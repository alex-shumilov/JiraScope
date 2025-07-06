"""Jira data extraction with hierarchical Epic->Story->Task structure."""

import time

from ..clients.mcp_client import MCPClient
from ..core.config import Config
from ..models import EpicHierarchy, EpicTree, ExtractionCost, WorkItem
from ..utils.logging import StructuredLogger

logger = StructuredLogger(__name__)


class JiraExtractor:
    """Extract hierarchical data structures from Jira."""

    # JQL query patterns for extraction
    ACTIVE_EPICS_JQL = """
        issuetype = Epic
        AND project = {project_key}
        AND status != Closed
        AND issueFunction in hasLinks("is parent of")
    """

    EPIC_CHILDREN_JQL = """
        parent = {epic_key}
        OR ("Epic Link" = {epic_key} AND issuetype != Epic)
        ORDER BY created ASC
    """

    INCREMENTAL_SYNC_JQL = """
        project = {project_key}
        AND updated >= "{last_sync}"
        AND (
            issuetype = Epic
            OR "Epic Link" in ({tracked_epics})
            OR parent in ({tracked_work_items})
        )
    """

    def __init__(self, config: Config):
        self.config = config
        self.cost_tracker = ExtractionCost()

    async def extract_active_hierarchies(self, project_key: str) -> list[EpicHierarchy]:
        """Extract all Epics that have active children (any status)."""
        logger.info(f"Starting extraction of active hierarchies for project {project_key}")
        start_time = time.time()

        try:
            async with MCPClient(self.config) as client:
                # First, get all active epics
                epics_jql = self.ACTIVE_EPICS_JQL.format(project_key=project_key)
                epics = await client.get_work_items(epics_jql)

                self.cost_tracker.add_call(
                    processing_time=time.time() - start_time, items_count=len(epics)
                )

                logger.info(f"Found {len(epics)} active epics")

                hierarchies = []
                for epic in epics:
                    try:
                        hierarchy = await self._extract_epic_hierarchy(client, epic)
                        hierarchies.append(hierarchy)
                    except Exception as e:
                        logger.error(
                            f"Failed to extract hierarchy for epic {epic.key}", error=str(e)
                        )
                        continue

                total_time = time.time() - start_time
                logger.log_operation(
                    "extract_active_hierarchies",
                    total_time,
                    success=True,
                    epics_count=len(epics),
                    hierarchies_count=len(hierarchies),
                )

                return hierarchies

        except Exception as e:
            logger.error("Failed to extract active hierarchies", error=str(e))
            raise

    async def get_epic_tree(self, epic_key: str) -> EpicTree:
        """Get complete Epic with all nested work items."""
        logger.info(f"Building complete epic tree for {epic_key}")
        start_time = time.time()

        try:
            async with MCPClient(self.config) as client:
                # Get the epic itself
                epic = await client.get_work_item(epic_key)
                if not epic:
                    raise ValueError(f"Epic {epic_key} not found")

                # Get all children recursively
                direct_children = await self._get_direct_children(client, epic_key)
                all_descendants = await self._get_all_descendants(client, epic_key)

                # Calculate hierarchy depth
                hierarchy_depth = self._calculate_hierarchy_depth(all_descendants)

                # Calculate completion percentage (simplified)
                completion_percentage = self._calculate_completion_percentage(
                    [epic] + all_descendants
                )

                tree = EpicTree(
                    epic=epic,
                    direct_children=direct_children,
                    all_descendants=all_descendants,
                    hierarchy_depth=hierarchy_depth,
                    completion_percentage=completion_percentage,
                )

                processing_time = time.time() - start_time
                self.cost_tracker.add_call(processing_time, len(all_descendants) + 1)

                logger.log_operation(
                    "get_epic_tree",
                    processing_time,
                    success=True,
                    epic_key=epic_key,
                    total_items=tree.total_items,
                )

                return tree

        except Exception as e:
            logger.error(f"Failed to build epic tree for {epic_key}", error=str(e))
            raise

    async def get_incremental_updates(
        self, project_key: str, last_sync: str, tracked_epics: set[str], tracked_items: set[str]
    ) -> list[WorkItem]:
        """Get work items updated since last sync."""
        logger.info(f"Getting incremental updates since {last_sync}")
        start_time = time.time()

        try:
            async with MCPClient(self.config) as client:
                # Build JQL for incremental sync
                tracked_epics_str = "'" + "','".join(tracked_epics) + "'" if tracked_epics else "''"
                tracked_items_str = "'" + "','".join(tracked_items) + "'" if tracked_items else "''"

                jql = self.INCREMENTAL_SYNC_JQL.format(
                    project_key=project_key,
                    last_sync=last_sync,
                    tracked_epics=tracked_epics_str,
                    tracked_work_items=tracked_items_str,
                )

                updated_items = await client.get_work_items(jql)

                processing_time = time.time() - start_time
                self.cost_tracker.add_call(processing_time, len(updated_items))

                logger.log_operation(
                    "get_incremental_updates",
                    processing_time,
                    success=True,
                    updated_items=len(updated_items),
                )

                return updated_items

        except Exception as e:
            logger.error("Failed to get incremental updates", error=str(e))
            raise

    def calculate_extraction_cost(self) -> ExtractionCost:
        """Get current extraction cost tracking."""
        return self.cost_tracker

    def reset_cost_tracking(self):
        """Reset cost tracking for new extraction session."""
        self.cost_tracker = ExtractionCost()

    async def _extract_epic_hierarchy(self, client: MCPClient, epic: WorkItem) -> EpicHierarchy:
        """Extract complete hierarchy for a single epic."""
        children_jql = self.EPIC_CHILDREN_JQL.format(epic_key=epic.key)
        children = await client.get_work_items(children_jql)

        # Categorize children by type
        stories = [item for item in children if item.issue_type.lower() in ["story", "user story"]]
        tasks = [item for item in children if item.issue_type.lower() in ["task", "improvement"]]
        subtasks = [item for item in children if item.issue_type.lower() in ["sub-task", "subtask"]]

        self.cost_tracker.add_call(items_count=len(children))

        hierarchy = EpicHierarchy(epic=epic, stories=stories, tasks=tasks, subtasks=subtasks)

        logger.debug(f"Extracted hierarchy for {epic.key}: {hierarchy.total_items} items")
        return hierarchy

    async def _get_direct_children(self, client: MCPClient, parent_key: str) -> list[WorkItem]:
        """Get direct children of a work item."""
        children_jql = f'parent = {parent_key} OR "Epic Link" = {parent_key}'
        children = await client.get_work_items(children_jql)
        self.cost_tracker.add_call(items_count=len(children))
        return children

    async def _get_all_descendants(self, client: MCPClient, root_key: str) -> list[WorkItem]:
        """Get all descendant work items recursively."""
        all_descendants = []
        visited = set()

        async def collect_descendants(parent_key: str):
            if parent_key in visited:
                return
            visited.add(parent_key)

            children = await self._get_direct_children(client, parent_key)
            all_descendants.extend(children)

            # Recursively collect descendants
            for child in children:
                await collect_descendants(child.key)

        await collect_descendants(root_key)
        return all_descendants

    def _calculate_hierarchy_depth(self, items: list[WorkItem]) -> int:
        """Calculate the maximum depth of the hierarchy."""
        # Simple heuristic: count parent relationships
        depth_map = {}

        for item in items:
            if item.parent_key:
                depth_map[item.key] = depth_map.get(item.parent_key, 0) + 1
            else:
                depth_map[item.key] = 1

        return max(depth_map.values()) if depth_map else 1

    def _calculate_completion_percentage(self, items: list[WorkItem]) -> float:
        """Calculate completion percentage based on status."""
        if not items:
            return 0.0

        completed_statuses = {"done", "closed", "resolved", "completed"}
        completed_items = sum(1 for item in items if item.status.lower() in completed_statuses)

        return (completed_items / len(items)) * 100.0
