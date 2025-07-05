"""Tests for Jira data extraction."""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from jirascope.core.config import Config
from jirascope.extractors.jira_extractor import JiraExtractor
from jirascope.models import EpicHierarchy, WorkItem


@pytest.fixture
def mock_config():
    return Config(jira_mcp_endpoint="http://test.com", jira_batch_size=10)


@pytest.fixture
def sample_epic():
    return WorkItem(
        key="PROJ-1",
        summary="Sample Epic",
        issue_type="Epic",
        status="In Progress",
        created=datetime.now(),
        updated=datetime.now(),
        reporter="test@example.com",
        description="Epic description",
        parent_key=None,
        epic_key=None,
        assignee=None,
        embedding=None,
    )


@pytest.fixture
def sample_story():
    return WorkItem(
        key="PROJ-2",
        summary="Sample Story",
        issue_type="Story",
        status="To Do",
        epic_key="PROJ-1",
        created=datetime.now(),
        updated=datetime.now(),
        reporter="test@example.com",
        description="Story description",
        parent_key=None,
        assignee=None,
        embedding=None,
    )


@pytest.mark.asyncio
async def test_extract_active_hierarchies(mock_config, sample_epic, sample_story):
    """Test extracting active hierarchies - verify actual hierarchy building logic."""
    extractor = JiraExtractor(mock_config)

    # Mock the MCP client
    mock_client = AsyncMock()
    mock_client.get_work_items.side_effect = [
        [sample_epic],  # Epics query
        [sample_story],  # Children query
    ]

    # Mock the async context manager
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_client
    mock_context.__aexit__.return_value = None

    with patch("jirascope.extractors.jira_extractor.MCPClient", return_value=mock_context):
        hierarchies = await extractor.extract_active_hierarchies("PROJ")

    # Test actual hierarchy building logic
    assert len(hierarchies) == 1
    assert isinstance(hierarchies[0], EpicHierarchy)

    # Test that the hierarchy was built correctly
    hierarchy = hierarchies[0]
    assert hierarchy.epic.key == "PROJ-1"
    assert hierarchy.epic.issue_type == "Epic"
    assert len(hierarchy.stories) == 1
    assert hierarchy.stories[0].key == "PROJ-2"
    assert hierarchy.stories[0].epic_key == "PROJ-1"  # Test relationship

    # Test that the hierarchy contains the expected work items
    assert hierarchy.stories[0].issue_type == "Story"
    assert hierarchy.stories[0].status == "To Do"

    # Test the epic-child relationship is maintained
    assert all(story.epic_key == hierarchy.epic.key for story in hierarchy.stories)


@pytest.mark.asyncio
async def test_get_epic_tree(mock_config, sample_epic, sample_story):
    """Test building complete epic tree - verify tree structure logic."""
    extractor = JiraExtractor(mock_config)

    # Create additional test data for better tree testing
    sub_task = WorkItem(
        key="PROJ-3",
        summary="Sub-task under story",
        issue_type="Sub-task",
        status="In Progress",
        epic_key="PROJ-1",
        parent_key="PROJ-2",  # Child of the story
        created=datetime.now(),
        updated=datetime.now(),
        reporter="test@example.com",
        description="Sub-task description",
        assignee=None,
        embedding=None,
    )

    mock_client = AsyncMock()
    mock_client.get_work_item.return_value = sample_epic
    mock_client.get_work_items.return_value = [sample_story, sub_task]

    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_client
    mock_context.__aexit__.return_value = None

    with patch("jirascope.extractors.jira_extractor.MCPClient", return_value=mock_context):
        tree = await extractor.get_epic_tree("PROJ-1")

    # Test actual tree structure logic
    assert tree.epic.key == "PROJ-1"
    assert tree.epic.issue_type == "Epic"

    # Test that direct children are identified correctly
    assert len(tree.direct_children) >= 1
    direct_child_keys = [child.key for child in tree.direct_children]
    assert "PROJ-2" in direct_child_keys

    # Test that all descendants are collected
    assert len(tree.all_descendants) >= 2  # Story + Sub-task
    descendant_keys = [desc.key for desc in tree.all_descendants]
    assert "PROJ-2" in descendant_keys
    assert "PROJ-3" in descendant_keys

    # Test hierarchy depth calculation
    assert tree.hierarchy_depth >= 2  # Epic -> Story -> Sub-task

    # Test total items calculation
    assert tree.total_items >= 3  # Epic + Story + Sub-task

    # Test that the tree maintains proper relationships
    for descendant in tree.all_descendants:
        assert descendant.epic_key == tree.epic.key or descendant.key == tree.epic.key


def test_cost_tracking(mock_config):
    """Test extraction cost tracking."""
    extractor = JiraExtractor(mock_config)

    # Initially zero
    cost = extractor.calculate_extraction_cost()
    assert cost.api_calls == 0
    assert cost.items_processed == 0

    # Add some costs
    extractor.cost_tracker.add_call(processing_time=1.5, items_count=10)

    cost = extractor.calculate_extraction_cost()
    assert cost.api_calls == 1
    assert cost.items_processed == 10
    assert cost.processing_time == 1.5
    assert cost.estimated_cost > 0


@pytest.mark.asyncio
async def test_incremental_updates(mock_config, sample_story):
    """Test getting incremental updates."""
    extractor = JiraExtractor(mock_config)

    mock_client = AsyncMock()
    mock_client.get_work_items.return_value = [sample_story]

    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_client
    mock_context.__aexit__.return_value = None

    with patch("jirascope.extractors.jira_extractor.MCPClient", return_value=mock_context):
        updates = await extractor.get_incremental_updates(
            "PROJ", "2024-01-01T00:00:00Z", {"PROJ-1"}, {"PROJ-2"}
        )

    assert len(updates) == 1
    assert updates[0].key == "PROJ-2"
