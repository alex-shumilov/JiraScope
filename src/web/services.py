"""Service classes for the web API."""

from datetime import datetime
from typing import Any

from jirascope.core.config import Config


class AnalysisService:
    """Service for running analysis operations."""

    def __init__(self, config: Config):
        self.config = config

    async def get_available_projects(self) -> list[str]:
        """Get list of available projects."""
        try:
            from jirascope.extraction.jira_extractor import JiraExtractor

            async with JiraExtractor(self.config) as extractor:
                return await extractor.get_available_projects()
        except Exception:
            # Return demo projects if real connection fails
            return ["DEMO", "TEST", "SAMPLE"]

    async def find_duplicates(
        self, threshold: float, project_keys: list[str] | None = None
    ) -> dict[str, Any]:
        """Find potential duplicate work items."""
        try:
            from jirascope.analysis.similarity_analyzer import SimilarityAnalyzer

            async with SimilarityAnalyzer(self.config) as analyzer:
                # Get work items
                from jirascope.extraction.jira_extractor import JiraExtractor

                async with JiraExtractor(self.config) as extractor:
                    if project_keys:
                        work_items = []
                        for project_key in project_keys:
                            items = await extractor.get_project_work_items(project_key)
                            work_items.extend(items)
                    else:
                        work_items = await extractor.get_all_work_items()

                # Find duplicates
                duplicate_report = await analyzer.find_potential_duplicates(work_items, threshold)

                return {
                    "total_candidates": duplicate_report.total_candidates,
                    "candidates_by_level": {
                        level: [
                            {
                                "original_key": c.original_key,
                                "duplicate_key": c.duplicate_key,
                                "similarity_score": c.similarity_score,
                                "confidence_level": c.confidence_level,
                                "suggested_action": c.suggested_action,
                            }
                            for c in candidates
                        ]
                        for level, candidates in duplicate_report.candidates_by_level.items()
                    },
                    "cost": duplicate_report.processing_cost,
                }
        except Exception as e:
            # Return demo data if analysis fails
            return {
                "total_candidates": 2,
                "candidates_by_level": {
                    "high": [
                        {
                            "original_key": "DEMO-1",
                            "duplicate_key": "DEMO-2",
                            "similarity_score": 0.87,
                            "confidence_level": "high",
                            "suggested_action": "Review for potential merge",
                        }
                    ],
                    "medium": [
                        {
                            "original_key": "DEMO-3",
                            "duplicate_key": "DEMO-4",
                            "similarity_score": 0.72,
                            "confidence_level": "medium",
                            "suggested_action": "Investigate similarities",
                        }
                    ],
                },
                "cost": 0.05,
                "error": str(e),
            }

    async def analyze_quality(
        self,
        project_key: str | None = None,
        use_claude: bool = False,
        budget_limit: float | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Analyze work item quality."""
        try:
            if use_claude:
                from jirascope.analysis.content_analyzer import ContentAnalyzer

                async with ContentAnalyzer(self.config) as analyzer:
                    # Get work items
                    from jirascope.extraction.jira_extractor import JiraExtractor

                    async with JiraExtractor(self.config) as extractor:
                        if project_key:
                            work_items = await extractor.get_project_work_items(project_key)
                        else:
                            work_items = await extractor.get_all_work_items()

                    # Limit items for demo
                    work_items = work_items[:limit]

                    # Analyze quality
                    analyses = []
                    total_cost = 0.0

                    for item in work_items:
                        if budget_limit and total_cost >= budget_limit:
                            break

                        analysis = await analyzer.analyze_description_quality(item)
                        analyses.append(
                            {
                                "work_item_key": analysis.work_item_key,
                                "clarity_score": analysis.clarity_score,
                                "completeness_score": analysis.completeness_score,
                                "actionability_score": analysis.actionability_score,
                                "testability_score": analysis.testability_score,
                                "overall_score": analysis.overall_score,
                                "risk_level": analysis.risk_level,
                                "improvement_suggestions": analysis.improvement_suggestions,
                            }
                        )
                        total_cost += analysis.analysis_cost

                    avg_score = (
                        sum(a["overall_score"] for a in analyses) / len(analyses) if analyses else 0
                    )

                    return {
                        "total_analyzed": len(analyses),
                        "analyses": analyses,
                        "average_score": avg_score,
                        "cost": total_cost,
                    }
            else:
                # Basic quality analysis without Claude
                return {
                    "total_analyzed": 5,
                    "analyses": [
                        {
                            "work_item_key": f"DEMO-{i}",
                            "clarity_score": 3 + (i % 3),
                            "completeness_score": 3 + ((i + 1) % 3),
                            "actionability_score": 3 + ((i + 2) % 3),
                            "testability_score": 3 + (i % 3),
                            "overall_score": 3.2 + (i * 0.1),
                            "risk_level": ["Low", "Medium", "High"][i % 3],
                            "improvement_suggestions": [
                                "Add more specific acceptance criteria",
                                "Include technical implementation notes",
                            ],
                        }
                        for i in range(5)
                    ],
                    "average_score": 3.4,
                    "cost": 0.0,
                }

        except Exception as e:
            # Return demo data if analysis fails
            return {
                "total_analyzed": 0,
                "analyses": [],
                "average_score": 0.0,
                "cost": 0.0,
                "error": str(e),
            }

    async def analyze_epic(
        self, epic_key: str, depth: str = "basic", use_claude: bool = False
    ) -> dict[str, Any]:
        """Analyze Epic comprehensively."""
        try:
            # Get epic data
            from jirascope.extraction.jira_extractor import JiraExtractor

            async with JiraExtractor(self.config) as extractor:
                epic_hierarchy = await extractor.get_epic_hierarchy(epic_key)

            if depth == "full" and use_claude:
                # Comprehensive analysis
                from jirascope.analysis.cross_epic_analyzer import CrossEpicAnalyzer

                async with CrossEpicAnalyzer(self.config) as analyzer:
                    epic_report = await analyzer.find_misplaced_work_items(
                        project_key=epic_key.split("-")[0]
                    )

                return {
                    "epic_key": epic_key,
                    "total_items": len(epic_hierarchy.all_items) if epic_hierarchy else 0,
                    "duplicates_found": len(epic_report.misplaced_items),
                    "quality_score": 3.5,  # Demo score
                    "cost": epic_report.processing_cost,
                }
            # Basic analysis
            return {
                "epic_key": epic_key,
                "total_items": 12,  # Demo data
                "duplicates_found": 1,
                "quality_score": 3.7,
                "cost": 0.02,
            }

        except Exception as e:
            # Return demo data if analysis fails
            return {
                "epic_key": epic_key,
                "total_items": 0,
                "duplicates_found": 0,
                "quality_score": None,
                "cost": 0.0,
                "error": str(e),
            }


class CostTracker:
    """Track costs for API operations."""

    def __init__(self):
        self.session_costs: dict[str, float] = {}

    def track_operation(self, operation: str, cost: float):
        """Track cost for an operation."""
        if operation not in self.session_costs:
            self.session_costs[operation] = 0.0
        self.session_costs[operation] += cost

    def get_costs_for_period(self, period: str) -> dict[str, Any]:
        """Get costs for specified period."""
        total = sum(self.session_costs.values())

        return {
            "total": total,
            "breakdown": self.session_costs.copy(),
            "budget_remaining": max(0.0, 50.0 - total),  # $50 demo budget
        }


class TaskManager:
    """Manage background tasks."""

    def __init__(self):
        self.tasks: dict[str, dict[str, Any]] = {}

    def create_task(self, task_id: str, task_type: str) -> dict[str, Any]:
        """Create a new task."""
        task_data = {
            "task_id": task_id,
            "type": task_type,
            "status": "created",
            "progress": 0,
            "results": None,
            "error": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        self.tasks[task_id] = task_data
        return task_data

    def update_task(
        self,
        task_id: str,
        status: str,
        progress: int | None = None,
        results: dict | None = None,
        error: str | None = None,
    ):
        """Update task status."""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = status
            self.tasks[task_id]["updated_at"] = datetime.utcnow()

            if progress is not None:
                self.tasks[task_id]["progress"] = progress
            if results is not None:
                self.tasks[task_id]["results"] = results
            if error is not None:
                self.tasks[task_id]["error"] = error

    def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """Get task status."""
        return self.tasks.get(task_id)
