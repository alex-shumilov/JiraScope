"""Pydantic models for Jira work items."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class WorkItem(BaseModel):
    """Represents a Jira work item (issue, story, task, etc.)."""

    key: str = Field(..., description="Jira issue key (e.g., PROJ-123)")
    summary: str = Field(..., description="Issue summary/title")
    description: str | None = Field(None, description="Issue description")
    issue_type: str = Field(..., description="Type of issue (Story, Task, Bug, etc.)")
    status: str = Field(..., description="Current status")
    parent_key: str | None = Field(None, description="Parent issue key if this is a subtask")
    epic_key: str | None = Field(None, description="Epic key this issue belongs to")
    created: datetime = Field(..., description="Creation timestamp")
    updated: datetime = Field(..., description="Last update timestamp")
    assignee: str | None = Field(None, description="Assigned user")
    reporter: str = Field(..., description="Reporter/creator")
    components: list[str] = Field(default_factory=list, description="Project components")
    labels: list[str] = Field(default_factory=list, description="Issue labels")
    embedding: list[float] | None = Field(
        None, description="Vector embedding for similarity analysis"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "key": "PROJ-123",
                "summary": "Implement login functionality",
                "description": "Create the login page with validation",
                "issue_type": "Story",
                "status": "In Progress",
                "created": "2023-01-15T10:00:00",
                "updated": "2023-01-16T14:30:00",
                "reporter": "jdoe",
            }
        }
    }

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        """Customize the JSON schema for serialization."""
        schema = super().model_json_schema(*args, **kwargs)
        return schema

    def model_dump_json(self, *args, **kwargs):
        """Customize JSON serialization."""
        kwargs["exclude_none"] = kwargs.get("exclude_none", True)
        data = self.model_dump(*args, **kwargs)

        # Convert datetime objects to ISO format
        if isinstance(data.get("created"), datetime):
            data["created"] = data["created"].isoformat()
        if isinstance(data.get("updated"), datetime):
            data["updated"] = data["updated"].isoformat()

        # Use default json serializer
        import json

        return json.dumps(data, default=str)


class EpicHierarchy(BaseModel):
    """Represents an epic with its child work items."""

    epic: WorkItem = Field(..., description="The epic work item")
    stories: list[WorkItem] = Field(default_factory=list, description="Stories under this epic")
    tasks: list[WorkItem] = Field(default_factory=list, description="Tasks under this epic")
    subtasks: list[WorkItem] = Field(default_factory=list, description="Subtasks under this epic")

    @property
    def total_items(self) -> int:
        """Total number of work items in this hierarchy."""
        return len(self.stories) + len(self.tasks) + len(self.subtasks)

    @property
    def all_items(self) -> list[WorkItem]:
        """All work items in this hierarchy including the epic."""
        return [self.epic] + self.stories + self.tasks + self.subtasks


class EpicTree(BaseModel):
    """Complete Epic hierarchy with all descendants."""

    epic: WorkItem = Field(..., description="The epic work item")
    direct_children: list[WorkItem] = Field(
        default_factory=list, description="Direct children of the epic"
    )
    all_descendants: list[WorkItem] = Field(
        default_factory=list, description="All descendant work items"
    )
    hierarchy_depth: int = Field(..., description="Maximum depth of the hierarchy")
    total_story_points: int | None = Field(None, description="Total story points for all items")
    completion_percentage: float = Field(0.0, ge=0.0, le=100.0, description="Completion percentage")

    @property
    def total_items(self) -> int:
        """Total number of work items in this tree."""
        return 1 + len(self.all_descendants)


class ExtractionCost(BaseModel):
    """Cost tracking for Jira data extraction."""

    api_calls: int = Field(0, description="Number of API calls made")
    items_processed: int = Field(0, description="Number of items processed")
    processing_time: float = Field(0.0, description="Processing time in seconds")
    estimated_cost: float = Field(0.0, description="Estimated monetary cost")
    rate_limit_hits: int = Field(0, description="Number of rate limit hits")

    def add_call(self, processing_time: float = 0.0, items_count: int = 0):
        """Add a single API call to the cost tracking."""
        self.api_calls += 1
        self.processing_time += processing_time
        self.items_processed += items_count
        # Rough estimate: $0.01 per API call
        self.estimated_cost += 0.01


class ProcessingResult(BaseModel):
    """Result of batch processing operations."""

    processed_items: int = Field(0, description="Number of items successfully processed")
    skipped_items: int = Field(0, description="Number of items skipped")
    failed_items: int = Field(0, description="Number of items that failed processing")
    total_cost: float = Field(0.0, description="Total monetary cost")
    processing_time: float = Field(0.0, description="Total processing time in seconds")
    batch_stats: dict[str, float] = Field(
        default_factory=dict, description="Batch processing statistics"
    )
    errors: list[str] = Field(default_factory=list, description="Error messages")

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.processed_items + self.failed_items
        if total == 0:
            return 0.0
        return (self.processed_items / total) * 100.0


class QualityReport(BaseModel):
    """Quality validation report for embeddings."""

    total_tests: int = Field(..., description="Total number of test queries")
    passed_tests: int = Field(..., description="Number of tests that passed")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall quality score")
    results: list[dict[str, Any]] = Field(default_factory=list, description="Detailed test results")
    recommendations: list[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100.0


class AnalysisResult(BaseModel):
    """Result of AI analysis on a work item."""

    work_item_key: str = Field(..., description="Key of the analyzed work item")
    analysis_type: str = Field(..., description="Type of analysis performed")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    insights: dict[str, Any] = Field(
        default_factory=dict, description="Analysis insights and results"
    )
    cost: float | None = Field(None, description="API cost for this analysis")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "work_item_key": "PROJ-123",
                "analysis_type": "quality",
                "confidence": 0.85,
                "insights": {"score": 4.2, "suggestions": ["Improve clarity"]},
                "timestamp": "2023-01-20T10:15:30",
            }
        }
    }

    def model_dump_json(self, *args, **kwargs):
        """Customize JSON serialization."""
        kwargs["exclude_none"] = kwargs.get("exclude_none", True)
        data = self.model_dump(*args, **kwargs)

        # Convert datetime objects to ISO format
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()

        # Use default json serializer
        import json

        return json.dumps(data, default=str)
