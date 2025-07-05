"""Global test configuration and utilities."""

import json
import random
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock

import pytest

from src.jirascope.core.config import Config
from src.jirascope.models import CrossEpicReport, QualityAnalysis, WorkItem


class TestConfig:
    """Centralized test configuration to eliminate hardcoding."""

    # Time configuration
    BASE_TIME = datetime.now()
    TIME_DELTA_DAYS = 1

    # Score ranges (for realistic testing)
    SCORE_RANGES = {
        "similarity": (0.5, 0.95),
        "quality": (1.0, 5.0),
        "confidence": (0.0, 1.0),
        "cost": (0.01, 0.10),
    }

    # Embedding dimensions (common sizes)
    EMBEDDING_DIMS = {
        "small": 384,
        "medium": 768,
        "large": 1536,
    }

    # Batch sizes for testing
    BATCH_SIZES = [8, 16, 32, 64]

    # Sample keys (avoid hardcoding specific keys)
    KEY_PREFIXES = ["TEST", "PROJ", "EPIC", "TASK"]

    @classmethod
    def random_score(cls, score_type: str) -> float:
        """Generate random score within appropriate range."""
        min_val, max_val = cls.SCORE_RANGES.get(score_type, (0.0, 1.0))
        return random.uniform(min_val, max_val)

    @classmethod
    def random_key(cls, prefix: Optional[str] = None) -> str:
        """Generate random but consistent test key."""
        if not prefix:
            prefix = random.choice(cls.KEY_PREFIXES)
        return f"{prefix}-{random.randint(100, 999)}"

    @classmethod
    def relative_time(cls, days_offset: int = 0) -> datetime:
        """Generate relative timestamps."""
        return cls.BASE_TIME + timedelta(days=days_offset)


class TestDataBuilder:
    """Builder pattern for creating test data without hardcoding."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset builder state."""
        self._work_items = []
        self._counter = 1
        return self

    def add_story(
        self, summary: Optional[str] = None, complexity: str = "simple", **kwargs
    ) -> "TestDataBuilder":
        """Add a story with configurable complexity."""
        key = kwargs.get("key", TestConfig.random_key("STORY"))

        descriptions = {
            "simple": "A simple user story with basic requirements.",
            "medium": self._medium_description(),
            "complex": self._complex_description(),
        }

        work_item = WorkItem(
            key=key,
            summary=summary or f"Test Story {self._counter}",
            description=descriptions.get(complexity, descriptions["simple"]),
            issue_type="Story",
            status=kwargs.get("status", "Open"),
            created=TestConfig.relative_time(-self._counter),
            updated=TestConfig.relative_time(),
            reporter=kwargs.get("reporter", "test.user"),
            components=kwargs.get("components", ["frontend"]),
            labels=kwargs.get("labels", ["test"]),
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["key", "status", "reporter", "components", "labels"]
            },
        )

        self._work_items.append(work_item)
        self._counter += 1
        return self

    def add_bug(self, severity: str = "medium", **kwargs) -> "TestDataBuilder":
        """Add a bug with configurable severity."""
        severities = {
            "low": "Minor UI issue with low impact",
            "medium": "Functional issue affecting some users",
            "high": "Critical system failure requiring immediate attention",
        }

        key = kwargs.get("key", TestConfig.random_key("BUG"))
        summary = kwargs.get("summary", f"Bug Report {self._counter}")

        # Remove conflicting keys from kwargs
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["key", "summary", "status", "reporter"]
        }

        work_item = WorkItem(
            key=key,
            summary=summary,
            description=severities.get(severity, severities["medium"]),
            issue_type="Bug",
            status=kwargs.get("status", "Open"),
            created=TestConfig.relative_time(-self._counter),
            updated=TestConfig.relative_time(),
            reporter=kwargs.get("reporter", "test.user"),
            **filtered_kwargs,
        )

        self._work_items.append(work_item)
        self._counter += 1
        return self

    def add_epic(self, scope: str = "medium", **kwargs) -> "TestDataBuilder":
        """Add an epic with configurable scope."""
        scopes = {
            "small": "A focused epic with clear boundaries",
            "medium": "A moderate epic spanning multiple teams",
            "large": "A complex epic with multiple phases and dependencies",
        }

        key = kwargs.get("key", TestConfig.random_key("EPIC"))
        summary = kwargs.get("summary", f"Epic {self._counter}")

        # Remove conflicting keys from kwargs
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["key", "summary", "status", "reporter"]
        }

        work_item = WorkItem(
            key=key,
            summary=summary,
            description=scopes.get(scope, scopes["medium"]),
            issue_type="Epic",
            status=kwargs.get("status", "In Progress"),
            created=TestConfig.relative_time(-self._counter * 7),  # Epics start earlier
            updated=TestConfig.relative_time(),
            reporter=kwargs.get("reporter", "product.manager"),
            **filtered_kwargs,
        )

        self._work_items.append(work_item)
        self._counter += 1
        return self

    def build(self) -> List[WorkItem]:
        """Return built work items."""
        return self._work_items.copy()

    def _medium_description(self) -> str:
        """Medium complexity description."""
        return """## Requirements
- Implement basic functionality
- Add input validation
- Include error handling

## Acceptance Criteria
- [ ] Feature works as expected
- [ ] Edge cases are handled
- [ ] Tests are added"""

    def _complex_description(self) -> str:
        """Complex description for testing."""
        return """## User Story
As a user, I want comprehensive functionality.

## Requirements
- Multiple system integrations
- Complex business logic
- Performance considerations
- Security requirements

## Acceptance Criteria
- [ ] All systems integrate properly
- [ ] Business rules are enforced
- [ ] Performance meets requirements
- [ ] Security standards are met

## Technical Notes
- Consider scalability
- Implement monitoring
- Add comprehensive testing"""


class MockHelper:
    """Helper for creating consistent mocks without hardcoding."""

    @staticmethod
    def mock_embedding(dimension: str = "medium") -> List[float]:
        """Create consistent mock embedding."""
        dim = TestConfig.EMBEDDING_DIMS[dimension]
        random.seed(42)  # Consistent for tests
        return [random.uniform(-1, 1) for _ in range(dim)]

    @staticmethod
    def mock_search_results(count: int = 3, score_range: tuple = (0.7, 0.9)) -> List[Dict]:
        """Create mock search results with realistic scores."""
        results = []
        for i in range(count):
            score = random.uniform(*score_range)
            results.append(
                {
                    "score": score,
                    "work_item": {
                        "key": TestConfig.random_key(),
                        "summary": f"Mock result {i+1}",
                        "issue_type": random.choice(["Story", "Bug", "Task"]),
                    },
                }
            )
        return results

    @staticmethod
    def mock_claude_response(response_type: str, **kwargs) -> Dict[str, Any]:
        """Create mock Claude responses based on type."""
        responses = {
            "quality_analysis": {
                "content": f'{{"clarity_score": {int(TestConfig.random_score("quality"))}, "completeness_score": {int(TestConfig.random_score("quality"))}, "actionability_score": {int(TestConfig.random_score("quality"))}, "testability_score": {int(TestConfig.random_score("quality"))}, "overall_score": {int(TestConfig.random_score("quality"))}, "improvement_suggestions": ["Add more specific acceptance criteria", "Include error handling scenarios"], "risk_level": "Medium"}}',
                "cost": TestConfig.random_score("cost"),
            },
            "template_inference": {
                "content": '{"title_template": "Test: {description}", "confidence_score": 0.8}',
                "cost": TestConfig.random_score("cost"),
            },
        }

        return responses.get(
            response_type, {"content": "{}", "cost": TestConfig.random_score("cost")}
        )


# Global fixtures
@pytest.fixture
def test_config():
    """Provide test configuration."""
    return TestConfig


@pytest.fixture
def data_builder():
    """Provide data builder."""
    return TestDataBuilder()


@pytest.fixture
def mock_helper():
    """Provide mock helper."""
    return MockHelper


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = Mock(spec=Config)
    config.jira_mcp_endpoint = "http://localhost:8000"
    config.jira_url = "https://test.atlassian.net"
    config.jira_username = "test@example.com"
    config.jira_password = "test_password"
    config.qdrant_url = "http://localhost:6333"
    config.qdrant_collection = "test_collection"
    config.lmstudio_url = "http://localhost:1234"
    config.lmstudio_model = "test-model"
    config.claude_model = "claude-3-5-sonnet-20241022"
    config.logging_level = "INFO"
    return config


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file."""
    config_content = """
jira:
  url: "https://test.atlassian.net"
  username: "test@example.com"
  password: "test_password"

qdrant:
  url: "http://localhost:6333"
  collection: "test_collection"

lmstudio:
  url: "http://localhost:1234"
  model: "test-model"

claude:
  model: "claude-3-5-sonnet-20241022"

logging:
  level: "INFO"
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content.strip())
        f.flush()
        yield f.name

    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def base_time():
    """Provide a consistent base time for test data."""
    return datetime.now(timezone.utc)


@pytest.fixture
def sample_work_items(base_time):
    """Create a comprehensive set of sample work items for testing."""
    return [
        WorkItem(
            key="PROJ-1",
            summary="Simple bug fix",
            issue_type="Bug",
            status="Open",
            created=base_time,
            updated=base_time,
            reporter="test@example.com",
            description="Fix a simple bug in the login system",
            assignee="dev@example.com",
            parent_key=None,
            epic_key=None,
            embedding=[0.1, 0.2, 0.3] * 100,
        ),
        WorkItem(
            key="PROJ-2",
            summary="User authentication feature",
            issue_type="Story",
            status="In Progress",
            created=base_time - timedelta(days=1),
            updated=base_time,
            reporter="pm@example.com",
            description="Implement OAuth2 authentication for user login",
            assignee="dev@example.com",
            parent_key=None,
            epic_key="PROJ-100",
            embedding=[0.4, 0.5, 0.6] * 100,
        ),
        WorkItem(
            key="PROJ-3",
            summary="Database performance improvement",
            issue_type="Task",
            status="Done",
            created=base_time - timedelta(days=2),
            updated=base_time - timedelta(hours=1),
            reporter="dba@example.com",
            description="Optimize database queries for better performance",
            assignee="dba@example.com",
            parent_key=None,
            epic_key=None,
            embedding=[0.7, 0.8, 0.9] * 100,
        ),
        WorkItem(
            key="PROJ-4",
            summary="Epic: User Management System",
            issue_type="Epic",
            status="In Progress",
            created=base_time - timedelta(days=30),
            updated=base_time,
            reporter="architect@example.com",
            description="Complete user management system with roles and permissions",
            assignee="lead@example.com",
            parent_key=None,
            epic_key=None,
            embedding=[0.2, 0.3, 0.4] * 100,
        ),
        WorkItem(
            key="PROJ-5",
            summary="Frontend component library",
            issue_type="Story",
            status="Open",
            created=base_time - timedelta(days=3),
            updated=base_time,
            reporter="designer@example.com",
            description="Create reusable UI components for frontend",
            assignee="frontend@example.com",
            parent_key=None,
            epic_key="PROJ-200",
            embedding=[0.5, 0.6, 0.7] * 100,
        ),
        WorkItem(
            key="PROJ-6",
            summary="Complex platform overhaul",
            issue_type="Story",
            status="Open",
            created=base_time - timedelta(days=5),
            updated=base_time,
            reporter="architect@example.com",
            description="Complete overhaul of the e-commerce platform including payment processing, inventory management, user interface redesign, mobile app development, and third-party integrations",
            assignee="team@example.com",
            parent_key=None,
            epic_key=None,
            embedding=[0.8, 0.9, 0.1] * 100,
        ),
    ]


@pytest.fixture
def epic_work_items(base_time):
    """Create work items specifically for epic-related testing."""
    return {
        "EPIC-1": [
            {
                "key": "PROJ-1",
                "summary": "Authentication feature",
                "description": "Implement user authentication",
                "embedding": [0.1, 0.2, 0.3] * 100,
                "epic_key": "EPIC-1",
            },
            {
                "key": "PROJ-2",
                "summary": "Login validation",
                "description": "Add login form validation",
                "embedding": [0.1, 0.2, 0.3] * 100,
                "epic_key": "EPIC-1",
            },
        ],
        "EPIC-2": [
            {
                "key": "PROJ-3",
                "summary": "Database migration",
                "description": "Migrate database to new schema",
                "embedding": [0.8, 0.9, 0.7] * 100,
                "epic_key": "EPIC-2",
            },
            {
                "key": "PROJ-4",
                "summary": "Data cleanup",
                "description": "Clean up legacy data",
                "embedding": [0.8, 0.9, 0.7] * 100,
                "epic_key": "EPIC-2",
            },
        ],
    }


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client with common methods."""
    client = AsyncMock()
    client.health_check = AsyncMock(return_value=True)
    client.store_work_items = AsyncMock(return_value=True)
    client.search_similar_work_items = AsyncMock(return_value=[])
    client.get_work_item_by_key = AsyncMock(return_value=None)
    client.get_all_work_items = AsyncMock(return_value=[])
    client.search_with_filters = AsyncMock(return_value=[])
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.fixture
def mock_lmstudio_client():
    """Create a mock LMStudio client with common methods."""
    client = AsyncMock()
    client.health_check = AsyncMock(return_value=True)
    client.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3] * 100])
    client.calculate_similarity = AsyncMock(return_value=0.85)
    client.analyze_text = AsyncMock(return_value={"analysis": "test"})
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.fixture
def mock_claude_client():
    """Create a mock Claude client with common methods."""
    client = AsyncMock()
    client.health_check = AsyncMock(return_value=True)
    client.analyze_text = AsyncMock(return_value={"analysis": "test", "confidence": 0.85})
    client.analyze = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.fixture
def mock_claude_responses():
    """Provide mock Claude API responses for different analysis types."""
    return {
        "quality_analysis": {
            "content": json.dumps(
                {
                    "clarity_score": 4,
                    "completeness_score": 3,
                    "actionability_score": 4,
                    "testability_score": 3,
                    "overall_score": 3.5,
                    "risk_level": "Medium",
                    "improvement_suggestions": [
                        "Add more specific acceptance criteria",
                        "Include technical implementation details",
                    ],
                }
            ),
            "cost": 0.02,
        },
        "split_analysis": {
            "content": json.dumps(
                {
                    "should_split": True,
                    "complexity_score": 0.8,
                    "reasoning": "Work item is too complex and contains multiple features",
                    "suggested_splits": [
                        {
                            "suggested_title": "Payment Processing Module",
                            "suggested_description": "Implement core payment processing functionality",
                            "estimated_story_points": 8,
                        },
                        {
                            "suggested_title": "User Interface Updates",
                            "suggested_description": "Update UI components for new payment flow",
                            "estimated_story_points": 5,
                        },
                    ],
                }
            ),
            "cost": 0.03,
        },
        "tech_debt_cluster_analysis": {
            "content": json.dumps(
                {
                    "theme": "Database Performance",
                    "priority_score": 0.8,
                    "estimated_effort": "Medium",
                    "dependencies": [],
                    "impact_assessment": "Performance impact on user experience",
                    "recommended_approach": "Optimize queries and add indexes",
                }
            ),
            "cost": 0.05,
        },
        "template_inference": {
            "content": json.dumps(
                {
                    "issue_type": "Story",
                    "title_template": "As a {user_type}, I want {functionality} so that {benefit}",
                    "description_template": "**Background:** {context}\n**Requirements:** {requirements}\n**Acceptance Criteria:** {criteria}",
                    "required_fields": ["summary", "description", "acceptance_criteria"],
                    "common_components": ["frontend", "backend"],
                    "common_labels": ["user-story", "feature"],
                    "confidence_score": 0.85,
                    "sample_count": 15,
                }
            ),
            "cost": 0.04,
        },
        "scope_change_analysis": {
            "content": json.dumps(
                {
                    "has_drift": True,
                    "drift_score": 0.7,
                    "drift_events": [
                        {
                            "type": "scope_expansion",
                            "confidence": 0.8,
                            "description": "Additional requirements added",
                            "impact_level": "moderate",
                        }
                    ],
                }
            ),
            "cost": 0.03,
        },
        "complexity_analysis": {
            "content": json.dumps({"complexity_score": 0.7, "factors": ["database", "ui"]}),
            "cost": 0.03,
        },
    }


@pytest.fixture
def mock_httpx_responses():
    """Provide mock HTTP responses for client testing."""
    return {
        "jira_search": {
            "issues": [
                {
                    "key": "PROJ-1",
                    "fields": {
                        "summary": "Test issue",
                        "description": "Test description",
                        "issuetype": {"name": "Story"},
                        "status": {"name": "Open"},
                        "created": "2024-01-01T00:00:00.000Z",
                        "updated": "2024-01-01T00:00:00.000Z",
                        "reporter": {"name": "test.user"},
                        "components": [],
                        "labels": [],
                    },
                }
            ]
        },
        "lmstudio_models": {"object": "list", "data": [{"id": "text-embedding-bge-large-en-v1.5"}]},
        "lmstudio_embeddings": {
            "object": "list",
            "data": [{"embedding": [0.1, 0.2, 0.3] * 341}],  # 1023 dimensions
        },
    }


@pytest.fixture
def sample_embeddings():
    """Provide sample embeddings for testing."""
    return [
        [0.1, 0.2, 0.3] * 100,  # 300-dimension embedding
        [0.4, 0.5, 0.6] * 100,  # 300-dimension embedding
        [0.7, 0.8, 0.9] * 100,  # 300-dimension embedding
    ]


@pytest.fixture
def sample_quality_analysis():
    """Create a sample quality analysis result."""
    return QualityAnalysis(
        work_item_key="PROJ-1",
        clarity_score=4,
        completeness_score=3,
        actionability_score=4,
        testability_score=3,
        overall_score=3.5,
        risk_level="Medium",
        improvement_suggestions=[
            "Add more specific acceptance criteria",
            "Include technical implementation details",
        ],
        analysis_cost=0.02,
    )


@pytest.fixture
def sample_cross_epic_report():
    """Create a sample cross-epic analysis report."""
    return CrossEpicReport(misplaced_items=[], epics_analyzed=2, processing_cost=0.15)


def create_work_item(
    key: str,
    summary: str,
    issue_type: str = "Story",
    epic_key: Optional[str] = None,
    parent_key: Optional[str] = None,
    base_time: Optional[datetime] = None,
) -> WorkItem:
    """Create a test work item with sensible defaults."""
    if base_time is None:
        base_time = datetime.now(timezone.utc)

    return WorkItem(
        key=key,
        summary=summary,
        issue_type=issue_type,
        status="Open",
        created=base_time,
        updated=base_time,
        reporter="test@example.com",
        description=f"Description for {key}",
        parent_key=parent_key,
        epic_key=epic_key,
        assignee="test@example.com",
        embedding=[0.1, 0.2, 0.3] * 100,
    )


def create_mock_scroll_result(work_items_data: List[Dict[str, Any]]):
    """Create a mock Qdrant scroll result from work items data."""
    mock_points = []
    for item_data in work_items_data:
        mock_point = Mock()
        mock_point.payload = item_data
        mock_point.vector = item_data.get("embedding", [0.1, 0.2, 0.3] * 100)
        mock_points.append(mock_point)

    return mock_points, None  # (points, next_page_offset)


def create_mock_search_result(work_items_data: List[Dict[str, Any]], scores: List[float]):
    """Create a mock Qdrant search result from work items data and scores."""
    mock_results = []
    for item_data, score in zip(work_items_data, scores):
        mock_result = Mock()
        mock_result.payload = item_data
        mock_result.score = score
        mock_results.append(mock_result)

    return mock_results


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "functional: mark test as a functional test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
