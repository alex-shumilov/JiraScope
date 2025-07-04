"""Global test configuration and utilities."""

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from src.jirascope.core.config import Config
from src.jirascope.models import WorkItem
from tests.fixtures.analysis_fixtures import AnalysisFixtures


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
    """Provide mock config without hardcoded values."""
    return Config(
        jira_mcp_endpoint="http://localhost:8080/mcp",
        lmstudio_endpoint="http://localhost:1234/v1",
        qdrant_url="http://localhost:6333",
        claude_api_key="test-key",
        embedding_batch_size=32,
        similarity_threshold=0.7,
    )


@pytest.fixture
def simple_work_items(data_builder):
    """Provide simple work items for basic testing."""
    return (
        data_builder.add_story("Simple login functionality")
        .add_bug("UI rendering issue", severity="low")
        .add_epic("User management system", scope="small")
        .build()
    )


# Additional fixtures needed by existing tests
@pytest.fixture
def sample_work_items(data_builder):
    """Provide sample work items matching test expectations."""
    return (
        data_builder.add_story("User authentication", complexity="simple", key="TEST-1")
        .add_bug(summary="Login form validation", severity="medium", key="TEST-2")
        .add_story("Database optimization", complexity="complex", key="TEST-3")
        .add_bug(summary="Performance issues", severity="high", key="TEST-4")
        .add_epic(summary="E-commerce Platform", scope="large", key="TEST-5")
        .add_story("Complete system overhaul", complexity="complex", key="TEST-6")
        .build()
    )


@pytest.fixture
def high_quality_stories(data_builder):
    """Provide high-quality story samples for template inference."""
    return (
        data_builder.add_story("User Profile Management", complexity="medium")
        .add_story("Shopping Cart Functionality", complexity="medium")
        .add_story("Payment Integration", complexity="complex")
        .build()
    )


@pytest.fixture
def high_quality_tasks(data_builder):
    """Provide high-quality task samples for template inference."""
    return (
        data_builder.add_story("Update documentation", complexity="simple")
        .add_story("Configure monitoring", complexity="medium")
        .add_story("Setup CI/CD pipeline", complexity="complex")
        .build()
    )


@pytest.fixture
def sample_embeddings(test_config):
    """Provide sample embeddings for testing."""
    dimension = test_config.EMBEDDING_DIMS["medium"]
    return [[random.uniform(-1, 1) for _ in range(dimension)] for _ in range(6)]


@pytest.fixture
def mock_claude_responses():
    """Provide mock Claude API responses."""
    return {
        "quality_analysis": MockHelper.mock_claude_response("quality_analysis"),
        "template_inference": MockHelper.mock_claude_response("template_inference"),
        "complexity_analysis": {
            "content": '{"complexity_score": 0.7, "factors": ["database", "ui"]}',
            "cost": 0.03,
        },
        "split_analysis": {
            "content": '{"should_split": true, "complexity_score": 0.8, "suggested_splits": [{"suggested_title": "UI Components", "suggested_description": "Frontend components implementation", "estimated_effort": "Medium", "dependencies": []}]}',
            "cost": 0.04,
        },
        "scope_drift": {
            "content": '{"has_drift": true, "drift_score": 0.6, "drift_events": []}',
            "cost": 0.02,
        },
        "tech_debt_cluster_analysis": {
            "content": '{"theme": "Database Performance", "priority_score": 0.8, "estimated_effort": "Medium", "dependencies": []}',
            "cost": 0.05,
        },
        "scope_change_analysis": {
            "content": '{"has_drift": true, "drift_score": 0.7, "drift_events": [{"type": "scope_expansion", "confidence": 0.8}]}',
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


# Mock client fixtures for different analyzers
@pytest.fixture
def mock_claude_client(mock_claude_responses):
    """Provide mock Claude client."""
    client = AsyncMock()
    client.analyze.return_value = AsyncMock(
        content=mock_claude_responses["quality_analysis"]["content"],
        cost=mock_claude_responses["quality_analysis"]["cost"],
    )
    return client


@pytest.fixture
def mock_qdrant_client():
    """Provide mock Qdrant client."""
    client = AsyncMock()
    client.search_similar.return_value = MockHelper.mock_search_results()
    client.scroll_work_items.return_value = ([], None)  # Empty results
    return client


@pytest.fixture
def mock_lmstudio_client():
    """Provide mock LMStudio client."""
    client = AsyncMock()
    client.generate_embeddings.return_value = [MockHelper.mock_embedding()]
    client.health_check.return_value = True
    return client


@pytest.fixture
def mock_mcp_client(sample_work_items):
    """Provide mock MCP client."""
    client = AsyncMock()
    client.get_work_items.return_value = sample_work_items
    client.get_work_item.return_value = sample_work_items[0] if sample_work_items else None
    return client


# Composite fixtures for different analyzer combinations
@pytest.fixture
def mock_clients(request):
    """Dynamic mock_clients based on test class context."""
    # Default to similarity analyzer clients for compatibility
    from unittest.mock import AsyncMock, MagicMock

    # Look at the test class to determine which clients to return
    test_class = request.node.cls.__name__ if request.node.cls else ""

    if "StructuralAnalyzer" in test_class:
        # Structural analyzer needs (qdrant_client, claude_client)
        qdrant_client = AsyncMock()
        claude_client = AsyncMock()

        # Mock Qdrant search for tech debt items
        tech_debt_items = AnalysisFixtures.create_sample_work_items()[2:4]
        mock_points = []

        for item in tech_debt_items:
            point = MagicMock()
            point.payload = item.model_dump()
            point.payload["embedding"] = AnalysisFixtures.create_mock_embeddings()[0]
            mock_points.append(point)

        def mock_scroll(*args, **kwargs):
            return [mock_points], None

        qdrant_client.client.scroll.side_effect = mock_scroll

        # Mock Claude analysis
        claude_client.analyze.return_value = AsyncMock(
            content='{"theme": "Database Performance", "priority_score": 0.8, "estimated_effort": "Medium", "dependencies": []}',
            cost=0.05,
        )

        return qdrant_client, claude_client

    elif "TemporalAnalyzer" in test_class:
        # Temporal analyzer needs (jira_client, lm_client, claude_client)
        jira_client = AsyncMock()
        lm_client = AsyncMock()
        claude_client = AsyncMock()

        # Mock change history
        scope_drift_history = AnalysisFixtures.create_scope_drift_history()
        jira_client.get_work_item.return_value = (
            scope_drift_history[0] if scope_drift_history else None
        )

        # Mock embedding generation
        mock_embeddings = AnalysisFixtures.create_mock_embeddings()
        lm_client.generate_embeddings.return_value = mock_embeddings[:3]
        lm_client.calculate_similarity.return_value = 0.25
        lm_client._generate_batch_embeddings = AsyncMock(return_value=mock_embeddings[:3])

        # Mock Claude analysis
        claude_client.analyze.return_value = AsyncMock(
            content='{"has_drift": true, "drift_score": 0.7, "drift_events": [{"type": "scope_expansion", "confidence": 0.8}]}',
            cost=0.03,
        )

        return jira_client, lm_client, claude_client

    else:
        # Default: similarity analyzer (qdrant + lmstudio)
        qdrant_client = AsyncMock()
        lmstudio_client = AsyncMock()

        qdrant_client.search_similar.return_value = MockHelper.mock_search_results()
        qdrant_client.scroll_work_items.return_value = ([], None)
        lmstudio_client.generate_embeddings.return_value = [MockHelper.mock_embedding()]
        lmstudio_client.health_check.return_value = True

        return qdrant_client, lmstudio_client
