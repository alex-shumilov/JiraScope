"""Pytest configuration and fixtures."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock
from src.jirascope.core.config import Config
from src.jirascope.models import WorkItem, AnalysisResult

# Import analysis fixtures to make them available
from .fixtures.analysis_fixtures import *


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return Config(
        jira_mcp_endpoint="http://localhost:8080/mcp",
        lmstudio_endpoint="http://localhost:1234/v1",
        qdrant_url="http://localhost:6333",
        claude_api_key="test-key",
        claude_model="claude-3-5-sonnet-20241022",
        embedding_batch_size=2,
        jira_batch_size=10,
        similarity_threshold=0.8,
        report_retention_days=30,
        cost_tracking=True,
        jira_dry_run=True
    )


# sample_work_items fixture is imported from analysis_fixtures


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] * 128,
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 128,
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 128
    ]


@pytest.fixture
def mock_mcp_client():
    """Mock MCP client for testing."""
    mock = AsyncMock()
    mock.get_work_items = AsyncMock()
    mock.get_work_item = AsyncMock()
    mock.update_work_item = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_lmstudio_client():
    """Mock LMStudio client for testing."""
    mock = AsyncMock()
    mock.generate_embeddings = AsyncMock()
    mock.health_check = AsyncMock(return_value=True)
    mock.calculate_similarity = Mock(return_value=0.85)
    return mock


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    mock = AsyncMock()
    mock.initialize_collection = AsyncMock()
    mock.store_work_items = AsyncMock()
    mock.search_similar_work_items = AsyncMock()
    mock.get_work_item_by_key = AsyncMock()
    mock.delete_work_item = AsyncMock(return_value=True)
    mock.health_check = AsyncMock(return_value=True)
    mock.get_collection_stats = AsyncMock(return_value={
        "points_count": 100,
        "segments_count": 1,
        "disk_data_size": 1024,
        "ram_data_size": 512
    })
    return mock


@pytest.fixture
def mock_claude_client():
    """Mock Claude client for testing."""
    mock = AsyncMock()
    mock.analyze_work_item = AsyncMock()
    mock.calculate_cost = Mock(return_value=0.001)
    mock.get_session_cost = Mock(return_value=0.05)
    mock.reset_session_cost = Mock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock()
    return mock


@pytest.fixture
def sample_analysis_result():
    """Sample analysis result for testing."""
    return AnalysisResult(
        work_item_key="PROJ-1",
        analysis_type="complexity",
        confidence=0.85,
        insights={
            "technical_complexity": 7,
            "business_complexity": 5,
            "risk_level": "medium",
            "dependencies": ["authentication", "database"],
            "effort_estimate": "5-8 days",
            "reasoning": "Complex authentication logic with database dependencies"
        },
        cost=0.0015
    )


@pytest.fixture
def mock_httpx_responses():
    """Mock HTTP responses for external API calls."""
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
                        "created": "2023-01-01T00:00:00.000Z",
                        "updated": "2023-01-01T00:00:00.000Z",
                        "reporter": {"displayName": "Test User"},
                        "assignee": {"displayName": "Test Assignee"},
                        "components": [{"name": "Backend"}],
                        "labels": ["test"]
                    }
                }
            ]
        },
        "lmstudio_embeddings": {
            "data": [
                {"embedding": [0.1, 0.2, 0.3] * 341}  # 1023 dimensions
            ]
        },
        "lmstudio_models": {
            "data": [
                {"id": "BAAI/bge-large-en-v1.5"}
            ]
        }
    }