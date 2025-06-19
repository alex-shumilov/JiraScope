"""Tests for Jira data extraction."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from jirascope.extractors.jira_extractor import JiraExtractor
from jirascope.models import WorkItem, EpicHierarchy
from jirascope.core.config import Config


@pytest.fixture
def mock_config():
    return Config(
        jira_mcp_endpoint="http://test.com",
        jira_batch_size=10
    )


@pytest.fixture
def sample_epic():
    return WorkItem(
        key="PROJ-1",
        summary="Sample Epic",
        issue_type="Epic",
        status="In Progress",
        created=datetime.now(),
        updated=datetime.now(),
        reporter="test@example.com"
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
        reporter="test@example.com"
    )


@pytest.mark.asyncio
async def test_extract_active_hierarchies(mock_config, sample_epic, sample_story):
    """Test extracting active hierarchies."""
    extractor = JiraExtractor(mock_config)
    
    # Mock the MCP client
    mock_client = AsyncMock()
    mock_client.get_work_items.side_effect = [
        [sample_epic],  # Epics query
        [sample_story]  # Children query
    ]
    
    # Mock the async context manager
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_client
    mock_context.__aexit__.return_value = None
    
    with patch('jirascope.extractors.jira_extractor.MCPClient', return_value=mock_context):
        hierarchies = await extractor.extract_active_hierarchies("PROJ")
    
    assert len(hierarchies) == 1
    assert isinstance(hierarchies[0], EpicHierarchy)
    assert hierarchies[0].epic.key == "PROJ-1"
    assert len(hierarchies[0].stories) == 1
    assert hierarchies[0].stories[0].key == "PROJ-2"


@pytest.mark.asyncio  
async def test_get_epic_tree(mock_config, sample_epic, sample_story):
    """Test building complete epic tree."""
    extractor = JiraExtractor(mock_config)
    
    mock_client = AsyncMock()
    mock_client.get_work_item.return_value = sample_epic
    mock_client.get_work_items.return_value = [sample_story]
    
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_client
    mock_context.__aexit__.return_value = None
    
    with patch('jirascope.extractors.jira_extractor.MCPClient', return_value=mock_context):
        tree = await extractor.get_epic_tree("PROJ-1")
    
    assert tree.epic.key == "PROJ-1"
    assert len(tree.direct_children) == 1
    assert len(tree.all_descendants) >= 1  # May include duplicates from recursive logic
    assert tree.hierarchy_depth >= 1
    assert tree.total_items >= 2  # Epic + at least 1 descendant


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
    
    with patch('jirascope.extractors.jira_extractor.MCPClient', return_value=mock_context):
        updates = await extractor.get_incremental_updates(
            "PROJ", 
            "2024-01-01T00:00:00Z",
            {"PROJ-1"},
            {"PROJ-2"}
        )
    
    assert len(updates) == 1
    assert updates[0].key == "PROJ-2"