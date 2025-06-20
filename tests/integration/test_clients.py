"""Integration tests for client classes."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from src.jirascope.clients import MCPClient, LMStudioClient, QdrantVectorClient, ClaudeClient


@pytest.mark.asyncio
async def test_mcp_client_get_work_items(mock_config, mock_httpx_responses):
    """Test MCP client work item retrieval."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.json.return_value = mock_httpx_responses["jira_search"]
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        async with MCPClient(mock_config) as client:
            work_items = await client.get_work_items("project = PROJ")
            
            assert len(work_items) == 1
            assert work_items[0].key == "PROJ-1"
            assert work_items[0].summary == "Test issue"
            assert work_items[0].issue_type == "Story"


@pytest.mark.asyncio
async def test_mcp_client_update_work_item_dry_run(mock_config):
    """Test MCP client dry run mode."""
    mock_config.jira_dry_run = True
    
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        async with MCPClient(mock_config) as client:
            result = await client.update_work_item("PROJ-1", {"summary": "Updated"})
            
            assert result is True
            mock_client.put.assert_not_called()  # Should not make actual API call


@pytest.mark.asyncio
async def test_lmstudio_client_health_check(mock_config, mock_httpx_responses):
    """Test LMStudio client health check."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.json.return_value = mock_httpx_responses["lmstudio_models"]
        mock_response.raise_for_status = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        async with LMStudioClient(mock_config) as client:
            health = await client.health_check()
            
            assert health is True
            mock_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_lmstudio_client_generate_embeddings(mock_config, mock_httpx_responses):
    """Test LMStudio client embedding generation."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.json.return_value = mock_httpx_responses["lmstudio_embeddings"]
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        async with LMStudioClient(mock_config) as client:
            embeddings = await client.generate_embeddings(["test text"])
            
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 1023  # 341 * 3
            mock_client.post.assert_called_once()


def test_lmstudio_client_calculate_similarity(mock_config):
    """Test LMStudio client similarity calculation."""
    client = LMStudioClient(mock_config)
    
    # Test identical vectors
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    similarity = client.calculate_similarity(vec1, vec2)
    assert similarity == 1.0
    
    # Test orthogonal vectors
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    similarity = client.calculate_similarity(vec1, vec2)
    assert similarity == 0.0
    
    # Test zero vectors
    vec1 = [0.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    similarity = client.calculate_similarity(vec1, vec2)
    assert similarity == 0.0


@pytest.mark.asyncio
async def test_qdrant_client_initialization(mock_config):
    """Test Qdrant client initialization."""
    with patch("qdrant_client.QdrantClient") as mock_qdrant:
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_qdrant.return_value = mock_client
        
        async with QdrantVectorClient(mock_config) as client:
            assert client.collection_name == "jirascope_work_items"
            mock_client.create_collection.assert_called_once()


@pytest.mark.asyncio
async def test_qdrant_client_store_work_items(mock_config, sample_work_items, sample_embeddings):
    """Test Qdrant client work item storage."""
    with patch("qdrant_client.QdrantClient") as mock_qdrant:
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(name="jirascope_work_items")]
        mock_client.get_collections.return_value = mock_collections
        mock_qdrant.return_value = mock_client
        
        async with QdrantVectorClient(mock_config) as client:
            await client.store_work_items(sample_work_items[:1], sample_embeddings[:1])
            
            mock_client.upsert.assert_called_once()
            call_args = mock_client.upsert.call_args
            points = call_args[1]["points"]
            assert len(points) == 1
            assert points[0].payload["key"] == "TEST-1"


def test_claude_client_calculate_cost(mock_config):
    """Test Claude client cost calculation."""
    client = ClaudeClient(mock_config)
    
    cost = client.calculate_cost(input_tokens=1000, output_tokens=500)
    expected = (1000 * 0.000003) + (500 * 0.000015)
    assert cost == expected


@pytest.mark.asyncio
async def test_claude_client_analyze_work_item(mock_config, sample_work_items):
    """Test Claude client work item analysis."""
    with patch.object(ClaudeClient, "__init__", return_value=None):
        client = ClaudeClient.__new__(ClaudeClient)
        client.config = mock_config
        client.session_cost = 0.0
        
        # Mock the Anthropic client
        mock_anthropic = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text='{"confidence": 0.8, "complexity": 5}')]
        mock_response.usage = Mock(input_tokens=100, output_tokens=50)
        mock_anthropic.messages.create.return_value = mock_response
        client.client = mock_anthropic
        
        result = await client.analyze_work_item(sample_work_items[0], "complexity")
        
        assert result.work_item_key == "TEST-1"
        assert result.analysis_type == "complexity"
        assert result.cost > 0
        assert "confidence" in result.insights