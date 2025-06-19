"""Tests for embedding processor."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from jirascope.pipeline.embedding_processor import EmbeddingProcessor, AdaptiveBatcher
from jirascope.models import WorkItem, ProcessingResult
from jirascope.core.config import Config


@pytest.fixture
def mock_config():
    return Config(
        jira_mcp_endpoint="http://test.com",
        embedding_batch_size=4
    )


@pytest.fixture
def sample_work_items():
    return [
        WorkItem(
            key="PROJ-1",
            summary="Short summary",
            issue_type="Story",
            status="Done",
            created=datetime.now(),
            updated=datetime.now(), 
            reporter="test@example.com"
        ),
        WorkItem(
            key="PROJ-2",
            summary="Very long summary with lots of details about what this work item is supposed to accomplish and how it fits into the larger project goals",
            description="This is a detailed description with even more information about the work item requirements, acceptance criteria, and implementation details that should be considered during development.",
            issue_type="Epic",
            status="In Progress",
            created=datetime.now(),
            updated=datetime.now(),
            reporter="test@example.com"
        )
    ]


def test_adaptive_batcher_calculates_optimal_size(sample_work_items):
    """Test adaptive batcher calculates optimal batch size."""
    batcher = AdaptiveBatcher(base_batch_size=32)
    
    # Short items should use full batch size
    short_items = [sample_work_items[0]] * 5
    size = batcher.calculate_optimal_batch_size(short_items)
    assert size == 32
    
    # Long items should use smaller batch size  
    long_items = [sample_work_items[1]] * 5
    size = batcher.calculate_optimal_batch_size(long_items)
    assert size <= 32  # Should be same or smaller


def test_prepare_embedding_text(sample_work_items):
    """Test text preparation for embeddings."""
    short_item = sample_work_items[0]
    text = EmbeddingProcessor.prepare_embedding_text(short_item) 
    
    assert "Title: Short summary" in text
    assert "Type: Story" in text
    assert "Status: Done" in text
    assert "|" in text  # Separator
    
    # Test with description
    long_item = sample_work_items[1]
    text = EmbeddingProcessor.prepare_embedding_text(long_item)
    
    assert "Description:" in text
    assert len(text) < 2000  # Should be truncated if too long


def test_clean_jira_markup():
    """Test cleaning Jira markup."""
    from jirascope.pipeline.embedding_processor import EmbeddingProcessor
    
    text_with_markup = """
    {code:java}
    public void test() {}
    {code}
    
    {quote}This is a quote{quote}
    
    *Bold text* and _italic text_
    
    [Link text|http://example.com]
    """
    
    cleaned = EmbeddingProcessor._clean_jira_markup(text_with_markup)
    
    assert "{code}" not in cleaned
    assert "{quote}" not in cleaned
    assert "*Bold text*" not in cleaned
    assert "Bold text" in cleaned
    assert "[CODE BLOCK]" in cleaned


@pytest.mark.asyncio
async def test_process_work_items(mock_config, sample_work_items):
    """Test processing work items with embeddings."""
    processor = EmbeddingProcessor(mock_config)
    
    # Mock clients
    mock_lm_client = AsyncMock()
    mock_lm_client.generate_embeddings.return_value = [
        [0.1] * 1024,  # Mock embedding for first item
        [0.2] * 1024   # Mock embedding for second item  
    ]
    
    mock_qdrant_client = AsyncMock()
    mock_qdrant_client.store_work_items.return_value = None
    
    # Mock context managers
    mock_lm_context = AsyncMock()
    mock_lm_context.__aenter__.return_value = mock_lm_client
    mock_lm_context.__aexit__.return_value = None
    
    mock_qdrant_context = AsyncMock() 
    mock_qdrant_context.__aenter__.return_value = mock_qdrant_client
    mock_qdrant_context.__aexit__.return_value = None
    
    with patch('jirascope.pipeline.embedding_processor.LMStudioClient', return_value=mock_lm_context):
        with patch('jirascope.pipeline.embedding_processor.QdrantVectorClient', return_value=mock_qdrant_context):
            # Mock the filtering method to return all items
            processor._filter_unchanged_items = MagicMock(return_value=sample_work_items)
            
            result = await processor.process_work_items(sample_work_items)
    
    assert isinstance(result, ProcessingResult)
    assert result.processed_items == 2
    assert result.failed_items == 0
    assert result.total_cost > 0
    
    # Verify embeddings were generated and stored
    mock_lm_client.generate_embeddings.assert_called()
    mock_qdrant_client.store_work_items.assert_called()


def test_calculate_item_hash(mock_config):
    """Test item hash calculation for change detection."""
    processor = EmbeddingProcessor(mock_config)
    
    item = WorkItem(
        key="PROJ-1",
        summary="Test item", 
        issue_type="Story",
        status="Done",
        created=datetime.now(),
        updated=datetime.now(),
        reporter="test@example.com"
    )
    
    hash1 = processor._calculate_item_hash(item)
    hash2 = processor._calculate_item_hash(item)
    
    # Same item should produce same hash
    assert hash1 == hash2
    assert len(hash1) == 32  # MD5 hash length
    
    # Changed item should produce different hash
    item.summary = "Changed summary"
    hash3 = processor._calculate_item_hash(item)
    assert hash1 != hash3


@pytest.mark.asyncio
async def test_process_batch_error_handling(mock_config, sample_work_items):
    """Test batch processing error handling."""
    processor = EmbeddingProcessor(mock_config)
    
    # Mock LM client that fails
    mock_lm_client = AsyncMock()
    mock_lm_client.generate_embeddings.side_effect = Exception("LM Studio error")
    
    mock_qdrant_client = AsyncMock()
    
    result = await processor._process_batch(
        sample_work_items,
        mock_lm_client,
        mock_qdrant_client
    )
    
    assert result.processed_items == 0
    assert result.failed_items == len(sample_work_items)
    assert len(result.errors) > 0
    assert "LM Studio error" in result.errors[0]