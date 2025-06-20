"""Tests for the RAG quality testing system."""

import pytest
from unittest.mock import AsyncMock, patch

from jirascope.pipeline.rag_quality_tester import (
    RAGQualityTester, 
    RagTestQuery, 
    RAGQualityReport
)
from jirascope.core.config import Config


@pytest.fixture
def mock_config():
    """Test configuration."""
    return Config(
        jira_mcp_endpoint="http://test.com"
    )


@pytest.fixture
def sample_test_queries():
    """Sample test queries for testing."""
    return [
        RagTestQuery(
            id="auth_query",
            query_text="user authentication and login",
            expected_work_items=["PROJ-123", "PROJ-456"],
            minimum_similarity=0.7,
            category="functional",
            description="Test for authentication features"
        ),
        RagTestQuery(
            id="db_query",
            query_text="database migration",
            expected_work_items=["PROJ-789", "PROJ-012"],
            minimum_similarity=0.6,
            category="technical",
            description="Test for database tasks"
        )
    ]


@pytest.mark.asyncio
async def test_run_quality_tests_success(mock_config, sample_test_queries):
    """Test successful RAG quality testing."""
    tester = RAGQualityTester(mock_config, sample_test_queries)
    
    # Mock successful search results that match both test queries' expected items
    auth_mock_results = [
        {
            "score": 0.8,
            "work_item": {
                "key": "PROJ-123",
                "summary": "Login functionality",
                "issue_type": "Story"
            }
        },
        {
            "score": 0.7, 
            "work_item": {
                "key": "PROJ-456",
                "summary": "User authentication",
                "issue_type": "Task"
            }
        }
    ]
    
    db_mock_results = [
        {
            "score": 0.8,
            "work_item": {
                "key": "PROJ-789",
                "summary": "Database migration",
                "issue_type": "Story"
            }
        },
        {
            "score": 0.7, 
            "work_item": {
                "key": "PROJ-012",
                "summary": "Schema updates",
                "issue_type": "Task"
            }
        }
    ]
    
    mock_lm_client = AsyncMock()
    mock_lm_client.generate_embeddings.return_value = [[0.1] * 1024]
    
    mock_qdrant_client = AsyncMock()
    # Use an index counter to return different results for each call
    call_index = 0
    def search_side_effect(*args, **kwargs):
        nonlocal call_index
        result = auth_mock_results if call_index == 0 else db_mock_results
        call_index += 1
        return result
        
    mock_qdrant_client.search_similar_work_items.side_effect = search_side_effect
    
    # Mock context managers
    mock_lm_context = AsyncMock()
    mock_lm_context.__aenter__.return_value = mock_lm_client
    mock_lm_context.__aexit__.return_value = None
    
    mock_qdrant_context = AsyncMock()
    mock_qdrant_context.__aenter__.return_value = mock_qdrant_client  
    mock_qdrant_context.__aexit__.return_value = None
    
    with patch('jirascope.pipeline.rag_quality_tester.LMStudioClient', return_value=mock_lm_context):
        with patch('jirascope.pipeline.rag_quality_tester.QdrantVectorClient', return_value=mock_qdrant_context):
            report = await tester.run_quality_tests()
    
    assert isinstance(report, RAGQualityReport)
    assert report.total_tests == 2
    # Both should pass with good results because mock returns all expected items
    assert report.passed_tests == 2  
    assert report.overall_f1_score > 0.5
    assert len(report.test_results) == 2
    
    # Check individual results
    for result in report.test_results:
        assert result.f1_score > 0  # Should have positive F1 scores
        assert result.precision > 0
        assert result.recall > 0


@pytest.mark.asyncio
async def test_validate_analysis_consistency(mock_config):
    """Test embedding consistency validation."""
    tester = RAGQualityTester(mock_config)
    
    # First test: high consistency
    # Mock _get_test_work_items to return predictable test items
    tester._get_test_work_items = AsyncMock(return_value=[
        AsyncMock(
            key="TEST-1",
            summary="Test item 1",
            description="Test description 1"
        ),
        AsyncMock(
            key="TEST-2",
            summary="Test item 2",
            description="Test description 2"
        )
    ])
    
    # Custom implementation for _cosine_similarity to force high consistency (above 0.99)
    tester._cosine_similarity = lambda vec1, vec2: 1.0
    
    # Run the test
    with patch('jirascope.pipeline.rag_quality_tester.LMStudioClient') as mock_lm_class:
        mock_lm_client = AsyncMock()
        mock_lm_client.generate_embeddings.return_value = [[0.1] * 1024]
        mock_lm_context = AsyncMock()
        mock_lm_context.__aenter__.return_value = mock_lm_client
        mock_lm_context.__aexit__.return_value = None
        mock_lm_class.return_value = mock_lm_context
        
        report = await tester.validate_analysis_consistency()
    
    # Should be highly consistent
    assert report.overall_consistency > 0.9
    assert report.consistent_items > 0
    
    # Second test: low consistency
    # Custom implementation for _cosine_similarity to force low consistency
    tester._cosine_similarity = lambda vec1, vec2: 0.5
    
    # Run the test
    with patch('jirascope.pipeline.rag_quality_tester.LMStudioClient') as mock_lm_class:
        mock_lm_client = AsyncMock()
        mock_lm_client.generate_embeddings.return_value = [[0.1] * 1024]
        mock_lm_context = AsyncMock()
        mock_lm_context.__aenter__.return_value = mock_lm_client
        mock_lm_context.__aexit__.return_value = None
        mock_lm_class.return_value = mock_lm_context
        
        report = await tester.validate_analysis_consistency()
    
    # Should have lower consistency
    assert report.overall_consistency < 0.9  


@pytest.mark.asyncio
async def test_benchmark_embedding_performance(mock_config):
    """Test performance benchmarking."""
    tester = RAGQualityTester(mock_config)
    
    # Mock embeddings generation
    mock_lm_client = AsyncMock()
    mock_lm_client.generate_embeddings.return_value = [[0.1] * 1024]
    
    # Mock context manager
    mock_lm_context = AsyncMock()
    mock_lm_context.__aenter__.return_value = mock_lm_client
    mock_lm_context.__aexit__.return_value = None
    
    # Mock _get_test_work_items to return test items
    tester._get_test_work_items = AsyncMock(return_value=[
        AsyncMock(
            key=f"TEST-{i}",
            summary=f"Test item {i}",
            description=f"Test description {i}"
        )
        for i in range(100)  # 100 test items
    ])
    
    with patch('jirascope.pipeline.rag_quality_tester.LMStudioClient', return_value=mock_lm_context):
        benchmark = await tester.benchmark_embedding_performance()
    
    assert len(benchmark.results) > 0
    assert benchmark.optimal_batch_size > 0
    assert benchmark.recommendation is not None
    
    # Verify batch sizes in results match expected values
    batch_sizes = [result.batch_size for result in benchmark.results]
    assert 8 in batch_sizes
    assert 16 in batch_sizes
    assert 32 in batch_sizes
    assert 64 in batch_sizes