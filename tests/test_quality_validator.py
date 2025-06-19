"""Tests for embedding quality validator."""

import pytest
from unittest.mock import AsyncMock, patch

from jirascope.pipeline.quality_validator import EmbeddingQualityValidator
from jirascope.models import QualityReport
from jirascope.core.config import Config


@pytest.fixture
def mock_config():
    return Config(
        jira_mcp_endpoint="http://test.com"
    )


@pytest.fixture  
def sample_test_queries():
    return [
        "authentication and login",
        "database migration",
        "user interface"
    ]


@pytest.mark.asyncio
async def test_validate_embedding_quality_success(mock_config, sample_test_queries):
    """Test successful quality validation."""
    validator = EmbeddingQualityValidator(mock_config, sample_test_queries)
    
    # Mock successful search results
    mock_search_results = [
        {
            "score": 0.8,
            "work_item": {
                "key": "PROJ-1",
                "summary": "Login functionality",
                "issue_type": "Story"
            }
        },
        {
            "score": 0.7, 
            "work_item": {
                "key": "PROJ-2",
                "summary": "User authentication",
                "issue_type": "Task"
            }
        }
    ]
    
    mock_lm_client = AsyncMock()
    mock_lm_client.generate_embeddings.return_value = [[0.1] * 1024]
    
    mock_qdrant_client = AsyncMock()
    mock_qdrant_client.search_similar_work_items.return_value = mock_search_results
    
    # Mock context managers
    mock_lm_context = AsyncMock()
    mock_lm_context.__aenter__.return_value = mock_lm_client
    mock_lm_context.__aexit__.return_value = None
    
    mock_qdrant_context = AsyncMock()
    mock_qdrant_context.__aenter__.return_value = mock_qdrant_client  
    mock_qdrant_context.__aexit__.return_value = None
    
    with patch('jirascope.pipeline.quality_validator.LMStudioClient', return_value=mock_lm_context):
        with patch('jirascope.pipeline.quality_validator.QdrantVectorClient', return_value=mock_qdrant_context):
            report = await validator.validate_embedding_quality()
    
    assert isinstance(report, QualityReport)
    assert report.total_tests == 3
    assert report.passed_tests == 3  # All should pass with good results
    assert report.overall_score == 100.0
    assert len(report.results) == 3
    
    # Check individual results
    for result in report.results:
        assert result["passed"]
        assert result["results_count"] == 2
        assert result["avg_similarity"] == 0.75  # (0.8 + 0.7) / 2


@pytest.mark.asyncio
async def test_validate_embedding_quality_poor_results(mock_config, sample_test_queries):
    """Test quality validation with poor results."""
    validator = EmbeddingQualityValidator(mock_config, sample_test_queries)
    
    # Mock poor search results (low similarity)
    mock_search_results = [
        {
            "score": 0.1,  # Very low similarity
            "work_item": {
                "key": "PROJ-1", 
                "summary": "Unrelated item",
                "issue_type": "Bug"
            }
        }
    ]
    
    mock_lm_client = AsyncMock()
    mock_lm_client.generate_embeddings.return_value = [[0.1] * 1024]
    
    mock_qdrant_client = AsyncMock()
    mock_qdrant_client.search_similar_work_items.return_value = mock_search_results
    
    mock_lm_context = AsyncMock()
    mock_lm_context.__aenter__.return_value = mock_lm_client
    mock_lm_context.__aexit__.return_value = None
    
    mock_qdrant_context = AsyncMock()
    mock_qdrant_context.__aenter__.return_value = mock_qdrant_client
    mock_qdrant_context.__aexit__.return_value = None
    
    with patch('jirascope.pipeline.quality_validator.LMStudioClient', return_value=mock_lm_context):
        with patch('jirascope.pipeline.quality_validator.QdrantVectorClient', return_value=mock_qdrant_context):
            report = await validator.validate_embedding_quality()
    
    assert report.passed_tests == 0  # All should fail due to low similarity
    assert report.overall_score == 0.0
    assert len(report.recommendations) > 0
    
    # Should recommend improvements
    rec_text = " ".join(report.recommendations)
    assert "poor" in rec_text.lower() or "low similarity" in rec_text.lower()


@pytest.mark.asyncio
async def test_validate_embedding_quality_no_results(mock_config, sample_test_queries):
    """Test quality validation with no search results."""
    validator = EmbeddingQualityValidator(mock_config, sample_test_queries)
    
    mock_lm_client = AsyncMock()
    mock_lm_client.generate_embeddings.return_value = [[0.1] * 1024]
    
    mock_qdrant_client = AsyncMock()
    mock_qdrant_client.search_similar_work_items.return_value = []  # No results
    
    mock_lm_context = AsyncMock()
    mock_lm_context.__aenter__.return_value = mock_lm_client
    mock_lm_context.__aexit__.return_value = None
    
    mock_qdrant_context = AsyncMock()
    mock_qdrant_context.__aenter__.return_value = mock_qdrant_client
    mock_qdrant_context.__aexit__.return_value = None
    
    with patch('jirascope.pipeline.quality_validator.LMStudioClient', return_value=mock_lm_context):
        with patch('jirascope.pipeline.quality_validator.QdrantVectorClient', return_value=mock_qdrant_context):
            report = await validator.validate_embedding_quality()
    
    assert report.passed_tests == 0
    assert report.overall_score == 0.0
    
    # Should have recommendations about no results
    rec_text = " ".join(report.recommendations)
    assert "no results" in rec_text.lower()


@pytest.mark.asyncio
async def test_test_single_query(mock_config):
    """Test single query testing."""
    validator = EmbeddingQualityValidator(mock_config)
    
    mock_lm_client = AsyncMock()
    mock_lm_client.generate_embeddings.return_value = [[0.1] * 1024]
    
    mock_qdrant_client = AsyncMock()
    mock_qdrant_client.search_similar_work_items.return_value = [
        {
            "score": 0.85,
            "work_item": {
                "key": "PROJ-1",
                "summary": "Authentication system", 
                "issue_type": "Story"
            }
        }
    ]
    
    result = await validator._test_single_query(
        "user authentication",
        mock_lm_client,
        mock_qdrant_client
    )
    
    assert result["query"] == "user authentication"
    assert result["results_count"] == 1
    assert result["avg_similarity"] == 0.85
    assert result["passed"] is True
    assert len(result["top_results"]) == 1
    assert result["top_results"][0]["key"] == "PROJ-1"


@pytest.mark.asyncio 
async def test_run_performance_test(mock_config, sample_test_queries):
    """Test performance testing."""
    validator = EmbeddingQualityValidator(mock_config, sample_test_queries)
    
    mock_lm_client = AsyncMock()
    mock_lm_client.generate_embeddings.return_value = [[0.1] * 1024]
    
    mock_qdrant_client = AsyncMock()
    mock_qdrant_client.search_similar_work_items.return_value = []
    
    mock_lm_context = AsyncMock()
    mock_lm_context.__aenter__.return_value = mock_lm_client
    mock_lm_context.__aexit__.return_value = None
    
    mock_qdrant_context = AsyncMock()
    mock_qdrant_context.__aenter__.return_value = mock_qdrant_client
    mock_qdrant_context.__aexit__.return_value = None
    
    with patch('jirascope.pipeline.quality_validator.LMStudioClient', return_value=mock_lm_context):
        with patch('jirascope.pipeline.quality_validator.QdrantVectorClient', return_value=mock_qdrant_context):
            metrics = await validator.run_performance_test(num_queries=2)
    
    assert "total_test_time" in metrics
    assert "queries_per_second" in metrics
    assert "avg_query_time" in metrics
    assert "avg_embedding_time" in metrics
    assert "avg_search_time" in metrics
    
    assert metrics["queries_per_second"] > 0
    assert metrics["avg_query_time"] > 0


def test_generate_recommendations():
    """Test recommendation generation."""
    validator = EmbeddingQualityValidator(Config(jira_mcp_endpoint="test"))
    
    # Test with good results - use enough results to avoid the sparse data warning
    good_results = [
        {"results_count": 5, "avg_similarity": 0.8, "passed": True},
        {"results_count": 5, "avg_similarity": 0.7, "passed": True}, 
        {"results_count": 5, "avg_similarity": 0.6, "passed": True},
        {"results_count": 5, "avg_similarity": 0.8, "passed": True},
        {"results_count": 5, "avg_similarity": 0.7, "passed": True},
        {"results_count": 5, "avg_similarity": 0.6, "passed": True},
        {"results_count": 5, "avg_similarity": 0.8, "passed": True},
        {"results_count": 5, "avg_similarity": 0.7, "passed": True},
        {"results_count": 5, "avg_similarity": 0.6, "passed": True},
        {"results_count": 5, "avg_similarity": 0.8, "passed": True}
    ]
    
    recommendations = validator._generate_recommendations(good_results, 100.0)
    assert len(recommendations) > 0
    assert any("good" in rec.lower() or "no specific improvements needed" in rec.lower() for rec in recommendations)
    
    # Test with poor results  
    poor_results = [
        {"results_count": 0, "avg_similarity": 0.0, "passed": False},
        {"results_count": 1, "avg_similarity": 0.1, "passed": False}
    ]
    
    recommendations = validator._generate_recommendations(poor_results, 0.0)
    assert len(recommendations) > 0 
    assert any("poor" in rec.lower() for rec in recommendations)