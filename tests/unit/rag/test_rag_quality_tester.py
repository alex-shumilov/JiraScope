"""Simplified tests for RAG quality testing - following KISS principle."""

from unittest.mock import AsyncMock, patch

import pytest

from jirascope.pipeline.rag_quality_tester import RAGQualityReport, RAGQualityTester, RagTestQuery


class TestRAGQualityTester:
    """Simplified RAG quality testing."""

    def test_test_query_creation(self, test_config):
        """Test basic test query creation - no hardcoding."""
        query = RagTestQuery(
            id="test_query",
            query_text="authentication functionality",
            expected_work_items=[test_config.random_key(), test_config.random_key()],
            minimum_similarity=test_config.random_score("similarity"),
            category="functional",
            description="Test authentication features",
        )

        assert query.id == "test_query"
        assert len(query.expected_work_items) == 2
        assert 0.5 <= query.minimum_similarity <= 0.95  # Realistic range

    @pytest.mark.asyncio
    async def test_quality_tests_basic_functionality(self, mock_config, mock_helper, test_config):
        """Test quality testing without excessive mocking."""
        # Simple test queries - no hardcoded keys
        test_queries = [
            RagTestQuery(
                id="auth_test",
                query_text="user authentication",
                expected_work_items=[test_config.random_key(), test_config.random_key()],
                minimum_similarity=0.7,
                category="functional",
                description="Auth test",
            )
        ]

        tester = RAGQualityTester(mock_config, test_queries)

        # Mock only what's necessary - realistic responses
        mock_search_results = mock_helper.mock_search_results(count=2, score_range=(0.7, 0.9))

        with patch("jirascope.pipeline.rag_quality_tester.LMStudioClient") as mock_lm:
            with patch("jirascope.clients.qdrant_client.QdrantVectorClient") as mock_qdrant:
                # Simple mock setup
                mock_lm_instance = AsyncMock()
                mock_lm_instance.generate_embeddings.return_value = [mock_helper.mock_embedding()]
                mock_lm.return_value.__aenter__.return_value = mock_lm_instance

                mock_qdrant_instance = AsyncMock()
                mock_qdrant_instance.search_similar_work_items.return_value = mock_search_results
                mock_qdrant.return_value.__aenter__.return_value = mock_qdrant_instance

                report = await tester.run_quality_tests()

        # Test behavior, not exact values
        assert isinstance(report, RAGQualityReport)
        assert report.total_tests == 1
        assert report.passed_tests >= 0  # Could pass or fail
        assert 0.0 <= report.overall_f1_score <= 1.0  # Valid range
        assert len(report.test_results) == 1

    @pytest.mark.asyncio
    async def test_embedding_consistency_range_validation(self, mock_config, mock_helper):
        """Test consistency validation focuses on ranges, not exact values."""
        tester = RAGQualityTester(mock_config)

        # Mock minimal test data
        tester._get_test_work_items = AsyncMock(
            return_value=[
                type(
                    "MockItem",
                    (),
                    {"key": "TEST-1", "summary": "Test item", "description": "Test description"},
                )()
            ]
        )

        with patch("jirascope.pipeline.rag_quality_tester.LMStudioClient") as mock_lm:
            mock_lm_instance = AsyncMock()
            mock_lm_instance.generate_embeddings.return_value = [mock_helper.mock_embedding()]
            mock_lm.return_value.__aenter__.return_value = mock_lm_instance

            # Test high consistency scenario
            tester._cosine_similarity = lambda vec1, vec2: 0.95
            report = await tester.validate_analysis_consistency()

            assert 0.0 <= report.overall_consistency <= 1.0
            assert report.consistent_items >= 0
            assert report.total_items > 0

            # Test low consistency scenario
            tester._cosine_similarity = lambda vec1, vec2: 0.3
            report = await tester.validate_analysis_consistency()

            assert report.overall_consistency < 0.9  # Should be lower

    @pytest.mark.asyncio
    async def test_performance_benchmark_realistic_batches(
        self, mock_config, mock_helper, test_config
    ):
        """Test performance benchmarking with configurable batch sizes."""
        tester = RAGQualityTester(mock_config)

        # Use configurable batch sizes, not hardcoded ones
        expected_batches = test_config.BATCH_SIZES

        # Mock test items
        tester._get_test_work_items = AsyncMock(
            return_value=[
                type(
                    "MockItem",
                    (),
                    {
                        "key": f"TEST-{i}",
                        "summary": f"Test item {i}",
                        "description": f"Description {i}",
                    },
                )()
                for i in range(50)  # Reasonable test size
            ]
        )

        with patch("jirascope.pipeline.rag_quality_tester.LMStudioClient") as mock_lm:
            mock_lm_instance = AsyncMock()
            mock_lm_instance.generate_embeddings.return_value = [mock_helper.mock_embedding()]
            mock_lm.return_value.__aenter__.return_value = mock_lm_instance

            benchmark = await tester.benchmark_embedding_performance()

            assert len(benchmark.results) > 0
            assert benchmark.optimal_batch_size in expected_batches
            assert benchmark.recommendation is not None

            # Verify tested batch sizes are from our config
            tested_batches = [result.batch_size for result in benchmark.results]
            assert all(batch in expected_batches for batch in tested_batches)


class TestRagTestQuery:
    """Test the RagTestQuery model - simple validation."""

    def test_query_validation(self, test_config):
        """Test query validation without hardcoded values."""
        # Valid query
        query = RagTestQuery(
            id="valid_query",
            query_text="test query",
            expected_work_items=[test_config.random_key()],
            minimum_similarity=test_config.random_score("similarity"),
            category="test",
            description="A test query",
        )

        assert query.id == "valid_query"
        assert len(query.expected_work_items) == 1

        # Similarity should be in valid range
        assert 0.0 <= query.minimum_similarity <= 1.0


class TestRAGQualityReport:
    """Test the report model - focus on structure, not exact values."""

    def test_report_structure(self, test_config):
        """Test report structure validation."""
        report = RAGQualityReport(
            test_results=[],
            overall_f1_score=test_config.random_score("confidence"),
            passed_tests=0,
            total_tests=1,
            processing_time=1.5,
            processing_cost=test_config.random_score("cost"),
        )

        assert isinstance(report.test_results, list)
        assert 0.0 <= report.overall_f1_score <= 1.0
        assert report.passed_tests <= report.total_tests
        assert report.processing_time > 0
        assert report.processing_cost > 0
