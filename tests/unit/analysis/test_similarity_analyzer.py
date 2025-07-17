"""Tests for similarity analyzer components."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jirascope.analysis.similarity_analyzer import MultiLevelSimilarityDetector, SimilarityAnalyzer
from jirascope.core.config import Config
from jirascope.models import DuplicateCandidate, DuplicateReport
from tests.fixtures.analysis_fixtures import AnalysisFixtures


class TestMultiLevelSimilarityDetector:
    """Test the multi-level similarity detection logic."""

    def test_similarity_thresholds(self, mock_qdrant_client, mock_lmstudio_client):
        """Test that similarity thresholds are correctly defined."""
        detector = MultiLevelSimilarityDetector(mock_qdrant_client, mock_lmstudio_client)
        expected_thresholds = {"exact": 0.95, "high": 0.85, "medium": 0.70, "low": 0.55}
        assert detector.similarity_thresholds == expected_thresholds

    def test_classify_similarity_exact(self, mock_qdrant_client, mock_lmstudio_client):
        """Test exact similarity classification."""
        detector = MultiLevelSimilarityDetector(mock_qdrant_client, mock_lmstudio_client)
        result = detector.classify_similarity(0.96)
        assert result == "exact"

    def test_classify_similarity_high(self, mock_qdrant_client, mock_lmstudio_client):
        """Test high similarity classification."""
        detector = MultiLevelSimilarityDetector(mock_qdrant_client, mock_lmstudio_client)
        result = detector.classify_similarity(0.87)
        assert result == "high"

    def test_classify_similarity_medium(self, mock_qdrant_client, mock_lmstudio_client):
        """Test medium similarity classification."""
        detector = MultiLevelSimilarityDetector(mock_qdrant_client, mock_lmstudio_client)
        result = detector.classify_similarity(0.72)
        assert result == "medium"

    def test_classify_similarity_low(self, mock_qdrant_client, mock_lmstudio_client):
        """Test low similarity classification."""
        detector = MultiLevelSimilarityDetector(mock_qdrant_client, mock_lmstudio_client)
        result = detector.classify_similarity(0.58)
        assert result == "low"

    def test_classify_similarity_below_threshold(self, mock_qdrant_client, mock_lmstudio_client):
        """Test similarity below all thresholds."""
        detector = MultiLevelSimilarityDetector(mock_qdrant_client, mock_lmstudio_client)
        result = detector.classify_similarity(0.30)
        assert result is None

    def test_generate_suggested_action_exact(self, mock_qdrant_client, mock_lmstudio_client):
        """Test suggested action for exact matches."""
        detector = MultiLevelSimilarityDetector(mock_qdrant_client, mock_lmstudio_client)
        action = detector.generate_suggested_action("exact", 0.96)
        assert "merge" in action.lower() or "duplicate" in action.lower()

    def test_generate_suggested_action_high(self, mock_qdrant_client, mock_lmstudio_client):
        """Test suggested action for high similarity."""
        detector = MultiLevelSimilarityDetector(mock_qdrant_client, mock_lmstudio_client)
        action = detector.generate_suggested_action("high", 0.87)
        assert "review" in action.lower()

    def test_generate_suggested_action_medium(self, mock_qdrant_client, mock_lmstudio_client):
        """Test suggested action for medium similarity."""
        detector = MultiLevelSimilarityDetector(mock_qdrant_client, mock_lmstudio_client)
        action = detector.generate_suggested_action("medium", 0.72)
        assert "investigate" in action.lower() or "compare" in action.lower()

    def test_generate_suggested_action_low(self, mock_qdrant_client, mock_lmstudio_client):
        """Test suggested action for low similarity."""
        detector = MultiLevelSimilarityDetector(mock_qdrant_client, mock_lmstudio_client)
        action = detector.generate_suggested_action("low", 0.58)
        assert "monitor" in action.lower() or "track" in action.lower()


class TestSimilarityAnalyzer:
    """Test the full similarity analyzer with mocked dependencies."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return MagicMock(spec=Config)

    @pytest.fixture
    def mock_clients(self):
        """Create mock clients."""
        lm_client = AsyncMock()
        qdrant_client = AsyncMock()

        # Mock embeddings generation
        mock_embeddings = AnalysisFixtures.create_mock_embeddings()
        lm_client.generate_embeddings.return_value = mock_embeddings[:2]  # Return first 2
        lm_client.calculate_similarity.return_value = 0.87  # High similarity

        # Mock Qdrant search
        work_items = AnalysisFixtures.create_sample_work_items()
        qdrant_client.search_similar_work_items.return_value = [
            {"work_item": item.model_dump(), "score": 0.87} for item in work_items[:2]
        ]

        return lm_client, qdrant_client

    @pytest.mark.asyncio
    async def test_find_potential_duplicates_success(
        self, mock_config, mock_clients, sample_work_items
    ):
        """Test successful duplicate detection."""
        lm_client, qdrant_client = mock_clients

        with (
            patch("jirascope.analysis.similarity_analyzer.LMStudioClient", return_value=lm_client),
            patch(
                "jirascope.analysis.similarity_analyzer.QdrantVectorClient",
                return_value=qdrant_client,
            ),
        ):
            async with SimilarityAnalyzer(mock_config) as analyzer:
                # Mock async context managers
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()

                report = await analyzer.find_potential_duplicates(sample_work_items[:2], 0.70)

                assert isinstance(report, DuplicateReport)
                assert report.total_candidates >= 0
                assert "high" in report.candidates_by_level
                assert len(report.candidates_by_level["high"]) > 0

    @pytest.mark.asyncio
    async def test_find_potential_duplicates_no_duplicates(
        self, mock_config, mock_clients, sample_work_items
    ):
        """Test when no duplicates are found."""
        lm_client, qdrant_client = mock_clients

        # Mock Qdrant to return no similar items (empty results)
        qdrant_client.search_similar_work_items.return_value = []

        with (
            patch("jirascope.analysis.similarity_analyzer.LMStudioClient", return_value=lm_client),
            patch(
                "jirascope.analysis.similarity_analyzer.QdrantVectorClient",
                return_value=qdrant_client,
            ),
        ):
            async with SimilarityAnalyzer(mock_config) as analyzer:
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()

                report = await analyzer.find_potential_duplicates(sample_work_items[:2], 0.70)

                assert report.total_candidates == 0
                assert all(
                    len(candidates) == 0 for candidates in report.candidates_by_level.values()
                )

    @pytest.mark.asyncio
    async def test_find_potential_duplicates_single_item(self, mock_config, sample_work_items):
        """Test with single work item (should return empty report)."""
        lm_client = AsyncMock()
        qdrant_client = AsyncMock()

        # Mock empty search results for single item
        qdrant_client.search_similar_work_items.return_value = []

        with (
            patch("jirascope.analysis.similarity_analyzer.LMStudioClient", return_value=lm_client),
            patch(
                "jirascope.analysis.similarity_analyzer.QdrantVectorClient",
                return_value=qdrant_client,
            ),
        ):
            async with SimilarityAnalyzer(mock_config) as analyzer:
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()

                report = await analyzer.find_potential_duplicates([sample_work_items[0]], 0.70)

                assert report.total_candidates == 0

    @pytest.mark.asyncio
    async def test_context_manager_initialization(self, mock_config):
        """Test that async context manager properly initializes clients."""
        with (
            patch("jirascope.analysis.similarity_analyzer.LMStudioClient") as mock_lm,
            patch("jirascope.analysis.similarity_analyzer.QdrantVectorClient") as mock_qdrant,
        ):
            mock_lm_instance = AsyncMock()
            mock_qdrant_instance = AsyncMock()
            mock_lm.return_value = mock_lm_instance
            mock_qdrant.return_value = mock_qdrant_instance

            async with SimilarityAnalyzer(mock_config) as analyzer:
                # Verify clients were created
                assert analyzer.lm_client is not None
                assert analyzer.qdrant_client is not None

                # Verify __aenter__ was called
                mock_lm_instance.__aenter__.assert_called_once()
                mock_qdrant_instance.__aenter__.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, mock_config):
        """Test that async context manager properly cleans up clients."""
        with (
            patch("jirascope.analysis.similarity_analyzer.LMStudioClient") as mock_lm,
            patch("jirascope.analysis.similarity_analyzer.QdrantVectorClient") as mock_qdrant,
        ):
            mock_lm_instance = AsyncMock()
            mock_qdrant_instance = AsyncMock()
            mock_lm.return_value = mock_lm_instance
            mock_qdrant.return_value = mock_qdrant_instance

            async with SimilarityAnalyzer(mock_config) as _:
                pass

            # Verify __aexit__ was called
            mock_lm_instance.__aexit__.assert_called_once()
            mock_qdrant_instance.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_embedding_failure(self, mock_config, sample_work_items):
        """Test handling of embedding generation failures."""
        lm_client = AsyncMock()
        qdrant_client = AsyncMock()

        # Mock embedding failure
        lm_client.generate_embeddings.side_effect = Exception("Embedding API error")

        with (
            patch("jirascope.analysis.similarity_analyzer.LMStudioClient", return_value=lm_client),
            patch(
                "jirascope.analysis.similarity_analyzer.QdrantVectorClient",
                return_value=qdrant_client,
            ),
        ):
            async with SimilarityAnalyzer(mock_config) as analyzer:
                lm_client.__aenter__ = AsyncMock(return_value=lm_client)
                lm_client.__aexit__ = AsyncMock()
                qdrant_client.__aenter__ = AsyncMock(return_value=qdrant_client)
                qdrant_client.__aexit__ = AsyncMock()

                # The analyzer should handle the error or re-raise it
                # If it re-raises, we catch it; if it handles gracefully, we check the result
                try:
                    report = await analyzer.find_potential_duplicates(sample_work_items[:2], 0.70)
                    # If no exception was raised, the analyzer handled it gracefully
                    assert report.total_candidates == 0
                except Exception as e:
                    # If an exception was raised, it should be our mock exception
                    assert "Embedding API error" in str(e)

    def test_duplicate_candidate_creation(self):
        """Test DuplicateCandidate model creation and validation."""
        candidate = DuplicateCandidate(
            original_key="TEST-1",
            duplicate_key="TEST-2",
            similarity_score=0.87,
            confidence_level="high",
            review_priority=2,
            suggested_action="Review for potential merge",
        )

        assert candidate.original_key == "TEST-1"
        assert candidate.duplicate_key == "TEST-2"
        assert candidate.similarity_score == 0.87
        assert candidate.confidence_level == "high"
        assert "review" in candidate.suggested_action.lower()

    def test_duplicate_report_creation(self):
        """Test DuplicateReport model creation and structure."""
        candidates = [
            DuplicateCandidate(
                original_key="TEST-1",
                duplicate_key="TEST-2",
                similarity_score=0.87,
                confidence_level="high",
                review_priority=2,
                suggested_action="Review for potential merge",
            )
        ]

        report = DuplicateReport(
            total_candidates=1,
            candidates_by_level={"high": candidates},
            processing_cost=0.05,
            items_analyzed=10,
        )

        assert report.total_candidates == 1
        assert len(report.candidates_by_level["high"]) == 1
        assert report.processing_cost == 0.05


class TestSimilarityCalculations:
    """Test similarity calculation utilities."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical vectors."""
        from jirascope.clients.lmstudio_client import LMStudioClient

        vector1 = [1.0, 2.0, 3.0]
        vector2 = [1.0, 2.0, 3.0]

        # Mock the calculation method
        client = LMStudioClient(Config(jira_mcp_endpoint="test"))
        similarity = client.calculate_similarity(vector1, vector2)

        # Identical vectors should have similarity close to 1.0
        assert abs(similarity - 1.0) < 0.01

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity with orthogonal vectors."""
        from jirascope.clients.lmstudio_client import LMStudioClient

        vector1 = [1.0, 0.0, 0.0]
        vector2 = [0.0, 1.0, 0.0]

        client = LMStudioClient(Config(jira_mcp_endpoint="test"))
        similarity = client.calculate_similarity(vector1, vector2)

        # Orthogonal vectors should have similarity close to 0.0
        assert abs(similarity) < 0.01

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity with opposite vectors."""
        from jirascope.clients.lmstudio_client import LMStudioClient

        vector1 = [1.0, 2.0, 3.0]
        vector2 = [-1.0, -2.0, -3.0]

        client = LMStudioClient(Config(jira_mcp_endpoint="test"))
        similarity = client.calculate_similarity(vector1, vector2)

        # Opposite vectors should have similarity close to -1.0
        assert abs(similarity + 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__])
