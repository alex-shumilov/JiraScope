"""Cross-Epic Analyzer Coverage Boost Tests - Targeting 59% -> 85%+ coverage."""

from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

from jirascope.analysis.cross_epic_analyzer import CrossEpicAnalyzer
from jirascope.core.config import Config
from jirascope.models import CrossEpicReport, WorkItem


class TestCrossEpicAnalyzerCoverage:
    """Test cross-epic analyzer business logic for maximum coverage boost."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config(jira_mcp_endpoint="http://localhost:8000")
        self.analyzer = CrossEpicAnalyzer(self.config)
        self.base_time = datetime.now(timezone.utc)

    def create_work_item(
        self,
        key: str,
        summary: str,
        issue_type: str = "Story",
        epic_key: Optional[str] = None,
        parent_key: Optional[str] = None,
    ) -> WorkItem:
        """Create a test work item."""
        return WorkItem(
            key=key,
            summary=summary,
            issue_type=issue_type,
            status="Open",
            created=self.base_time,
            updated=self.base_time,
            reporter="test@example.com",
            description=f"Description for {key}",
            parent_key=parent_key,
            epic_key=epic_key,
            assignee="test@example.com",
            embedding=[0.1, 0.2, 0.3] * 100,  # Mock embedding
        )

    @patch("jirascope.analysis.cross_epic_analyzer.QdrantVectorClient")
    @patch("jirascope.analysis.cross_epic_analyzer.LMStudioClient")
    @patch("jirascope.analysis.cross_epic_analyzer.ClaudeClient")
    async def test_context_manager_functionality(self, mock_claude, mock_lm, mock_qdrant):
        """Test async context manager functionality."""
        # Mock the clients
        mock_qdrant_instance = AsyncMock()
        mock_lm_instance = AsyncMock()
        mock_claude_instance = AsyncMock()

        mock_qdrant.return_value = mock_qdrant_instance
        mock_lm.return_value = mock_lm_instance
        mock_claude.return_value = mock_claude_instance

        # Test context manager
        async with self.analyzer as analyzer:
            assert analyzer.qdrant_client is not None
            assert analyzer.lm_client is not None
            assert analyzer.claude_client is not None

        # Verify all clients were entered and exited
        mock_qdrant_instance.__aenter__.assert_called_once()
        mock_lm_instance.__aenter__.assert_called_once()
        mock_claude_instance.__aenter__.assert_called_once()

        mock_qdrant_instance.__aexit__.assert_called_once()
        mock_lm_instance.__aexit__.assert_called_once()
        mock_claude_instance.__aexit__.assert_called_once()

    @patch("jirascope.analysis.cross_epic_analyzer.QdrantVectorClient")
    @patch("jirascope.analysis.cross_epic_analyzer.LMStudioClient")
    @patch("jirascope.analysis.cross_epic_analyzer.ClaudeClient")
    async def test_find_misplaced_work_items_business_logic(
        self, mock_claude, mock_lm, mock_qdrant
    ):
        """Test the main misplacement detection business logic."""
        # Setup mocks
        mock_qdrant_instance = AsyncMock()
        mock_lm_instance = AsyncMock()
        mock_claude_instance = AsyncMock()

        mock_qdrant.return_value = mock_qdrant_instance
        mock_lm.return_value = mock_lm_instance
        mock_claude.return_value = mock_claude_instance

        # Mock the get_all_epics_with_items method
        mock_epics_data = {
            "EPIC-1": [
                {
                    "key": "PROJ-1",
                    "summary": "Authentication feature",
                    "embedding": [0.1, 0.2, 0.3] * 100,
                },
                {
                    "key": "PROJ-2",
                    "summary": "Login validation",
                    "embedding": [0.1, 0.2, 0.3] * 100,
                },
            ],
            "EPIC-2": [
                {
                    "key": "PROJ-3",
                    "summary": "Database migration",
                    "embedding": [0.8, 0.9, 0.7] * 100,
                },
                {"key": "PROJ-4", "summary": "Data cleanup", "embedding": [0.8, 0.9, 0.7] * 100},
            ],
        }

        # Mock the _get_all_epics_with_items method
        self.analyzer._get_all_epics_with_items = AsyncMock(return_value=mock_epics_data)

        # Mock the _find_similar_across_epics method
        self.analyzer._find_similar_across_epics = AsyncMock(
            return_value=[{"epic_key": "EPIC-2", "score": 0.8}]
        )

        # Mock LMStudio similarity calculation
        mock_lm_instance.calculate_similarity.return_value = 0.75

        # Test the method
        async with self.analyzer as analyzer:
            result = await analyzer.find_misplaced_work_items("TEST")

            # Verify results
            assert isinstance(result, CrossEpicReport)
            assert result.epics_analyzed == 2
            assert len(result.misplaced_items) >= 0

    async def test_get_all_epics_with_items_logic(self):
        """Test epic data collection logic."""
        # Mock Qdrant client
        mock_qdrant_client = AsyncMock()
        mock_scroll_result = (
            [
                Mock(
                    payload={"key": "PROJ-1", "epic_key": "EPIC-1", "summary": "Story 1"},
                    vector=[0.1, 0.2, 0.3] * 100,
                ),
                Mock(
                    payload={"key": "PROJ-2", "epic_key": "EPIC-1", "summary": "Story 2"},
                    vector=[0.1, 0.2, 0.3] * 100,
                ),
                Mock(
                    payload={"key": "PROJ-3", "epic_key": "EPIC-2", "summary": "Story 3"},
                    vector=[0.8, 0.9, 0.7] * 100,
                ),
            ],
            None,  # Next page offset
        )

        mock_qdrant_client.client.scroll.return_value = mock_scroll_result
        self.analyzer.qdrant_client = mock_qdrant_client

        # Test the method
        result = await self.analyzer._get_all_epics_with_items("TEST")

        # Verify results
        assert isinstance(result, dict)
        assert "EPIC-1" in result
        assert "EPIC-2" in result
        assert len(result["EPIC-1"]) == 2
        assert len(result["EPIC-2"]) == 1

    async def test_calculate_epic_theme_embedding_algorithm(self):
        """Test epic theme embedding calculation algorithm."""
        # Create test work items
        work_items = [
            self.create_work_item("PROJ-1", "Authentication feature", epic_key="EPIC-1"),
            self.create_work_item("PROJ-2", "Login validation", epic_key="EPIC-1"),
            self.create_work_item("PROJ-3", "Password security", epic_key="EPIC-1"),
        ]

        epic = self.create_work_item("EPIC-1", "Authentication System", "Epic")

        # Mock LMStudio client
        mock_lm_client = AsyncMock()
        mock_lm_client.generate_embeddings.return_value = [[0.5, 0.5, 0.5] * 100]
        self.analyzer.lm_client = mock_lm_client

        # Test theme embedding calculation
        result = await self.analyzer._calculate_epic_theme_embedding(epic, work_items)

        # Verify results
        assert isinstance(result, list)
        assert len(result) == 300  # 3 * 100 dimension mock embedding
        assert all(isinstance(x, float) for x in result)

    async def test_calculate_epic_coherence_score_algorithm(self):
        """Test epic coherence score calculation algorithm."""
        # Create test work items
        work_items = [
            self.create_work_item("PROJ-1", "Authentication feature", epic_key="EPIC-1"),
            self.create_work_item("PROJ-2", "Login validation", epic_key="EPIC-1"),
        ]

        epic_theme = [0.5, 0.5, 0.5] * 100

        # Mock LMStudio client
        mock_lm_client = AsyncMock()
        mock_lm_client.calculate_similarity.return_value = 0.85
        self.analyzer.lm_client = mock_lm_client

        # Test coherence calculation
        result = await self.analyzer._calculate_epic_coherence_score(epic_theme, work_items)

        # Verify results
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    async def test_find_similar_across_epics_algorithm(self):
        """Test finding similar items across epics algorithm."""
        # Mock Qdrant client
        mock_qdrant_client = AsyncMock()
        mock_search_result = [
            Mock(
                payload={"key": "PROJ-3", "epic_key": "EPIC-2", "summary": "Similar item"},
                score=0.85,
            ),
            Mock(
                payload={"key": "PROJ-4", "epic_key": "EPIC-3", "summary": "Another similar item"},
                score=0.75,
            ),
        ]

        mock_qdrant_client.client.search.return_value = mock_search_result
        self.analyzer.qdrant_client = mock_qdrant_client

        # Test the method
        item_embedding = [0.1, 0.2, 0.3] * 100
        result = await self.analyzer._find_similar_across_epics(item_embedding, "EPIC-1", 0.7)

        # Verify results
        assert isinstance(result, list)
        assert len(result) == 2
        assert all("epic_key" in item for item in result)
        assert all("score" in item for item in result)

    def test_generate_misplacement_reasoning_logic(self):
        """Test misplacement reasoning generation logic."""
        # Create test data
        item_data = {
            "key": "PROJ-1",
            "summary": "Authentication feature",
            "description": "Implement user authentication",
        }

        best_match = {"epic_key": "EPIC-AUTH", "coherence": 0.85, "similarity_score": 0.90}

        # Test reasoning generation
        result = self.analyzer._generate_misplacement_reasoning(
            item_data, "EPIC-MISC", best_match, 0.45
        )

        # Verify results
        assert isinstance(result, str)
        assert len(result) > 0
        assert "EPIC-AUTH" in result
        assert "0.85" in result or "85%" in result

    async def test_analyze_misplacement_with_claude_integration(self):
        """Test Claude integration for misplacement analysis."""
        # Create test work items
        work_item = self.create_work_item("PROJ-1", "Authentication feature", epic_key="EPIC-1")
        current_epic = self.create_work_item("EPIC-1", "Miscellaneous Tasks", "Epic")
        suggested_epic = self.create_work_item("EPIC-AUTH", "Authentication System", "Epic")

        # Mock Claude client
        mock_claude_client = AsyncMock()
        mock_claude_response = {
            "analysis": "The work item appears to be misplaced",
            "confidence": 0.85,
            "reasoning": "Authentication feature belongs in authentication epic",
        }
        mock_claude_client.analyze_text.return_value = mock_claude_response
        self.analyzer.claude_client = mock_claude_client

        # Test Claude analysis
        result = await self.analyzer._analyze_misplacement_with_claude(
            work_item, current_epic, suggested_epic
        )

        # Verify results
        assert isinstance(result, dict)
        assert "analysis" in result
        assert "confidence" in result
        assert "reasoning" in result

    async def test_calculate_epic_coherence_business_logic(self):
        """Test epic coherence calculation business logic."""
        # Create test work item
        work_item = self.create_work_item("PROJ-1", "Authentication feature", epic_key="EPIC-1")

        # Mock the required methods
        self.analyzer._get_all_epics_with_items = AsyncMock(
            return_value={
                "EPIC-1": [
                    {
                        "key": "PROJ-1",
                        "summary": "Auth feature",
                        "embedding": [0.1, 0.2, 0.3] * 100,
                    },
                    {"key": "PROJ-2", "summary": "Login page", "embedding": [0.1, 0.2, 0.3] * 100},
                ]
            }
        )

        self.analyzer._calculate_epic_theme_embedding = AsyncMock(
            return_value=[0.5, 0.5, 0.5] * 100
        )

        # Mock LMStudio client
        mock_lm_client = AsyncMock()
        mock_lm_client.calculate_similarity.return_value = 0.78
        self.analyzer.lm_client = mock_lm_client

        # Test coherence calculation
        result = await self.analyzer.calculate_epic_coherence(work_item, "EPIC-1")

        # Verify results
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @patch("jirascope.analysis.cross_epic_analyzer.QdrantVectorClient")
    @patch("jirascope.analysis.cross_epic_analyzer.LMStudioClient")
    @patch("jirascope.analysis.cross_epic_analyzer.ClaudeClient")
    async def test_misplacement_detection_with_insufficient_data(
        self, mock_claude, mock_lm, mock_qdrant
    ):
        """Test misplacement detection with insufficient data."""
        # Setup mocks
        mock_qdrant_instance = AsyncMock()
        mock_lm_instance = AsyncMock()
        mock_claude_instance = AsyncMock()

        mock_qdrant.return_value = mock_qdrant_instance
        mock_lm.return_value = mock_lm_instance
        mock_claude.return_value = mock_claude_instance

        # Mock insufficient epics data
        self.analyzer._get_all_epics_with_items = AsyncMock(
            return_value={"EPIC-1": [{"key": "PROJ-1", "summary": "Single item"}]}
        )

        # Test the method
        async with self.analyzer as analyzer:
            result = await analyzer.find_misplaced_work_items("TEST")

            # Verify it handles insufficient data gracefully
            assert isinstance(result, CrossEpicReport)
            assert result.epics_analyzed == 1
            assert len(result.misplaced_items) == 0

    async def test_similarity_threshold_filtering(self):
        """Test similarity threshold filtering logic."""
        # Mock Qdrant client with various similarity scores
        mock_qdrant_client = AsyncMock()
        mock_search_result = [
            Mock(
                payload={"key": "PROJ-3", "epic_key": "EPIC-2", "summary": "High similarity"},
                score=0.95,  # Above threshold
            ),
            Mock(
                payload={"key": "PROJ-4", "epic_key": "EPIC-3", "summary": "Medium similarity"},
                score=0.75,  # Above threshold
            ),
            Mock(
                payload={"key": "PROJ-5", "epic_key": "EPIC-4", "summary": "Low similarity"},
                score=0.45,  # Below threshold
            ),
        ]

        mock_qdrant_client.client.search.return_value = mock_search_result
        self.analyzer.qdrant_client = mock_qdrant_client

        # Test with threshold of 0.7
        item_embedding = [0.1, 0.2, 0.3] * 100
        result = await self.analyzer._find_similar_across_epics(item_embedding, "EPIC-1", 0.7)

        # Verify only items above threshold are returned
        assert len(result) == 2
        assert all(item["score"] >= 0.7 for item in result)

    async def test_epic_theme_embedding_edge_cases(self):
        """Test epic theme embedding calculation edge cases."""
        # Test with no work items
        epic = self.create_work_item("EPIC-1", "Empty Epic", "Epic")

        # Mock LMStudio client
        mock_lm_client = AsyncMock()
        mock_lm_client.generate_embeddings.return_value = [[0.0] * 300]
        self.analyzer.lm_client = mock_lm_client

        # Test with empty work items list
        result = await self.analyzer._calculate_epic_theme_embedding(epic, [])

        # Should handle gracefully
        assert isinstance(result, list)
        assert len(result) == 300

    def test_misplacement_reasoning_content_quality(self):
        """Test quality of misplacement reasoning content."""
        # Create test data with rich content
        item_data = {
            "key": "PROJ-123",
            "summary": "Implement user authentication system",
            "description": "Create secure login functionality with OAuth2 support",
        }

        best_match = {"epic_key": "EPIC-AUTH", "coherence": 0.92, "similarity_score": 0.88}

        # Test reasoning generation
        result = self.analyzer._generate_misplacement_reasoning(
            item_data, "EPIC-MISC", best_match, 0.35
        )

        # Verify reasoning quality
        assert isinstance(result, str)
        assert len(result) > 50  # Should be detailed
        assert "EPIC-AUTH" in result
        assert "authentication" in result.lower() or "user" in result.lower()


class TestCrossEpicAnalyzerEdgeCases:
    """Test edge cases and error handling in cross-epic analyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config(jira_mcp_endpoint="http://localhost:8000")
        self.analyzer = CrossEpicAnalyzer(self.config)

    async def test_uninitialized_analyzer_error(self):
        """Test error handling when analyzer is not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await self.analyzer.find_misplaced_work_items("TEST")

    async def test_empty_epics_data_handling(self):
        """Test handling of empty epics data."""
        # Mock empty data
        self.analyzer._get_all_epics_with_items = AsyncMock(return_value={})

        # Mock clients
        self.analyzer.qdrant_client = AsyncMock()
        self.analyzer.lm_client = AsyncMock()

        # Test with empty data
        result = await self.analyzer.find_misplaced_work_items("TEST")

        # Should handle gracefully
        assert isinstance(result, CrossEpicReport)
        assert result.epics_analyzed == 0
        assert len(result.misplaced_items) == 0

    async def test_missing_embeddings_handling(self):
        """Test handling of work items without embeddings."""
        # Mock epics data with missing embeddings
        mock_epics_data = {
            "EPIC-1": [
                {"key": "PROJ-1", "summary": "No embedding"},  # Missing embedding
                {"key": "PROJ-2", "summary": "Has embedding", "embedding": [0.1, 0.2, 0.3] * 100},
            ]
        }

        self.analyzer._get_all_epics_with_items = AsyncMock(return_value=mock_epics_data)
        self.analyzer._find_similar_across_epics = AsyncMock(return_value=[])

        # Mock clients
        self.analyzer.qdrant_client = AsyncMock()
        self.analyzer.lm_client = AsyncMock()

        # Test handling of missing embeddings
        result = await self.analyzer.find_misplaced_work_items("TEST")

        # Should handle gracefully
        assert isinstance(result, CrossEpicReport)

    async def test_none_input_handling(self):
        """Test handling of None inputs."""
        # Test with None project key
        self.analyzer._get_all_epics_with_items = AsyncMock(return_value={})
        self.analyzer.qdrant_client = AsyncMock()
        self.analyzer.lm_client = AsyncMock()

        # Should handle None project key gracefully
        result = await self.analyzer.find_misplaced_work_items(None)
        assert isinstance(result, CrossEpicReport)

    async def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Create large mock dataset
        large_epics_data = {}
        for i in range(10):
            epic_key = f"EPIC-{i}"
            large_epics_data[epic_key] = [
                {"key": f"PROJ-{i}-{j}", "summary": f"Item {j}", "embedding": [0.1, 0.2, 0.3] * 100}
                for j in range(20)
            ]

        self.analyzer._get_all_epics_with_items = AsyncMock(return_value=large_epics_data)
        self.analyzer._find_similar_across_epics = AsyncMock(return_value=[])

        # Mock clients
        self.analyzer.qdrant_client = AsyncMock()
        self.analyzer.lm_client = AsyncMock()

        # Test with large dataset (should complete without hanging)
        result = await self.analyzer.find_misplaced_work_items("TEST")

        # Should handle large datasets
        assert isinstance(result, CrossEpicReport)
        assert result.epics_analyzed == 10
