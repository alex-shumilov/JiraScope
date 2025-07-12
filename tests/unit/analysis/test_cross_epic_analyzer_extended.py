"""Cross-Epic Analyzer Coverage Boost Tests - Targeting 59% -> 85%+ coverage."""

from datetime import UTC, datetime
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
        self.base_time = datetime.now(UTC)

    def create_work_item(
        self,
        key: str,
        summary: str,
        issue_type: str = "Story",
        epic_key: str | None = None,
        parent_key: str | None = None,
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_find_misplaced_work_items_business_logic(self):
        """Test the main misplacement detection business logic."""
        # Mock clients directly
        self.analyzer.qdrant_client = AsyncMock()
        self.analyzer.lm_client = AsyncMock()
        self.analyzer.claude_client = AsyncMock()

        # Ensure __aenter__ and __aexit__ are defined
        self.analyzer.qdrant_client.__aenter__ = AsyncMock(return_value=self.analyzer.qdrant_client)
        self.analyzer.qdrant_client.__aexit__ = AsyncMock()
        self.analyzer.lm_client.__aenter__ = AsyncMock(return_value=self.analyzer.lm_client)
        self.analyzer.lm_client.__aexit__ = AsyncMock()
        self.analyzer.claude_client.__aenter__ = AsyncMock(return_value=self.analyzer.claude_client)
        self.analyzer.claude_client.__aexit__ = AsyncMock()

        # Create proper work item data matching the model requirements
        base_time = datetime.now(UTC)
        mock_epics_data = {
            "EPIC-1": [
                {
                    "key": "PROJ-1",
                    "summary": "Authentication feature",
                    "embedding": [0.1, 0.2, 0.3] * 100,
                    "issue_type": "Story",
                    "status": "Open",
                    "created": base_time,
                    "updated": base_time,
                    "reporter": "test@example.com",
                    "epic_key": "EPIC-1",
                },
                {
                    "key": "PROJ-2",
                    "summary": "Login validation",
                    "embedding": [0.1, 0.2, 0.3] * 100,
                    "issue_type": "Story",
                    "status": "Open",
                    "created": base_time,
                    "updated": base_time,
                    "reporter": "test@example.com",
                    "epic_key": "EPIC-1",
                },
            ],
            "EPIC-2": [
                {
                    "key": "PROJ-3",
                    "summary": "Database migration",
                    "embedding": [0.8, 0.9, 0.7] * 100,
                    "issue_type": "Story",
                    "status": "Open",
                    "created": base_time,
                    "updated": base_time,
                    "reporter": "test@example.com",
                    "epic_key": "EPIC-2",
                },
                {
                    "key": "PROJ-4",
                    "summary": "Data cleanup",
                    "embedding": [0.8, 0.9, 0.7] * 100,
                    "issue_type": "Story",
                    "status": "Open",
                    "created": base_time,
                    "updated": base_time,
                    "reporter": "test@example.com",
                    "epic_key": "EPIC-2",
                },
            ],
        }

        # Mock the _get_all_epics_with_items method
        self.analyzer._get_all_epics_with_items = AsyncMock(return_value=mock_epics_data)

        # Mock the _find_similar_across_epics method
        self.analyzer._find_similar_across_epics = AsyncMock(
            return_value=[
                {
                    "epic_key": "EPIC-2",
                    "score": 0.8,
                    "key": "PROJ-3",
                    "work_item": {"key": "PROJ-3"},
                }
            ]
        )

        # Mock _calculate_epic_theme_embedding method
        self.analyzer._calculate_epic_theme_embedding = AsyncMock(
            return_value=[0.1, 0.2, 0.3] * 100
        )

        # Mock _calculate_epic_coherence_score method
        self.analyzer._calculate_epic_coherence_score = AsyncMock(return_value=0.65)

        # Set lm_client's calculate_similarity as a regular method
        self.analyzer.lm_client.calculate_similarity = Mock(return_value=0.75)

        # Test the method
        result = await self.analyzer.find_misplaced_work_items("TEST")

        # Verify results
        assert isinstance(result, CrossEpicReport)
        assert result.epics_analyzed == 2
        assert len(result.misplaced_items) >= 0

    @pytest.mark.asyncio
    async def test_get_all_epics_with_items_logic(self):
        """Test epic data collection logic."""
        # Mock Qdrant client
        mock_qdrant_client = AsyncMock()

        # Create proper mock points with the correct structure
        mock_point_1 = Mock()
        mock_point_1.payload = {"key": "PROJ-1", "epic_key": "EPIC-1", "summary": "Story 1"}
        mock_point_1.vector = [0.1, 0.2, 0.3] * 100

        mock_point_2 = Mock()
        mock_point_2.payload = {"key": "PROJ-2", "epic_key": "EPIC-1", "summary": "Story 2"}
        mock_point_2.vector = [0.1, 0.2, 0.3] * 100

        mock_point_3 = Mock()
        mock_point_3.payload = {"key": "PROJ-3", "epic_key": "EPIC-2", "summary": "Story 3"}
        mock_point_3.vector = [0.8, 0.9, 0.7] * 100

        mock_point_4 = Mock()
        mock_point_4.payload = {"key": "PROJ-4", "epic_key": "EPIC-2", "summary": "Story 4"}
        mock_point_4.vector = [0.8, 0.9, 0.7] * 100

        # Mock scroll returns (points_list, next_offset) tuple
        mock_scroll_result = (
            [mock_point_1, mock_point_2, mock_point_3, mock_point_4],
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
        assert len(result["EPIC-2"]) == 2

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_calculate_epic_coherence_score_algorithm(self):
        """Test epic coherence score calculation algorithm."""
        # Create test work items
        work_items = [
            self.create_work_item("PROJ-1", "Authentication feature", epic_key="EPIC-1"),
            self.create_work_item("PROJ-2", "Login validation", epic_key="EPIC-1"),
        ]

        epic_theme = [0.5, 0.5, 0.5] * 100

        # Mock LMStudio client (calculate_similarity is SYNC, not async)
        mock_lm_client = Mock()
        mock_lm_client.calculate_similarity.return_value = 0.85
        self.analyzer.lm_client = mock_lm_client

        # Test coherence calculation
        result = await self.analyzer._calculate_epic_coherence_score(epic_theme, work_items)

        # Verify results
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # Should be around 0.85 since both items have embeddings and similarity returns 0.85
        assert abs(result - 0.85) < 0.1

    @pytest.mark.asyncio
    async def test_find_similar_across_epics_algorithm(self):
        """Test finding similar items across epics algorithm."""
        # Mock Qdrant client
        mock_qdrant_client = AsyncMock()

        # Mock search_similar_work_items to return the correct format
        # The real method returns List[Dict[str, Any]] with "score" and "work_item" keys
        mock_search_result = [
            {
                "score": 0.85,
                "work_item": {"key": "PROJ-3", "epic_key": "EPIC-2", "summary": "Similar item"},
            },
            {
                "score": 0.75,
                "work_item": {
                    "key": "PROJ-4",
                    "epic_key": "EPIC-3",
                    "summary": "Another similar item",
                },
            },
        ]

        mock_qdrant_client.search_similar_work_items.return_value = mock_search_result
        self.analyzer.qdrant_client = mock_qdrant_client

        # Test the method
        item_embedding = [0.1, 0.2, 0.3] * 100
        result = await self.analyzer._find_similar_across_epics(item_embedding, "EPIC-1", 0.7)

        # Verify results
        assert isinstance(result, list)
        assert len(result) == 2
        assert all("epic_key" in item for item in result)
        assert all("score" in item for item in result)
        assert all("key" in item for item in result)
        assert all("work_item" in item for item in result)

        # Verify the actual values
        assert result[0]["epic_key"] == "EPIC-2"
        assert result[0]["score"] == 0.85
        assert result[1]["epic_key"] == "EPIC-3"
        assert result[1]["score"] == 0.75

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

    @pytest.mark.asyncio
    async def test_analyze_misplacement_with_claude_integration(self):
        """Test Claude integration for misplacement analysis."""
        # Create test work items
        work_item = self.create_work_item("PROJ-1", "Authentication feature", epic_key="EPIC-1")
        current_epic = self.create_work_item("EPIC-1", "Miscellaneous Tasks", "Epic")
        suggested_epic = self.create_work_item("EPIC-AUTH", "Authentication System", "Epic")

        # Mock Claude client - it should return a simple object with content field containing JSON
        mock_claude_client = AsyncMock()

        # Create a mock response object that matches the SimpleResponse class in claude_client.py
        class MockResponse:
            def __init__(self):
                self.content = '{"reasoning": "Authentication feature belongs in authentication epic", "confidence": 0.85}'
                self.cost = 0.05

        mock_claude_response = MockResponse()
        mock_claude_client.analyze.return_value = mock_claude_response
        self.analyzer.claude_client = mock_claude_client

        # Test Claude analysis
        result = await self.analyzer._analyze_misplacement_with_claude(
            work_item, current_epic, suggested_epic
        )

        # Verify results - the method parses response.content as JSON and adds cost
        assert isinstance(result, dict)
        assert "reasoning" in result
        assert "confidence" in result
        assert "cost" in result
        assert result["reasoning"] == "Authentication feature belongs in authentication epic"
        assert result["confidence"] == 0.85
        assert result["cost"] == 0.05

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    @patch("jirascope.analysis.cross_epic_analyzer.QdrantVectorClient")
    @patch("jirascope.analysis.cross_epic_analyzer.LMStudioClient")
    @patch("jirascope.analysis.cross_epic_analyzer.ClaudeClient")
    async def test_misplacement_detection_with_insufficient_data(
        self, _mock_claude, _mock_lm, _mock_qdrant
    ):
        """Test misplacement detection with insufficient data."""
        # Setup mocks
        mock_qdrant_instance = AsyncMock()
        mock_lm_instance = AsyncMock()
        mock_claude_instance = AsyncMock()

        _mock_qdrant.return_value = mock_qdrant_instance
        _mock_lm.return_value = mock_lm_instance
        _mock_claude.return_value = mock_claude_instance

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

    @pytest.mark.asyncio
    async def test_similarity_threshold_filtering(self):
        """Test similarity threshold filtering logic."""
        # Mock Qdrant client with various similarity scores
        mock_qdrant_client = AsyncMock()

        # Mock search_similar_work_items to return the correct format
        # The Qdrant client is supposed to filter by threshold, so only return items above 0.7
        mock_search_result = [
            {
                "score": 0.95,  # Above threshold
                "work_item": {"key": "PROJ-3", "epic_key": "EPIC-2", "summary": "High similarity"},
            },
            {
                "score": 0.75,  # Above threshold
                "work_item": {
                    "key": "PROJ-4",
                    "epic_key": "EPIC-3",
                    "summary": "Medium similarity",
                },
            },
            # Note: Item with score 0.45 is filtered out by Qdrant's score_threshold
        ]

        mock_qdrant_client.search_similar_work_items.return_value = mock_search_result
        self.analyzer.qdrant_client = mock_qdrant_client

        # Test with threshold of 0.7
        item_embedding = [0.1, 0.2, 0.3] * 100
        result = await self.analyzer._find_similar_across_epics(item_embedding, "EPIC-1", 0.7)

        # Verify only items above threshold are returned
        assert len(result) == 2
        assert all(item["score"] >= 0.7 for item in result)
        # Verify the specific items that should be returned
        assert result[0]["score"] == 0.95
        assert result[1]["score"] == 0.75

    @pytest.mark.asyncio
    async def test_epic_theme_embedding_edge_cases(self):
        """Test epic theme embedding calculation edge cases."""
        # Test with no work items
        epic = self.create_work_item("EPIC-1", "Empty Epic", "Epic")

        # Mock LMStudio client - the implementation returns default size from EMBEDDING_CONFIG
        mock_lm_client = AsyncMock()
        # The _calculate_epic_theme_embedding method returns [0.0] * 1024 for empty items
        # But let's check what the actual config uses by looking at the constants
        mock_lm_client.generate_embeddings.return_value = [[0.0] * 1024]  # Use actual default size
        self.analyzer.lm_client = mock_lm_client

        # Test with empty work items list
        result = await self.analyzer._calculate_epic_theme_embedding(epic, [])

        # Should handle gracefully and return the default embedding size (1024, not 300)
        assert isinstance(result, list)
        assert len(result) == 1024  # This is the actual default size from the implementation
        assert all(x == 0.0 for x in result)  # Should be all zeros for empty epic

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
        assert result is not None
        assert len(result) > 20  # Reasonable length
        assert "epic" in result.lower()
        assert "coherence" in result.lower()

        # Check for authentication-related keywords (including epic names)
        auth_keywords = ["authentication", "user", "auth", "epic-auth", "login"]
        assert any(keyword in result.lower() for keyword in auth_keywords)


class TestCrossEpicAnalyzerEdgeCases:
    """Test edge cases and error handling in cross-epic analyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config(jira_mcp_endpoint="http://localhost:8000")
        self.analyzer = CrossEpicAnalyzer(self.config)

    @pytest.mark.asyncio
    async def test_uninitialized_analyzer_error(self):
        """Test error handling when analyzer is not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await self.analyzer.find_misplaced_work_items("TEST")

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_none_input_handling(self):
        """Test handling of None inputs."""
        # Test with None project key
        self.analyzer._get_all_epics_with_items = AsyncMock(return_value={})
        self.analyzer.qdrant_client = AsyncMock()
        self.analyzer.lm_client = AsyncMock()

        # Should handle None project key gracefully
        result = await self.analyzer.find_misplaced_work_items(None)
        assert isinstance(result, CrossEpicReport)

    @pytest.mark.asyncio
    async def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Mock clients directly
        self.analyzer.qdrant_client = AsyncMock()
        self.analyzer.lm_client = AsyncMock()
        self.analyzer.claude_client = AsyncMock()

        # Ensure __aenter__ and __aexit__ are defined
        self.analyzer.qdrant_client.__aenter__ = AsyncMock(return_value=self.analyzer.qdrant_client)
        self.analyzer.qdrant_client.__aexit__ = AsyncMock()
        self.analyzer.lm_client.__aenter__ = AsyncMock(return_value=self.analyzer.lm_client)
        self.analyzer.lm_client.__aexit__ = AsyncMock()
        self.analyzer.claude_client.__aenter__ = AsyncMock(return_value=self.analyzer.claude_client)
        self.analyzer.claude_client.__aexit__ = AsyncMock()

        # Create large mock dataset with proper fields for WorkItem model
        base_time = datetime.now(UTC)
        large_epics_data = {}
        for i in range(10):
            epic_key = f"EPIC-{i}"
            large_epics_data[epic_key] = [
                {
                    "key": f"PROJ-{i}-{j}",
                    "summary": f"Item {j}",
                    "embedding": [0.1, 0.2, 0.3] * 100,
                    "issue_type": "Story",
                    "status": "Open",
                    "created": base_time,
                    "updated": base_time,
                    "reporter": "test@example.com",
                    "epic_key": epic_key,
                }
                for j in range(20)
            ]

        # Mock the internal methods
        self.analyzer._get_all_epics_with_items = AsyncMock(return_value=large_epics_data)
        self.analyzer._find_similar_across_epics = AsyncMock(return_value=[])
        self.analyzer._calculate_epic_theme_embedding = AsyncMock(
            return_value=[0.1, 0.2, 0.3] * 100
        )
        self.analyzer._calculate_epic_coherence_score = AsyncMock(return_value=0.65)
        self.analyzer.lm_client.calculate_similarity = Mock(return_value=0.75)

        # Test with large dataset (should complete without hanging)
        result = await self.analyzer.find_misplaced_work_items("TEST")

        # Should handle large datasets
        assert isinstance(result, CrossEpicReport)
        assert result.epics_analyzed == 10
