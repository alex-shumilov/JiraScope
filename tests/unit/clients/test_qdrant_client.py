"""Tests for Qdrant client functionality."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from qdrant_client.http.models import (
    CollectionInfo,
    CollectionsResponse,
    Distance,
    PointStruct,
    VectorParams,
)

from src.jirascope.clients.qdrant_client import JIRASCOPE_COLLECTION_CONFIG, QdrantVectorClient
from src.jirascope.core.config import Config
from src.jirascope.models import WorkItem


class MockChunk:
    """Mock chunk class for testing."""

    def __init__(self, chunk_id: str, text: str, metadata=None):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata or MockChunkMetadata()


class MockChunkMetadata:
    """Mock chunk metadata class for testing."""

    def to_qdrant_payload(self):
        return {
            "chunk_type": "description",
            "source_key": "TEST-1",
            "created_month": "2024-01",
            "team": "Backend",
        }


class TestQdrantVectorClient:
    """Test Qdrant vector client functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config(
            jira_mcp_endpoint="https://test.atlassian.net",
            qdrant_url="http://localhost:6333",
            embedding_dimensions=1536,
            similarity_threshold=0.7,
        )
        self.client = QdrantVectorClient(self.config)

        # Create test work item
        self.work_item = WorkItem(
            key="TEST-123",
            summary="Test work item",
            description="Test description",
            issue_type="Story",
            status="Open",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
            reporter="test@example.com",
            assignee="dev@example.com",
            components=["Backend", "API"],
            labels=["test", "unit"],
            parent_key="EPIC-1",
            epic_key="EPIC-1",
            embedding=[0.1, 0.2, 0.3] * 512,  # 1536 dimensions
        )

        # Create test embedding
        self.test_embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions

    def test_qdrant_client_initialization(self):
        """Test Qdrant client initialization."""
        assert self.client.config == self.config
        assert self.client.collection_name == "jirascope_work_items"
        assert self.client.client is not None

    def test_collection_config_constants(self):
        """Test collection configuration constants."""
        config = JIRASCOPE_COLLECTION_CONFIG

        assert config["vectors"]["size"] == 1536
        assert config["vectors"]["distance"] == "Cosine"
        assert "epic_key" in config["payload_schema"]
        assert "item_type" in config["payload_schema"]
        assert "chunk_type" in config["payload_schema"]

    @pytest.mark.asyncio
    async def test_context_manager_entry(self):
        """Test async context manager entry."""
        with patch.object(self.client, "initialize_collection") as mock_init:
            mock_init.return_value = None

            result = await self.client.__aenter__()

            assert result == self.client
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_exit(self):
        """Test async context manager exit."""
        # Should not raise any exceptions
        await self.client.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_initialize_collection_creates_new(self):
        """Test collection initialization when collection doesn't exist."""
        # Mock Qdrant client
        mock_collections_response = CollectionsResponse(collections=[])

        with patch.object(self.client.client, "get_collections") as mock_get:
            mock_get.return_value = mock_collections_response

            with patch.object(self.client.client, "create_collection") as mock_create:
                await self.client.initialize_collection()

                mock_create.assert_called_once_with(
                    collection_name="jirascope_work_items",
                    vectors_config=VectorParams(
                        size=self.config.embedding_dimensions, distance=Distance.COSINE
                    ),
                )

    @pytest.mark.asyncio
    async def test_initialize_collection_exists(self):
        """Test collection initialization when collection already exists."""
        # Mock existing collection
        existing_collection = CollectionInfo(name="jirascope_work_items")
        mock_collections_response = CollectionsResponse(collections=[existing_collection])

        with patch.object(self.client.client, "get_collections") as mock_get:
            mock_get.return_value = mock_collections_response

            with patch.object(self.client.client, "create_collection") as mock_create:
                await self.client.initialize_collection()

                # Should not create collection if it exists
                mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_collection_error(self):
        """Test collection initialization error handling."""
        with patch.object(self.client.client, "get_collections") as mock_get:
            mock_get.side_effect = Exception("Connection error")

            with pytest.raises(Exception, match="Connection error"):
                await self.client.initialize_collection()

    @pytest.mark.asyncio
    async def test_store_work_items_success(self):
        """Test successful work item storage."""
        work_items = [self.work_item]
        embeddings = [self.test_embedding]

        with patch.object(self.client.client, "upsert") as mock_upsert:
            await self.client.store_work_items(work_items, embeddings)

            mock_upsert.assert_called_once()
            call_args = mock_upsert.call_args

            # Verify collection name
            assert call_args[1]["collection_name"] == "jirascope_work_items"

            # Verify points structure
            points = call_args[1]["points"]
            assert len(points) == 1
            point = points[0]
            assert isinstance(point, PointStruct)
            assert point.vector == self.test_embedding
            assert point.payload["key"] == "TEST-123"
            assert point.payload["summary"] == "Test work item"

    @pytest.mark.asyncio
    async def test_store_work_items_mismatched_lengths(self):
        """Test work item storage with mismatched lengths."""
        work_items = [self.work_item]
        embeddings = [self.test_embedding, [0.4, 0.5, 0.6] * 512]  # More embeddings than items

        with pytest.raises(
            ValueError, match="Number of work items must match number of embeddings"
        ):
            await self.client.store_work_items(work_items, embeddings)

    @pytest.mark.asyncio
    async def test_store_work_items_error(self):
        """Test work item storage error handling."""
        work_items = [self.work_item]
        embeddings = [self.test_embedding]

        with patch.object(self.client.client, "upsert") as mock_upsert:
            mock_upsert.side_effect = Exception("Storage error")

            with pytest.raises(Exception, match="Storage error"):
                await self.client.store_work_items(work_items, embeddings)

    @pytest.mark.asyncio
    async def test_store_chunks_success(self):
        """Test successful chunk storage."""
        chunk = MockChunk("chunk-1", "Test chunk text")
        chunks = [chunk]
        embeddings = [self.test_embedding]

        with patch.object(self.client.client, "upsert") as mock_upsert:
            await self.client.store_chunks(chunks, embeddings)

            mock_upsert.assert_called_once()
            call_args = mock_upsert.call_args

            points = call_args[1]["points"]
            assert len(points) == 1
            point = points[0]
            assert point.vector == self.test_embedding
            assert point.payload["text"] == "Test chunk text"
            assert "chunk_type" in point.payload

    @pytest.mark.asyncio
    async def test_store_chunks_mismatched_lengths(self):
        """Test chunk storage with mismatched lengths."""
        chunks = [MockChunk("chunk-1", "Test chunk")]
        embeddings = [self.test_embedding, [0.4, 0.5, 0.6] * 512]

        with pytest.raises(ValueError, match="Number of chunks must match number of embeddings"):
            await self.client.store_chunks(chunks, embeddings)

    @pytest.mark.asyncio
    async def test_search_similar_work_items_success(self):
        """Test successful similarity search."""
        # Mock search result
        mock_hit = Mock()
        mock_hit.score = 0.85
        mock_hit.payload = {"key": "TEST-123", "summary": "Test item"}

        with patch.object(self.client.client, "search") as mock_search:
            mock_search.return_value = [mock_hit]

            results = await self.client.search_similar_work_items(self.test_embedding, limit=5)

            assert len(results) == 1
            assert results[0]["score"] == 0.85
            assert results[0]["work_item"]["key"] == "TEST-123"

            # Verify search parameters
            mock_search.assert_called_once_with(
                collection_name="jirascope_work_items",
                query_vector=self.test_embedding,
                limit=5,
                score_threshold=self.config.similarity_threshold,
            )

    @pytest.mark.asyncio
    async def test_search_similar_work_items_custom_threshold(self):
        """Test similarity search with custom threshold."""
        mock_hit = Mock()
        mock_hit.score = 0.9
        mock_hit.payload = {"key": "TEST-456"}

        with patch.object(self.client.client, "search") as mock_search:
            mock_search.return_value = [mock_hit]

            await self.client.search_similar_work_items(
                self.test_embedding, limit=10, score_threshold=0.8
            )

            mock_search.assert_called_once_with(
                collection_name="jirascope_work_items",
                query_vector=self.test_embedding,
                limit=10,
                score_threshold=0.8,
            )

    @pytest.mark.asyncio
    async def test_search_similar_work_items_error(self):
        """Test similarity search error handling."""
        with patch.object(self.client.client, "search") as mock_search:
            mock_search.side_effect = Exception("Search error")

            with pytest.raises(Exception, match="Search error"):
                await self.client.search_similar_work_items(self.test_embedding)

    @pytest.mark.asyncio
    async def test_search_with_filters_success(self):
        """Test filtered search success."""
        mock_hit = Mock()
        mock_hit.score = 0.8
        mock_hit.payload = {"key": "TEST-789", "status": "Open"}

        filters = {"status": "Open", "item_type": "Story"}

        with patch.object(self.client.client, "search") as mock_search:
            mock_search.return_value = [mock_hit]

            results = await self.client.search_with_filters(
                self.test_embedding, filters=filters, limit=15, score_threshold=0.6
            )

            assert len(results) == 1
            assert results[0]["score"] == 0.8

            # Verify search was called with filters
            mock_search.assert_called_once()
            call_args = mock_search.call_args[1]
            assert call_args["limit"] == 15
            assert call_args["score_threshold"] == 0.6
            assert call_args["query_filter"] is not None

    @pytest.mark.asyncio
    async def test_search_with_filters_list_values(self):
        """Test filtered search with list values."""
        filters = {"item_type": ["Story", "Bug"], "status": "Open"}

        with patch.object(self.client.client, "search") as mock_search:
            mock_search.return_value = []

            await self.client.search_with_filters(self.test_embedding, filters=filters)

            # Should handle list values in filters
            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_no_filters(self):
        """Test filtered search with no filters provided."""
        with patch.object(self.client.client, "search") as mock_search:
            mock_search.return_value = []

            await self.client.search_with_filters(self.test_embedding, filters=None)

            # Should call search without filter
            call_args = mock_search.call_args[1]
            assert call_args.get("query_filter") is None

    @pytest.mark.asyncio
    async def test_search_by_epic_success(self):
        """Test search by epic functionality."""
        with patch.object(self.client, "search_with_filters") as mock_search:
            mock_search.return_value = [{"score": 0.9, "work_item": {"key": "TEST-EPIC"}}]

            results = await self.client.search_by_epic(self.test_embedding, "EPIC-123", limit=8)

            assert len(results) == 1
            mock_search.assert_called_once_with(
                self.test_embedding, filters={"epic_key": "EPIC-123"}, limit=8
            )

    @pytest.mark.asyncio
    async def test_search_by_item_type_success(self):
        """Test search by item type functionality."""
        item_types = ["Story", "Bug"]

        with patch.object(self.client, "search_with_filters") as mock_search:
            mock_search.return_value = [{"score": 0.85, "work_item": {"key": "TEST-TYPE"}}]

            results = await self.client.search_by_item_type(
                self.test_embedding, item_types, limit=12
            )

            assert len(results) == 1
            mock_search.assert_called_once_with(
                self.test_embedding, filters={"item_type": item_types}, limit=12
            )

    @pytest.mark.asyncio
    async def test_get_work_item_by_key_found(self):
        """Test getting work item by key when found."""
        mock_hit = Mock()
        mock_hit.payload = {"key": "TEST-123", "summary": "Found item"}

        with patch.object(self.client.client, "scroll") as mock_scroll:
            mock_scroll.return_value = ([mock_hit], None)

            result = await self.client.get_work_item_by_key("TEST-123")

            assert result is not None
            assert result["key"] == "TEST-123"
            assert result["summary"] == "Found item"

    @pytest.mark.asyncio
    async def test_get_work_item_by_key_not_found(self):
        """Test getting work item by key when not found."""
        with patch.object(self.client.client, "scroll") as mock_scroll:
            mock_scroll.return_value = ([], None)

            result = await self.client.get_work_item_by_key("NOT-FOUND")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_work_item_by_key_error(self):
        """Test getting work item by key error handling."""
        with patch.object(self.client.client, "scroll") as mock_scroll:
            mock_scroll.side_effect = Exception("Scroll error")

            with pytest.raises(Exception, match="Scroll error"):
                await self.client.get_work_item_by_key("TEST-123")

    @pytest.mark.asyncio
    async def test_delete_work_item_success(self):
        """Test successful work item deletion."""
        with patch.object(self.client.client, "delete") as mock_delete:
            mock_delete.return_value = True

            result = await self.client.delete_work_item("TEST-123")

            assert result is True
            mock_delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_work_item_error(self):
        """Test work item deletion error handling."""
        with patch.object(self.client.client, "delete") as mock_delete:
            mock_delete.side_effect = Exception("Delete error")

            result = await self.client.delete_work_item("TEST-123")

            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        mock_collection_info = CollectionInfo(name="jirascope_work_items")

        with patch.object(self.client.client, "get_collection") as mock_get:
            mock_get.return_value = mock_collection_info

            result = await self.client.health_check()

            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check failure."""
        with patch.object(self.client.client, "get_collection") as mock_get:
            mock_get.side_effect = Exception("Health check failed")

            result = await self.client.health_check()

            assert result is False

    @pytest.mark.asyncio
    async def test_get_collection_stats_success(self):
        """Test successful collection statistics retrieval."""
        mock_collection_info = Mock()
        mock_collection_info.points_count = 1000
        mock_collection_info.segments_count = 5
        mock_collection_info.vectors_count = 1000

        with patch.object(self.client.client, "get_collection") as mock_get:
            mock_get.return_value = mock_collection_info

            stats = await self.client.get_collection_stats()

            assert stats["points_count"] == 1000
            assert stats["segments_count"] == 5
            assert stats["vectors_count"] == 1000
            assert stats["collection_name"] == "jirascope_work_items"

    @pytest.mark.asyncio
    async def test_get_collection_stats_error(self):
        """Test collection statistics error handling."""
        with patch.object(self.client.client, "get_collection") as mock_get:
            mock_get.side_effect = Exception("Stats error")

            with pytest.raises(Exception, match="Stats error"):
                await self.client.get_collection_stats()

    def test_work_item_payload_creation(self):
        """Test work item payload structure in store operation."""
        work_items = [self.work_item]
        embeddings = [self.test_embedding]

        with patch.object(self.client.client, "upsert") as mock_upsert:
            # Run the store operation
            import asyncio

            asyncio.run(self.client.store_work_items(work_items, embeddings))

            # Extract the point payload
            points = mock_upsert.call_args[1]["points"]
            payload = points[0].payload

            # Verify all expected fields are present
            assert payload["key"] == "TEST-123"
            assert payload["summary"] == "Test work item"
            assert payload["description"] == "Test description"
            assert payload["issue_type"] == "Story"
            assert payload["status"] == "Open"
            assert payload["parent_key"] == "EPIC-1"
            assert payload["epic_key"] == "EPIC-1"
            assert payload["assignee"] == "dev@example.com"
            assert payload["reporter"] == "test@example.com"
            assert payload["components"] == ["Backend", "API"]
            assert payload["labels"] == ["test", "unit"]
            assert "created" in payload
            assert "updated" in payload

    def test_point_id_generation(self):
        """Test point ID generation from work item key."""
        work_items = [self.work_item]
        embeddings = [self.test_embedding]

        with patch.object(self.client.client, "upsert") as mock_upsert:
            import asyncio

            asyncio.run(self.client.store_work_items(work_items, embeddings))

            points = mock_upsert.call_args[1]["points"]
            point_id = points[0].id

            # Verify ID is generated from hash of key
            expected_id = abs(hash("TEST-123"))
            assert point_id == expected_id

    @pytest.mark.asyncio
    async def test_context_manager_full_workflow(self):
        """Test full context manager workflow."""
        with patch.object(self.client, "initialize_collection") as mock_init:
            mock_init.return_value = None

            async with self.client as client:
                assert client == self.client
                mock_init.assert_called_once()

            # No exception should be raised on exit
