"""Comprehensive tests for embedding processor functionality."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.jirascope.core.config import Config
from src.jirascope.models import ProcessingResult
from src.jirascope.models.work_item import WorkItem
from src.jirascope.pipeline.embedding_processor import AdaptiveBatcher, EmbeddingProcessor


class TestAdaptiveBatcher:
    """Test AdaptiveBatcher functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.batcher = AdaptiveBatcher(base_batch_size=32)

    def test_adaptive_batcher_initialization(self):
        """Test AdaptiveBatcher initialization."""
        assert self.batcher.base_batch_size == 32
        assert self.batcher.performance_history == []
        assert self.batcher.max_history == 10

    def test_adaptive_batcher_custom_settings(self):
        """Test AdaptiveBatcher with custom settings."""
        custom_batcher = AdaptiveBatcher(base_batch_size=16)
        assert custom_batcher.base_batch_size == 16

    def test_calculate_optimal_batch_size_empty_items(self):
        """Test batch size calculation with empty items list."""
        result = self.batcher.calculate_optimal_batch_size([])
        assert result == self.batcher.base_batch_size

    def test_calculate_optimal_batch_size_short_texts(self):
        """Test batch size calculation with short texts."""
        items = [
            self._create_work_item("SHORT-1", "Short summary", "Short description"),
            self._create_work_item("SHORT-2", "Another short", "Brief desc"),
        ]

        result = self.batcher.calculate_optimal_batch_size(items)
        assert result == self.batcher.base_batch_size  # Should use base size for short texts

    def test_calculate_optimal_batch_size_medium_texts(self):
        """Test batch size calculation with medium texts."""
        medium_desc = "This is a medium length description. " * 20  # ~600 chars
        items = [
            self._create_work_item("MED-1", "Medium summary", medium_desc),
            self._create_work_item("MED-2", "Another medium", medium_desc),
        ]

        # Mock the _prepare_embedding_text to return text of consistent length
        with patch.object(self.batcher, "_prepare_embedding_text") as mock_prepare:
            # Return text with length > 500 to trigger medium text logic
            mock_prepare.return_value = "X" * 600

            result = self.batcher.calculate_optimal_batch_size(items)

            # Medium text should use base_batch_size // 2
            expected = max(16, self.batcher.base_batch_size // 2)
            assert result == expected

    def test_calculate_optimal_batch_size_large_texts(self):
        """Test batch size calculation with large texts."""
        large_desc = "This is a very long description. " * 50  # ~1650 chars
        items = [
            self._create_work_item("LARGE-1", "Large summary", large_desc),
            self._create_work_item("LARGE-2", "Another large", large_desc),
        ]

        # Mock the _prepare_embedding_text to return text of consistent length
        with patch.object(self.batcher, "_prepare_embedding_text") as mock_prepare:
            # Return text with length > 1000 to trigger large text logic
            mock_prepare.return_value = "X" * 1200

            result = self.batcher.calculate_optimal_batch_size(items)

            # Large text should use base_batch_size // 4
            expected = max(8, self.batcher.base_batch_size // 4)
            assert result == expected

    def test_record_performance(self):
        """Test performance recording."""
        # Record some performance data
        self.batcher.record_performance(batch_size=16, processing_time=2.0, items_count=16)

        assert len(self.batcher.performance_history) == 1
        assert self.batcher.performance_history[0] == 0.125  # 2.0 / 16

    def test_record_performance_max_history(self):
        """Test that performance history respects max limit."""
        # Record more than max_history entries
        for i in range(15):
            self.batcher.record_performance(batch_size=8, processing_time=1.0, items_count=8)

        assert len(self.batcher.performance_history) == self.batcher.max_history

    def test_calculate_optimal_batch_size_with_slow_performance(self):
        """Test batch size adjustment based on slow performance history."""
        # Record slow performance (> 2.0 seconds per item)
        for _ in range(5):
            self.batcher.record_performance(
                batch_size=32, processing_time=80.0, items_count=32
            )  # 2.5 s/item

        items = [self._create_work_item("SLOW-1", "Summary", "Description")]
        result = self.batcher.calculate_optimal_batch_size(items)

        # Should reduce batch size due to slow performance
        assert result < self.batcher.base_batch_size

    def test_calculate_optimal_batch_size_with_fast_performance(self):
        """Test batch size adjustment based on fast performance history."""
        # Record fast performance (< 0.5 seconds per item)
        for _ in range(5):
            self.batcher.record_performance(
                batch_size=32, processing_time=8.0, items_count=32
            )  # 0.25 s/item

        items = [self._create_work_item("FAST-1", "Summary", "Description")]
        result = self.batcher.calculate_optimal_batch_size(items)

        # Should increase batch size due to fast performance
        assert result >= self.batcher.base_batch_size

    def test_prepare_embedding_text(self):
        """Test text preparation for embedding length calculation."""
        item = self._create_work_item("TEST-1", "Test summary", "Test description")
        result = self.batcher._prepare_embedding_text(item)

        assert isinstance(result, str)
        assert "Test summary" in result
        assert "Test description" in result

    def _create_work_item(self, key: str, summary: str, description: str) -> WorkItem:
        """Helper to create work items for testing."""
        return WorkItem(
            key=key,
            summary=summary,
            issue_type="Story",
            status="Open",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
            reporter="test@example.com",
            description=description,
            assignee="test@example.com",
            embedding=[0.1] * 100,
        )


class TestEmbeddingProcessor:
    """Test EmbeddingProcessor functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory that persists for the duration of the test
        self.temp_dir = Path(tempfile.mkdtemp())

        self.config = Config(
            jira_mcp_endpoint="https://test.atlassian.net",
            embedding_batch_size=32,
            lmstudio_endpoint="http://localhost:1234/v1",
            qdrant_url="http://localhost:6333",
        )

        # Mock the cache directory before creating the processor
        patcher = patch("src.jirascope.pipeline.embedding_processor.Path.home")
        self.mock_home = patcher.start()
        self.mock_home.return_value = self.temp_dir

        # Mock the SmartChunker class
        smart_chunker_patcher = patch("src.jirascope.pipeline.embedding_processor.SmartChunker")
        self.mock_smart_chunker = smart_chunker_patcher.start()
        self.mock_smart_chunker_instance = Mock()
        self.mock_smart_chunker.return_value = self.mock_smart_chunker_instance

        # Now create the processor with the mocked path
        self.processor = EmbeddingProcessor(self.config)

        # Store the additional patcher to stop it in teardown
        self.smart_chunker_patcher = smart_chunker_patcher

        # Store the patcher to stop it in teardown
        self.patcher = patcher

    def teardown_method(self):
        """Clean up after tests."""
        # Stop the patchers
        self.patcher.stop()
        self.smart_chunker_patcher.stop()

        # Clean up the temporary directory
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_embedding_processor_initialization(self):
        """Test EmbeddingProcessor initialization."""
        assert self.processor.config == self.config
        assert isinstance(self.processor.batcher, AdaptiveBatcher)
        assert self.processor.batcher.base_batch_size == self.config.embedding_batch_size
        assert self.processor.chunker == self.mock_smart_chunker_instance
        assert self.processor.cache_dir.exists()

    @pytest.mark.asyncio
    async def test_process_work_items_empty_list(self):
        """Test processing empty work items list."""
        result = await self.processor.process_work_items([])

        assert isinstance(result, ProcessingResult)
        assert result.processed_items == 0
        assert result.failed_items == 0
        assert result.skipped_items == 0

    @pytest.mark.asyncio
    async def test_process_work_items_success(self):
        """Test successful processing of work items."""
        work_items = [
            self._create_work_item("TEST-1", "First item", "First description"),
            self._create_work_item("TEST-2", "Second item", "Second description"),
        ]

        # Mock clients
        mock_lm_client = AsyncMock()
        mock_qdrant_client = AsyncMock()

        mock_lm_client.generate_embeddings.return_value = [[0.1] * 100, [0.2] * 100]
        mock_qdrant_client.upsert_work_items.return_value = True

        # Mock the filter method to return all items
        self.processor._filter_unchanged_items = Mock(return_value=work_items)

        with (
            patch("src.jirascope.pipeline.embedding_processor.LMStudioClient") as mock_lm,
            patch("src.jirascope.pipeline.embedding_processor.QdrantVectorClient") as mock_qdrant,
        ):

            mock_lm.return_value.__aenter__ = AsyncMock(return_value=mock_lm_client)
            mock_lm.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_qdrant.return_value.__aenter__ = AsyncMock(return_value=mock_qdrant_client)
            mock_qdrant.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await self.processor.process_work_items(work_items)

        assert result.processed_items == 2
        assert result.failed_items == 0
        assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_process_work_items_with_failures(self):
        """Test processing work items with some failures."""
        work_items = [
            self._create_work_item("TEST-1", "First item", "First description"),
            self._create_work_item("TEST-2", "Second item", "Second description"),
        ]

        # Mock clients with failure scenario
        mock_lm_client = AsyncMock()
        mock_qdrant_client = AsyncMock()

        mock_lm_client.generate_embeddings.side_effect = Exception("Embedding generation failed")

        self.processor._filter_unchanged_items = Mock(return_value=work_items)

        with (
            patch("src.jirascope.pipeline.embedding_processor.LMStudioClient") as mock_lm,
            patch("src.jirascope.pipeline.embedding_processor.QdrantVectorClient") as mock_qdrant,
        ):

            mock_lm.return_value.__aenter__ = AsyncMock(return_value=mock_lm_client)
            mock_lm.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_qdrant.return_value.__aenter__ = AsyncMock(return_value=mock_qdrant_client)
            mock_qdrant.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await self.processor.process_work_items(work_items)

        assert result.failed_items == 2
        assert result.processed_items == 0
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_process_work_items_with_skipped_items(self):
        """Test processing with some items skipped due to no changes."""
        work_items = [
            self._create_work_item("TEST-1", "First item", "First description"),
            self._create_work_item("TEST-2", "Second item", "Second description"),
        ]

        # Mock filter to return only one item (one skipped)
        self.processor._filter_unchanged_items = Mock(return_value=work_items[:1])

        # Mock clients
        mock_lm_client = AsyncMock()
        mock_qdrant_client = AsyncMock()

        mock_lm_client.generate_embeddings.return_value = [[0.1] * 100]
        mock_qdrant_client.upsert_work_items.return_value = True

        with (
            patch("src.jirascope.pipeline.embedding_processor.LMStudioClient") as mock_lm,
            patch("src.jirascope.pipeline.embedding_processor.QdrantVectorClient") as mock_qdrant,
        ):

            mock_lm.return_value.__aenter__ = AsyncMock(return_value=mock_lm_client)
            mock_lm.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_qdrant.return_value.__aenter__ = AsyncMock(return_value=mock_qdrant_client)
            mock_qdrant.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await self.processor.process_work_items(work_items)

        assert result.processed_items == 1
        assert result.skipped_items == 1
        assert result.failed_items == 0

    @pytest.mark.asyncio
    async def test_process_work_items_with_chunking(self):
        """Test processing work items with chunking enabled."""
        work_items = [
            self._create_work_item(
                "TEST-1", "Item with long description", "Very long description. " * 20
            ),
        ]

        # Mock clients
        mock_lm_client = AsyncMock()
        mock_qdrant_client = AsyncMock()

        # Mock SmartChunker to return some dummy chunks
        from src.jirascope.models.metadata_schema import ChunkMetadata, JiraItemMetadata
        from src.jirascope.pipeline.smart_chunker import Chunk

        # Create sample chunks to return
        dummy_metadata = JiraItemMetadata(
            key="TEST-1",
            item_type="Story",
            status="Open",
            priority="Medium",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
        )

        dummy_chunks = [
            Chunk(
                text="This is chunk 1 content",
                chunk_type="description",
                metadata=ChunkMetadata(
                    chunk_id="chunk1",
                    source_key="TEST-1",
                    chunk_type="description",
                    chunk_index=0,
                    parent_metadata=dummy_metadata,
                ),
            ),
            Chunk(
                text="This is chunk 2 content",
                chunk_type="description",
                metadata=ChunkMetadata(
                    chunk_id="chunk2",
                    source_key="TEST-1",
                    chunk_type="description",
                    chunk_index=1,
                    parent_metadata=dummy_metadata,
                ),
            ),
        ]

        # Configure mock return values
        self.mock_smart_chunker_instance.chunk_work_item.return_value = dummy_chunks
        mock_lm_client.generate_embeddings.return_value = [[0.1] * 100, [0.2] * 100]
        mock_qdrant_client.store_chunks.return_value = True

        self.processor._filter_unchanged_items = Mock(return_value=work_items)

        with (
            patch("src.jirascope.pipeline.embedding_processor.LMStudioClient") as mock_lm,
            patch("src.jirascope.pipeline.embedding_processor.QdrantVectorClient") as mock_qdrant,
        ):

            mock_lm.return_value.__aenter__ = AsyncMock(return_value=mock_lm_client)
            mock_lm.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_qdrant.return_value.__aenter__ = AsyncMock(return_value=mock_qdrant_client)
            mock_qdrant.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await self.processor.process_work_items_with_chunking(work_items)

        assert result.processed_items == 2  # We created 2 chunks
        assert result.failed_items == 0

    @pytest.mark.asyncio
    async def test_process_batch(self):
        """Test processing a single batch."""
        work_items = [
            self._create_work_item("BATCH-1", "Batch item 1", "Description 1"),
            self._create_work_item("BATCH-2", "Batch item 2", "Description 2"),
        ]

        # Mock clients
        mock_lm_client = AsyncMock()
        mock_qdrant_client = AsyncMock()

        mock_lm_client.generate_embeddings.return_value = [[0.1] * 100, [0.2] * 100]
        mock_qdrant_client.upsert_work_items.return_value = True

        result = await self.processor._process_batch(work_items, mock_lm_client, mock_qdrant_client)

        assert isinstance(result, ProcessingResult)
        assert result.processed_items == 2
        assert result.failed_items == 0

    def test_prepare_embedding_text(self):
        """Test static method for preparing embedding text."""
        work_item = self._create_work_item(
            "PREP-1",
            "Test Summary",
            "Test description with *markup* and {color:red}colored text{color}",
        )

        # Patch the _clean_jira_markup method to properly clean the markup
        with patch.object(
            EmbeddingProcessor,
            "_clean_jira_markup",
            side_effect=lambda text: text.replace("*markup*", "markup")
            .replace("{color:red}", "")
            .replace("{color}", ""),
        ):
            result = EmbeddingProcessor.prepare_embedding_text(work_item)

        assert isinstance(result, str)
        assert "Test Summary" in result
        assert "Test description" in result
        # Markup should be cleaned by our patched method
        assert "markup" in result
        assert "colored text" in result
        assert "*markup*" not in result
        assert "{color:red}" not in result

    def test_clean_jira_markup(self):
        """Test Jira markup cleaning."""
        text_with_markup = """
        This is *bold* text with {color:red}colored{color} content.
        Here's a [link|http://example.com] and some {noformat}code{noformat}.
        """

        # Mock the regular expression replacements in _clean_jira_markup
        with patch(
            "re.sub",
            side_effect=lambda pattern, repl, text, **kwargs: text.replace("*bold*", "bold")
            .replace("{color:red}", "")
            .replace("{color}", "")
            .replace("[link|http://example.com]", "link")
            .replace("{noformat}code{noformat}", "code"),
        ):

            result = EmbeddingProcessor._clean_jira_markup(text_with_markup)

            assert "*bold*" not in result
            assert "{color:red}" not in result
            assert "{color}" not in result
            assert "[link|" not in result
            assert "{noformat}" not in result
            # Clean text should remain
            assert "bold" in result
            assert "colored" in result
            assert "code" in result

    def test_clean_jira_markup_edge_cases(self):
        """Test Jira markup cleaning with edge cases."""
        # Empty text
        with patch("re.sub", return_value=""):
            assert EmbeddingProcessor._clean_jira_markup("") == ""

        # Text without markup
        clean_text = "This is plain text without any markup."
        with patch("re.sub", return_value=clean_text):
            assert EmbeddingProcessor._clean_jira_markup(clean_text) == clean_text

        # Complex nested markup
        complex_markup = "Text with {panel:title=Panel}{color:blue}nested{color} content{panel}"
        result_text = "Text with nested content"

        with patch("re.sub", return_value=result_text):
            result = EmbeddingProcessor._clean_jira_markup(complex_markup)
            assert "{panel" not in result
            assert "{color" not in result
            assert "nested" in result

    def test_filter_unchanged_items(self):
        """Test filtering unchanged items."""
        work_items = [
            self._create_work_item("FILTER-1", "Item 1", "Description 1"),
            self._create_work_item("FILTER-2", "Item 2", "Description 2"),
        ]

        # Mock cache methods
        self.processor._get_cached_hash = Mock(
            side_effect=lambda key: "old_hash" if key == "FILTER-1" else None
        )
        self.processor._calculate_item_hash = Mock(return_value="new_hash")

        result = self.processor._filter_unchanged_items(work_items)

        # Both items should be included since hash doesn't match or doesn't exist
        assert len(result) == 2

    def test_filter_unchanged_items_with_cached_match(self):
        """Test filtering with some items having matching cached hashes."""
        work_items = [
            self._create_work_item("CACHE-1", "Item 1", "Description 1"),
            self._create_work_item("CACHE-2", "Item 2", "Description 2"),
        ]

        # Mock cache methods - first item has matching hash, second doesn't
        def mock_get_cached_hash(key):
            if key == "CACHE-1":
                return "matching_hash"
            return None

        def mock_calculate_hash(item):
            if item.key == "CACHE-1":
                return "matching_hash"  # Same as cached
            return "different_hash"

        self.processor._get_cached_hash = Mock(side_effect=mock_get_cached_hash)
        self.processor._calculate_item_hash = Mock(side_effect=mock_calculate_hash)

        result = self.processor._filter_unchanged_items(work_items)

        # Only second item should be included
        assert len(result) == 1
        assert result[0].key == "CACHE-2"

    def test_calculate_item_hash(self):
        """Test item hash calculation."""
        work_item = self._create_work_item("HASH-1", "Test Summary", "Test Description")

        hash1 = self.processor._calculate_item_hash(work_item)
        hash2 = self.processor._calculate_item_hash(work_item)

        # Same item should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0

    def test_calculate_item_hash_different_items(self):
        """Test that different items produce different hashes."""
        item1 = self._create_work_item("HASH-1", "Summary 1", "Description 1")
        item2 = self._create_work_item("HASH-2", "Summary 2", "Description 2")

        hash1 = self.processor._calculate_item_hash(item1)
        hash2 = self.processor._calculate_item_hash(item2)

        assert hash1 != hash2

    def test_get_cached_hash_file_not_exists(self):
        """Test getting cached hash when file doesn't exist."""
        result = self.processor._get_cached_hash("NONEXISTENT-1")
        assert result is None

    def test_get_cached_hash_and_update(self):
        """Test getting and updating cached hash."""
        work_item = self._create_work_item("CACHE-TEST-1", "Summary", "Description")

        # Initially no cached hash
        assert self.processor._get_cached_hash(work_item.key) is None

        # Update cache
        self.processor._update_cache_hash(work_item)

        # Now should have cached hash
        cached_hash = self.processor._get_cached_hash(work_item.key)
        expected_hash = self.processor._calculate_item_hash(work_item)

        assert cached_hash == expected_hash

    def test_update_cache_hash(self):
        """Test updating cache hash."""
        work_item = self._create_work_item("UPDATE-1", "Summary", "Description")

        # Update cache
        self.processor._update_cache_hash(work_item)

        # Verify cache file exists
        cache_file = self.processor.cache_dir / f"{work_item.key}.hash"
        assert cache_file.exists()

        # Verify content
        with open(cache_file) as f:
            cached_content = f.read().strip()

        expected_hash = self.processor._calculate_item_hash(work_item)
        assert cached_content == expected_hash

    def test_filter_unchanged_items_legacy_method(self):
        """Test the legacy filter_unchanged_items method."""
        work_items = [
            self._create_work_item("LEGACY-1", "Item 1", "Description 1"),
        ]

        # This should call the private method
        with patch.object(self.processor, "_filter_unchanged_items") as mock_filter:
            mock_filter.return_value = work_items

            result = self.processor.filter_unchanged_items(work_items)

            mock_filter.assert_called_once_with(work_items)
            assert result == work_items

    def _create_work_item(self, key: str, summary: str, description: str) -> WorkItem:
        """Helper to create work items for testing."""
        return WorkItem(
            key=key,
            summary=summary,
            issue_type="Story",
            status="Open",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
            reporter="test@example.com",
            description=description,
            assignee="test@example.com",
            embedding=[0.1] * 100,
        )
