"""Optimized embedding pipeline with adaptive batching."""

import hashlib
import re
import time
from pathlib import Path

from ..clients.lmstudio_client import LMStudioClient
from ..clients.qdrant_client import QdrantVectorClient
from ..core.config import Config
from ..models import ProcessingResult, WorkItem
from ..pipeline.smart_chunker import SmartChunker
from ..utils.logging import StructuredLogger

logger = StructuredLogger(__name__)


class AdaptiveBatcher:
    """Adaptive batching based on text length and performance history."""

    def __init__(self, base_batch_size: int = 32):
        self.base_batch_size = base_batch_size
        self.performance_history = []
        self.max_history = 10

    def calculate_optimal_batch_size(self, items: list[WorkItem]) -> int:
        """Adjust batch size based on text length and performance history."""
        if not items:
            return self.base_batch_size

        # Calculate average text length from sample
        sample_size = min(10, len(items))
        sample_texts = [self._prepare_embedding_text(item) for item in items[:sample_size]]
        avg_text_length = sum(len(text) for text in sample_texts) / len(sample_texts)

        # Adjust based on text length
        if avg_text_length > 1000:
            suggested_size = max(8, self.base_batch_size // 4)  # Large texts
        elif avg_text_length > 500:
            suggested_size = max(16, self.base_batch_size // 2)  # Medium texts
        else:
            suggested_size = self.base_batch_size  # Short texts

        # Adjust based on performance history
        if self.performance_history:
            avg_performance = sum(self.performance_history) / len(self.performance_history)
            if avg_performance > 2.0:  # Slow processing
                suggested_size = max(8, suggested_size // 2)
            elif avg_performance < 0.5:  # Fast processing
                suggested_size = min(64, suggested_size * 2)

        logger.debug(
            f"Calculated optimal batch size: {suggested_size} (avg_text_length: {avg_text_length:.0f})"
        )
        return suggested_size

    def record_performance(self, batch_size: int, processing_time: float, items_count: int):
        """Record performance metrics for future optimization."""
        time_per_item = processing_time / max(1, items_count)
        self.performance_history.append(time_per_item)

        # Keep only recent history
        if len(self.performance_history) > self.max_history:
            self.performance_history.pop(0)

    def _prepare_embedding_text(self, item: WorkItem) -> str:
        """Prepare text for length calculation."""
        return EmbeddingProcessor.prepare_embedding_text(item)


class EmbeddingProcessor:
    """Efficient batch processing with quality validation and cost optimization."""

    def __init__(self, config: Config):
        self.config = config
        self.batcher = AdaptiveBatcher(config.embedding_batch_size)
        self.chunker = SmartChunker(max_chunk_size=500)
        self.cache_dir = Path.home() / ".jirascope" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def process_work_items(self, items: list[WorkItem]) -> ProcessingResult:
        """Batch process work items with adaptive batch sizing."""
        logger.info(f"Starting processing of {len(items)} work items")
        start_time = time.time()

        result = ProcessingResult()

        try:
            async with (
                LMStudioClient(self.config) as lm_client,
                QdrantVectorClient(self.config) as qdrant_client,
            ):
                # Filter out items that haven't changed (for incremental processing)
                items_to_process = self._filter_unchanged_items(items)
                result.skipped_items = len(items) - len(items_to_process)

                if not items_to_process:
                    logger.info("No items need processing")
                    return result

                # Process in adaptive batches
                batch_size = self.batcher.calculate_optimal_batch_size(items_to_process)

                for i in range(0, len(items_to_process), batch_size):
                    batch_start = time.time()
                    batch = items_to_process[i : i + batch_size]

                    try:
                        batch_result = await self._process_batch(batch, lm_client, qdrant_client)

                        result.processed_items += batch_result.processed_items
                        result.failed_items += batch_result.failed_items
                        result.total_cost += batch_result.total_cost
                        result.errors.extend(batch_result.errors)

                        # Record performance for adaptive batching
                        batch_time = time.time() - batch_start
                        self.batcher.record_performance(len(batch), batch_time, len(batch))

                        # Update batch stats
                        batch_num = i // batch_size + 1
                        result.batch_stats[f"batch_{batch_num}_time"] = batch_time
                        result.batch_stats[f"batch_{batch_num}_items"] = len(batch)

                        logger.debug(
                            f"Processed batch {batch_num}: {len(batch)} items in {batch_time:.2f}s"
                        )

                    except Exception as e:
                        logger.exception(
                            f"Failed to process batch {i//batch_size + 1}", error=str(e)
                        )
                        result.failed_items += len(batch)
                        result.errors.append(f"Batch {i//batch_size + 1}: {e!s}")
                        continue

            result.processing_time = time.time() - start_time

            logger.log_operation(
                "process_work_items",
                result.processing_time,
                success=result.failed_items == 0,
                processed=result.processed_items,
                failed=result.failed_items,
                skipped=result.skipped_items,
                success_rate=result.success_rate,
            )

            return result

        except Exception as e:
            logger.exception("Failed to process work items", error=str(e))
            result.errors.append(f"Processing failed: {e!s}")
            result.processing_time = time.time() - start_time
            return result

    async def process_work_items_with_chunking(self, items: list[WorkItem]) -> ProcessingResult:
        """Process work items using smart chunking strategy."""
        logger.info(f"Starting chunked processing of {len(items)} work items")
        start_time = time.time()

        result = ProcessingResult()

        try:
            async with (
                LMStudioClient(self.config) as lm_client,
                QdrantVectorClient(self.config) as qdrant_client,
            ):
                # Filter unchanged items
                items_to_process = self._filter_unchanged_items(items)
                result.skipped_items = len(items) - len(items_to_process)

                if not items_to_process:
                    logger.info("No items need processing")
                    return result

                # Generate chunks for all items
                all_chunks = []
                for item in items_to_process:
                    chunks = self.chunker.chunk_work_item(item)
                    all_chunks.extend(chunks)

                logger.info(
                    f"Generated {len(all_chunks)} chunks from {len(items_to_process)} items"
                )

                # Process chunks in batches
                batch_size = self.batcher.calculate_optimal_batch_size(items_to_process)

                for i in range(0, len(all_chunks), batch_size):
                    batch_start = time.time()
                    chunk_batch = all_chunks[i : i + batch_size]

                    try:
                        # Extract text from chunks
                        texts = [chunk.text for chunk in chunk_batch]

                        # Generate embeddings
                        embeddings = await lm_client.generate_embeddings(texts)

                        if len(embeddings) != len(chunk_batch):
                            raise ValueError(
                                f"Embedding count mismatch: {len(embeddings)} != {len(chunk_batch)}"
                            )

                        # Store chunks in Qdrant
                        await qdrant_client.store_chunks(chunk_batch, embeddings)

                        result.processed_items += len(chunk_batch)
                        result.total_cost += len(chunk_batch) * 0.0001

                        # Record performance
                        batch_time = time.time() - batch_start
                        self.batcher.record_performance(
                            len(chunk_batch), batch_time, len(chunk_batch)
                        )

                        batch_num = i // batch_size + 1
                        result.batch_stats[f"batch_{batch_num}_time"] = batch_time
                        result.batch_stats[f"batch_{batch_num}_chunks"] = len(chunk_batch)

                        logger.debug(
                            f"Processed chunk batch {batch_num}: {len(chunk_batch)} chunks in {batch_time:.2f}s"
                        )

                    except Exception as e:
                        logger.exception(
                            f"Failed to process chunk batch {i//batch_size + 1}", error=str(e)
                        )
                        result.failed_items += len(chunk_batch)
                        result.errors.append(f"Chunk batch {i//batch_size + 1}: {e!s}")
                        continue

                # Update cache hashes for processed items
                for item in items_to_process:
                    self._update_cache_hash(item)

            result.processing_time = time.time() - start_time

            logger.log_operation(
                "process_work_items_with_chunking",
                result.processing_time,
                success=result.failed_items == 0,
                processed=result.processed_items,
                failed=result.failed_items,
                skipped=result.skipped_items,
                success_rate=result.success_rate,
            )

            return result

        except Exception as e:
            logger.exception("Failed to process work items with chunking", error=str(e))
            result.errors.append(f"Chunked processing failed: {e!s}")
            result.processing_time = time.time() - start_time
            return result

    async def _process_batch(
        self, batch: list[WorkItem], lm_client: LMStudioClient, qdrant_client: QdrantVectorClient
    ) -> ProcessingResult:
        """Process a single batch of work items."""
        result = ProcessingResult()

        try:
            # Prepare texts for embedding
            texts = [self.prepare_embedding_text(item) for item in batch]

            # Generate embeddings
            embeddings = await lm_client.generate_embeddings(texts)

            if len(embeddings) != len(batch):
                raise ValueError(f"Embedding count mismatch: {len(embeddings)} != {len(batch)}")

            # Store in Qdrant
            await qdrant_client.store_work_items(batch, embeddings)

            # Update cache hashes
            for item in batch:
                self._update_cache_hash(item)

            result.processed_items = len(batch)
            # Rough cost estimate: $0.0001 per embedding
            result.total_cost = len(batch) * 0.0001

            if logger.cost_tracker:
                logger.log_cost(
                    "embedding",
                    "generate_and_store",
                    result.total_cost,
                    {"items_count": len(batch), "embeddings_count": len(embeddings)},
                )

        except Exception as e:
            result.failed_items = len(batch)
            result.errors.append(str(e))

        return result

    @staticmethod
    def prepare_embedding_text(item: WorkItem) -> str:
        """Optimized text preparation for BGE-large-en-v1.5."""
        parts = [f"Title: {item.summary}", f"Type: {item.issue_type}", f"Status: {item.status}"]

        if item.description:
            # Clean Jira markup and truncate
            clean_desc = EmbeddingProcessor._clean_jira_markup(item.description)
            if len(clean_desc) > 300:
                clean_desc = clean_desc[:300] + "..."
            parts.append(f"Description: {clean_desc}")

        if item.components:
            parts.append(f"Components: {', '.join(item.components)}")

        if item.labels:
            parts.append(f"Labels: {', '.join(item.labels)}")

        # Add hierarchy context
        if item.epic_key:
            parts.append(f"Epic: {item.epic_key}")
        if item.parent_key:
            parts.append(f"Parent: {item.parent_key}")

        text = " | ".join(parts)

        # Ensure we don't exceed max tokens (default 512 tokens * 4 chars/token estimate)
        max_chars = 512 * 4  # Rough estimate for max tokens
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        return text

    @staticmethod
    def _clean_jira_markup(text: str) -> str:
        """Clean Jira markup from text."""
        if not text:
            return ""

        # Remove common Jira markup
        # Remove {code} blocks
        text = re.sub(r"\{code[^}]*\}.*?\{code\}", "[CODE BLOCK]", text, flags=re.DOTALL)
        # Remove {quote} blocks
        text = re.sub(r"\{quote\}.*?\{quote\}", "[QUOTE]", text, flags=re.DOTALL)
        # Remove {panel} blocks
        text = re.sub(r"\{panel[^}]*\}.*?\{panel\}", "[PANEL]", text, flags=re.DOTALL)
        # Remove links [text|url]
        text = re.sub(r"\[[^|\]]+\|[^\]]+\]", lambda m: m.group(0).split("|")[0][1:], text)
        # Remove formatting
        text = re.sub(r"\*([^*]+)\*", r"\1", text)  # Bold
        text = re.sub(r"_([^_]+)_", r"\1", text)  # Italic
        text = re.sub(r"\^([^\^]+)\^", r"\1", text)  # Superscript
        text = re.sub(r"~([^~]+)~", r"\1", text)  # Subscript

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _filter_unchanged_items(self, items: list[WorkItem]) -> list[WorkItem]:
        """Filter out items that haven't changed since last processing."""
        items_to_process = []

        for item in items:
            cached_hash = self._get_cached_hash(item.key)
            current_hash = self._calculate_item_hash(item)

            if cached_hash != current_hash:
                items_to_process.append(item)

        return items_to_process

    def _calculate_item_hash(self, item: WorkItem) -> str:
        """Calculate hash of item content for change detection."""
        content = f"{item.key}|{item.summary}|{item.description or ''}|{item.updated.isoformat()}"
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

    def _get_cached_hash(self, item_key: str) -> str | None:
        """Get cached hash for an item."""
        cache_file = self.cache_dir / f"{item_key}.hash"

        if cache_file.exists():
            try:
                return cache_file.read_text().strip()
            except Exception:
                return None

        return None

    def _update_cache_hash(self, item: WorkItem):
        """Update cached hash for an item."""
        cache_file = self.cache_dir / f"{item.key}.hash"
        current_hash = self._calculate_item_hash(item)

        try:
            cache_file.write_text(current_hash)
        except Exception as e:
            logger.warning(f"Failed to update cache hash for {item.key}", error=str(e))

    def filter_unchanged_items(self, items, *args, **kwargs):
        """Public method for filtering unchanged items (for test compatibility)."""
        return self._filter_unchanged_items(items)
