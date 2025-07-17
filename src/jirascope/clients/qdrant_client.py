"""Qdrant client for vector storage and retrieval."""

import logging
from typing import TYPE_CHECKING, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from ..core.config import Config
from ..models import WorkItem

if TYPE_CHECKING:
    from ..pipeline.smart_chunker import Chunk

logger = logging.getLogger(__name__)


# Enhanced collection schema for hierarchical Jira data
JIRASCOPE_COLLECTION_CONFIG = {
    "vectors": {"size": 1536, "distance": "Cosine"},  # OpenAI/BGE embedding size
    "payload_schema": {
        # Hierarchical filters
        "epic_key": {"type": "keyword", "index": True},
        "item_type": {"type": "keyword", "index": True},
        "status": {"type": "keyword", "index": True},
        "priority": {"type": "keyword", "index": True},
        # Team/component filters
        "team": {"type": "keyword", "index": True},
        "components": {"type": "keyword", "index": True},
        # Temporal filters
        "created_month": {"type": "keyword", "index": True},
        # Chunk-specific filters
        "chunk_type": {"type": "keyword", "index": True},
        "source_key": {"type": "keyword", "index": True},
        # Relationship filters
        "has_children": {"type": "bool", "index": True},
        "dependency_count": {"type": "integer", "index": True},
    },
}


class QdrantVectorClient:
    """Client for managing work item vectors in Qdrant."""

    def __init__(self, config: Config):
        self.config = config
        self.client = QdrantClient(url=config.qdrant_url)
        self.collection_name = "jirascope_work_items"

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize_collection()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""

    async def initialize_collection(self):
        """Initialize the Qdrant collection if it doesn't exist."""
        try:
            collections_info = self.client.get_collections()
            collection_names = [c.name for c in collections_info.collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.embedding_dimensions, distance=Distance.COSINE
                    ),
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            logger.exception(f"Failed to initialize collection: {e}")
            raise

    async def store_work_items(self, work_items: list[WorkItem], embeddings: list[list[float]]):
        """Store work items with their embeddings in Qdrant."""
        if len(work_items) != len(embeddings):
            raise ValueError("Number of work items must match number of embeddings")

        points = []
        for work_item, embedding in zip(work_items, embeddings, strict=False):
            point = PointStruct(
                id=abs(hash(work_item.key)),  # Use absolute hash of key as ID
                vector=embedding,
                payload={
                    "key": work_item.key,
                    "summary": work_item.summary,
                    "description": work_item.description or "",
                    "issue_type": work_item.issue_type,
                    "status": work_item.status,
                    "parent_key": work_item.parent_key,
                    "epic_key": work_item.epic_key,
                    "created": work_item.created.isoformat(),
                    "updated": work_item.updated.isoformat(),
                    "assignee": work_item.assignee,
                    "reporter": work_item.reporter,
                    "components": work_item.components,
                    "labels": work_item.labels,
                },
            )
            points.append(point)

        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Stored {len(points)} work items in Qdrant")

        except Exception as e:
            logger.exception(f"Failed to store work items: {e}")
            raise

    async def store_chunks(self, chunks: list["Chunk"], embeddings: list[list[float]]):
        """Store text chunks with their embeddings and enhanced metadata."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        points = []
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            # Use chunk metadata for payload
            payload = chunk.metadata.to_qdrant_payload()
            payload["text"] = chunk.text  # Include the actual text

            point = PointStruct(
                id=abs(hash(chunk.chunk_id)),
                vector=embedding,
                payload=payload,  # Use chunk ID hash
            )
            points.append(point)

        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Stored {len(points)} chunks in Qdrant")

        except Exception as e:
            logger.exception(f"Failed to store chunks: {e}")
            raise

    async def search_similar_work_items(
        self, query_embedding: list[float], limit: int = 10, score_threshold: float | None = None
    ) -> list[dict[str, Any]]:
        """Search for similar work items using vector similarity."""
        try:
            score_threshold = score_threshold or self.config.similarity_threshold

            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
            )

            results = []
            for hit in search_result:
                result = {"score": hit.score, "work_item": hit.payload}
                results.append(result)

            logger.info(f"Found {len(results)} similar work items")
            return results

        except Exception as e:
            logger.exception(f"Failed to search similar work items: {e}")
            raise

    async def search_with_filters(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Enhanced search with metadata filtering."""
        try:
            score_threshold = score_threshold or self.config.similarity_threshold

            # Build Qdrant filter
            qdrant_filter = None
            if filters:
                conditions = []

                for key, value in filters.items():
                    if isinstance(value, list):
                        # Match any value in list
                        conditions.append(
                            models.FieldCondition(key=key, match=models.MatchAny(any=value))
                        )
                    else:
                        # Exact match
                        conditions.append(
                            models.FieldCondition(key=key, match=models.MatchValue(value=value))
                        )

                if conditions:
                    qdrant_filter = models.Filter(must=conditions)

            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=limit,
                score_threshold=score_threshold,
            )

            results = []
            for hit in search_result:
                result = {"score": hit.score, "content": hit.payload}
                results.append(result)

            logger.info(f"Found {len(results)} filtered results")
            return results

        except Exception as e:
            logger.exception(f"Failed to search with filters: {e}")
            raise

    async def search_by_epic(
        self, query_embedding: list[float], epic_key: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Search within a specific Epic hierarchy."""
        filters = {"epic_key": epic_key}
        return await self.search_with_filters(query_embedding, filters, limit)

    async def search_by_item_type(
        self, query_embedding: list[float], item_types: list[str], limit: int = 10
    ) -> list[dict[str, Any]]:
        """Search for specific item types (Story, Task, Bug, etc.)."""
        filters = {"item_type": item_types}
        return await self.search_with_filters(query_embedding, filters, limit)

    async def get_work_item_by_key(self, key: str) -> dict[str, Any] | None:
        """Retrieve a work item by its key."""
        try:
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="key", match=models.MatchValue(value=key))]
                ),
                limit=1,
            )

            if search_result[0]:  # Points found
                point = search_result[0][0]
                return point.payload

            return None

        except Exception as e:
            logger.exception(f"Failed to get work item {key}: {e}")
            raise

    async def delete_work_item(self, key: str) -> bool:
        """Delete a work item from the collection."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[hash(key)]),
            )
            logger.info(f"Deleted work item {key}")
            return True

        except Exception as e:
            logger.exception(f"Failed to delete work item {key}: {e}")
            return False

    async def health_check(self) -> bool:
        """Check if Qdrant is running and accessible."""
        try:
            self.client.get_collections()
        except Exception as e:
            logger.exception(f"Qdrant health check failed: {e}")
            return False
        else:
            logger.info("Qdrant health check passed")
            return True

    async def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status,
            }

        except Exception as e:
            logger.exception(f"Failed to get collection stats: {e}")
            return {}
