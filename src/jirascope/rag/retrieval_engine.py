"""Contextual retrieval engine for Jira semantic search."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .query_processor import ExpandedQuery, FilterSet


@dataclass
class RetrievalResult:
    """Result from semantic search with metadata."""

    score: float
    content: Dict[str, Any]
    chunk_id: str
    source_key: str
    item_type: str

    @property
    def jira_key(self) -> str:
        """Get the Jira key for this result."""
        return self.content.get("key", self.source_key)


@dataclass
class ContextTree:
    """Hierarchical context for a Jira item."""

    root_item: Dict[str, Any]
    parent_context: Optional[Dict[str, Any]] = None
    child_items: Optional[List[Dict[str, Any]]] = None
    related_items: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self):
        """Initialize list fields if None."""
        if self.child_items is None:
            self.child_items = []
        if self.related_items is None:
            self.related_items = []

    @property
    def total_items(self) -> int:
        """Total number of items in this context tree."""
        return 1 + len(self.child_items or []) + len(self.related_items or [])


class ContextualRetriever:
    """Advanced retrieval engine with hierarchical context awareness."""

    def __init__(self, qdrant_client, embedding_client):
        self.qdrant = qdrant_client
        self.embedder = embedding_client

    async def semantic_search(
        self, query: ExpandedQuery, filters: FilterSet, limit: int = 10
    ) -> List[RetrievalResult]:
        """Multi-stage semantic retrieval with filtering and ranking."""
        # Generate embedding for the expanded query
        query_embedding = await self.embedder.generate_embeddings([query.full_query_text])

        if not query_embedding:
            return []

        # Convert filters to Qdrant format
        qdrant_filters = filters.to_qdrant_filter()

        # Perform semantic search with filters
        search_results = await self.qdrant.search_with_filters(
            query_embedding=query_embedding[0],
            filters=qdrant_filters,
            limit=limit * 2,  # Get more results for re-ranking
        )

        # Convert to RetrievalResult objects
        results = []
        for result in search_results:
            retrieval_result = RetrievalResult(
                score=result["score"],
                content=result["content"],
                chunk_id=result["content"].get("chunk_id", ""),
                source_key=result["content"].get("source_key", ""),
                item_type=result["content"].get("item_type", "Unknown"),
            )
            results.append(retrieval_result)

        # Re-rank results based on additional factors
        ranked_results = self._rerank_results(results, query, filters)

        return ranked_results[:limit]

    async def hierarchical_retrieval(self, item_key: str, depth: int = 2) -> ContextTree:
        """Retrieve item with full hierarchical context."""
        # Get the root item
        root_item = await self.qdrant.get_work_item_by_key(item_key)

        if not root_item:
            return ContextTree(root_item={})

        context_tree = ContextTree(root_item=root_item)

        # Get parent context if item has a parent
        if root_item.get("epic_key") or root_item.get("parent_key"):
            parent_key = root_item.get("epic_key") or root_item.get("parent_key")
            parent_item = await self.qdrant.get_work_item_by_key(parent_key)
            if parent_item:
                context_tree.parent_context = parent_item

        # Get child items if this is an Epic or parent item
        if root_item.get("item_type") == "Epic":
            child_results = await self.qdrant.search_with_filters(
                query_embedding=[0.0] * 1536,  # Dummy embedding for filter-only search
                filters={"epic_key": [item_key]},
                limit=50,
            )
            context_tree.child_items = [result["content"] for result in child_results]

        return context_tree

    async def search_by_epic(
        self, query_embedding: List[float], epic_key: str, limit: int = 10
    ) -> List[RetrievalResult]:
        """Search within a specific Epic hierarchy."""
        search_results = await self.qdrant.search_with_filters(
            query_embedding=query_embedding, filters={"epic_key": [epic_key]}, limit=limit
        )

        results = []
        for result in search_results:
            retrieval_result = RetrievalResult(
                score=result["score"],
                content=result["content"],
                chunk_id=result["content"].get("chunk_id", ""),
                source_key=result["content"].get("source_key", ""),
                item_type=result["content"].get("item_type", "Unknown"),
            )
            results.append(retrieval_result)

        return results

    def _rerank_results(
        self, results: List[RetrievalResult], query: ExpandedQuery, filters: FilterSet
    ) -> List[RetrievalResult]:
        """Re-rank results based on additional relevance factors."""
        for result in results:
            # Start with semantic similarity score
            relevance_score = result.score

            # Boost based on item type preferences
            if filters.item_types and result.item_type in filters.item_types:
                relevance_score *= 1.2

            # Boost high priority items
            if result.content.get("priority") == "High":
                relevance_score *= 1.1

            # Boost recent items
            created_month = result.content.get("created_month", "")
            if created_month and "2024" in created_month:
                relevance_score *= 1.05

            # Update the score
            result.score = relevance_score

        # Sort by updated relevance score
        return sorted(results, key=lambda x: x.score, reverse=True)
