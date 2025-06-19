"""Qdrant client for vector storage and retrieval."""

import logging
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from ..core.config import Config, EMBEDDING_CONFIG
from ..models import WorkItem


logger = logging.getLogger(__name__)


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
        pass
    
    async def initialize_collection(self):
        """Initialize the Qdrant collection if it doesn't exist."""
        try:
            collections_info = self.client.get_collections()
            collection_names = [c.name for c in collections_info.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=EMBEDDING_CONFIG["dimensions"],
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    async def store_work_items(
        self, 
        work_items: List[WorkItem], 
        embeddings: List[List[float]]
    ):
        """Store work items with their embeddings in Qdrant."""
        if len(work_items) != len(embeddings):
            raise ValueError("Number of work items must match number of embeddings")
        
        points = []
        for work_item, embedding in zip(work_items, embeddings):
            point = PointStruct(
                id=hash(work_item.key),  # Use hash of key as ID
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
                    "labels": work_item.labels
                }
            )
            points.append(point)
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Stored {len(points)} work items in Qdrant")
            
        except Exception as e:
            logger.error(f"Failed to store work items: {e}")
            raise
    
    async def search_similar_work_items(
        self, 
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar work items using vector similarity."""
        try:
            score_threshold = score_threshold or self.config.similarity_threshold
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            
            results = []
            for hit in search_result:
                result = {
                    "score": hit.score,
                    "work_item": hit.payload
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar work items")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar work items: {e}")
            raise
    
    async def get_work_item_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a work item by its key."""
        try:
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="key",
                            match=models.MatchValue(value=key)
                        )
                    ]
                ),
                limit=1
            )
            
            if search_result[0]:  # Points found
                point = search_result[0][0]
                return point.payload
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get work item {key}: {e}")
            raise
    
    async def delete_work_item(self, key: str) -> bool:
        """Delete a work item from the collection."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[hash(key)]
                )
            )
            logger.info(f"Deleted work item {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete work item {key}: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check if Qdrant is running and accessible."""
        try:
            self.client.get_collections()
            logger.info("Qdrant health check passed")
            return True
            
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "disk_data_size": info.disk_data_size,
                "ram_data_size": info.ram_data_size
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}