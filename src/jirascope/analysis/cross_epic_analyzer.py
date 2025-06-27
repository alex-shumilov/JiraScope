"""Cross-Epic analysis for finding misplaced work items."""

import time
from collections import defaultdict
from typing import Dict, List, Optional

from ..clients.claude_client import ClaudeClient
from ..clients.lmstudio_client import LMStudioClient
from ..clients.qdrant_client import QdrantVectorClient
from ..core.config import Config
from ..models import CrossEpicReport, MisplacedWorkItem, WorkItem
from ..utils.logging import StructuredLogger

logger = StructuredLogger(__name__)


class CrossEpicAnalyzer:
    """Analyze work items across Epics to find misplaced items."""

    def __init__(self, config: Config):
        self.config = config
        self.qdrant_client: Optional[QdrantVectorClient] = None
        self.lm_client: Optional[LMStudioClient] = None
        self.claude_client: Optional[ClaudeClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.qdrant_client = QdrantVectorClient(self.config)
        self.lm_client = LMStudioClient(self.config)
        self.claude_client = ClaudeClient(self.config)

        await self.qdrant_client.__aenter__()
        await self.lm_client.__aenter__()
        await self.claude_client.__aenter__()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.qdrant_client:
            await self.qdrant_client.__aexit__(exc_type, exc_val, exc_tb)
        if self.lm_client:
            await self.lm_client.__aexit__(exc_type, exc_val, exc_tb)
        if self.claude_client:
            await self.claude_client.__aexit__(exc_type, exc_val, exc_tb)

    async def find_misplaced_work_items(self, project_key: Optional[str] = None) -> CrossEpicReport:
        """Find work items that might belong to different Epics."""
        logger.info(f"Starting cross-Epic analysis for project {project_key or 'all'}")
        start_time = time.time()

        if not self.qdrant_client or not self.lm_client:
            raise RuntimeError("CrossEpicAnalyzer not initialized. Use async context manager.")

        try:
            # Get all Epics and their work items
            epics_data = await self._get_all_epics_with_items(project_key)

            if len(epics_data) < 2:
                logger.info("Less than 2 Epics found, skipping cross-Epic analysis")
                return CrossEpicReport(epics_analyzed=len(epics_data))

            misplaced_items = []

            # Analyze each Epic's work items
            for epic_key, epic_items in epics_data.items():
                if len(epic_items) < 2:  # Skip Epics with too few items
                    continue

                logger.debug(f"Analyzing Epic {epic_key} with {len(epic_items)} items")

                # Calculate Epic's thematic coherence
                # Skip embedding calculation for now - not needed for this part

                # Check each work item against other Epics
                for item_data in epic_items:
                    item_key = item_data["key"]
                    item_embedding = item_data.get("embedding")

                    if not item_embedding:
                        continue

                    # Find similar items across other Epics
                    cross_epic_matches = await self._find_similar_across_epics(
                        item_embedding, exclude_epic=epic_key, threshold=0.65
                    )

                    if not cross_epic_matches:
                        continue

                    # Calculate coherence with current Epic (placeholder for now)
                    current_coherence = (
                        0.5  # Will be properly calculated when epic theme embedding is implemented
                    )

                    # Find best matching Epic
                    best_match = None
                    best_coherence = current_coherence

                    for match in cross_epic_matches:
                        match_epic = match["epic_key"]

                        # Calculate coherence with this Epic
                        if match_epic in epics_data:
                            # Skip other epic theme calculation for now
                            other_epic_theme = [0.0] * 1024  # Default embedding
                            other_coherence = self.lm_client.calculate_similarity(
                                item_embedding, other_epic_theme
                            )

                            if other_coherence > best_coherence + 0.15:  # Significant difference
                                best_coherence = other_coherence
                                best_match = {
                                    "epic_key": match_epic,
                                    "coherence": other_coherence,
                                    "similarity_score": match["score"],
                                }

                    # If we found a significantly better Epic
                    if best_match:
                        reasoning = self._generate_misplacement_reasoning(
                            item_data, epic_key, best_match, current_coherence
                        )

                        misplaced_item = MisplacedWorkItem(
                            work_item_key=item_key,
                            current_epic_key=epic_key,
                            suggested_epic_key=best_match["epic_key"],
                            confidence_score=best_match["coherence"],
                            coherence_difference=best_coherence - current_coherence,
                            reasoning=reasoning,
                        )

                        misplaced_items.append(misplaced_item)

                        logger.debug(
                            f"Found misplaced item: {item_key} (current: {epic_key}, suggested: {best_match['epic_key']})"
                        )

            processing_time = time.time() - start_time
            estimated_cost = sum(len(items) for items in epics_data.values()) * 0.0002

            logger.log_operation(
                "find_misplaced_work_items",
                processing_time,
                success=True,
                epics_analyzed=len(epics_data),
                misplaced_items_found=len(misplaced_items),
            )

            return CrossEpicReport(
                misplaced_items=misplaced_items,
                epics_analyzed=len(epics_data),
                processing_cost=estimated_cost,
            )

        except Exception as e:
            logger.error("Failed to find misplaced work items", error=str(e))
            raise

    async def _get_all_epics_with_items(
        self, project_key: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """Get all Epics and their work items from Qdrant."""
        epics_data = defaultdict(list)

        # Build filter for project if specified
        scroll_filter = None
        if project_key:
            scroll_filter = {"must": [{"key": "key", "match": {"value": f"{project_key}-*"}}]}

        # Get all work items from Qdrant
        scroll_result = await self.qdrant_client.client.scroll(
            collection_name="jirascope_work_items",
            scroll_filter=scroll_filter,
            limit=10000,  # Adjust based on your data size
            with_vectors=True,
        )

        # Group by Epic
        for point in scroll_result[0]:
            payload = point.payload
            epic_key = payload.get("epic_key")

            if epic_key:  # Only include items that belong to an Epic
                item_data = payload.copy()
                item_data["embedding"] = point.vector
                epics_data[epic_key].append(item_data)

        # Filter out Epics with too few items
        epics_data = {k: v for k, v in epics_data.items() if len(v) >= 2}

        logger.debug(f"Found {len(epics_data)} Epics with sufficient work items")
        return dict(epics_data)

    async def _calculate_epic_theme_embedding(
        self, epic: WorkItem, work_items: List[WorkItem]
    ) -> List[float]:
        """Calculate a representative embedding for an Epic's theme."""
        epic_key = epic.key

        # Filter work items for this epic
        epic_items = [item for item in work_items if getattr(item, "epic_key", None) == epic_key]

        if not epic_items:
            return [0.0] * 1024  # Default embedding size

        # Get embeddings for the epic items
        if not self.lm_client:
            raise RuntimeError("LM client not initialized")

        # Prepare texts for embedding
        from ..pipeline.embedding_processor import EmbeddingProcessor

        texts = [EmbeddingProcessor.prepare_embedding_text(item) for item in epic_items]
        embeddings = await self.lm_client.generate_embeddings(texts)

        if not embeddings:
            return [0.0] * 1024

        # Calculate component-wise average
        avg_embedding = []
        embedding_length = len(embeddings[0])

        for i in range(embedding_length):
            avg_value = sum(emb[i] for emb in embeddings) / len(embeddings)
            avg_embedding.append(avg_value)

        return avg_embedding

    async def _calculate_epic_coherence_score(
        self, epic_theme: List[float], work_items: List[WorkItem]
    ) -> float:
        """Calculate coherence score between epic theme and work items."""
        if len(work_items) < 1:
            return 0.0

        # Get embeddings for all items
        embeddings = []
        items_needing_embeddings = []

        # Collect items that already have embeddings and those that don't
        for item in work_items:
            if item.embedding:
                embeddings.append(item.embedding)
            else:
                items_needing_embeddings.append(item)

        # Generate embeddings for items that don't have them
        if items_needing_embeddings:
            if not self.lm_client:
                raise RuntimeError("LM client not initialized")
            from ..pipeline.embedding_processor import EmbeddingProcessor

            texts = [
                EmbeddingProcessor.prepare_embedding_text(item) for item in items_needing_embeddings
            ]
            new_embeddings = await self.lm_client.generate_embeddings(texts)
            embeddings.extend(new_embeddings)

        if not embeddings:
            return 0.0

        # Calculate similarities between epic theme and work items
        similarities = []
        for embedding in embeddings:
            similarity = self.lm_client.calculate_similarity(epic_theme, embedding)
            similarities.append(similarity)

        # Return average similarity as coherence score
        return sum(similarities) / len(similarities) if similarities else 0.0

    async def _find_similar_across_epics(
        self, item_embedding: List[float], exclude_epic: str, threshold: float = 0.65
    ) -> List[Dict]:
        """Find similar work items across other Epics."""
        # Search for similar items in Qdrant
        similar_items = await self.qdrant_client.search_similar_work_items(
            item_embedding, limit=10, score_threshold=threshold
        )

        # Filter out items from the same Epic
        cross_epic_matches = []
        for item in similar_items:
            item_epic = item["work_item"].get("epic_key")
            if item_epic and item_epic != exclude_epic:
                cross_epic_matches.append(
                    {
                        "key": item["work_item"]["key"],
                        "epic_key": item_epic,
                        "score": item["score"],
                        "work_item": item["work_item"],
                    }
                )

        return cross_epic_matches

    def _generate_misplacement_reasoning(
        self, item_data: Dict, current_epic: str, best_match: Dict, current_coherence: float
    ) -> str:
        """Generate human-readable reasoning for why an item might be misplaced."""
        reasons = []

        coherence_diff = best_match["coherence"] - current_coherence

        if coherence_diff > 0.3:
            reasons.append("significantly higher semantic similarity")
        elif coherence_diff > 0.15:
            reasons.append("notably higher semantic similarity")
        else:
            reasons.append("higher semantic similarity")

        # Check for shared components or labels
        # This would require additional Epic metadata, simplified for now
        reasons.append(f"with Epic {best_match['epic_key']}")

        # Add confidence level
        if best_match["coherence"] > 0.8:
            confidence_text = "high confidence"
        elif best_match["coherence"] > 0.6:
            confidence_text = "moderate confidence"
        else:
            confidence_text = "low confidence"

        reasoning = f"Work item shows {', '.join(reasons)} ({confidence_text}). "
        reasoning += f"Current Epic coherence: {current_coherence:.2f}, "
        reasoning += f"Suggested Epic coherence: {best_match['coherence']:.2f}."

        return reasoning

    async def _analyze_misplacement_with_claude(
        self, work_item: WorkItem, current_epic: WorkItem, suggested_epic: WorkItem
    ) -> Dict:
        """Analyze work item misplacement using Claude."""
        if not self.claude_client:
            raise RuntimeError("Claude client not initialized")

        prompt = f"""Analyze whether this work item is better suited for a different Epic:

Work Item: {work_item.key}
Title: {work_item.summary}
Description: {work_item.description or 'No description'}

Current Epic: {current_epic.key}
Title: {current_epic.summary}
Description: {current_epic.description or 'No description'}

Suggested Epic: {suggested_epic.key}
Title: {suggested_epic.summary}
Description: {suggested_epic.description or 'No description'}

Based on content analysis, evaluate if this work item would be better aligned with the suggested Epic. Consider:
1. Thematic consistency
2. Functional relationships
3. Business value alignment
4. Technical dependencies

Respond in JSON format:
{{
    "reasoning": "detailed explanation of your analysis",
    "confidence": 0.0-1.0 (confidence in the recommendation)
}}"""

        try:
            response = await self.claude_client.analyze(
                prompt=prompt, analysis_type="misplacement_analysis"
            )

            import json

            analysis = json.loads(response.content)
            analysis["cost"] = response.cost
            return analysis

        except Exception as e:
            logger.warning(f"Failed to analyze misplacement for {work_item.key}: {str(e)}")
            return {
                "reasoning": "Unable to analyze misplacement due to analysis error",
                "confidence": 0.5,
                "cost": 0.0,
            }

    async def calculate_epic_coherence(self, work_item: WorkItem, epic_key: str) -> float:
        """Calculate how well a work item fits with an Epic's theme."""
        # This is a simplified version - in practice you'd want to get the Epic's
        # theme embedding and compare it with the work item's embedding

        # For now, return a placeholder implementation
        # In a real implementation, you'd:
        # 1. Get all work items for the Epic
        # 2. Calculate the Epic's theme embedding
        # 3. Compare work item embedding with Epic theme

        return 0.5  # Placeholder
