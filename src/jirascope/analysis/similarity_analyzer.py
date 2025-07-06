"""Similarity analysis with multi-level duplicate detection."""

import time

from qdrant_client.http import models

from ..clients.lmstudio_client import LMStudioClient
from ..clients.qdrant_client import QdrantVectorClient
from ..core.config import Config
from ..models import CoherenceAnalysis, DuplicateCandidate, DuplicateReport, WorkItem
from ..utils.logging import StructuredLogger

logger = StructuredLogger(__name__)


class MultiLevelSimilarityDetector:
    """Multi-level similarity detection with confidence scoring."""

    def __init__(self, qdrant_client: QdrantVectorClient, lm_client: LMStudioClient):
        self.qdrant = qdrant_client
        self.lm_client = lm_client
        self.similarity_thresholds = {
            "exact": 0.95,  # Almost identical
            "high": 0.85,  # Very similar, likely duplicates
            "medium": 0.70,  # Similar, worth reviewing
            "low": 0.55,  # Somewhat related
        }

    def classify_similarity(self, similarity_score: float) -> str | None:
        """Classify similarity score into confidence levels."""
        for level, threshold in sorted(
            self.similarity_thresholds.items(), key=lambda x: x[1], reverse=True
        ):
            if similarity_score >= threshold:
                return level
        return None

    def _get_confidence_level(self, similarity_score: float) -> str | None:
        """Get confidence level based on similarity score."""
        return self.classify_similarity(similarity_score)

    def _calculate_review_priority(self, original: WorkItem, similar_item: dict) -> int:
        """Calculate review priority (1-5, higher is more urgent)."""
        priority = 1

        # Same issue type increases priority
        if original.issue_type == similar_item.get("issue_type"):
            priority += 1

        # Same status increases priority
        if original.status == similar_item.get("status"):
            priority += 1

        # Same Epic increases priority
        if original.epic_key and original.epic_key == similar_item.get("epic_key"):
            priority += 1

        # Recent items get higher priority
        try:
            similar_created = similar_item.get("created")
            if similar_created and (original.created - similar_created).days < 30:
                priority += 1
        except (TypeError, AttributeError):
            pass

        return min(priority, 5)

    def generate_suggested_action(self, confidence_level: str, similarity_score: float) -> str:
        """Generate suggested action based on confidence level and score."""
        if confidence_level == "exact":
            return "Immediate review - likely exact duplicate"
        if confidence_level == "high":
            return "High priority review - consider merging"
        if confidence_level == "medium":
            return "Investigate potential relationship - compare for related work"
        return "Low priority - monitor for patterns"

    def _determine_suggested_action(self, candidate: DuplicateCandidate) -> str:
        """Determine suggested action based on candidate properties."""
        return self.generate_suggested_action(
            candidate.confidence_level, candidate.similarity_score
        )

    def _analyze_similarity_reasons(
        self, original: WorkItem, similar_item: dict, score: float
    ) -> list[str]:
        """Analyze why items are similar."""
        reasons = []

        # Check title similarity
        if any(
            word in similar_item.get("summary", "").lower()
            for word in original.summary.lower().split()
            if len(word) > 3
        ):
            reasons.append("Similar titles/keywords")

        # Check same Epic
        if original.epic_key == similar_item.get("epic_key"):
            reasons.append("Same Epic")

        # Check same components
        original_components = set(original.components)
        similar_components = set(similar_item.get("components", []))
        if original_components & similar_components:
            reasons.append("Shared components")

        # Check same labels
        original_labels = set(original.labels)
        similar_labels = set(similar_item.get("labels", []))
        if original_labels & similar_labels:
            reasons.append("Shared labels")

        # High semantic similarity
        if score > 0.8:
            reasons.append("High semantic similarity")

        return reasons or ["Semantic content similarity"]

    async def find_duplicates_by_level(
        self, work_items: list[WorkItem], min_threshold: float = 0.55
    ) -> dict[str, list[DuplicateCandidate]]:
        """Find duplicates at different confidence levels."""
        logger.info(f"Starting duplicate detection for {len(work_items)} work items")
        start_time = time.time()

        duplicates_by_level = {level: [] for level in self.similarity_thresholds}
        processed_pairs = set()  # Avoid duplicate comparisons

        try:
            for i, item in enumerate(work_items):
                # Generate embedding for this item if not available
                if not hasattr(item, "embedding") or not item.embedding:
                    # Get embedding from LMStudio
                    from ..pipeline.embedding_processor import EmbeddingProcessor

                    text = EmbeddingProcessor.prepare_embedding_text(item)
                    embeddings = await self.lm_client.generate_embeddings([text])
                    item_embedding = embeddings[0] if embeddings else None
                else:
                    item_embedding = item.embedding

                if not item_embedding:
                    logger.warning(f"Could not get embedding for {item.key}")
                    continue

                # Search for similar items in Qdrant
                similar_items = await self.qdrant.search_similar_work_items(
                    item_embedding, limit=20, score_threshold=min_threshold
                )

                for similar in similar_items:
                    similar_key = similar["work_item"]["key"]

                    # Skip self-matches and already processed pairs
                    if similar_key == item.key:
                        continue

                    pair_key = tuple(sorted([item.key, similar_key]))
                    if pair_key in processed_pairs:
                        continue
                    processed_pairs.add(pair_key)

                    confidence_level = self._get_confidence_level(similar["score"])
                    if not confidence_level:
                        continue

                    # Create duplicate candidate
                    candidate = DuplicateCandidate(
                        original_key=item.key,
                        duplicate_key=similar_key,
                        similarity_score=similar["score"],
                        confidence_level=confidence_level,
                        review_priority=self._calculate_review_priority(item, similar["work_item"]),
                        suggested_action="",  # Will be set below
                        similarity_reasons=self._analyze_similarity_reasons(
                            item, similar["work_item"], similar["score"]
                        ),
                    )

                    candidate.suggested_action = self._determine_suggested_action(candidate)
                    duplicates_by_level[confidence_level].append(candidate)

                if (i + 1) % 10 == 0:
                    logger.debug(
                        f"Processed {i + 1}/{len(work_items)} items for duplicate detection"
                    )

        except Exception as e:
            logger.error("Error during duplicate detection", error=str(e))
            raise

        processing_time = time.time() - start_time
        total_candidates = sum(len(candidates) for candidates in duplicates_by_level.values())

        logger.log_operation(
            "find_duplicates_by_level",
            processing_time,
            success=True,
            items_processed=len(work_items),
            candidates_found=total_candidates,
        )

        return duplicates_by_level


class SimilarityAnalyzer:
    """Main similarity analyzer with duplicate detection and coherence analysis."""

    def __init__(self, config: Config):
        self.config = config
        self.detector: MultiLevelSimilarityDetector | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.qdrant_client = QdrantVectorClient(self.config)
        self.lm_client = LMStudioClient(self.config)

        await self.qdrant_client.__aenter__()
        await self.lm_client.__aenter__()

        self.detector = MultiLevelSimilarityDetector(self.qdrant_client, self.lm_client)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.qdrant_client:
            await self.qdrant_client.__aexit__(exc_type, exc_val, exc_tb)
        if self.lm_client:
            await self.lm_client.__aexit__(exc_type, exc_val, exc_tb)

    async def find_potential_duplicates(
        self, work_items: list[WorkItem], threshold: float = 0.70
    ) -> DuplicateReport:
        """Find duplicates with confidence scoring and human review flags."""
        logger.info(f"Starting comprehensive duplicate analysis for {len(work_items)} items")
        start_time = time.time()

        if not self.detector:
            raise RuntimeError("SimilarityAnalyzer not initialized. Use async context manager.")

        try:
            # Find duplicates by confidence level
            candidates_by_level = await self.detector.find_duplicates_by_level(
                work_items, min_threshold=threshold
            )

            total_candidates = sum(len(candidates) for candidates in candidates_by_level.values())
            processing_time = time.time() - start_time

            # Estimate cost (rough approximation based on API calls)
            estimated_cost = len(work_items) * 0.0001  # Vector search cost

            if logger.cost_tracker:
                logger.log_cost(
                    "similarity",
                    "duplicate_detection",
                    estimated_cost,
                    {"items_analyzed": len(work_items), "candidates_found": total_candidates},
                )

            report = DuplicateReport(
                total_candidates=total_candidates,
                candidates_by_level=candidates_by_level,
                processing_cost=estimated_cost,
                items_analyzed=len(work_items),
            )

            logger.log_operation(
                "find_potential_duplicates",
                processing_time,
                success=True,
                total_candidates=total_candidates,
                exact_matches=len(candidates_by_level.get("exact", [])),
                high_confidence=len(candidates_by_level.get("high", [])),
            )

            return report

        except Exception as e:
            logger.error("Failed to find potential duplicates", error=str(e))
            raise

    async def hierarchical_coherence(self, epic_key: str) -> CoherenceAnalysis:
        """Analyze if Epic's children are semantically coherent."""
        logger.info(f"Analyzing hierarchical coherence for Epic {epic_key}")
        start_time = time.time()

        try:
            # Get all work items for this Epic from Qdrant
            _ = []  # Placeholder for epic_items - for future enhancement

            # Search for work items belonging to this Epic
            # Note: This assumes epic_key is stored in payload during embedding storage
            scroll_result = self.qdrant_client.client.scroll(
                collection_name="jirascope_work_items",
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="epic_key", match=models.MatchValue(value=epic_key)
                        )
                    ]
                ),
                limit=1000,
            )

            if not scroll_result[0]:  # No items found
                return CoherenceAnalysis(
                    epic_key=epic_key,
                    coherence_score=0.0,
                    work_items_count=0,
                    theme_consistency=0.0,
                )

            # Extract work item data and embeddings
            work_item_data = []
            embeddings = []

            for point in scroll_result[0]:
                work_item_data.append(point.payload)
                embeddings.append(point.vector)

            if len(embeddings) < 2:
                return CoherenceAnalysis(
                    epic_key=epic_key,
                    coherence_score=1.0 if len(embeddings) == 1 else 0.0,
                    work_items_count=len(embeddings),
                    theme_consistency=1.0 if len(embeddings) == 1 else 0.0,
                )

            # Calculate pairwise similarities
            similarities = []
            outlier_candidates = []

            for i, emb1 in enumerate(embeddings):
                item_similarities = []

                for j, emb2 in enumerate(embeddings):
                    if i != j:
                        similarity = self.lm_client.calculate_similarity(emb1, emb2)
                        similarities.append(similarity)
                        item_similarities.append(similarity)

                # Check if this item is an outlier (low average similarity to others)
                avg_similarity = sum(item_similarities) / len(item_similarities)
                if avg_similarity < 0.5:  # Threshold for outlier detection
                    outlier_candidates.append(
                        {"key": work_item_data[i]["key"], "avg_similarity": avg_similarity}
                    )

            # Calculate overall coherence metrics
            overall_coherence = sum(similarities) / len(similarities) if similarities else 0.0
            theme_consistency = (overall_coherence + 0.2) / 1.2  # Normalize and adjust

            # Identify clear outliers (bottom 20% or those below 0.4 avg similarity)
            outlier_threshold = min(
                0.4,
                (
                    sorted([oc["avg_similarity"] for oc in outlier_candidates])[
                        : len(outlier_candidates) // 5
                    ][-1]
                    if outlier_candidates
                    else 0.4
                ),
            )
            outlier_items = [
                oc["key"] for oc in outlier_candidates if oc["avg_similarity"] < outlier_threshold
            ]

            # Generate recommendations
            recommendations = []
            if overall_coherence < 0.6:
                recommendations.append(
                    "Epic has low coherence - consider splitting or reorganizing"
                )
            if len(outlier_items) > 0:
                recommendations.append(
                    f"Review {len(outlier_items)} outlier items for correct Epic assignment"
                )
            if theme_consistency < 0.5:
                recommendations.append("Epic theme is inconsistent - review work item alignment")
            if not recommendations:
                recommendations.append("Epic shows good coherence and theme consistency")

            processing_time = time.time() - start_time
            estimated_cost = len(embeddings) * 0.0001  # noqa: F841

            logger.log_operation(
                "hierarchical_coherence",
                processing_time,
                success=True,
                epic_key=epic_key,
                work_items_count=len(work_item_data),
                coherence_score=overall_coherence,
                outliers_found=len(outlier_items),
            )

            return CoherenceAnalysis(
                epic_key=epic_key,
                coherence_score=overall_coherence,
                work_items_count=len(work_item_data),
                outlier_items=outlier_items,
                theme_consistency=theme_consistency,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Failed to analyze coherence for Epic {epic_key}", error=str(e))
            raise
