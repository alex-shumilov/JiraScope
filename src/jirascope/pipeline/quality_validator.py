"""Embedding quality validation with predefined test queries."""

import time
from typing import Any

from ..clients.lmstudio_client import LMStudioClient
from ..clients.qdrant_client import QdrantVectorClient
from ..core.config import Config
from ..models import QualityReport
from ..utils.logging import StructuredLogger

logger = StructuredLogger(__name__)


class EmbeddingQualityValidator:
    """Validate embedding quality with test queries."""

    DEFAULT_TEST_QUERIES = [
        "authentication and login functionality",
        "database migration and schema changes",
        "user interface improvements",
        "API endpoint documentation",
        "performance optimization tasks",
        "bug fixes and error handling",
        "integration testing",
        "security vulnerabilities",
        "mobile application features",
        "reporting and analytics",
    ]

    def __init__(self, config: Config, test_queries: list[str] | None = None):
        self.config = config
        self.test_queries = test_queries or self.DEFAULT_TEST_QUERIES

    async def validate_embedding_quality(self) -> QualityReport:
        """Run test queries to ensure embeddings are working correctly."""
        logger.info(
            f"Starting embedding quality validation with {len(self.test_queries)} test queries"
        )
        start_time = time.time()

        results = []
        passed_tests = 0

        try:
            async with LMStudioClient(self.config) as lm_client:
                async with QdrantVectorClient(self.config) as qdrant_client:

                    for i, query in enumerate(self.test_queries):
                        try:
                            result = await self._test_single_query(query, lm_client, qdrant_client)
                            results.append(result)

                            if result["passed"]:
                                passed_tests += 1

                            logger.debug(
                                f"Test query {i+1}/{len(self.test_queries)}: {'PASS' if result['passed'] else 'FAIL'}"
                            )

                        except Exception as e:
                            logger.error(f"Failed test query '{query}'", error=str(e))
                            results.append(
                                {
                                    "query": query,
                                    "results_count": 0,
                                    "avg_similarity": 0.0,
                                    "passed": False,
                                    "error": str(e),
                                }
                            )

            # Calculate overall score
            overall_score = (passed_tests / len(self.test_queries)) * 100.0

            # Generate recommendations
            recommendations = self._generate_recommendations(results, overall_score)

            report = QualityReport(
                total_tests=len(self.test_queries),
                passed_tests=passed_tests,
                overall_score=overall_score,
                results=results,
                recommendations=recommendations,
            )

            processing_time = time.time() - start_time
            logger.log_operation(
                "validate_embedding_quality",
                processing_time,
                success=overall_score >= 70.0,  # Consider 70% pass rate as success
                overall_score=overall_score,
                passed_tests=passed_tests,
                total_tests=len(self.test_queries),
            )

            return report

        except Exception as e:
            logger.error("Failed to validate embedding quality", error=str(e))
            # Return minimal report on failure
            return QualityReport(
                total_tests=len(self.test_queries),
                passed_tests=0,
                overall_score=0.0,
                results=[],
                recommendations=["Failed to run quality validation - check service connectivity"],
            )

    async def _test_single_query(
        self, query: str, lm_client: LMStudioClient, qdrant_client: QdrantVectorClient
    ) -> dict[str, Any]:
        """Test a single query and return results."""

        # Generate embedding for the query
        query_embeddings = await lm_client.generate_embeddings([query])
        if not query_embeddings:
            raise ValueError(f"Failed to generate embedding for query: {query}")

        query_embedding = query_embeddings[0]

        # Search for similar work items
        search_results = await qdrant_client.search_similar_work_items(
            query_embedding, limit=5, score_threshold=0.3  # Lower threshold for testing
        )

        # Calculate metrics
        results_count = len(search_results)
        avg_similarity = 0.0

        if search_results:
            avg_similarity = sum(result["score"] for result in search_results) / len(search_results)

        # Define pass criteria
        # - At least 1 result found
        # - Average similarity > 0.3 (reasonable threshold for semantic search)
        passed = results_count > 0 and avg_similarity > 0.3

        return {
            "query": query,
            "results_count": results_count,
            "avg_similarity": avg_similarity,
            "passed": passed,
            "top_results": [
                {
                    "key": result["work_item"]["key"],
                    "summary": result["work_item"]["summary"],
                    "score": result["score"],
                }
                for result in search_results[:3]  # Top 3 results
            ],
        }

    def _generate_recommendations(
        self, results: list[dict[str, Any]], overall_score: float
    ) -> list[str]:
        """Generate improvement recommendations based on test results."""
        recommendations = []

        if overall_score < 50.0:
            recommendations.append(
                "Overall quality is poor - consider re-generating embeddings with different parameters"
            )
        elif overall_score < 70.0:
            recommendations.append("Quality is moderate - some queries may need refinement")

        # Check for queries with no results
        no_result_queries = [r for r in results if r["results_count"] == 0]
        if no_result_queries:
            recommendations.append(
                f"{len(no_result_queries)} queries returned no results - consider adding more diverse work items"
            )

        # Check for low similarity scores
        low_similarity_queries = [r for r in results if r.get("avg_similarity", 0) < 0.3]
        if low_similarity_queries:
            recommendations.append(
                f"{len(low_similarity_queries)} queries had low similarity scores - embedding model may need tuning"
            )

        # Check if we have enough data
        total_results = sum(r["results_count"] for r in results)
        if total_results < len(self.test_queries) * 2:  # Less than 2 results per query on average
            recommendations.append(
                "Limited search results suggest sparse data - consider adding more work items"
            )

        if not recommendations:
            recommendations.append("Embedding quality looks good - no specific improvements needed")

        return recommendations

    async def run_performance_test(self, num_queries: int = 10) -> dict[str, float]:
        """Run performance test to measure search speed."""
        logger.info(f"Running performance test with {num_queries} queries")

        test_queries = self.test_queries[:num_queries]
        start_time = time.time()

        search_times = []

        try:
            async with LMStudioClient(self.config) as lm_client:
                async with QdrantVectorClient(self.config) as qdrant_client:

                    for query in test_queries:
                        query_start = time.time()

                        # Generate embedding
                        embedding_start = time.time()
                        query_embeddings = await lm_client.generate_embeddings([query])
                        embedding_time = time.time() - embedding_start

                        # Search
                        search_start = time.time()
                        await qdrant_client.search_similar_work_items(query_embeddings[0], limit=10)
                        search_time = time.time() - search_start

                        total_query_time = time.time() - query_start
                        search_times.append(
                            {
                                "total": total_query_time,
                                "embedding": embedding_time,
                                "search": search_time,
                            }
                        )

            # Calculate performance metrics
            total_time = time.time() - start_time
            avg_total_time = sum(t["total"] for t in search_times) / len(search_times)
            avg_embedding_time = sum(t["embedding"] for t in search_times) / len(search_times)
            avg_search_time = sum(t["search"] for t in search_times) / len(search_times)

            performance_metrics = {
                "total_test_time": total_time,
                "queries_per_second": len(test_queries) / total_time,
                "avg_query_time": avg_total_time,
                "avg_embedding_time": avg_embedding_time,
                "avg_search_time": avg_search_time,
            }

            logger.log_operation(
                "performance_test",
                total_time,
                success=True,
                queries_tested=len(test_queries),
                **performance_metrics,
            )

            return performance_metrics

        except Exception as e:
            logger.error("Performance test failed", error=str(e))
            return {"error": str(e)}
