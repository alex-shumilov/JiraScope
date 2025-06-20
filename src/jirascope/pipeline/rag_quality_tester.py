"""Comprehensive RAG quality testing system for JiraScope."""

import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Set

from pydantic import BaseModel, Field

from ..clients.claude_client import ClaudeClient
from ..clients.lmstudio_client import LMStudioClient
from ..clients.qdrant_client import QdrantVectorClient
from ..clients.mcp_client import MCPClient
from ..core.config import Config
from ..models.work_item import WorkItem
from ..utils.logging import StructuredLogger

logger = StructuredLogger(__name__)


class RagTestQuery(BaseModel):
    """Test query for RAG quality testing."""
    
    id: str = Field(..., description="Unique identifier for this test query")
    query_text: str = Field(..., description="The query text")
    expected_work_items: List[str] = Field(default_factory=list, description="Known relevant work item keys")
    minimum_similarity: float = Field(0.5, description="Minimum similarity score threshold")
    category: str = Field(..., description="Category of test (functional, technical, business)")
    description: str = Field(..., description="Description of what this test is checking")


class RAGTestResult(BaseModel):
    """Result of a single RAG quality test."""
    
    query_id: str = Field(..., description="Test query ID")
    precision: float = Field(..., description="Precision score (0-1)")
    recall: float = Field(..., description="Recall score (0-1)")
    f1_score: float = Field(..., description="F1 score (0-1)")
    min_similarity_met: bool = Field(..., description="Whether minimum similarity was met")
    total_results: int = Field(..., description="Total search results")
    relevant_results: int = Field(..., description="Number of relevant results found")
    search_time: float = Field(0.0, description="Search time in seconds")


class RAGQualityReport(BaseModel):
    """Comprehensive RAG quality report."""
    
    test_results: List[RAGTestResult] = Field(default_factory=list)
    overall_f1_score: float = Field(..., description="Overall F1 score across all tests")
    passed_tests: int = Field(..., description="Number of tests that passed")
    total_tests: int = Field(..., description="Total number of tests")
    processing_time: float = Field(0.0, description="Total processing time in seconds")
    processing_cost: float = Field(0.0, description="Total processing cost")


class EmbeddingConsistencyResult(BaseModel):
    """Result of embedding consistency test for a single item."""
    
    work_item_key: str = Field(..., description="Work item key")
    average_similarity: float = Field(..., description="Average cosine similarity between embeddings")
    is_consistent: bool = Field(..., description="Whether embeddings are consistent")


class EmbeddingConsistencyReport(BaseModel):
    """Report on embedding consistency."""
    
    results: List[EmbeddingConsistencyResult] = Field(default_factory=list)
    overall_consistency: float = Field(..., description="Overall consistency score")
    consistent_items: int = Field(..., description="Number of items with consistent embeddings")
    total_items: int = Field(..., description="Total items tested")


class BatchSizeBenchmark(BaseModel):
    """Performance benchmark for a specific batch size."""
    
    batch_size: int = Field(..., description="Batch size tested")
    avg_processing_time: float = Field(..., description="Average processing time per batch")
    avg_memory_usage: float = Field(..., description="Average memory usage")
    items_per_second: float = Field(..., description="Items processed per second")
    memory_per_item: float = Field(..., description="Memory usage per item")


class PerformanceBenchmark(BaseModel):
    """Performance benchmark results for different configurations."""
    
    results: List[BatchSizeBenchmark] = Field(default_factory=list)
    optimal_batch_size: int = Field(..., description="Optimal batch size based on tests")
    recommendation: str = Field(..., description="Recommendation based on benchmarks")


class QualityTestSuite:
    """Collection of test queries for RAG quality testing."""
    
    def __init__(self):
        self.test_queries = [
            RagTestQuery(
                id="auth_functionality",
                query_text="user authentication and login functionality",
                expected_work_items=["PROJ-123", "PROJ-456"],
                minimum_similarity=0.7,
                category="functional",
                description="Should find authentication-related work items"
            ),
            RagTestQuery(
                id="database_migration",
                query_text="database schema changes and migrations",
                expected_work_items=["PROJ-789", "PROJ-012"],
                minimum_similarity=0.6,
                category="technical",
                description="Should identify database-related tasks"
            ),
            RagTestQuery(
                id="ui_improvements",
                query_text="user interface and user experience improvements",
                expected_work_items=["PROJ-345", "PROJ-678"],
                minimum_similarity=0.65,
                category="functional",
                description="Should find UI/UX related work"
            ),
            RagTestQuery(
                id="performance_optimization",
                query_text="application performance and speed optimization",
                expected_work_items=["PROJ-901", "PROJ-234"],
                minimum_similarity=0.7,
                category="technical",
                description="Should identify performance-related tasks"
            ),
            RagTestQuery(
                id="api_documentation",
                query_text="REST API documentation and endpoint specifications",
                expected_work_items=["PROJ-567", "PROJ-890"],
                minimum_similarity=0.6,
                category="business",
                description="Should find documentation tasks"
            )
        ]
    
    async def create_custom_test_from_epic(self, epic_key: str, jira_client: MCPClient, claude_client: ClaudeClient) -> RagTestQuery:
        """Generate test query from Epic content."""
        epic_data = await jira_client.get_issue(epic_key)
        
        # Use Claude to generate a semantic query from Epic description
        prompt = f"""
        Based on this Epic description, create a semantic search query that should find related work items:
        
        Epic: {epic_data.get('summary', '')}
        Description: {epic_data.get('description', '')}
        
        Generate a natural language query (2-8 words) that captures the main theme.
        """
        
        response = await claude_client.generate_text(prompt)
        query = response.strip()
        
        # Get all work items in the Epic as expected results
        epic_work_items = await jira_client.get_epic_issues(epic_key)
        work_item_keys = [item["key"] for item in epic_work_items]
        
        return RagTestQuery(
            id=f"epic_{epic_key.lower()}",
            query_text=query,
            expected_work_items=work_item_keys,
            minimum_similarity=0.5,  # Lower threshold for generated queries
            category="generated",
            description=f"Generated from Epic {epic_key}"
        )


class RAGQualityTester:
    """Comprehensive RAG quality testing system."""
    
    def __init__(self, config: Config, test_queries: Optional[List[RagTestQuery]] = None):
        self.config = config
        self.test_queries = test_queries or QualityTestSuite().test_queries
        self.baseline_results = {}
        self.performance_history = []
    
    async def run_quality_tests(self) -> RAGQualityReport:
        """Run predefined queries and validate results."""
        logger.info(f"Starting RAG quality tests with {len(self.test_queries)} test queries")
        start_time = time.time()
        test_results = []
        total_cost = 0.0
        
        try:
            async with LMStudioClient(self.config) as lm_client:
                async with QdrantVectorClient(self.config) as qdrant_client:
                    for test_query in self.test_queries:
                        try:
                            logger.debug(f"Testing query: {test_query.query_text}")
                            
                            # Measure search time
                            query_start_time = time.time()
                            
                            # Generate embedding for the query
                            embeddings = await lm_client.generate_embeddings([test_query.query_text])
                            if not embeddings:
                                raise ValueError(f"Failed to generate embeddings for query: {test_query.query_text}")
                            
                            # Search for similar work items
                            search_results = await qdrant_client.search_similar_work_items(
                                embeddings[0], 
                                limit=10, 
                                score_threshold=0.3
                            )
                            
                            query_time = time.time() - query_start_time
                            
                            # Calculate metrics
                            expected_set = set(test_query.expected_work_items)
                            found_set = set(result["work_item"]["key"] for result in search_results[:5])
                            
                            precision = len(expected_set & found_set) / len(found_set) if found_set else 0
                            recall = len(expected_set & found_set) / len(expected_set) if expected_set else 0
                            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                            
                            # Check minimum similarity threshold
                            min_similarity_met = all(
                                result["score"] >= test_query.minimum_similarity 
                                for result in search_results if result["work_item"]["key"] in expected_set
                            ) if expected_set & found_set else False
                            
                            # Add result
                            test_results.append(RAGTestResult(
                                query_id=test_query.id,
                                precision=precision,
                                recall=recall,
                                f1_score=f1_score,
                                min_similarity_met=min_similarity_met,
                                total_results=len(search_results),
                                relevant_results=len(expected_set & found_set),
                                search_time=query_time
                            ))
                            
                        except Exception as e:
                            logger.error(f"Error testing query {test_query.id}", error=str(e))
                            # Add a failed result
                            test_results.append(RAGTestResult(
                                query_id=test_query.id,
                                precision=0.0,
                                recall=0.0,
                                f1_score=0.0,
                                min_similarity_met=False,
                                total_results=0,
                                relevant_results=0,
                                search_time=0.0
                            ))
            
            # Calculate overall metrics
            if test_results:
                overall_f1 = sum(result.f1_score for result in test_results) / len(test_results)
                passed_tests = sum(1 for r in test_results if r.f1_score > 0.5)
            else:
                overall_f1 = 0.0
                passed_tests = 0
            
            processing_time = time.time() - start_time
            
            # Estimate cost: $0.01 per query
            cost_per_query = 0.01
            total_cost = len(self.test_queries) * cost_per_query
            
            report = RAGQualityReport(
                test_results=test_results,
                overall_f1_score=overall_f1,
                passed_tests=passed_tests,
                total_tests=len(self.test_queries),
                processing_time=processing_time,
                processing_cost=total_cost
            )
            
            # Log results
            logger.log_operation(
                "rag_quality_tests",
                processing_time,
                success=overall_f1 >= 0.7,
                overall_f1_score=overall_f1,
                passed_tests=passed_tests,
                total_tests=len(self.test_queries),
                cost=total_cost
            )
            
            # Store as baseline if first run or better than previous
            if not self.baseline_results or overall_f1 > self.baseline_results.get("overall_f1_score", 0):
                self.baseline_results = {
                    "overall_f1_score": overall_f1,
                    "passed_tests": passed_tests,
                    "total_tests": len(self.test_queries),
                    "timestamp": time.time()
                }
            
            # Track performance history
            self.performance_history.append({
                "timestamp": time.time(),
                "overall_f1_score": overall_f1,
                "passed_tests": passed_tests
            })
            
            return report
            
        except Exception as e:
            logger.error("Failed to run RAG quality tests", error=str(e))
            
            # Return minimal report on failure
            return RAGQualityReport(
                test_results=[],
                overall_f1_score=0.0,
                passed_tests=0,
                total_tests=len(self.test_queries),
                processing_time=time.time() - start_time,
                processing_cost=0.0
            )
    
    async def benchmark_embedding_performance(self) -> PerformanceBenchmark:
        """Test different batch sizes and configurations."""
        logger.info("Starting embedding performance benchmarking")
        
        # Define batch sizes to test
        batch_sizes = [8, 16, 32, 64]
        
        # Load test items
        sample_items = await self._get_test_work_items(100)
        benchmark_results = []
        
        async with LMStudioClient(self.config) as lm_client:
            for batch_size in batch_sizes:
                logger.info(f"Testing batch size: {batch_size}")
                batch_times = []
                memory_usages = []
                
                # Run multiple tests for statistical significance
                for _ in range(3):
                    start_time = time.time()
                    memory_start = 0  # We'd use psutil here in a real implementation
                    
                    # Process items in batches
                    for i in range(0, len(sample_items), batch_size):
                        batch = sample_items[i:i+batch_size]
                        texts = [item.summary + " " + (item.description or "") for item in batch]
                        await lm_client.generate_embeddings(texts)
                    
                    end_time = time.time()
                    memory_end = 0  # We'd use psutil here
                    
                    batch_times.append(end_time - start_time)
                    memory_usages.append(memory_end - memory_start)
                
                # Calculate averages
                avg_time = statistics.mean(batch_times) if batch_times else 0
                avg_memory = statistics.mean(memory_usages) if memory_usages else 0
                
                items_per_second = len(sample_items) / avg_time if avg_time > 0 else 0
                memory_per_item = avg_memory / len(sample_items) if len(sample_items) > 0 else 0
                
                benchmark_results.append(BatchSizeBenchmark(
                    batch_size=batch_size,
                    avg_processing_time=avg_time,
                    avg_memory_usage=avg_memory,
                    items_per_second=items_per_second,
                    memory_per_item=memory_per_item
                ))
        
        # Find optimal batch size based on processing speed
        optimal_batch_size = max(benchmark_results, key=lambda x: x.items_per_second)
        
        return PerformanceBenchmark(
            results=benchmark_results,
            optimal_batch_size=optimal_batch_size.batch_size,
            recommendation=f"Use batch size {optimal_batch_size.batch_size} for optimal performance"
        )
    
    async def validate_analysis_consistency(self) -> EmbeddingConsistencyReport:
        """Ensure analysis results are consistent across runs."""
        logger.info("Testing embedding consistency")
        
        sample_items = await self._get_test_work_items(20)
        consistency_results = []
        
        async with LMStudioClient(self.config) as lm_client:
            for item in sample_items:
                # Generate embedding multiple times
                embeddings = []
                text = item.summary + " " + (item.description or "")
                
                for _ in range(3):
                    embedding = await lm_client.generate_embeddings([text])
                    if embedding and embedding[0]:
                        embeddings.append(embedding[0])
                
                # Calculate pairwise similarities if we have enough embeddings
                if len(embeddings) >= 2:
                    similarities = []
                    for i in range(len(embeddings)):
                        for j in range(i+1, len(embeddings)):
                            sim = self._cosine_similarity(embeddings[i], embeddings[j])
                            similarities.append(sim)
                    
                    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
                    
                    consistency_results.append(EmbeddingConsistencyResult(
                        work_item_key=item.key,
                        average_similarity=avg_similarity,
                        is_consistent=avg_similarity > 0.99  # Very high threshold for consistency
                    ))
        
        # Calculate overall consistency
        if consistency_results:
            overall_consistency = sum(r.average_similarity for r in consistency_results) / len(consistency_results)
            consistent_items = sum(1 for r in consistency_results if r.is_consistent)
        else:
            overall_consistency = 0.0
            consistent_items = 0
        
        return EmbeddingConsistencyReport(
            results=consistency_results,
            overall_consistency=overall_consistency,
            consistent_items=consistent_items,
            total_items=len(consistency_results)
        )
    
    async def get_regression_report(self) -> Dict[str, Any]:
        """Compare current performance with baseline."""
        current_report = await self.run_quality_tests()
        
        if not self.baseline_results:
            return {
                "has_regression": False,
                "message": "No baseline results available for comparison",
                "current_f1": current_report.overall_f1_score,
                "baseline_f1": 0.0,
                "change": 0.0
            }
        
        f1_change = current_report.overall_f1_score - self.baseline_results["overall_f1_score"]
        has_regression = f1_change < -0.05  # 5% drop is considered regression
        
        return {
            "has_regression": has_regression,
            "message": "Quality regression detected" if has_regression else "No quality regression",
            "current_f1": current_report.overall_f1_score,
            "baseline_f1": self.baseline_results["overall_f1_score"],
            "change": f1_change,
            "current_passed_tests": current_report.passed_tests,
            "baseline_passed_tests": self.baseline_results["passed_tests"]
        }
    
    async def _get_test_work_items(self, sample_size: int = 100) -> List[WorkItem]:
        """Get sample work items for testing."""
        # This would normally load from database or API
        # Here we'll create some sample items
        sample_items = []
        
        for i in range(1, sample_size + 1):
            sample_items.append(WorkItem(
                key=f"SAMPLE-{i}",
                summary=f"Sample work item {i}",
                description=f"This is a sample work item for testing consistency with ID {i}",
                issue_type="Task",
                status="Open",
                created=time.time(),
                updated=time.time(),
                reporter="system"
            ))
        
        return sample_items
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        # Simple implementation - a real system would use numpy
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        
        if mag1 * mag2 == 0:
            return 0.0
            
        return dot_product / (mag1 * mag2)


class ComprehensiveQualityTester:
    """Runs comprehensive quality testing across all aspects of the system."""
    
    def __init__(self, config: Config):
        self.config = config
        self.rag_tester = RAGQualityTester(config)
    
    async def run_full_quality_assessment(self) -> Dict[str, Any]:
        """Run comprehensive quality assessment across all areas."""
        start_time = time.time()
        logger.info("Starting comprehensive quality assessment")
        
        results = {}
        
        # 1. RAG Query Quality Tests
        logger.info("Running RAG quality tests")
        results['rag_quality'] = await self.rag_tester.run_quality_tests()
        
        # 2. Embedding Consistency Tests
        logger.info("Running embedding consistency tests")
        results['embedding_consistency'] = await self.rag_tester.validate_analysis_consistency()
        
        # 3. Performance Benchmarking
        logger.info("Running performance benchmarking")
        results['performance_benchmark'] = await self.rag_tester.benchmark_embedding_performance()
        
        # 4. Regression Testing
        logger.info("Checking for quality regression")
        results['regression_analysis'] = await self.rag_tester.get_regression_report()
        
        processing_time = time.time() - start_time
        logger.log_operation(
            "comprehensive_quality_assessment",
            processing_time,
            success=True,
            rag_f1_score=results['rag_quality'].overall_f1_score,
            embedding_consistency=results['embedding_consistency'].overall_consistency
        )
        
        return {
            "results": results,
            "processing_time": processing_time,
            "timestamp": time.time(),
            "overall_health": self._calculate_overall_health(results)
        }
    
    def _calculate_overall_health(self, results: Dict[str, Any]) -> float:
        """Calculate overall system health score based on test results."""
        # Define weights for different metrics
        weights = {
            "rag_quality": 0.5,
            "embedding_consistency": 0.3,
            "regression": 0.2
        }
        
        # Calculate weighted score
        rag_score = results['rag_quality'].overall_f1_score
        consistency_score = results['embedding_consistency'].overall_consistency
        regression_factor = 1.0 if not results['regression_analysis']["has_regression"] else 0.7
        
        weighted_score = (
            weights["rag_quality"] * rag_score + 
            weights["embedding_consistency"] * consistency_score
        ) * regression_factor
        
        return min(1.0, max(0.0, weighted_score))