"""Main RAG pipeline orchestrator for JiraScope."""

from ..clients.lmstudio_client import LMStudioClient
from ..clients.qdrant_client import QdrantVectorClient
from .context_assembler import ContextAssembler
from .query_processor import JiraQueryProcessor
from .retrieval_engine import ContextualRetriever, RetrievalResult


class JiraRAGPipeline:
    """Main RAG pipeline for intelligent Jira query processing."""

    def __init__(
        self,
        qdrant_client: QdrantVectorClient,
        embedding_client: LMStudioClient,
        max_context_tokens: int = 8000,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            qdrant_client: Vector database client
            embedding_client: Client for generating embeddings
            max_context_tokens: Maximum tokens for context assembly
        """
        self.query_processor = JiraQueryProcessor()
        self.retriever = ContextualRetriever(qdrant_client, embedding_client)
        self.context_assembler = ContextAssembler(max_tokens=max_context_tokens)

    async def process_query(self, user_query: str, include_hierarchy: bool = True) -> dict:
        """
        Process a natural language query end-to-end.

        Args:
            user_query: Natural language query from user
            include_hierarchy: Whether to include hierarchical context

        Returns:
            Dictionary with query results and metadata
        """
        try:
            # Step 1: Query Understanding
            query_plan = self.query_processor.analyze_query(user_query)

            # Step 2: Semantic Retrieval
            retrieval_results = await self.retriever.semantic_search(
                query=query_plan.expanded_query, filters=query_plan.filters, limit=15
            )

            # Step 3: Hierarchical Context (if requested)
            hierarchical_context = []
            if include_hierarchy and retrieval_results:
                # Get hierarchical context for top results
                for result in retrieval_results[:3]:  # Top 3 results
                    if result.jira_key:
                        context_tree = await self.retriever.hierarchical_retrieval(result.jira_key)
                        if context_tree.root_item:
                            hierarchical_context.append(context_tree)

            # Step 4: Context Assembly
            assembled_context = self.context_assembler.assemble_context(
                retrieval_results=retrieval_results,
                query_plan=query_plan,
                hierarchical_context=hierarchical_context,
            )

            # Step 5: Format Response
            return {
                "query": user_query,
                "intent": query_plan.intent,
                "expected_output": query_plan.expected_output,
                "results_count": len(retrieval_results),
                "context_summary": assembled_context.summary.to_text(),
                "formatted_context": assembled_context.formatted_text,
                "token_count": assembled_context.token_count,
                "jira_keys": assembled_context.jira_keys,
                "filters_applied": query_plan.filters.to_qdrant_filter(),
                "success": True,
            }

        except Exception as e:
            return {
                "query": user_query,
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    async def search_by_epic(self, epic_key: str, query: str = "") -> dict:
        """
        Search within a specific Epic hierarchy.

        Args:
            epic_key: Epic key to search within
            query: Optional search query

        Returns:
            Dictionary with Epic search results
        """
        try:
            # Get Epic context
            epic_context = await self.retriever.hierarchical_retrieval(epic_key)

            # If query provided, do semantic search within Epic
            if query:
                query_plan = self.query_processor.analyze_query(query)
                query_embedding = await self.retriever.embedder.generate_embeddings([query])

                if query_embedding:
                    results = await self.retriever.search_by_epic(
                        query_embedding=query_embedding[0], epic_key=epic_key, limit=10
                    )
                else:
                    results = []
            else:
                query_plan = self.query_processor.analyze_query(f"Epic {epic_key}")
                results = []

            # Assemble context
            assembled_context = self.context_assembler.assemble_context(
                retrieval_results=results,
                query_plan=query_plan,
                hierarchical_context=[epic_context] if epic_context.root_item else [],
            )

            return {
                "epic_key": epic_key,
                "query": query,
                "epic_context": epic_context.root_item,
                "child_count": len(epic_context.child_items or []),
                "results_count": len(results),
                "formatted_context": assembled_context.formatted_text,
                "success": True,
            }

        except Exception as e:
            return {"epic_key": epic_key, "query": query, "success": False, "error": str(e)}

    async def analyze_technical_debt(self, team: str | None = None) -> dict:
        """
        Specialized analysis for technical debt items.

        Args:
            team: Optional team filter

        Returns:
            Technical debt analysis results
        """
        try:
            # Construct technical debt query
            base_query = "technical debt refactoring cleanup code quality maintenance"
            if team:
                base_query += f" team:{team}"

            query_plan = self.query_processor.analyze_query(base_query)

            # Search for technical debt items
            results = await self.retriever.semantic_search(
                query=query_plan.expanded_query, filters=query_plan.filters, limit=20
            )

            # Assemble context
            assembled_context = self.context_assembler.assemble_context(
                retrieval_results=results, query_plan=query_plan
            )

            # Analyze patterns
            debt_analysis = self._analyze_debt_patterns(results)

            return {
                "query": base_query,
                "team_filter": team,
                "debt_items_found": len(results),
                "debt_analysis": debt_analysis,
                "formatted_context": assembled_context.formatted_text,
                "recommendations": self._generate_debt_recommendations(debt_analysis),
                "success": True,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _analyze_debt_patterns(self, results: list[RetrievalResult]) -> dict:
        """Analyze patterns in technical debt items."""
        patterns = {
            "by_component": {},
            "by_priority": {},
            "by_status": {},
            "total_items": len(results),
        }

        for result in results:
            content = result.content

            # Group by component
            components = content.get("components", [])
            for component in components:
                patterns["by_component"][component] = patterns["by_component"].get(component, 0) + 1

            # Group by priority
            priority = content.get("priority", "Unknown")
            patterns["by_priority"][priority] = patterns["by_priority"].get(priority, 0) + 1

            # Group by status
            status = content.get("status", "Unknown")
            patterns["by_status"][status] = patterns["by_status"].get(status, 0) + 1

        return patterns

    def _generate_debt_recommendations(self, debt_analysis: dict) -> list[str]:
        """Generate recommendations based on debt analysis."""
        recommendations = []

        # Component-based recommendations
        if debt_analysis["by_component"]:
            top_component = max(debt_analysis["by_component"].items(), key=lambda x: x[1])
            recommendations.append(
                f"Focus on {top_component[0]} component with {top_component[1]} debt items"
            )

        # Priority-based recommendations
        high_priority_count = debt_analysis["by_priority"].get("High", 0)
        if high_priority_count > 0:
            recommendations.append(f"Address {high_priority_count} high-priority debt items first")

        # Status-based recommendations
        open_count = debt_analysis["by_status"].get("Open", 0)
        if open_count > 0:
            recommendations.append(f"Prioritize {open_count} open debt items for sprint planning")

        return recommendations
