"""Comprehensive RAG Pipeline Testing - Testing actual business logic."""

from unittest.mock import AsyncMock, Mock

import pytest

from jirascope.rag.context_assembler import ContextAssembler
from jirascope.rag.pipeline import JiraRAGPipeline
from jirascope.rag.query_processor import ExpandedQuery, FilterSet, JiraQueryProcessor, QueryPlan
from jirascope.rag.retrieval_engine import ContextTree, RetrievalResult


class TestRAGPipelineBusinessLogic:
    """Test actual RAG pipeline business logic - not just mocks."""

    def test_query_processor_analysis_logic(self):
        """Test actual query analysis business logic."""
        processor = JiraQueryProcessor()

        # Test complex query analysis
        complex_query = "Show me blocked high priority frontend bugs from last sprint with authentication issues"
        plan = processor.analyze_query(complex_query)

        # Test actual business logic
        assert plan.original_query == complex_query
        assert plan.intent == "search"
        assert plan.expected_output == "list"

        # Test filter extraction logic
        filters = plan.filters
        assert filters.item_types and "Bug" in filters.item_types
        assert filters.priorities and "High" in filters.priorities
        assert filters.statuses and "Blocked" in filters.statuses
        assert filters.components and any("frontend" in comp.lower() for comp in filters.components)

        # Test query expansion logic
        expanded = plan.expanded_query
        assert len(expanded.expanded_terms) > 0
        assert any("waiting" in term.lower() for term in expanded.expanded_terms)  # Blocked synonym

        # Test Jira-specific synonyms
        assert "defect" in expanded.expanded_terms or "issue" in expanded.expanded_terms

        # Test priority boosts
        assert plan.priority_boost is not None
        assert plan.priority_boost.get("status_boost", 0) > 1.0  # Blocked should get boost

    def test_context_assembler_ranking_logic(self):
        """Test actual relevance ranking business logic."""
        assembler = ContextAssembler(max_tokens=1000)

        # Create test results with different relevance factors
        results = [
            RetrievalResult(
                score=0.8,
                content={
                    "key": "LOW-RELEVANCE",
                    "item_type": "Task",
                    "priority": "Low",
                    "status": "Open",
                    "has_children": False,
                    "dependency_count": 0,
                },
                chunk_id="chunk1",
                source_key="LOW-RELEVANCE",
                item_type="Task",
            ),
            RetrievalResult(
                score=0.7,  # Lower base score
                content={
                    "key": "HIGH-RELEVANCE",
                    "item_type": "Bug",
                    "priority": "High",
                    "status": "Blocked",
                    "has_children": True,
                    "dependency_count": 3,
                },
                chunk_id="chunk2",
                source_key="HIGH-RELEVANCE",
                item_type="Bug",
            ),
        ]

        # Create query plan that should boost second item
        query_plan = QueryPlan(
            original_query="show blocked high priority bugs",
            expanded_query=ExpandedQuery(
                original_query="show blocked high priority bugs",
                expanded_terms=[],
                semantic_variants=[],
                jira_synonyms={},
            ),
            filters=FilterSet(
                item_types=["Bug"],
                priorities=["High"],
                statuses=["Blocked"],
            ),
            intent="analysis",  # Should boost items with dependencies
            expected_output="list",
            priority_boost={"status_boost": 2.0, "priority_boost": 1.5},
        )

        # Test ranking business logic
        ranked = assembler.rank_by_relevance(results, query_plan)

        # Second item should be ranked higher despite lower base score
        assert ranked[0].jira_key == "HIGH-RELEVANCE"
        assert ranked[1].jira_key == "LOW-RELEVANCE"

        # Verify the business logic applied boosts correctly
        assert ranked[0].score > ranked[1].score
        assert ranked[0].score > 0.7  # Should be boosted from original 0.7

    def test_context_summary_business_logic(self):
        """Test context summary generation business logic."""
        assembler = ContextAssembler()

        # Create complex result set
        results = [
            RetrievalResult(
                score=0.9,
                content={
                    "key": "PROJ-1",
                    "item_type": "Epic",
                    "priority": "High",
                    "status": "In Progress",
                    "epic_key": "PROJ-1",
                    "team": "Frontend",
                    "components": ["ui", "auth"],
                },
                chunk_id="chunk1",
                source_key="PROJ-1",
                item_type="Epic",
            ),
            RetrievalResult(
                score=0.8,
                content={
                    "key": "PROJ-2",
                    "item_type": "Story",
                    "priority": "Medium",
                    "status": "Done",
                    "epic_key": "PROJ-1",
                    "team": "Backend",
                    "components": ["api"],
                },
                chunk_id="chunk2",
                source_key="PROJ-2",
                item_type="Story",
            ),
            RetrievalResult(
                score=0.7,
                content={
                    "key": "PROJ-3",
                    "item_type": "Bug",
                    "priority": "High",
                    "status": "Open",
                    "epic_key": "PROJ-1",
                    "team": "Frontend",
                    "components": ["ui"],
                },
                chunk_id="chunk3",
                source_key="PROJ-3",
                item_type="Bug",
            ),
        ]

        # Create hierarchical context
        context_tree = ContextTree(
            root_item={"key": "PROJ-1", "item_type": "Epic"},
            child_items=[
                {"key": "PROJ-2", "item_type": "Story"},
                {"key": "PROJ-3", "item_type": "Bug"},
            ],
        )

        # Test summary generation logic
        summary = assembler.create_context_summary(results, [context_tree])

        # Verify business logic calculations
        assert summary.total_items == 6  # 3 results + 3 from context tree
        assert summary.item_types["Epic"] == 1
        assert summary.item_types["Story"] == 1
        assert summary.item_types["Bug"] == 1

        assert summary.priority_distribution["High"] == 2
        assert summary.priority_distribution["Medium"] == 1

        assert summary.status_distribution["In Progress"] == 1
        assert summary.status_distribution["Done"] == 1
        assert summary.status_distribution["Open"] == 1

        assert "PROJ-1" in summary.epics_covered
        assert "Frontend" in summary.teams_involved
        assert "Backend" in summary.teams_involved

    def test_token_limit_enforcement_logic(self):
        """Test token limit enforcement in context formatting."""
        assembler = ContextAssembler(max_tokens=200)  # Very low limit

        # Create results with very long content
        results = []
        for i in range(10):
            long_content = "This is a very long description that should be truncated. " * 20
            results.append(
                RetrievalResult(
                    score=0.9,
                    content={
                        "key": f"PROJ-{i}",
                        "item_type": "Story",
                        "priority": "High",
                        "status": "Open",
                        "text": long_content,
                    },
                    chunk_id=f"chunk{i}",
                    source_key=f"PROJ-{i}",
                    item_type="Story",
                )
            )

        query_plan = QueryPlan(
            original_query="test query",
            expanded_query=ExpandedQuery(
                original_query="test query",
                expanded_terms=[],
                semantic_variants=[],
                jira_synonyms={},
            ),
            filters=FilterSet(),
            intent="search",
            expected_output="list",
        )

        # Test token limit enforcement
        formatted_text, token_count = assembler._format_context_text(results, [], query_plan)

        # Verify business logic
        assert token_count <= 200  # Should respect token limit
        assert len(formatted_text) <= 200 * 4  # Rough character estimate
        assert "..." in formatted_text  # Should truncate long content

    @pytest.mark.asyncio
    async def test_error_handling_business_logic(self):
        """Test error handling business logic in pipeline."""
        # Mock clients that will fail
        mock_qdrant = Mock()
        mock_qdrant.search_with_filters = AsyncMock(
            side_effect=Exception("Qdrant connection failed")
        )

        mock_lm = Mock()
        mock_lm.generate_embeddings = AsyncMock(return_value=[[0.1] * 1536])

        pipeline = JiraRAGPipeline(
            qdrant_client=mock_qdrant,
            embedding_client=mock_lm,
            max_context_tokens=1000,
        )

        # Test error handling business logic
        result = await pipeline.process_query("test query")

        # Verify error handling logic
        assert result["success"] is False
        assert "error" in result
        assert "error_type" in result
        assert result["query"] == "test query"
        assert "Qdrant connection failed" in result["error"]

    def test_debt_analysis_business_logic(self):
        """Test technical debt analysis business logic."""
        # Create mock results with different debt patterns
        debt_results = [
            RetrievalResult(
                score=0.9,
                content={
                    "key": "DEBT-1",
                    "components": ["auth", "ui"],
                    "priority": "High",
                    "status": "Open",
                },
                chunk_id="chunk1",
                source_key="DEBT-1",
                item_type="Task",
            ),
            RetrievalResult(
                score=0.8,
                content={
                    "key": "DEBT-2",
                    "components": ["auth", "api"],
                    "priority": "High",
                    "status": "Blocked",
                },
                chunk_id="chunk2",
                source_key="DEBT-2",
                item_type="Task",
            ),
            RetrievalResult(
                score=0.7,
                content={
                    "key": "DEBT-3",
                    "components": ["database"],
                    "priority": "Medium",
                    "status": "Open",
                },
                chunk_id="chunk3",
                source_key="DEBT-3",
                item_type="Task",
            ),
        ]

        pipeline = JiraRAGPipeline(
            qdrant_client=Mock(),
            embedding_client=Mock(),
            max_context_tokens=1000,
        )

        # Test debt pattern analysis logic
        patterns = pipeline._analyze_debt_patterns(debt_results)

        # Verify business logic
        assert patterns["total_items"] == 3
        assert patterns["by_component"]["auth"] == 2  # Most common component
        assert patterns["by_component"]["ui"] == 1
        assert patterns["by_component"]["api"] == 1
        assert patterns["by_component"]["database"] == 1

        assert patterns["by_priority"]["High"] == 2
        assert patterns["by_priority"]["Medium"] == 1

        assert patterns["by_status"]["Open"] == 2
        assert patterns["by_status"]["Blocked"] == 1

        # Test recommendation generation logic
        recommendations = pipeline._generate_debt_recommendations(patterns)

        # Verify recommendation business logic
        assert len(recommendations) > 0
        assert any("auth" in rec and "2 debt items" in rec for rec in recommendations)
        assert any("2 high-priority debt items" in rec for rec in recommendations)
        assert any("2 open debt items" in rec for rec in recommendations)

    def test_epic_search_business_logic(self):
        """Test Epic search business logic."""
        # This would test the actual epic search logic
        # Including hierarchical context assembly and filtering
        # Placeholder for epic search tests

    def test_query_expansion_edge_cases(self):
        """Test query expansion edge cases."""
        processor = JiraQueryProcessor()

        # Test empty query
        empty_plan = processor.analyze_query("")
        assert empty_plan.expanded_query.expanded_terms == []

        # Test single word query
        single_plan = processor.analyze_query("bug")
        assert len(single_plan.expanded_query.expanded_terms) > 0
        assert "defect" in single_plan.expanded_query.expanded_terms

        # Test technical debt queries
        debt_plan = processor.analyze_query("technical debt cleanup")
        assert any(
            "refactoring" in term.lower() for term in debt_plan.expanded_query.expanded_terms
        )
        assert any(
            "maintenance" in term.lower() for term in debt_plan.expanded_query.expanded_terms
        )

    def test_filter_to_qdrant_conversion_logic(self):
        """Test filter conversion to Qdrant format."""
        filters = FilterSet(
            item_types=["Bug", "Story"],
            priorities=["High", "Critical"],
            statuses=["Open", "In Progress"],
            assignees=["john.doe", "jane.smith"],
            components=["frontend", "backend"],
        )

        # Test conversion logic
        qdrant_filters = filters.to_qdrant_filter()

        # Verify conversion business logic
        assert "item_type" in qdrant_filters
        assert "priority" in qdrant_filters
        assert "status" in qdrant_filters
        assert "assignee" in qdrant_filters
        assert "components" in qdrant_filters

        # Test that arrays are properly formatted
        assert isinstance(qdrant_filters["item_type"], list)
        assert "Bug" in qdrant_filters["item_type"]
        assert "Story" in qdrant_filters["item_type"]


class TestRAGPipelineIntegration:
    """Integration tests for RAG pipeline components."""

    def test_query_to_context_integration(self):
        """Test integration from query processing to context assembly."""
        # Test the full flow without external dependencies
        processor = JiraQueryProcessor()
        assembler = ContextAssembler(max_tokens=1000)

        # Process complex query
        query = "blocked high priority authentication bugs"
        plan = processor.analyze_query(query)

        # Create mock results that match the query
        matching_results = [
            RetrievalResult(
                score=0.9,
                content={
                    "key": "AUTH-1",
                    "item_type": "Bug",
                    "priority": "High",
                    "status": "Blocked",
                    "components": ["auth"],
                    "text": "Authentication service login failure",
                },
                chunk_id="chunk1",
                source_key="AUTH-1",
                item_type="Bug",
            ),
        ]

        # Test integration
        assembled = assembler.assemble_context(
            retrieval_results=matching_results,
            query_plan=plan,
            hierarchical_context=[],
        )

        # Verify integration works correctly
        assert len(assembled.primary_results) == 1
        assert assembled.primary_results[0].jira_key == "AUTH-1"
        assert assembled.summary.item_types["Bug"] == 1
        assert assembled.summary.priority_distribution["High"] == 1
        assert assembled.summary.status_distribution["Blocked"] == 1
        assert "AUTH-1" in assembled.jira_keys
        assert "authentication" in assembled.formatted_text.lower()
