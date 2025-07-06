"""Context assembly and ranking for RAG pipeline."""

from dataclasses import dataclass

from .query_processor import QueryPlan
from .retrieval_engine import ContextTree, RetrievalResult


@dataclass
class ContextSummary:
    """Summary of assembled context."""

    total_items: int
    item_types: dict[str, int]
    status_distribution: dict[str, int]
    priority_distribution: dict[str, int]
    epics_covered: list[str]
    teams_involved: list[str]

    def to_text(self) -> str:
        """Convert summary to readable text."""
        lines = [
            f"Context Summary: {self.total_items} items",
            f"Types: {', '.join(f'{k}({v})' for k, v in self.item_types.items())}",
            f"Status: {', '.join(f'{k}({v})' for k, v in self.status_distribution.items())}",
        ]

        if self.epics_covered:
            lines.append(f"Epics: {', '.join(self.epics_covered[:3])}")

        if self.teams_involved:
            lines.append(f"Teams: {', '.join(self.teams_involved[:3])}")

        return " | ".join(lines)


@dataclass
class AssembledContext:
    """Complete assembled context for LLM consumption."""

    primary_results: list[RetrievalResult]
    hierarchical_context: list[ContextTree]
    summary: ContextSummary
    formatted_text: str
    token_count: int

    @property
    def jira_keys(self) -> list[str]:
        """Get all Jira keys mentioned in this context."""
        keys = []
        for result in self.primary_results:
            keys.append(result.jira_key)

        for tree in self.hierarchical_context:
            if tree.root_item.get("key"):
                keys.append(tree.root_item["key"])

        return list(set(keys))


class ContextAssembler:
    """Assembles retrieval results into coherent context for LLM consumption."""

    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens

    def assemble_context(
        self,
        retrieval_results: list[RetrievalResult],
        query_plan: QueryPlan,
        hierarchical_context: list[ContextTree] | None = None,
    ) -> AssembledContext:
        """Intelligently combine retrieved content into coherent context."""
        if hierarchical_context is None:
            hierarchical_context = []

        # Rank and filter results by relevance
        ranked_results = self.rank_by_relevance(retrieval_results, query_plan)

        # Create context summary
        summary = self.create_context_summary(ranked_results, hierarchical_context)

        # Format context text within token limits
        formatted_text, token_count = self._format_context_text(
            ranked_results, hierarchical_context, query_plan
        )

        return AssembledContext(
            primary_results=ranked_results,
            hierarchical_context=hierarchical_context,
            summary=summary,
            formatted_text=formatted_text,
            token_count=token_count,
        )

    def create_context_summary(
        self, results: list[RetrievalResult], hierarchical_context: list[ContextTree]
    ) -> ContextSummary:
        """Generate concise summary of the assembled context."""
        # Count items by type
        item_types = {}
        status_distribution = {}
        priority_distribution = {}
        epics_covered = set()
        teams_involved = set()

        # Analyze primary results
        for result in results:
            content = result.content

            # Count item types
            item_type = content.get("item_type", "Unknown")
            item_types[item_type] = item_types.get(item_type, 0) + 1

            # Count statuses
            status = content.get("status", "Unknown")
            status_distribution[status] = status_distribution.get(status, 0) + 1

            # Count priorities
            priority = content.get("priority", "Unknown")
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1

            # Collect epics
            if content.get("epic_key"):
                epics_covered.add(content["epic_key"])

            # Collect teams
            if content.get("team"):
                teams_involved.add(content["team"])

        # Analyze hierarchical context
        for tree in hierarchical_context:
            if tree.root_item.get("key"):
                epics_covered.add(tree.root_item["key"])

        total_items = len(results) + sum(tree.total_items for tree in hierarchical_context)

        return ContextSummary(
            total_items=total_items,
            item_types=item_types,
            status_distribution=status_distribution,
            priority_distribution=priority_distribution,
            epics_covered=list(epics_covered),
            teams_involved=list(teams_involved),
        )

    def rank_by_relevance(
        self, results: list[RetrievalResult], query_plan: QueryPlan
    ) -> list[RetrievalResult]:
        """Multi-factor relevance ranking."""
        for result in results:
            relevance_score = result.score

            # Boost based on query intent
            if query_plan.intent == "analysis":
                # For analysis queries, boost items with more metadata
                if result.content.get("dependency_count", 0) > 0:
                    relevance_score *= 1.3
                if result.content.get("has_children"):
                    relevance_score *= 1.2

            # Boost items matching filters exactly
            filters = query_plan.filters
            if filters.item_types and result.item_type in filters.item_types:
                relevance_score *= 1.4

            if filters.priorities and result.content.get("priority") in filters.priorities:
                relevance_score *= 1.3

            # Apply priority boosts from query plan
            if query_plan.priority_boost:
                for boost_type, boost_value in query_plan.priority_boost.items():
                    if (
                        boost_type == "priority_boost" and result.content.get("priority") == "High"
                    ) or (
                        boost_type == "status_boost" and result.content.get("status") == "Blocked"
                    ):
                        relevance_score *= boost_value

            # Update score
            result.score = relevance_score

        # Sort by updated relevance score
        return sorted(results, key=lambda x: x.score, reverse=True)

    def _format_context_text(
        self,
        results: list[RetrievalResult],
        hierarchical_context: list[ContextTree],
        query_plan: QueryPlan,
    ) -> tuple[str, int]:
        """Format context text within token limits."""
        sections = []
        estimated_tokens = 0

        # Add query context
        sections.append(f"Query: {query_plan.original_query}")
        sections.append(f"Intent: {query_plan.intent}")
        estimated_tokens += 20

        # Add primary results
        if results:
            sections.append("\n## Relevant Items:")
            for _i, result in enumerate(results[:10]):  # Limit to top 10
                content = result.content

                # Format item header
                item_header = (
                    f"\n### {content.get('key', 'Unknown')} - {content.get('item_type', 'Unknown')}"
                )
                sections.append(item_header)

                # Add key metadata
                metadata_parts = []
                if content.get("status"):
                    metadata_parts.append(f"Status: {content['status']}")
                if content.get("priority"):
                    metadata_parts.append(f"Priority: {content['priority']}")
                if content.get("epic_key"):
                    metadata_parts.append(f"Epic: {content['epic_key']}")

                if metadata_parts:
                    sections.append(f"**{' | '.join(metadata_parts)}**")

                # Add content text
                text_content = content.get("text", "")
                if text_content:
                    # Truncate if too long
                    if len(text_content) > 500:
                        text_content = text_content[:500] + "..."
                    sections.append(text_content)

                # Estimate tokens (rough: 4 chars per token)
                section_tokens = len("\n".join(sections[-4:])) // 4
                estimated_tokens += section_tokens

                # Stop if approaching token limit
                if estimated_tokens > self.max_tokens * 0.8:
                    break

        # Add hierarchical context if space allows
        if hierarchical_context and estimated_tokens < self.max_tokens * 0.6:
            sections.append("\n## Epic Context:")
            for tree in hierarchical_context[:3]:  # Limit to 3 epics
                if tree.root_item.get("key"):
                    epic_header = f"\n### Epic: {tree.root_item['key']}"
                    sections.append(epic_header)

                    if tree.child_items:
                        child_keys = [item.get("key", "Unknown") for item in tree.child_items[:5]]
                        sections.append(f"Child Items: {', '.join(child_keys)}")

                    # Estimate tokens and check limit
                    section_tokens = len("\n".join(sections[-2:])) // 4
                    estimated_tokens += section_tokens

                    if estimated_tokens > self.max_tokens * 0.9:
                        break

        formatted_text = "\n".join(sections)

        # Final token count estimation
        final_token_count = len(formatted_text) // 4

        return formatted_text, final_token_count
