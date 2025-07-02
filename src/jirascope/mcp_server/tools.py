"""MCP Tools implementation for JiraScope.

This module contains the tool definitions and implementations that are exposed
through the MCP server. These tools provide access to JiraScope's RAG capabilities.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..rag.pipeline import JiraRAGPipeline


@dataclass
class ToolResult:
    """Standard result format for MCP tools."""

    status: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    error: Optional[str] = None


class JiraScopeMCPTools:
    """Collection of MCP tools that expose JiraScope RAG capabilities."""

    def __init__(self, rag_pipeline: JiraRAGPipeline):
        self.rag_pipeline = rag_pipeline

    async def search_jira_issues(
        self, query: str, filters: Optional[Dict[str, Any]] = None, limit: int = 10
    ) -> ToolResult:
        """Search Jira issues using natural language query."""
        try:
            result = await self.rag_pipeline.process_query(user_query=query, include_hierarchy=True)

            return ToolResult(
                status="success",
                data={
                    "query": query,
                    "results": result.get("formatted_context", ""),
                    "jira_keys": result.get("jira_keys", []),
                },
                metadata={
                    "total_items": result.get("results_count", 0),
                    "query_analysis": {
                        "intent": result.get("intent", ""),
                        "filters_applied": result.get("filters_applied", {}),
                        "expected_output": result.get("expected_output", ""),
                    },
                },
            )

        except Exception as e:
            return ToolResult(status="error", data={}, metadata={}, error=str(e))

    async def analyze_technical_debt(self, component: Optional[str] = None) -> ToolResult:
        """Analyze technical debt patterns."""
        try:
            result = await self.rag_pipeline.analyze_technical_debt(team=component)

            return ToolResult(
                status="success",
                data={
                    "component": component,
                    "analysis": result.get("formatted_context", ""),
                    "patterns": result.get("jira_keys", []),
                },
                metadata={"total_debt_items": result.get("debt_items_found", 0)},
            )

        except Exception as e:
            return ToolResult(status="error", data={}, metadata={}, error=str(e))

    async def detect_scope_drift(self, epic_key: str) -> ToolResult:
        """Detect scope drift for an Epic."""
        try:
            result = await self.rag_pipeline.search_by_epic(epic_key, query="")

            return ToolResult(
                status="success",
                data={
                    "epic_key": epic_key,
                    "scope_analysis": result.get("formatted_context", ""),
                    "epic_context": result.get("epic_context"),
                    "related_items": result.get("jira_keys", []),
                },
                metadata={
                    "total_related_items": result.get("results_count", 0),
                    "child_count": result.get("child_count", 0),
                },
            )

        except Exception as e:
            return ToolResult(status="error", data={}, metadata={}, error=str(e))

    async def map_dependencies(
        self, epic_key: Optional[str] = None, team: Optional[str] = None
    ) -> ToolResult:
        """Map dependencies and blockers."""
        try:
            # Build dependency search query
            query = "blocked depends dependency waiting blocker"
            if epic_key:
                query += f" {epic_key}"
            if team:
                query += f" {team}"

            result = await self.rag_pipeline.process_query(user_query=query, include_hierarchy=True)

            return ToolResult(
                status="success",
                data={
                    "epic_key": epic_key,
                    "team": team,
                    "dependency_map": result.get("formatted_context", ""),
                    "blocked_items": result.get("jira_keys", []),
                },
                metadata={
                    "total_dependencies": result.get("results_count", 0),
                    "query_analysis": result.get("filters_applied", {}),
                },
            )

        except Exception as e:
            return ToolResult(status="error", data={}, metadata={}, error=str(e))
