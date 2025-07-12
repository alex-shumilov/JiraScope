"""JiraScope MCP Server implementation using FastMCP."""

import asyncio
import time
from typing import Any

from mcp.server.fastmcp import FastMCP

from ..clients.lmstudio_client import LMStudioClient
from ..clients.qdrant_client import QdrantVectorClient
from ..core.config import Config
from ..rag.pipeline import JiraRAGPipeline
from ..utils.logging import StructuredLogger

logger = StructuredLogger(__name__)

# Create the MCP server instance
mcp = FastMCP("JiraScope", dependencies=["qdrant-client", "httpx", "pydantic"])

# Global variables for components (initialized in main)
rag_pipeline: JiraRAGPipeline | None = None
config: Config | None = None


async def init_components():
    """Initialize the RAG pipeline and other components."""
    global rag_pipeline, config

    try:
        logger.info("Initializing JiraScope MCP Server components...")

        # Load configuration
        config = Config.from_env()

        # Initialize clients
        qdrant_client = QdrantVectorClient(config)
        lm_client = LMStudioClient(config)

        # Initialize async contexts
        await qdrant_client.__aenter__()
        await lm_client.__aenter__()

        # Initialize RAG pipeline
        rag_pipeline = JiraRAGPipeline(qdrant_client, lm_client)

        logger.info("JiraScope MCP Server components initialized successfully")

    except Exception as e:
        logger.exception(f"Failed to initialize components: {e}")
        raise


@mcp.tool()
async def search_jira_issues(query: str, limit: int | None = 10) -> dict[str, Any]:
    """Search Jira issues using natural language query.

    Args:
        query: Natural language search query (e.g., "high priority bugs in frontend")
        limit: Maximum number of results to return (default: 10)

    Returns:
        Dict containing search results, metadata, and query analysis
    """
    if not rag_pipeline:
        return {"status": "error", "error": "RAG pipeline not initialized"}

    try:
        start_time = time.time()
        result = await rag_pipeline.process_query(user_query=query, include_hierarchy=True)

        return {
            "status": "success",
            "query": query,
            "results": result.get("formatted_context", ""),
            "metadata": {
                "total_items": result.get("results_count", 0),
                "processing_time": time.time() - start_time,
                "query_analysis": {
                    "intent": result.get("intent", ""),
                    "filters_applied": result.get("filters_applied", {}),
                    "expected_output": result.get("expected_output", ""),
                },
            },
            "jira_keys": result.get("jira_keys", []),
        }

    except Exception as e:
        logger.exception(f"Error in search_jira_issues: {e}")
        return {"status": "error", "error": str(e), "query": query}


@mcp.tool()
async def analyze_technical_debt(
    component: str | None = None, time_range: str | None = None
) -> dict[str, Any]:
    """Analyze technical debt patterns across Jira issues.

    Args:
        component: Optional component to focus on (e.g., "frontend", "backend")
        time_range: Optional time range (e.g., "last month", "last quarter")

    Returns:
        Dict containing technical debt analysis and recommendations
    """
    if not rag_pipeline:
        return {"status": "error", "error": "RAG pipeline not initialized"}

    try:
        result = await rag_pipeline.analyze_technical_debt(team=component)

        return {
            "status": "success",
            "component": component,
            "time_range": time_range,
            "analysis": result.get("formatted_context", ""),
            "patterns": result.get("jira_keys", []),
            "metadata": {
                "total_debt_items": result.get("debt_items_found", 0),
                "analysis_cost": 0.0,
            },
        }

    except Exception as e:
        logger.exception(f"Error in analyze_technical_debt: {e}")
        return {"status": "error", "error": str(e), "component": component}


@mcp.tool()
async def detect_scope_drift(epic_key: str) -> dict[str, Any]:
    """Detect and analyze scope drift for a specific Epic.

    Args:
        epic_key: Epic key to analyze (e.g., "PROJ-123")

    Returns:
        Dict containing scope drift analysis and timeline
    """
    if not rag_pipeline:
        return {"status": "error", "error": "RAG pipeline not initialized"}

    try:
        result = await rag_pipeline.search_by_epic(epic_key, query="")

        return {
            "status": "success",
            "epic_key": epic_key,
            "scope_analysis": result.get("formatted_context", ""),
            "epic_context": result.get("epic_context"),
            "metadata": {
                "total_related_items": result.get("results_count", 0),
                "child_count": result.get("child_count", 0),
            },
            "related_items": result.get("jira_keys", []),
        }

    except Exception as e:
        logger.exception(f"Error in detect_scope_drift: {e}")
        return {"status": "error", "error": str(e), "epic_key": epic_key}


@mcp.tool()
async def map_dependencies(epic_key: str | None = None, team: str | None = None) -> dict[str, Any]:
    """Map dependencies and blockers across teams and Epics.

    Args:
        epic_key: Optional Epic key to focus on
        team: Optional team name to analyze dependencies for

    Returns:
        Dict containing dependency mapping and blocker analysis
    """
    if not rag_pipeline:
        return {"status": "error", "error": "RAG pipeline not initialized"}

    try:
        # Build dependency search query
        query = "blocked depends dependency waiting blocker"
        if epic_key:
            query += f" {epic_key}"
        if team:
            query += f" {team}"

        result = await rag_pipeline.process_query(user_query=query, include_hierarchy=True)

        return {
            "status": "success",
            "epic_key": epic_key,
            "team": team,
            "dependency_map": result.get("formatted_context", ""),
            "blocked_items": result.get("jira_keys", []),
            "metadata": {
                "total_dependencies": result.get("results_count", 0),
                "query_analysis": result.get("filters_applied", {}),
            },
        }

    except Exception as e:
        logger.exception(f"Error in map_dependencies: {e}")
        return {"status": "error", "error": str(e), "epic_key": epic_key, "team": team}


@mcp.resource("jira://config")
def get_config() -> str:
    """Get current JiraScope configuration and status."""
    if not config:
        return "Configuration not loaded"

    config_info = {
        "qdrant_url": config.qdrant_url,
        "lm_endpoint": config.lmstudio_endpoint,
        "jira_endpoint": config.jira_mcp_endpoint,
        "claude_model": config.claude_model,
        "embedding_batch_size": config.embedding_batch_size,
    }
    return f"JiraScope Configuration:\n{config_info}"


@mcp.prompt()
def jira_analysis_prompt(analysis_type: str, focus_area: str | None = None) -> str:
    """Generate prompts for different types of Jira analysis.

    Args:
        analysis_type: Type of analysis (e.g., "technical_debt", "scope_drift", "dependencies")
        focus_area: Optional focus area (e.g., component name, Epic key)
    """
    base_prompt = f"Analyze the following Jira data for {analysis_type}"
    if focus_area:
        base_prompt += f" focusing on {focus_area}"

    base_prompt += ". Provide insights, patterns, and actionable recommendations."
    return base_prompt


@mcp.prompt()
def sprint_planning_prompt(team: str, sprint_goal: str | None = None) -> str:
    """Generate sprint planning analysis prompt.

    Args:
        team: Team name for sprint planning
        sprint_goal: Optional sprint goal or theme
    """
    prompt = f"Analyze the current backlog and work items for {team} team"
    if sprint_goal:
        prompt += f" with sprint goal: {sprint_goal}"

    prompt += (
        ". Identify capacity, dependencies, risks, and provide sprint planning recommendations."
    )
    return prompt


async def main():
    """Main entry point for running the MCP server."""
    try:
        # Initialize components first
        await init_components()

        logger.info("Starting JiraScope MCP Server...")

        # Run the MCP server
        mcp.run()

    except Exception as e:
        logger.exception(f"Failed to start MCP server: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
