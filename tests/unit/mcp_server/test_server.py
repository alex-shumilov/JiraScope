"""Tests for MCP server functionality."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.jirascope.clients.lmstudio_client import LMStudioClient
from src.jirascope.clients.qdrant_client import QdrantVectorClient
from src.jirascope.core.config import Config
from src.jirascope.mcp_server import server
from src.jirascope.rag.pipeline import JiraRAGPipeline


class TestMCPServerInitialization:
    """Test MCP server initialization functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset global variables
        server.rag_pipeline = None
        server.config = None

    @pytest.mark.asyncio
    @patch("src.jirascope.mcp_server.server.Config.from_env")
    @patch("src.jirascope.mcp_server.server.QdrantVectorClient")
    @patch("src.jirascope.mcp_server.server.LMStudioClient")
    @patch("src.jirascope.mcp_server.server.JiraRAGPipeline")
    async def test_init_components_success(self, mock_pipeline, mock_lm, mock_qdrant, mock_config):
        """Test successful component initialization."""
        # Setup mocks
        mock_config_instance = Mock(spec=Config)
        mock_config.return_value = mock_config_instance

        mock_qdrant_instance = AsyncMock(spec=QdrantVectorClient)
        mock_qdrant.return_value = mock_qdrant_instance

        mock_lm_instance = AsyncMock(spec=LMStudioClient)
        mock_lm.return_value = mock_lm_instance

        mock_pipeline_instance = Mock(spec=JiraRAGPipeline)
        mock_pipeline.return_value = mock_pipeline_instance

        # Execute
        await server.init_components()

        # Verify
        assert server.config is mock_config_instance
        assert server.rag_pipeline is mock_pipeline_instance

        # Verify clients were initialized
        mock_qdrant.assert_called_once_with(mock_config_instance)
        mock_lm.assert_called_once_with(mock_config_instance)

        # Verify async context managers were entered
        mock_qdrant_instance.__aenter__.assert_called_once()
        mock_lm_instance.__aenter__.assert_called_once()

        # Verify pipeline was created with clients
        mock_pipeline.assert_called_once_with(mock_qdrant_instance, mock_lm_instance)

    @pytest.mark.asyncio
    @patch("src.jirascope.mcp_server.server.Config.from_env")
    async def test_init_components_config_error(self, mock_config):
        """Test initialization failure due to config error."""
        # Setup mock to raise exception
        mock_config.side_effect = Exception("Config load failed")

        # Execute and verify exception is raised
        with pytest.raises(Exception, match="Config load failed"):
            await server.init_components()

        # Verify global variables remain None
        assert server.config is None
        assert server.rag_pipeline is None

    @pytest.mark.asyncio
    @patch("src.jirascope.mcp_server.server.Config.from_env")
    @patch("src.jirascope.mcp_server.server.QdrantVectorClient")
    async def test_init_components_client_error(self, mock_qdrant, mock_config):
        """Test initialization failure due to client error."""
        # Setup mocks
        mock_config_instance = Mock(spec=Config)
        mock_config.return_value = mock_config_instance

        mock_qdrant_instance = AsyncMock(spec=QdrantVectorClient)
        mock_qdrant_instance.__aenter__.side_effect = Exception("Qdrant connection failed")
        mock_qdrant.return_value = mock_qdrant_instance

        # Execute and verify exception is raised
        with pytest.raises(Exception, match="Qdrant connection failed"):
            await server.init_components()


class TestMCPServerTools:
    """Test MCP server tool functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Set up global rag_pipeline mock
        server.rag_pipeline = AsyncMock(spec=JiraRAGPipeline)

    def teardown_method(self):
        """Clean up after tests."""
        server.rag_pipeline = None

    @pytest.mark.asyncio
    async def test_search_jira_issues_success(self):
        """Test successful Jira issue search."""
        # Setup mock response
        mock_result = {
            "formatted_context": "Found 3 authentication issues",
            "results_count": 3,
            "intent": "search",
            "filters_applied": {"component": "auth"},
            "expected_output": "issue list",
            "jira_keys": ["AUTH-1", "AUTH-2", "AUTH-3"],
        }
        server.rag_pipeline.process_query.return_value = mock_result

        # Execute
        result = await server.search_jira_issues("authentication issues", limit=10)

        # Verify
        assert result["status"] == "success"
        assert result["query"] == "authentication issues"
        assert result["results"] == "Found 3 authentication issues"
        assert result["metadata"]["total_items"] == 3
        assert result["metadata"]["query_analysis"]["intent"] == "search"
        assert result["jira_keys"] == ["AUTH-1", "AUTH-2", "AUTH-3"]
        assert "processing_time" in result["metadata"]

        # Verify pipeline was called correctly
        server.rag_pipeline.process_query.assert_called_once_with(
            user_query="authentication issues", include_hierarchy=True
        )

    @pytest.mark.asyncio
    async def test_search_jira_issues_no_pipeline(self):
        """Test search when pipeline is not initialized."""
        # Clear the pipeline
        server.rag_pipeline = None

        # Execute
        result = await server.search_jira_issues("test query")

        # Verify error response
        assert result["status"] == "error"
        assert result["error"] == "RAG pipeline not initialized"

    @pytest.mark.asyncio
    async def test_search_jira_issues_pipeline_error(self):
        """Test search when pipeline raises an exception."""
        # Setup mock to raise exception
        server.rag_pipeline.process_query.side_effect = Exception("Pipeline error")

        # Execute
        result = await server.search_jira_issues("test query")

        # Verify error response
        assert result["status"] == "error"
        assert result["error"] == "Pipeline error"
        assert result["query"] == "test query"

    @pytest.mark.asyncio
    async def test_search_jira_issues_default_limit(self):
        """Test search with default limit parameter."""
        mock_result = {
            "formatted_context": "Results",
            "results_count": 5,
            "jira_keys": ["ITEM-1", "ITEM-2"],
        }
        server.rag_pipeline.process_query.return_value = mock_result

        # Execute without specifying limit
        result = await server.search_jira_issues("test query")

        # Verify
        assert result["status"] == "success"
        # The function doesn't actually use the limit parameter in the implementation
        # but we verify it accepts the default

    @pytest.mark.asyncio
    async def test_analyze_technical_debt_success(self):
        """Test successful technical debt analysis."""
        mock_result = {
            "formatted_context": "Found 5 debt items in frontend",
            "jira_keys": ["DEBT-1", "DEBT-2"],
            "debt_items_found": 5,
        }
        server.rag_pipeline.analyze_technical_debt.return_value = mock_result

        # Execute
        result = await server.analyze_technical_debt("frontend", "last month")

        # Verify
        assert result["status"] == "success"
        assert result["component"] == "frontend"
        assert result["time_range"] == "last month"
        assert result["analysis"] == "Found 5 debt items in frontend"
        assert result["patterns"] == ["DEBT-1", "DEBT-2"]
        assert result["metadata"]["total_debt_items"] == 5

        # Verify pipeline was called correctly
        server.rag_pipeline.analyze_technical_debt.assert_called_once_with(team="frontend")

    @pytest.mark.asyncio
    async def test_analyze_technical_debt_no_pipeline(self):
        """Test technical debt analysis when pipeline is not initialized."""
        server.rag_pipeline = None

        result = await server.analyze_technical_debt("frontend")

        assert result["status"] == "error"
        assert result["error"] == "RAG pipeline not initialized"

    @pytest.mark.asyncio
    async def test_analyze_technical_debt_error(self):
        """Test technical debt analysis error handling."""
        server.rag_pipeline.analyze_technical_debt.side_effect = Exception("Analysis failed")

        result = await server.analyze_technical_debt("backend")

        assert result["status"] == "error"
        assert result["error"] == "Analysis failed"
        assert result["component"] == "backend"

    @pytest.mark.asyncio
    async def test_detect_scope_drift_success(self):
        """Test successful scope drift detection."""
        mock_result = {
            "formatted_context": "Epic has expanded beyond original scope",
            "epic_context": {"epic_key": "EPIC-123", "summary": "User Auth"},
            "results_count": 8,
            "child_count": 12,
            "jira_keys": ["STORY-1", "STORY-2"],
        }
        server.rag_pipeline.search_by_epic.return_value = mock_result

        # Execute
        result = await server.detect_scope_drift("EPIC-123")

        # Verify
        assert result["status"] == "success"
        assert result["epic_key"] == "EPIC-123"
        assert result["scope_analysis"] == "Epic has expanded beyond original scope"
        assert result["epic_context"]["epic_key"] == "EPIC-123"
        assert result["metadata"]["total_related_items"] == 8
        assert result["metadata"]["child_count"] == 12
        assert result["related_items"] == ["STORY-1", "STORY-2"]

        # Verify pipeline was called correctly
        server.rag_pipeline.search_by_epic.assert_called_once_with("EPIC-123", query="")

    @pytest.mark.asyncio
    async def test_detect_scope_drift_no_pipeline(self):
        """Test scope drift detection when pipeline is not initialized."""
        server.rag_pipeline = None

        result = await server.detect_scope_drift("EPIC-123")

        assert result["status"] == "error"
        assert result["error"] == "RAG pipeline not initialized"

    @pytest.mark.asyncio
    async def test_detect_scope_drift_error(self):
        """Test scope drift detection error handling."""
        server.rag_pipeline.search_by_epic.side_effect = Exception("Epic not found")

        result = await server.detect_scope_drift("INVALID-EPIC")

        assert result["status"] == "error"
        assert result["error"] == "Epic not found"
        assert result["epic_key"] == "INVALID-EPIC"

    @pytest.mark.asyncio
    async def test_map_dependencies_success(self):
        """Test successful dependency mapping."""
        mock_result = {
            "formatted_context": "Found critical dependencies and blockers",
            "results_count": 6,
            "filters_applied": {"status": "blocked"},
            "jira_keys": ["BLOCKED-1", "BLOCKED-2", "BLOCKED-3"],
        }
        server.rag_pipeline.process_query.return_value = mock_result

        # Execute
        result = await server.map_dependencies("EPIC-456", "frontend")

        # Verify
        assert result["status"] == "success"
        assert result["epic_key"] == "EPIC-456"
        assert result["team"] == "frontend"
        assert result["dependency_map"] == "Found critical dependencies and blockers"
        assert result["blocked_items"] == ["BLOCKED-1", "BLOCKED-2", "BLOCKED-3"]
        assert result["metadata"]["total_dependencies"] == 6

        # Verify the query was built correctly
        expected_query = "blocked depends dependency waiting blocker EPIC-456 frontend"
        server.rag_pipeline.process_query.assert_called_once_with(
            user_query=expected_query, include_hierarchy=True
        )

    @pytest.mark.asyncio
    async def test_map_dependencies_epic_only(self):
        """Test dependency mapping with only epic key."""
        mock_result = {
            "formatted_context": "Epic dependencies",
            "results_count": 2,
            "filters_applied": {},
            "jira_keys": ["DEP-1"],
        }
        server.rag_pipeline.process_query.return_value = mock_result

        result = await server.map_dependencies("EPIC-789")

        assert result["status"] == "success"
        assert result["epic_key"] == "EPIC-789"
        assert result["team"] is None

        # Verify query excludes team
        expected_query = "blocked depends dependency waiting blocker EPIC-789"
        server.rag_pipeline.process_query.assert_called_once_with(
            user_query=expected_query, include_hierarchy=True
        )

    @pytest.mark.asyncio
    async def test_map_dependencies_team_only(self):
        """Test dependency mapping with only team."""
        mock_result = {
            "formatted_context": "Team dependencies",
            "results_count": 3,
            "filters_applied": {},
            "jira_keys": ["TEAM-DEP-1"],
        }
        server.rag_pipeline.process_query.return_value = mock_result

        result = await server.map_dependencies(team="backend")

        assert result["status"] == "success"
        assert result["epic_key"] is None
        assert result["team"] == "backend"

        # Verify query excludes epic
        expected_query = "blocked depends dependency waiting blocker backend"
        server.rag_pipeline.process_query.assert_called_once_with(
            user_query=expected_query, include_hierarchy=True
        )

    @pytest.mark.asyncio
    async def test_map_dependencies_no_params(self):
        """Test dependency mapping with no parameters."""
        mock_result = {
            "formatted_context": "General dependencies",
            "results_count": 1,
            "filters_applied": {},
            "jira_keys": ["GENERAL-1"],
        }
        server.rag_pipeline.process_query.return_value = mock_result

        result = await server.map_dependencies()

        assert result["status"] == "success"
        assert result["epic_key"] is None
        assert result["team"] is None

        # Verify query is base query only
        expected_query = "blocked depends dependency waiting blocker"
        server.rag_pipeline.process_query.assert_called_once_with(
            user_query=expected_query, include_hierarchy=True
        )

    @pytest.mark.asyncio
    async def test_map_dependencies_no_pipeline(self):
        """Test dependency mapping when pipeline is not initialized."""
        server.rag_pipeline = None

        result = await server.map_dependencies("EPIC-123", "qa")

        assert result["status"] == "error"
        assert result["error"] == "RAG pipeline not initialized"

    @pytest.mark.asyncio
    async def test_map_dependencies_error(self):
        """Test dependency mapping error handling."""
        server.rag_pipeline.process_query.side_effect = Exception("Dependency analysis failed")

        result = await server.map_dependencies("EPIC-999", "qa")

        assert result["status"] == "error"
        assert result["error"] == "Dependency analysis failed"
        assert result["epic_key"] == "EPIC-999"
        assert result["team"] == "qa"


class TestMCPServerResources:
    """Test MCP server resource functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Set up global config mock
        server.config = Mock(spec=Config)
        server.config.qdrant_url = "http://localhost:6333"
        server.config.lmstudio_endpoint = "http://localhost:1234/v1"
        server.config.jira_mcp_endpoint = "http://localhost:8000"
        server.config.claude_model = "claude-3-sonnet"
        server.config.embedding_batch_size = 32

    def teardown_method(self):
        """Clean up after tests."""
        server.config = None

    def test_get_config_success(self):
        """Test successful config retrieval."""
        result = server.get_config()

        # Verify config information is returned as formatted string
        assert "JiraScope Configuration" in result
        assert "http://localhost:6333" in result
        assert "http://localhost:1234/v1" in result
        assert "http://localhost:8000" in result
        assert "claude-3-sonnet" in result
        assert "32" in result

    def test_get_config_no_config(self):
        """Test config retrieval when config is not initialized."""
        server.config = None

        result = server.get_config()

        assert result == "Configuration not loaded"


class TestMCPServerPrompts:
    """Test MCP server prompt functions."""

    def test_jira_analysis_prompt_with_focus(self):
        """Test Jira analysis prompt with focus area."""
        result = server.jira_analysis_prompt("technical_debt", "frontend")

        assert "technical_debt" in result
        assert "frontend" in result
        assert "analyze" in result.lower()
        assert len(result) > 50  # Ensure it's a substantial prompt

    def test_jira_analysis_prompt_no_focus(self):
        """Test Jira analysis prompt without focus area."""
        result = server.jira_analysis_prompt("scope_drift")

        assert "scope_drift" in result
        assert "analyze" in result.lower()
        assert len(result) > 50

    def test_sprint_planning_prompt_with_goal(self):
        """Test sprint planning prompt with sprint goal."""
        result = server.sprint_planning_prompt("frontend", "Implement user authentication")

        assert "frontend" in result
        assert "Implement user authentication" in result
        assert "sprint" in result.lower()
        assert "planning" in result.lower()
        assert len(result) > 50

    def test_sprint_planning_prompt_no_goal(self):
        """Test sprint planning prompt without sprint goal."""
        result = server.sprint_planning_prompt("backend")

        assert "backend" in result
        assert "sprint" in result.lower()
        assert "planning" in result.lower()
        assert len(result) > 50


class TestMCPServerIntegration:
    """Test MCP server integration functionality."""

    @pytest.mark.asyncio
    async def test_processing_time_measurement(self):
        """Test that processing time is measured correctly."""
        # Setup
        server.rag_pipeline = AsyncMock(spec=JiraRAGPipeline)

        # Mock response with proper structure
        mock_result = {
            "formatted_context": "Results",
            "results_count": 1,
            "intent": "search",
            "filters_applied": {},
            "expected_output": "issues",
            "jira_keys": ["ITEM-1"],
        }
        server.rag_pipeline.process_query.return_value = mock_result

        # Execute
        result = await server.search_jira_issues("test query")

        # Verify timing was measured
        assert result["status"] == "success"
        assert "processing_time" in result["metadata"]
        processing_time = result["metadata"]["processing_time"]
        assert processing_time >= 0  # Should be non-negative

    @pytest.mark.asyncio
    async def test_default_value_handling(self):
        """Test handling of missing values in pipeline responses."""
        server.rag_pipeline = AsyncMock(spec=JiraRAGPipeline)

        # Return minimal response missing many expected keys
        server.rag_pipeline.process_query.return_value = {
            "formatted_context": "Minimal response"
            # Missing results_count, jira_keys, intent, etc.
        }

        result = await server.search_jira_issues("test query")

        # Verify defaults are handled correctly
        assert result["status"] == "success"
        assert result["results"] == "Minimal response"
        assert result["metadata"]["total_items"] == 0  # Default for missing results_count
        assert result["jira_keys"] == []  # Default for missing jira_keys
        assert result["metadata"]["query_analysis"]["intent"] == ""  # Default

    def test_mcp_server_instance_creation(self):
        """Test that MCP server instance is created correctly."""
        # Verify the mcp instance exists and has correct configuration
        assert server.mcp is not None
        assert hasattr(server.mcp, "name")
        # Note: We can't easily test the FastMCP instance deeply without
        # importing the actual library, but we can verify basic attributes
