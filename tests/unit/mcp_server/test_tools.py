"""Tests for MCP server tools functionality."""

from unittest.mock import AsyncMock, Mock

import pytest

from src.jirascope.mcp_server.tools import JiraScopeMCPTools, ToolResult
from src.jirascope.rag.pipeline import JiraRAGPipeline


class TestToolResult:
    """Test ToolResult dataclass functionality."""

    def test_tool_result_creation_success(self):
        """Test creating a successful ToolResult."""
        result = ToolResult(status="success", data={"key": "value"}, metadata={"count": 1})

        assert result.status == "success"
        assert result.data == {"key": "value"}
        assert result.metadata == {"count": 1}
        assert result.error is None

    def test_tool_result_creation_with_error(self):
        """Test creating a ToolResult with error."""
        result = ToolResult(status="error", data={}, metadata={}, error="Something went wrong")

        assert result.status == "error"
        assert result.data == {}
        assert result.metadata == {}
        assert result.error == "Something went wrong"


class TestJiraScopeMCPTools:
    """Test JiraScopeMCPTools functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_rag_pipeline = AsyncMock(spec=JiraRAGPipeline)
        self.tools = JiraScopeMCPTools(self.mock_rag_pipeline)

    @pytest.mark.asyncio
    async def test_search_jira_issues_success(self):
        """Test successful Jira issue search."""
        # Setup mock response
        mock_result = {
            "formatted_context": "Found 2 issues related to authentication",
            "jira_keys": ["PROJ-123", "PROJ-124"],
            "results_count": 2,
            "intent": "search",
            "filters_applied": {"component": "auth"},
            "expected_output": "Jira issues",
        }
        self.mock_rag_pipeline.process_query.return_value = mock_result

        # Execute
        result = await self.tools.search_jira_issues("authentication issues", limit=10)

        # Verify
        assert result.status == "success"
        assert result.data["query"] == "authentication issues"
        assert result.data["results"] == "Found 2 issues related to authentication"
        assert result.data["jira_keys"] == ["PROJ-123", "PROJ-124"]
        assert result.metadata["total_items"] == 2
        assert result.metadata["query_analysis"]["intent"] == "search"
        assert result.error is None

        # Verify pipeline was called correctly
        self.mock_rag_pipeline.process_query.assert_called_once_with(
            user_query="authentication issues", include_hierarchy=True
        )

    @pytest.mark.asyncio
    async def test_search_jira_issues_with_filters(self):
        """Test Jira issue search with filters."""
        mock_result = {
            "formatted_context": "Filtered results",
            "jira_keys": ["PROJ-125"],
            "results_count": 1,
            "intent": "search",
            "filters_applied": {"priority": "high"},
            "expected_output": "Filtered issues",
        }
        self.mock_rag_pipeline.process_query.return_value = mock_result

        # Execute with filters
        filters = {"priority": "high", "status": "open"}
        result = await self.tools.search_jira_issues("bugs", filters=filters, limit=5)

        # Verify
        assert result.status == "success"
        assert result.data["jira_keys"] == ["PROJ-125"]
        assert result.metadata["total_items"] == 1

    @pytest.mark.asyncio
    async def test_search_jira_issues_error_handling(self):
        """Test error handling in search_jira_issues."""
        # Setup mock to raise exception
        self.mock_rag_pipeline.process_query.side_effect = Exception("Pipeline error")

        # Execute
        result = await self.tools.search_jira_issues("test query")

        # Verify error handling
        assert result.status == "error"
        assert result.data == {}
        assert result.metadata == {}
        assert result.error == "Pipeline error"

    @pytest.mark.asyncio
    async def test_analyze_technical_debt_success(self):
        """Test successful technical debt analysis."""
        mock_result = {
            "formatted_context": "Found 5 technical debt items",
            "jira_keys": ["DEBT-1", "DEBT-2", "DEBT-3"],
            "debt_items_found": 5,
        }
        self.mock_rag_pipeline.analyze_technical_debt.return_value = mock_result

        # Execute
        result = await self.tools.analyze_technical_debt("frontend")

        # Verify
        assert result.status == "success"
        assert result.data["component"] == "frontend"
        assert result.data["analysis"] == "Found 5 technical debt items"
        assert result.data["patterns"] == ["DEBT-1", "DEBT-2", "DEBT-3"]
        assert result.metadata["total_debt_items"] == 5

        # Verify pipeline was called correctly
        self.mock_rag_pipeline.analyze_technical_debt.assert_called_once_with(team="frontend")

    @pytest.mark.asyncio
    async def test_analyze_technical_debt_no_component(self):
        """Test technical debt analysis without specifying component."""
        mock_result = {
            "formatted_context": "Overall technical debt analysis",
            "jira_keys": ["DEBT-4"],
            "debt_items_found": 1,
        }
        self.mock_rag_pipeline.analyze_technical_debt.return_value = mock_result

        # Execute without component
        result = await self.tools.analyze_technical_debt()

        # Verify
        assert result.status == "success"
        assert result.data["component"] is None
        assert result.metadata["total_debt_items"] == 1

        # Verify pipeline was called with None
        self.mock_rag_pipeline.analyze_technical_debt.assert_called_once_with(team=None)

    @pytest.mark.asyncio
    async def test_analyze_technical_debt_error_handling(self):
        """Test error handling in analyze_technical_debt."""
        self.mock_rag_pipeline.analyze_technical_debt.side_effect = Exception("Analysis failed")

        # Execute
        result = await self.tools.analyze_technical_debt("backend")

        # Verify error handling
        assert result.status == "error"
        assert result.error == "Analysis failed"

    @pytest.mark.asyncio
    async def test_detect_scope_drift_success(self):
        """Test successful scope drift detection."""
        mock_result = {
            "formatted_context": "Epic scope has expanded significantly",
            "epic_context": {"epic_key": "EPIC-123", "summary": "User Management"},
            "jira_keys": ["STORY-1", "STORY-2"],
            "results_count": 2,
            "child_count": 5,
        }
        self.mock_rag_pipeline.search_by_epic.return_value = mock_result

        # Execute
        result = await self.tools.detect_scope_drift("EPIC-123")

        # Verify
        assert result.status == "success"
        assert result.data["epic_key"] == "EPIC-123"
        assert result.data["scope_analysis"] == "Epic scope has expanded significantly"
        assert result.data["epic_context"]["epic_key"] == "EPIC-123"
        assert result.data["related_items"] == ["STORY-1", "STORY-2"]
        assert result.metadata["total_related_items"] == 2
        assert result.metadata["child_count"] == 5

        # Verify pipeline was called correctly
        self.mock_rag_pipeline.search_by_epic.assert_called_once_with("EPIC-123", query="")

    @pytest.mark.asyncio
    async def test_detect_scope_drift_error_handling(self):
        """Test error handling in detect_scope_drift."""
        self.mock_rag_pipeline.search_by_epic.side_effect = Exception("Epic not found")

        # Execute
        result = await self.tools.detect_scope_drift("INVALID-EPIC")

        # Verify error handling
        assert result.status == "error"
        assert result.error == "Epic not found"

    @pytest.mark.asyncio
    async def test_map_dependencies_with_epic_and_team(self):
        """Test dependency mapping with both epic and team specified."""
        mock_result = {
            "formatted_context": "Found dependencies and blockers",
            "jira_keys": ["BLOCKED-1", "BLOCKED-2"],
            "results_count": 2,
            "filters_applied": {"component": "frontend", "status": "blocked"},
        }
        self.mock_rag_pipeline.process_query.return_value = mock_result

        # Execute
        result = await self.tools.map_dependencies("EPIC-456", "frontend")

        # Verify
        assert result.status == "success"
        assert result.data["epic_key"] == "EPIC-456"
        assert result.data["team"] == "frontend"
        assert result.data["dependency_map"] == "Found dependencies and blockers"
        assert result.data["blocked_items"] == ["BLOCKED-1", "BLOCKED-2"]
        assert result.metadata["total_dependencies"] == 2

        # Verify the query was built correctly
        expected_query = "blocked depends dependency waiting blocker EPIC-456 frontend"
        self.mock_rag_pipeline.process_query.assert_called_once_with(
            user_query=expected_query, include_hierarchy=True
        )

    @pytest.mark.asyncio
    async def test_map_dependencies_epic_only(self):
        """Test dependency mapping with only epic specified."""
        mock_result = {
            "formatted_context": "Epic dependencies",
            "jira_keys": ["EPIC-DEP-1"],
            "results_count": 1,
            "filters_applied": {},
        }
        self.mock_rag_pipeline.process_query.return_value = mock_result

        # Execute
        result = await self.tools.map_dependencies(epic_key="EPIC-789")

        # Verify
        assert result.status == "success"
        assert result.data["epic_key"] == "EPIC-789"
        assert result.data["team"] is None

        # Verify the query was built correctly (no team)
        expected_query = "blocked depends dependency waiting blocker EPIC-789"
        self.mock_rag_pipeline.process_query.assert_called_once_with(
            user_query=expected_query, include_hierarchy=True
        )

    @pytest.mark.asyncio
    async def test_map_dependencies_team_only(self):
        """Test dependency mapping with only team specified."""
        mock_result = {
            "formatted_context": "Team dependencies",
            "jira_keys": ["TEAM-DEP-1"],
            "results_count": 1,
            "filters_applied": {},
        }
        self.mock_rag_pipeline.process_query.return_value = mock_result

        # Execute
        result = await self.tools.map_dependencies(team="backend")

        # Verify
        assert result.status == "success"
        assert result.data["epic_key"] is None
        assert result.data["team"] == "backend"

        # Verify the query was built correctly (no epic)
        expected_query = "blocked depends dependency waiting blocker backend"
        self.mock_rag_pipeline.process_query.assert_called_once_with(
            user_query=expected_query, include_hierarchy=True
        )

    @pytest.mark.asyncio
    async def test_map_dependencies_no_params(self):
        """Test dependency mapping with no parameters."""
        mock_result = {
            "formatted_context": "General dependencies",
            "jira_keys": ["GENERAL-DEP-1"],
            "results_count": 1,
            "filters_applied": {},
        }
        self.mock_rag_pipeline.process_query.return_value = mock_result

        # Execute
        result = await self.tools.map_dependencies()

        # Verify
        assert result.status == "success"
        assert result.data["epic_key"] is None
        assert result.data["team"] is None

        # Verify the query was built correctly (base query only)
        expected_query = "blocked depends dependency waiting blocker"
        self.mock_rag_pipeline.process_query.assert_called_once_with(
            user_query=expected_query, include_hierarchy=True
        )

    @pytest.mark.asyncio
    async def test_map_dependencies_error_handling(self):
        """Test error handling in map_dependencies."""
        self.mock_rag_pipeline.process_query.side_effect = Exception("Dependency analysis failed")

        # Execute
        result = await self.tools.map_dependencies("EPIC-999", "qa")

        # Verify error handling
        assert result.status == "error"
        assert result.error == "Dependency analysis failed"

    def test_tools_initialization(self):
        """Test proper initialization of JiraScopeMCPTools."""
        assert self.tools.rag_pipeline is self.mock_rag_pipeline
        assert isinstance(self.tools, JiraScopeMCPTools)

    @pytest.mark.asyncio
    async def test_empty_response_handling(self):
        """Test handling of empty responses from the pipeline."""
        # Test with empty/minimal response
        empty_result = {}
        self.mock_rag_pipeline.process_query.return_value = empty_result

        result = await self.tools.search_jira_issues("empty query")

        # Verify defaults are handled correctly
        assert result.status == "success"
        assert result.data["results"] == ""  # default for missing formatted_context
        assert result.data["jira_keys"] == []  # default for missing jira_keys
        assert result.metadata["total_items"] == 0  # default for missing results_count

    @pytest.mark.asyncio
    async def test_missing_keys_in_response(self):
        """Test handling of responses with missing expected keys."""
        # Test with partial response missing some keys
        partial_result = {
            "formatted_context": "Some results",
            # Missing jira_keys, results_count, etc.
        }
        self.mock_rag_pipeline.analyze_technical_debt.return_value = partial_result

        result = await self.tools.analyze_technical_debt("frontend")

        # Verify missing keys are handled with defaults
        assert result.status == "success"
        assert result.data["analysis"] == "Some results"
        assert result.data["patterns"] == []  # default for missing jira_keys
        assert result.metadata["total_debt_items"] == 0  # default for missing debt_items_found
