"""Tests for Claude client functionality."""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest

from src.jirascope.clients.claude_client import ClaudeClient
from src.jirascope.core.config import Config
from src.jirascope.models import AnalysisResult, WorkItem


class TestClaudeClient:
    """Test Claude client functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config(
            jira_mcp_endpoint="https://test.atlassian.net",
            claude_api_key="test_api_key",
            claude_model="claude-3-sonnet-20240229",
            claude_max_tokens=4000,
            claude_temperature=0.1,
            claude_input_cost_per_token=0.000003,
            claude_output_cost_per_token=0.000015,
            claude_session_budget=10.0,
        )
        self.client = ClaudeClient(self.config)

        # Create test work item
        self.work_item = WorkItem(
            key="TEST-123",
            summary="Implement user authentication",
            description="Add OAuth2 authentication to the application",
            parent_key=None,
            epic_key=None,
            issue_type="Story",
            status="Open",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
            reporter="test@example.com",
            assignee="dev@example.com",
            components=["Backend", "Security"],
            labels=["authentication", "security"],
            embedding=[0.1, 0.2, 0.3] * 100,
        )

    def test_claude_client_initialization(self):
        """Test Claude client initialization."""
        assert self.client.config == self.config
        assert self.client.session_cost == 0.0
        assert self.client.client is not None

    @pytest.mark.asyncio
    async def test_context_manager_methods(self):
        """Test async context manager functionality."""
        async with self.client as client:
            assert client == self.client
        # No exception should be raised

    def test_calculate_cost(self):
        """Test cost calculation logic."""
        input_tokens = 1000
        output_tokens = 500

        expected_cost = (
            input_tokens * self.config.claude_input_cost_per_token
            + output_tokens * self.config.claude_output_cost_per_token
        )

        actual_cost = self.client.calculate_cost(input_tokens, output_tokens)
        assert actual_cost == expected_cost
        assert actual_cost > 0

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        cost = self.client.calculate_cost(0, 0)
        assert cost == 0.0

    def test_session_cost_management(self):
        """Test session cost tracking."""
        assert self.client.get_session_cost() == 0.0

        # Simulate adding cost
        self.client.session_cost = 5.25
        assert self.client.get_session_cost() == 5.25

        # Reset cost
        self.client.reset_session_cost()
        assert self.client.get_session_cost() == 0.0

    def test_build_analysis_prompt_general(self):
        """Test building general analysis prompt."""
        prompt = self.client._build_analysis_prompt(self.work_item, "general")

        assert "TEST-123" in prompt
        assert "Implement user authentication" in prompt
        assert "OAuth2 authentication" in prompt
        assert "Story" in prompt
        assert "Open" in prompt
        assert "Backend" in prompt
        assert "Security" in prompt
        assert "authentication" in prompt
        assert "clarity_score" in prompt
        assert "completeness_score" in prompt

    def test_build_analysis_prompt_complexity(self):
        """Test building complexity analysis prompt."""
        prompt = self.client._build_analysis_prompt(self.work_item, "complexity")

        assert "TEST-123" in prompt
        assert "technical_complexity" in prompt
        assert "business_complexity" in prompt
        assert "risk_level" in prompt
        assert "dependencies" in prompt
        assert "effort_estimate" in prompt

    def test_build_analysis_prompt_similarity(self):
        """Test building similarity analysis prompt."""
        prompt = self.client._build_analysis_prompt(self.work_item, "similarity")

        assert "TEST-123" in prompt
        assert "similarity_indicators" in prompt
        assert "duplicate_risk" in prompt
        assert "recommended_actions" in prompt

    def test_build_analysis_prompt_with_context(self):
        """Test building prompt with context items."""
        context_item = WorkItem(
            key="CONTEXT-1",
            summary="Related authentication feature",
            description="Context item",
            parent_key=None,
            epic_key=None,
            issue_type="Task",
            status="Done",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
            assignee=None,
            reporter="test@example.com",
            embedding=[0.1, 0.2, 0.3] * 100,
        )

        prompt = self.client._build_analysis_prompt(self.work_item, "general", [context_item])

        assert "TEST-123" in prompt
        assert "Related work items for context" in prompt
        assert "CONTEXT-1" in prompt
        assert "Related authentication feature" in prompt

    def test_build_analysis_prompt_with_multiple_context(self):
        """Test building prompt with multiple context items (should limit to 3)."""
        context_items = [
            WorkItem(
                key=f"CONTEXT-{i}",
                summary=f"Context item {i}",
                description="Context",
                parent_key=None,
                epic_key=None,
                issue_type="Task",
                status="Done",
                created=datetime.now(UTC),
                updated=datetime.now(UTC),
                assignee=None,
                reporter="test@example.com",
                embedding=[0.1, 0.2, 0.3] * 100,
            )
            for i in range(5)  # Create 5 context items
        ]

        prompt = self.client._build_analysis_prompt(self.work_item, "general", context_items)

        # Should only include first 3 context items
        assert "CONTEXT-0" in prompt
        assert "CONTEXT-1" in prompt
        assert "CONTEXT-2" in prompt
        assert "CONTEXT-3" not in prompt
        assert "CONTEXT-4" not in prompt

    def test_parse_analysis_response_valid_json(self):
        """Test parsing valid JSON response."""
        response = """
        Some text before JSON
        {
            "clarity_score": 8,
            "completeness_score": 7,
            "confidence": 0.85,
            "reasoning": "Well defined requirements"
        }
        Some text after JSON
        """

        result = self.client._parse_analysis_response(response, "general")

        assert result["clarity_score"] == 8
        assert result["completeness_score"] == 7
        assert result["confidence"] == 0.85
        assert result["reasoning"] == "Well defined requirements"

    def test_parse_analysis_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        response = "This is not valid JSON"

        result = self.client._parse_analysis_response(response, "general")

        assert result["raw_response"] == response
        assert result["confidence"] == 0.5
        assert result["reasoning"] == "Failed to parse structured response"

    def test_parse_analysis_response_no_json(self):
        """Test parsing response with no JSON."""
        response = "This response has no JSON brackets"

        result = self.client._parse_analysis_response(response, "general")

        assert result["raw_response"] == response
        assert result["confidence"] == 0.5
        assert result["reasoning"] == "Failed to parse structured response"

    def test_parse_analysis_response_malformed_json(self):
        """Test parsing response with malformed JSON."""
        response = """
        {
            "clarity_score": 8,
            "completeness_score": 7,
            "confidence": 0.85,
            "reasoning": "Missing closing brace"
        """

        result = self.client._parse_analysis_response(response, "general")

        assert result["raw_response"] == response
        assert result["confidence"] == 0.5
        assert result["reasoning"] == "Invalid JSON response"

    @pytest.mark.asyncio
    async def test_analyze_work_item_success(self):
        """Test successful work item analysis."""
        # Mock Anthropic client
        mock_response = Mock()
        mock_response.content = [Mock(text='{"confidence": 0.8, "reasoning": "Test analysis"}')]
        mock_response.usage.input_tokens = 1000
        mock_response.usage.output_tokens = 500

        with patch.object(self.client, "client") as mock_anthropic:
            mock_anthropic.messages.create.return_value = mock_response

            result = await self.client.analyze_work_item(self.work_item)

            assert isinstance(result, AnalysisResult)
            assert result.work_item_key == "TEST-123"
            assert result.analysis_type == "general"
            assert result.confidence == 0.8
            assert result.cost > 0
            assert self.client.session_cost > 0

    @pytest.mark.asyncio
    async def test_analyze_work_item_complexity_type(self):
        """Test work item analysis with complexity type."""
        mock_response = Mock()
        mock_response.content = [Mock(text='{"technical_complexity": 7, "confidence": 0.9}')]
        mock_response.usage.input_tokens = 1000
        mock_response.usage.output_tokens = 500

        with patch.object(self.client, "client") as mock_anthropic:
            mock_anthropic.messages.create.return_value = mock_response

            result = await self.client.analyze_work_item(self.work_item, analysis_type="complexity")

            assert result.analysis_type == "complexity"
            assert result.insights["technical_complexity"] == 7

    @pytest.mark.asyncio
    async def test_analyze_work_item_with_context(self):
        """Test work item analysis with context."""
        context_item = WorkItem(
            key="CONTEXT-1",
            summary="Related item",
            description="Context",
            issue_type="Task",
            status="Done",
            parent_key=None,
            epic_key=None,
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
            assignee=None,
            reporter="test@example.com",
            embedding=[0.1, 0.2, 0.3] * 100,
        )

        mock_response = Mock()
        mock_response.content = [Mock(text='{"confidence": 0.8, "reasoning": "Test analysis"}')]
        mock_response.usage.input_tokens = 1000
        mock_response.usage.output_tokens = 500

        with patch.object(self.client, "client") as mock_anthropic:
            mock_anthropic.messages.create.return_value = mock_response

            result = await self.client.analyze_work_item(self.work_item, context=[context_item])

            assert isinstance(result, AnalysisResult)
            # Verify context was included in prompt
            mock_anthropic.messages.create.assert_called_once()
            call_args = mock_anthropic.messages.create.call_args
            prompt = call_args[1]["messages"][0]["content"]
            assert "CONTEXT-1" in prompt

    @pytest.mark.asyncio
    async def test_analyze_work_item_budget_exceeded(self):
        """Test analysis when budget is exceeded."""
        # Set session cost to exceed budget
        self.client.session_cost = 15.0  # Exceeds 10.0 budget

        with pytest.raises(ValueError, match="Session budget.*exceeded"):
            await self.client.analyze_work_item(self.work_item)

    @pytest.mark.asyncio
    async def test_analyze_work_item_api_error(self):
        """Test analysis when API call fails."""
        with patch.object(self.client, "client") as mock_anthropic:
            mock_anthropic.messages.create.side_effect = Exception("API Error")

            with pytest.raises(Exception, match="API Error"):
                await self.client.analyze_work_item(self.work_item)

    @pytest.mark.asyncio
    async def test_analyze_generic_method_success(self):
        """Test generic analyze method."""
        mock_response = Mock()
        mock_response.content = [Mock(text='{"confidence": 0.7, "result": "Generic analysis"}')]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 250

        with patch.object(self.client, "client") as mock_anthropic:
            mock_anthropic.messages.create.return_value = mock_response

            # Create a simple response class to match the implementation
            class SimpleResponse:
                def __init__(self, content: str, cost: float):
                    self.content = content
                    self.cost = cost

            with patch.object(self.client, "_parse_analysis_response") as mock_parse:
                mock_parse.return_value = {"confidence": 0.7, "result": "Generic analysis"}

                result = await self.client.analyze("Test prompt", "custom")

                # The analyze method returns a SimpleResponse object
                assert hasattr(result, "content")
                assert hasattr(result, "cost")

    @pytest.mark.asyncio
    async def test_analyze_generic_method_budget_exceeded(self):
        """Test generic analyze method when budget is exceeded."""
        self.client.session_cost = 15.0  # Exceeds budget

        with pytest.raises(ValueError, match="Session budget.*exceeded"):
            await self.client.analyze("Test prompt")

    def test_cost_tracking_across_multiple_calls(self):
        """Test that costs accumulate across multiple API calls."""
        # Simulate multiple cost additions
        initial_cost = self.client.session_cost

        cost1 = self.client.calculate_cost(1000, 500)
        self.client.session_cost += cost1

        cost2 = self.client.calculate_cost(800, 300)
        self.client.session_cost += cost2

        expected_total = initial_cost + cost1 + cost2
        assert self.client.get_session_cost() == expected_total

    def test_work_item_with_minimal_data(self):
        """Test analysis with work item that has minimal data."""
        minimal_item = WorkItem(
            key="MIN-1",
            summary="Minimal item",
            description=None,
            parent_key=None,
            epic_key=None,
            issue_type="Task",
            status="Open",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
            assignee=None,
            reporter="test@example.com",
            embedding=[0.1, 0.2, 0.3] * 100,
        )

        prompt = self.client._build_analysis_prompt(minimal_item, "general")

        assert "MIN-1" in prompt
        assert "Minimal item" in prompt
        assert "No description" in prompt
        assert "Task" in prompt
        assert "None" in prompt  # For components and labels

    def test_work_item_with_empty_collections(self):
        """Test analysis with work item that has empty components/labels."""
        item_with_empty = WorkItem(
            key="EMPTY-1",
            summary="Item with empty collections",
            description="Test description",
            parent_key=None,
            epic_key=None,
            issue_type="Bug",
            status="In Progress",
            created=datetime.now(UTC),
            updated=datetime.now(UTC),
            assignee=None,
            reporter="test@example.com",
            components=[],  # Empty list
            labels=[],  # Empty list
            embedding=[0.1, 0.2, 0.3] * 100,
        )

        prompt = self.client._build_analysis_prompt(item_with_empty, "general")

        assert "EMPTY-1" in prompt
        assert "Components: None" in prompt
        assert "Labels: None" in prompt
