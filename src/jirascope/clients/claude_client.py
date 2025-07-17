"""Claude client for AI analysis."""

import logging
from typing import Any

from anthropic import Anthropic

from ..core.config import Config
from ..models import AnalysisResult, WorkItem

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Client for AI analysis using Claude."""

    def __init__(self, config: Config):
        self.config = config
        self.client = Anthropic(api_key=config.claude_api_key)
        self.session_cost = 0.0

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # No cleanup needed for Anthropic client

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost of a Claude API call."""
        input_cost = input_tokens * self.config.claude_input_cost_per_token
        output_cost = output_tokens * self.config.claude_output_cost_per_token
        return input_cost + output_cost

    async def analyze_work_item(
        self,
        work_item: WorkItem,
        analysis_type: str = "general",
        context: list[WorkItem] | None = None,
    ) -> AnalysisResult:
        """Analyze a work item using Claude."""

        # Check budget constraints
        if self.session_cost >= self.config.claude_session_budget:
            raise ValueError(f"Session budget of ${self.config.claude_session_budget} exceeded")

        try:
            prompt = self._build_analysis_prompt(work_item, analysis_type, context)

            response = self.client.messages.create(
                model=self.config.claude_model,
                max_tokens=self.config.claude_max_tokens,
                temperature=self.config.claude_temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            # Calculate cost
            cost = self.calculate_cost(response.usage.input_tokens, response.usage.output_tokens)
            self.session_cost += cost

            # Parse response
            insights = self._parse_analysis_response(response.content[0].text, analysis_type)

            result = AnalysisResult(
                work_item_key=work_item.key,
                analysis_type=analysis_type,
                confidence=insights.get("confidence", 0.8),
                insights=insights,
                cost=cost,
            )

            logger.info(f"Analyzed {work_item.key} ({analysis_type}) - Cost: ${cost:.4f}")
            return result

        except Exception as e:
            logger.exception(f"Failed to analyze work item {work_item.key}: {e}")
            raise

    def _build_analysis_prompt(
        self, work_item: WorkItem, analysis_type: str, context: list[WorkItem] | None = None
    ) -> str:
        """Build the analysis prompt for Claude."""
        base_prompt = f"""
Analyze this Jira work item:

Key: {work_item.key}
Summary: {work_item.summary}
Description: {work_item.description or 'No description'}
Type: {work_item.issue_type}
Status: {work_item.status}
Components: {', '.join(work_item.components) if work_item.components else 'None'}
Labels: {', '.join(work_item.labels) if work_item.labels else 'None'}
"""

        if context:
            base_prompt += "\n\nRelated work items for context:\n"
            for item in context[:3]:  # Limit context to avoid token limits
                base_prompt += f"- {item.key}: {item.summary}\n"

        if analysis_type == "complexity":
            return (
                base_prompt
                + """
Analyze the complexity of this work item. Consider:
1. Technical complexity (1-10 scale)
2. Business complexity (1-10 scale)
3. Risk factors
4. Dependencies
5. Estimated effort

Respond in JSON format with these fields:
- technical_complexity: number
- business_complexity: number
- risk_level: "low"|"medium"|"high"
- dependencies: array of strings
- effort_estimate: string
- confidence: number (0-1)
- reasoning: string
"""
            )

        if analysis_type == "similarity":
            return (
                base_prompt
                + """
Analyze this work item for potential duplicates or highly similar items. Consider:
1. Functional similarity
2. Technical overlap
3. Scope similarity

Respond in JSON format with these fields:
- similarity_indicators: array of strings
- duplicate_risk: "low"|"medium"|"high"
- recommended_actions: array of strings
- confidence: number (0-1)
- reasoning: string
"""
            )

        # general analysis
        return (
            base_prompt
            + """
Provide a general analysis of this work item. Consider:
1. Clarity and completeness of requirements
2. Potential issues or risks
3. Suggestions for improvement
4. Priority assessment

Respond in JSON format with these fields:
- clarity_score: number (1-10)
- completeness_score: number (1-10)
- risk_factors: array of strings
- suggestions: array of strings
- priority_recommendation: "low"|"medium"|"high"|"critical"
- confidence: number (0-1)
- reasoning: string
"""
        )

    def _parse_analysis_response(self, response: str, analysis_type: str) -> dict[str, Any]:
        """Parse Claude's analysis response."""
        try:
            import json

            # Try to extract JSON from the response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            # Fallback: return raw response
            return {
                "raw_response": response,
                "confidence": 0.5,
                "reasoning": "Failed to parse structured response",
            }

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response for {analysis_type}")
            return {
                "raw_response": response,
                "confidence": 0.5,
                "reasoning": "Invalid JSON response",
            }

    def get_session_cost(self) -> float:
        """Get the total cost for this session."""
        return self.session_cost

    def reset_session_cost(self):
        """Reset the session cost counter."""
        self.session_cost = 0.0

    async def analyze(self, prompt: str, analysis_type: str = "general") -> AnalysisResult:
        """Generic analysis method using Claude."""
        # Check budget constraints
        if self.session_cost >= self.config.claude_session_budget:
            raise ValueError(f"Session budget of ${self.config.claude_session_budget} exceeded")

        try:
            response = self.client.messages.create(
                model=self.config.claude_model,
                max_tokens=self.config.claude_max_tokens,
                temperature=self.config.claude_temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            # Calculate cost
            cost = self.calculate_cost(response.usage.input_tokens, response.usage.output_tokens)
            self.session_cost += cost

            # Create a simple response object
            class SimpleResponse:
                def __init__(self, content: str, cost: float):
                    self.content = content
                    self.cost = cost

            result = SimpleResponse(content=response.content[0].text, cost=cost)

            logger.info(f"Analyzed prompt ({analysis_type}) - Cost: ${cost:.4f}")
            return result

        except Exception as e:
            logger.exception(f"Failed to analyze prompt: {e}")
            raise
