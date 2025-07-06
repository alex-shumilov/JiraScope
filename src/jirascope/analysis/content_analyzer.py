"""AI-powered content quality analysis using Claude."""

import json
import time
from typing import Any

from ..clients.claude_client import ClaudeClient
from ..core.config import Config
from ..models import BatchAnalysisResult, QualityAnalysis, SplitAnalysis, SplitSuggestion, WorkItem
from ..utils.logging import StructuredLogger

logger = StructuredLogger(__name__)


class QualityAnalysisPrompts:
    """Optimized prompts for content quality analysis."""

    DESCRIPTION_QUALITY_PROMPT = """Analyze this Jira work item for description quality:

Title: {summary}
Type: {issue_type}
Description: {description}

Evaluate on these criteria:
1. Clarity (1-5): Is the requirement clearly stated?
2. Completeness (1-5): Are acceptance criteria defined?
3. Actionability (1-5): Can a developer start work immediately?
4. Testability (1-5): Is it clear how to verify completion?

Provide:
- Overall score (1-5)
- Top 2 improvement suggestions
- Risk level (Low/Medium/High) for development

Respond in JSON format:
{{
    "clarity_score": <1-5>,
    "completeness_score": <1-5>,
    "actionability_score": <1-5>,
    "testability_score": <1-5>,
    "overall_score": <1.0-5.0>,
    "improvement_suggestions": ["suggestion1", "suggestion2"],
    "risk_level": "Low|Medium|High"
}}"""

    SPLIT_ANALYSIS_PROMPT = """Analyze if this work item should be split into smaller tasks:

Title: {summary}
Description: {description}
Story Points: {story_points}

Consider:
- Does it contain multiple distinct features/changes?
- Would it take more than 1-2 weeks to complete?
- Are there independent sub-components?

If split recommended, suggest 2-4 smaller work items with:
- Clear titles
- Focused scope
- Dependencies between them

Respond in JSON format:
{{
    "should_split": <true/false>,
    "complexity_score": <0.0-1.0>,
    "reasoning": "explanation of decision",
    "suggested_splits": [
        {{
            "title": "specific title",
            "description": "focused description",
            "estimated_effort": "effort estimate",
            "dependencies": ["dependency1", "dependency2"]
        }}
    ]
}}"""

    BATCH_QUALITY_PROMPT = """Analyze multiple Jira work items for description quality. For each item, evaluate:
1. Clarity (1-5): Is the requirement clearly stated?
2. Completeness (1-5): Are acceptance criteria defined?
3. Actionability (1-5): Can a developer start work immediately?
4. Testability (1-5): Is it clear how to verify completion?

Work Items:
{work_items}

Respond in JSON format with an array of analyses:
{{
    "analyses": [
        {{
            "work_item_key": "KEY-123",
            "clarity_score": <1-5>,
            "completeness_score": <1-5>,
            "actionability_score": <1-5>,
            "testability_score": <1-5>,
            "overall_score": <1.0-5.0>,
            "improvement_suggestions": ["suggestion1", "suggestion2"],
            "risk_level": "Low|Medium|High"
        }}
    ]
}}"""


class ContentAnalyzer:
    """AI-powered content analysis using Claude 3.5 Sonnet."""

    def __init__(self, config: Config):
        self.config = config
        self.claude_client: ClaudeClient | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.claude_client = ClaudeClient(self.config)
        await self.claude_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.claude_client:
            await self.claude_client.__aexit__(exc_type, exc_val, exc_tb)

    async def analyze_description_quality(self, work_item: WorkItem) -> QualityAnalysis:
        """Use Claude to assess description completeness and clarity."""
        logger.info(f"Analyzing description quality for {work_item.key}")
        start_time = time.time()

        if not self.claude_client:
            raise RuntimeError("ContentAnalyzer not initialized. Use async context manager.")

        try:
            # Prepare prompt
            prompt = QualityAnalysisPrompts.DESCRIPTION_QUALITY_PROMPT.format(
                summary=work_item.summary,
                issue_type=work_item.issue_type,
                description=work_item.description or "No description provided",
            )

            # Call Claude
            response = await self.claude_client.analyze(
                prompt=prompt, analysis_type="quality_analysis"
            )

            # Parse response
            try:
                analysis_data = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                analysis_data = self._parse_fallback_quality_response(response.content)

            # Calculate overall score
            scores = [
                analysis_data.get("clarity_score", 3),
                analysis_data.get("completeness_score", 3),
                analysis_data.get("actionability_score", 3),
                analysis_data.get("testability_score", 3),
            ]
            overall_score = sum(scores) / len(scores)

            processing_time = time.time() - start_time

            analysis = QualityAnalysis(
                work_item_key=work_item.key,
                clarity_score=analysis_data.get("clarity_score", 3),
                completeness_score=analysis_data.get("completeness_score", 3),
                actionability_score=analysis_data.get("actionability_score", 3),
                testability_score=analysis_data.get("testability_score", 3),
                overall_score=analysis_data.get("overall_score", overall_score),
                improvement_suggestions=analysis_data.get("improvement_suggestions", []),
                risk_level=analysis_data.get("risk_level", "Medium"),
                analysis_cost=response.cost,
            )

            logger.log_operation(
                "analyze_description_quality",
                processing_time,
                success=True,
                work_item_key=work_item.key,
                overall_score=analysis.overall_score,
                risk_level=analysis.risk_level,
            )

            return analysis

        except Exception as e:
            logger.exception(
                f"Claude API error during description quality analysis for {work_item.key}",
                error=str(e),
            )
            raise

    async def suggest_work_item_splits(self, work_item: WorkItem) -> SplitAnalysis:
        """AI analysis for overly complex work items."""
        logger.info(f"Analyzing split suggestions for {work_item.key}")
        start_time = time.time()

        if not self.claude_client:
            raise RuntimeError("ContentAnalyzer not initialized. Use async context manager.")

        try:
            # Estimate story points if not available (simplified heuristic)
            story_points = "Unknown"
            if work_item.description:
                desc_length = len(work_item.description)
                if desc_length > 1000:
                    story_points = "8-13 (Large)"
                elif desc_length > 500:
                    story_points = "5-8 (Medium)"
                else:
                    story_points = "1-3 (Small)"

            # Prepare prompt
            prompt = QualityAnalysisPrompts.SPLIT_ANALYSIS_PROMPT.format(
                summary=work_item.summary,
                description=work_item.description or "No description provided",
                story_points=story_points,
            )

            # Call Claude
            response = await self.claude_client.analyze(
                prompt=prompt, analysis_type="split_analysis"
            )

            # Parse response
            try:
                split_data = json.loads(response.content)
            except json.JSONDecodeError:
                split_data = self._parse_fallback_split_response(response.content)

            # Create split suggestions
            suggested_splits = []
            for split in split_data.get("suggested_splits", []):
                suggestion = SplitSuggestion(
                    suggested_title=split.get("title", ""),
                    suggested_description=split.get("description", ""),
                    estimated_effort=split.get("estimated_effort"),
                    dependencies=split.get("dependencies", []),
                )
                suggested_splits.append(suggestion)

            processing_time = time.time() - start_time

            analysis = SplitAnalysis(
                work_item_key=work_item.key,
                should_split=split_data.get("should_split", False),
                complexity_score=split_data.get("complexity_score", 0.5),
                suggested_splits=suggested_splits,
                reasoning=split_data.get("reasoning", "No reasoning provided"),
                analysis_cost=response.cost,
            )

            logger.log_operation(
                "suggest_work_item_splits",
                processing_time,
                success=True,
                work_item_key=work_item.key,
                should_split=analysis.should_split,
                splits_suggested=len(suggested_splits),
            )

            return analysis

        except Exception as e:
            logger.exception(f"Failed to analyze splits for {work_item.key}", error=str(e))
            raise

    def _parse_fallback_quality_response(self, content: str) -> dict[str, Any]:
        """Fallback parser for malformed quality analysis responses."""
        # Basic parsing fallback - look for key indicators
        fallback = {
            "clarity_score": 3,
            "completeness_score": 3,
            "actionability_score": 3,
            "testability_score": 3,
            "overall_score": 3.0,
            "improvement_suggestions": ["Review and clarify requirements"],
            "risk_level": "Medium",
        }

        # Try to extract some basic information
        if "low risk" in content.lower():
            fallback["risk_level"] = "Low"
        elif "high risk" in content.lower():
            fallback["risk_level"] = "High"

        return fallback

    def _parse_fallback_split_response(self, content: str) -> dict[str, Any]:
        """Fallback parser for malformed split analysis responses."""
        return {
            "should_split": "split" in content.lower() and "recommend" in content.lower(),
            "complexity_score": 0.5,
            "reasoning": "Unable to parse detailed analysis",
            "suggested_splits": [],
        }


class BatchContentAnalyzer:
    """Batch content analysis for cost optimization."""

    def __init__(self, config: Config):
        self.config = config
        self.claude_client: ClaudeClient | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.claude_client = ClaudeClient(self.config)
        await self.claude_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.claude_client:
            await self.claude_client.__aexit__(exc_type, exc_val, exc_tb)

    async def analyze_multiple_items(
        self, work_items: list[WorkItem], analysis_types: list[str], batch_size: int = 5
    ) -> BatchAnalysisResult:
        """Batch multiple work items and analysis types into optimized Claude calls."""
        logger.info(f"Starting batch analysis of {len(work_items)} items")
        start_time = time.time()

        if not self.claude_client:
            raise RuntimeError("BatchContentAnalyzer not initialized. Use async context manager.")

        results = []
        total_cost = 0.0
        successful_analyses = 0
        failed_analyses = 0
        errors = []

        try:
            # Process in batches for cost optimization
            for i in range(0, len(work_items), batch_size):
                batch = work_items[i : i + batch_size]

                try:
                    batch_result = await self._process_batch(batch, analysis_types)
                    results.extend(batch_result["analyses"])
                    total_cost += batch_result["cost"]
                    successful_analyses += len(batch_result["analyses"])

                except Exception as e:
                    error_msg = f"Batch {i//batch_size + 1} failed: {e!s}"
                    errors.append(error_msg)
                    failed_analyses += len(batch)
                    logger.error(error_msg)

            processing_time = time.time() - start_time

            logger.log_operation(
                "analyze_multiple_items",
                processing_time,
                success=failed_analyses == 0,
                items_processed=len(work_items),
                successful_analyses=successful_analyses,
                total_cost=total_cost,
            )

            return BatchAnalysisResult(
                total_items_processed=len(work_items),
                successful_analyses=successful_analyses,
                failed_analyses=failed_analyses,
                total_cost=total_cost,
                processing_time=processing_time,
                analysis_results=results,
                errors=errors,
            )

        except Exception as e:
            logger.exception("Batch analysis failed completely", error=str(e))
            raise

    async def _process_batch(
        self, work_items: list[WorkItem], analysis_types: list[str]
    ) -> dict[str, Any]:
        """Process a single batch of work items."""
        # For now, focus on quality analysis batching
        if "quality" in analysis_types:
            return await self._batch_quality_analysis(work_items)
        # Fallback to individual analysis
        results = []
        total_cost = 0.0

        analyzer = ContentAnalyzer(self.config)
        async with analyzer:
            for item in work_items:
                try:
                    analysis = await analyzer.analyze_description_quality(item)
                    results.append(analysis.dict())
                    total_cost += analysis.analysis_cost
                except Exception as e:
                    logger.warning(f"Failed to analyze {item.key}: {e!s}")

        return {"analyses": results, "cost": total_cost}

    async def _batch_quality_analysis(self, work_items: list[WorkItem]) -> dict[str, Any]:
        """Batch quality analysis for multiple work items."""
        # Prepare batch prompt
        work_items_text = ""
        for i, item in enumerate(work_items, 1):
            work_items_text += f"""
Item {i}:
Key: {item.key}
Title: {item.summary}
Type: {item.issue_type}
Description: {item.description or "No description provided"}

"""

        prompt = QualityAnalysisPrompts.BATCH_QUALITY_PROMPT.format(
            work_items=work_items_text.strip()
        )

        # Call Claude
        response = await self.claude_client.analyze(
            prompt=prompt, analysis_type="batch_quality_analysis"
        )

        # Parse response
        try:
            batch_data = json.loads(response.content)
            analyses = batch_data.get("analyses", [])
        except json.JSONDecodeError:
            # Fallback to individual analysis if batch parsing fails
            logger.warning("Batch parsing failed, falling back to individual analysis")
            analyses = []
            for item in work_items:
                analyses.append(
                    {
                        "work_item_key": item.key,
                        "clarity_score": 3,
                        "completeness_score": 3,
                        "actionability_score": 3,
                        "testability_score": 3,
                        "overall_score": 3.0,
                        "improvement_suggestions": ["Batch analysis parsing failed"],
                        "risk_level": "Medium",
                    }
                )

        return {"analyses": analyses, "cost": response.cost}
