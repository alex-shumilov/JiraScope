"""Advanced cost optimization and analysis for JiraScope."""

import statistics
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from ..core.config import Config
from ..utils.logging import CostTracker, StructuredLogger

logger = StructuredLogger(__name__)


class APICall(BaseModel):
    """API call record for cost analysis."""

    service: str = Field(..., description="Service name (claude, embeddings, etc.)")
    operation: str = Field(..., description="Operation performed")
    timestamp: str | datetime = Field(..., description="Call timestamp")
    input_tokens: int = Field(0, description="Number of input tokens")
    output_tokens: int = Field(0, description="Number of output tokens")
    cost: float = Field(0.0, description="Cost of this call")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details")


class PromptCategoryAnalysis(BaseModel):
    """Analysis of a prompt category."""

    category: str = Field(..., description="Prompt category")
    total_calls: int = Field(0, description="Total calls in this category")
    avg_input_tokens: float = Field(0.0, description="Average input tokens")
    avg_output_tokens: float = Field(0.0, description="Average output tokens")
    avg_cost: float = Field(0.0, description="Average cost per call")
    expensive_calls: int = Field(0, description="Number of expensive calls")
    optimization_potential: float = Field(0.0, description="Estimated savings potential")


class PromptEfficiencyAnalysis(BaseModel):
    """Analysis of prompt efficiency."""

    categories: list[PromptCategoryAnalysis] = Field(default_factory=list)
    overall_efficiency_score: float = Field(0.0, description="Overall efficiency score (0-1)")


class OptimizationSuggestion(BaseModel):
    """Suggestion for cost optimization."""

    type: str = Field(..., description="Type of optimization")
    current_value: int | float | str = Field(..., description="Current configuration value")
    suggested_value: int | float | str = Field(..., description="Suggested new value")
    potential_savings: float = Field(0.0, description="Estimated monthly savings")
    risk_level: str = Field(..., description="Risk level (low, medium, high)")
    description: str = Field(..., description="Suggestion description")


class BatchOptimizationSuggestions(BaseModel):
    """Collection of batch optimization suggestions."""

    suggestions: list[OptimizationSuggestion] = Field(default_factory=list)
    estimated_total_savings: float = Field(0.0, description="Total estimated monthly savings")


class ClaudeUsageAnalysis(BaseModel):
    """Analysis of Claude API usage patterns."""

    prompt_efficiency: PromptEfficiencyAnalysis = Field(...)
    batch_opportunities: BatchOptimizationSuggestions = Field(...)
    caching_opportunities: dict[str, Any] = Field(default_factory=dict)
    cost_per_analysis_type: dict[str, float] = Field(default_factory=dict)
    total_calls: int = Field(0, description="Total calls analyzed")
    total_cost: float = Field(0.0, description="Total cost of calls analyzed")
    analysis_period_days: int = Field(0, description="Analysis period in days")


class CostTrends(BaseModel):
    """Cost trends over time."""

    daily_costs: list[float] = Field(default_factory=list, description="Daily costs")
    weekly_costs: list[float] = Field(default_factory=list, description="Weekly costs")
    monthly_costs: list[float] = Field(default_factory=list, description="Monthly costs")
    trend_direction: str = Field(
        "stable", description="Trend direction (increasing, stable, decreasing)"
    )
    growth_rate: float = Field(0.0, description="Monthly growth rate")


class CostPredictions(BaseModel):
    """Predicted future costs."""

    next_week: float = Field(0.0, description="Predicted cost for next week")
    next_month: float = Field(0.0, description="Predicted cost for next month")
    confidence: float = Field(0.0, description="Prediction confidence (0-1)")
    growth_rate: float = Field(0.0, description="Predicted growth rate")


class ComprehensiveCostReport(BaseModel):
    """Comprehensive cost report with analysis."""

    period: str = Field(..., description="Report period")
    total_cost: float = Field(0.0, description="Total cost for the period")
    service_breakdown: dict[str, float] = Field(default_factory=dict)
    trends: CostTrends = Field(...)
    predictions: CostPredictions = Field(...)
    optimization_opportunities: list[OptimizationSuggestion] = Field(default_factory=list)
    cost_per_analysis: dict[str, float] = Field(default_factory=dict)
    efficiency_metrics: dict[str, Any] = Field(default_factory=dict)


class BudgetAlert(BaseModel):
    """Budget threshold alert."""

    type: str = Field(..., description="Alert type (daily_budget, monthly_budget)")
    threshold: float = Field(..., description="Threshold percentage")
    current_usage: float = Field(..., description="Current usage percentage")
    severity: str = Field(..., description="Severity level")
    message: str = Field(..., description="Alert message")
    timestamp: datetime = Field(default_factory=datetime.now)


class CostOptimizer:
    """Advanced cost optimization and management."""

    def __init__(self, config: Config, cost_tracker: CostTracker):
        self.config = config
        self.cost_tracker = cost_tracker
        self.usage_analyzer = UsagePatternAnalyzer(config, cost_tracker)

        # Load cost configuration
        self.daily_budget = getattr(config, "daily_budget", 50.0)
        self.monthly_budget = getattr(config, "monthly_budget", 1000.0)

        # Set up alert thresholds
        self.alert_thresholds = [0.5, 0.75, 0.9, 1.0]
        self.sent_alerts: dict[str, datetime] = {}

    async def analyze_api_usage_patterns(self) -> ClaudeUsageAnalysis:
        """Identify opportunities for cost reduction."""
        return await self.usage_analyzer.analyze_claude_usage_patterns()

    async def suggest_batch_optimizations(self) -> BatchOptimizationSuggestions:
        """Recommend better batching strategies."""
        return await self.usage_analyzer.suggest_batch_optimizations()

    async def generate_cost_report(self, period: str = "month") -> ComprehensiveCostReport:
        """Generate detailed cost breakdown with recommendations."""
        logger.info(f"Generating comprehensive cost report for period: {period}")

        # Get detailed cost data
        cost_data = await self._get_detailed_costs(period)

        # Service-specific cost breakdown
        service_costs = await self._get_service_breakdown(cost_data)

        # Analyze cost trends
        trends = await self._analyze_cost_trends(cost_data, period)

        # Generate predictions
        predictions = await self._predict_future_costs(trends)

        # Find optimization opportunities
        optimizations = await self._identify_cost_optimizations(cost_data)

        # Calculate cost per analysis type
        analysis_costs = await self._calculate_cost_per_analysis_type(cost_data)

        # Calculate efficiency metrics
        efficiency_metrics = await self._calculate_efficiency_metrics(cost_data)

        return ComprehensiveCostReport(
            period=period,
            total_cost=sum(service_costs.values()),
            service_breakdown=service_costs,
            trends=trends,
            predictions=predictions,
            optimization_opportunities=optimizations,
            cost_per_analysis=analysis_costs,
            efficiency_metrics=efficiency_metrics,
        )

    async def check_budget_alerts(self) -> list[BudgetAlert]:
        """Check for budget threshold violations."""
        daily_cost = await self._get_daily_cost()
        monthly_cost = await self._get_monthly_cost()

        alerts = []

        # Daily budget alerts
        daily_usage = daily_cost / self.daily_budget if self.daily_budget > 0 else 0
        for threshold in self.alert_thresholds:
            alert_key = f"daily_{threshold}"
            if daily_usage >= threshold and not self._alert_already_sent(alert_key):
                alerts.append(
                    BudgetAlert(
                        type="daily_budget",
                        threshold=threshold,
                        current_usage=daily_usage,
                        severity=self._get_alert_severity(threshold),
                        message=f"Daily budget {threshold*100}% reached: ${daily_cost:.2f} of ${self.daily_budget:.2f}",
                    )
                )
                self._mark_alert_sent(alert_key)

        # Monthly budget alerts
        monthly_usage = monthly_cost / self.monthly_budget if self.monthly_budget > 0 else 0
        for threshold in self.alert_thresholds:
            alert_key = f"monthly_{threshold}"
            if monthly_usage >= threshold and not self._alert_already_sent(alert_key):
                alerts.append(
                    BudgetAlert(
                        type="monthly_budget",
                        threshold=threshold,
                        current_usage=monthly_usage,
                        severity=self._get_alert_severity(threshold),
                        message=f"Monthly budget {threshold*100}% reached: ${monthly_cost:.2f} of ${self.monthly_budget:.2f}",
                    )
                )
                self._mark_alert_sent(alert_key)

        return alerts

    async def reset_daily_alerts(self):
        """Reset daily alerts at the end of the day."""
        for key in list(self.sent_alerts.keys()):
            if key.startswith("daily_"):
                del self.sent_alerts[key]

    async def reset_monthly_alerts(self):
        """Reset monthly alerts at the end of the month."""
        for key in list(self.sent_alerts.keys()):
            if key.startswith("monthly_"):
                del self.sent_alerts[key]

    def _get_alert_severity(self, threshold: float) -> str:
        """Determine alert severity based on threshold."""
        if threshold >= 1.0:
            return "critical"
        elif threshold >= 0.9:
            return "high"
        elif threshold >= 0.75:
            return "medium"
        else:
            return "low"

    def _alert_already_sent(self, alert_key: str) -> bool:
        """Check if an alert has already been sent."""
        return alert_key in self.sent_alerts

    def _mark_alert_sent(self, alert_key: str):
        """Mark an alert as sent."""
        self.sent_alerts[alert_key] = datetime.now()

    async def _get_daily_cost(self) -> float:
        """Get cost for the current day."""
        _ = datetime.now().date()  # today calculated but not used in this simplified implementation

        # In a real implementation, this would filter the cost tracker data
        # For simplicity, we'll estimate based on the total
        if not self.cost_tracker:
            return 0.0

        total = self.cost_tracker.get_total_cost()

        # Simulate by returning a portion of total cost
        return total * 0.1  # 10% of total as daily cost for simulation

    async def _get_monthly_cost(self) -> float:
        """Get cost for the current month."""
        if not self.cost_tracker:
            return 0.0

        # Simply return the total for simulation
        return self.cost_tracker.get_total_cost()

    async def _get_detailed_costs(self, period: str) -> dict[str, Any]:
        """Get detailed cost data for the specified period."""
        if not self.cost_tracker:
            return {
                "claude_calls": [],
                "embedding_operations": [],
                "qdrant_costs": 0.0,
                "jira_api_costs": 0.0,
            }

        # In a real implementation, this would filter by date
        # For now, we'll use all available data
        claude_costs = self.cost_tracker.costs.get("claude", [])
        embedding_costs = self.cost_tracker.costs.get("embeddings", [])

        return {
            "claude_calls": claude_costs,
            "embedding_operations": embedding_costs,
            "qdrant_costs": 0.1 * self.cost_tracker.get_total_cost("embeddings"),  # Estimate
            "jira_api_costs": 0.0,  # Usually free but track for rate limiting
        }

    async def _get_service_breakdown(self, cost_data: dict[str, Any]) -> dict[str, float]:
        """Calculate cost breakdown by service."""
        claude_cost = sum(call.get("cost", 0.0) for call in cost_data.get("claude_calls", []))
        embedding_cost = sum(
            op.get("cost", 0.0) for op in cost_data.get("embedding_operations", [])
        )
        qdrant_cost = cost_data.get("qdrant_costs", 0.0)
        jira_api_cost = cost_data.get("jira_api_costs", 0.0)

        return {
            "claude_api": claude_cost,
            "embedding_processing": embedding_cost,
            "vector_storage": qdrant_cost,
            "jira_api": jira_api_cost,
        }

    async def _analyze_cost_trends(self, cost_data: dict[str, Any], period: str) -> CostTrends:
        """Analyze cost trends over time."""
        # For a simple implementation, we'll use simulated data
        daily_costs = [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 1.8]  # Last 7 days
        weekly_costs = [5.0, 7.0, 9.0, 12.0]  # Last 4 weeks
        monthly_costs = [30.0, 40.0, 50.0]  # Last 3 months

        # Calculate trend direction
        if len(monthly_costs) >= 2:
            recent = monthly_costs[-1]
            previous = monthly_costs[-2]
            growth_rate = (recent - previous) / previous if previous > 0 else 0

            if growth_rate > 0.1:
                trend = "increasing"
            elif growth_rate < -0.1:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
            growth_rate = 0.0

        return CostTrends(
            daily_costs=daily_costs,
            weekly_costs=weekly_costs,
            monthly_costs=monthly_costs,
            trend_direction=trend,
            growth_rate=growth_rate,
        )

    async def _predict_future_costs(self, trends: CostTrends) -> CostPredictions:
        """Predict future costs based on trends."""
        daily_costs = trends.daily_costs
        monthly_growth = trends.growth_rate

        if len(daily_costs) >= 7:
            # Calculate recent average
            recent_avg = statistics.mean(daily_costs[-7:])

            # Project for next week and month
            next_week = recent_avg * 7 * (1 + monthly_growth / 4)  # Weekly growth
            next_month = recent_avg * 30 * (1 + monthly_growth)  # Monthly growth

            confidence = min(0.9, len(daily_costs) / 30)  # More data = higher confidence
        else:
            recent_avg = statistics.mean(daily_costs) if daily_costs else 0
            next_week = recent_avg * 7
            next_month = recent_avg * 30
            confidence = 0.5

        return CostPredictions(
            next_week=next_week,
            next_month=next_month,
            confidence=confidence,
            growth_rate=monthly_growth,
        )

    async def _identify_cost_optimizations(
        self, cost_data: dict[str, Any]
    ) -> list[OptimizationSuggestion]:
        """Identify potential cost optimizations."""
        optimizations = []

        # Check for batch optimization opportunities
        batch_opts = await self.suggest_batch_optimizations()
        optimizations.extend(batch_opts.suggestions)

        # Check for inefficient prompts
        usage_analysis = await self.analyze_api_usage_patterns()

        # Identify high-cost prompts
        for category in usage_analysis.prompt_efficiency.categories:
            if category.optimization_potential > 5.0:  # $5/month savings potential
                optimizations.append(
                    OptimizationSuggestion(
                        type="prompt_optimization",
                        current_value=f"{category.avg_input_tokens:.0f} tokens",
                        suggested_value=f"{int(category.avg_input_tokens * 0.8):.0f} tokens",
                        potential_savings=category.optimization_potential,
                        risk_level="low",
                        description=f"Optimize prompts for {category.category} to reduce token usage",
                    )
                )

        # Check for caching opportunities
        if usage_analysis.caching_opportunities.get("potential_savings", 0.0) > 5.0:
            optimizations.append(
                OptimizationSuggestion(
                    type="implement_caching",
                    current_value="disabled",
                    suggested_value="enabled",
                    potential_savings=usage_analysis.caching_opportunities.get(
                        "potential_savings", 0.0
                    ),
                    risk_level="low",
                    description="Implement result caching for common queries",
                )
            )

        # Add model downgrade suggestion if appropriate
        claude_cost = sum(call.get("cost", 0.0) for call in cost_data.get("claude_calls", []))
        if claude_cost > 50.0:  # Substantial Claude usage
            optimizations.append(
                OptimizationSuggestion(
                    type="model_downgrade",
                    current_value=self.config.claude_model,
                    suggested_value="claude-3-haiku-20240307",
                    potential_savings=claude_cost * 0.4,  # 40% savings estimate
                    risk_level="medium",
                    description="Consider downgrading to Claude 3 Haiku for non-critical analyses",
                )
            )

        return optimizations

    async def _calculate_cost_per_analysis_type(
        self, cost_data: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate costs per analysis type."""
        # Group claude calls by operation type
        claude_calls = cost_data.get("claude_calls", [])
        cost_by_type = {}

        for call in claude_calls:
            operation = call.get("operation", "unknown")
            cost = call.get("cost", 0.0)

            operation_type = operation.split(".")[0] if "." in operation else operation

            if operation_type not in cost_by_type:
                cost_by_type[operation_type] = 0.0

            cost_by_type[operation_type] += cost

        return cost_by_type

    async def _calculate_efficiency_metrics(self, cost_data: dict[str, Any]) -> dict[str, Any]:
        """Calculate cost efficiency metrics."""
        claude_calls = cost_data.get("claude_calls", [])

        # Initialize metrics
        metrics = {
            "avg_cost_per_call": 0.0,
            "cost_per_token": 0.0,
            "token_efficiency_score": 0.0,
            "highest_cost_operation": "",
            "highest_cost_value": 0.0,
        }

        if not claude_calls:
            return metrics

        # Calculate average cost
        total_cost = sum(call.get("cost", 0.0) for call in claude_calls)
        metrics["avg_cost_per_call"] = total_cost / len(claude_calls) if claude_calls else 0.0

        # Calculate token metrics
        total_input_tokens = sum(
            call.get("details", {}).get("input_tokens", 0) for call in claude_calls
        )
        total_output_tokens = sum(
            call.get("details", {}).get("output_tokens", 0) for call in claude_calls
        )
        total_tokens = total_input_tokens + total_output_tokens

        if total_tokens > 0:
            metrics["cost_per_token"] = total_cost / total_tokens
            metrics["token_efficiency_score"] = total_output_tokens / (
                total_input_tokens + 1
            )  # Avoid division by 0

        # Find highest cost operation
        operation_costs = {}
        for call in claude_calls:
            operation = call.get("operation", "unknown")
            cost = call.get("cost", 0.0)

            if operation not in operation_costs:
                operation_costs[operation] = 0.0

            operation_costs[operation] += cost

        if operation_costs:
            highest_cost_operation = max(operation_costs.items(), key=lambda x: x[1])
            metrics["highest_cost_operation"] = highest_cost_operation[0]
            metrics["highest_cost_value"] = highest_cost_operation[1]

        return metrics


class UsagePatternAnalyzer:
    """Analyzer for API usage patterns."""

    def __init__(self, config: Config, cost_tracker: CostTracker):
        self.config = config
        self.cost_tracker = cost_tracker

    async def analyze_claude_usage_patterns(self) -> ClaudeUsageAnalysis:
        """Analyze Claude API usage for optimization opportunities."""

        # Get historical usage data (last 30 days)
        usage_data = await self._get_claude_usage_history(30)

        # Initialize analysis components
        prompt_efficiency = await self._analyze_prompt_efficiency(usage_data)
        batch_opportunities = await self._identify_batch_opportunities(usage_data)
        caching_opportunities = await self._identify_caching_opportunities(usage_data)
        cost_per_analysis_type = self._calculate_cost_per_analysis_type(usage_data)

        # Calculate total cost and calls
        total_cost = sum(call.cost for call in usage_data)
        total_calls = len(usage_data)

        return ClaudeUsageAnalysis(
            prompt_efficiency=prompt_efficiency,
            batch_opportunities=batch_opportunities,
            caching_opportunities=caching_opportunities,
            cost_per_analysis_type=cost_per_analysis_type,
            total_calls=total_calls,
            total_cost=total_cost,
            analysis_period_days=30,
        )

    async def _get_claude_usage_history(self, days: int = 30) -> list[APICall]:
        """Get historical Claude API usage data."""
        if not self.cost_tracker or "claude" not in self.cost_tracker.costs:
            # If no real data, return simulated data
            return self._get_simulated_usage_data()

        # Convert cost tracker data to APICall objects
        calls = []
        for entry in self.cost_tracker.costs["claude"]:
            calls.append(
                APICall(
                    service="claude",
                    operation=entry.get("operation", "unknown"),
                    timestamp=entry.get("timestamp", datetime.now().isoformat()),
                    input_tokens=entry.get("details", {}).get("input_tokens", 0),
                    output_tokens=entry.get("details", {}).get("output_tokens", 0),
                    cost=entry.get("cost", 0.0),
                    details=entry.get("details", {}),
                )
            )

        return calls

    def _get_simulated_usage_data(self) -> list[APICall]:
        """Generate simulated usage data for testing."""
        operations = [
            "similarity_search",
            "duplicate_detection",
            "scope_drift_analysis",
            "template_inference",
            "quality_analysis",
        ]

        calls = []
        now = datetime.now()

        # Generate 100 simulated calls over 30 days
        for i in range(100):
            # Random operation
            operation = operations[i % len(operations)]

            # Random date in last 30 days
            days_ago = i % 30
            timestamp = (now - timedelta(days=days_ago)).isoformat()

            # Simulated token counts based on operation
            input_tokens = {
                "similarity_search": 200,
                "duplicate_detection": 1500,
                "scope_drift_analysis": 2000,
                "template_inference": 3000,
                "quality_analysis": 1000,
            }.get(operation, 500)

            output_tokens = {
                "similarity_search": 100,
                "duplicate_detection": 300,
                "scope_drift_analysis": 500,
                "template_inference": 700,
                "quality_analysis": 400,
            }.get(operation, 200)

            # Add some variation
            input_tokens = int(input_tokens * (0.8 + 0.4 * (i % 5) / 4.0))
            output_tokens = int(output_tokens * (0.8 + 0.4 * (i % 7) / 6.0))

            # Calculate cost
            cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)

            calls.append(
                APICall(
                    service="claude",
                    operation=operation,
                    timestamp=timestamp,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                    details={
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "model": self.config.claude_model,
                    },
                )
            )

        return calls

    async def _analyze_prompt_efficiency(
        self, usage_data: list[APICall]
    ) -> PromptEfficiencyAnalysis:
        """Analyze prompt efficiency and suggest improvements."""

        prompt_categories: dict[str, list[APICall]] = {}

        for call in usage_data:
            # Categorize by operation type
            category = call.operation.split(".")[0] if "." in call.operation else call.operation

            if category not in prompt_categories:
                prompt_categories[category] = []
            prompt_categories[category].append(call)

        efficiency_results = []

        for category, calls in prompt_categories.items():
            if not calls:
                continue

            avg_input_tokens = statistics.mean(call.input_tokens for call in calls)
            avg_output_tokens = statistics.mean(call.output_tokens for call in calls)
            avg_cost = statistics.mean(call.cost for call in calls)

            # Identify unusually expensive calls
            expensive_calls = [call for call in calls if call.cost > avg_cost * 1.5]

            # Calculate optimization potential
            # Assume we can save 20% on input tokens for inefficient prompts
            potential_savings = (
                sum(call.cost * 0.2 for call in expensive_calls) * 30
            )  # Monthly estimate

            efficiency_results.append(
                PromptCategoryAnalysis(
                    category=category,
                    total_calls=len(calls),
                    avg_input_tokens=avg_input_tokens,
                    avg_output_tokens=avg_output_tokens,
                    avg_cost=avg_cost,
                    expensive_calls=len(expensive_calls),
                    optimization_potential=potential_savings,
                )
            )

        # Calculate overall efficiency score
        total_calls = sum(result.total_calls for result in efficiency_results)
        if total_calls > 0:
            overall_efficiency_score = 1.0 - (
                sum(result.expensive_calls for result in efficiency_results) / total_calls
            )
        else:
            overall_efficiency_score = 0.0

        return PromptEfficiencyAnalysis(
            categories=efficiency_results, overall_efficiency_score=overall_efficiency_score
        )

    async def _identify_batch_opportunities(
        self, usage_data: list[APICall]
    ) -> BatchOptimizationSuggestions:
        """Identify opportunities for batching API calls."""
        suggestions = []

        # Group calls by type and timestamp proximity
        batches = self._identify_potential_batches(usage_data)
        total_potential_savings = 0.0

        # Analyze each batch for savings
        for batch_type, calls in batches.items():
            if len(calls) >= 3:  # Only consider significant batching opportunities
                # Calculate potential savings
                current_cost = sum(call.cost for call in calls)

                # Estimate batch cost - assuming 30% saving from batching
                batch_cost = current_cost * 0.7
                savings = current_cost - batch_cost

                # Extrapolate to monthly savings
                days_covered = self._get_days_covered(calls)
                if days_covered > 0:
                    monthly_savings = (savings / days_covered) * 30
                else:
                    monthly_savings = savings

                if monthly_savings >= 1.0:  # Only suggest if savings is significant
                    suggestions.append(
                        OptimizationSuggestion(
                            type=f"{batch_type}_batch_optimization",
                            current_value=1,  # Individual calls
                            suggested_value=min(10, len(calls)),  # Batch size
                            potential_savings=monthly_savings,
                            risk_level="low",
                            description=f"Batch {batch_type} operations for cost savings",
                        )
                    )

                    total_potential_savings += monthly_savings

        # Embedding batch optimization
        if "embeddings" in self.cost_tracker.costs:
            current_size = getattr(self.config, "embedding_batch_size", 32)
            if current_size < 64:
                suggestions.append(
                    OptimizationSuggestion(
                        type="embedding_batch_increase",
                        current_value=current_size,
                        suggested_value=64,
                        potential_savings=5.0,  # Estimated monthly savings
                        risk_level="low",
                        description="Increase embedding batch size for better throughput",
                    )
                )

                total_potential_savings += 5.0

        return BatchOptimizationSuggestions(
            suggestions=suggestions, estimated_total_savings=total_potential_savings
        )

    def _identify_potential_batches(self, usage_data: list[APICall]) -> dict[str, list[APICall]]:
        """Identify potential batch operations."""
        batches: dict[str, list[APICall]] = {}

        # Group calls by operation type
        for call in usage_data:
            batch_type = call.operation

            if batch_type not in batches:
                batches[batch_type] = []

            batches[batch_type].append(call)

        # Filter only batch types with enough calls
        return {k: v for k, v in batches.items() if len(v) >= 3}

    def _get_days_covered(self, calls: list[APICall]) -> int:
        """Calculate how many days are covered by the calls."""
        if not calls:
            return 0

        # Convert timestamps to datetime objects
        dates = []
        for call in calls:
            ts = call.timestamp
            if isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts)
                    dates.append(dt.date())
                except ValueError:
                    # Handle non-standard format
                    continue
            elif isinstance(ts, datetime):
                dates.append(ts.date())

        # Count unique dates
        unique_dates = set(dates)
        return len(unique_dates)

    async def _identify_caching_opportunities(self, usage_data: list[APICall]) -> dict[str, Any]:
        """Identify opportunities for result caching."""
        # Group calls by operation and input similarity
        # This is simplified - a real implementation would analyze input content

        duplicate_call_count = 0
        estimated_savings = 0.0

        # Group by operation
        operations: dict[str, list[APICall]] = {}
        for call in usage_data:
            if call.operation not in operations:
                operations[call.operation] = []
            operations[call.operation].append(call)

        # For each operation, estimate duplicate calls
        for operation, calls in operations.items():
            if len(calls) <= 1:
                continue

            # Simple heuristic - assume 10% of calls could be cached
            potential_dupes = int(len(calls) * 0.1)
            duplicate_call_count += potential_dupes

            # Calculate potential savings
            avg_cost = statistics.mean(call.cost for call in calls)
            estimated_savings += potential_dupes * avg_cost * (30 / 7)  # Weekly to monthly scaling

        return {
            "duplicate_call_count": duplicate_call_count,
            "potential_savings": estimated_savings,
            "cacheable_operations": [op for op, calls in operations.items() if len(calls) > 5],
            "recommendation": (
                "Implement results caching"
                if estimated_savings > 5.0
                else "No significant caching opportunity"
            ),
        }

    def _calculate_cost_per_analysis_type(self, usage_data: list[APICall]) -> dict[str, float]:
        """Calculate cost per analysis type."""
        costs_by_type = {}

        for call in usage_data:
            operation_type = (
                call.operation.split(".")[0] if "." in call.operation else call.operation
            )

            if operation_type not in costs_by_type:
                costs_by_type[operation_type] = 0.0

            costs_by_type[operation_type] += call.cost

        return costs_by_type

    async def suggest_batch_optimizations(self) -> BatchOptimizationSuggestions:
        """Suggest better batching strategies."""
        # Get usage data
        usage_data = await self._get_claude_usage_history(days=30)

        # Check current batching patterns
        current_batching = await self._analyze_current_batching_patterns()

        suggestions = []

        # Embedding batch optimization
        if current_batching.get("embedding_batch_size", 32) < 64:
            suggestions.append(
                OptimizationSuggestion(
                    type="embedding_batch_increase",
                    current_value=current_batching.get("embedding_batch_size", 32),
                    suggested_value=64,
                    potential_savings=10.0,  # Estimated savings
                    risk_level="low",
                    description="Increase embedding batch size for better throughput",
                )
            )

        # Claude batch optimization
        batchable_ops = await self._identify_batchable_claude_operations(usage_data)
        for op_name, op_data in batchable_ops.items():
            if op_data["potential_savings"] >= 5.0:  # Only suggest if savings is significant
                suggestions.append(
                    OptimizationSuggestion(
                        type=f"{op_name}_batch_processing",
                        current_value=1,
                        suggested_value=op_data["suggested_batch_size"],
                        potential_savings=op_data["potential_savings"],
                        risk_level="medium",
                        description=f"Batch {op_name} operations in a single API call",
                    )
                )

        # Calculate total potential savings
        total_savings = sum(suggestion.potential_savings for suggestion in suggestions)

        return BatchOptimizationSuggestions(
            suggestions=suggestions, estimated_total_savings=total_savings
        )

    async def _analyze_current_batching_patterns(self) -> dict[str, Any]:
        """Analyze current batching patterns."""
        return {
            "embedding_batch_size": getattr(self.config, "embedding_batch_size", 32),
            "jira_batch_size": getattr(self.config, "jira_batch_size", 50),
            "claude_batching": "disabled",  # Assume no batching by default
        }

    async def _identify_batchable_claude_operations(
        self, usage_data: list[APICall]
    ) -> dict[str, Any]:
        """Identify Claude operations that could be batched."""
        operations = {}

        # Group calls by operation
        for call in usage_data:
            op = call.operation

            if op not in operations:
                operations[op] = {
                    "calls": [],
                    "count": 0,
                    "avg_cost": 0.0,
                    "potential_savings": 0.0,
                    "suggested_batch_size": 0,
                }

            operations[op]["calls"].append(call)
            operations[op]["count"] += 1

        # Analyze each operation for batching potential
        for op_name, op_data in operations.items():
            if op_data["count"] < 5:  # Ignore operations with few calls
                continue

            # Calculate average cost
            avg_cost = sum(call.cost for call in op_data["calls"]) / op_data["count"]
            op_data["avg_cost"] = avg_cost

            # Calculate time distribution to identify potential batches
            timestamps = []
            for call in op_data["calls"]:
                ts = call.timestamp
                if isinstance(ts, str):
                    try:
                        dt = datetime.fromisoformat(ts)
                        timestamps.append(dt)
                    except ValueError:
                        continue
                elif isinstance(ts, datetime):
                    timestamps.append(ts)

            # Sort timestamps
            timestamps.sort()

            # Identify clusters of timestamps (potential batches)
            clusters = []
            current_cluster = []

            for i, ts in enumerate(timestamps):
                if not current_cluster:
                    current_cluster.append(ts)
                    continue

                # If within 5 seconds of previous, add to cluster
                if (ts - current_cluster[-1]).total_seconds() <= 5:
                    current_cluster.append(ts)
                else:
                    # Start new cluster
                    if len(current_cluster) > 1:
                        clusters.append(current_cluster)
                    current_cluster = [ts]

            # Add final cluster
            if len(current_cluster) > 1:
                clusters.append(current_cluster)

            # Calculate batching potential
            batchable_calls = sum(len(cluster) for cluster in clusters)
            batch_count = len(clusters)

            if batch_count > 0 and batchable_calls > 0:
                avg_batch_size = batchable_calls / batch_count

                # Estimate savings (assume 30% cost reduction from batching)
                single_cost = batchable_calls * avg_cost
                batch_cost = batch_count * (avg_cost * avg_batch_size * 0.7)
                savings = single_cost - batch_cost

                # Scale to monthly savings
                days_covered = (timestamps[-1] - timestamps[0]).days + 1
                monthly_savings = (savings / days_covered) * 30 if days_covered > 0 else savings

                op_data["potential_savings"] = monthly_savings
                op_data["suggested_batch_size"] = min(
                    10, int(avg_batch_size * 1.5)
                )  # Suggest slightly larger batch

            # Remove calls to reduce memory footprint
            op_data.pop("calls", None)

        return operations
