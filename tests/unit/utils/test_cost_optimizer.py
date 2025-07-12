"""Comprehensive tests for cost optimizer functionality."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.jirascope.core.config import Config
from src.jirascope.utils.cost_optimizer import (
    APICall,
    BatchOptimizationSuggestions,
    BudgetAlert,
    ClaudeUsageAnalysis,
    ComprehensiveCostReport,
    CostOptimizer,
    CostPredictions,
    CostTrends,
    OptimizationSuggestion,
    PromptCategoryAnalysis,
    PromptEfficiencyAnalysis,
    UsagePatternAnalyzer,
)
from src.jirascope.utils.logging import CostTracker


class TestAPICall:
    """Test APICall model functionality."""

    def test_api_call_creation(self):
        """Test creating APICall with all fields."""
        api_call = APICall(
            service="claude",
            operation="analyze_quality",
            timestamp=datetime.now(),
            input_tokens=100,
            output_tokens=50,
            cost=0.015,
            details={"model": "claude-3-sonnet"},
        )

        assert api_call.service == "claude"
        assert api_call.operation == "analyze_quality"
        assert api_call.input_tokens == 100
        assert api_call.output_tokens == 50
        assert api_call.cost == 0.015
        assert api_call.details["model"] == "claude-3-sonnet"

    def test_api_call_defaults(self):
        """Test APICall with default values."""
        api_call = APICall(
            service="embeddings", operation="generate", timestamp="2024-01-01T10:00:00"
        )

        assert api_call.input_tokens == 0
        assert api_call.output_tokens == 0
        assert api_call.cost == 0.0
        assert api_call.details == {}


class TestPromptCategoryAnalysis:
    """Test PromptCategoryAnalysis model functionality."""

    def test_prompt_category_analysis_creation(self):
        """Test creating PromptCategoryAnalysis."""
        analysis = PromptCategoryAnalysis(
            category="quality_analysis",
            total_calls=25,
            avg_input_tokens=150.5,
            avg_output_tokens=75.2,
            avg_cost=0.022,
            expensive_calls=3,
            optimization_potential=0.15,
        )

        assert analysis.category == "quality_analysis"
        assert analysis.total_calls == 25
        assert analysis.avg_input_tokens == 150.5
        assert analysis.avg_output_tokens == 75.2
        assert analysis.avg_cost == 0.022
        assert analysis.expensive_calls == 3
        assert analysis.optimization_potential == 0.15

    def test_prompt_category_analysis_defaults(self):
        """Test PromptCategoryAnalysis with defaults."""
        analysis = PromptCategoryAnalysis(category="test_category")

        assert analysis.total_calls == 0
        assert analysis.avg_input_tokens == 0.0
        assert analysis.avg_output_tokens == 0.0
        assert analysis.avg_cost == 0.0
        assert analysis.expensive_calls == 0
        assert analysis.optimization_potential == 0.0


class TestOptimizationSuggestion:
    """Test OptimizationSuggestion model functionality."""

    def test_optimization_suggestion_creation(self):
        """Test creating OptimizationSuggestion."""
        suggestion = OptimizationSuggestion(
            type="batch_size",
            current_value=8,
            suggested_value=32,
            potential_savings=15.50,
            risk_level="low",
            description="Increase batch size for embedding operations",
        )

        assert suggestion.type == "batch_size"
        assert suggestion.current_value == 8
        assert suggestion.suggested_value == 32
        assert suggestion.potential_savings == 15.50
        assert suggestion.risk_level == "low"
        assert "batch size" in suggestion.description


class TestBudgetAlert:
    """Test BudgetAlert model functionality."""

    def test_budget_alert_creation(self):
        """Test creating BudgetAlert."""
        alert = BudgetAlert(
            type="daily_budget",
            threshold=0.75,
            current_usage=0.82,
            severity="medium",
            message="Daily budget at 82% of limit",
        )

        assert alert.type == "daily_budget"
        assert alert.threshold == 0.75
        assert alert.current_usage == 0.82
        assert alert.severity == "medium"
        assert "82%" in alert.message
        assert isinstance(alert.timestamp, datetime)

    def test_budget_alert_with_timestamp(self):
        """Test BudgetAlert with custom timestamp."""
        custom_time = datetime(2024, 1, 1, 10, 0, 0)
        alert = BudgetAlert(
            type="monthly_budget",
            threshold=0.9,
            current_usage=0.95,
            severity="high",
            message="Monthly budget exceeded",
            timestamp=custom_time,
        )

        assert alert.timestamp == custom_time


class TestCostTrends:
    """Test CostTrends model functionality."""

    def test_cost_trends_creation(self):
        """Test creating CostTrends."""
        trends = CostTrends(
            daily_costs=[10.5, 12.3, 9.8, 11.2],
            weekly_costs=[75.5, 82.1],
            monthly_costs=[320.5],
            trend_direction="increasing",
            growth_rate=0.15,
        )

        assert len(trends.daily_costs) == 4
        assert len(trends.weekly_costs) == 2
        assert len(trends.monthly_costs) == 1
        assert trends.trend_direction == "increasing"
        assert trends.growth_rate == 0.15

    def test_cost_trends_defaults(self):
        """Test CostTrends with defaults."""
        trends = CostTrends()

        assert trends.daily_costs == []
        assert trends.weekly_costs == []
        assert trends.monthly_costs == []
        assert trends.trend_direction == "stable"
        assert trends.growth_rate == 0.0


class TestCostPredictions:
    """Test CostPredictions model functionality."""

    def test_cost_predictions_creation(self):
        """Test creating CostPredictions."""
        predictions = CostPredictions(
            next_week=85.5, next_month=350.0, confidence=0.82, growth_rate=0.12
        )

        assert predictions.next_week == 85.5
        assert predictions.next_month == 350.0
        assert predictions.confidence == 0.82
        assert predictions.growth_rate == 0.12

    def test_cost_predictions_defaults(self):
        """Test CostPredictions with defaults."""
        predictions = CostPredictions()

        assert predictions.next_week == 0.0
        assert predictions.next_month == 0.0
        assert predictions.confidence == 0.0
        assert predictions.growth_rate == 0.0


class TestCostOptimizer:
    """Test CostOptimizer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config(
            jira_mcp_endpoint="https://test.atlassian.net", daily_budget=50.0, monthly_budget=1000.0
        )
        self.mock_cost_tracker = Mock(spec=CostTracker)
        self.optimizer = CostOptimizer(self.config, self.mock_cost_tracker)

    def test_cost_optimizer_initialization(self):
        """Test CostOptimizer initialization."""
        assert self.optimizer.config == self.config
        assert self.optimizer.cost_tracker == self.mock_cost_tracker
        assert self.optimizer.daily_budget == 50.0
        assert self.optimizer.monthly_budget == 1000.0
        assert len(self.optimizer.alert_thresholds) == 4
        assert self.optimizer.sent_alerts == {}

    def test_cost_optimizer_with_missing_budget_config(self):
        """Test CostOptimizer with missing budget configuration."""
        config_without_budget = Config(jira_mcp_endpoint="https://test.atlassian.net")
        optimizer = CostOptimizer(config_without_budget, self.mock_cost_tracker)

        assert optimizer.daily_budget == 50.0  # Default value
        assert optimizer.monthly_budget == 1000.0  # Default value

    @pytest.mark.asyncio
    async def test_analyze_api_usage_patterns(self):
        """Test analyzing API usage patterns."""
        # Mock the usage analyzer
        mock_usage_analysis = ClaudeUsageAnalysis(
            prompt_efficiency=PromptEfficiencyAnalysis(),
            batch_opportunities=BatchOptimizationSuggestions(),
            total_calls=100,
            total_cost=25.50,
            analysis_period_days=30,
        )

        self.optimizer.usage_analyzer.analyze_claude_usage_patterns = AsyncMock(
            return_value=mock_usage_analysis
        )

        result = await self.optimizer.analyze_api_usage_patterns()

        assert isinstance(result, ClaudeUsageAnalysis)
        assert result.total_calls == 100
        assert result.total_cost == 25.50
        assert result.analysis_period_days == 30

    @pytest.mark.asyncio
    async def test_suggest_batch_optimizations(self):
        """Test suggesting batch optimizations."""
        mock_suggestions = BatchOptimizationSuggestions(
            suggestions=[
                OptimizationSuggestion(
                    type="batch_size",
                    current_value=8,
                    suggested_value=32,
                    potential_savings=20.0,
                    risk_level="low",
                    description="Increase batch size",
                )
            ],
            estimated_total_savings=20.0,
        )

        self.optimizer.usage_analyzer.suggest_batch_optimizations = AsyncMock(
            return_value=mock_suggestions
        )

        result = await self.optimizer.suggest_batch_optimizations()

        assert isinstance(result, BatchOptimizationSuggestions)
        assert len(result.suggestions) == 1
        assert result.estimated_total_savings == 20.0

    @pytest.mark.asyncio
    async def test_generate_cost_report(self):
        """Test generating comprehensive cost report."""
        # Mock all the internal methods
        self.optimizer._get_detailed_costs = AsyncMock(return_value={"test": "data"})
        self.optimizer._get_service_breakdown = AsyncMock(
            return_value={"claude": 100.0, "embeddings": 50.0}
        )
        self.optimizer._analyze_cost_trends = AsyncMock(return_value=CostTrends())
        self.optimizer._predict_future_costs = AsyncMock(return_value=CostPredictions())
        self.optimizer._identify_cost_optimizations = AsyncMock(return_value=[])
        self.optimizer._calculate_cost_per_analysis_type = AsyncMock(return_value={"quality": 15.0})
        self.optimizer._calculate_efficiency_metrics = AsyncMock(return_value={"efficiency": 0.85})

        result = await self.optimizer.generate_cost_report("month")

        assert isinstance(result, ComprehensiveCostReport)
        assert result.period == "month"
        assert result.total_cost == 150.0  # Sum of service breakdown
        assert "claude" in result.service_breakdown
        assert "embeddings" in result.service_breakdown
        assert "quality" in result.cost_per_analysis
        assert "efficiency" in result.efficiency_metrics

    @pytest.mark.asyncio
    async def test_check_budget_alerts_daily_threshold(self):
        """Test checking budget alerts for daily threshold."""
        # Mock daily cost that exceeds threshold
        self.optimizer._get_daily_cost = AsyncMock(return_value=40.0)  # 80% of 50.0 budget
        self.optimizer._get_monthly_cost = AsyncMock(return_value=500.0)  # 50% of 1000.0 budget

        alerts = await self.optimizer.check_budget_alerts()

        assert len(alerts) >= 1
        daily_alerts = [a for a in alerts if a.type == "daily_budget"]
        assert len(daily_alerts) >= 1

        # Should have alerts for 50% and 75% thresholds (80% usage exceeds both)
        thresholds_alerted = [a.threshold for a in daily_alerts]
        assert 0.5 in thresholds_alerted
        assert 0.75 in thresholds_alerted

    @pytest.mark.asyncio
    async def test_check_budget_alerts_monthly_threshold(self):
        """Test checking budget alerts for monthly threshold."""
        self.optimizer._get_daily_cost = AsyncMock(return_value=10.0)  # 20% of daily budget
        self.optimizer._get_monthly_cost = AsyncMock(return_value=950.0)  # 95% of monthly budget

        alerts = await self.optimizer.check_budget_alerts()

        monthly_alerts = [a for a in alerts if a.type == "monthly_budget"]
        assert len(monthly_alerts) >= 1

        # Should have alerts for 50%, 75%, and 90% thresholds (95% usage exceeds all)
        thresholds_alerted = [a.threshold for a in monthly_alerts]
        assert 0.5 in thresholds_alerted
        assert 0.75 in thresholds_alerted
        assert 0.9 in thresholds_alerted

    @pytest.mark.asyncio
    async def test_check_budget_alerts_no_alerts_needed(self):
        """Test checking budget alerts when no alerts are needed."""
        self.optimizer._get_daily_cost = AsyncMock(return_value=20.0)  # 40% of daily budget
        self.optimizer._get_monthly_cost = AsyncMock(return_value=400.0)  # 40% of monthly budget

        alerts = await self.optimizer.check_budget_alerts()

        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_reset_daily_alerts(self):
        """Test resetting daily alerts."""
        # Set some alerts as sent
        self.optimizer.sent_alerts = {
            "daily_0.5": datetime.now(),
            "daily_0.75": datetime.now(),
            "monthly_0.5": datetime.now(),
        }

        await self.optimizer.reset_daily_alerts()

        # Daily alerts should be cleared, monthly should remain
        remaining_alerts = [key for key in self.optimizer.sent_alerts.keys() if "daily_" in key]
        assert len(remaining_alerts) == 0
        assert "monthly_0.5" in self.optimizer.sent_alerts

    @pytest.mark.asyncio
    async def test_reset_monthly_alerts(self):
        """Test resetting monthly alerts."""
        # Set some alerts as sent
        self.optimizer.sent_alerts = {
            "daily_0.5": datetime.now(),
            "monthly_0.5": datetime.now(),
            "monthly_0.75": datetime.now(),
        }

        await self.optimizer.reset_monthly_alerts()

        # Monthly alerts should be cleared, daily should remain
        remaining_alerts = [key for key in self.optimizer.sent_alerts.keys() if "monthly_" in key]
        assert len(remaining_alerts) == 0
        assert "daily_0.5" in self.optimizer.sent_alerts

    def test_get_alert_severity(self):
        """Test alert severity calculation."""
        assert self.optimizer._get_alert_severity(0.5) == "low"
        assert self.optimizer._get_alert_severity(0.75) == "medium"
        assert self.optimizer._get_alert_severity(0.9) == "high"
        assert self.optimizer._get_alert_severity(1.0) == "critical"
        assert self.optimizer._get_alert_severity(1.1) == "critical"

    def test_alert_already_sent(self):
        """Test checking if alert was already sent."""
        # No alerts sent yet
        assert not self.optimizer._alert_already_sent("daily_0.5")

        # Mark an alert as sent
        self.optimizer._mark_alert_sent("daily_0.5")
        assert self.optimizer._alert_already_sent("daily_0.5")

        # Different alert should not be marked as sent
        assert not self.optimizer._alert_already_sent("daily_0.75")

    def test_mark_alert_sent(self):
        """Test marking alert as sent."""
        alert_key = "monthly_0.75"
        assert alert_key not in self.optimizer.sent_alerts

        self.optimizer._mark_alert_sent(alert_key)

        assert alert_key in self.optimizer.sent_alerts
        assert isinstance(self.optimizer.sent_alerts[alert_key], datetime)

    @pytest.mark.asyncio
    async def test_get_daily_cost(self):
        """Test getting daily cost."""
        # Mock cost tracker to return specific daily cost
        self.mock_cost_tracker.get_daily_cost.return_value = 35.75

        result = await self.optimizer._get_daily_cost()

        assert result == 35.75
        self.mock_cost_tracker.get_daily_cost.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_monthly_cost(self):
        """Test getting monthly cost."""
        # Mock cost tracker to return specific monthly cost
        self.mock_cost_tracker.get_monthly_cost.return_value = 750.25

        result = await self.optimizer._get_monthly_cost()

        assert result == 750.25
        self.mock_cost_tracker.get_monthly_cost.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_detailed_costs_month(self):
        """Test getting detailed costs for month period."""
        mock_data = {"claude_calls": 150, "embedding_calls": 200, "total_tokens": 50000}
        self.mock_cost_tracker.get_detailed_costs.return_value = mock_data

        result = await self.optimizer._get_detailed_costs("month")

        assert result == mock_data
        self.mock_cost_tracker.get_detailed_costs.assert_called_once_with("month")

    @pytest.mark.asyncio
    async def test_get_service_breakdown(self):
        """Test getting service cost breakdown."""
        cost_data = {
            "services": {
                "claude": {"total_cost": 125.50},
                "embeddings": {"total_cost": 75.25},
                "qdrant": {"total_cost": 0.0},
            }
        }

        result = await self.optimizer._get_service_breakdown(cost_data)

        expected = {"claude": 125.50, "embeddings": 75.25, "qdrant": 0.0}
        assert result == expected

    @pytest.mark.asyncio
    async def test_get_service_breakdown_empty_data(self):
        """Test getting service breakdown with empty data."""
        cost_data = {
            "claude_calls": [],
            "embedding_operations": [],
            "qdrant_costs": 0.0,
            "jira_api_costs": 0.0,
        }

        result = await self.optimizer._get_service_breakdown(cost_data)

        # Should return dict with zero values
        expected = {
            "claude_api": 0.0,
            "embedding_processing": 0.0,
            "vector_storage": 0.0,
            "jira_api": 0.0,
        }
        assert result == expected

    @pytest.mark.asyncio
    async def test_analyze_cost_trends(self):
        """Test analyzing cost trends."""
        cost_data = {
            "daily_costs": [10.0, 12.0, 15.0, 18.0, 20.0],
            "period_costs": {"week1": 75.0, "week2": 85.0, "week3": 95.0},
        }

        result = await self.optimizer._analyze_cost_trends(cost_data, "month")

        assert isinstance(result, CostTrends)
        assert len(result.daily_costs) > 0
        assert result.trend_direction in ["increasing", "stable", "decreasing"]
        assert isinstance(result.growth_rate, float)

    @pytest.mark.asyncio
    async def test_predict_future_costs(self):
        """Test predicting future costs."""
        trends = CostTrends(
            daily_costs=[10.0, 12.0, 15.0],
            weekly_costs=[75.0, 85.0],
            trend_direction="increasing",
            growth_rate=0.15,
        )

        result = await self.optimizer._predict_future_costs(trends)

        assert isinstance(result, CostPredictions)
        assert result.next_week > 0
        assert result.next_month > 0
        assert 0 <= result.confidence <= 1
        assert isinstance(result.growth_rate, float)

    @pytest.mark.asyncio
    async def test_identify_cost_optimizations(self):
        """Test identifying cost optimization opportunities."""
        cost_data = {
            "high_cost_operations": ["quality_analysis", "similarity_detection"],
            "batch_opportunities": True,
            "avg_batch_size": 8,
        }

        result = await self.optimizer._identify_cost_optimizations(cost_data)

        assert isinstance(result, list)
        for suggestion in result:
            assert isinstance(suggestion, OptimizationSuggestion)
            assert hasattr(suggestion, "type")
            assert hasattr(suggestion, "potential_savings")
            assert hasattr(suggestion, "risk_level")

    @pytest.mark.asyncio
    async def test_calculate_cost_per_analysis_type(self):
        """Test calculating cost per analysis type."""
        cost_data = {
            "analysis_costs": {
                "quality": {"total_cost": 150.0, "count": 10},
                "similarity": {"total_cost": 200.0, "count": 8},
                "complexity": {"total_cost": 100.0, "count": 5},
            }
        }

        result = await self.optimizer._calculate_cost_per_analysis_type(cost_data)

        expected = {
            "quality": 15.0,  # 150.0 / 10
            "similarity": 25.0,  # 200.0 / 8
            "complexity": 20.0,  # 100.0 / 5
        }
        assert result == expected

    @pytest.mark.asyncio
    async def test_calculate_efficiency_metrics(self):
        """Test calculating efficiency metrics."""
        cost_data = {
            "total_operations": 100,
            "total_cost": 250.0,
            "avg_response_time": 2.5,
            "success_rate": 0.95,
        }

        result = await self.optimizer._calculate_efficiency_metrics(cost_data)

        assert isinstance(result, dict)
        assert "cost_per_operation" in result
        assert "efficiency_score" in result
        assert result["cost_per_operation"] == 2.5  # 250.0 / 100
        assert isinstance(result["efficiency_score"], float)


class TestUsagePatternAnalyzer:
    """Test UsagePatternAnalyzer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config(jira_mcp_endpoint="https://test.atlassian.net")
        self.mock_cost_tracker = Mock(spec=CostTracker)
        self.analyzer = UsagePatternAnalyzer(self.config, self.mock_cost_tracker)

    def test_usage_pattern_analyzer_initialization(self):
        """Test UsagePatternAnalyzer initialization."""
        assert self.analyzer.config == self.config
        assert self.analyzer.cost_tracker == self.mock_cost_tracker

    @pytest.mark.asyncio
    async def test_analyze_claude_usage_patterns(self):
        """Test analyzing Claude usage patterns."""
        # Mock the internal methods
        self.analyzer._get_claude_usage_history = AsyncMock(return_value=[])
        self.analyzer._analyze_prompt_efficiency = AsyncMock(
            return_value=PromptEfficiencyAnalysis()
        )
        self.analyzer._identify_batch_opportunities = AsyncMock(
            return_value=BatchOptimizationSuggestions()
        )
        self.analyzer._identify_caching_opportunities = AsyncMock(
            return_value={"potential_savings": 25.0}
        )

        result = await self.analyzer.analyze_claude_usage_patterns()

        assert isinstance(result, ClaudeUsageAnalysis)
        assert hasattr(result, "prompt_efficiency")
        assert hasattr(result, "batch_opportunities")
        assert hasattr(result, "caching_opportunities")

    def test_get_simulated_usage_data(self):
        """Test getting simulated usage data."""
        result = self.analyzer._get_simulated_usage_data()

        assert isinstance(result, list)
        assert len(result) > 0

        for api_call in result[:5]:  # Check first 5 calls
            assert isinstance(api_call, APICall)
            assert api_call.service == "claude"
            assert api_call.input_tokens > 0
            assert api_call.output_tokens > 0
            assert api_call.cost > 0

    @pytest.mark.asyncio
    async def test_analyze_prompt_efficiency(self):
        """Test analyzing prompt efficiency."""
        # Create sample usage data
        usage_data = [
            APICall(
                service="claude",
                operation="quality_analysis",
                timestamp=datetime.now(),
                input_tokens=150,
                output_tokens=75,
                cost=0.025,
            ),
            APICall(
                service="claude",
                operation="quality_analysis",
                timestamp=datetime.now(),
                input_tokens=200,
                output_tokens=100,
                cost=0.035,
            ),
            APICall(
                service="claude",
                operation="similarity_detection",
                timestamp=datetime.now(),
                input_tokens=300,
                output_tokens=50,
                cost=0.045,
            ),
        ]

        result = await self.analyzer._analyze_prompt_efficiency(usage_data)

        assert isinstance(result, PromptEfficiencyAnalysis)
        assert len(result.categories) > 0
        assert 0 <= result.overall_efficiency_score <= 1

    @pytest.mark.asyncio
    async def test_identify_batch_opportunities(self):
        """Test identifying batch opportunities."""
        # Create usage data that could be batched
        usage_data = []
        base_time = datetime.now()

        for i in range(10):
            usage_data.append(
                APICall(
                    service="claude",
                    operation="quality_analysis",
                    timestamp=base_time + timedelta(minutes=i * 5),  # 5 minutes apart
                    input_tokens=100,
                    output_tokens=50,
                    cost=0.02,
                )
            )

        result = await self.analyzer._identify_batch_opportunities(usage_data)

        assert isinstance(result, BatchOptimizationSuggestions)
        assert hasattr(result, "suggestions")
        assert hasattr(result, "estimated_total_savings")

    def test_identify_potential_batches(self):
        """Test identifying potential batches."""
        # Create usage data with similar operations
        usage_data = []
        base_time = datetime.now()

        for i in range(5):
            usage_data.append(
                APICall(
                    service="claude",
                    operation="quality_analysis",
                    timestamp=base_time + timedelta(minutes=i * 2),
                    input_tokens=100,
                    output_tokens=50,
                    cost=0.02,
                )
            )

        result = self.analyzer._identify_potential_batches(usage_data)

        assert isinstance(result, dict)
        # Should group similar operations together

    def test_get_days_covered(self):
        """Test calculating days covered by API calls."""
        # Create calls with specific dates 5 days apart
        today = datetime.now()
        five_days_ago = today - timedelta(days=5)

        calls = [
            APICall(service="claude", operation="test", timestamp=five_days_ago, cost=0.01),
            APICall(
                service="claude", operation="test", timestamp=today - timedelta(days=2), cost=0.01
            ),
            APICall(service="claude", operation="test", timestamp=today, cost=0.01),
        ]

        result = self.analyzer._get_days_covered(calls)

        # Should be 6 days (day 0 through day 5)
        assert result == 6

    @pytest.mark.asyncio
    async def test_identify_caching_opportunities(self):
        """Test identifying caching opportunities."""
        # Create usage data with repeated similar calls
        usage_data = []
        base_time = datetime.now()

        # Add some duplicate-like operations
        for i in range(3):
            usage_data.append(
                APICall(
                    service="claude",
                    operation="quality_analysis",
                    timestamp=base_time + timedelta(hours=i),
                    input_tokens=150,  # Same input size
                    output_tokens=75,
                    cost=0.025,
                )
            )

        result = await self.analyzer._identify_caching_opportunities(usage_data)

        assert isinstance(result, dict)
        assert "potential_savings" in result
        assert "cacheable_operations" in result
        assert "cache_hit_rate" in result

    def test_calculate_cost_per_analysis_type(self):
        """Test calculating cost per analysis type."""
        usage_data = [
            APICall(service="claude", operation="quality", timestamp=datetime.now(), cost=0.02),
            APICall(service="claude", operation="quality", timestamp=datetime.now(), cost=0.03),
            APICall(service="claude", operation="similarity", timestamp=datetime.now(), cost=0.04),
            APICall(service="claude", operation="similarity", timestamp=datetime.now(), cost=0.05),
            APICall(service="claude", operation="similarity", timestamp=datetime.now(), cost=0.06),
        ]

        result = self.analyzer._calculate_cost_per_analysis_type(usage_data)

        assert isinstance(result, dict)
        assert "quality" in result
        assert "similarity" in result
        assert result["quality"] == 0.05  # total cost = 0.02 + 0.03 = 0.05
        assert result["similarity"] == 0.15  # total cost = 0.04 + 0.05 + 0.06 = 0.15

    @pytest.mark.asyncio
    async def test_suggest_batch_optimizations(self):
        """Test suggesting batch optimizations."""
        # Mock the internal methods
        self.analyzer._get_claude_usage_history = AsyncMock(return_value=[])
        self.analyzer._analyze_current_batching_patterns = AsyncMock(
            return_value={"current_avg_batch_size": 8}
        )
        self.analyzer._identify_batchable_claude_operations = AsyncMock(
            return_value={"batchable_ops": 25, "potential_savings": 15.0}
        )

        result = await self.analyzer.suggest_batch_optimizations()

        assert isinstance(result, BatchOptimizationSuggestions)
        assert hasattr(result, "suggestions")
        assert hasattr(result, "estimated_total_savings")

    @pytest.mark.asyncio
    async def test_analyze_current_batching_patterns(self):
        """Test analyzing current batching patterns."""
        result = await self.analyzer._analyze_current_batching_patterns()

        assert isinstance(result, dict)
        # Should contain batching metrics

    @pytest.mark.asyncio
    async def test_identify_batchable_claude_operations(self):
        """Test identifying batchable Claude operations."""
        usage_data = [
            APICall(service="claude", operation="quality", timestamp=datetime.now(), cost=0.02),
            APICall(service="claude", operation="quality", timestamp=datetime.now(), cost=0.02),
            APICall(service="claude", operation="similarity", timestamp=datetime.now(), cost=0.03),
        ]

        result = await self.analyzer._identify_batchable_claude_operations(usage_data)

        assert isinstance(result, dict)
        assert "batchable_operations" in result
        assert "potential_batch_savings" in result
