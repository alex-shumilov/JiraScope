"""Tests for the cost optimization and reporting systems."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from jirascope.core.config import Config
from jirascope.utils.logging import CostTracker
from jirascope.utils.cost_optimizer import (
    CostOptimizer,
    APICall,
    ComprehensiveCostReport,
    BudgetAlert
)
from jirascope.utils.cost_reporter import AdvancedCostReporter


@pytest.fixture
def mock_config():
    """Test configuration."""
    return Config(
        jira_mcp_endpoint="http://test.com",
        claude_model="claude-3-haiku-20240307",
        daily_budget=50.0,
        monthly_budget=1000.0
    )


@pytest.fixture
def mock_cost_tracker():
    """Mock cost tracker with sample data."""
    tracker = CostTracker()
    
    # Add sample costs
    tracker.costs = {
        "claude": [
            {
                "timestamp": datetime.now().isoformat(),
                "operation": "similarity_search",
                "cost": 0.05,
                "details": {
                    "input_tokens": 500,
                    "output_tokens": 200
                }
            },
            {
                "timestamp": datetime.now().isoformat(),
                "operation": "duplicate_detection",
                "cost": 0.12,
                "details": {
                    "input_tokens": 1500,
                    "output_tokens": 300
                }
            }
        ],
        "embeddings": [
            {
                "timestamp": datetime.now().isoformat(),
                "operation": "vector_generation",
                "cost": 0.03,
                "details": {
                    "vectors": 10
                }
            }
        ]
    }
    
    return tracker


@pytest.mark.asyncio
async def test_cost_optimizer_initialization(mock_config, mock_cost_tracker):
    """Test cost optimizer initialization."""
    optimizer = CostOptimizer(mock_config, mock_cost_tracker)
    
    # Check initialization
    assert optimizer.config == mock_config
    assert optimizer.cost_tracker == mock_cost_tracker
    assert optimizer.daily_budget == mock_config.daily_budget
    assert optimizer.monthly_budget == mock_config.monthly_budget
    assert len(optimizer.alert_thresholds) > 0


@pytest.mark.asyncio
async def test_cost_optimizer_analyze_api_usage_patterns(mock_config, mock_cost_tracker):
    """Test API usage pattern analysis."""
    optimizer = CostOptimizer(mock_config, mock_cost_tracker)
    
    # Mock method to return test data
    optimizer.usage_analyzer._get_claude_usage_history = AsyncMock(return_value=[
        APICall(
            service="claude",
            operation="similarity_search",
            timestamp=datetime.now().isoformat(),
            input_tokens=500,
            output_tokens=200,
            cost=0.05
        ),
        APICall(
            service="claude",
            operation="duplicate_detection",
            timestamp=datetime.now().isoformat(),
            input_tokens=1500,
            output_tokens=300,
            cost=0.12
        )
    ])
    
    # Analyze API usage
    analysis = await optimizer.analyze_api_usage_patterns()
    
    # Check analysis
    assert analysis.total_calls == 2
    assert analysis.total_cost > 0
    assert analysis.prompt_efficiency is not None
    assert len(analysis.prompt_efficiency.categories) > 0


@pytest.mark.asyncio
async def test_cost_optimizer_generate_cost_report(mock_config, mock_cost_tracker):
    """Test cost report generation."""
    optimizer = CostOptimizer(mock_config, mock_cost_tracker)
    
    # Generate report
    report = await optimizer.generate_cost_report()
    
    # Check report
    assert isinstance(report, ComprehensiveCostReport)
    assert report.total_cost >= 0
    assert len(report.service_breakdown) > 0
    assert report.trends is not None
    assert report.predictions is not None


@pytest.mark.asyncio
async def test_cost_optimizer_check_budget_alerts(mock_config, mock_cost_tracker):
    """Test budget alerts."""
    optimizer = CostOptimizer(mock_config, mock_cost_tracker)
    
    # Mock _get_daily_cost and _get_monthly_cost
    optimizer._get_daily_cost = AsyncMock(return_value=45.0)  # 90% of daily budget
    optimizer._get_monthly_cost = AsyncMock(return_value=900.0)  # 90% of monthly budget
    
    # Check for alerts
    alerts = await optimizer.check_budget_alerts()
    
    # Should get alerts for 50%, 75%, and 90% thresholds
    assert isinstance(alerts, list)
    assert len(alerts) > 0
    assert all(isinstance(alert, BudgetAlert) for alert in alerts)


@pytest.mark.asyncio
async def test_advanced_cost_reporter(mock_config, mock_cost_tracker):
    """Test advanced cost reporter."""
    # Create cost optimizer
    optimizer = CostOptimizer(mock_config, mock_cost_tracker)
    
    # Create reporter
    reporter = AdvancedCostReporter(mock_config, mock_cost_tracker, optimizer)
    
    # Mock methods to avoid actual file operations
    reporter._save_report = AsyncMock(return_value=True)
    
    # Generate comprehensive report
    report = await reporter.generate_comprehensive_cost_report()
    
    # Check report
    assert report.total_cost >= 0
    assert report.usage_breakdown is not None
    assert report.predictions is not None
    assert report.efficiency is not None
    assert report.budget_status is not None
    
    # Generate daily summary
    daily_summary = await reporter.generate_daily_cost_summary()
    assert daily_summary["date"] is not None
    assert "current_cost" in daily_summary
    
    # Generate monthly projection
    monthly_projection = await reporter.generate_monthly_cost_projection()
    assert "month" in monthly_projection
    assert "projected_month_total" in monthly_projection