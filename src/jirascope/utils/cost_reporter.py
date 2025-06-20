"""Advanced cost reporting and visualization for JiraScope."""

import json
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Dict, List, Optional, Any, Union

from pydantic import BaseModel, Field

from ..core.config import Config
from ..utils.logging import StructuredLogger, CostTracker
from .cost_optimizer import CostOptimizer, ComprehensiveCostReport, CostPredictions, CostTrends

logger = StructuredLogger(__name__)


class ServiceUsageTrend(BaseModel):
    """Usage trend for a service."""
    
    service: str = Field(..., description="Service name")
    usage: List[float] = Field(default_factory=list, description="Usage values over time")
    trend_direction: str = Field("stable", description="Trend direction (increasing, stable, decreasing)")
    percentage_change: float = Field(0.0, description="Percentage change")


class ApiUsageBreakdown(BaseModel):
    """Breakdown of API usage by service and operation."""
    
    services: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    top_operations: Dict[str, float] = Field(default_factory=dict)
    trends: Dict[str, ServiceUsageTrend] = Field(default_factory=dict)


class CostEfficiencyReport(BaseModel):
    """Report on cost efficiency and optimization opportunities."""
    
    efficiency_score: float = Field(0.0, description="Overall cost efficiency score (0-100)")
    breakdown_by_service: Dict[str, float] = Field(default_factory=dict)
    optimization_suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    potential_savings: float = Field(0.0, description="Potential monthly savings")
    risk_assessment: Dict[str, Any] = Field(default_factory=dict)


class AdvancedCostReport(BaseModel):
    """Comprehensive cost report with advanced analytics."""
    
    report_period: str = Field(..., description="Report period")
    timestamp: datetime = Field(default_factory=datetime.now)
    total_cost: float = Field(0.0, description="Total cost for the period")
    cost_by_service: Dict[str, float] = Field(default_factory=dict)
    daily_costs: List[Dict[str, Any]] = Field(default_factory=list)
    usage_breakdown: ApiUsageBreakdown = Field(...)
    predictions: CostPredictions = Field(...)
    efficiency: CostEfficiencyReport = Field(...)
    budget_status: Dict[str, Any] = Field(default_factory=dict)
    execution_stats: Dict[str, Any] = Field(default_factory=dict)


class CostThreshold(BaseModel):
    """Budget threshold configuration."""
    
    threshold: float = Field(..., description="Threshold value (e.g. 0.75 for 75%)")
    notification_type: str = Field("email", description="Notification type")
    message_template: str = Field(..., description="Message template")
    recipients: List[str] = Field(default_factory=list, description="Notification recipients")
    enabled: bool = Field(True, description="Whether this threshold is enabled")


class BudgetConfig(BaseModel):
    """Budget configuration."""
    
    daily_budget: float = Field(0.0, description="Daily budget")
    monthly_budget: float = Field(0.0, description="Monthly budget")
    thresholds: List[CostThreshold] = Field(default_factory=list, description="Budget thresholds")
    auto_pause: bool = Field(False, description="Whether to auto-pause on budget reached")
    rollover_unused: bool = Field(False, description="Whether to roll over unused budget")


class AdvancedCostReporter:
    """Advanced cost reporting with detailed analysis."""
    
    def __init__(self, config: Config, cost_tracker: CostTracker, cost_optimizer: Optional[CostOptimizer] = None):
        self.config = config
        self.cost_tracker = cost_tracker
        self.cost_optimizer = cost_optimizer or CostOptimizer(config, cost_tracker)
        
        # Report history
        self.report_history = []
        self.report_path = Path.home() / ".jirascope" / "reports" / "cost"
        self.report_path.mkdir(parents=True, exist_ok=True)
        
        # Budget configuration
        self.budget = BudgetConfig(
            daily_budget=getattr(config, "daily_budget", 50.0),
            monthly_budget=getattr(config, "monthly_budget", 1000.0),
            thresholds=[
                CostThreshold(
                    threshold=0.5,
                    notification_type="info",
                    message_template="Budget is now at {percent}%: ${current:.2f} of ${total:.2f}",
                    recipients=[]
                ),
                CostThreshold(
                    threshold=0.75,
                    notification_type="warning",
                    message_template="WARNING: Budget is now at {percent}%: ${current:.2f} of ${total:.2f}",
                    recipients=[]
                ),
                CostThreshold(
                    threshold=0.9,
                    notification_type="alert",
                    message_template="ALERT: Budget is nearly exhausted at {percent}%: ${current:.2f} of ${total:.2f}",
                    recipients=[]
                ),
                CostThreshold(
                    threshold=1.0,
                    notification_type="critical",
                    message_template="CRITICAL: Budget has been reached: ${current:.2f} of ${total:.2f}",
                    recipients=[]
                )
            ]
        )
    
    async def generate_comprehensive_cost_report(self, period: str = "month") -> AdvancedCostReport:
        """Generate detailed cost report with trends and predictions."""
        logger.info(f"Generating advanced cost report for period: {period}")
        start_time = time.time()
        
        # Get base cost report from optimizer
        base_report = await self.cost_optimizer.generate_cost_report(period)
        
        # Get API usage breakdown
        usage_breakdown = await self._analyze_api_usage()
        
        # Prepare efficiency report
        efficiency_report = await self._analyze_cost_efficiency(base_report)
        
        # Check budget status
        budget_status = await self._check_budget_status()
        
        # Build the advanced report
        report = AdvancedCostReport(
            report_period=period,
            total_cost=base_report.total_cost,
            cost_by_service=base_report.service_breakdown,
            daily_costs=await self._get_daily_costs(period),
            usage_breakdown=usage_breakdown,
            predictions=base_report.predictions,
            efficiency=efficiency_report,
            budget_status=budget_status,
            execution_stats={
                "processing_time": time.time() - start_time,
                "api_calls_analyzed": await self._count_analyzed_calls(),
                "services_tracked": len(base_report.service_breakdown)
            }
        )
        
        # Save report to history
        self.report_history.append(report)
        if len(self.report_history) > 10:
            self.report_history.pop(0)  # Keep only the last 10 reports
        
        # Save report to file
        report_file = self.report_path / f"cost_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        await self._save_report(report, report_file)
        
        return report
    
    async def generate_daily_cost_summary(self) -> Dict[str, Any]:
        """Generate daily cost summary."""
        logger.info("Generating daily cost summary")
        
        # Get today's date
        today = datetime.now().date()
        
        # Calculate today's costs from tracker
        today_costs = await self._get_costs_for_date(today)
        
        # Compare with average daily cost from recent history
        recent_daily_costs = await self._get_recent_daily_costs(7)  # Last week
        avg_daily_cost = sum(recent_daily_costs) / len(recent_daily_costs) if recent_daily_costs else 0
        
        # Check budget status
        daily_budget = self.budget.daily_budget
        budget_percentage = (today_costs / daily_budget) * 100 if daily_budget > 0 else 0
        
        # Predict end of day cost
        hours_passed = datetime.now().hour + (datetime.now().minute / 60)
        full_day_projection = today_costs * (24 / hours_passed) if hours_passed > 0 else today_costs
        
        return {
            "date": today.isoformat(),
            "current_cost": today_costs,
            "average_daily_cost": avg_daily_cost,
            "budget_percentage": budget_percentage,
            "projected_eod_cost": full_day_projection,
            "budget_status": "over_budget" if today_costs > daily_budget else "under_budget",
            "vs_average": ((today_costs - avg_daily_cost) / avg_daily_cost) * 100 if avg_daily_cost > 0 else 0
        }
    
    async def generate_monthly_cost_projection(self) -> Dict[str, Any]:
        """Generate monthly cost projection based on current trends."""
        logger.info("Generating monthly cost projection")
        
        # Get current month-to-date cost
        now = datetime.now()
        month_start = datetime(now.year, now.month, 1).date()
        mtd_cost = await self._get_costs_for_period(month_start, now.date())
        
        # Calculate days passed and remaining
        days_passed = (now.date() - month_start).days + 1
        days_in_month = self._days_in_month(now.year, now.month)
        days_remaining = days_in_month - days_passed
        
        # Calculate daily rate and projection
        daily_rate = mtd_cost / days_passed if days_passed > 0 else 0
        projected_month_total = mtd_cost + (daily_rate * days_remaining)
        
        # Compare with budget
        monthly_budget = self.budget.monthly_budget
        budget_percentage = (projected_month_total / monthly_budget) * 100 if monthly_budget > 0 else 0
        
        # Get cost trends
        daily_costs = await self._get_daily_costs("month")
        trend = self._calculate_cost_trend([item.get("cost", 0) for item in daily_costs])
        
        return {
            "month": now.strftime("%Y-%m"),
            "days_passed": days_passed,
            "days_remaining": days_remaining,
            "mtd_cost": mtd_cost,
            "projected_month_total": projected_month_total,
            "monthly_budget": monthly_budget,
            "budget_percentage": budget_percentage,
            "daily_rate": daily_rate,
            "trend": trend,
            "is_projected_over_budget": projected_month_total > monthly_budget
        }
    
    async def generate_cost_trend_analysis(self, days: int = 90) -> Dict[str, Any]:
        """Generate long-term cost trend analysis."""
        logger.info(f"Generating cost trend analysis for the last {days} days")
        
        # Get daily costs
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Create date range
        date_range = []
        current_date = start_date
        while current_date <= end_date:
            date_range.append(current_date)
            current_date += timedelta(days=1)
        
        # Get costs for each date
        daily_costs = []
        for date in date_range:
            cost = await self._get_costs_for_date(date)
            daily_costs.append({
                "date": date.isoformat(),
                "cost": cost
            })
        
        # Calculate weekly and monthly averages
        weekly_costs = []
        for i in range(0, len(daily_costs), 7):
            week_costs = [item["cost"] for item in daily_costs[i:i+7]]
            if week_costs:
                weekly_costs.append({
                    "week_starting": daily_costs[i]["date"],
                    "avg_cost": sum(week_costs) / len(week_costs),
                    "total_cost": sum(week_costs)
                })
        
        monthly_costs = []
        month_data = {}
        
        for item in daily_costs:
            date = datetime.fromisoformat(item["date"])
            month_key = f"{date.year}-{date.month:02d}"
            
            if month_key not in month_data:
                month_data[month_key] = {
                    "month": month_key,
                    "costs": [],
                    "total": 0
                }
            
            month_data[month_key]["costs"].append(item["cost"])
            month_data[month_key]["total"] += item["cost"]
        
        for month_key, data in month_data.items():
            monthly_costs.append({
                "month": data["month"],
                "avg_cost": sum(data["costs"]) / len(data["costs"]) if data["costs"] else 0,
                "total_cost": data["total"]
            })
        
        # Calculate trends
        daily_trend = self._calculate_cost_trend([item["cost"] for item in daily_costs])
        weekly_trend = self._calculate_cost_trend([item["total_cost"] for item in weekly_costs])
        monthly_trend = self._calculate_cost_trend([item["total_cost"] for item in monthly_costs])
        
        return {
            "period": f"{start_date.isoformat()} to {end_date.isoformat()}",
            "days_analyzed": days,
            "daily_costs": daily_costs,
            "weekly_costs": weekly_costs,
            "monthly_costs": monthly_costs,
            "trends": {
                "daily": daily_trend,
                "weekly": weekly_trend,
                "monthly": monthly_trend
            },
            "latest_daily_cost": daily_costs[-1]["cost"] if daily_costs else 0,
            "latest_weekly_cost": weekly_costs[-1]["total_cost"] if weekly_costs else 0,
            "latest_monthly_cost": monthly_costs[-1]["total_cost"] if monthly_costs else 0
        }
    
    async def generate_service_comparison_report(self) -> Dict[str, Any]:
        """Generate cost comparison between services."""
        logger.info("Generating service cost comparison report")
        
        # Get service costs
        service_costs = await self._get_service_costs()
        
        # Calculate percentages
        total_cost = sum(service_costs.values())
        service_percentages = {
            service: (cost / total_cost) * 100 if total_cost > 0 else 0
            for service, cost in service_costs.items()
        }
        
        # Get historical comparison for trend
        historical_comparison = await self._get_historical_service_costs()
        
        # Calculate trend for each service
        service_trends = {}
        for service, history in historical_comparison.items():
            service_trends[service] = self._calculate_cost_trend(history)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_cost": total_cost,
            "service_costs": service_costs,
            "service_percentages": service_percentages,
            "service_trends": service_trends,
            "highest_cost_service": max(service_costs.items(), key=lambda x: x[1])[0] if service_costs else None,
            "cost_ratio": {
                "claude_to_embedding": service_costs.get("claude_api", 0) / service_costs.get("embedding_processing", 1) 
                if service_costs.get("embedding_processing", 0) > 0 else 0
            }
        }
    
    async def generate_cost_anomaly_report(self) -> Dict[str, Any]:
        """Detect and report cost anomalies."""
        logger.info("Generating cost anomaly report")
        
        # Get recent daily costs
        daily_costs = await self._get_recent_daily_costs(30)  # Last 30 days
        
        if len(daily_costs) < 7:
            return {
                "error": "Not enough historical data for anomaly detection",
                "data_points": len(daily_costs)
            }
        
        # Calculate baseline (average and standard deviation)
        baseline_avg = sum(daily_costs[:-1]) / (len(daily_costs) - 1)  # Exclude most recent day
        baseline_std = (sum((cost - baseline_avg) ** 2 for cost in daily_costs[:-1]) / (len(daily_costs) - 1)) ** 0.5
        
        # Check most recent day for anomaly
        latest_cost = daily_costs[-1]
        z_score = (latest_cost - baseline_avg) / baseline_std if baseline_std > 0 else 0
        
        # Define anomaly threshold
        threshold = 2.0  # 2 standard deviations
        is_anomaly = abs(z_score) > threshold
        
        # Additional checks for anomalies in specific services
        service_anomalies = await self._check_service_anomalies()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "baseline_avg_cost": baseline_avg,
            "latest_cost": latest_cost,
            "z_score": z_score,
            "is_anomaly": is_anomaly,
            "anomaly_direction": "high" if z_score > 0 else "low" if z_score < 0 else "none",
            "percent_difference": ((latest_cost - baseline_avg) / baseline_avg) * 100 if baseline_avg > 0 else 0,
            "service_anomalies": service_anomalies
        }
    
    async def _analyze_api_usage(self) -> ApiUsageBreakdown:
        """Analyze API usage patterns."""
        # In a full implementation, this would analyze actual API call logs
        # For simplicity, we'll use simulated data
        
        services = {
            "claude_api": {
                "similarity_search": 25.0,
                "duplicate_detection": 35.0,
                "scope_drift_analysis": 20.0,
                "quality_analysis": 15.0
            },
            "embedding_processing": {
                "vector_generation": 40.0,
                "batch_processing": 5.0
            },
            "vector_storage": {
                "storage_fees": 2.0,
                "query_costs": 3.0
            }
        }
        
        # Top operations overall
        top_operations = {
            "duplicate_detection": 35.0,
            "vector_generation": 40.0,
            "similarity_search": 25.0
        }
        
        # Service usage trends
        trends = {
            "claude_api": ServiceUsageTrend(
                service="claude_api",
                usage=[85.0, 87.0, 92.0, 95.0],
                trend_direction="increasing",
                percentage_change=10.5
            ),
            "embedding_processing": ServiceUsageTrend(
                service="embedding_processing",
                usage=[45.0, 44.0, 45.0, 45.0],
                trend_direction="stable",
                percentage_change=0.0
            ),
            "vector_storage": ServiceUsageTrend(
                service="vector_storage",
                usage=[4.0, 4.5, 5.0, 5.0],
                trend_direction="stable", 
                percentage_change=5.0
            )
        }
        
        return ApiUsageBreakdown(
            services=services,
            top_operations=top_operations,
            trends=trends
        )
    
    async def _analyze_cost_efficiency(self, base_report: ComprehensiveCostReport) -> CostEfficiencyReport:
        """Analyze cost efficiency and provide optimization suggestions."""
        # Get optimization suggestions from the optimizer
        optimizations = base_report.optimization_opportunities
        
        # Calculate efficiency score (0-100)
        # This formula is simplified; a real system would use more factors
        potential_savings = sum(opt.potential_savings for opt in optimizations)
        total_cost = base_report.total_cost or 1  # Avoid division by zero
        
        # Higher efficiency score = fewer optimization opportunities relative to total cost
        efficiency_score = 100 * (1 - min(0.5, potential_savings / total_cost))
        
        # Create risk assessment
        risk_assessment = {
            "high_cost_risk": total_cost > self.budget.monthly_budget * 0.8,
            "budget_overrun_probability": min(100, (total_cost / self.budget.monthly_budget) * 100) if self.budget.monthly_budget > 0 else 0,
            "largest_cost_factor": max(base_report.service_breakdown.items(), key=lambda x: x[1])[0] if base_report.service_breakdown else None,
            "optimization_priority": "high" if potential_savings > 50 else "medium" if potential_savings > 20 else "low"
        }
        
        return CostEfficiencyReport(
            efficiency_score=efficiency_score,
            breakdown_by_service=base_report.service_breakdown,
            optimization_suggestions=[opt.dict() for opt in optimizations],
            potential_savings=potential_savings,
            risk_assessment=risk_assessment
        )
    
    async def _check_budget_status(self) -> Dict[str, Any]:
        """Check current budget status."""
        # Get current costs
        daily_cost = await self._get_daily_cost()
        monthly_cost = await self._get_monthly_cost()
        
        # Calculate percentages
        daily_percent = (daily_cost / self.budget.daily_budget) * 100 if self.budget.daily_budget > 0 else 0
        monthly_percent = (monthly_cost / self.budget.monthly_budget) * 100 if self.budget.monthly_budget > 0 else 0
        
        # Determine status
        daily_status = "over_budget" if daily_cost > self.budget.daily_budget else "under_budget"
        monthly_status = "over_budget" if monthly_cost > self.budget.monthly_budget else "under_budget"
        
        # Get active alerts
        active_alerts = []
        
        for threshold in self.budget.thresholds:
            if monthly_percent >= threshold.threshold * 100 and threshold.enabled:
                active_alerts.append({
                    "type": threshold.notification_type,
                    "message": threshold.message_template.format(
                        percent=f"{monthly_percent:.1f}",
                        current=monthly_cost,
                        total=self.budget.monthly_budget
                    ),
                    "threshold": threshold.threshold
                })
        
        return {
            "daily": {
                "budget": self.budget.daily_budget,
                "current": daily_cost,
                "percentage": daily_percent,
                "status": daily_status
            },
            "monthly": {
                "budget": self.budget.monthly_budget,
                "current": monthly_cost, 
                "percentage": monthly_percent,
                "status": monthly_status
            },
            "active_alerts": active_alerts,
            "days_until_budget_exhausted": self._calculate_days_until_budget_exhausted(monthly_cost, monthly_percent)
        }
    
    def _calculate_days_until_budget_exhausted(self, monthly_cost: float, monthly_percent: float) -> int:
        """Calculate days until budget exhaustion based on current rate."""
        if monthly_percent >= 100:
            return 0  # Already exhausted
            
        if monthly_percent <= 0:
            return 30  # No usage yet
        
        # Get current date info
        now = datetime.now()
        days_in_month = self._days_in_month(now.year, now.month)
        days_elapsed = now.day
        
        # Calculate burn rate (% per day)
        burn_rate_per_day = monthly_percent / days_elapsed if days_elapsed > 0 else 0
        
        if burn_rate_per_day <= 0:
            return days_in_month - days_elapsed  # Will last the whole month
            
        # Calculate days remaining at current rate
        days_remaining = (100 - monthly_percent) / burn_rate_per_day
        
        return int(min(days_remaining, days_in_month - days_elapsed))
    
    def _days_in_month(self, year: int, month: int) -> int:
        """Get the number of days in a specific month."""
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        
        return (next_month - datetime(year, month, 1)).days
    
    async def _get_daily_costs(self, period: str = "month") -> List[Dict[str, Any]]:
        """Get daily costs for the specified period."""
        # Calculate date range
        end_date = datetime.now().date()
        
        if period == "week":
            start_date = end_date - timedelta(days=7)
        elif period == "month":
            start_date = end_date - timedelta(days=30)
        elif period == "quarter":
            start_date = end_date - timedelta(days=90)
        else:  # default to month
            start_date = end_date - timedelta(days=30)
        
        # In a real implementation, this would query actual daily costs
        # For simplicity, we'll use simulated data
        daily_costs = []
        current_date = start_date
        
        while current_date <= end_date:
            cost = await self._get_costs_for_date(current_date)
            daily_costs.append({
                "date": current_date.isoformat(),
                "cost": cost
            })
            current_date += timedelta(days=1)
        
        return daily_costs
    
    async def _get_costs_for_date(self, date: datetime.date) -> float:
        """Get costs for a specific date."""
        # In a real implementation, this would filter the cost tracker data by date
        # For simplicity, we'll return a simulated value
        
        if not self.cost_tracker:
            return 0.0
        
        total_cost = self.cost_tracker.get_total_cost()
        days_since_start = (datetime.now().date() - date).days
        
        # Simulate reasonable daily cost with some randomization
        if days_since_start >= 30:
            return 0.0
        
        base_daily = total_cost / 30  # Assume costs are spread over 30 days
        variation = base_daily * (date.day % 5) / 10  # Slight variation based on day of month
        
        return max(0, base_daily + variation)
    
    async def _get_costs_for_period(self, start_date: datetime.date, end_date: datetime.date) -> float:
        """Get total costs between two dates."""
        current_date = start_date
        total = 0.0
        
        while current_date <= end_date:
            total += await self._get_costs_for_date(current_date)
            current_date += timedelta(days=1)
        
        return total
    
    async def _get_recent_daily_costs(self, days: int) -> List[float]:
        """Get daily costs for the specified number of recent days."""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        result = []
        current_date = start_date
        
        while current_date <= end_date:
            cost = await self._get_costs_for_date(current_date)
            result.append(cost)
            current_date += timedelta(days=1)
        
        return result
    
    async def _get_daily_cost(self) -> float:
        """Get cost for the current day."""
        return await self._get_costs_for_date(datetime.now().date())
    
    async def _get_monthly_cost(self) -> float:
        """Get cost for the current month."""
        today = datetime.now().date()
        month_start = datetime(today.year, today.month, 1).date()
        
        return await self._get_costs_for_period(month_start, today)
    
    async def _get_service_costs(self) -> Dict[str, float]:
        """Get costs broken down by service."""
        if not self.cost_tracker:
            return {}
        
        # In a real implementation, this would aggregate costs by service
        # For now, we'll return the total costs by service
        return {
            service: self.cost_tracker.get_total_cost(service)
            for service in self.cost_tracker.costs.keys()
        }
    
    async def _get_historical_service_costs(self) -> Dict[str, List[float]]:
        """Get historical costs by service."""
        # In a real implementation, this would return actual historical data
        # For simplicity, we'll return simulated data
        
        return {
            "claude_api": [80.0, 82.0, 85.0, 90.0, 95.0],
            "embedding_processing": [40.0, 42.0, 45.0, 45.0, 45.0],
            "vector_storage": [3.0, 3.5, 4.0, 4.5, 5.0]
        }
    
    async def _check_service_anomalies(self) -> List[Dict[str, Any]]:
        """Check for anomalies in specific services."""
        # In a real implementation, this would analyze actual service metrics
        # For simplicity, we'll return simulated data
        
        return [
            {
                "service": "claude_api",
                "metric": "avg_tokens_per_call",
                "current_value": 4500,
                "baseline_value": 3200,
                "percent_increase": 40.6,
                "severity": "high",
                "recommendation": "Review prompt design for inefficiencies"
            }
        ]
    
    def _calculate_cost_trend(self, costs: List[float]) -> str:
        """Calculate trend direction from a series of costs."""
        if not costs or len(costs) < 3:
            return "insufficient_data"
        
        # Calculate average change
        changes = [costs[i] - costs[i-1] for i in range(1, len(costs))]
        avg_change = sum(changes) / len(changes) if changes else 0
        
        # Calculate normalized change (as percentage of average cost)
        avg_cost = sum(costs) / len(costs) if costs else 1
        normalized_change = (avg_change / avg_cost) * 100 if avg_cost > 0 else 0
        
        # Determine trend direction
        if normalized_change > 5:
            return "sharply_increasing"
        elif normalized_change > 2:
            return "increasing"
        elif normalized_change < -5:
            return "sharply_decreasing"
        elif normalized_change < -2:
            return "decreasing"
        else:
            return "stable"
    
    async def _count_analyzed_calls(self) -> int:
        """Count the total number of API calls analyzed."""
        if not self.cost_tracker:
            return 0
        
        return sum(len(calls) for calls in self.cost_tracker.costs.values())
    
    async def _save_report(self, report: AdvancedCostReport, file_path: Path) -> bool:
        """Save the report to a file."""
        try:
            with open(file_path, 'w') as f:
                f.write(report.json(indent=2))
            
            logger.info(f"Saved cost report to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save cost report to {file_path}", error=str(e))
            return False