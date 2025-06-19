"""CLI utility functions and classes."""

import json
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console
from rich.table import Table

from jirascope.core.config import Config

console = Console()


class CostTracker:
    """Track costs for CLI operations with budget warnings."""
    
    def __init__(self):
        self.session_costs: Dict[str, float] = {}
        self.warnings_shown = set()
    
    def track_operation(self, operation: str, cost: float):
        """Track cost for CLI operation"""
        if operation not in self.session_costs:
            self.session_costs[operation] = 0.0
        
        self.session_costs[operation] += cost
        
        # Check for budget warnings
        total_cost = sum(self.session_costs.values())
        if total_cost > 5.0 and "budget_5" not in self.warnings_shown:
            console.print("[yellow]⚠️  Session cost exceeded $5.00[/yellow]")
            self.warnings_shown.add("budget_5")
        elif total_cost > 10.0 and "budget_10" not in self.warnings_shown:
            console.print("[red]🚨 Session cost exceeded $10.00[/red]")
            self.warnings_shown.add("budget_10")
    
    def get_total_cost(self) -> float:
        """Get total session cost"""
        return sum(self.session_costs.values())
    
    def display_cost_summary(self):
        """Display cost breakdown table"""
        if not self.session_costs:
            console.print("[dim]No costs tracked this session[/dim]")
            return
        
        cost_table = Table(title="Session Cost Summary")
        cost_table.add_column("Operation", style="cyan")
        cost_table.add_column("Cost", style="green")
        
        for operation, cost in self.session_costs.items():
            cost_table.add_row(operation, f"${cost:.4f}")
        
        total = sum(self.session_costs.values())
        cost_table.add_row("TOTAL", f"${total:.4f}", style="bold")
        
        console.print(cost_table)


def load_config_file(config_path: Path) -> Config:
    """Load configuration from file"""
    try:
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        else:
            # Assume YAML/TOML format
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        
        # Create config with loaded data
        config = Config()
        
        # Apply configuration values
        if 'jira' in config_data:
            for key, value in config_data['jira'].items():
                setattr(config.jira, key, value)
        
        if 'claude' in config_data:
            for key, value in config_data['claude'].items():
                setattr(config.claude, key, value)
        
        if 'lmstudio' in config_data:
            for key, value in config_data['lmstudio'].items():
                setattr(config.lmstudio, key, value)
        
        if 'qdrant' in config_data:
            for key, value in config_data['qdrant'].items():
                setattr(config.qdrant, key, value)
        
        return config
        
    except Exception as e:
        console.print(f"[red]Error loading config file: {e}[/red]")
        console.print("[yellow]Using default configuration[/yellow]")
        return Config()


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_cost(cost: float) -> str:
    """Format cost with appropriate precision"""
    if cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1.0:
        return f"${cost:.3f}"
    else:
        return f"${cost:.2f}"


def validate_threshold(value: float) -> float:
    """Validate similarity threshold value"""
    if not (0.0 <= value <= 1.0):
        raise ValueError("Threshold must be between 0.0 and 1.0")
    return value


def validate_project_key(value: str) -> str:
    """Validate Jira project key format"""
    if not value or not value.replace('-', '').replace('_', '').isalnum():
        raise ValueError("Invalid project key format")
    return value.upper()


def confirm_expensive_operation(estimated_cost: float, budget: Optional[float] = None) -> bool:
    """Confirm expensive operations with user"""
    if budget and estimated_cost > budget:
        console.print(f"[red]Operation cost ${estimated_cost:.4f} exceeds budget ${budget:.2f}[/red]")
        return False
    
    if estimated_cost > 5.0:
        console.print(f"[yellow]This operation will cost approximately ${estimated_cost:.2f}[/yellow]")
        return console.input("Continue? [y/N]: ").lower().startswith('y')
    
    return True