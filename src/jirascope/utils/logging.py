"""Structured logging configuration with cost metrics."""

import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class CostTracker:
    """Track API costs and usage metrics."""

    def __init__(self):
        self.costs = {}
        self.session_start = datetime.now()

    def add_cost(
        self, service: str, operation: str, cost: float, details: dict[str, Any] | None = None
    ):
        """Add a cost entry."""
        timestamp = datetime.now()

        if service not in self.costs:
            self.costs[service] = []

        entry = {
            "timestamp": timestamp.isoformat(),
            "operation": operation,
            "cost": cost,
            "details": details or {},
        }

        self.costs[service].append(entry)

    def get_total_cost(self, service: str | None = None) -> float:
        """Get total cost for a service or all services."""
        if service:
            return sum(entry["cost"] for entry in self.costs.get(service, []))

        return sum(sum(entry["cost"] for entry in entries) for entries in self.costs.values())

    def get_session_summary(self) -> dict[str, Any]:
        """Get summary of session costs."""
        session_duration = (datetime.now() - self.session_start).total_seconds()

        return {
            "session_duration_seconds": session_duration,
            "total_cost": self.get_total_cost(),
            "costs_by_service": {service: self.get_total_cost(service) for service in self.costs},
            "operation_count": sum(len(entries) for entries in self.costs.values()),
        }


class JiraScopeLogger(logging.Logger):
    """Custom logger class with cost tracking capabilities."""

    cost_tracker: CostTracker | None = None

    def log_cost(
        self, service: str, operation: str, cost: float, details: dict[str, Any] | None = None
    ):
        """Log a cost entry if cost tracking is enabled."""
        if self.cost_tracker:
            self.cost_tracker.add_cost(service, operation, cost, details)
            self.info(f"{service}.{operation}", extra={"cost": cost})


logging.setLoggerClass(JiraScopeLogger)
logging.LogRecord.cost = 0.0


class CostTrackingFormatter(logging.Formatter):
    """Custom formatter that includes cost information."""

    def format(self, record):
        # Add cost information if available
        if hasattr(record, "cost"):
            record.msg = f"[COST: ${record.cost:.4f}] {record.msg}"  # type: ignore[attr-defined]

        return super().format(record)


def setup_logging(
    log_level: str = "INFO", log_file: Path | None = None, enable_cost_tracking: bool = True
) -> CostTracker | None:
    """Setup structured logging with optional cost tracking."""

    log_level = getattr(logging, log_level.upper())

    # Create formatters
    console_formatter = CostTrackingFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)

    handlers = [console_handler]

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)  # type: ignore[arg-type]

    # Configure root logger
    logging.basicConfig(level=log_level, handlers=handlers, force=True)

    # Set specific logger levels
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.INFO)

    # Initialize cost tracker
    cost_tracker = CostTracker() if enable_cost_tracking else None

    # Get the cost logger and assign the tracker
    cost_logger = logging.getLogger("jirascope.costs")
    if isinstance(cost_logger, JiraScopeLogger):
        cost_logger.cost_tracker = cost_tracker

    return cost_tracker


class StructuredLogger:
    """Structured logger for consistent logging across the application."""

    def __init__(self, name: str):
        self.logger: JiraScopeLogger = logging.getLogger(name)  # type: ignore
        self.cost_tracker = self.logger.cost_tracker

    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        extra = {"structured_data": kwargs} if kwargs else {}
        self.logger.info(message, extra=extra)

    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        extra = {"structured_data": kwargs} if kwargs else {}
        self.logger.warning(message, extra=extra)

    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        extra = {"structured_data": kwargs} if kwargs else {}
        self.logger.error(message, extra=extra)

    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        extra = {"structured_data": kwargs} if kwargs else {}
        self.logger.debug(message, extra=extra)

    def log_operation(self, operation: str, duration: float, success: bool = True, **kwargs):
        """Log an operation with timing and success status."""
        status = "SUCCESS" if success else "FAILED"
        message = f"Operation {operation} {status} in {duration:.2f}s"

        extra = {"operation": operation, "duration": duration, "success": success, **kwargs}

        if success:
            self.info(message, **extra)
        else:
            self.error(message, **extra)

    def log_cost(
        self, service: str, operation: str, cost: float, details: dict[str, Any] | None = None
    ):
        """Log cost information."""
        self.logger.log_cost(service, operation, cost, details)
        self.logger.info(
            f"Cost: {service}.{operation} = ${cost:.4f}",
            extra={"cost": cost, "service": service, "operation": operation},
        )
