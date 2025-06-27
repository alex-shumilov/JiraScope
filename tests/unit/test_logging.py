"""Tests for logging utilities."""

import logging
import tempfile
from pathlib import Path

import pytest

from src.jirascope.utils.logging import CostTracker, StructuredLogger, setup_logging


def test_cost_tracker():
    """Test CostTracker functionality."""
    tracker = CostTracker()

    # Add some costs
    tracker.add_cost("claude", "analysis", 0.001, {"tokens": 1000})
    tracker.add_cost("claude", "analysis", 0.002, {"tokens": 2000})
    tracker.add_cost("lmstudio", "embedding", 0.0005, {"vectors": 10})

    # Test total costs
    assert tracker.get_total_cost() == 0.0035
    assert tracker.get_total_cost("claude") == 0.003
    assert tracker.get_total_cost("lmstudio") == 0.0005
    assert tracker.get_total_cost("nonexistent") == 0.0

    # Test session summary
    summary = tracker.get_session_summary()
    assert summary["total_cost"] == 0.0035
    assert summary["costs_by_service"]["claude"] == 0.003
    assert summary["costs_by_service"]["lmstudio"] == 0.0005
    assert summary["operation_count"] == 3


def test_setup_logging_console_only():
    """Test logging setup with console output only."""
    cost_tracker = setup_logging(log_level="DEBUG", enable_cost_tracking=True)

    assert isinstance(cost_tracker, CostTracker)

    # Test that root logger is configured
    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG

    # Test that cost logger exists
    cost_logger = logging.getLogger("jirascope.costs")
    assert hasattr(cost_logger, "cost_tracker")
    assert hasattr(cost_logger, "log_cost")


def test_setup_logging_with_file():
    """Test logging setup with file output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.log"

        cost_tracker = setup_logging(log_level="INFO", log_file=log_file, enable_cost_tracking=True)

        assert isinstance(cost_tracker, CostTracker)

        # Log a message
        logger = logging.getLogger("test")
        logger.info("Test message")

        # Check file was created and contains the message
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content


def test_structured_logger():
    """Test StructuredLogger functionality."""
    # Setup logging first
    setup_logging(log_level="DEBUG", enable_cost_tracking=True)

    logger = StructuredLogger("test")

    # Test different log levels
    logger.info("Info message", key1="value1", key2="value2")
    logger.warning("Warning message", error_code=404)
    logger.error("Error message", exception="TestException")
    logger.debug("Debug message", debug_info={"level": "verbose"})

    # Test operation logging
    logger.log_operation("test_operation", 1.5, success=True, items=5)
    logger.log_operation("failed_operation", 0.5, success=False, error="timeout")

    # Test cost logging
    logger.log_cost("test_service", "test_operation", 0.001, {"tokens": 500})


def test_structured_logger_without_cost_tracking():
    """Test StructuredLogger when cost tracking is disabled."""
    setup_logging(log_level="INFO", enable_cost_tracking=False)

    logger = StructuredLogger("test")

    # Should not have cost tracker
    assert logger.cost_tracker is None

    # Should still be able to log costs (but won't track them)
    logger.log_cost("test_service", "test_operation", 0.001)


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration after each test."""
    yield
    # Clear all handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Reset level
    root_logger.setLevel(logging.WARNING)
