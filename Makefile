.PHONY: help lint lint-fix format type-check check-all install-dev clean

# Default target
help:
	@echo "JiraScope Development Commands"
	@echo "============================="
	@echo "type-check    - Run mypy type checker"
	@echo "check-all     - Run all checks (lint + format + type)"
	@echo "install-dev   - Install development dependencies"
	@echo "clean         - Clean cache files"

# Type checking
type-check:
	@echo "🔍 Running mypy type checker..."
	mypy src/ --show-error-codes --pretty

# Combined checks
check-all: lint format-check type-check
	@echo "✅ All checks completed!"

# Development setup
install-dev:
	@echo "📦 Installing development dependencies..."
	pip install -e .

# Cleanup
clean:
	@echo "🧹 Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage .pytest_cache/
