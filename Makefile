.PHONY: help lint lint-fix format type-check check-all install-dev clean

# Default target
help:
	@echo "JiraScope Development Commands"
	@echo "============================="
	@echo "lint          - Run ruff linter (check only)"
	@echo "lint-fix      - Run ruff linter with auto-fix"
	@echo "format        - Format code with ruff"
	@echo "format-check  - Check code formatting without changes"
	@echo "type-check    - Run mypy type checker"
	@echo "check-all     - Run all checks (lint + format + type)"
	@echo "install-dev   - Install development dependencies"
	@echo "clean         - Clean cache files"

# Linting
lint:
	@echo "🔍 Running ruff linter..."
	ruff check . --show-fixes

lint-fix:
	@echo "🔧 Running ruff linter with auto-fix..."
	ruff check . --fix --show-fixes

# Formatting
format:
	@echo "🎨 Formatting code with ruff..."
	ruff format .

format-check:
	@echo "🎨 Checking code formatting..."
	ruff format --check .

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
	pip install ruff mypy black

# Cleanup
clean:
	@echo "🧹 Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage .pytest_cache/
