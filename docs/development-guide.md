# JiraScope Development Guide

This guide consolidates all development-related information for JiraScope contributors and maintainers.

## 📋 Table of Contents

- [Development Setup](#development-setup)
- [Code Quality Standards](#code-quality-standards)
- [Architecture & Patterns](#architecture--patterns)
- [Testing Guidelines](#testing-guidelines)
- [Contribution Workflow](#contribution-workflow)
- [Debugging & Troubleshooting](#debugging--troubleshooting)

## 🚀 Development Setup

### Prerequisites

- **Python 3.11+** (defined in [pyproject.toml](../pyproject.toml))
- **Poetry** for dependency management
- **Docker** for external services (Qdrant)
- **External Services**: LMStudio, Qdrant, Claude API access

### Installation & Environment

```bash
# Clone and install dependencies
git clone <repository-url>
cd jirascope
poetry install
poetry shell

# Install in development mode
pip install -e .
```

### ⚠️ Important: Always Use Poetry

**The system Python is not compatible with this project.** Always use `poetry run` for Python commands:

- ✅ **Correct**: `poetry run python -m jirascope`
- ✅ **Correct**: `poetry run pytest`
- ✅ **Correct**: `poetry run mypy src/`
- ❌ **Incorrect**: `python -m jirascope`
- ❌ **Incorrect**: `pytest`
- ❌ **Incorrect**: `mypy src/`

### Environment Configuration

1. **Copy Configuration Template**:
   ```bash
   cp config/env.example .env
   ```

2. **Configure Required Services**:
   ```bash
   # Required
   JIRA_MCP_ENDPOINT=https://your-jira.atlassian.net
   JIRA_USERNAME=your.email@company.com
   JIRA_API_TOKEN=your_jira_api_token

   # Optional (with defaults)
   QDRANT_URL=http://localhost:6333
   LMSTUDIO_ENDPOINT=http://localhost:1234/v1
   CLAUDE_API_KEY=your_claude_api_key
   ```

3. **Start External Services**:
   ```bash
   # Start Qdrant vector database
   docker run -p 6333:6333 qdrant/qdrant

   # Start LMStudio (if using local AI)
   # Download and run LMStudio from https://lmstudio.ai/
   ```

## 📏 Code Quality Standards

### Formatting and Linting

- **Black Formatter**: 100 character line length
- **MyPy Type Checker**: Static type checking for Python
- **Configuration**: [pyproject.toml](../pyproject.toml)

```bash
# Format code
poetry run black src/ tests/

# Lint code
poetry run mypy src/

# Auto-fix linting issues
poetry run mypy src/ --fix

# Type checking (if configured)
poetry run mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
poetry run pre-commit install

# Run manually
poetry run pre-commit run --all-files
```

### Code Quality Principles

#### Anti-Overcoding Rules
- **YAGNI**: Don't implement features until actually needed
- **Simplicity First**: Choose straightforward implementations over clever ones
- **Function Limits**: Maximum 20 lines per function (prefer 10-15)
- **Class Limits**: Maximum 200 lines per class (prefer 100-150)
- **File Limits**: Maximum 500 lines per file

#### SOLID Principles
- **Single Responsibility**: Each class/function has one clear purpose
- **Open-Closed**: Extend behavior through composition, not modification
- **Liskov Substitution**: Subclasses must be substitutable for base classes
- **Interface Segregation**: Create focused, specific interfaces
- **Dependency Inversion**: Depend on abstractions, not concrete implementations

#### KISS Principle
- Prefer simple solutions over complex ones
- Write code readable by developers of all skill levels
- Avoid unnecessary abstractions and patterns
- Question every layer of complexity

## 🏗️ Architecture & Patterns

### Code Organization

```
src/jirascope/
├── analysis/           # Analysis engine modules
├── clients/           # External service clients
├── core/             # Core configuration and utilities
├── extractors/       # Data extraction components
├── mcp_server/       # MCP protocol server
├── pipeline/         # Data processing pipeline
├── rag/             # RAG implementation
└── utils/           # Shared utilities
```

### Design Patterns

#### Async/Await Pattern
- All external service calls use `async`/`await`
- Main CLI commands wrap async functions with `asyncio.run()`
- Client classes use async context managers (`async with`)

```python
# Correct pattern
async def main():
    async with QdrantClient(config) as client:
        await client.health_check()

# CLI entry point
def cli_command():
    asyncio.run(main())
```

#### Configuration Management
- All modules receive config via dependency injection
- Configuration loaded once in CLI entry point
- Config class defined in `src/jirascope/core/config.py`

```python
# Correct pattern
class AnalysisEngine:
    def __init__(self, config: Config):
        self.config = config
```

#### Client Pattern
All external service clients follow this structure:
- Async context manager implementation
- Health check method
- Configuration-based initialization
- Located in `src/jirascope/clients/`

```python
class ExampleClient:
    async def __aenter__(self):
        # Setup connection
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup connection
        pass

    async def health_check(self) -> bool:
        # Verify service availability
        pass
```

#### Error Handling
- Use try/except blocks for external service calls
- Log errors using the logging utility from `src/jirascope/utils/logging.py`
- Return structured error responses where appropriate

```python
from jirascope.utils.logging import logger

async def process_data():
    try:
        result = await external_service.call()
        return result
    except ServiceError as e:
        logger.error(f"Service call failed: {e}")
        return ErrorResponse(message=str(e))
```

### Type Hints
- Use type hints throughout the codebase
- Leverage Pydantic models for data validation
- Import types from typing module as needed

```python
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class WorkItem(BaseModel):
    key: str
    summary: str
    description: Optional[str] = None

async def process_items(items: List[WorkItem]) -> Dict[str, Any]:
    # Implementation
    pass
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/              # Unit tests (isolated components)
├── integration/       # Integration tests (components together)
├── functional/        # Functional tests (user-facing features)
├── performance/       # Performance and scalability tests
├── fixtures/          # Test data and utilities
└── conftest.py       # Shared fixtures
```

### Running Tests

```bash
# Run all tests with coverage
poetry run pytest

# Run specific test categories
poetry run pytest tests/unit/
poetry run pytest tests/integration/
poetry run pytest tests/functional/

# Run with verbose output
poetry run pytest -v

# Run with coverage report
poetry run pytest --cov-report=html
open htmlcov/index.html
```

### Test Quality Standards

#### ✅ Good Test Patterns
- **Clear Purpose**: Test name explains what behavior is being verified
- **Meaningful Assertions**: Verify actual business logic outcomes
- **Realistic Data**: Test data reflects real-world scenarios
- **Error Coverage**: Both success and failure scenarios tested
- **Behavior Focus**: Test what code does, not how it does it

```python
async def test_similarity_analyzer_identifies_duplicates():
    """Test that similar work items are correctly identified as potential duplicates."""
    # Given
    similar_items = [
        WorkItem(key="PROJ-1", summary="Login bug fix"),
        WorkItem(key="PROJ-2", summary="Authentication issue repair")
    ]

    # When
    result = await analyzer.find_duplicates(similar_items, threshold=0.8)

    # Then
    assert len(result.duplicate_groups) == 1
    assert result.duplicate_groups[0].similarity_score >= 0.8
    assert "PROJ-1" in [item.key for item in result.duplicate_groups[0].items]
    assert "PROJ-2" in [item.key for item in result.duplicate_groups[0].items]
```

#### ❌ Anti-Patterns to Avoid
- **Mock Overuse**: Mocking everything instead of testing behavior
- **Implementation Testing**: Testing internal method calls vs outcomes
- **Magic Values**: Hardcoded numbers without explanation
- **Meaningless Assertions**: Assertions that always pass
- **Copy-Paste Tests**: Duplicate test logic without clear purpose

### Test Fixtures

Use shared fixtures from `conftest.py`:

```python
# Available fixtures
def test_example(mock_config, sample_work_items, mock_qdrant_client):
    # Use pre-configured test data
    pass
```

### Coverage Requirements

- **Minimum Coverage**: 80% (configurable in `pyproject.toml`)
- **HTML Reports**: Generated in `htmlcov/` directory
- **XML Reports**: Available for CI/CD integration

## 🔄 Contribution Workflow

### 1. Development Process

```bash
# 1. Create feature branch
git checkout -b feature/your-feature-name

# 2. Make changes following code quality standards
# 3. Add/update tests
# 4. Run quality checks
poetry run mypy src/
poetry run pytest

# 5. Commit with descriptive message
git commit -m "feat: add similarity threshold configuration"

# 6. Push and create pull request
git push origin feature/your-feature-name
```

### 2. Code Review Checklist

- [ ] **Code Quality**: Follows anti-overcoding, SOLID, and KISS principles
- [ ] **Tests**: Adequate test coverage with meaningful assertions
- [ ] **Documentation**: Updated relevant documentation
- [ ] **Type Hints**: Proper type annotations
- [ ] **Error Handling**: Appropriate error handling and logging
- [ ] **Performance**: No obvious performance regressions
- [ ] **Security**: No security vulnerabilities introduced

### 3. Pull Request Guidelines

- **Title**: Use conventional commit format (`feat:`, `fix:`, `docs:`, etc.)
- **Description**: Explain what changes were made and why
- **Tests**: Include test results and coverage information
- **Documentation**: Note any documentation updates needed

## 🐛 Debugging & Troubleshooting

### Logging

```python
from jirascope.utils.logging import logger

# Use appropriate log levels
logger.debug("Detailed debugging information")
logger.info("General information")
logger.warning("Warning messages")
logger.error("Error messages")
logger.critical("Critical issues")
```

### Debug Mode

```bash
# Enable verbose logging
poetry run jirascope --verbose --log-file debug.log [command]

# Environment variable for debug mode
DEBUG=true poetry run python -m jirascope
```

### Common Issues

#### Connection Problems
```bash
# Check service health
poetry run jirascope health

# Verify configuration
poetry run jirascope status

# Test individual services
curl http://localhost:6333/health  # Qdrant
curl http://localhost:1234/health  # LMStudio
```

#### Import Errors
```bash
# Ensure proper installation
poetry install
pip install -e .

# Check Python path
poetry run python -c "import jirascope; print(jirascope.__file__)"
```

#### Test Failures
```bash
# Run specific failing test
poetry run pytest tests/path/to/test.py::test_function -v

# Run with debugging
poetry run pytest --pdb tests/path/to/test.py::test_function

# Check test data
poetry run pytest --fixtures tests/
```

### Performance Debugging

```bash
# Profile application
poetry run python -m cProfile -o profile.stats -m jirascope [command]

# Analyze profile
poetry run python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Memory profiling (if memory-profiler installed)
poetry run mprof run python -m jirascope [command]
poetry run mprof plot
```

## 📚 Additional Resources

- **[Testing Guide](../tests/README.md)**: Comprehensive testing documentation
- **[Test Quality Analysis](test-integrity-analysis.md)**: Test improvement recommendations
- **[Linting Guide](LINTING.md)**: Code formatting and linting standards
- **[API Documentation](api/README.md)**: REST API reference
- **[Phase 5 Summary](PHASE5_SUMMARY.md)**: Advanced analytics architecture

## ✅ Quick Reference

### Daily Development Commands
```bash
# Start development environment
poetry shell
docker run -p 6333:6333 qdrant/qdrant

# Code quality checks
poetry run mypy src/ --fix
poetry run pytest

# Run application
poetry run jirascope health
poetry run jirascope mcp-server
```

### Before Committing
```bash
# Full quality check
poetry run mypy src/
poetry run pytest --cov-fail-under=80
poetry run pre-commit run --all-files
```

### Emergency Debugging
```bash
# Quick health check
poetry run jirascope health

# Full diagnostic
poetry run jirascope status --verbose

# Reset environment
rm -rf __pycache__ .pytest_cache htmlcov/
poetry install --no-cache
```

---

**Remember**: Follow the anti-overcoding rules, SOLID principles, and KISS principle. Write simple, testable, maintainable code that solves real problems without unnecessary complexity.
