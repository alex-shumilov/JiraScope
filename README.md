# JiraScope

AI-powered Jira work item analysis and management tool.

## Installation

```bash
poetry install
```

## Testing

JiraScope has a comprehensive test suite with unit, integration, and component tests.

### Prerequisites

Before running tests, ensure the following dependencies are set up:

#### 1. LMStudio
- Install and run [LMStudio](https://lmstudio.ai/)
- Ensure the local server is running on the default port

#### 2. Embedding Model
Set the embedding model configuration in your environment variables:
```bash
export EMBEDDING_MODEL="your-embedding-model-name"
export EMBEDDING_MODEL_URL="http://localhost:1234/v1"  # LMStudio default
```

#### 3. Claude Code
- Install [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI
- Ensure you have valid Anthropic API credentials configured

#### 4. MCP Configuration
- Set up MCP (Model Context Protocol) configuration
- Ensure MCP clients are properly configured for Jira integration

### Running Tests

Run all tests:
```bash
pytest
```

Run specific test categories:
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only  
pytest tests/integration/

# Component tests only
pytest tests/test_*.py
```

Run with verbose output:
```bash
pytest -v
```

### Test Structure

- **Unit Tests** (`tests/unit/`) - Configuration, logging, and model validation
- **Integration Tests** (`tests/integration/`) - External client integrations
- **Component Tests** (root level) - Analysis components and pipeline processors

## Usage

Coming soon...