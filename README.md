# JiraScope

AI-powered Jira work item analysis and management tool that provides semantic analysis of work items to improve quality, identify duplicates, and analyze scope drift over time.

## Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/JiraScope.git
cd JiraScope

# Install dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell

# Install the CLI
pip install -e .

# Verify installation
jirascope --version
```

### Using pip

```bash
# Install directly from source
pip install -e .

# Verify installation
jirascope --version
```

### Using Docker

```bash
# Build and start services
docker-compose up -d

# Run commands in container
docker-compose exec jirascope-dev jirascope health
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

## CLI Usage

JiraScope provides a comprehensive command-line interface (CLI) for semantic work item analysis.

### Setting Up the CLI

The CLI tool is installed as an entry point when you install the package:

1. Ensure the package is installed using either pip or poetry:
   ```bash
   # If using pip
   pip install -e .

   # Or if using poetry
   poetry install
   ```

2. After installation, the `jirascope` command will be available in your PATH

3. Verify the installation:
   ```bash
   jirascope --version
   ```

4. If the command is not found, ensure:
   - Your virtual environment is activated (if using one)
   - The package was installed with the `-e` flag for development mode
   - The `src` directory is in your Python path

> **Note:** The current implementation has two CLI modules: `src/cli/` and `src/jirascope/cli/`. The entry point in `pyproject.toml` points to `jirascope.cli.main:cli`, which is the one that will be available after installation.

### Basic Commands

#### Health Check

Check if all services (Jira, LM Studio, Qdrant, Claude) are running correctly:

```bash
jirascope health
```

#### Fetch Data

Extract and process Jira data:

```bash
# Basic fetch for a specific project
jirascope fetch --project PROJECT_KEY

# Incremental fetch (only updates changed items)
jirascope fetch --project PROJECT_KEY --incremental

# With additional JQL filtering
jirascope fetch --project PROJECT_KEY --jql "created >= -30d"
```

### Analysis Commands

#### Find Duplicates

Detect potential duplicate work items:

```bash
# Basic duplicate detection
jirascope analyze duplicates --threshold 0.7

# For a specific project
jirascope analyze duplicates --project PROJECT_KEY --threshold 0.8
```

#### Quality Analysis

Analyze the quality of work item descriptions:

```bash
# Analyze a specific work item
jirascope analyze quality PROJ-123
```

#### Cross-Epic Analysis

Find work items that might belong to different epics:

```bash
# For all projects
jirascope analyze cross-epic

# For a specific project
jirascope analyze cross-epic --project PROJECT_KEY
```

#### Technical Debt Analysis

Cluster technical debt items for prioritization:

```bash
# For all projects
jirascope analyze tech-debt

# For a specific project
jirascope analyze tech-debt --project PROJECT_KEY
```

#### Template Generation

Generate template from high-quality examples:

```bash
# For all projects
jirascope analyze template --issue-type Story

# For a specific project
jirascope analyze template --issue-type Bug --project PROJECT_KEY
```

### Cost Management

Display session cost summary:

```bash
jirascope cost
```

JiraScope includes built-in cost tracking:
- Automatic cost calculation for all operations
- Budget warnings at threshold levels
- Cost tracking by service type
- Session cost summaries

### Web Dashboard

Start the web interface:

```bash
jirascope web

# Custom host and port
jirascope web --host 0.0.0.0 --port 8080

# With auto-reload for development
jirascope web --reload
```

### Global Options

Options available for all commands:

- `--config, -c`: Specify configuration file path
- `--dry-run`: Preview actions without executing
- `--verbose, -v`: Enable verbose output
- `--help`: Show help information

### Configuration

Create a configuration file (JSON or YAML):

```json
{
  "jira": {
    "mcp_endpoint": "your-jira-mcp-endpoint"
  },
  "claude": {
    "api_key": "your-claude-api-key",
    "model": "claude-3-5-sonnet-20241022"
  },
  "lmstudio": {
    "endpoint": "http://localhost:1234/v1",
    "embedding_model": "bge-large-en-v1.5"
  },
  "qdrant": {
    "url": "http://localhost:6333"
  }
}
```

### Export Formats

Results can be exported in multiple formats:

- **JSON**: Structured data with full details
- **CSV**: Tabular format for spreadsheet analysis

### Complete Workflow Example

```bash
# 1. Check health of all services
jirascope health

# 2. Fetch project data
jirascope fetch --project PROJECT_KEY

# 3. Find duplicates
jirascope analyze duplicates --project PROJECT_KEY --threshold 0.8

# 4. Quality analysis for a specific work item
jirascope analyze quality PROJECT_KEY-123

# 5. Search for similar work items
jirascope search --query "authentication fails when API token is invalid"

# 6. Show cost summary
jirascope cost
```

For more detailed instructions, see the [CLI_USAGE.md](CLI_USAGE.md) document.

## Features

- **Complete Data Pipeline**: Extract work items from Jira and build semantic vector database
- **Analysis Engines**:
  - Similarity Analysis: Find duplicate work items across projects
  - Content Quality: Analyze and improve work item descriptions
  - Cross-Epic Analysis: Ensure work items belong in the right epics
  - Scope Drift Detection: Track changes in requirements over time
  - Structure Analysis: Identify and cluster technical debt
  - Template Inference: Generate templates from high-quality examples
- **Interface Options**:
  - Comprehensive CLI with cost tracking
  - Web dashboard with real-time updates
- **Cost Management**:
  - Real-time cost tracking
  - Budget controls and warnings
  - Cost optimization with batching and incremental updates

## System Requirements

- Python 3.11+
- 4GB+ RAM (8GB recommended)
- 1GB+ storage for vector data
- Network access to Jira and AI services

## License

This project is proprietary software.

## Environment Setup

### Prerequisites
- Python 3.11 or higher
- Poetry for dependency management
- Qdrant vector database (running on localhost:6333)
- LMStudio or compatible embedding service (running on localhost:1234)

### Configuration

1. **Copy the environment template:**
   ```bash
   cp .env.dist .env
   ```

2. **Edit the `.env` file with your configuration:**
   ```bash
   # Required: Set your Jira MCP endpoint
   JIRA_MCP_ENDPOINT=http://your-jira-mcp-server:port

   # Optional: Adjust other settings as needed
   QDRANT_URL=http://localhost:6333
   LMSTUDIO_ENDPOINT=http://localhost:1234/v1
   CLAUDE_API_KEY=your-claude-api-key
   ```

3. **Install dependencies:**
   ```bash
   poetry install
   ```

4. **Activate the environment:**
   ```bash
   source $(poetry env info --path)/bin/activate
   ```

   After activation, you can use `jirascope` command directly instead of `poetry run jirascope`.

5. **Verify setup:**
   ```bash
   jirascope health
   ```

### SSE Authentication (Atlassian Cloud)

For SSE-based MCP endpoints like Atlassian Cloud, JiraScope supports OAuth/SSO authentication with browser redirect flow, similar to AWS CLI.

**Configuration for Atlassian Cloud:**
```bash
# Set SSE endpoint in .env
JIRA_MCP_ENDPOINT=https://mcp.atlassian.com/v1/sse

# Optional: Configure OAuth settings
JIRA_SSE_CLIENT_ID=jirascope-cli
JIRA_SSE_REDIRECT_PORT=8081
```

**Authentication Commands:**

```bash
# Authenticate with SSE endpoint (opens browser)
jirascope auth

# Check authentication status
jirascope auth-status

# Clear cached authentication tokens
jirascope auth-clear
```

**How SSE Authentication Works:**

1. **Automatic Detection**: JiraScope detects SSE endpoints automatically
2. **Browser OAuth Flow**: Opens browser for OAuth/SSO authentication
3. **Token Caching**: Securely caches tokens in `~/.jirascope/auth_cache.json`
4. **Auto-Refresh**: Automatically refreshes expired tokens
5. **PKCE Security**: Uses PKCE for secure OAuth flow

**Authentication Flow:**
```bash
$ jirascope auth
🔐 Opening browser for authentication...
⏳ Waiting for authentication...
✅ Authentication successful!
   Token expires: 1640995200.0
   Saved to: /Users/user/.jirascope/auth_cache.json
```

After authentication, all JiraScope commands will automatically use the cached tokens.
