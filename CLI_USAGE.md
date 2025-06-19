# JiraScope CLI Usage Guide

JiraScope provides a comprehensive command-line interface for semantic work item analysis.

## Installation

```bash
# Install from source
pip install -e .

# Or using Docker
docker-compose up -d
```

## Basic Commands

### Health Check
Check if all services are running:
```bash
jirascope health-check
```

### Sync Data
Extract and process Jira data:
```bash
# Basic sync
jirascope sync --project AAA

# With cost tracking
jirascope sync --project AAA --show-costs

# Incremental sync
jirascope sync --project AAA --incremental

# Custom batch size
jirascope sync --project AAA --batch-size 50
```

### Analysis Commands

#### Find Duplicates
```bash
# Basic duplicate detection
jirascope analyze duplicates --threshold 0.8

# With cost estimation
jirascope analyze duplicates --threshold 0.8 --cost-estimate

# Export results
jirascope analyze duplicates --threshold 0.8 --output duplicates.json
```

#### Quality Analysis
```bash
# Basic quality check
jirascope analyze quality --project AAA

# Using Claude AI
jirascope analyze quality --project AAA --use-claude

# With budget limit
jirascope analyze quality --project AAA --use-claude --budget 5.00
```

## Global Options

- `--config, -c`: Specify configuration file
- `--dry-run`: Preview actions without executing
- `--verbose, -v`: Enable verbose output
- `--help`: Show help information

## Configuration

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

## Docker Usage

### Development
```bash
# Start development environment
docker-compose up -d

# Run commands in container
docker-compose exec jirascope-dev jirascope health-check
docker-compose exec jirascope-dev jirascope sync --project AAA
```

### Production
```bash
# Build production image
docker build --target prod -t jirascope:prod .

# Run production container
docker run -d --name jirascope \
  -e CLAUDE_API_KEY=your_key \
  -e JIRA_MCP_ENDPOINT=your_endpoint \
  jirascope:prod
```

## Cost Management

JiraScope includes built-in cost tracking:

- Automatic cost calculation for all operations
- Budget warnings at $5 and $10 thresholds
- Cost estimates for expensive operations
- Session cost summaries

## Export Formats

Results can be exported in multiple formats:

- **JSON**: Structured data with full details
- **CSV**: Tabular format for spreadsheet analysis

## Examples

### Complete Workflow
```bash
# 1. Check health
jirascope health-check

# 2. Sync project data
jirascope sync --project AAA --show-costs

# 3. Find duplicates
jirascope analyze duplicates --threshold 0.8 --output duplicates.json

# 4. Quality analysis
jirascope analyze quality --project AAA --use-claude --budget 10.00
```

### Dry Run Mode
Test commands without executing:
```bash
jirascope --dry-run sync --project AAA
jirascope --dry-run analyze duplicates --threshold 0.8
```

## Troubleshooting

### Common Issues

1. **Service connection errors**: Run `jirascope health-check` to diagnose
2. **Configuration issues**: Use `--verbose` flag for detailed logging
3. **Cost concerns**: Use `--cost-estimate` flag before expensive operations

### Getting Help
```bash
jirascope --help
jirascope analyze --help
jirascope sync --help
```