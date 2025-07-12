# JiraScope CLI Usage Guide

This guide covers all available command-line interface commands for JiraScope.

## Installation & Setup

```bash
# Install JiraScope
pip install -e .

# Set up configuration (choose one method)
cp config/jirascope.yaml.example config/jirascope.yaml
# OR
cp config/env.example .env

# Edit your configuration file with your settings
```

## Global Options

```bash
jirascope --help                    # Show help
jirascope --config CONFIG_FILE      # Use specific config file
jirascope --verbose                 # Enable verbose logging
jirascope --log-file LOG_FILE       # Write logs to file
```

## Core Commands

### Health Check

Check the status of all JiraScope components:

```bash
jirascope health                    # Check all service connections
jirascope status                    # Detailed status report
```

### Data Extraction & Processing

```bash
# Fetch data from Jira
jirascope fetch --project PROJ               # Extract project data
jirascope fetch --project PROJ --incremental # Incremental sync only
jirascope fetch --project PROJ --jql "priority = High"

# Legacy extraction command
jirascope extract --output-dir ./data       # Extract to specific directory

# Process and generate embeddings
jirascope process --batch-size 50           # Custom batch size
jirascope process --force-reprocess          # Reprocess all data
```

### Search & Query

```bash
# Search work items
jirascope search --query "authentication bugs" --limit 10

# Interactive query mode
jirascope query --interactive

# Single query
jirascope query --query "high priority frontend issues"
```

### Analysis Commands

JiraScope provides several specialized analysis commands:

#### Duplicate Analysis
```bash
jirascope analyze duplicates --project PROJ --threshold 0.85
```
Find potentially duplicate work items using semantic similarity.

#### Cross-Epic Analysis
```bash
jirascope analyze cross-epic --project PROJ
```
Find work items that might belong to different Epics.

#### Quality Analysis
```bash
jirascope analyze quality PROJ-123
```
Analyze the quality and completeness of a specific work item.

#### Template Analysis
```bash
jirascope analyze template --issue-type Story --project PROJ
```
Generate templates based on high-quality examples.

#### Technical Debt Analysis
```bash
jirascope analyze tech-debt --project PROJ
```
Identify and cluster technical debt items.

### Authentication

For Jira SSE/OAuth authentication:

```bash
jirascope auth                      # Start authentication flow
jirascope auth-status              # Check authentication status
jirascope auth-clear               # Clear cached tokens
```

### MCP Server

Run JiraScope as an MCP server for LMStudio integration:

```bash
jirascope mcp-server                       # Default stdio transport
jirascope mcp-server --transport sse       # Server-Sent Events
jirascope mcp-server --transport sse --port 8080
```

### Maintenance

```bash
jirascope cleanup --days 30        # Clean cache files older than 30 days
jirascope validate                  # Validate system and run test queries
jirascope cost                      # Show cost analysis and budget status
```

## Configuration Examples

### Environment Variables (.env)
```bash
JIRA_MCP_ENDPOINT=https://company.atlassian.net
QDRANT_URL=http://localhost:6333
LMSTUDIO_ENDPOINT=http://localhost:1234/v1
CLAUDE_API_KEY=your_api_key
```

### YAML Configuration (jirascope.yaml)
```yaml
jira:
  mcp_endpoint: "https://company.atlassian.net"
  batch_size: 100

lmstudio:
  endpoint: "http://localhost:1234/v1"

processing:
  embedding_batch_size: 32
  similarity_threshold: 0.8
```

## Common Workflows

### Initial Setup
```bash
# 1. Configure JiraScope
cp config/env.example .env
# Edit .env with your settings

# 2. Check connectivity
jirascope health

# 3. Extract and process data
jirascope fetch --project YOUR_PROJECT
jirascope process

# 4. Validate setup
jirascope validate
```

### Daily Analysis
```bash
# Incremental data sync
jirascope fetch --project PROJ --incremental

# Find duplicates
jirascope analyze duplicates --threshold 0.8

# Check for misplaced work items
jirascope analyze cross-epic --project PROJ

# Interactive exploration
jirascope query --interactive
```

### LMStudio Integration
```bash
# Start MCP server for LMStudio
jirascope mcp-server

# In another terminal, you can now use LMStudio to chat with your Jira data
```

## Troubleshooting

### Common Issues

**Connection Errors:**
```bash
jirascope health                    # Check service status
jirascope status                    # Detailed diagnostics
```

**Authentication Issues:**
```bash
jirascope auth-clear               # Clear cached tokens
jirascope auth                     # Re-authenticate
```

**Performance Issues:**
```bash
jirascope cleanup --days 7         # Clean old cache files
jirascope process --batch-size 16  # Reduce batch size
```

### Debug Mode
```bash
jirascope --verbose --log-file debug.log [command]
```

### Configuration Validation
```bash
jirascope validate                  # Run system validation
jirascope cost                      # Check budget and usage
```

## Advanced Usage

### Custom JQL Queries
```bash
jirascope fetch --project PROJ --jql "component = 'frontend' AND priority = High"
```

### Batch Processing
```bash
jirascope process --batch-size 64 --force-reprocess
```

### Cost Optimization
```bash
jirascope cost                      # View current usage
jirascope cleanup --days 14        # Regular maintenance
```

For more advanced usage patterns and integration examples, see [LMStudio Prompts](examples/lmstudio_prompts.md).
