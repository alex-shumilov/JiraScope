# JiraScope Web Interface Usage Guide

This guide covers the web-based interface for JiraScope, which provides a browser-based alternative to the CLI.

## Starting the Web Interface

```bash
# Start the web server
python src/web/main.py

# Or use the CLI command (if implemented)
jirascope web-server --port 8000
```

The web interface will be available at `http://localhost:8000`

## Web Interface Features

### Dashboard Overview

The main dashboard provides:
- **System Status**: Health check indicators for all services
- **Cost Tracking**: Real-time budget and usage monitoring
- **Quick Actions**: Common analysis tasks with one-click access
- **Recent Activity**: Latest analysis results and queries

### Search & Query Interface

#### Natural Language Search
- Enter queries in plain English
- Examples:
  - "Show me high priority bugs from last week"
  - "Find technical debt in authentication components"
  - "What work items are blocked for the mobile team?"

#### Advanced Search Filters
- **Project**: Filter by specific Jira projects
- **Date Range**: Limit results to specific time periods
- **Work Item Types**: Stories, Bugs, Tasks, Epics
- **Status**: Open, In Progress, Done, Blocked
- **Priority**: High, Medium, Low
- **Components**: Frontend, Backend, Mobile, etc.

### Analysis Tools

#### Duplicate Detection
1. Navigate to **Analysis > Duplicates**
2. Select project(s) to analyze
3. Adjust similarity threshold (0.0-1.0)
4. Click "Find Duplicates"
5. Review results with similarity scores and suggestions

#### Cross-Epic Analysis
1. Go to **Analysis > Cross-Epic**
2. Choose project scope
3. Review items that may belong to different Epics
4. Export recommendations for team review

#### Quality Assessment
1. Access **Analysis > Quality**
2. Enter specific work item key (e.g., PROJ-123)
3. View comprehensive quality metrics:
   - Completeness score
   - Description quality
   - Acceptance criteria coverage
   - Estimation accuracy

#### Technical Debt Clustering
1. Visit **Analysis > Tech Debt**
2. Select project or component
3. Review clustered debt items by:
   - Priority score
   - Estimated effort
   - Impact assessment
   - Recommended approach

### Real-time Features

#### WebSocket Integration
- Live updates for long-running analyses
- Real-time progress indicators
- Instant notification of completion

#### Streaming Results
- Results appear as they're processed
- Cancel long-running operations
- Progress bars for batch operations

## API Endpoints

The web interface exposes REST API endpoints for programmatic access:

### Search Endpoints
```http
GET /api/search?query=high%20priority%20bugs&limit=10
POST /api/search
{
  "query": "authentication issues",
  "filters": {
    "project": "PROJ",
    "priority": ["High", "Critical"]
  }
}
```

### Analysis Endpoints
```http
POST /api/analysis/duplicates
{
  "project": "PROJ",
  "threshold": 0.85
}

POST /api/analysis/cross-epic
{
  "project": "PROJ"
}

POST /api/analysis/quality
{
  "work_item_key": "PROJ-123"
}

POST /api/analysis/tech-debt
{
  "project": "PROJ",
  "component": "frontend"
}
```

### Task Management
```http
GET /api/tasks                    # List all tasks
GET /api/tasks/{task_id}          # Get task status
DELETE /api/tasks/{task_id}       # Cancel task
```

### System Endpoints
```http
GET /api/health                   # System health check
GET /api/cost                     # Cost and usage summary
GET /api/config                   # Current configuration
```

## Configuration

### Web Server Configuration

Environment variables for web interface:

```bash
# Server Configuration
WEB_HOST=0.0.0.0
WEB_PORT=8000
WEB_DEBUG=false

# Security
WEB_SECRET_KEY=your_secret_key_here
WEB_CORS_ORIGINS=["http://localhost:3000"]

# Features
WEB_ENABLE_API=true
WEB_ENABLE_WEBSOCKETS=true
WEB_MAX_WORKERS=4
```

### Frontend Configuration

The web interface can be customized via `src/web/static/config.js`:

```javascript
window.JiraScopeConfig = {
  apiBase: '/api',
  maxResults: 50,
  refreshInterval: 30000,
  theme: 'light',
  features: {
    realtime: true,
    exports: true,
    sharing: false
  }
};
```

## Usage Examples

### Sprint Planning Workflow
1. **Navigate to Dashboard**
2. **Search for Sprint Items**: "items planned for Sprint 23"
3. **Analyze Dependencies**: Use Cross-Epic analysis
4. **Check Quality**: Review acceptance criteria completeness
5. **Export Results**: Download analysis for team review

### Technical Debt Review
1. **Access Tech Debt Analysis**
2. **Filter by Component**: Select "authentication" or "payment"
3. **Review Clusters**: Examine grouped debt items
4. **Prioritize Work**: Sort by impact and effort scores
5. **Create Action Items**: Export recommendations

### Quality Assurance
1. **Bulk Quality Check**: Upload list of work item keys
2. **Review Scores**: Check completeness and clarity metrics
3. **Identify Issues**: Find items needing improvement
4. **Generate Reports**: Export quality assessment

## Troubleshooting

### Common Issues

**Web Server Won't Start:**
```bash
# Check port availability
netstat -an | grep 8000

# Use different port
WEB_PORT=8080 python src/web/main.py
```

**API Errors:**
- Check browser console for JavaScript errors
- Verify backend services are running
- Review server logs for API failures

**Performance Issues:**
- Reduce batch sizes in analysis requests
- Enable result pagination
- Clear browser cache

### Debug Mode

Enable debug mode for detailed logging:

```bash
WEB_DEBUG=true python src/web/main.py
```

### Browser Compatibility

Supported browsers:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

Required features:
- WebSocket support for real-time updates
- ES6+ JavaScript support
- Fetch API for HTTP requests

## Integration with Other Tools

### LMStudio Integration
The web interface complements LMStudio MCP integration:
- Use web interface for detailed analysis
- Use LMStudio for natural language queries
- Share results between both interfaces

### CLI Integration
Web interface and CLI share the same backend:
- Results from CLI commands appear in web interface
- Web analysis tasks can be monitored via CLI
- Shared configuration and data storage

### External Tools
Export capabilities for integration:
- JSON/CSV exports for external analysis
- Webhook notifications for CI/CD integration
- REST API for custom tool integration

## Advanced Features

### Batch Operations
- Upload CSV files with work item keys
- Bulk analysis across multiple projects
- Scheduled analysis tasks

### Custom Dashboards
- Create project-specific dashboards
- Configure favorite queries and filters
- Team-specific views and permissions

### Reporting
- Generate PDF reports from analysis results
- Schedule automated reports
- Email distribution for team updates

For more information on API integration and advanced usage, see the [API Documentation](api/README.md).
