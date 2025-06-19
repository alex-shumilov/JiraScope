# JiraScope Web Dashboard Usage Guide

The JiraScope web dashboard provides an intuitive interface for semantic work item analysis with real-time updates and cost tracking.

## Quick Start

### Local Development
```bash
# Start the web dashboard
jirascope web --host 0.0.0.0 --port 8000 --reload

# Or using Docker
docker compose up jirascope-dev
```

### Production
```bash
# Using Docker
docker-compose up jirascope-web

# Dashboard available at: http://localhost:8080
```

## Dashboard Features

### 1. Duplicate Analysis
- **Purpose**: Find potentially duplicate work items across projects
- **Parameters**:
  - Similarity Threshold (0.0-1.0): How similar items must be to be considered duplicates
  - Project Keys: Comma-separated list of projects to analyze (optional)
- **Output**: Table showing duplicate candidates with similarity scores and suggested actions

### 2. Quality Analysis
- **Purpose**: Analyze work item quality using AI
- **Parameters**:
  - Project Key: Specific project to analyze (optional)
  - Use Claude AI: Enable AI-powered analysis (costs apply)
  - Budget Limit: Maximum cost for Claude analysis ($0-$50)
- **Output**: Quality scores chart and detailed analysis table

### 3. Epic Analysis
- **Purpose**: Comprehensive Epic-level analysis
- **Parameters**:
  - Epic Key: The Epic to analyze (e.g., EPIC-123)
  - Analysis Depth: Basic (free) or Full (with Claude AI)
- **Output**: Epic statistics including total items, duplicates found, and quality score

## Real-time Features

### Progress Tracking
- All analyses run in the background
- Real-time progress bars show completion status
- WebSocket updates provide live status information

### Cost Monitoring
- Session cost displayed in header
- Automatic budget warnings
- Cost breakdown available per operation

## API Endpoints

### Analysis Endpoints
```
POST /api/analysis/duplicates    # Start duplicate analysis
POST /api/analysis/quality       # Start quality analysis
POST /api/analysis/epic/{key}    # Start epic analysis
```

### Task Management
```
GET /api/tasks/{task_id}         # Get task status
GET /api/export/{task_id}        # Export results (JSON/CSV)
```

### Cost Tracking
```
GET /api/costs/summary           # Get cost summary
```

### WebSocket
```
WS /ws/tasks/{task_id}          # Real-time task updates
```

## Usage Examples

### 1. Find Duplicates in Multiple Projects
1. Enter threshold: `0.8`
2. Enter project keys: `PROJ1, PROJ2, PROJ3`
3. Click "Find Duplicates"
4. Monitor progress bar
5. Review results table when complete

### 2. Quality Analysis with Budget Control
1. Enter project key: `MYPROJECT`
2. Check "Use Claude AI"
3. Set budget limit: `$5.00`
4. Click "Analyze Quality"
5. View quality chart and suggestions

### 3. Comprehensive Epic Analysis
1. Enter epic key: `EPIC-123`
2. Select depth: "Full (with Claude)"
3. Click "Analyze Epic"
4. Review comprehensive statistics

## Cost Management

### Budget Controls
- Session costs displayed in real-time
- Budget limits enforced for Claude operations
- Automatic cost warnings at thresholds

### Cost Optimization
- Use basic analysis for initial exploration
- Enable Claude AI only when detailed insights needed
- Set budget limits to control spending
- Monitor session costs regularly

## Technical Details

### Architecture
- **Backend**: FastAPI with async support
- **Frontend**: Vanilla JavaScript with Vue.js
- **Real-time**: WebSocket connections
- **Charts**: Chart.js for visualizations

### Performance
- Background task processing
- Non-blocking API operations
- Efficient WebSocket updates
- Responsive design for mobile

### Security
- CORS protection
- Input validation
- Budget enforcement
- Task isolation

## Troubleshooting

### Common Issues

1. **Analysis stuck "Processing"**
   - Check backend logs for errors
   - Verify service dependencies (Qdrant, LM Studio)
   - Try restarting the analysis

2. **High costs with Claude**
   - Set lower budget limits
   - Use basic analysis first
   - Limit number of items analyzed

3. **WebSocket connection issues**
   - Check firewall settings
   - Verify WebSocket support in browser
   - Try refreshing the page

### Getting Help
- Check browser developer console for errors
- Review FastAPI logs for backend issues
- Use CLI health check: `jirascope health-check`

## Development

### Running Locally
```bash
# Backend only
cd src
python -m web.main

# With auto-reload
jirascope web --reload

# Custom host/port
jirascope web --host 0.0.0.0 --port 8080
```

### API Documentation
- Interactive docs: http://localhost:8000/docs
- OpenAPI spec: http://localhost:8000/openapi.json

### Customization
- Frontend: Edit `src/web/static/index.html`
- Backend: Modify `src/web/main.py`
- Models: Update `src/web/models.py`