# JiraScope API Documentation

This document provides comprehensive API documentation for JiraScope's REST API endpoints.

## Table of Contents

- [Authentication](#authentication)
- [Base URLs](#base-urls)
- [Response Formats](#response-formats)
- [Error Handling](#error-handling)
- [Search Endpoints](#search-endpoints)
- [Analysis Endpoints](#analysis-endpoints)
- [Task Management](#task-management)
- [System Endpoints](#system-endpoints)

## Authentication

JiraScope API uses the same authentication mechanism as the underlying Jira instance.

### Headers
```http
Authorization: Bearer <your_token>
Content-Type: application/json
```

## Base URLs

- **Development**: `http://localhost:8000/api`
- **Production**: `https://your-domain.com/api`

## Response Formats

### Success Response
```json
{
  "success": true,
  "data": {
    // Response data
  },
  "meta": {
    "timestamp": "2024-01-01T12:00:00Z",
    "version": "1.0.0"
  }
}
```

### Error Response
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": {}
  },
  "meta": {
    "timestamp": "2024-01-01T12:00:00Z",
    "version": "1.0.0"
  }
}
```

## Error Handling

### HTTP Status Codes

- `200` - Success
- `400` - Bad Request (invalid parameters)
- `401` - Unauthorized (authentication required)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found (resource doesn't exist)
- `429` - Too Many Requests (rate limited)
- `500` - Internal Server Error

### Common Error Codes

- `INVALID_QUERY` - Search query is malformed
- `ANALYSIS_FAILED` - Analysis operation failed
- `TASK_NOT_FOUND` - Requested task doesn't exist
- `SERVICE_UNAVAILABLE` - External service is unavailable

## Search Endpoints

### GET /api/search

Search work items using query string parameters.

**Parameters:**
- `query` (string, required) - Search query
- `limit` (integer, optional) - Maximum results (default: 10, max: 100)
- `offset` (integer, optional) - Result offset (default: 0)
- `project` (string, optional) - Filter by project key
- `priority` (array, optional) - Filter by priority levels

**Example:**
```http
GET /api/search?query=authentication%20bugs&limit=20&project=PROJ
```

**Response:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "key": "PROJ-123",
        "summary": "Authentication bug in login flow",
        "description": "Users cannot log in...",
        "priority": "High",
        "status": "Open",
        "assignee": "john.doe@company.com",
        "created": "2024-01-01T12:00:00Z",
        "updated": "2024-01-01T15:30:00Z",
        "similarity_score": 0.95
      }
    ],
    "total": 1,
    "limit": 20,
    "offset": 0
  }
}
```

### POST /api/search

Advanced search with complex filters.

**Request Body:**
```json
{
  "query": "authentication issues",
  "filters": {
    "project": ["PROJ", "MOBILE"],
    "priority": ["High", "Critical"],
    "status": ["Open", "In Progress"],
    "assignee": ["john.doe@company.com"],
    "created_after": "2024-01-01",
    "created_before": "2024-12-31",
    "components": ["frontend", "backend"]
  },
  "limit": 50,
  "offset": 0,
  "sort": {
    "field": "priority",
    "order": "desc"
  }
}
```

## Analysis Endpoints

### POST /api/analysis/duplicates

Find potentially duplicate work items.

**Request Body:**
```json
{
  "project": "PROJ",
  "threshold": 0.85,
  "limit": 100
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "duplicates": [
      {
        "group_id": "group_1",
        "items": [
          {
            "key": "PROJ-123",
            "summary": "Login bug fix",
            "similarity_score": 0.92
          },
          {
            "key": "PROJ-456",
            "summary": "Authentication issue",
            "similarity_score": 0.88
          }
        ],
        "average_similarity": 0.90
      }
    ],
    "total_groups": 1,
    "threshold": 0.85
  }
}
```

### POST /api/analysis/cross-epic

Analyze work items that might belong to different Epics.

**Request Body:**
```json
{
  "project": "PROJ",
  "epic_key": "PROJ-EPIC-1"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "misplaced_items": [
      {
        "key": "PROJ-789",
        "current_epic": "PROJ-EPIC-1",
        "suggested_epic": "PROJ-EPIC-2",
        "confidence": 0.87,
        "reasoning": "This item relates more to mobile features than web features"
      }
    ],
    "analysis_summary": {
      "total_items_analyzed": 45,
      "potentially_misplaced": 3,
      "confidence_threshold": 0.75
    }
  }
}
```

### POST /api/analysis/quality

Analyze the quality and completeness of work items.

**Request Body:**
```json
{
  "work_item_key": "PROJ-123"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "quality_score": 0.75,
    "completeness": 0.80,
    "metrics": {
      "description_quality": 0.85,
      "acceptance_criteria": 0.70,
      "technical_details": 0.60,
      "business_value": 0.90
    },
    "recommendations": [
      "Add more detailed acceptance criteria",
      "Include technical implementation notes"
    ],
    "missing_fields": ["acceptance_criteria", "story_points"]
  }
}
```

### POST /api/analysis/tech-debt

Identify and cluster technical debt items.

**Request Body:**
```json
{
  "project": "PROJ",
  "component": "frontend"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "clusters": [
      {
        "cluster_id": "auth_debt",
        "name": "Authentication Technical Debt",
        "items": [
          {
            "key": "PROJ-101",
            "summary": "Refactor login component",
            "effort_estimate": "Large",
            "impact_score": 0.85
          }
        ],
        "priority": "High",
        "estimated_effort": "2 weeks",
        "business_impact": "High"
      }
    ],
    "summary": {
      "total_debt_items": 15,
      "clusters_found": 4,
      "high_priority_clusters": 2
    }
  }
}
```

## Task Management

### GET /api/tasks

List all tasks (analysis operations that are running or completed).

**Parameters:**
- `status` (string, optional) - Filter by status: `pending`, `running`, `completed`, `failed`
- `limit` (integer, optional) - Maximum results (default: 20)

**Response:**
```json
{
  "success": true,
  "data": {
    "tasks": [
      {
        "task_id": "task_123",
        "type": "duplicate_analysis",
        "status": "completed",
        "progress": 100,
        "created": "2024-01-01T12:00:00Z",
        "started": "2024-01-01T12:00:05Z",
        "completed": "2024-01-01T12:02:30Z",
        "result_url": "/api/tasks/task_123/result"
      }
    ]
  }
}
```

### GET /api/tasks/{task_id}

Get specific task status and details.

**Response:**
```json
{
  "success": true,
  "data": {
    "task_id": "task_123",
    "type": "duplicate_analysis",
    "status": "running",
    "progress": 65,
    "created": "2024-01-01T12:00:00Z",
    "started": "2024-01-01T12:00:05Z",
    "estimated_completion": "2024-01-01T12:03:00Z",
    "current_step": "Processing work items",
    "total_steps": 4
  }
}
```

### DELETE /api/tasks/{task_id}

Cancel a running task.

**Response:**
```json
{
  "success": true,
  "data": {
    "task_id": "task_123",
    "status": "cancelled",
    "message": "Task cancelled successfully"
  }
}
```

## System Endpoints

### GET /api/health

Check system health and service availability.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "services": {
      "jira": {
        "status": "healthy",
        "response_time": 250,
        "last_check": "2024-01-01T12:00:00Z"
      },
      "qdrant": {
        "status": "healthy",
        "response_time": 15,
        "last_check": "2024-01-01T12:00:00Z"
      },
      "lmstudio": {
        "status": "healthy",
        "response_time": 45,
        "last_check": "2024-01-01T12:00:00Z"
      }
    },
    "version": "1.0.0",
    "uptime": 86400
  }
}
```

### GET /api/cost

Get cost analysis and budget status.

**Response:**
```json
{
  "success": true,
  "data": {
    "current_month": {
      "total_cost": 45.67,
      "budget": 100.00,
      "usage_percentage": 45.67,
      "remaining": 54.33
    },
    "breakdown": {
      "claude_api": 30.50,
      "embeddings": 12.17,
      "vector_storage": 3.00
    },
    "alerts": [
      {
        "type": "budget_warning",
        "message": "Approaching 50% of monthly budget",
        "threshold": 50
      }
    ]
  }
}
```

### GET /api/config

Get current system configuration (non-sensitive data only).

**Response:**
```json
{
  "success": true,
  "data": {
    "version": "1.0.0",
    "features": {
      "api_enabled": true,
      "websockets_enabled": true,
      "cost_tracking": true
    },
    "limits": {
      "max_search_results": 100,
      "max_concurrent_analyses": 5,
      "request_timeout": 30
    },
    "supported_formats": ["json"],
    "api_version": "v1"
  }
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Search endpoints**: 100 requests per minute
- **Analysis endpoints**: 10 requests per minute
- **System endpoints**: 300 requests per minute

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## SDKs and Examples

### Python Example
```python
import requests

# Search for work items
response = requests.get(
    'http://localhost:8000/api/search',
    params={'query': 'authentication bugs', 'limit': 10},
    headers={'Authorization': 'Bearer your_token'}
)
data = response.json()

# Start duplicate analysis
response = requests.post(
    'http://localhost:8000/api/analysis/duplicates',
    json={'project': 'PROJ', 'threshold': 0.85},
    headers={'Authorization': 'Bearer your_token'}
)
analysis = response.json()
```

### JavaScript Example
```javascript
// Search for work items
const searchResponse = await fetch('/api/search?query=bugs&limit=10', {
  headers: {
    'Authorization': 'Bearer your_token',
    'Content-Type': 'application/json'
  }
});
const searchData = await searchResponse.json();

// Start quality analysis
const qualityResponse = await fetch('/api/analysis/quality', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer your_token',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    work_item_key: 'PROJ-123'
  })
});
const qualityData = await qualityResponse.json();
```

## Webhooks

JiraScope supports webhooks for real-time notifications:

### Configuration
```json
{
  "url": "https://your-app.com/webhooks/jirascope",
  "events": ["analysis_completed", "task_failed"],
  "secret": "your_webhook_secret"
}
```

### Webhook Payload
```json
{
  "event": "analysis_completed",
  "data": {
    "task_id": "task_123",
    "type": "duplicate_analysis",
    "completed_at": "2024-01-01T12:05:00Z",
    "result": {
      // Analysis results
    }
  },
  "timestamp": "2024-01-01T12:05:00Z"
}
```

For more information, see the [Web Interface Guide](../WEB_USAGE.md) and [CLI Usage Guide](../CLI_USAGE.md).
