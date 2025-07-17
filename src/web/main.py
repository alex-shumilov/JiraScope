"""FastAPI web application for JiraScope."""

import asyncio
import uuid
from datetime import datetime, timedelta

import sentry_sdk
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from jirascope.core.config import Config

from .models import (
    AnalysisResponse,
    CostSummary,
    DuplicateAnalysisRequest,
    EpicAnalysisRequest,
    QualityAnalysisRequest,
    TaskStatusResponse,
)
from .services import AnalysisService, CostTracker, TaskManager

# Load configuration
config = Config.from_env()

# Initialize Sentry if DSN is configured
if config.sentry_dsn:
    sentry_sdk.init(
        dsn=config.sentry_dsn,
        # Add data like request headers and IP for users,
        # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
        send_default_pii=True,
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for tracing.
        traces_sample_rate=1.0,
    )

# Initialize FastAPI app
app = FastAPI(
    title="JiraScope API", description="Semantic Work Item Analysis Platform", version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")

# Global services
analysis_service = AnalysisService(config)
cost_tracker = CostTracker()
task_manager = TaskManager()


# Root endpoint - serve the web UI
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main web UI"""
    with open("src/web/static/index.html") as f:
        return HTMLResponse(content=f.read())


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check for load balancers"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}


# Analysis endpoints
@app.get("/api/projects")
async def get_projects() -> list[str]:
    """Get list of available projects"""
    try:
        return await analysis_service.get_available_projects()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analysis/duplicates")
async def analyze_duplicates(
    request: DuplicateAnalysisRequest, background_tasks: BackgroundTasks
) -> AnalysisResponse:
    """Find potential duplicate work items"""

    # Validate threshold
    if not (0.0 <= request.threshold <= 1.0):
        raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0")

    # Create task for analysis
    task_id = str(uuid.uuid4())
    task_manager.create_task(task_id, "duplicate_analysis")

    # Run analysis in background
    background_tasks.add_task(_run_duplicate_analysis, task_id, request)

    return AnalysisResponse(
        task_id=task_id,
        status="processing",
        estimated_completion=datetime.utcnow() + timedelta(minutes=2),
    )


@app.post("/api/analysis/quality")
async def analyze_quality(
    request: QualityAnalysisRequest, background_tasks: BackgroundTasks
) -> AnalysisResponse:
    """Analyze work item quality"""

    # Budget validation
    if request.budget_limit and request.budget_limit > 50.0:
        raise HTTPException(status_code=400, detail="Budget limit cannot exceed $50.00")

    # Create task
    task_id = str(uuid.uuid4())
    task_manager.create_task(task_id, "quality_analysis")

    # Run analysis in background
    background_tasks.add_task(_run_quality_analysis, task_id, request)

    return AnalysisResponse(
        task_id=task_id,
        status="processing",
        estimated_completion=datetime.utcnow() + timedelta(minutes=3),
    )


@app.post("/api/analysis/epic/{epic_key}")
async def analyze_epic(
    epic_key: str, request: EpicAnalysisRequest, background_tasks: BackgroundTasks
) -> AnalysisResponse:
    """Analyze Epic comprehensively"""

    # Create task
    task_id = str(uuid.uuid4())
    task_manager.create_task(task_id, "epic_analysis")

    # Run analysis in background
    background_tasks.add_task(_run_epic_analysis, task_id, epic_key, request)

    return AnalysisResponse(
        task_id=task_id,
        status="processing",
        estimated_completion=datetime.utcnow() + timedelta(minutes=5),
    )


@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """Get status of background analysis task"""
    task_status = task_manager.get_task_status(task_id)

    if not task_status:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatusResponse(**task_status)


# Cost tracking endpoints
@app.get("/api/costs/summary")
async def get_cost_summary(period: str = "session") -> CostSummary:
    """Get cost summary for specified period"""

    if period not in ["session", "daily", "weekly", "monthly"]:
        raise HTTPException(status_code=400, detail="Invalid period")

    costs = cost_tracker.get_costs_for_period(period)

    return CostSummary(
        period=period,
        total_cost=costs["total"],
        breakdown=costs["breakdown"],
        budget_remaining=costs.get("budget_remaining", 0.0),
    )


# WebSocket for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, task_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[task_id] = websocket

    def disconnect(self, task_id: str):
        if task_id in self.active_connections:
            del self.active_connections[task_id]

    async def send_update(self, task_id: str, data: dict):
        if task_id in self.active_connections:
            try:
                await self.active_connections[task_id].send_json(data)
            except Exception:
                self.disconnect(task_id)


connection_manager = ConnectionManager()


@app.websocket("/ws/tasks/{task_id}")
async def websocket_task_updates(websocket: WebSocket, task_id: str):
    await connection_manager.connect(task_id, websocket)
    try:
        while True:
            # Send periodic updates about task progress
            task_status = task_manager.get_task_status(task_id)

            if not task_status:
                await websocket.send_json({"error": "Task not found"})
                break

            await websocket.send_json(task_status)

            if task_status["status"] in ["completed", "failed"]:
                break

            await asyncio.sleep(2)

    except WebSocketDisconnect:
        connection_manager.disconnect(task_id)


# Export endpoints
@app.get("/api/export/{task_id}")
async def export_results(task_id: str, format: str = "json") -> dict:
    """Export analysis results"""

    task_status = task_manager.get_task_status(task_id)

    if not task_status:
        raise HTTPException(status_code=404, detail="Task not found")

    if task_status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")

    results = task_status.get("results", {})

    if format == "json":
        return results
    if format == "csv":
        # Simple CSV conversion for demo
        return {"csv_data": "key,value\n" + "\n".join(f"{k},{v}" for k, v in results.items())}
    raise HTTPException(status_code=400, detail="Unsupported format")


# Background task functions
async def _run_duplicate_analysis(task_id: str, request: DuplicateAnalysisRequest):
    """Run duplicate analysis in background"""
    try:
        task_manager.update_task(task_id, "running", progress=10)

        # Simulate progress
        for i in range(20, 100, 20):
            await asyncio.sleep(1)
            task_manager.update_task(task_id, "running", progress=i)

        results = await analysis_service.find_duplicates(
            threshold=request.threshold, project_keys=request.project_keys
        )

        task_manager.update_task(task_id, "completed", progress=100, results=results)
        cost_tracker.track_operation("duplicate_analysis", results.get("cost", 0.0))

    except Exception as e:
        task_manager.update_task(task_id, "failed", error=str(e))


async def _run_quality_analysis(task_id: str, request: QualityAnalysisRequest):
    """Run quality analysis in background"""
    try:
        task_manager.update_task(task_id, "running", progress=10)

        # Simulate progress
        for i in range(20, 100, 20):
            await asyncio.sleep(1)
            task_manager.update_task(task_id, "running", progress=i)

        results = await analysis_service.analyze_quality(
            project_key=request.project_key,
            use_claude=request.use_claude,
            budget_limit=request.budget_limit,
            limit=request.limit,
        )

        task_manager.update_task(task_id, "completed", progress=100, results=results)
        cost_tracker.track_operation("quality_analysis", results.get("cost", 0.0))

    except Exception as e:
        task_manager.update_task(task_id, "failed", error=str(e))


async def _run_epic_analysis(task_id: str, epic_key: str, request: EpicAnalysisRequest):
    """Run epic analysis in background"""
    try:
        task_manager.update_task(task_id, "running", progress=10)

        # Simulate progress
        for i in range(20, 100, 15):
            await asyncio.sleep(1)
            task_manager.update_task(task_id, "running", progress=i)

        results = await analysis_service.analyze_epic(
            epic_key=epic_key, depth=request.depth, use_claude=request.use_claude
        )

        task_manager.update_task(task_id, "completed", progress=100, results=results)
        cost_tracker.track_operation("epic_analysis", results.get("cost", 0.0))

    except Exception as e:
        task_manager.update_task(task_id, "failed", error=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104
