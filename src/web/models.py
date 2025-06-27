"""Pydantic models for the web API."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Request models
class DuplicateAnalysisRequest(BaseModel):
    threshold: float = Field(ge=0.0, le=1.0, description="Similarity threshold")
    project_keys: Optional[List[str]] = None
    issue_types: Optional[List[str]] = None


class QualityAnalysisRequest(BaseModel):
    project_key: Optional[str] = None
    use_claude: bool = False
    budget_limit: Optional[float] = Field(None, ge=0.0, le=50.0)
    limit: int = Field(10, ge=1, le=100, description="Maximum items to analyze")


class EpicAnalysisRequest(BaseModel):
    depth: str = Field("basic", pattern="^(basic|full)$")
    use_claude: bool = False
    budget_limit: Optional[float] = Field(None, ge=0.0, le=50.0)


# Response models
class AnalysisResponse(BaseModel):
    task_id: str
    status: str
    estimated_completion: Optional[datetime] = None
    message: Optional[str] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int = 0
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class CostSummary(BaseModel):
    period: str
    total_cost: float
    breakdown: Dict[str, float]
    budget_remaining: Optional[float] = None


# Analysis result models
class DuplicateCandidate(BaseModel):
    original_key: str
    duplicate_key: str
    similarity_score: float
    confidence_level: str
    suggested_action: str


class DuplicateResults(BaseModel):
    total_candidates: int
    candidates_by_level: Dict[str, List[DuplicateCandidate]]
    processing_cost: float


class QualityAnalysis(BaseModel):
    work_item_key: str
    clarity_score: int
    completeness_score: int
    actionability_score: int
    testability_score: int
    overall_score: float
    risk_level: str
    improvement_suggestions: List[str]


class QualityResults(BaseModel):
    total_analyzed: int
    analyses: List[QualityAnalysis]
    average_score: float
    processing_cost: float


class EpicResults(BaseModel):
    epic_key: str
    total_items: int
    duplicates_found: int
    quality_score: Optional[float] = None
    processing_cost: float
