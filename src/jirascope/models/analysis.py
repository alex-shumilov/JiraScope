"""Analysis result models for JiraScope."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DuplicateCandidate(BaseModel):
    """Candidate for duplicate work item."""
    
    original_key: str = Field(..., description="Original work item key")
    duplicate_key: str = Field(..., description="Potential duplicate work item key")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    confidence_level: str = Field(..., description="Confidence level (exact, high, medium, low)")
    review_priority: int = Field(..., ge=1, le=5, description="Priority for manual review")
    suggested_action: str = Field(..., description="Suggested action to take")
    similarity_reasons: List[str] = Field(default_factory=list, description="Reasons for similarity")


class DuplicateReport(BaseModel):
    """Report of duplicate analysis results."""
    
    total_candidates: int = Field(..., description="Total number of duplicate candidates")
    candidates_by_level: Dict[str, List[DuplicateCandidate]] = Field(
        default_factory=dict, 
        description="Candidates grouped by confidence level"
    )
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    processing_cost: float = Field(0.0, description="Cost of analysis")
    items_analyzed: int = Field(0, description="Number of items analyzed")


class MisplacedWorkItem(BaseModel):
    """Work item that might belong to a different Epic."""
    
    work_item_key: str = Field(..., description="Work item key")
    current_epic_key: str = Field(..., description="Current Epic key")
    suggested_epic_key: str = Field(..., description="Suggested Epic key")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in suggestion")
    coherence_difference: float = Field(..., description="Difference in coherence scores")
    reasoning: str = Field(..., description="Reasoning for the suggestion")


class CrossEpicReport(BaseModel):
    """Report of cross-Epic analysis."""
    
    misplaced_items: List[MisplacedWorkItem] = Field(default_factory=list)
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    epics_analyzed: int = Field(0, description="Number of Epics analyzed")
    processing_cost: float = Field(0.0, description="Cost of analysis")


class CoherenceAnalysis(BaseModel):
    """Analysis of Epic coherence."""
    
    epic_key: str = Field(..., description="Epic key")
    coherence_score: float = Field(..., ge=0.0, le=1.0, description="Overall coherence score")
    work_items_count: int = Field(..., description="Number of work items in Epic")
    outlier_items: List[str] = Field(default_factory=list, description="Work item keys that are outliers")
    theme_consistency: float = Field(..., ge=0.0, le=1.0, description="Theme consistency score")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")


class QualityAnalysis(BaseModel):
    """Content quality analysis result."""
    
    work_item_key: str = Field(..., description="Work item key")
    clarity_score: int = Field(..., ge=1, le=5, description="Clarity score (1-5)")
    completeness_score: int = Field(..., ge=1, le=5, description="Completeness score (1-5)")
    actionability_score: int = Field(..., ge=1, le=5, description="Actionability score (1-5)")
    testability_score: int = Field(..., ge=1, le=5, description="Testability score (1-5)")
    overall_score: float = Field(..., ge=1.0, le=5.0, description="Overall quality score")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    risk_level: str = Field(..., description="Risk level (Low, Medium, High)")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    analysis_cost: float = Field(0.0, description="Cost of this analysis")


class SplitSuggestion(BaseModel):
    """Suggestion for splitting a work item."""
    
    suggested_title: str = Field(..., description="Suggested title for split item")
    suggested_description: str = Field(..., description="Suggested description")
    estimated_effort: Optional[str] = Field(None, description="Estimated effort")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies on other splits")


class SplitAnalysis(BaseModel):
    """Analysis of whether work item should be split."""
    
    work_item_key: str = Field(..., description="Work item key")
    should_split: bool = Field(..., description="Whether item should be split")
    complexity_score: float = Field(..., ge=0.0, le=1.0, description="Complexity score")
    suggested_splits: List[SplitSuggestion] = Field(default_factory=list)
    reasoning: str = Field(..., description="Reasoning for split decision")
    analysis_cost: float = Field(0.0, description="Cost of analysis")


class TemplateInference(BaseModel):
    """Inferred template from high-quality examples."""
    
    issue_type: str = Field(..., description="Issue type this template is for")
    title_template: str = Field(..., description="Title template with placeholders")
    description_template: str = Field(..., description="Description template")
    required_fields: List[str] = Field(default_factory=list, description="Required fields checklist")
    common_components: List[str] = Field(default_factory=list, description="Common components")
    common_labels: List[str] = Field(default_factory=list, description="Common labels")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in template quality")
    sample_count: int = Field(..., description="Number of samples used")
    generation_cost: float = Field(0.0, description="Cost to generate template")


class ScopeDriftEvent(BaseModel):
    """Individual scope drift event."""
    
    timestamp: datetime = Field(..., description="When the change occurred")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity to previous version")
    change_type: str = Field(..., description="Type of change (expansion, reduction, pivot, clarification)")
    impact_level: str = Field(..., description="Impact level (minor, moderate, major)")
    description: str = Field(..., description="Description of the change")
    changed_by: Optional[str] = Field(None, description="User who made the change")


class ScopeDriftAnalysis(BaseModel):
    """Analysis of scope drift over time."""
    
    work_item_key: str = Field(..., description="Work item key")
    has_drift: bool = Field(..., description="Whether scope drift was detected")
    drift_events: List[ScopeDriftEvent] = Field(default_factory=list)
    overall_drift_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall drift severity")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    total_changes: int = Field(0, description="Total number of changes analyzed")


class TechDebtCluster(BaseModel):
    """Cluster of related technical debt items."""
    
    cluster_id: int = Field(..., description="Cluster identifier")
    work_item_keys: List[str] = Field(..., description="Work item keys in this cluster")
    theme: str = Field(..., description="Common theme of the cluster")
    priority_score: float = Field(..., ge=0.0, le=1.0, description="Priority score for addressing")
    estimated_effort: str = Field(..., description="Estimated effort to address cluster")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies between items")
    impact_assessment: str = Field(..., description="Impact assessment if not addressed")
    recommended_approach: str = Field(..., description="Recommended approach to address")


class TechDebtReport(BaseModel):
    """Report of technical debt clustering analysis."""
    
    clusters: List[TechDebtCluster] = Field(default_factory=list)
    total_tech_debt_items: int = Field(0, description="Total tech debt items analyzed")
    clustering_algorithm: str = Field("DBSCAN", description="Algorithm used for clustering")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    processing_cost: float = Field(0.0, description="Cost of analysis")


class LabelingAnalysis(BaseModel):
    """Analysis of labeling and component patterns."""
    
    label_usage_stats: Dict[str, int] = Field(default_factory=dict, description="Label usage statistics")
    component_usage_stats: Dict[str, int] = Field(default_factory=dict, description="Component usage statistics")
    suggested_label_cleanup: List[str] = Field(default_factory=list, description="Labels to clean up")
    suggested_new_labels: List[str] = Field(default_factory=list, description="Suggested new labels")
    inconsistency_issues: List[str] = Field(default_factory=list, description="Inconsistency issues found")
    optimization_suggestions: List[str] = Field(default_factory=list, description="Optimization suggestions")


class BatchAnalysisResult(BaseModel):
    """Result of batch analysis operation."""
    
    total_items_processed: int = Field(..., description="Total items processed")
    successful_analyses: int = Field(..., description="Successful analyses")
    failed_analyses: int = Field(..., description="Failed analyses")
    total_cost: float = Field(..., description="Total cost of batch analysis")
    processing_time: float = Field(..., description="Total processing time in seconds")
    analysis_results: List[Dict[str, Any]] = Field(default_factory=list, description="Individual analysis results")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    timestamp: datetime = Field(default_factory=datetime.now)


class EvolutionReport(BaseModel):
    """Report of Epic evolution over time."""
    
    epic_key: str = Field(..., description="Epic key")
    time_period_days: int = Field(..., description="Time period analyzed in days")
    coherence_trend: List[float] = Field(default_factory=list, description="Coherence scores over time")
    work_items_added: List[str] = Field(default_factory=list, description="Work items added during period")
    work_items_removed: List[str] = Field(default_factory=list, description="Work items removed during period")
    theme_stability: float = Field(..., ge=0.0, le=1.0, description="Theme stability score")
    major_changes: List[Dict[str, Any]] = Field(default_factory=list, description="Major changes detected")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for Epic management")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)