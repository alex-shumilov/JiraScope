"""Pydantic models for JiraScope."""

from .work_item import (
    WorkItem, 
    EpicHierarchy, 
    EpicTree,
    ExtractionCost,
    ProcessingResult,
    QualityReport,
    AnalysisResult
)

from .analysis import (
    DuplicateCandidate,
    DuplicateReport,
    MisplacedWorkItem,
    CrossEpicReport,
    CoherenceAnalysis,
    QualityAnalysis,
    SplitAnalysis,
    SplitSuggestion,
    TemplateInference,
    ScopeDriftEvent,
    ScopeDriftAnalysis,
    TechDebtCluster,
    TechDebtReport,
    LabelingAnalysis,
    BatchAnalysisResult,
    EvolutionReport
)

__all__ = [
    # Work item models
    "WorkItem", 
    "EpicHierarchy", 
    "EpicTree",
    "ExtractionCost",
    "ProcessingResult", 
    "QualityReport",
    "AnalysisResult",
    
    # Analysis models
    "DuplicateCandidate",
    "DuplicateReport", 
    "MisplacedWorkItem",
    "CrossEpicReport",
    "CoherenceAnalysis",
    "QualityAnalysis",
    "SplitAnalysis",
    "SplitSuggestion",
    "TemplateInference",
    "ScopeDriftEvent",
    "ScopeDriftAnalysis",
    "TechDebtCluster",
    "TechDebtReport",
    "LabelingAnalysis",
    "BatchAnalysisResult",
    "EvolutionReport"
]