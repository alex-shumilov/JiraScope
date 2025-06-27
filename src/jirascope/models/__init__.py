"""Pydantic models for JiraScope."""

from .analysis import (
    BatchAnalysisResult,
    CoherenceAnalysis,
    CrossEpicReport,
    DuplicateCandidate,
    DuplicateReport,
    EvolutionReport,
    LabelingAnalysis,
    MisplacedWorkItem,
    QualityAnalysis,
    ScopeDriftAnalysis,
    ScopeDriftEvent,
    SplitAnalysis,
    SplitSuggestion,
    TechDebtCluster,
    TechDebtReport,
    TemplateInference,
)
from .work_item import (
    AnalysisResult,
    EpicHierarchy,
    EpicTree,
    ExtractionCost,
    ProcessingResult,
    QualityReport,
    WorkItem,
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
    "EvolutionReport",
]
