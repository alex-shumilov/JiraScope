"""JiraScope - AI-powered Jira work item analysis and management tool."""

__version__ = "0.1.0"

from .extractors import JiraExtractor
from .models import AnalysisResult, EpicHierarchy, EpicTree, WorkItem
from .pipeline import EmbeddingProcessor, EmbeddingQualityValidator, IncrementalProcessor

__all__ = [
    "JiraExtractor",
    "EmbeddingProcessor",
    "EmbeddingQualityValidator",
    "IncrementalProcessor",
    "WorkItem",
    "EpicHierarchy",
    "EpicTree",
    "AnalysisResult",
]
