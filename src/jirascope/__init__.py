"""JiraScope - AI-powered Jira work item analysis and management tool."""

__version__ = "0.1.0"

from .extractors import JiraExtractor
from .pipeline import EmbeddingProcessor, EmbeddingQualityValidator, IncrementalProcessor
from .models import WorkItem, EpicHierarchy, EpicTree, AnalysisResult

__all__ = [
    "JiraExtractor",
    "EmbeddingProcessor", 
    "EmbeddingQualityValidator",
    "IncrementalProcessor",
    "WorkItem",
    "EpicHierarchy", 
    "EpicTree",
    "AnalysisResult"
]