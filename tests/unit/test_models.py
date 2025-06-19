"""Tests for Pydantic models."""

from datetime import datetime
import pytest
from pydantic import ValidationError
from src.jirascope.models import WorkItem, EpicHierarchy, AnalysisResult


def test_work_item_creation():
    """Test WorkItem model creation and validation."""
    work_item = WorkItem(
        key="PROJ-123",
        summary="Test summary",
        description="Test description",
        issue_type="Story",
        status="Open",
        created=datetime.now(),
        updated=datetime.now(),
        reporter="Test User"
    )
    
    assert work_item.key == "PROJ-123"
    assert work_item.summary == "Test summary"
    assert work_item.issue_type == "Story"
    assert work_item.assignee is None
    assert work_item.components == []
    assert work_item.labels == []


def test_work_item_required_fields():
    """Test WorkItem requires mandatory fields."""
    with pytest.raises(ValidationError):
        WorkItem()  # Missing required fields
    
    with pytest.raises(ValidationError):
        WorkItem(key="PROJ-123")  # Missing summary, issue_type, etc.


def test_work_item_json_serialization():
    """Test WorkItem JSON serialization."""
    now = datetime.now()
    work_item = WorkItem(
        key="PROJ-123",
        summary="Test",
        issue_type="Story",
        status="Open",
        created=now,
        updated=now,
        reporter="Test User"
    )
    
    json_data = work_item.model_dump_json()
    assert "PROJ-123" in json_data
    assert now.isoformat() in json_data


def test_epic_hierarchy():
    """Test EpicHierarchy model."""
    epic = WorkItem(
        key="EPIC-1",
        summary="Epic summary",
        issue_type="Epic",
        status="Open",
        created=datetime.now(),
        updated=datetime.now(),
        reporter="Epic Reporter"
    )
    
    story = WorkItem(
        key="STORY-1",
        summary="Story summary",
        issue_type="Story",
        status="Open",
        epic_key="EPIC-1",
        created=datetime.now(),
        updated=datetime.now(),
        reporter="Story Reporter"
    )
    
    hierarchy = EpicHierarchy(
        epic=epic,
        stories=[story]
    )
    
    assert hierarchy.epic.key == "EPIC-1"
    assert len(hierarchy.stories) == 1
    assert hierarchy.stories[0].key == "STORY-1"
    assert hierarchy.total_items == 1
    assert len(hierarchy.all_items) == 2  # Epic + 1 story


def test_analysis_result_creation():
    """Test AnalysisResult model creation."""
    result = AnalysisResult(
        work_item_key="PROJ-123",
        analysis_type="complexity",
        confidence=0.85,
        insights={
            "complexity_score": 7,
            "risk_factors": ["database", "authentication"]
        },
        cost=0.0015
    )
    
    assert result.work_item_key == "PROJ-123"
    assert result.analysis_type == "complexity"
    assert result.confidence == 0.85
    assert result.insights["complexity_score"] == 7
    assert result.cost == 0.0015
    assert isinstance(result.timestamp, datetime)


def test_analysis_result_confidence_validation():
    """Test AnalysisResult confidence validation."""
    # Valid confidence values
    AnalysisResult(
        work_item_key="PROJ-1",
        analysis_type="test",
        confidence=0.0
    )
    
    AnalysisResult(
        work_item_key="PROJ-1",
        analysis_type="test",
        confidence=1.0
    )
    
    # Invalid confidence values
    with pytest.raises(ValidationError):
        AnalysisResult(
            work_item_key="PROJ-1",
            analysis_type="test",
            confidence=-0.1
        )
    
    with pytest.raises(ValidationError):
        AnalysisResult(
            work_item_key="PROJ-1",
            analysis_type="test",
            confidence=1.1
        )