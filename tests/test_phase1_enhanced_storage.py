"""Tests for Phase 1: Enhanced Vector Storage components."""

from datetime import datetime

import pytest

from src.jirascope.models.metadata_schema import ChunkMetadata, JiraItemMetadata
from src.jirascope.models.work_item import WorkItem
from src.jirascope.pipeline.smart_chunker import SmartChunker


class TestJiraItemMetadata:
    """Test the enhanced metadata schema."""

    def test_metadata_creation(self):
        """Test creating metadata with all fields."""
        metadata = JiraItemMetadata(
            key="PROJ-123",
            item_type="Story",
            status="In Progress",
            priority="High",
            created=datetime.now(),
            updated=datetime.now(),
            epic_key="PROJ-100",
            components=["frontend", "backend"],
            labels=["feature", "priority"],
        )

        assert metadata.key == "PROJ-123"
        assert metadata.item_type == "Story"
        assert metadata.components == ["frontend", "backend"]
        assert metadata.labels == ["feature", "priority"]

    def test_metadata_to_qdrant_payload(self):
        """Test converting metadata to Qdrant payload format."""
        now = datetime.now()
        metadata = JiraItemMetadata(
            key="PROJ-123",
            item_type="Story",
            status="In Progress",
            priority="High",
            created=now,
            updated=now,
            epic_key="PROJ-100",
        )

        payload = metadata.to_qdrant_payload()

        assert payload["key"] == "PROJ-123"
        assert payload["item_type"] == "Story"
        assert payload["epic_key"] == "PROJ-100"
        assert payload["created_month"] == now.strftime("%Y-%m")
        assert "has_children" in payload
        assert "dependency_count" in payload


class TestSmartChunker:
    """Test the smart chunking functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = SmartChunker(max_chunk_size=200)
        self.sample_work_item = WorkItem(
            key="PROJ-123",
            summary="Implement user authentication",
            description="Create login functionality with validation and error handling",
            issue_type="Story",
            status="In Progress",
            created=datetime.now(),
            updated=datetime.now(),
            reporter="testuser",
            components=["frontend"],
            labels=["auth"],
            parent_key=None,
            epic_key=None,
            assignee=None,
            embedding=None,
        )

    def test_chunk_story(self):
        """Test chunking a story work item."""
        chunks = self.chunker.chunk_work_item(self.sample_work_item)

        assert len(chunks) >= 1
        assert any(chunk.chunk_type == "story_summary" for chunk in chunks)

        # Check that chunks have proper metadata
        for chunk in chunks:
            assert chunk.metadata.source_key == "PROJ-123"
            assert chunk.metadata.parent_metadata.item_type == "Story"

    def test_chunk_epic(self):
        """Test chunking an epic work item."""
        epic_item = WorkItem(
            key="PROJ-100",
            summary="User Management Epic",
            description="Complete user management system including auth, profiles, and permissions",
            issue_type="Epic",
            status="In Progress",
            created=datetime.now(),
            updated=datetime.now(),
            reporter="testuser",
            parent_key=None,
            epic_key=None,
            assignee=None,
            embedding=None,
        )

        chunks = self.chunker.chunk_work_item(epic_item)

        assert len(chunks) >= 1
        assert any(chunk.chunk_type == "epic_summary" for chunk in chunks)

    def test_chunk_bug_with_sections(self):
        """Test chunking a bug with structured sections."""
        bug_description = """
        Symptoms: Login fails with 500 error

        Steps to reproduce:
        1. Navigate to login page
        2. Enter valid credentials
        3. Click login button

        Solution: Fix database connection timeout
        """

        bug_item = WorkItem(
            key="PROJ-124",
            summary="Login failure bug",
            description=bug_description,
            issue_type="Bug",
            status="Open",
            created=datetime.now(),
            updated=datetime.now(),
            reporter="testuser",
            parent_key=None,
            epic_key=None,
            assignee=None,
            embedding=None,
        )

        chunks = self.chunker.chunk_work_item(bug_item)

        # Should have summary plus structured sections
        assert len(chunks) >= 2
        assert any(chunk.chunk_type == "bug_summary" for chunk in chunks)

        # Check for structured sections
        chunk_types = [chunk.chunk_type for chunk in chunks]
        assert any("bug_" in chunk_type for chunk_type in chunk_types)

    def test_chunk_story_with_acceptance_criteria(self):
        """Test chunking story with acceptance criteria."""
        story_description = """
        As a user, I want to login to the system.

        Acceptance Criteria:
        - User can enter username and password
        - System validates credentials
        - User is redirected to dashboard on success

        Additional notes: Consider implementing 2FA in future.
        """

        story_item = WorkItem(
            key="PROJ-125",
            summary="User login story",
            description=story_description,
            issue_type="Story",
            status="Open",
            created=datetime.now(),
            updated=datetime.now(),
            reporter="testuser",
            parent_key=None,
            epic_key=None,
            assignee=None,
            embedding=None,
        )

        chunks = self.chunker.chunk_work_item(story_item)

        # Should have summary, AC, and description chunks
        assert len(chunks) >= 2
        chunk_types = [chunk.chunk_type for chunk in chunks]
        assert "acceptance_criteria" in chunk_types

    def test_content_hash_calculation(self):
        """Test that content hash is calculated correctly."""
        chunks = self.chunker.chunk_work_item(self.sample_work_item)

        # All chunks should have the same content hash from their parent
        content_hashes = [chunk.metadata.parent_metadata.content_hash for chunk in chunks]
        assert len(set(content_hashes)) == 1  # All should be the same
        assert content_hashes[0]  # Should not be empty


class TestChunkMetadata:
    """Test chunk metadata functionality."""

    def test_chunk_metadata_creation(self):
        """Test creating chunk metadata."""
        base_metadata = JiraItemMetadata(
            key="PROJ-123",
            item_type="Story",
            status="In Progress",
            priority="High",
            created=datetime.now(),
            updated=datetime.now(),
        )

        chunk_metadata = ChunkMetadata(
            chunk_id="chunk_123",
            source_key="PROJ-123",
            chunk_type="summary",
            chunk_index=0,
            parent_metadata=base_metadata,
        )

        assert chunk_metadata.chunk_id == "chunk_123"
        assert chunk_metadata.source_key == "PROJ-123"
        assert chunk_metadata.chunk_type == "summary"

    def test_chunk_metadata_to_qdrant_payload(self):
        """Test converting chunk metadata to Qdrant payload."""
        base_metadata = JiraItemMetadata(
            key="PROJ-123",
            item_type="Story",
            status="In Progress",
            priority="High",
            created=datetime.now(),
            updated=datetime.now(),
        )

        chunk_metadata = ChunkMetadata(
            chunk_id="chunk_123",
            source_key="PROJ-123",
            chunk_type="summary",
            chunk_index=0,
            parent_metadata=base_metadata,
        )

        payload = chunk_metadata.to_qdrant_payload()

        # Should include both chunk-specific and parent metadata
        assert payload["chunk_id"] == "chunk_123"
        assert payload["chunk_type"] == "summary"
        assert payload["source_key"] == "PROJ-123"
        assert payload["key"] == "PROJ-123"  # From parent
        assert payload["item_type"] == "Story"  # From parent


if __name__ == "__main__":
    pytest.main([__file__])
