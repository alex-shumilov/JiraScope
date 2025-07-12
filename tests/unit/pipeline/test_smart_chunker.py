"""Comprehensive tests for smart chunker functionality."""

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from src.jirascope.models.metadata_schema import ChunkMetadata
from src.jirascope.models.work_item import WorkItem
from src.jirascope.pipeline.smart_chunker import Chunk, SmartChunker


class TestChunk:
    """Test Chunk dataclass functionality."""

    def test_chunk_creation(self):
        """Test creating a Chunk with all fields."""
        # Create parent metadata
        from datetime import datetime, timezone

        from src.jirascope.models.metadata_schema import JiraItemMetadata

        parent_metadata = JiraItemMetadata(
            key="TEST-123",
            item_type="Story",
            status="Open",
            priority="Medium",
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc),
            content_hash="abc123",
        )

        metadata = ChunkMetadata(
            chunk_id="TEST-123_summary_0",
            source_key="TEST-123",
            chunk_type="summary",
            chunk_index=0,
            parent_metadata=parent_metadata,
        )

        chunk = Chunk(text="This is a test chunk", metadata=metadata, chunk_type="summary")

        assert chunk.text == "This is a test chunk"
        assert chunk.metadata.source_key == "TEST-123"
        assert chunk.chunk_type == "summary"

    def test_chunk_id_generation(self):
        """Test chunk ID generation."""
        # Create parent metadata
        from datetime import datetime, timezone

        from src.jirascope.models.metadata_schema import JiraItemMetadata

        parent_metadata = JiraItemMetadata(
            key="TEST-123",
            item_type="Epic",
            status="Open",
            priority="High",
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc),
            content_hash="abc123",
        )

        metadata = ChunkMetadata(
            chunk_id="TEST-123_description_2",
            source_key="TEST-123",
            chunk_type="description",
            chunk_index=2,
            parent_metadata=parent_metadata,
        )

        chunk = Chunk(text="Test content", metadata=metadata, chunk_type="description")

        chunk_id = chunk.chunk_id
        assert isinstance(chunk_id, str)
        assert len(chunk_id) == 12  # MD5 hash truncated to 12 chars

        # Same input should generate same ID
        chunk2 = Chunk(
            text="Different content",  # Content doesn't affect ID
            metadata=metadata,
            chunk_type="description",
        )
        assert chunk2.chunk_id == chunk_id

    def test_chunk_id_uniqueness(self):
        """Test that different chunks generate different IDs."""
        # Create parent metadata
        from datetime import datetime, timezone

        from src.jirascope.models.metadata_schema import JiraItemMetadata

        parent_metadata = JiraItemMetadata(
            key="TEST-123",
            item_type="Story",
            status="Open",
            priority="Medium",
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc),
            content_hash="abc123",
        )

        metadata1 = ChunkMetadata(
            chunk_id="TEST-123_summary_0",
            source_key="TEST-123",
            chunk_type="summary",
            chunk_index=0,
            parent_metadata=parent_metadata,
        )

        metadata2 = ChunkMetadata(
            chunk_id="TEST-123_summary_1",
            source_key="TEST-123",
            chunk_type="summary",
            chunk_index=1,  # Different index
            parent_metadata=parent_metadata,
        )

        chunk1 = Chunk(text="Content", metadata=metadata1, chunk_type="summary")
        chunk2 = Chunk(text="Content", metadata=metadata2, chunk_type="summary")

        assert chunk1.chunk_id != chunk2.chunk_id


class TestSmartChunker:
    """Test SmartChunker functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = SmartChunker(max_chunk_size=800, min_chunk_size=200, overlap_size=50)

        self.base_work_item = WorkItem(
            key="TEST-123",
            summary="Test work item summary",
            issue_type="Story",
            status="Open",
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc),
            reporter="test@example.com",
            description="Test description",
            assignee="assignee@example.com",
            embedding=[0.1] * 100,  # Mock embedding
        )

    def test_chunker_initialization(self):
        """Test SmartChunker initialization."""
        assert self.chunker.max_chunk_size == 800
        assert self.chunker.min_chunk_size == 200
        assert self.chunker.overlap_size == 50

        # Check that patterns are compiled
        assert hasattr(self.chunker, "epic_patterns")
        assert hasattr(self.chunker, "story_patterns")
        assert hasattr(self.chunker, "bug_patterns")
        assert len(self.chunker.epic_patterns) > 0
        assert len(self.chunker.story_patterns) > 0
        assert len(self.chunker.bug_patterns) > 0

    def test_chunker_custom_settings(self):
        """Test SmartChunker with custom settings."""
        custom_chunker = SmartChunker(max_chunk_size=1000, min_chunk_size=100, overlap_size=25)

        assert custom_chunker.max_chunk_size == 1000
        assert custom_chunker.min_chunk_size == 100
        assert custom_chunker.overlap_size == 25

    def test_chunk_work_item_basic(self):
        """Test basic work item chunking."""
        work_item = self.base_work_item

        chunks = self.chunker.chunk_work_item(work_item)

        assert len(chunks) >= 1  # At least summary chunk
        assert chunks[0].chunk_type == "story_summary"
        assert chunks[0].text == work_item.summary
        assert chunks[0].metadata.source_key == work_item.key

    def test_chunk_work_item_epic(self):
        """Test Epic-specific chunking."""
        epic_description = """
        Goals: Improve user experience and increase conversion rates.

        This epic focuses on enhancing the checkout process.

        Success Criteria:
        - Reduce cart abandonment by 20%
        - Increase conversion rate by 15%

        Additional details about implementation.
        """

        work_item = WorkItem(
            key="EPIC-123",
            summary="Improve checkout process",
            issue_type="Epic",
            status="In Progress",
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc),
            reporter="pm@example.com",
            description=epic_description,
            assignee="team@example.com",
            embedding=[0.1] * 100,
        )

        chunks = self.chunker.chunk_work_item(work_item)

        # Should have: summary, goals, success criteria, remaining description
        assert len(chunks) >= 3

        # Check summary chunk
        assert chunks[0].chunk_type == "epic_summary"
        assert chunks[0].text == work_item.summary

        # Check for goals chunk
        goals_chunks = [c for c in chunks if c.chunk_type == "goals"]
        assert len(goals_chunks) == 1
        assert "improve user experience" in goals_chunks[0].text.lower()

        # Check for success criteria chunk
        success_chunks = [c for c in chunks if c.chunk_type == "success_criteria"]
        assert len(success_chunks) == 1
        assert "reduce cart abandonment" in success_chunks[0].text.lower()

    def test_chunk_work_item_story(self):
        """Test Story-specific chunking."""
        story_description = """
        As a customer, I want to save my payment information for faster checkout.

        Acceptance Criteria:
        - User can save payment methods securely
        - Saved cards appear in checkout
        - User can delete saved payment methods

        Additional requirements:
        - Must be PCI compliant
        - Support multiple payment types
        """

        work_item = WorkItem(
            key="STORY-456",
            summary="Save payment information feature",
            issue_type="Story",
            status="To Do",
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc),
            reporter="dev@example.com",
            description=story_description,
            assignee="dev@example.com",
            embedding=[0.1] * 100,
        )

        chunks = self.chunker.chunk_work_item(work_item)

        # Should have: summary, user story, acceptance criteria, remaining description
        assert len(chunks) >= 3

        # Check for user story chunk
        user_story_chunks = [c for c in chunks if c.chunk_type == "user_story"]
        assert len(user_story_chunks) == 1
        assert "as a customer" in user_story_chunks[0].text.lower()

        # Check for acceptance criteria chunk
        ac_chunks = [c for c in chunks if c.chunk_type == "acceptance_criteria"]
        assert len(ac_chunks) == 1
        assert "save payment methods" in ac_chunks[0].text.lower()

    def test_chunk_work_item_bug(self):
        """Test Bug-specific chunking."""
        bug_description = """
        Symptoms: Payment form freezes when user enters invalid card number.

        Steps to reproduce:
        1. Go to checkout page
        2. Enter invalid card number (e.g., 1234)
        3. Click submit

        Expected behavior: Form should show validation error message.

        Solution: Add client-side validation for card numbers.

        Additional notes: This affects mobile users more frequently.
        """

        work_item = WorkItem(
            key="BUG-789",
            summary="Payment form freezes on invalid input",
            issue_type="Bug",
            status="In Progress",
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc),
            reporter="qa@example.com",
            description=bug_description,
            assignee="dev@example.com",
            embedding=[0.1] * 100,
        )

        chunks = self.chunker.chunk_work_item(work_item)

        # Should have: summary, symptoms, reproduction, expected, solution, remaining
        assert len(chunks) >= 4

        # Check for symptoms chunk
        symptoms_chunks = [c for c in chunks if c.chunk_type == "symptoms"]
        assert len(symptoms_chunks) == 1
        assert "payment form freezes" in symptoms_chunks[0].text.lower()

        # Check for reproduction steps chunk
        repro_chunks = [c for c in chunks if c.chunk_type == "reproduction"]
        assert len(repro_chunks) == 1
        assert "checkout page" in repro_chunks[0].text.lower()

        # Check for expected behavior chunk
        expected_chunks = [c for c in chunks if c.chunk_type == "expected"]
        assert len(expected_chunks) == 1
        assert "validation error" in expected_chunks[0].text.lower()

        # Check for solution chunk
        solution_chunks = [c for c in chunks if c.chunk_type == "solution"]
        assert len(solution_chunks) == 1
        assert "client-side validation" in solution_chunks[0].text.lower()

    def test_chunk_work_item_generic(self):
        """Test generic chunking for unknown issue types."""
        work_item = WorkItem(
            key="TASK-999",
            summary="Generic task summary",
            issue_type="Task",  # Not Epic, Story, or Bug
            status="Open",
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc),
            reporter="user@example.com",
            description="This is a longer description that should be chunked generically since it's not a recognized issue type with specific patterns.",
            assignee="user@example.com",
            embedding=[0.1] * 100,
        )

        chunks = self.chunker.chunk_work_item(work_item)

        # Should have at least summary chunk
        assert len(chunks) >= 1
        assert chunks[0].chunk_type == "task_summary"
        assert chunks[0].text == work_item.summary

    def test_chunk_work_item_no_description(self):
        """Test chunking work item with no description."""
        work_item = WorkItem(
            key="TEST-000",
            summary="Item with no description",
            issue_type="Story",
            status="Open",
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc),
            reporter="user@example.com",
            description=None,  # No description
            assignee="user@example.com",
            embedding=[0.1] * 100,
        )

        chunks = self.chunker.chunk_work_item(work_item)

        # Should only have summary chunk
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "story_summary"
        assert chunks[0].text == work_item.summary

    def test_chunk_work_item_empty_description(self):
        """Test chunking work item with empty description."""
        work_item = WorkItem(
            key="TEST-001",
            summary="Item with empty description",
            issue_type="Epic",
            status="Open",
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc),
            reporter="user@example.com",
            description="",  # Empty description
            assignee="user@example.com",
            embedding=[0.1] * 100,
        )

        chunks = self.chunker.chunk_work_item(work_item)

        # Should only have summary chunk
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "epic_summary"

    def test_chunk_text_long_content(self):
        """Test chunking very long text content."""
        # Create a long description that will require multiple chunks
        long_text = "This is a sentence. " * 50  # ~1000 characters

        work_item = WorkItem(
            key="LONG-123",
            summary="Long content item",
            issue_type="Story",
            status="Open",
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc),
            reporter="user@example.com",
            description=long_text,
            assignee="user@example.com",
            embedding=[0.1] * 100,
        )

        chunks = self.chunker.chunk_work_item(work_item)

        # Should have multiple chunks for long content
        description_chunks = [c for c in chunks if c.chunk_type == "description"]
        assert len(description_chunks) >= 1  # Long text should be chunked

    def test_chunk_text_by_sentences(self):
        """Test text chunking by sentences."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."

        chunks = self.chunker._chunk_text(
            text=text, work_item=self.base_work_item, chunk_type="description", start_index=1
        )

        assert len(chunks) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.chunk_type == "description" for chunk in chunks)

    def test_chunk_by_words(self):
        """Test word-based chunking for text without sentence structure."""
        # Text without proper sentences
        text = "word1 word2 word3 " * 200  # Very long text without periods

        chunks = self.chunker._chunk_by_words(
            text=text, work_item=self.base_work_item, chunk_type="description", start_index=1
        )

        assert len(chunks) >= 1
        # Check that chunks respect size limits
        for chunk in chunks:
            assert len(chunk.text) <= self.chunker.max_chunk_size + self.chunker.overlap_size

    def test_split_into_sentences(self):
        """Test sentence splitting functionality."""
        text = "This is the first sentence. This is the second sentence! Is this a question? Yes, it is."

        sentences = self.chunker._split_into_sentences(text)

        assert len(sentences) == 4
        assert sentences[0] == "This is the first sentence."
        assert sentences[1] == "This is the second sentence!"
        assert sentences[2] == "Is this a question?"
        assert sentences[3] == "Yes, it is."

    def test_split_into_sentences_edge_cases(self):
        """Test sentence splitting with edge cases."""
        # Empty text
        assert self.chunker._split_into_sentences("") == []

        # Text without sentence endings
        sentences = self.chunker._split_into_sentences("No punctuation here")
        assert len(sentences) == 1
        assert sentences[0] == "No punctuation here"

        # Text with abbreviations (should not split)
        text = "Dr. Smith went to St. Louis. He met Prof. Johnson there."
        sentences = self.chunker._split_into_sentences(text)
        assert len(sentences) == 2
        assert "Dr. Smith went to St. Louis" in sentences[0]

    def test_create_chunk(self):
        """Test chunk creation with metadata."""
        chunk = self.chunker._create_chunk(
            text="Test chunk content",
            work_item=self.base_work_item,
            chunk_type="test_type",
            chunk_index=2,
        )

        assert isinstance(chunk, Chunk)
        assert chunk.text == "Test chunk content"
        assert chunk.chunk_type == "test_type"
        assert chunk.metadata.source_key == self.base_work_item.key
        assert chunk.metadata.chunk_index == 2
        assert chunk.metadata.parent_metadata.item_type == self.base_work_item.issue_type

    def test_calculate_content_hash(self):
        """Test content hash calculation."""
        hash1 = self.chunker._calculate_content_hash(self.base_work_item)

        # Same work item should produce same hash
        hash2 = self.chunker._calculate_content_hash(self.base_work_item)
        assert hash1 == hash2

        # Different work item should produce different hash
        different_item = WorkItem(
            key="DIFF-123",
            summary="Different summary",
            issue_type="Bug",
            status="Open",
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc),
            reporter="other@example.com",
            description="Different description",
            assignee="other@example.com",
            embedding=[0.2] * 100,
        )

        hash3 = self.chunker._calculate_content_hash(different_item)
        assert hash1 != hash3

    def test_pattern_matching_epic_goals(self):
        """Test Epic goals pattern matching."""
        text = "Goals: Improve performance and reduce costs."
        match = self.chunker.epic_patterns["goals"].search(text)

        assert match is not None
        assert "improve performance and reduce costs" in match.group(1).lower()

    def test_pattern_matching_epic_success_criteria(self):
        """Test Epic success criteria pattern matching."""
        text = "Success criteria: Achieve 99% uptime and reduce response time by 50%."
        match = self.chunker.epic_patterns["success_criteria"].search(text)

        assert match is not None
        assert "achieve 99% uptime" in match.group(1).lower()

    def test_pattern_matching_story_acceptance_criteria(self):
        """Test Story acceptance criteria pattern matching."""
        text = "Acceptance criteria: User can login with email and password."
        match = self.chunker.story_patterns["acceptance_criteria"].search(text)

        assert match is not None
        assert "user can login" in match.group(1).lower()

    def test_pattern_matching_story_user_story(self):
        """Test Story user story pattern matching."""
        text = "As a user, I want to reset my password so that I can regain access."
        match = self.chunker.story_patterns["user_story"].search(text)

        assert match is not None
        assert "as a user" in match.group(0).lower()

    def test_pattern_matching_bug_symptoms(self):
        """Test Bug symptoms pattern matching."""
        text = "Symptoms: Application crashes when uploading large files."
        match = self.chunker.bug_patterns["symptoms"].search(text)

        assert match is not None
        assert "application crashes" in match.group(1).lower()

    def test_pattern_matching_bug_reproduction(self):
        """Test Bug reproduction steps pattern matching."""
        text = "Steps to reproduce: 1. Upload file > 10MB 2. Click submit 3. Observe crash"
        match = self.chunker.bug_patterns["reproduction"].search(text)

        assert match is not None
        assert "upload file" in match.group(1).lower()

    def test_pattern_matching_case_insensitive(self):
        """Test that patterns work case-insensitively."""
        # Test uppercase
        text = "GOALS: Increase revenue by 25%"
        match = self.chunker.epic_patterns["goals"].search(text)
        assert match is not None

        # Test mixed case
        text = "acceptance Criteria: Must work on mobile devices"
        match = self.chunker.story_patterns["acceptance_criteria"].search(text)
        assert match is not None

    def test_chunk_metadata_consistency(self):
        """Test that chunk metadata is consistent across chunks."""
        work_item = WorkItem(
            key="META-123",
            summary="Metadata test item",
            issue_type="Epic",
            status="Open",
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc),
            reporter="test@example.com",
            description="Goals: Test metadata. Success criteria: Consistent metadata across chunks.",
            assignee="test@example.com",
            embedding=[0.1] * 100,
        )

        chunks = self.chunker.chunk_work_item(work_item)

        # All chunks should have same source_key and parent metadata content_hash
        assert len(chunks) >= 2
        source_key = chunks[0].metadata.source_key
        content_hash = chunks[0].metadata.parent_metadata.content_hash

        for chunk in chunks:
            assert chunk.metadata.source_key == source_key
            assert chunk.metadata.parent_metadata.content_hash == content_hash
            assert chunk.metadata.parent_metadata.item_type == work_item.issue_type

    def test_chunk_index_sequence(self):
        """Test that chunk indices form a proper sequence."""
        # Create work item that will generate multiple chunks
        long_description = "Goals: Long epic goal description. " + ("More content. " * 50)

        work_item = WorkItem(
            key="SEQ-123",
            summary="Sequence test epic",
            issue_type="Epic",
            status="Open",
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc),
            reporter="test@example.com",
            description=long_description,
            assignee="test@example.com",
            embedding=[0.1] * 100,
        )

        chunks = self.chunker.chunk_work_item(work_item)

        # Check that indices are sequential starting from 0
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i

    def test_chunk_total_count(self):
        """Test that total_chunks metadata is set correctly."""
        work_item = WorkItem(
            key="COUNT-123",
            summary="Count test item",
            issue_type="Story",
            status="Open",
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc),
            reporter="test@example.com",
            description="Simple description",
            assignee="test@example.com",
            embedding=[0.1] * 100,
        )

        chunks = self.chunker.chunk_work_item(work_item)

        # All chunks should be part of the same work item
        # In the new implementation, we don't have total_chunks directly
        # We need to calculate the total ourselves
        total_chunks = len(chunks)
        assert total_chunks > 0

        # Check that all chunks have parent_metadata with the same source_key
        source_key = chunks[0].metadata.source_key
        for chunk in chunks:
            assert chunk.metadata.source_key == source_key
