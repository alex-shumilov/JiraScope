"""Smart content chunking for Jira work items with content-aware strategies."""

import hashlib
import re
from dataclasses import dataclass
from typing import List

from ..models.metadata_schema import ChunkMetadata
from ..models.work_item import WorkItem


@dataclass
class Chunk:
    """A chunk of content with associated metadata."""

    text: str
    metadata: ChunkMetadata
    chunk_type: str  # 'summary', 'description', 'acceptance_criteria', etc.

    @property
    def chunk_id(self) -> str:
        """Generate a unique chunk ID."""
        content = f"{self.metadata.source_key}_{self.chunk_type}_{self.metadata.chunk_index}"
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:12]


class SmartChunker:
    """Content-aware chunker that adapts strategy based on Jira item type."""

    def __init__(
        self, max_chunk_size: int = 800, min_chunk_size: int = 200, overlap_size: int = 50
    ):
        """
        Initialize the smart chunker.

        Args:
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk
            overlap_size: Character overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size

        # Content patterns for different item types
        self.epic_patterns = {
            "goals": re.compile(
                r"(?:goals?|objectives?|outcomes?):\s*(.+?)(?:\n\n|\n[A-Z]|$)",
                re.IGNORECASE | re.DOTALL,
            ),
            "success_criteria": re.compile(
                r"(?:success criteria|definition of done):\s*(.+?)(?:\n\n|\n[A-Z]|$)",
                re.IGNORECASE | re.DOTALL,
            ),
        }

        self.story_patterns = {
            "acceptance_criteria": re.compile(
                r"(?:acceptance criteria|ac):\s*(.+?)(?:\n\n|\n[A-Z]|$)", re.IGNORECASE | re.DOTALL
            ),
            "user_story": re.compile(r"(?:as a|as an)\s+(.+?)(?:\n|$)", re.IGNORECASE),
            "requirements": re.compile(
                r"(?:requirements?|specs?):\s*(.+?)(?:\n\n|\n[A-Z]|$)", re.IGNORECASE | re.DOTALL
            ),
        }

        self.bug_patterns = {
            "symptoms": re.compile(
                r"(?:symptoms?|observed behavior):\s*(.+?)(?:\n\n|\n[A-Z]|$)",
                re.IGNORECASE | re.DOTALL,
            ),
            "reproduction": re.compile(
                r"(?:steps to reproduce|repro steps?):\s*(.+?)(?:\n\n|\n[A-Z]|$)",
                re.IGNORECASE | re.DOTALL,
            ),
            "expected": re.compile(
                r"(?:expected behavior|expected result):\s*(.+?)(?:\n\n|\n[A-Z]|$)",
                re.IGNORECASE | re.DOTALL,
            ),
            "solution": re.compile(
                r"(?:solution|fix|resolution):\s*(.+?)(?:\n\n|\n[A-Z]|$)", re.IGNORECASE | re.DOTALL
            ),
        }

    def chunk_work_item(self, work_item: WorkItem) -> List[Chunk]:
        """
        Chunk a work item using content-aware strategy.

        Args:
            work_item: The work item to chunk

        Returns:
            List of content chunks with metadata
        """
        chunks = []

        # Always include summary as first chunk
        if work_item.summary:
            summary_chunk = self._create_chunk(
                text=work_item.summary, work_item=work_item, chunk_type="summary", chunk_index=0
            )
            chunks.append(summary_chunk)

        # Apply type-specific chunking strategy
        if work_item.issue_type == "Epic":
            chunks.extend(self._chunk_epic(work_item, start_index=len(chunks)))
        elif work_item.issue_type == "Story":
            chunks.extend(self._chunk_story(work_item, start_index=len(chunks)))
        elif work_item.issue_type == "Bug":
            chunks.extend(self._chunk_bug(work_item, start_index=len(chunks)))
        else:
            # Generic chunking for other types
            chunks.extend(self._chunk_generic(work_item, start_index=len(chunks)))

        return chunks

    def _chunk_epic(self, work_item: WorkItem, start_index: int) -> List[Chunk]:
        """Epic-specific chunking strategy."""
        chunks = []
        current_index = start_index

        if work_item.description:
            # Extract goals and success criteria
            goals_match = self.epic_patterns["goals"].search(work_item.description)
            if goals_match:
                goals_chunk = self._create_chunk(
                    text=f"Epic Goals: {goals_match.group(1).strip()}",
                    work_item=work_item,
                    chunk_type="goals",
                    chunk_index=current_index,
                )
                chunks.append(goals_chunk)
                current_index += 1

            success_match = self.epic_patterns["success_criteria"].search(work_item.description)
            if success_match:
                success_chunk = self._create_chunk(
                    text=f"Success Criteria: {success_match.group(1).strip()}",
                    work_item=work_item,
                    chunk_type="success_criteria",
                    chunk_index=current_index,
                )
                chunks.append(success_chunk)
                current_index += 1

            # Chunk remaining description
            remaining_desc = work_item.description
            if goals_match:
                remaining_desc = remaining_desc.replace(goals_match.group(0), "")
            if success_match:
                remaining_desc = remaining_desc.replace(success_match.group(0), "")

            if remaining_desc.strip():
                desc_chunks = self._chunk_text(
                    text=remaining_desc.strip(),
                    work_item=work_item,
                    chunk_type="description",
                    start_index=current_index,
                )
                chunks.extend(desc_chunks)

        return chunks

    def _chunk_story(self, work_item: WorkItem, start_index: int) -> List[Chunk]:
        """Story-specific chunking strategy."""
        chunks = []
        current_index = start_index

        if work_item.description:
            # Extract acceptance criteria
            ac_match = self.story_patterns["acceptance_criteria"].search(work_item.description)
            if ac_match:
                ac_chunk = self._create_chunk(
                    text=f"Acceptance Criteria: {ac_match.group(1).strip()}",
                    work_item=work_item,
                    chunk_type="acceptance_criteria",
                    chunk_index=current_index,
                )
                chunks.append(ac_chunk)
                current_index += 1

            # Extract user story format
            user_story_match = self.story_patterns["user_story"].search(work_item.description)
            if user_story_match:
                user_story_chunk = self._create_chunk(
                    text=f"User Story: {user_story_match.group(0).strip()}",
                    work_item=work_item,
                    chunk_type="user_story",
                    chunk_index=current_index,
                )
                chunks.append(user_story_chunk)
                current_index += 1

            # Chunk remaining description
            remaining_desc = work_item.description
            if ac_match:
                remaining_desc = remaining_desc.replace(ac_match.group(0), "")
            if user_story_match:
                remaining_desc = remaining_desc.replace(user_story_match.group(0), "")

            if remaining_desc.strip():
                desc_chunks = self._chunk_text(
                    text=remaining_desc.strip(),
                    work_item=work_item,
                    chunk_type="description",
                    start_index=current_index,
                )
                chunks.extend(desc_chunks)

        return chunks

    def _chunk_bug(self, work_item: WorkItem, start_index: int) -> List[Chunk]:
        """Bug-specific chunking strategy."""
        chunks = []
        current_index = start_index

        if work_item.description:
            # Extract structured bug information
            for section_name, pattern in self.bug_patterns.items():
                match = pattern.search(work_item.description)
                if match:
                    section_chunk = self._create_chunk(
                        text=f"{section_name.replace('_', ' ').title()}: {match.group(1).strip()}",
                        work_item=work_item,
                        chunk_type=section_name,
                        chunk_index=current_index,
                    )
                    chunks.append(section_chunk)
                    current_index += 1

            # Chunk any remaining description
            remaining_desc = work_item.description
            for pattern in self.bug_patterns.values():
                match = pattern.search(remaining_desc)
                if match:
                    remaining_desc = remaining_desc.replace(match.group(0), "")

            if remaining_desc.strip():
                desc_chunks = self._chunk_text(
                    text=remaining_desc.strip(),
                    work_item=work_item,
                    chunk_type="description",
                    start_index=current_index,
                )
                chunks.extend(desc_chunks)

        return chunks

    def _chunk_generic(self, work_item: WorkItem, start_index: int) -> List[Chunk]:
        """Generic chunking strategy for other item types."""
        chunks = []

        if work_item.description:
            desc_chunks = self._chunk_text(
                text=work_item.description,
                work_item=work_item,
                chunk_type="description",
                start_index=start_index,
            )
            chunks.extend(desc_chunks)

        return chunks

    def _chunk_text(
        self, text: str, work_item: WorkItem, chunk_type: str, start_index: int
    ) -> List[Chunk]:
        """
        Chunk text content while preserving sentence boundaries.

        Args:
            text: Text to chunk
            work_item: Source work item
            chunk_type: Type of chunk
            start_index: Starting chunk index

        Returns:
            List of text chunks
        """
        if len(text) <= self.max_chunk_size:
            # Text fits in one chunk
            chunk = self._create_chunk(
                text=text, work_item=work_item, chunk_type=chunk_type, chunk_index=start_index
            )
            return [chunk]

        chunks = []
        sentences = self._split_into_sentences(text)

        current_chunk = ""
        current_index = start_index

        for sentence in sentences:
            # Check if adding this sentence would exceed max size
            if len(current_chunk) + len(sentence) + 1 > self.max_chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunk = self._create_chunk(
                        text=current_chunk.strip(),
                        work_item=work_item,
                        chunk_type=chunk_type,
                        chunk_index=current_index,
                    )
                    chunks.append(chunk)
                    current_index += 1

                    # Start new chunk with overlap
                    if self.overlap_size > 0 and len(current_chunk) > self.overlap_size:
                        overlap_text = current_chunk[-self.overlap_size :]
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    # Single sentence is too long, chunk it by words
                    word_chunks = self._chunk_by_words(
                        sentence, work_item, chunk_type, current_index
                    )
                    chunks.extend(word_chunks)
                    current_index += len(word_chunks)
                    current_chunk = ""
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        # Add final chunk if it has content
        if current_chunk.strip():
            chunk = self._create_chunk(
                text=current_chunk.strip(),
                work_item=work_item,
                chunk_type=chunk_type,
                chunk_index=current_index,
            )
            chunks.append(chunk)

        return chunks

    def _chunk_by_words(
        self, text: str, work_item: WorkItem, chunk_type: str, start_index: int
    ) -> List[Chunk]:
        """Chunk text by words when sentences are too long."""
        words = text.split()
        chunks = []
        current_chunk = ""
        current_index = start_index

        for word in words:
            if len(current_chunk) + len(word) + 1 > self.max_chunk_size:
                if current_chunk:
                    chunk = self._create_chunk(
                        text=current_chunk.strip(),
                        work_item=work_item,
                        chunk_type=chunk_type,
                        chunk_index=current_index,
                    )
                    chunks.append(chunk)
                    current_index += 1
                    current_chunk = word
                else:
                    # Single word is too long, truncate it
                    truncated_word = word[: self.max_chunk_size - 3] + "..."
                    chunk = self._create_chunk(
                        text=truncated_word,
                        work_item=work_item,
                        chunk_type=chunk_type,
                        chunk_index=current_index,
                    )
                    chunks.append(chunk)
                    current_index += 1
            else:
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word

        if current_chunk.strip():
            chunk = self._create_chunk(
                text=current_chunk.strip(),
                work_item=work_item,
                chunk_type=chunk_type,
                chunk_index=current_index,
            )
            chunks.append(chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics."""
        # Simple sentence splitting - can be enhanced with more sophisticated NLP
        sentences = re.split(r"[.!?]+\s+", text)

        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _create_chunk(
        self, text: str, work_item: WorkItem, chunk_type: str, chunk_index: int
    ) -> Chunk:
        """Create a chunk with associated metadata."""
        # Create parent metadata from work item
        from ..models.metadata_schema import JiraItemMetadata

        parent_metadata = JiraItemMetadata(
            key=work_item.key,
            item_type=work_item.issue_type,
            status=work_item.status,
            priority=getattr(work_item, "priority", "Medium"),
            created=work_item.created,
            updated=work_item.updated,
            epic_key=work_item.epic_key,
            parent_key=work_item.parent_key,
            components=work_item.components,
            labels=work_item.labels,
            content_hash=self._calculate_content_hash(work_item),
        )

        # Create chunk metadata
        chunk_id = f"{work_item.key}_{chunk_type}_{chunk_index}"
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            source_key=work_item.key,
            chunk_type=chunk_type,
            chunk_index=chunk_index,
            parent_metadata=parent_metadata,
        )

        return Chunk(text=text, metadata=metadata, chunk_type=chunk_type)

    def _calculate_content_hash(self, work_item: WorkItem) -> str:
        """Calculate content hash for change detection."""
        content = f"{work_item.summary}|{work_item.description or ''}|{work_item.status}|{work_item.updated.isoformat()}"
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()
