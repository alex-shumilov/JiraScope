"""Incremental processing with change detection and caching."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from ..models import ProcessingResult, WorkItem
from ..utils.logging import StructuredLogger
from .embedding_processor import EmbeddingProcessor

logger = StructuredLogger(__name__)


class IncrementalProcessor:
    """Handle incremental updates and change detection."""

    def __init__(self, config, cache_dir: Path | None = None):
        self.config = config
        self.cache_dir = cache_dir or Path.home() / ".jirascope" / "incremental_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.cache_dir / "metadata.json"
        self.tracked_items_file = self.cache_dir / "tracked_items.json"

        self.embedding_processor = EmbeddingProcessor(config)

    async def process_incremental_updates(
        self, new_items: list[WorkItem], updated_items: list[WorkItem]
    ) -> ProcessingResult:
        """Process only new and updated items efficiently."""
        logger.info(
            f"Processing incremental updates: {len(new_items)} new, {len(updated_items)} updated"
        )
        start_time = time.time()

        try:
            # Load existing metadata
            metadata = self._load_metadata()
            tracked_items = self._load_tracked_items()

            # Filter out items that haven't actually changed
            items_to_process = []

            # Add all new items
            items_to_process.extend(new_items)

            # Filter updated items
            for item in updated_items:
                cached_hash = tracked_items.get(item.key, {}).get("content_hash")
                current_hash = self._calculate_content_hash(item)

                if cached_hash != current_hash:
                    items_to_process.append(item)
                    logger.debug(f"Item {item.key} has changed, will reprocess")
                else:
                    logger.debug(f"Item {item.key} unchanged, skipping")

            if not items_to_process:
                logger.info("No items need processing after change detection")
                return ProcessingResult(
                    skipped_items=len(updated_items), processing_time=time.time() - start_time
                )

            # Process the filtered items
            result = await self.embedding_processor.process_work_items(items_to_process)

            # Update tracking metadata
            self._update_tracking_data(items_to_process, metadata, tracked_items)

            # Save updated metadata
            self._save_metadata(metadata)
            self._save_tracked_items(tracked_items)

            logger.log_operation(
                "process_incremental_updates",
                result.processing_time,
                success=result.failed_items == 0,
                new_items=len(new_items),
                updated_items=len(updated_items),
                actually_processed=result.processed_items,
                skipped=result.skipped_items,
            )

            return result

        except Exception as e:
            logger.exception("Failed to process incremental updates", error=str(e))
            return ProcessingResult(
                failed_items=len(new_items) + len(updated_items),
                errors=[f"Incremental processing failed: {e!s}"],
                processing_time=time.time() - start_time,
            )

    def get_last_sync_timestamp(self, project_key: str) -> str | None:
        """Get the last sync timestamp for a project."""
        metadata = self._load_metadata()
        project_data = metadata.get("projects", {}).get(project_key, {})
        return project_data.get("last_sync")

    def update_last_sync_timestamp(self, project_key: str, timestamp: str | None = None):
        """Update the last sync timestamp for a project."""
        if timestamp is None:
            timestamp = datetime.now(UTC).isoformat()

        metadata = self._load_metadata()
        if "projects" not in metadata:
            metadata["projects"] = {}
        if project_key not in metadata["projects"]:
            metadata["projects"][project_key] = {}

        metadata["projects"][project_key]["last_sync"] = timestamp
        self._save_metadata(metadata)

        logger.info(f"Updated last sync timestamp for {project_key}: {timestamp}")

    def get_tracked_items(self, project_key: str | None = None) -> set[str]:
        """Get set of tracked item keys, optionally filtered by project."""
        tracked_items = self._load_tracked_items()

        if project_key:
            return {
                key for key, data in tracked_items.items() if data.get("project_key") == project_key
            }

        return set(tracked_items.keys())

    def get_tracked_epics(self, project_key: str | None = None) -> set[str]:
        """Get set of tracked epic keys."""
        tracked_items = self._load_tracked_items()

        epic_keys = set()
        for key, data in tracked_items.items():
            if data.get("issue_type", "").lower() == "epic":
                if not project_key or data.get("project_key") == project_key:
                    epic_keys.add(key)

            # Also include epic_key references
            epic_key = data.get("epic_key")
            if epic_key and (not project_key or data.get("project_key") == project_key):
                epic_keys.add(epic_key)

        return epic_keys

    def cleanup_old_cache(self, days: int = 30):
        """Clean up old cache files."""
        logger.info(f"Cleaning up cache files older than {days} days")

        cutoff_time = time.time() - (days * 24 * 60 * 60)
        cleaned_count = 0

        try:
            for cache_file in self.cache_dir.glob("*.hash"):
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    cleaned_count += 1

            logger.info(f"Cleaned up {cleaned_count} old cache files")

        except Exception as e:
            logger.exception("Failed to cleanup old cache files", error=str(e))

    def get_cache_statistics(self) -> dict[str, any]:
        """Get statistics about the cache."""
        try:
            metadata = self._load_metadata()
            tracked_items = self._load_tracked_items()

            # Count files in cache directory
            hash_files = list(self.cache_dir.glob("*.hash"))

            # Calculate cache size
            total_size = sum(f.stat().st_size for f in self.cache_dir.iterdir() if f.is_file())

            return {
                "cache_directory": str(self.cache_dir),
                "tracked_items_count": len(tracked_items),
                "hash_files_count": len(hash_files),
                "total_cache_size_bytes": total_size,
                "projects_tracked": len(metadata.get("projects", {})),
                "last_cleanup": metadata.get("last_cleanup"),
                "cache_created": metadata.get("created_at"),
            }

        except Exception as e:
            logger.exception("Failed to get cache statistics", error=str(e))
            return {"error": str(e)}

    def _load_metadata(self) -> dict:
        """Load incremental processing metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load metadata, starting fresh", error=str(e))
            else:
                # Successfully loaded metadata from file
                pass

        # Return default metadata
        return {
            "created_at": datetime.now(UTC).isoformat(),
            "version": "1.0",
            "projects": {},
        }

    def _save_metadata(self, metadata: dict):
        """Save incremental processing metadata."""
        try:
            metadata["updated_at"] = datetime.now(UTC).isoformat()
            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.exception("Failed to save metadata", error=str(e))
        else:
            # Successfully saved metadata
            pass

    def _load_tracked_items(self) -> dict[str, dict]:
        """Load tracked items data."""
        if self.tracked_items_file.exists():
            try:
                with open(self.tracked_items_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load tracked items, starting fresh", error=str(e))
            else:
                # Successfully loaded tracked items from file
                pass

        return {}

    def _save_tracked_items(self, tracked_items: dict[str, dict]):
        """Save tracked items data."""
        try:
            with open(self.tracked_items_file, "w") as f:
                json.dump(tracked_items, f, indent=2)
        except Exception as e:
            logger.exception("Failed to save tracked items", error=str(e))
        else:
            # Successfully saved tracked items
            pass

    def _calculate_content_hash(self, item: WorkItem) -> str:
        """Calculate hash of item content for change detection."""
        return self.embedding_processor._calculate_item_hash(item)

    def _update_tracking_data(
        self, items: list[WorkItem], metadata: dict, tracked_items: dict[str, dict]
    ):
        """Update tracking data with processed items."""
        for item in items:
            # Extract project key from item key (e.g., "PROJ-123" -> "PROJ")
            project_key = item.key.split("-")[0] if "-" in item.key else "UNKNOWN"

            tracked_items[item.key] = {
                "content_hash": self._calculate_content_hash(item),
                "last_processed": datetime.now(UTC).isoformat(),
                "project_key": project_key,
                "issue_type": item.issue_type,
                "epic_key": item.epic_key,
                "parent_key": item.parent_key,
                "last_updated": item.updated.isoformat(),
            }

            # Update project statistics
            if "projects" not in metadata:
                metadata["projects"] = {}
            if project_key not in metadata["projects"]:
                metadata["projects"][project_key] = {
                    "items_count": 0,
                    "first_seen": datetime.now(UTC).isoformat(),
                }

            # Count items per project (rough estimate)
            project_items = sum(
                1 for data in tracked_items.values() if data.get("project_key") == project_key
            )
            metadata["projects"][project_key]["items_count"] = project_items
