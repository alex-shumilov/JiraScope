"""LMStudio client for embeddings generation."""

import logging

import httpx
import numpy as np

from ..core.config import Config

logger = logging.getLogger(__name__)


class LMStudioClient:
    """Client for generating embeddings via LMStudio."""

    def __init__(self, config: Config):
        self.config = config
        self.endpoint = config.lmstudio_endpoint
        self.session: httpx.AsyncClient | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.embedding_timeout),
            limits=httpx.Limits(max_connections=5),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.aclose()

    async def generate_embeddings(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not self.session:
            raise RuntimeError("LMStudio client not initialized. Use async context manager.")

        batch_size = batch_size or self.config.embedding_batch_size
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await self._generate_batch_embeddings(batch)
            all_embeddings.extend(batch_embeddings)

            logger.debug(
                f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
            )

        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings

    async def _generate_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        try:
            # Prepare texts with instruction prefix
            prepared_texts = [
                f"{self.config.embedding_instruction_prefix}{text[:self.config.embedding_max_tokens]}"
                for text in texts
            ]

            response = await self.session.post(
                f"{self.endpoint}/embeddings",
                json={"model": self.config.embedding_model, "input": prepared_texts},
            )
            response.raise_for_status()

            data = response.json()
            return [item["embedding"] for item in data["data"]]

        except httpx.HTTPError as e:
            logger.exception(f"Failed to generate embeddings: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if LMStudio is running and responsive."""
        if not self.session:
            raise RuntimeError("LMStudio client not initialized. Use async context manager.")

        try:
            response = await self.session.get(f"{self.endpoint}/models")
            response.raise_for_status()

            models = response.json()
            available_models = [model["id"] for model in models.get("data", [])]

            if self.config.embedding_model in available_models:
                logger.info("LMStudio health check passed")
                return True
            logger.warning(f"Embedding model {self.config.embedding_model} not available")
            return False

        except httpx.HTTPError as e:
            logger.exception(f"LMStudio health check failed: {e}")
            return False

    def calculate_similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
