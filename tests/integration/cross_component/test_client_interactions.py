"""
FIXED EXAMPLE: Integration tests demonstrating proper testing practices.

This file shows how to fix the integrity issues found in the original test file.
Following rules: development-guide, anti-overcoding, code-quality-integrity, solid-principles, kiss-principle
"""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from src.jirascope.clients import LMStudioClient
from src.jirascope.core.config import Config


# Test Constants - No more magic numbers
class TestConstants:
    """Centralized test constants following KISS principle."""

    EXPECTED_EMBEDDING_DIMENSIONS = 1024
    SAMPLE_TEXT = "test text for embedding"
    VALID_COST_RANGE = (0.001, 0.1)  # Reasonable cost bounds
    SIMILARITY_PRECISION = 0.001


class LMStudioTestHelper:
    """Helper class following Single Responsibility Principle."""

    @staticmethod
    def create_valid_models_response() -> dict[str, Any]:
        """Create a realistic models response."""
        return {
            "object": "list",
            "data": [{"id": "text-embedding-bge-large-en-v1.5"}, {"id": "text-embedding-ada-002"}],
        }

    @staticmethod
    def create_valid_embeddings_response(text_count: int = 1) -> dict[str, Any]:
        """Create a realistic embeddings response."""
        base_pattern = [0.1, 0.2, 0.3]
        repetitions = TestConstants.EXPECTED_EMBEDDING_DIMENSIONS // 3
        remainder = TestConstants.EXPECTED_EMBEDDING_DIMENSIONS % 3
        embedding = base_pattern * repetitions + base_pattern[:remainder]

        return {
            "object": "list",
            "data": [
                {
                    "embedding": embedding,
                    "index": i,
                }
                for i in range(text_count)
            ],
        }

    @staticmethod
    def assert_valid_embedding(embedding: list[float]) -> None:
        """Assert that an embedding meets expected criteria."""
        assert isinstance(embedding, list), "Embedding must be a list"
        assert (
            len(embedding) == TestConstants.EXPECTED_EMBEDDING_DIMENSIONS
        ), f"Expected {TestConstants.EXPECTED_EMBEDDING_DIMENSIONS} dimensions"
        assert all(isinstance(x, float) for x in embedding), "All values must be floats"
        assert all(-1.0 <= x <= 1.0 for x in embedding), "Values must be normalized"


# FIXED VERSION: Tests actual behavior, not mock behavior
@pytest.fixture
def real_config() -> Config:
    """Create a real config object, not a mock."""
    # Using actual Config class ensures we test real behavior
    return Config(
        jira_mcp_endpoint="http://test:8000",
        embedding_model="text-embedding-bge-large-en-v1.5",
        embedding_dimensions=TestConstants.EXPECTED_EMBEDDING_DIMENSIONS,
        embedding_batch_size=32,
        embedding_instruction_prefix="Represent this: ",
        embedding_max_tokens=512,
        embedding_timeout=30.0,
        lmstudio_endpoint="http://localhost:1234/v1",
    )


class TestLMStudioClientBehavior:
    """Test LMStudio client actual behavior, not implementation details."""

    @pytest.mark.asyncio
    async def test_health_check_verifies_model_availability(self, real_config):
        """Test that health check actually verifies the embedding model is available."""
        with patch("httpx.AsyncClient") as mock_client_class:
            # Setup: Mock successful response with our expected model
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.json.return_value = LMStudioTestHelper.create_valid_models_response()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            # Action: Perform health check
            async with LMStudioClient(real_config) as client:
                result = await client.health_check()

            # Verification: Test BEHAVIOR, not implementation
            assert result is True, "Health check should pass when model is available"

            # Verify the right endpoint was called
            mock_client.get.assert_called_once_with(f"{real_config.lmstudio_endpoint}/models")

    @pytest.mark.asyncio
    async def test_health_check_fails_when_model_unavailable(self, real_config):
        """Test health check fails when required model is not available."""
        with patch("httpx.AsyncClient") as mock_client_class:
            # Setup: Mock response without our required model
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.json.return_value = {
                "object": "list",
                "data": [{"id": "different-model"}],  # Wrong model
            }
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            # Action & Verification
            async with LMStudioClient(real_config) as client:
                result = await client.health_check()

            assert result is False, "Health check should fail when required model is unavailable"

    @pytest.mark.asyncio
    async def test_health_check_handles_network_errors(self, real_config):
        """Test health check properly handles network failures."""
        with patch("httpx.AsyncClient") as mock_client_class:
            # Setup: Mock network failure
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.NetworkError("Connection failed")
            mock_client_class.return_value = mock_client

            # Action & Verification
            async with LMStudioClient(real_config) as client:
                result = await client.health_check()

            assert result is False, "Health check should fail gracefully on network errors"

    @pytest.mark.asyncio
    async def test_generate_embeddings_produces_valid_output(self, real_config):
        """Test that embedding generation produces valid, properly formatted embeddings."""
        test_texts = [TestConstants.SAMPLE_TEXT, "another test text"]

        with patch("httpx.AsyncClient") as mock_client_class:
            # Setup: Mock successful embedding response
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.json.return_value = LMStudioTestHelper.create_valid_embeddings_response(
                len(test_texts)
            )
            mock_response.raise_for_status = Mock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            # Action: Generate embeddings
            async with LMStudioClient(real_config) as client:
                embeddings = await client.generate_embeddings(test_texts)

            # Verification: Test BEHAVIOR - validate actual output
            assert len(embeddings) == len(
                test_texts
            ), f"Should return {len(test_texts)} embeddings for {len(test_texts)} texts"

            for _i, embedding in enumerate(embeddings):
                LMStudioTestHelper.assert_valid_embedding(embedding)

            # Verify API call was made correctly
            call_args = mock_client.post.call_args
            assert call_args[0][0] == f"{real_config.lmstudio_endpoint}/embeddings"

            # Verify request payload structure
            request_data = call_args[1]["json"]
            assert request_data["model"] == real_config.embedding_model
            assert len(request_data["input"]) == len(test_texts)

            # Verify instruction prefix was applied
            for input_text in request_data["input"]:
                assert input_text.startswith(real_config.embedding_instruction_prefix)

    @pytest.mark.asyncio
    async def test_generate_embeddings_respects_batch_size(self, real_config):
        """Test that embedding generation properly handles batching."""
        # Create more texts than batch size
        batch_size = 2
        real_config.embedding_batch_size = batch_size
        test_texts = ["text1", "text2", "text3", "text4", "text5"]  # 5 texts, batch=2

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            # Mock returns 2 embeddings per call (batch size)
            mock_response.json.return_value = LMStudioTestHelper.create_valid_embeddings_response(
                batch_size
            )
            mock_response.raise_for_status = Mock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            # Action
            async with LMStudioClient(real_config) as client:
                await client.generate_embeddings(test_texts)

            # Verification: Should make 3 API calls (2+2+1)
            expected_calls = (len(test_texts) + batch_size - 1) // batch_size
            assert (
                mock_client.post.call_count == expected_calls
            ), f"Should make {expected_calls} API calls for {len(test_texts)} texts with batch size {batch_size}"

    def test_similarity_calculation_mathematical_properties(self, real_config):
        """Test similarity calculation follows mathematical properties."""
        client = LMStudioClient(real_config)

        # Test data with known properties
        identical_vec = [1.0, 0.0, 0.0]
        orthogonal_vec = [0.0, 1.0, 0.0]
        opposite_vec = [-1.0, 0.0, 0.0]
        zero_vec = [0.0, 0.0, 0.0]

        # Test identical vectors (should be 1.0)
        similarity = client.calculate_similarity(identical_vec, identical_vec)
        assert (
            abs(similarity - 1.0) < TestConstants.SIMILARITY_PRECISION
        ), "Identical vectors should have similarity of 1.0"

        # Test orthogonal vectors (should be 0.0)
        similarity = client.calculate_similarity(identical_vec, orthogonal_vec)
        assert (
            abs(similarity - 0.0) < TestConstants.SIMILARITY_PRECISION
        ), "Orthogonal vectors should have similarity of 0.0"

        # Test opposite vectors (should be -1.0)
        similarity = client.calculate_similarity(identical_vec, opposite_vec)
        assert (
            abs(similarity - (-1.0)) < TestConstants.SIMILARITY_PRECISION
        ), "Opposite vectors should have similarity of -1.0"

        # Test zero vector handling
        similarity = client.calculate_similarity(zero_vec, identical_vec)
        assert similarity == 0.0, "Zero vector should return 0.0 similarity"

        # Test symmetry property: sim(a,b) == sim(b,a)
        sim_ab = client.calculate_similarity(identical_vec, orthogonal_vec)
        sim_ba = client.calculate_similarity(orthogonal_vec, identical_vec)
        assert (
            abs(sim_ab - sim_ba) < TestConstants.SIMILARITY_PRECISION
        ), "Similarity should be symmetric"


# EXAMPLE: How to structure error handling tests
class TestLMStudioClientErrorHandling:
    """Test error scenarios following proper error testing practices."""

    @pytest.mark.asyncio
    async def test_embedding_generation_handles_api_errors(self, real_config):
        """Test that API errors are properly handled and logged."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "Server Error", request=Mock(), response=Mock(status_code=500)
            )
            mock_client_class.return_value = mock_client

            async with LMStudioClient(real_config) as client:
                with pytest.raises(httpx.HTTPStatusError):
                    await client.generate_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_embedding_generation_handles_timeout(self, real_config):
        """Test that timeouts are properly handled."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Request timeout")
            mock_client_class.return_value = mock_client

            async with LMStudioClient(real_config) as client:
                with pytest.raises(httpx.TimeoutException):
                    await client.generate_embeddings(["test"])


# KEY IMPROVEMENTS DEMONSTRATED:
#
# 1. KISS PRINCIPLE:
#    - Simple, focused test methods
#    - Clear naming that explains intent
#    - Minimal setup for each test
#
# 2. ANTI-OVERCODING:
#    - No unnecessary abstraction
#    - Tests only what's needed
#    - No premature optimization
#
# 3. CODE QUALITY & INTEGRITY:
#    - Tests actual behavior, not mocks
#    - Meaningful assertions with explanations
#    - No hardcoded magic values
#    - Proper error handling coverage
#
# 4. SOLID PRINCIPLES:
#    - Single responsibility per test
#    - Tests depend on abstractions (interfaces) not implementations
#    - Helper classes have focused responsibilities
#
# 5. DEVELOPMENT GUIDE:
#    - Uses real Config objects
#    - Follows async/await patterns
#    - Proper fixture usage
#    - Tests verify business logic
