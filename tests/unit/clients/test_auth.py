"""Tests for auth client functionality."""

import json
import socket
import time
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest

from src.jirascope.clients.auth import (
    AuthCallbackHandler,
    AuthError,
    AuthHTTPServer,
    AuthTokens,
    SSEAuthenticator,
    find_available_port,
)


class TestAuthTokens:
    """Test AuthTokens dataclass functionality."""

    def test_auth_tokens_creation(self):
        """Test creating AuthTokens with required fields."""
        tokens = AuthTokens(
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            expires_at=time.time() + 3600,
            token_type="Bearer",
            scope="read:jira-work",
        )

        assert tokens.access_token == "test_access_token"
        assert tokens.refresh_token == "test_refresh_token"
        assert tokens.token_type == "Bearer"
        assert tokens.scope == "read:jira-work"
        assert tokens.expires_at is not None

    def test_auth_tokens_minimal_creation(self):
        """Test creating AuthTokens with only required fields."""
        tokens = AuthTokens(access_token="minimal_token")

        assert tokens.access_token == "minimal_token"
        assert tokens.refresh_token is None
        assert tokens.expires_at is None
        assert tokens.token_type == "Bearer"
        assert tokens.scope is None

    def test_is_expired_not_expired(self):
        """Test token expiry check when token is not expired."""
        future_time = time.time() + 3600  # 1 hour in future
        tokens = AuthTokens(access_token="test_token", expires_at=future_time)

        assert not tokens.is_expired

    def test_is_expired_expired(self):
        """Test token expiry check when token is expired."""
        past_time = time.time() - 3600  # 1 hour in past
        tokens = AuthTokens(access_token="test_token", expires_at=past_time)

        assert tokens.is_expired

    def test_is_expired_no_expiry(self):
        """Test token expiry check when no expiry is set."""
        tokens = AuthTokens(access_token="test_token")

        assert not tokens.is_expired

    def test_is_expired_buffer(self):
        """Test token expiry check with 60-second buffer."""
        # Token expires in 30 seconds - should be considered expired due to buffer
        near_future = time.time() + 30
        tokens = AuthTokens(access_token="test_token", expires_at=near_future)

        assert tokens.is_expired

    def test_to_dict(self):
        """Test converting AuthTokens to dictionary."""
        tokens = AuthTokens(
            access_token="test_access",
            refresh_token="test_refresh",
            expires_at=1234567890.0,
            token_type="Bearer",
            scope="read:jira-work",
        )

        expected = {
            "access_token": "test_access",
            "refresh_token": "test_refresh",
            "expires_at": 1234567890.0,
            "token_type": "Bearer",
            "scope": "read:jira-work",
        }

        assert tokens.to_dict() == expected

    def test_from_dict(self):
        """Test creating AuthTokens from dictionary."""
        data = {
            "access_token": "test_access",
            "refresh_token": "test_refresh",
            "expires_at": 1234567890.0,
            "token_type": "Bearer",
            "scope": "read:jira-work",
        }

        tokens = AuthTokens.from_dict(data)

        assert tokens.access_token == "test_access"
        assert tokens.refresh_token == "test_refresh"
        assert tokens.expires_at == 1234567890.0
        assert tokens.token_type == "Bearer"
        assert tokens.scope == "read:jira-work"


class TestAuthError:
    """Test AuthError exception."""

    def test_auth_error_creation(self):
        """Test creating AuthError exception."""
        error = AuthError("Authentication failed")
        assert str(error) == "Authentication failed"
        assert isinstance(error, Exception)

    def test_auth_error_inheritance(self):
        """Test AuthError inheritance from Exception."""
        error = AuthError("Test error")
        assert isinstance(error, Exception)


class TestAuthHTTPServer:
    """Test AuthHTTPServer functionality."""

    def test_auth_http_server_initialization(self):
        """Test AuthHTTPServer initialization."""
        server = AuthHTTPServer(("localhost", 0), AuthCallbackHandler)

        assert server.auth_code is None
        assert server.auth_error is None
        assert server.auth_state is None

        server.server_close()

    def test_auth_http_server_attributes(self):
        """Test AuthHTTPServer has required attributes."""
        server = AuthHTTPServer(("localhost", 0), AuthCallbackHandler)

        # Check that attributes exist and can be set
        server.auth_code = "test_code"
        server.auth_error = "test_error"
        server.auth_state = "test_state"

        assert server.auth_code == "test_code"
        assert server.auth_error == "test_error"
        assert server.auth_state == "test_state"

        server.server_close()


class TestAuthCallbackHandler:
    """Test AuthCallbackHandler functionality."""

    def _get_handler(self, mock_server):
        """Helper to create a configured handler instance."""
        handler = AuthCallbackHandler.__new__(AuthCallbackHandler)
        handler.server = mock_server
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        handler.wfile = Mock()
        handler.wfile.write = Mock()
        return handler

    def test_callback_handler_success_path(self):
        """Test handling successful OAuth callback."""
        mock_server = Mock(spec=AuthHTTPServer)
        mock_server.auth_code = None
        mock_server.auth_state = None
        handler = self._get_handler(mock_server)

        # Mock the path with OAuth success parameters
        handler.path = "/callback?code=test_auth_code&state=test_state"

        # Execute
        handler.do_GET()

        # Verify server state was updated
        assert mock_server.auth_code == "test_auth_code"
        assert mock_server.auth_state == "test_state"

        # Verify HTTP response
        handler.send_response.assert_called_with(200)
        handler.send_header.assert_called_with("Content-type", "text/html")
        handler.end_headers.assert_called_once()

        # Verify write was called with bytes
        handler.wfile.write.assert_called_once()
        call_args = handler.wfile.write.call_args[0][0]
        assert isinstance(call_args, bytes)
        assert b"Authentication Successful" in call_args

    def test_callback_handler_error_path(self):
        """Test handling OAuth callback with error."""
        mock_server = Mock(spec=AuthHTTPServer)
        mock_server.auth_error = None
        handler = self._get_handler(mock_server)

        # Mock the path with OAuth error parameters
        handler.path = "/callback?error=access_denied&error_description=User%20denied%20access"

        # Execute
        handler.do_GET()

        # Verify server error state was updated
        assert mock_server.auth_error == "access_denied: User denied access"

        # Verify HTTP error response
        handler.send_response.assert_called_with(400)
        handler.send_header.assert_called_with("Content-type", "text/html")
        handler.end_headers.assert_called_once()

        # Verify write was called with bytes
        handler.wfile.write.assert_called_once()
        call_args = handler.wfile.write.call_args[0][0]
        assert isinstance(call_args, bytes)
        assert b"Authentication Error" in call_args

    def test_callback_handler_log_message_suppressed(self):
        """Test that log_message does nothing."""
        mock_server = Mock(spec=AuthHTTPServer)
        handler = self._get_handler(mock_server)
        # This test ensures that the log_message method is overridden to be silent.
        # We can just call it and assert that no exceptions are raised.
        try:
            handler.log_message("GET %s %s", "/test", "200")
        except Exception as e:
            pytest.fail(f"log_message raised an unexpected exception: {e}")


class TestSSEAuthenticator:
    """Test SSEAuthenticator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_endpoint = "https://test-jira.atlassian.net"
        self.authenticator = SSEAuthenticator(endpoint="http://test-endpoint.com")

    def test_authenticator_initialization_basic(self):
        """Test basic SSEAuthenticator initialization."""
        auth = SSEAuthenticator(self.test_endpoint)

        assert auth.endpoint == self.test_endpoint
        assert auth.client_id == "jirascope-cli"
        assert auth.client_secret is None
        assert auth.preferred_redirect_port is None
        assert auth.scope == "read:jira-work read:jira-user write:jira-work"
        assert auth.cache_file.name == "auth_cache.json"

    def test_authenticator_initialization_with_params(self):
        """Test SSEAuthenticator initialization with custom parameters."""
        auth = SSEAuthenticator(
            endpoint=self.test_endpoint,
            client_id="custom-client",
            client_secret="secret123",
            redirect_port=8090,
            scope="read:jira-work",
        )

        assert auth.endpoint == self.test_endpoint
        assert auth.client_id == "custom-client"
        assert auth.client_secret == "secret123"
        assert auth.preferred_redirect_port == 8090
        assert auth.scope == "read:jira-work"

    def test_authenticator_initialization_strips_endpoint_slash(self):
        """Test that endpoint trailing slash is stripped."""
        auth = SSEAuthenticator("https://test-jira.atlassian.net/")
        assert auth.endpoint == "https://test-jira.atlassian.net"

    def test_authenticator_initialization_empty_endpoint_error(self):
        """Test that empty endpoint raises AuthError."""
        with pytest.raises(AuthError, match="Endpoint cannot be empty"):
            SSEAuthenticator("")

    @patch("src.jirascope.clients.auth.Path.mkdir")
    def test_authenticator_cache_directory_creation(self, mock_mkdir):
        """Test that the cache directory is created if it doesn't exist."""
        self.authenticator = SSEAuthenticator(endpoint="http://test-endpoint.com")
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_load_cached_tokens_no_file(self):
        """Test loading tokens when cache file doesn't exist."""
        with patch("src.jirascope.clients.auth.Path.exists", return_value=False):
            tokens = self.authenticator._load_cached_tokens()
            assert tokens is None

    def test_load_cached_tokens_invalid_json(self):
        """Test loading tokens when cache file has invalid JSON."""
        with (
            patch("src.jirascope.clients.auth.Path.exists", return_value=True),
            patch("src.jirascope.clients.auth.Path.open", mock_open(read_data="invalid json")),
        ):
            tokens = self.authenticator._load_cached_tokens()
            assert tokens is None

    def test_load_cached_tokens_success(self):
        """Test loading tokens successfully from cache."""
        future_time = time.time() + 3600
        token_data = {
            "access_token": "cached_access",
            "refresh_token": "cached_refresh",
            "expires_at": future_time,
            "token_type": "Bearer",
            "scope": "read:all",
        }

        with (
            patch("src.jirascope.clients.auth.Path.exists", return_value=True),
            patch(
                "src.jirascope.clients.auth.Path.open", mock_open(read_data=json.dumps(token_data))
            ),
        ):
            tokens = self.authenticator._load_cached_tokens()

            assert tokens is not None
            assert tokens.access_token == "cached_access"
            assert tokens.refresh_token == "cached_refresh"
            assert tokens.expires_at == future_time

    @patch("src.jirascope.clients.auth.Path.write_text")
    def test_save_tokens(self, mock_write_text):
        """Test saving tokens to cache file."""
        tokens = AuthTokens(
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            expires_at=12345.67,
            token_type="Bearer",
            scope="read",
        )
        self.authenticator._save_tokens(tokens)

        expected_json = json.dumps(tokens.to_dict(), indent=2)
        mock_write_text.assert_called_once_with(expected_json)

    def test_clear_cache_file_exists(self):
        """Test clearing cache when file exists."""
        with (
            patch("src.jirascope.clients.auth.Path.exists", return_value=True),
            patch("src.jirascope.clients.auth.Path.unlink") as mock_unlink,
        ):
            self.authenticator.clear_cache()
            mock_unlink.assert_called_once()

    def test_clear_cache_file_not_exists(self):
        """Test clearing cache when file does not exist."""
        with (
            patch("src.jirascope.clients.auth.Path.exists", return_value=False),
            patch("src.jirascope.clients.auth.Path.unlink") as mock_unlink,
        ):
            self.authenticator.clear_cache()
            mock_unlink.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_token_success(self):
        """Test successful token validation."""
        auth = SSEAuthenticator(self.test_endpoint)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await auth.validate_token("valid_token")

            assert result is True
            mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_token_failure(self):
        """Test token validation failure."""
        auth = SSEAuthenticator(self.test_endpoint)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 401
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await auth.validate_token("invalid_token")

            assert result is False

    @pytest.mark.asyncio
    async def test_validate_token_network_error(self):
        """Test token validation with network error."""
        auth = SSEAuthenticator(self.test_endpoint)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Network error")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await auth.validate_token("test_token")

            assert result is False

    @pytest.mark.asyncio
    async def test_get_auth_tokens_cached_valid(self):
        """Test getting tokens when cached tokens are valid."""
        auth = SSEAuthenticator(self.test_endpoint)

        valid_tokens = AuthTokens(access_token="cached_valid_token", expires_at=time.time() + 3600)

        with patch.object(auth, "_load_cached_tokens", return_value=valid_tokens):
            result = await auth.get_auth_tokens()

            assert result is valid_tokens
            assert result.access_token == "cached_valid_token"

    @pytest.mark.asyncio
    async def test_get_auth_tokens_force_refresh(self):
        """Test forcing token refresh."""
        auth = SSEAuthenticator(self.test_endpoint)

        new_tokens = AuthTokens(access_token="new_token")

        with (
            patch.object(auth, "_load_cached_tokens") as mock_load,
            patch.object(auth, "_perform_oauth_flow", return_value=new_tokens) as mock_oauth,
            patch.object(auth, "_save_tokens") as mock_save,
        ):

            result = await auth.get_auth_tokens(force_refresh=True)

            # Should not load cached tokens when force_refresh=True
            mock_load.assert_not_called()
            mock_oauth.assert_called_once()
            mock_save.assert_called_once_with(new_tokens)
            assert result is new_tokens

    @pytest.mark.asyncio
    async def test_refresh_tokens_success(self):
        """Test successful token refresh."""
        auth = SSEAuthenticator(self.test_endpoint)

        refresh_response = {
            "access_token": "refreshed_token",
            "refresh_token": "new_refresh_token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = refresh_response
            mock_response.raise_for_status = Mock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await auth._refresh_tokens("old_refresh_token")

            assert result.access_token == "refreshed_token"
            assert result.refresh_token == "new_refresh_token"
            assert result.token_type == "Bearer"
            assert result.expires_at is not None

    @pytest.mark.asyncio
    async def test_refresh_tokens_http_error(self):
        """Test token refresh with HTTP error."""
        auth = SSEAuthenticator(self.test_endpoint)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = Exception("HTTP 400")
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(AuthError):
                await auth._refresh_tokens("invalid_refresh_token")


class TestFindAvailablePort:
    """Test find_available_port utility function."""

    def test_find_available_port_default_range(self):
        """Test finding a port within the default range."""
        # This is a basic test and might fail if ports are genuinely occupied.
        # It serves as a simple sanity check.
        port = find_available_port()
        assert 8080 <= port <= 8200

    def test_find_available_port_custom_range(self):
        """Test finding a port within a custom range."""
        port = find_available_port(start_port=9000, max_port=9005)
        assert 9000 <= port <= 9005

    @patch("socket.socket")
    def test_find_available_port_socket_operations(self, mock_socket_class):
        """Test socket operations during port finding."""
        mock_socket_instance = Mock()
        mock_socket_class.return_value.__enter__.return_value = mock_socket_instance

        # Mock behavior
        mock_socket_instance.bind.side_effect = [None, OSError(98)]  # First port free, second busy

        # Test finding the first available port
        port = find_available_port(start_port=8080, max_port=8081)
        assert port == 8080

        mock_socket_class.assert_called_with(socket.AF_INET, socket.SOCK_STREAM)
        mock_socket_instance.bind.assert_called_with(("localhost", 8080))

    @patch("socket.socket")
    def test_find_available_port_all_ports_busy(self, mock_socket_class):
        """Test behavior when all ports in the range are busy."""
        mock_socket_instance = Mock()
        mock_socket_class.return_value.__enter__.return_value = mock_socket_instance

        # Mock all ports as busy
        mock_socket_instance.bind.side_effect = OSError(98)

        with pytest.raises(RuntimeError, match="No available ports found between 8080 and 8081"):
            find_available_port(start_port=8080, max_port=8081)  # Small range to test quickly

        assert mock_socket_instance.bind.call_count == 1
