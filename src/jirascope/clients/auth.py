"""Authentication module for JiraScope MCP client."""

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import socket
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


@dataclass
class AuthTokens:
    """Container for authentication tokens."""

    access_token: str
    refresh_token: str | None = None
    expires_at: float | None = None
    token_type: str = "Bearer"
    scope: str | None = None  # Track granted scopes

    @property
    def is_expired(self) -> bool:
        """Check if the access token is expired."""
        if not self.expires_at:
            return False
        return time.time() >= self.expires_at - 60  # 60 second buffer

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "token_type": self.token_type,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AuthTokens":
        """Create from dictionary."""
        return cls(**data)


class AuthError(Exception):
    """Authentication-related errors."""


class AuthHTTPServer(HTTPServer):
    """Custom HTTP server to store OAuth callback data."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auth_code: str | None = None
        self.auth_error: str | None = None
        self.auth_state: str | None = None


class AuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    def do_GET(self):
        """Handle GET request for OAuth callback."""
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)

        # Type cast to our custom server class
        auth_server: AuthHTTPServer = self.server  # type: ignore

        # Store the authorization code
        if "code" in params:
            auth_server.auth_code = params["code"][0]
            # Also store state for validation
            if "state" in params:
                auth_server.auth_state = params["state"][0]

            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            success_html = """
                <html>
                <head><title>Authentication Successful</title></head>
                <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; text-align: center; padding: 50px;">
                    <h1 style="color: #2e8b57;">Authentication Successful!</h1>
                    <p>You can close this browser window and return to JiraScope.</p>
                    <p style="color: #666; font-size: 0.9em;">This window will close automatically in 3 seconds.</p>
                    <script>
                        setTimeout(() => window.close(), 3000);
                    </script>
                </body>
                </html>
            """
            self.wfile.write(success_html.encode("utf-8"))
        elif "error" in params:
            error_description = params.get("error_description", ["Unknown error"])[0]
            auth_server.auth_error = f"{params['error'][0]}: {error_description}"
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            error_html = f"""
                <html>
                <head><title>Authentication Error</title></head>
                <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; text-align: center; padding: 50px;">
                    <h1 style="color: #dc3545;">Authentication Error</h1>
                    <p><strong>Error:</strong> {params['error'][0]}</p>
                    <p><strong>Description:</strong> {error_description}</p>
                    <p>Please close this window and try again.</p>
                </body>
                </html>
            """
            self.wfile.write(error_html.encode("utf-8"))

    def log_message(self, format, *args):
        """Suppress default HTTP server logging."""


class SSEAuthenticator:
    """OAuth/SSO authenticator for SSE-based MCP endpoints."""

    def __init__(
        self,
        endpoint: str,
        client_id: str | None = None,
        client_secret: str | None = None,
        redirect_port: int | None = None,
        scope: str = "read:jira-work read:jira-user write:jira-work",
    ):
        self.endpoint = endpoint.rstrip("/")
        self.client_id = client_id or "jirascope-cli"
        self.client_secret = client_secret
        # Store the preferred port, but don't find available port yet
        self.preferred_redirect_port = redirect_port
        self.scope = scope
        self.cache_file = Path.home() / ".jirascope" / "auth_cache.json"

        # Validate endpoint
        if not self.endpoint:
            raise AuthError("Endpoint cannot be empty")

        # Ensure cache directory exists
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

    async def get_auth_tokens(self, force_refresh: bool = False) -> AuthTokens:
        """Get valid authentication tokens, refreshing if necessary."""
        try:
            # Try to load cached tokens first
            if not force_refresh:
                cached_tokens = self._load_cached_tokens()
                if cached_tokens and not cached_tokens.is_expired:
                    logger.debug("Using cached authentication tokens")
                    return cached_tokens

                # Try to refresh if we have a refresh token
                if cached_tokens and cached_tokens.refresh_token:
                    try:
                        refreshed_tokens = await self._refresh_tokens(cached_tokens.refresh_token)
                        self._save_tokens(refreshed_tokens)
                        logger.info("Successfully refreshed authentication tokens")
                        return refreshed_tokens
                    except Exception as e:
                        logger.warning(f"Failed to refresh tokens: {e}, will re-authenticate")

            # Perform full authentication flow
            logger.info("Starting OAuth authentication flow...")
            tokens = await self._perform_oauth_flow()
            self._save_tokens(tokens)
            logger.info("Authentication successful!")
            return tokens

        except Exception as e:
            logger.exception(f"Authentication failed: {e}")
            raise AuthError(f"Authentication failed: {e}") from e

    async def _perform_oauth_flow(self) -> AuthTokens:
        """Perform the complete OAuth flow with browser redirect."""
        # Find an available port for the OAuth callback
        redirect_port = (
            self.preferred_redirect_port
            if self.preferred_redirect_port is not None
            else find_available_port()
        )

        # Try to use the preferred/found port, but if it fails, find another one
        max_retries = 5
        for attempt in range(max_retries):
            try:
                redirect_uri = f"http://localhost:{redirect_port}/callback"
                # Generate PKCE parameters
                code_verifier = (
                    base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8").rstrip("=")
                )
                code_challenge = (
                    base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode("utf-8")).digest())
                    .decode("utf-8")
                    .rstrip("=")
                )

                state = secrets.token_urlsafe(32)

                # Start local callback server
                server = AuthHTTPServer(("localhost", redirect_port), AuthCallbackHandler)
                break  # Successfully created server, exit retry loop
            except OSError as e:
                if e.errno == 48:  # Address already in use
                    logger.debug(f"Port {redirect_port} is in use, trying next available port...")
                    redirect_port = find_available_port(redirect_port + 1)
                    if attempt == max_retries - 1:
                        raise AuthError(
                            f"Could not find an available port after {max_retries} attempts"
                        )
                    continue
                raise  # Re-raise other OSErrors

        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            # Discover OAuth endpoints
            auth_url, token_url = await self._discover_oauth_endpoints()

            # Build authorization URL
            auth_params = {
                "audience": "api.atlassian.com",  # Required for Atlassian OAuth
                "response_type": "code",
                "client_id": self.client_id,
                "redirect_uri": redirect_uri,
                "scope": self.scope,
                "state": state,
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
                "prompt": "consent",  # Force consent to get refresh token
            }

            full_auth_url = f"{auth_url}?" + urllib.parse.urlencode(auth_params)

            # Open browser for authentication
            if not webbrowser.open(full_auth_url):
                logger.warning("Failed to open browser automatically")
                logger.info(f"Please manually open this URL to authenticate: {full_auth_url}")
                # Continue with authentication flow - the user can still complete it manually

            # Wait for callback
            timeout = 300  # 5 minutes
            start_time = time.time()

            while server.auth_code is None and server.auth_error is None:
                if time.time() - start_time > timeout:
                    raise TimeoutError("Authentication timed out after 5 minutes")
                await asyncio.sleep(1)

            if server.auth_error:
                raise AuthError(f"Authentication failed: {server.auth_error}")

            # Validate state parameter
            if server.auth_state != state:
                raise AuthError("Invalid state parameter - possible CSRF attack")

            auth_code = server.auth_code

            # Exchange code for tokens
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                token_data = {
                    "grant_type": "authorization_code",
                    "client_id": self.client_id,
                    "code": auth_code,
                    "redirect_uri": redirect_uri,
                    "code_verifier": code_verifier,
                }

                # Add client secret for confidential clients
                if self.client_secret:
                    token_data["client_secret"] = self.client_secret

                headers = {
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                }

                response = await client.post(token_url, data=token_data, headers=headers)

                if response.status_code != 200:
                    error_text = response.text
                    raise AuthError(f"Token exchange failed: {response.status_code} - {error_text}")

                token_response = response.json()

                if "error" in token_response:
                    raise AuthError(f"Token exchange error: {token_response['error']}")

                expires_at = None
                if "expires_in" in token_response:
                    expires_at = time.time() + token_response["expires_in"]

                return AuthTokens(
                    access_token=token_response["access_token"],
                    refresh_token=token_response.get("refresh_token"),
                    expires_at=expires_at,
                    token_type=token_response.get("token_type", "Bearer"),
                    scope=token_response.get("scope"),
                )

        finally:
            server.shutdown()

    async def _discover_oauth_endpoints(self) -> tuple[str, str]:
        """Discover OAuth endpoints from the SSE endpoint."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            discovery_urls = [
                f"{self.endpoint}/.well-known/openid_configuration",
                f"{self.endpoint}/.well-known/oauth-authorization-server",
                # Try base domain variants
                f"{self.endpoint.split('/v1')[0]}/.well-known/openid_configuration",
                f"{self.endpoint.split('/sse')[0]}/.well-known/openid_configuration",
            ]

            for discovery_url in discovery_urls:
                try:
                    logger.debug(f"Trying discovery URL: {discovery_url}")
                    response = await client.get(discovery_url)
                    if response.status_code == 200:
                        config = response.json()
                        auth_endpoint = config.get("authorization_endpoint")
                        token_endpoint = config.get("token_endpoint")

                        if auth_endpoint and token_endpoint:
                            logger.info(f"Discovered OAuth endpoints via {discovery_url}")
                            return auth_endpoint, token_endpoint
                except Exception as e:
                    logger.debug(f"Discovery failed for {discovery_url}: {e}")

            # Fallback to proper OAuth endpoints
            if "atlassian.com" in self.endpoint:
                # For Atlassian endpoints, use the proper OAuth service
                auth_url = "https://auth.atlassian.com/authorize"
                token_url = "https://auth.atlassian.com/oauth/token"  # nosec B105
            elif "/v1/sse" in self.endpoint:
                # For other MCP endpoints
                base_url = self.endpoint.replace("/sse", "")  # Keep the /v1 part
                auth_url = f"{base_url}/authorize"
                token_url = f"{base_url}/token"
            else:
                # Generic fallback
                base_url = self.endpoint.replace("/sse", "")
                auth_url = f"{base_url}/oauth/authorize"
                token_url = f"{base_url}/oauth/token"

            logger.warning(
                f"Could not discover OAuth endpoints, using fallback: {auth_url}, {token_url}"
            )
            return auth_url, token_url

    async def _refresh_tokens(self, refresh_token: str) -> AuthTokens:
        """Refresh access token using refresh token."""
        _, token_url = await self._discover_oauth_endpoints()

        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            token_data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
            }

            # Add client secret for confidential clients
            if self.client_secret:
                token_data["client_secret"] = self.client_secret

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            }

            response = await client.post(token_url, data=token_data, headers=headers)

            if response.status_code != 200:
                error_text = response.text
                raise AuthError(f"Token refresh failed: {response.status_code} - {error_text}")

            token_response = response.json()

            if "error" in token_response:
                raise AuthError(f"Token refresh error: {token_response['error']}")

            expires_at = None
            if "expires_in" in token_response:
                expires_at = time.time() + token_response["expires_in"]

            return AuthTokens(
                access_token=token_response["access_token"],
                refresh_token=token_response.get("refresh_token", refresh_token),
                expires_at=expires_at,
                token_type=token_response.get("token_type", "Bearer"),
                scope=token_response.get("scope"),
            )

    def _load_cached_tokens(self) -> AuthTokens | None:
        """Load tokens from cache file."""
        try:
            if self.cache_file.exists():
                data = json.loads(self.cache_file.read_text())
                tokens = AuthTokens.from_dict(data)
                logger.debug(f"Loaded cached tokens, expires at: {tokens.expires_at}")
                return tokens
        except Exception as e:
            logger.debug(f"Failed to load cached tokens: {e}")
        return None

    def _save_tokens(self, tokens: AuthTokens):
        """Save tokens to cache file."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(tokens.to_dict(), indent=2))
            # Set file permissions to be readable only by user
            self.cache_file.chmod(0o600)
            logger.debug(f"Saved tokens to cache: {self.cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save tokens to cache: {e}")

    def clear_cache(self):
        """Clear cached authentication tokens."""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                logger.info("Cleared authentication cache")
        except Exception as e:
            logger.warning(f"Failed to clear auth cache: {e}")

    async def validate_token(self, token: str) -> bool:
        """Validate an access token by making a test request."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                headers = {"Authorization": f"Bearer {token}"}
                # Try a simple endpoint that should be available
                test_url = f"{self.endpoint}/myself"  # Common Atlassian endpoint
                response = await client.get(test_url, headers=headers)
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Token validation failed: {e}")
            return False


def find_available_port(start_port: int = 8080, max_port: int = 8200) -> int:
    """Find an available port starting from start_port."""
    # Ensure we don't exceed max_port
    if start_port >= max_port:
        start_port = 8080
        max_port = 8200

    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports found between {start_port} and {max_port}")
