"""
Authentication and authorization for The Noiseless Newspaper API.

Strategy
--------
Production (Supabase Auth)
  - Frontend obtains a JWT from Supabase after sign-in.
  - Every request sends the token as: ``Authorization: Bearer <token>``
  - This module verifies the token against SUPABASE_JWT_SECRET and extracts
    the user's UUID (the ``sub`` claim).

Development / testing
  - If SUPABASE_JWT_SECRET is not set and ENVIRONMENT == "development",
    a ``X-Debug-User-ID`` header is accepted in place of a real JWT.
    This lets you hit the API with curl/Postman without a Supabase project.
  - Example: ``curl -H "X-Debug-User-ID: test-user-1" http://localhost:8000/api/v1/users/me/stats``

Admin routes
  - Protected by an ``X-Admin-API-Key`` header checked against ADMIN_API_KEY.
  - If ADMIN_API_KEY is not set the admin routes return 503.
"""
from typing import Optional

import structlog
from fastapi import Depends, Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import ExpiredSignatureError, JWTError, jwt

from app.config import get_settings

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Security schemes
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)
_admin_api_key_header = APIKeyHeader(name="X-Admin-API-Key", auto_error=False)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _decode_supabase_jwt(token: str, secret: str) -> str:
    """Decode a Supabase-issued JWT and return the subject (user UUID).

    Supabase JWTs are signed with HS256.  The audience claim is typically
    "authenticated" but we skip audience verification so the helper works
    with both Supabase Cloud and self-hosted instances.
    """
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError as exc:
        logger.warning("JWT decode failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: Optional[str] = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing the subject claim.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user_id


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

async def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
    x_debug_user_id: Optional[str] = Header(default=None, alias="X-Debug-User-ID"),
) -> str:
    """Return the authenticated user's ID.

    Raises HTTP 401 if the request is not authenticated.
    """
    settings = get_settings()

    # ------------------------------------------------------------------
    # Development fallback: accept X-Debug-User-ID header when JWT secret
    # is not configured.  Never active in production.
    # ------------------------------------------------------------------
    if settings.environment == "development" and not settings.supabase_jwt_secret:
        if x_debug_user_id:
            logger.debug("Auth: using debug user ID", user_id=x_debug_user_id)
            return x_debug_user_id
        if credentials:
            # Treat the raw token as the user ID (handy for quick local tests)
            logger.debug("Auth: dev mode – treating bearer token as user ID")
            return credentials.credentials
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Not authenticated. "
                "In development mode, pass X-Debug-User-ID header or set SUPABASE_JWT_SECRET."
            ),
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ------------------------------------------------------------------
    # Production: require a valid Supabase JWT
    # ------------------------------------------------------------------
    if not settings.supabase_jwt_secret:
        logger.error("SUPABASE_JWT_SECRET is not configured in non-development environment")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured on this server.",
        )

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return _decode_supabase_jwt(credentials.credentials, settings.supabase_jwt_secret)


async def get_optional_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
    x_debug_user_id: Optional[str] = Header(default=None, alias="X-Debug-User-ID"),
) -> Optional[str]:
    """Return the authenticated user's ID, or ``None`` if unauthenticated.

    Use this for endpoints that work both with and without auth (e.g. public
    article browsing that optionally personalises results).
    """
    try:
        return await get_current_user_id(credentials, x_debug_user_id)
    except HTTPException:
        return None


async def require_admin(
    api_key: Optional[str] = Security(_admin_api_key_header),
) -> None:
    """Guard for admin-only endpoints.

    Verifies the ``X-Admin-API-Key`` header against ADMIN_API_KEY.
    Raises HTTP 403 on mismatch, HTTP 503 if key is not configured.
    """
    settings = get_settings()

    # In development with no key configured, allow access but log a warning
    if settings.environment == "development" and not settings.admin_api_key:
        logger.warning(
            "Admin route accessed without ADMIN_API_KEY set – "
            "this would be rejected in production"
        )
        return

    if not settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin access is not configured on this server.",
        )

    if api_key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin API key.",
        )
