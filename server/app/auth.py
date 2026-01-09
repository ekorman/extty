"""API key authentication."""

import os
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)


def get_api_key() -> str | None:
    """Get the configured API key from environment."""
    return os.environ.get("EXTTY_API_KEY")


def verify_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> None:
    """
    Verify API key for write operations.

    If EXTTY_API_KEY is not set, authentication is disabled.
    If set, requires a valid Bearer token.
    """
    api_key = get_api_key()

    if api_key is None:
        return

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.credentials != api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
