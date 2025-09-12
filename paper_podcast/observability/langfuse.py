from __future__ import annotations

import os
from typing import Optional


def _langfuse_env_present() -> bool:
    """Return True if Langfuse config appears present in env.

    Uses common envs: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST.
    Only the keys are required; host is optional (defaults to cloud/local config).
    """
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_openai_client(api_key: Optional[str] = None):
    """Return an OpenAI client, instrumented with Langfuse if available.

    - If the `langfuse` package is installed and Langfuse env vars are present, 
      returns the Langfuse-instrumented OpenAI client from `langfuse.openai`.
    - Otherwise falls back to `openai.OpenAI`.
    """
    if _langfuse_env_present():
        try:
            # Use the Langfuse OpenAI integration pattern
            from langfuse.openai import openai  # type: ignore
            
            # Return the instrumented openai module's client
            return openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
        except Exception:
            # If Langfuse integration is not available, silently fall back
            pass

    # Fallback to vanilla OpenAI client
    from openai import OpenAI  # lazy import to avoid hard dependency at import time

    return OpenAI(api_key=api_key)

