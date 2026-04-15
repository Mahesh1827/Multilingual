"""
Agent Error Handling Middleware
Provides @safe_agent_call decorator for standardized error handling,
retry logic, and circuit-breaker pattern across all agent nodes.
"""

import logging
import time
import functools
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Structured Error
# ─────────────────────────────────────────────

@dataclass
class AgentError:
    agent_name: str
    error_type: str
    message: str
    timestamp: float = field(default_factory=time.time)
    is_retryable: bool = False

    def __str__(self):
        return f"[{self.agent_name}] {self.error_type}: {self.message}"


# ─────────────────────────────────────────────
# Circuit Breaker
# ─────────────────────────────────────────────

class CircuitBreaker:
    """
    If an agent fails `threshold` times consecutively,
    the breaker opens and the agent is skipped for `cooldown` seconds.
    """
    def __init__(self, threshold: int = 3, cooldown: float = 60.0):
        self.threshold = threshold
        self.cooldown = cooldown
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}

    def record_success(self, agent_name: str):
        self._failures[agent_name] = 0

    def record_failure(self, agent_name: str):
        self._failures[agent_name] = self._failures.get(agent_name, 0) + 1
        if self._failures[agent_name] >= self.threshold:
            self._open_until[agent_name] = time.time() + self.cooldown
            logger.warning(
                f"🔴 [Circuit Breaker] {agent_name} tripped! "
                f"Skipping for {self.cooldown}s after {self.threshold} consecutive failures."
            )

    def is_open(self, agent_name: str) -> bool:
        deadline = self._open_until.get(agent_name)
        if deadline is None:
            return False
        if time.time() >= deadline:
            # Cooldown expired — reset
            self._open_until.pop(agent_name, None)
            self._failures[agent_name] = 0
            logger.info(f"🟢 [Circuit Breaker] {agent_name} cooldown expired. Re-enabling.")
            return False
        return True


# Global circuit breaker instance
_breaker = CircuitBreaker(threshold=3, cooldown=60.0)


# ─────────────────────────────────────────────
# Retryable exception detection
# ─────────────────────────────────────────────

_RETRYABLE_ERRORS = (
    TimeoutError,
    ConnectionError,
    OSError,
)

def _is_retryable(exc: Exception) -> bool:
    """Check if the exception is transient and worth retrying."""
    if isinstance(exc, _RETRYABLE_ERRORS):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in ["timeout", "connection", "unavailable", "502", "503"])


# ─────────────────────────────────────────────
# @safe_agent_call decorator
# ─────────────────────────────────────────────

def safe_agent_call(
    agent_name: str,
    fallback_state: Optional[dict] = None,
    max_retries: int = 1,
    retry_delay: float = 2.0,
):
    """
    Decorator that wraps a LangGraph node function with:
      - Circuit breaker check
      - Exception catching + structured logging
      - Retry for transient errors
      - Graceful fallback state on failure

    Usage:
        @safe_agent_call("rag_agent", fallback_state={"needs_web_fallback": True})
        def _rag_node(state):
            ...
    """
    if fallback_state is None:
        fallback_state = {}

    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(state: dict) -> dict:
            # Circuit breaker: skip if tripped
            if _breaker.is_open(agent_name):
                logger.warning(f"⚡ [Error Handler] {agent_name} circuit open — skipping.")
                return {
                    **fallback_state,
                    "error": f"{agent_name} temporarily disabled (circuit breaker).",
                }

            last_exc = None
            attempts = 1 + max_retries

            for attempt in range(1, attempts + 1):
                try:
                    result = fn(state)
                    _breaker.record_success(agent_name)
                    return result

                except Exception as exc:
                    last_exc = exc
                    err = AgentError(
                        agent_name=agent_name,
                        error_type=type(exc).__name__,
                        message=str(exc),
                        is_retryable=_is_retryable(exc),
                    )
                    logger.error(f"❌ [Error Handler] {err} (attempt {attempt}/{attempts})")

                    if _is_retryable(exc) and attempt < attempts:
                        logger.info(f"🔄 [Error Handler] Retrying {agent_name} in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        break

            # All attempts exhausted
            _breaker.record_failure(agent_name)
            logger.error(
                f"💀 [Error Handler] {agent_name} failed after {attempts} attempt(s). "
                f"Returning fallback state."
            )
            return {
                **fallback_state,
                "error": f"{agent_name} failed: {last_exc}",
            }

        return wrapper
    return decorator
