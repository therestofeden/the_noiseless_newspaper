"""
Rate limiting for API requests.

Ensures we respect rate limits across all sources.
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter with per-source tracking.

    Features:
    - Per-source rate limits
    - Async-safe with locks
    - Automatic request spacing
    - Burst handling
    """

    # Default limits per source (requests, period_seconds)
    DEFAULT_LIMITS = {
        "arxiv": (1, 3),          # 1 request per 3 seconds
        "semantic_scholar": (100, 300),  # 100 per 5 minutes
        "pubmed": (3, 1),         # 3 per second (without API key)
        "fred": (120, 60),        # 120 per minute
        "congress": (5000, 86400),  # 5000 per day
        "default": (60, 60),      # 60 per minute default
    }

    def __init__(self):
        self._request_times: dict[str, list[datetime]] = defaultdict(list)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._custom_limits: dict[str, tuple[int, int]] = {}

    def set_limit(self, source: str, requests: int, period_seconds: int):
        """Set custom rate limit for a source."""
        self._custom_limits[source] = (requests, period_seconds)

    def _get_limit(self, source: str) -> tuple[int, int]:
        """Get rate limit for a source."""
        if source in self._custom_limits:
            return self._custom_limits[source]
        return self.DEFAULT_LIMITS.get(source, self.DEFAULT_LIMITS["default"])

    async def acquire(self, source: str, timeout: Optional[float] = 30.0) -> bool:
        """
        Acquire permission to make a request.

        Args:
            source: Source name for rate limiting
            timeout: Max seconds to wait (None = wait forever)

        Returns:
            True if acquired, False if timed out
        """
        start_time = datetime.now()
        max_requests, period_seconds = self._get_limit(source)

        async with self._locks[source]:
            while True:
                now = datetime.now()
                cutoff = now - timedelta(seconds=period_seconds)

                # Clean old request times
                self._request_times[source] = [
                    t for t in self._request_times[source] if t > cutoff
                ]

                current_count = len(self._request_times[source])

                if current_count < max_requests:
                    # Can make request
                    self._request_times[source].append(now)
                    return True

                # Calculate wait time
                oldest = min(self._request_times[source])
                wait_until = oldest + timedelta(seconds=period_seconds)
                wait_seconds = (wait_until - now).total_seconds()

                # Check timeout
                if timeout is not None:
                    elapsed = (now - start_time).total_seconds()
                    if elapsed + wait_seconds > timeout:
                        logger.warning(
                            f"Rate limit timeout for {source}: "
                            f"would need to wait {wait_seconds:.1f}s"
                        )
                        return False

                logger.debug(
                    f"Rate limited for {source}, waiting {wait_seconds:.1f}s"
                )
                await asyncio.sleep(min(wait_seconds + 0.1, 1.0))

    async def wait_if_needed(self, source: str):
        """
        Wait until we can make a request (no timeout).

        Simpler API when you always want to wait.
        """
        await self.acquire(source, timeout=None)

    def get_status(self, source: str) -> dict:
        """Get current rate limit status for a source."""
        max_requests, period_seconds = self._get_limit(source)
        now = datetime.now()
        cutoff = now - timedelta(seconds=period_seconds)

        recent_requests = [
            t for t in self._request_times[source] if t > cutoff
        ]

        return {
            "source": source,
            "max_requests": max_requests,
            "period_seconds": period_seconds,
            "current_requests": len(recent_requests),
            "available": max_requests - len(recent_requests),
        }

    def get_all_status(self) -> list[dict]:
        """Get status for all tracked sources."""
        sources = set(self._request_times.keys()) | set(self._custom_limits.keys())
        return [self.get_status(s) for s in sorted(sources)]


# Global rate limiter instance
_global_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = RateLimiter()
    return _global_limiter
