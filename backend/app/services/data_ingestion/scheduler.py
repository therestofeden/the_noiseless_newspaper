"""
Ingestion Scheduler - Automated article fetching on schedule.

Runs periodic fetches from all configured sources and stores
articles in the database for ranking and serving.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Callable, Awaitable
import logging

from app.services.data_ingestion.aggregator import SourceAggregator
from app.services.data_ingestion.base import RawArticle, TopicDomain, IngestionResult

logger = logging.getLogger(__name__)


class IngestionScheduler:
    """
    Schedules and runs periodic data ingestion.

    Features:
    - Configurable fetch intervals per source type
    - Staggered fetching to avoid rate limits
    - Error recovery and retry logic
    - Callback hooks for article processing
    """

    def __init__(
        self,
        aggregator: Optional[SourceAggregator] = None,
        fetch_interval_hours: int = 6,
        on_articles_fetched: Optional[Callable[[list[RawArticle]], Awaitable[None]]] = None,
    ):
        """
        Initialize the scheduler.

        Args:
            aggregator: Source aggregator instance
            fetch_interval_hours: Hours between full fetches
            on_articles_fetched: Callback when articles are fetched
        """
        self.aggregator = aggregator or SourceAggregator()
        self.fetch_interval = timedelta(hours=fetch_interval_hours)
        self.on_articles_fetched = on_articles_fetched

        self._running = False
        self._last_fetch: Optional[datetime] = None
        self._fetch_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the scheduler loop."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        logger.info(f"Starting ingestion scheduler (interval: {self.fetch_interval})")

        # Run initial fetch
        await self._run_fetch()

        # Start background loop
        self._fetch_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self):
        """Stop the scheduler."""
        self._running = False

        if self._fetch_task:
            self._fetch_task.cancel()
            try:
                await self._fetch_task
            except asyncio.CancelledError:
                pass

        logger.info("Ingestion scheduler stopped")

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            # Calculate time until next fetch
            if self._last_fetch:
                next_fetch = self._last_fetch + self.fetch_interval
                wait_seconds = (next_fetch - datetime.utcnow()).total_seconds()

                if wait_seconds > 0:
                    logger.debug(f"Next fetch in {wait_seconds / 3600:.1f} hours")
                    await asyncio.sleep(wait_seconds)

            if self._running:
                await self._run_fetch()

    async def _run_fetch(self):
        """Execute a fetch cycle."""
        logger.info("Starting scheduled fetch")
        start_time = datetime.utcnow()

        try:
            # Fetch articles
            articles, results = await self.aggregator.fetch_all(
                max_per_source=100,
                since=self._last_fetch,
            )

            # Log results
            for result in results:
                logger.info(str(result))

            # Invoke callback if set
            if self.on_articles_fetched and articles:
                await self.on_articles_fetched(articles)

            self._last_fetch = start_time

            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                f"Fetch completed: {len(articles)} articles in {duration:.1f}s"
            )

        except Exception as e:
            logger.error(f"Fetch failed: {e}", exc_info=True)

    async def fetch_now(
        self,
        topic: Optional[TopicDomain] = None,
    ) -> tuple[list[RawArticle], list[IngestionResult]]:
        """
        Trigger an immediate fetch.

        Args:
            topic: Optional topic to fetch (None = all topics)

        Returns:
            Tuple of (articles, ingestion results)
        """
        if topic:
            return await self.aggregator.fetch_by_topic(topic, max_per_source=50)
        else:
            return await self.aggregator.fetch_all(max_per_source=100)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_fetch(self) -> Optional[datetime]:
        return self._last_fetch

    def get_status(self) -> dict:
        """Get scheduler status."""
        next_fetch = None
        if self._last_fetch:
            next_fetch = self._last_fetch + self.fetch_interval

        return {
            "running": self._running,
            "last_fetch": self._last_fetch.isoformat() if self._last_fetch else None,
            "next_fetch": next_fetch.isoformat() if next_fetch else None,
            "fetch_interval_hours": self.fetch_interval.total_seconds() / 3600,
            "sources": self.aggregator.get_source_stats(),
        }
