"""
Source Aggregator - Orchestrates data collection from all sources.

This module coordinates fetching from multiple sources, handles
deduplication, and provides a unified interface for data ingestion.
"""

import asyncio
import hashlib
import re
from collections import defaultdict
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Optional
from urllib.parse import urlparse, parse_qs
import logging

from app.services.data_ingestion.base import (
    BaseSource,
    RawArticle,
    IngestionResult,
    TopicDomain,
)
from app.services.data_ingestion.arxiv import ArxivSource
from app.services.data_ingestion.pubmed import PubMedSource
from app.services.data_ingestion.rss import RSSSource
from app.services.data_ingestion.semantic_scholar import SemanticScholarSource

logger = logging.getLogger(__name__)


class SourceAggregator:
    """
    Aggregates articles from multiple sources with deduplication.

    Features:
    - Concurrent fetching from all sources
    - URL and content-based deduplication
    - DOI/arXiv ID matching for academic papers
    - Priority-based source selection for duplicates
    """

    def __init__(
        self,
        arxiv_enabled: bool = True,
        pubmed_enabled: bool = True,
        rss_enabled: bool = True,
        semantic_scholar_enabled: bool = True,
        semantic_scholar_api_key: Optional[str] = None,
        pubmed_api_key: Optional[str] = None,
    ):
        """
        Initialize the aggregator with selected sources.

        Args:
            arxiv_enabled: Enable arXiv source
            pubmed_enabled: Enable PubMed source
            rss_enabled: Enable RSS feeds
            semantic_scholar_enabled: Enable Semantic Scholar
            semantic_scholar_api_key: API key for higher rate limits
            pubmed_api_key: API key for higher rate limits
        """
        self.sources: list[BaseSource] = []

        if arxiv_enabled:
            self.sources.append(ArxivSource())

        if pubmed_enabled:
            from app.services.data_ingestion.pubmed import create_pubmed_config
            config = create_pubmed_config(pubmed_api_key)
            self.sources.append(PubMedSource(config))

        if rss_enabled:
            self.sources.append(RSSSource())

        if semantic_scholar_enabled:
            from app.services.data_ingestion.semantic_scholar import create_semantic_scholar_config
            config = create_semantic_scholar_config(semantic_scholar_api_key)
            self.sources.append(SemanticScholarSource(config))

        logger.info(f"Initialized aggregator with {len(self.sources)} sources")

    async def fetch_all(
        self,
        max_per_source: int = 100,
        since: Optional[datetime] = None,
    ) -> tuple[list[RawArticle], list[IngestionResult]]:
        """
        Fetch articles from all sources concurrently.

        Args:
            max_per_source: Maximum articles per source
            since: Only fetch articles after this date

        Returns:
            Tuple of (deduplicated articles, ingestion results per source)
        """
        if since is None:
            since = datetime.utcnow() - timedelta(days=7)

        # Fetch from all sources concurrently
        tasks = []
        for source in self.sources:
            tasks.append(self._fetch_from_source(source, max_per_source, since))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect articles and results
        all_articles = []
        ingestion_results = []

        for source, result in zip(self.sources, results):
            if isinstance(result, Exception):
                logger.error(f"Source {source.name} failed: {result}")
                ingestion_results.append(IngestionResult(
                    source_name=source.name,
                    articles_fetched=0,
                    articles_new=0,
                    articles_updated=0,
                    articles_skipped=0,
                    errors=[str(result)],
                ))
            else:
                articles, ingestion_result = result
                all_articles.extend(articles)
                ingestion_results.append(ingestion_result)

        # Deduplicate
        deduplicated = self._deduplicate(all_articles)

        logger.info(
            f"Aggregated {len(all_articles)} articles, "
            f"{len(deduplicated)} after deduplication"
        )

        return deduplicated, ingestion_results

    async def fetch_by_topic(
        self,
        topic: TopicDomain,
        max_per_source: int = 50,
    ) -> tuple[list[RawArticle], list[IngestionResult]]:
        """
        Fetch articles for a specific topic from all sources.

        Args:
            topic: Topic domain to fetch
            max_per_source: Maximum articles per source

        Returns:
            Tuple of (deduplicated articles, ingestion results)
        """
        tasks = []
        for source in self.sources:
            tasks.append(self._fetch_topic_from_source(source, topic, max_per_source))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles = []
        ingestion_results = []

        for source, result in zip(self.sources, results):
            if isinstance(result, Exception):
                logger.error(f"Source {source.name} failed for {topic}: {result}")
                ingestion_results.append(IngestionResult(
                    source_name=source.name,
                    articles_fetched=0,
                    articles_new=0,
                    articles_updated=0,
                    articles_skipped=0,
                    errors=[str(result)],
                ))
            else:
                articles, ingestion_result = result
                all_articles.extend(articles)
                ingestion_results.append(ingestion_result)

        deduplicated = self._deduplicate(all_articles)

        return deduplicated, ingestion_results

    async def _fetch_from_source(
        self,
        source: BaseSource,
        max_results: int,
        since: datetime,
    ) -> tuple[list[RawArticle], IngestionResult]:
        """Fetch from a single source with timing."""
        start_time = datetime.utcnow()

        try:
            articles = await source.fetch_recent(max_results, since)

            duration = (datetime.utcnow() - start_time).total_seconds()

            return articles, IngestionResult(
                source_name=source.name,
                articles_fetched=len(articles),
                articles_new=len(articles),  # Will be updated after dedup
                articles_updated=0,
                articles_skipped=0,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            raise

    async def _fetch_topic_from_source(
        self,
        source: BaseSource,
        topic: TopicDomain,
        max_results: int,
    ) -> tuple[list[RawArticle], IngestionResult]:
        """Fetch topic-specific articles from a source."""
        start_time = datetime.utcnow()

        try:
            articles = await source.fetch_by_topic(topic, max_results)

            duration = (datetime.utcnow() - start_time).total_seconds()

            return articles, IngestionResult(
                source_name=source.name,
                articles_fetched=len(articles),
                articles_new=len(articles),
                articles_updated=0,
                articles_skipped=0,
                duration_seconds=duration,
            )

        except Exception as e:
            raise

    def _deduplicate(self, articles: list[RawArticle]) -> list[RawArticle]:
        """
        Deduplicate articles using multiple strategies.

        Strategies (in order):
        1. DOI matching (exact)
        2. arXiv ID matching (exact)
        3. URL normalization (exact)
        4. Title similarity (fuzzy, >0.9 threshold)

        When duplicates found, keep the one from highest-priority source.
        """
        # Index articles by various keys
        by_doi: dict[str, list[RawArticle]] = defaultdict(list)
        by_arxiv: dict[str, list[RawArticle]] = defaultdict(list)
        by_url: dict[str, list[RawArticle]] = defaultdict(list)

        for article in articles:
            if article.doi:
                by_doi[article.doi.lower()].append(article)
            if article.arxiv_id:
                by_arxiv[article.arxiv_id.lower()].append(article)

            normalized_url = self._normalize_url(article.url)
            by_url[normalized_url].append(article)

        # Track which articles to keep
        seen_ids = set()
        unique_articles = []

        # First pass: exact matches
        for article in articles:
            article_key = (article.source_name, article.external_id)
            if article_key in seen_ids:
                continue

            duplicates = []

            # Check DOI duplicates
            if article.doi:
                duplicates.extend(by_doi.get(article.doi.lower(), []))

            # Check arXiv duplicates
            if article.arxiv_id:
                duplicates.extend(by_arxiv.get(article.arxiv_id.lower(), []))

            # Check URL duplicates
            normalized_url = self._normalize_url(article.url)
            duplicates.extend(by_url.get(normalized_url, []))

            # Remove self and deduplicate
            duplicates = [
                d for d in duplicates
                if (d.source_name, d.external_id) != article_key
            ]

            # Mark all duplicates as seen
            for dup in duplicates:
                seen_ids.add((dup.source_name, dup.external_id))

            # Keep the best version (highest authority score)
            all_versions = [article] + duplicates
            best = max(all_versions, key=lambda a: a.source_authority_score)

            seen_ids.add((best.source_name, best.external_id))
            unique_articles.append(best)

        # Second pass: fuzzy title matching for remaining articles
        final_articles = []
        title_index: list[tuple[str, RawArticle]] = []

        for article in unique_articles:
            normalized_title = self._normalize_title(article.title)

            # Check for similar titles
            is_duplicate = False
            for existing_title, existing_article in title_index:
                similarity = SequenceMatcher(
                    None, normalized_title, existing_title
                ).ratio()

                if similarity > 0.9:
                    # Keep higher authority version
                    if article.source_authority_score > existing_article.source_authority_score:
                        # Replace existing with better version
                        final_articles.remove(existing_article)
                        final_articles.append(article)
                        title_index.remove((existing_title, existing_article))
                        title_index.append((normalized_title, article))
                    is_duplicate = True
                    break

            if not is_duplicate:
                final_articles.append(article)
                title_index.append((normalized_title, article))

        return final_articles

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        parsed = urlparse(url)

        # Remove common tracking parameters
        tracking_params = {
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_content',
            'ref', 'source', 'via', 'fbclid', 'gclid',
        }

        params = parse_qs(parsed.query)
        clean_params = {
            k: v for k, v in params.items()
            if k.lower() not in tracking_params
        }

        # Rebuild URL without tracking
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if clean_params:
            clean_url += "?" + "&".join(
                f"{k}={v[0]}" for k, v in sorted(clean_params.items())
            )

        return clean_url.lower()

    def _normalize_title(self, title: str) -> str:
        """Normalize title for fuzzy matching."""
        # Lowercase
        title = title.lower()

        # Remove punctuation
        title = re.sub(r'[^\w\s]', ' ', title)

        # Normalize whitespace
        title = ' '.join(title.split())

        return title

    async def health_check(self) -> dict[str, bool]:
        """Check health of all sources."""
        results = {}

        for source in self.sources:
            try:
                is_healthy = await source.health_check()
                results[source.name] = is_healthy
            except Exception as e:
                logger.error(f"Health check failed for {source.name}: {e}")
                results[source.name] = False

        return results

    def get_source_stats(self) -> dict:
        """Get statistics about configured sources."""
        return {
            "total_sources": len(self.sources),
            "sources": [
                {
                    "name": s.name,
                    "type": s.config.source_type.value,
                    "topics": [t.value for t in s.config.topics],
                    "priority": s.config.priority,
                }
                for s in self.sources
            ],
        }
