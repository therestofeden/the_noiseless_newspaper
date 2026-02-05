"""
Data Ingestion Services for The Noiseless Newspaper.

This module provides connectors to fetch articles from various sources:
- Academic APIs (arXiv, PubMed, Semantic Scholar)
- RSS Feed aggregation
- Rate limiting and caching
- Content deduplication
"""

from app.services.data_ingestion.base import (
    BaseSource,
    SourceConfig,
    RawArticle,
    IngestionResult,
)
from app.services.data_ingestion.rate_limiter import RateLimiter
from app.services.data_ingestion.arxiv import ArxivSource
from app.services.data_ingestion.pubmed import PubMedSource
from app.services.data_ingestion.rss import RSSSource
from app.services.data_ingestion.semantic_scholar import SemanticScholarSource
from app.services.data_ingestion.aggregator import SourceAggregator

__all__ = [
    "BaseSource",
    "SourceConfig",
    "RawArticle",
    "IngestionResult",
    "RateLimiter",
    "ArxivSource",
    "PubMedSource",
    "RSSSource",
    "SemanticScholarSource",
    "SourceAggregator",
]
