"""
Base classes and data models for data ingestion.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class SourceType(str, Enum):
    """Type of content source."""
    ACADEMIC_API = "academic_api"
    RSS_FEED = "rss_feed"
    NEWS_API = "news_api"
    THINK_TANK = "think_tank"
    GOVERNMENT = "government"


class TopicDomain(str, Enum):
    """Top-level topic domains."""
    AI_ML = "ai-ml"
    PHYSICS = "physics"
    ECONOMICS = "economics"
    BIOTECH = "biotech"
    POLITICS = "politics"


@dataclass
class SourceConfig:
    """Configuration for a data source."""
    name: str
    source_type: SourceType
    base_url: str
    api_key: Optional[str] = None
    rate_limit_requests: int = 60
    rate_limit_period: int = 60  # seconds
    topics: list[TopicDomain] = field(default_factory=list)
    enabled: bool = True
    priority: int = 1  # Higher = more authoritative


@dataclass
class RawArticle:
    """
    Raw article data from a source before processing.

    This is the intermediate format between source-specific data
    and our normalized Article model.
    """
    # Required fields
    external_id: str  # Source-specific ID (DOI, arXiv ID, URL hash)
    source_name: str
    title: str
    url: str

    # Content
    abstract: Optional[str] = None
    content: Optional[str] = None

    # Metadata
    authors: list[str] = field(default_factory=list)
    published_at: Optional[datetime] = None
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    # Classification
    topics: list[str] = field(default_factory=list)  # Source's own categories
    matched_domains: list[TopicDomain] = field(default_factory=list)

    # Academic metadata
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None  # PubMed ID
    citations: list[str] = field(default_factory=list)  # External IDs of cited works
    citation_count: int = 0

    # Quality signals
    peer_reviewed: bool = False
    source_authority_score: float = 0.5

    def __hash__(self):
        return hash((self.source_name, self.external_id))

    def __eq__(self, other):
        if not isinstance(other, RawArticle):
            return False
        return self.source_name == other.source_name and self.external_id == other.external_id


@dataclass
class IngestionResult:
    """Result of an ingestion operation."""
    source_name: str
    articles_fetched: int
    articles_new: int
    articles_updated: int
    articles_skipped: int
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return (
            f"{status} {self.source_name}: "
            f"fetched={self.articles_fetched}, new={self.articles_new}, "
            f"updated={self.articles_updated}, skipped={self.articles_skipped}, "
            f"errors={len(self.errors)}, time={self.duration_seconds:.1f}s"
        )


class BaseSource(ABC):
    """
    Abstract base class for data sources.

    Each source implementation handles:
    - Fetching articles from its specific API/feed
    - Parsing source-specific data format
    - Mapping to normalized RawArticle format
    - Rate limiting compliance
    """

    def __init__(self, config: SourceConfig):
        self.config = config
        self.name = config.name

    @abstractmethod
    async def fetch_recent(
        self,
        max_results: int = 100,
        since: Optional[datetime] = None,
    ) -> list[RawArticle]:
        """
        Fetch recent articles from this source.

        Args:
            max_results: Maximum number of articles to fetch
            since: Only fetch articles published after this date

        Returns:
            List of RawArticle objects
        """
        pass

    @abstractmethod
    async def fetch_by_topic(
        self,
        topic: TopicDomain,
        max_results: int = 50,
    ) -> list[RawArticle]:
        """
        Fetch articles for a specific topic domain.

        Args:
            topic: The topic domain to fetch
            max_results: Maximum number of articles

        Returns:
            List of RawArticle objects matching the topic
        """
        pass

    async def health_check(self) -> bool:
        """Check if the source is accessible."""
        try:
            articles = await self.fetch_recent(max_results=1)
            return len(articles) > 0
        except Exception:
            return False

    def map_to_domain(self, source_categories: list[str]) -> list[TopicDomain]:
        """
        Map source-specific categories to our topic domains.

        Override in subclasses for source-specific mapping logic.
        """
        return []
