"""
Base interface for article sources.
All sources (arXiv, Semantic Scholar, NewsAPI, etc.) implement this interface.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncIterator, Optional

from app.models.domain import Article, ArticleSource, TopicPath


class ArticleSourceAdapter(ABC):
    """Abstract base class for article source adapters."""

    @property
    @abstractmethod
    def source_type(self) -> ArticleSource:
        """Return the source type identifier."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the source."""
        pass

    @abstractmethod
    async def search(
        self,
        topic: TopicPath,
        keywords: list[str],
        since: Optional[datetime] = None,
        max_results: int = 50,
    ) -> AsyncIterator[Article]:
        """
        Search for articles matching the given topic and keywords.

        Args:
            topic: The topic path to search for
            keywords: List of keywords to search
            since: Only return articles published after this date
            max_results: Maximum number of results to return

        Yields:
            Article objects matching the search criteria
        """
        pass

    @abstractmethod
    async def get_article(self, article_id: str) -> Optional[Article]:
        """
        Fetch a specific article by its ID.

        Args:
            article_id: The article's unique identifier

        Returns:
            The article if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_citations(self, article_id: str) -> tuple[list[str], list[str]]:
        """
        Get citation information for an article.

        Args:
            article_id: The article's unique identifier

        Returns:
            Tuple of (citing_paper_ids, cited_paper_ids)
        """
        pass

    async def health_check(self) -> bool:
        """Check if the source is available."""
        return True
