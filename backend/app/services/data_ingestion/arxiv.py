"""
arXiv API integration.

arXiv provides free access to preprints in physics, mathematics,
computer science, and other fields.

API Documentation: https://info.arxiv.org/help/api/basics.html
"""

import asyncio
import re
from datetime import datetime
from typing import Optional
from xml.etree import ElementTree
import logging

import httpx

from app.services.data_ingestion.base import (
    BaseSource,
    RawArticle,
    SourceConfig,
    SourceType,
    TopicDomain,
)
from app.services.data_ingestion.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

# arXiv category to topic domain mapping
ARXIV_CATEGORY_MAP = {
    # AI/ML categories
    "cs.AI": TopicDomain.AI_ML,
    "cs.LG": TopicDomain.AI_ML,
    "cs.CL": TopicDomain.AI_ML,
    "cs.CV": TopicDomain.AI_ML,
    "cs.NE": TopicDomain.AI_ML,
    "stat.ML": TopicDomain.AI_ML,

    # Physics categories
    "physics": TopicDomain.PHYSICS,
    "hep-th": TopicDomain.PHYSICS,
    "hep-ph": TopicDomain.PHYSICS,
    "hep-ex": TopicDomain.PHYSICS,
    "hep-lat": TopicDomain.PHYSICS,
    "cond-mat": TopicDomain.PHYSICS,
    "quant-ph": TopicDomain.PHYSICS,
    "astro-ph": TopicDomain.PHYSICS,
    "gr-qc": TopicDomain.PHYSICS,
    "nucl-th": TopicDomain.PHYSICS,
    "nucl-ex": TopicDomain.PHYSICS,

    # Economics
    "econ": TopicDomain.ECONOMICS,
    "q-fin": TopicDomain.ECONOMICS,

    # Biotech (quantitative biology)
    "q-bio": TopicDomain.BIOTECH,
}

# Topic domain to arXiv categories for querying
DOMAIN_TO_ARXIV = {
    TopicDomain.AI_ML: ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML"],
    TopicDomain.PHYSICS: ["physics", "hep-th", "hep-ph", "cond-mat", "quant-ph", "astro-ph"],
    TopicDomain.ECONOMICS: ["econ", "q-fin"],
    TopicDomain.BIOTECH: ["q-bio"],
}

# XML namespaces
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def create_arxiv_config() -> SourceConfig:
    """Create default arXiv source configuration."""
    return SourceConfig(
        name="arxiv",
        source_type=SourceType.ACADEMIC_API,
        base_url="http://export.arxiv.org/api/query",
        rate_limit_requests=1,
        rate_limit_period=3,
        topics=[TopicDomain.AI_ML, TopicDomain.PHYSICS, TopicDomain.ECONOMICS, TopicDomain.BIOTECH],
        priority=5,  # High authority for academic content
    )


class ArxivSource(BaseSource):
    """
    arXiv API source implementation.

    Fetches preprints from arXiv's Atom feed API.
    """

    def __init__(self, config: Optional[SourceConfig] = None):
        super().__init__(config or create_arxiv_config())
        self.rate_limiter = get_rate_limiter()

    async def fetch_recent(
        self,
        max_results: int = 100,
        since: Optional[datetime] = None,
    ) -> list[RawArticle]:
        """
        Fetch recent articles across all supported categories.

        Args:
            max_results: Maximum articles to fetch
            since: Only articles after this date (note: arXiv API has limited date filtering)

        Returns:
            List of RawArticle objects
        """
        # Build query for all our categories
        all_categories = []
        for cats in DOMAIN_TO_ARXIV.values():
            all_categories.extend(cats)

        # Remove duplicates while preserving order
        seen = set()
        categories = [c for c in all_categories if not (c in seen or seen.add(c))]

        return await self._fetch_by_categories(categories, max_results)

    async def fetch_by_topic(
        self,
        topic: TopicDomain,
        max_results: int = 50,
    ) -> list[RawArticle]:
        """
        Fetch articles for a specific topic domain.

        Args:
            topic: Topic domain to fetch
            max_results: Maximum articles

        Returns:
            List of RawArticle objects
        """
        categories = DOMAIN_TO_ARXIV.get(topic, [])
        if not categories:
            logger.warning(f"No arXiv categories mapped for topic: {topic}")
            return []

        return await self._fetch_by_categories(categories, max_results)

    async def _fetch_by_categories(
        self,
        categories: list[str],
        max_results: int,
    ) -> list[RawArticle]:
        """Fetch articles matching any of the given categories."""
        # Build OR query for categories
        cat_query = " OR ".join(f"cat:{cat}" for cat in categories)

        params = {
            "search_query": cat_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        # Respect rate limits
        await self.rate_limiter.wait_if_needed("arxiv")

        try:
            # Create client without proxy to avoid socksio issues
            async with httpx.AsyncClient(timeout=30.0, proxy=None) as client:
                response = await client.get(self.config.base_url, params=params)
                response.raise_for_status()

            return self._parse_atom_feed(response.text)

        except httpx.HTTPError as e:
            logger.error(f"arXiv API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch from arXiv: {e}")
            return []

    def _parse_atom_feed(self, xml_content: str) -> list[RawArticle]:
        """Parse arXiv Atom feed into RawArticle objects."""
        articles = []

        try:
            root = ElementTree.fromstring(xml_content)
        except ElementTree.ParseError as e:
            logger.error(f"Failed to parse arXiv XML: {e}")
            return []

        for entry in root.findall(f"{ATOM_NS}entry"):
            try:
                article = self._parse_entry(entry)
                if article:
                    articles.append(article)
            except Exception as e:
                logger.warning(f"Failed to parse arXiv entry: {e}")
                continue

        return articles

    def _parse_entry(self, entry: ElementTree.Element) -> Optional[RawArticle]:
        """Parse a single Atom entry into a RawArticle."""
        # Extract arXiv ID from the entry ID
        entry_id = entry.findtext(f"{ATOM_NS}id", "")
        arxiv_id = self._extract_arxiv_id(entry_id)
        if not arxiv_id:
            return None

        # Title (clean up whitespace)
        title = entry.findtext(f"{ATOM_NS}title", "")
        title = " ".join(title.split())

        # Abstract/summary
        abstract = entry.findtext(f"{ATOM_NS}summary", "")
        abstract = " ".join(abstract.split())

        # Authors
        authors = []
        for author in entry.findall(f"{ATOM_NS}author"):
            name = author.findtext(f"{ATOM_NS}name")
            if name:
                authors.append(name)

        # Publication date
        published_str = entry.findtext(f"{ATOM_NS}published", "")
        published_at = self._parse_date(published_str)

        # Categories
        categories = []
        for category in entry.findall(f"{ATOM_NS}category"):
            term = category.get("term")
            if term:
                categories.append(term)

        # Map to our topic domains
        matched_domains = self.map_to_domain(categories)

        # PDF link
        pdf_link = None
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("title") == "pdf":
                pdf_link = link.get("href")
                break

        # Abstract page URL
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"

        # DOI if available
        doi = entry.findtext(f"{ARXIV_NS}doi")

        return RawArticle(
            external_id=arxiv_id,
            source_name="arxiv",
            title=title,
            url=abs_url,
            abstract=abstract,
            authors=authors,
            published_at=published_at,
            topics=categories,
            matched_domains=matched_domains,
            doi=doi,
            arxiv_id=arxiv_id,
            peer_reviewed=False,  # arXiv is preprints
            source_authority_score=0.7,  # High but not peer-reviewed
        )

    def _extract_arxiv_id(self, entry_id: str) -> Optional[str]:
        """Extract arXiv ID from entry ID URL."""
        # Entry ID format: http://arxiv.org/abs/2401.12345v1
        match = re.search(r"arxiv\.org/abs/(.+?)(?:v\d+)?$", entry_id)
        if match:
            return match.group(1)

        # Try alternate format
        match = re.search(r"(\d{4}\.\d{4,5})", entry_id)
        if match:
            return match.group(1)

        return None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse arXiv date string."""
        if not date_str:
            return None

        try:
            # Format: 2024-01-15T12:00:00Z
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            try:
                # Try without timezone
                return datetime.fromisoformat(date_str[:19])
            except ValueError:
                return None

    def map_to_domain(self, source_categories: list[str]) -> list[TopicDomain]:
        """Map arXiv categories to our topic domains."""
        domains = set()

        for cat in source_categories:
            # Check exact match
            if cat in ARXIV_CATEGORY_MAP:
                domains.add(ARXIV_CATEGORY_MAP[cat])
                continue

            # Check prefix match (e.g., "cond-mat.str-el" -> "cond-mat")
            prefix = cat.split(".")[0]
            if prefix in ARXIV_CATEGORY_MAP:
                domains.add(ARXIV_CATEGORY_MAP[prefix])

        return list(domains)
