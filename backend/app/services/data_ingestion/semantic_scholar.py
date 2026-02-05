"""
Semantic Scholar API integration.

Semantic Scholar provides access to academic papers with rich metadata
including citation graphs, author information, and paper embeddings.

API Documentation: https://api.semanticscholar.org/api-docs/
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
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

# Search queries for each topic domain
DOMAIN_QUERIES = {
    TopicDomain.AI_ML: [
        "machine learning",
        "deep learning",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
    ],
    TopicDomain.PHYSICS: [
        "quantum physics",
        "particle physics",
        "condensed matter",
        "astrophysics",
        "theoretical physics",
    ],
    TopicDomain.ECONOMICS: [
        "macroeconomics",
        "microeconomics",
        "econometrics",
        "financial economics",
        "behavioral economics",
    ],
    TopicDomain.BIOTECH: [
        "biotechnology",
        "genomics",
        "CRISPR gene editing",
        "synthetic biology",
        "bioinformatics",
    ],
    TopicDomain.POLITICS: [
        "political science",
        "public policy",
        "international relations",
        "political economy",
    ],
}

# Fields to request from the API
PAPER_FIELDS = [
    "paperId",
    "externalIds",
    "title",
    "abstract",
    "year",
    "publicationDate",
    "authors",
    "citationCount",
    "influentialCitationCount",
    "fieldsOfStudy",
    "s2FieldsOfStudy",
    "publicationTypes",
    "journal",
    "url",
    "openAccessPdf",
]


def create_semantic_scholar_config(api_key: Optional[str] = None) -> SourceConfig:
    """Create default Semantic Scholar source configuration."""
    return SourceConfig(
        name="semantic_scholar",
        source_type=SourceType.ACADEMIC_API,
        base_url="https://api.semanticscholar.org/graph/v1",
        api_key=api_key,
        rate_limit_requests=100 if api_key else 10,
        rate_limit_period=300,  # 5 minutes
        topics=list(TopicDomain),
        priority=5,  # High authority
    )


class SemanticScholarSource(BaseSource):
    """
    Semantic Scholar API source implementation.

    Provides access to academic papers with rich citation data.
    """

    def __init__(self, config: Optional[SourceConfig] = None):
        super().__init__(config or create_semantic_scholar_config())
        self.rate_limiter = get_rate_limiter()

        # Configure rate limit based on API key
        if self.config.api_key:
            self.rate_limiter.set_limit("semantic_scholar", 100, 300)
        else:
            self.rate_limiter.set_limit("semantic_scholar", 10, 300)

    async def fetch_recent(
        self,
        max_results: int = 100,
        since: Optional[datetime] = None,
    ) -> list[RawArticle]:
        """
        Fetch recent papers across all topics.

        Args:
            max_results: Maximum papers to fetch
            since: Only papers after this date

        Returns:
            List of RawArticle objects
        """
        all_articles = []
        per_topic = max_results // len(TopicDomain)

        for topic in TopicDomain:
            articles = await self.fetch_by_topic(topic, per_topic)
            all_articles.extend(articles)

        # Filter by date if specified
        if since:
            all_articles = [
                a for a in all_articles
                if a.published_at and a.published_at > since
            ]

        return all_articles[:max_results]

    async def fetch_by_topic(
        self,
        topic: TopicDomain,
        max_results: int = 50,
    ) -> list[RawArticle]:
        """
        Fetch papers for a specific topic domain.

        Args:
            topic: Topic domain to fetch
            max_results: Maximum papers

        Returns:
            List of RawArticle objects
        """
        queries = DOMAIN_QUERIES.get(topic, [])
        if not queries:
            return []

        all_articles = []
        per_query = max(10, max_results // len(queries))

        for query in queries:
            articles = await self._search_papers(query, per_query, topic)
            all_articles.extend(articles)

            if len(all_articles) >= max_results:
                break

        return all_articles[:max_results]

    async def _search_papers(
        self,
        query: str,
        limit: int,
        topic: TopicDomain,
    ) -> list[RawArticle]:
        """Search for papers matching a query."""
        await self.rate_limiter.wait_if_needed("semantic_scholar")

        # Build year filter for recent papers
        current_year = datetime.now().year
        year_filter = f"{current_year - 1}-{current_year}"

        params = {
            "query": query,
            "limit": min(limit, 100),  # API max is 100
            "fields": ",".join(PAPER_FIELDS),
            "year": year_filter,
        }

        headers = {}
        if self.config.api_key:
            headers["x-api-key"] = self.config.api_key

        url = f"{self.config.base_url}/paper/search"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()

            data = response.json()
            papers = data.get("data", [])

            return [
                self._parse_paper(paper, topic)
                for paper in papers
                if self._parse_paper(paper, topic) is not None
            ]

        except httpx.HTTPError as e:
            logger.error(f"Semantic Scholar API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch from Semantic Scholar: {e}")
            return []

    def _parse_paper(
        self,
        paper: dict,
        topic: TopicDomain,
    ) -> Optional[RawArticle]:
        """Parse a paper from the API response."""
        paper_id = paper.get("paperId")
        title = paper.get("title")

        if not paper_id or not title:
            return None

        # External IDs
        external_ids = paper.get("externalIds", {})
        doi = external_ids.get("DOI")
        arxiv_id = external_ids.get("ArXiv")
        pmid = external_ids.get("PubMed")

        # URL
        url = paper.get("url") or f"https://www.semanticscholar.org/paper/{paper_id}"

        # Open access PDF
        pdf_info = paper.get("openAccessPdf")
        if pdf_info and isinstance(pdf_info, dict):
            pdf_url = pdf_info.get("url")
        else:
            pdf_url = None

        # Authors
        authors = []
        for author in paper.get("authors", []):
            name = author.get("name")
            if name:
                authors.append(name)

        # Publication date
        pub_date_str = paper.get("publicationDate")
        year = paper.get("year")
        published_at = self._parse_date(pub_date_str, year)

        # Fields of study
        fields = []
        for field in paper.get("s2FieldsOfStudy", []):
            category = field.get("category")
            if category:
                fields.append(category)

        # Publication types
        pub_types = paper.get("publicationTypes", [])
        peer_reviewed = "JournalArticle" in pub_types

        # Citation count as authority signal
        citation_count = paper.get("citationCount", 0)
        influential_citations = paper.get("influentialCitationCount", 0)

        # Map fields to our domains
        matched_domains = self.map_to_domain(fields)
        if not matched_domains:
            matched_domains = [topic]

        return RawArticle(
            external_id=paper_id,
            source_name="semantic_scholar",
            title=title,
            url=url,
            abstract=paper.get("abstract"),
            authors=authors,
            published_at=published_at,
            topics=fields,
            matched_domains=matched_domains,
            doi=doi,
            arxiv_id=arxiv_id,
            pmid=pmid,
            citation_count=citation_count,
            peer_reviewed=peer_reviewed,
            source_authority_score=self._calculate_authority(
                citation_count, influential_citations, peer_reviewed
            ),
        )

    def _parse_date(
        self,
        date_str: Optional[str],
        year: Optional[int],
    ) -> Optional[datetime]:
        """Parse publication date."""
        if date_str:
            try:
                return datetime.fromisoformat(date_str)
            except ValueError:
                pass

        if year:
            return datetime(year, 1, 1)

        return None

    def _calculate_authority(
        self,
        citations: int,
        influential: int,
        peer_reviewed: bool,
    ) -> float:
        """Calculate authority score based on citation metrics."""
        base_score = 0.6 if peer_reviewed else 0.4

        # Citation bonus (logarithmic scale)
        if citations > 0:
            import math
            citation_bonus = min(0.2, math.log10(citations + 1) * 0.05)
            base_score += citation_bonus

        # Influential citation bonus
        if influential > 0:
            influential_bonus = min(0.1, influential * 0.01)
            base_score += influential_bonus

        return min(1.0, base_score)

    def map_to_domain(self, source_categories: list[str]) -> list[TopicDomain]:
        """Map Semantic Scholar fields to our topic domains."""
        domain_mapping = {
            "Computer Science": TopicDomain.AI_ML,
            "Artificial Intelligence": TopicDomain.AI_ML,
            "Machine Learning": TopicDomain.AI_ML,
            "Physics": TopicDomain.PHYSICS,
            "Astrophysics": TopicDomain.PHYSICS,
            "Economics": TopicDomain.ECONOMICS,
            "Business": TopicDomain.ECONOMICS,
            "Biology": TopicDomain.BIOTECH,
            "Medicine": TopicDomain.BIOTECH,
            "Chemistry": TopicDomain.BIOTECH,
            "Political Science": TopicDomain.POLITICS,
            "Sociology": TopicDomain.POLITICS,
        }

        domains = set()
        for field in source_categories:
            if field in domain_mapping:
                domains.add(domain_mapping[field])

        return list(domains)
