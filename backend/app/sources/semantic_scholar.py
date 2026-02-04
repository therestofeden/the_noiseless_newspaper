"""
Semantic Scholar API adapter for academic papers with citation data.
API docs: https://api.semanticscholar.org/api-docs/
"""
import asyncio
from datetime import datetime
from typing import AsyncIterator, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.models.domain import Article, ArticleSource, Author, ContentType, TopicPath
from app.sources.base import ArticleSourceAdapter

# Field of study mappings for our taxonomy
TOPIC_TO_FIELDS = {
    # AI & ML
    "ai-ml/llms": ["Computer Science", "Artificial Intelligence"],
    "ai-ml/computer-vision": ["Computer Science", "Computer Vision"],
    "ai-ml/reinforcement": ["Computer Science", "Machine Learning"],
    "ai-ml/ml-theory": ["Computer Science", "Machine Learning", "Mathematics"],

    # Physics
    "physics/complexity": ["Physics", "Mathematics"],
    "physics/quantum": ["Physics", "Quantum Physics"],
    "physics/condensed": ["Physics", "Materials Science"],
    "physics/astro": ["Physics", "Astrophysics"],

    # Biotech
    "biotech/gene-editing": ["Biology", "Genetics"],
    "biotech/drug-discovery": ["Medicine", "Biology", "Chemistry"],
    "biotech/synbio": ["Biology", "Bioengineering"],
    "biotech/neuro": ["Neuroscience", "Biology"],

    # Economics (Semantic Scholar has limited coverage here)
    "economics/macro": ["Economics"],
    "economics/markets": ["Economics", "Business"],
    "economics/behavioral": ["Economics", "Psychology"],
    "economics/development": ["Economics", "Political Science"],
}


class SemanticScholarAdapter(ArticleSourceAdapter):
    """Adapter for fetching articles from Semantic Scholar."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self):
        self.settings = get_settings()
        self._rate_limiter = asyncio.Semaphore(self.settings.semantic_scholar_rate_limit)

    @property
    def source_type(self) -> ArticleSource:
        return ArticleSource.SEMANTIC_SCHOLAR

    @property
    def name(self) -> str:
        return "Semantic Scholar"

    def _get_headers(self) -> dict:
        """Get request headers, including API key if available."""
        headers = {"Content-Type": "application/json"}
        if self.settings.semantic_scholar_api_key:
            headers["x-api-key"] = self.settings.semantic_scholar_api_key
        return headers

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _fetch(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Fetch from Semantic Scholar API with rate limiting."""
        async with self._rate_limiter:
            async with httpx.AsyncClient() as client:
                url = f"{self.BASE_URL}/{endpoint}"
                response = await client.get(
                    url,
                    params=params,
                    headers=self._get_headers(),
                    timeout=30.0,
                )
                response.raise_for_status()
                return response.json()

    def _parse_paper(self, paper: dict) -> Optional[Article]:
        """Parse a Semantic Scholar paper into an Article."""
        paper_id = paper.get("paperId")
        if not paper_id:
            return None

        title = paper.get("title")
        if not title:
            return None

        # Parse authors
        authors = []
        for author in paper.get("authors", []):
            authors.append(Author(
                name=author.get("name", "Unknown"),
                affiliation=None,  # S2 doesn't provide this in basic response
            ))

        # Parse publication date
        pub_date = paper.get("publicationDate")
        if pub_date:
            try:
                published_at = datetime.strptime(pub_date, "%Y-%m-%d")
            except ValueError:
                # Sometimes it's just a year
                try:
                    year = int(paper.get("year", 2000))
                    published_at = datetime(year, 1, 1)
                except (ValueError, TypeError):
                    published_at = datetime.now()
        else:
            year = paper.get("year")
            if year:
                published_at = datetime(int(year), 1, 1)
            else:
                published_at = datetime.now()

        # Get URL
        url = paper.get("url") or f"https://www.semanticscholar.org/paper/{paper_id}"

        # Get external IDs for linking
        external_ids = paper.get("externalIds", {})
        arxiv_id = external_ids.get("ArXiv")
        doi = external_ids.get("DOI")

        # Determine article ID (prefer arXiv ID if available)
        if arxiv_id:
            article_id = f"arxiv:{arxiv_id}"
        else:
            article_id = f"s2:{paper_id}"

        return Article(
            id=article_id,
            source=ArticleSource.SEMANTIC_SCHOLAR,
            content_type=ContentType.RESEARCH,
            title=title,
            abstract=paper.get("abstract"),
            authors=authors,
            published_at=published_at,
            url=url,
            citation_count=paper.get("citationCount", 0),
            citing_paper_ids=[],  # Fetched separately
            cited_paper_ids=[],
        )

    async def search(
        self,
        topic: TopicPath,
        keywords: list[str],
        since: Optional[datetime] = None,
        max_results: int = 50,
    ) -> AsyncIterator[Article]:
        """Search Semantic Scholar for papers matching the topic."""
        # Build query from keywords
        query = " ".join(keywords[:10])  # S2 limits query length

        # Get fields of study for filtering
        topic_key = f"{topic.category_id}/{topic.subtopic_id}"
        fields_of_study = TOPIC_TO_FIELDS.get(topic_key, [])

        # Build year filter
        year_filter = None
        if since:
            year_filter = f"{since.year}-"

        params = {
            "query": query,
            "limit": min(max_results, 100),  # S2 max is 100
            "fields": "paperId,title,abstract,authors,year,publicationDate,citationCount,url,externalIds",
        }

        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study[:3])

        if year_filter:
            params["year"] = year_filter

        try:
            data = await self._fetch("paper/search", params)

            for paper in data.get("data", []):
                article = self._parse_paper(paper)
                if article:
                    # Apply date filter (S2's year filter is coarse)
                    if since and article.published_at < since:
                        continue

                    # Apply minimum citation filter for cold start
                    if article.citation_count >= self.settings.min_citations_academic:
                        yield article

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Rate limited, wait and skip
                await asyncio.sleep(60)
            raise

    async def get_article(self, article_id: str) -> Optional[Article]:
        """Fetch a specific paper by ID."""
        # Handle different ID formats
        if article_id.startswith("arxiv:"):
            paper_id = f"ArXiv:{article_id[6:]}"
        elif article_id.startswith("s2:"):
            paper_id = article_id[3:]
        else:
            paper_id = article_id

        params = {
            "fields": "paperId,title,abstract,authors,year,publicationDate,citationCount,url,externalIds,citations,references",
        }

        try:
            data = await self._fetch(f"paper/{paper_id}", params)
            return self._parse_paper(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def get_citations(self, article_id: str) -> tuple[list[str], list[str]]:
        """Get citation relationships for a paper."""
        # Handle different ID formats
        if article_id.startswith("arxiv:"):
            paper_id = f"ArXiv:{article_id[6:]}"
        elif article_id.startswith("s2:"):
            paper_id = article_id[3:]
        else:
            paper_id = article_id

        citing_ids = []
        cited_ids = []

        try:
            # Get papers that cite this one
            params = {"fields": "paperId,externalIds", "limit": 500}
            data = await self._fetch(f"paper/{paper_id}/citations", params)

            for item in data.get("data", []):
                citing_paper = item.get("citingPaper", {})
                ext_ids = citing_paper.get("externalIds", {})
                if arxiv_id := ext_ids.get("ArXiv"):
                    citing_ids.append(f"arxiv:{arxiv_id}")
                elif s2_id := citing_paper.get("paperId"):
                    citing_ids.append(f"s2:{s2_id}")

            # Get papers this one cites
            data = await self._fetch(f"paper/{paper_id}/references", params)

            for item in data.get("data", []):
                cited_paper = item.get("citedPaper", {})
                ext_ids = cited_paper.get("externalIds", {})
                if arxiv_id := ext_ids.get("ArXiv"):
                    cited_ids.append(f"arxiv:{arxiv_id}")
                elif s2_id := cited_paper.get("paperId"):
                    cited_ids.append(f"s2:{s2_id}")

        except httpx.HTTPStatusError:
            pass  # Return empty lists on error

        return citing_ids, cited_ids

    async def health_check(self) -> bool:
        """Check if Semantic Scholar API is available."""
        try:
            await self._fetch("paper/search", {"query": "test", "limit": 1})
            return True
        except Exception:
            return False

    async def get_citation_velocity(self, article_id: str, months: int = 6) -> float:
        """Calculate citations per month over recent period."""
        try:
            article = await self.get_article(article_id)
            if not article:
                return 0.0

            # Calculate age in months
            age_days = (datetime.now() - article.published_at).days
            age_months = max(1, age_days / 30)

            # If newer than our window, use actual age
            if age_months < months:
                return article.citation_count / age_months

            # Otherwise estimate recent velocity (would need historical data for accuracy)
            return article.citation_count / age_months

        except Exception:
            return 0.0
