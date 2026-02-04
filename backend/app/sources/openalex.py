"""
OpenAlex API adapter for academic papers.
OpenAlex is completely free and open, with 250M+ works.
API docs: https://docs.openalex.org/
"""
import asyncio
from datetime import datetime
from typing import AsyncIterator, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.models.domain import Article, ArticleSource, Author, ContentType, TopicPath
from app.sources.base import ArticleSourceAdapter

# OpenAlex concept IDs for our taxonomy
# These are OpenAlex's hierarchical concept identifiers
TOPIC_TO_CONCEPTS = {
    # AI & ML
    "ai-ml/llms": ["C154945302", "C119857082"],  # NLP, Machine Learning
    "ai-ml/computer-vision": ["C31972630"],  # Computer Vision
    "ai-ml/reinforcement": ["C119857082"],  # Machine Learning
    "ai-ml/ml-theory": ["C119857082", "C33923547"],  # ML, Mathematics

    # Physics
    "physics/complexity": ["C62520636", "C121332964"],  # Complex systems, Networks
    "physics/quantum": ["C62520636", "C182306322"],  # Physics, Quantum mechanics
    "physics/condensed": ["C205649164"],  # Condensed matter
    "physics/astro": ["C1965285"],  # Astrophysics

    # Biotech
    "biotech/gene-editing": ["C54355233", "C86803240"],  # Genetics, Biology
    "biotech/drug-discovery": ["C71924100", "C185592680"],  # Medicine, Pharmacology
    "biotech/synbio": ["C86803240"],  # Biology
    "biotech/neuro": ["C134018914"],  # Neuroscience

    # Economics
    "economics/macro": ["C162324750"],  # Economics
    "economics/markets": ["C162324750", "C144133560"],  # Economics, Business
    "economics/behavioral": ["C162324750", "C15744967"],  # Economics, Psychology
    "economics/development": ["C162324750"],  # Economics
}


class OpenAlexAdapter(ArticleSourceAdapter):
    """Adapter for fetching articles from OpenAlex."""

    BASE_URL = "https://api.openalex.org"

    def __init__(self, email: str = "noiseless@example.com"):
        self.settings = get_settings()
        self.email = email  # OpenAlex asks for email in polite pool
        self._rate_limiter = asyncio.Semaphore(100)  # OpenAlex allows 100 req/sec

    @property
    def source_type(self) -> ArticleSource:
        return ArticleSource.OPENLEX

    @property
    def name(self) -> str:
        return "OpenAlex"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def _fetch(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Fetch from OpenAlex API."""
        async with self._rate_limiter:
            async with httpx.AsyncClient() as client:
                url = f"{self.BASE_URL}/{endpoint}"
                params = params or {}
                params["mailto"] = self.email  # Polite pool
                response = await client.get(url, params=params, timeout=30.0)
                response.raise_for_status()
                return response.json()

    def _parse_work(self, work: dict) -> Optional[Article]:
        """Parse an OpenAlex work into an Article."""
        work_id = work.get("id", "").replace("https://openalex.org/", "")
        if not work_id:
            return None

        title = work.get("title") or work.get("display_name")
        if not title:
            return None

        # Parse authors
        authors = []
        for authorship in work.get("authorships", [])[:10]:  # Limit authors
            author_data = authorship.get("author", {})
            name = author_data.get("display_name", "Unknown")
            institutions = authorship.get("institutions", [])
            affiliation = institutions[0].get("display_name") if institutions else None
            orcid = author_data.get("orcid")
            authors.append(Author(name=name, affiliation=affiliation, orcid=orcid))

        # Parse publication date
        pub_date = work.get("publication_date")
        if pub_date:
            try:
                published_at = datetime.strptime(pub_date, "%Y-%m-%d")
            except ValueError:
                published_at = datetime.now()
        else:
            year = work.get("publication_year", 2000)
            published_at = datetime(year, 1, 1)

        # Get best URL
        primary_location = work.get("primary_location", {}) or {}
        url = (
            primary_location.get("landing_page_url")
            or work.get("doi")
            or f"https://openalex.org/{work_id}"
        )
        if url and url.startswith("https://doi.org/"):
            url = url  # Keep DOI URL

        # Get abstract (OpenAlex stores inverted index, we reconstruct)
        abstract = None
        abstract_inverted_index = work.get("abstract_inverted_index")
        if abstract_inverted_index:
            # Reconstruct abstract from inverted index
            word_positions = []
            for word, positions in abstract_inverted_index.items():
                for pos in positions:
                    word_positions.append((pos, word))
            word_positions.sort()
            abstract = " ".join(word for _, word in word_positions)

        # Get external IDs
        ids = work.get("ids", {})
        arxiv_id = None
        if openalex_ids := ids.get("openalex"):
            pass  # Already have work_id
        if pmid := ids.get("pmid"):
            pass  # Could use PubMed ID
        # Check for arXiv ID in locations
        for location in work.get("locations", []):
            source = location.get("source", {}) or {}
            if source.get("display_name") == "arXiv":
                # Extract arXiv ID from URL if available
                pdf_url = location.get("pdf_url", "")
                if "arxiv.org" in pdf_url:
                    import re
                    match = re.search(r"(\d{4}\.\d{4,5})", pdf_url)
                    if match:
                        arxiv_id = match.group(1)

        # Determine article ID
        if arxiv_id:
            article_id = f"arxiv:{arxiv_id}"
        else:
            article_id = f"openalex:{work_id}"

        return Article(
            id=article_id,
            source=ArticleSource.OPENLEX,
            content_type=ContentType.RESEARCH,
            title=title,
            abstract=abstract,
            authors=authors,
            published_at=published_at,
            url=url,
            citation_count=work.get("cited_by_count", 0),
            citing_paper_ids=[],
            cited_paper_ids=[],
        )

    async def search(
        self,
        topic: TopicPath,
        keywords: list[str],
        since: Optional[datetime] = None,
        max_results: int = 50,
    ) -> AsyncIterator[Article]:
        """Search OpenAlex for works matching the topic."""
        # Build filters
        filters = []

        # Get concept IDs for this topic
        topic_key = f"{topic.category_id}/{topic.subtopic_id}"
        concept_ids = TOPIC_TO_CONCEPTS.get(topic_key, [])
        if concept_ids:
            concept_filter = "|".join(concept_ids)
            filters.append(f"concepts.id:{concept_filter}")

        # Date filter
        if since:
            filters.append(f"from_publication_date:{since.strftime('%Y-%m-%d')}")

        # Minimum citations filter
        filters.append(f"cited_by_count:>{self.settings.min_citations_academic}")

        # Build query
        query = " ".join(keywords[:10]) if keywords else None

        params = {
            "per_page": min(max_results, 200),  # OpenAlex max is 200
            "sort": "cited_by_count:desc",  # Most cited first
            "select": "id,title,display_name,abstract_inverted_index,authorships,publication_date,publication_year,cited_by_count,primary_location,locations,ids",
        }

        if filters:
            params["filter"] = ",".join(filters)

        if query:
            params["search"] = query

        try:
            data = await self._fetch("works", params)

            for work in data.get("results", []):
                article = self._parse_work(work)
                if article:
                    yield article

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                await asyncio.sleep(10)
            raise

    async def get_article(self, article_id: str) -> Optional[Article]:
        """Fetch a specific work by ID."""
        # Handle different ID formats
        if article_id.startswith("openalex:"):
            work_id = article_id[9:]
        elif article_id.startswith("arxiv:"):
            # Search by arXiv ID
            params = {
                "filter": f"ids.openalex:{article_id}",
                "per_page": 1,
            }
            try:
                data = await self._fetch("works", params)
                works = data.get("results", [])
                if works:
                    return self._parse_work(works[0])
            except Exception:
                pass
            return None
        else:
            work_id = article_id

        try:
            data = await self._fetch(f"works/{work_id}")
            return self._parse_work(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def get_citations(self, article_id: str) -> tuple[list[str], list[str]]:
        """Get citation relationships for a work."""
        citing_ids = []
        cited_ids = []

        # Normalize ID
        if article_id.startswith("openalex:"):
            work_id = article_id[9:]
        else:
            return [], []

        try:
            # Get works that cite this one
            params = {
                "filter": f"cites:{work_id}",
                "per_page": 200,
                "select": "id,ids",
            }
            data = await self._fetch("works", params)

            for work in data.get("results", []):
                wid = work.get("id", "").replace("https://openalex.org/", "")
                if wid:
                    citing_ids.append(f"openalex:{wid}")

            # Get works this one cites
            params = {
                "filter": f"cited_by:{work_id}",
                "per_page": 200,
                "select": "id,ids",
            }
            data = await self._fetch("works", params)

            for work in data.get("results", []):
                wid = work.get("id", "").replace("https://openalex.org/", "")
                if wid:
                    cited_ids.append(f"openalex:{wid}")

        except Exception:
            pass

        return citing_ids, cited_ids

    async def health_check(self) -> bool:
        """Check if OpenAlex API is available."""
        try:
            await self._fetch("works", {"per_page": 1})
            return True
        except Exception:
            return False
