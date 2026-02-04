"""
arXiv API adapter for fetching academic preprints.
arXiv API docs: https://info.arxiv.org/help/api/index.html
"""
import asyncio
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import AsyncIterator, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.models.domain import Article, ArticleSource, Author, ContentType, TopicPath
from app.sources.base import ArticleSourceAdapter

# arXiv category mappings for our taxonomy
TOPIC_TO_ARXIV_CATEGORIES = {
    # AI & ML
    "ai-ml/llms/interpretability": ["cs.CL", "cs.LG", "cs.AI"],
    "ai-ml/llms/architectures": ["cs.CL", "cs.LG"],
    "ai-ml/llms/scaling": ["cs.CL", "cs.LG"],
    "ai-ml/llms/applications": ["cs.CL", "cs.AI"],
    "ai-ml/computer-vision/generative": ["cs.CV", "cs.LG"],
    "ai-ml/computer-vision/detection": ["cs.CV"],
    "ai-ml/computer-vision/video": ["cs.CV"],
    "ai-ml/reinforcement/robotics": ["cs.RO", "cs.LG"],
    "ai-ml/reinforcement/games": ["cs.LG", "cs.AI"],
    "ai-ml/reinforcement/multi-agent": ["cs.MA", "cs.LG"],
    "ai-ml/ml-theory/optimization": ["cs.LG", "stat.ML"],
    "ai-ml/ml-theory/causality": ["stat.ML", "cs.LG"],
    "ai-ml/ml-theory/fairness": ["cs.LG", "cs.CY"],

    # Physics
    "physics/complexity/networks": ["physics.soc-ph", "cs.SI", "nlin.AO"],
    "physics/complexity/emergence": ["nlin.AO", "cond-mat.stat-mech"],
    "physics/complexity/chaos": ["nlin.CD", "physics.class-ph"],
    "physics/complexity/info-theory": ["cs.IT", "quant-ph"],
    "physics/quantum/computing": ["quant-ph", "cs.ET"],
    "physics/quantum/foundations": ["quant-ph"],
    "physics/quantum/materials": ["cond-mat.mtrl-sci", "quant-ph"],
    "physics/condensed/superconductivity": ["cond-mat.supr-con"],
    "physics/condensed/topological": ["cond-mat.mes-hall"],
    "physics/condensed/soft-matter": ["cond-mat.soft"],
    "physics/astro/exoplanets": ["astro-ph.EP"],
    "physics/astro/dark-matter": ["astro-ph.CO", "hep-ph"],
    "physics/astro/gravitational": ["gr-qc", "astro-ph.HE"],

    # Biotech
    "biotech/gene-editing/crispr": ["q-bio.GN", "q-bio.MN"],
    "biotech/gene-editing/gene-therapy": ["q-bio.GN"],
    "biotech/gene-editing/epigenetics": ["q-bio.GN"],
    "biotech/drug-discovery/ai-pharma": ["q-bio.BM", "cs.LG"],
    "biotech/drug-discovery/clinical-trials": ["stat.AP", "q-bio.QM"],
    "biotech/drug-discovery/small-molecules": ["q-bio.BM"],
    "biotech/synbio/metabolic": ["q-bio.MN"],
    "biotech/synbio/cell-free": ["q-bio.MN"],
    "biotech/synbio/biofuels": ["q-bio.MN"],
    "biotech/neuro/brain-computer": ["q-bio.NC", "cs.HC"],
    "biotech/neuro/connectomics": ["q-bio.NC"],
    "biotech/neuro/neurodegeneration": ["q-bio.NC"],
}

# Namespaces for arXiv Atom feed
NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class ArxivAdapter(ArticleSourceAdapter):
    """Adapter for fetching articles from arXiv."""

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self):
        self.settings = get_settings()
        self._rate_limiter = asyncio.Semaphore(self.settings.arxiv_rate_limit)

    @property
    def source_type(self) -> ArticleSource:
        return ArticleSource.ARXIV

    @property
    def name(self) -> str:
        return "arXiv"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _fetch(self, params: dict) -> str:
        """Fetch from arXiv API with rate limiting and retries."""
        async with self._rate_limiter:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.BASE_URL, params=params, timeout=30.0)
                response.raise_for_status()
                # arXiv asks for 3 second delay between requests
                await asyncio.sleep(3)
                return response.text

    def _parse_article(self, entry: ET.Element) -> Article:
        """Parse an arXiv Atom entry into an Article."""
        # Extract ID (remove version number for consistency)
        arxiv_id = entry.find("atom:id", NAMESPACES).text
        arxiv_id = arxiv_id.replace("http://arxiv.org/abs/", "")
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

        # Extract title (remove newlines)
        title = entry.find("atom:title", NAMESPACES).text
        title = " ".join(title.split())

        # Extract abstract
        abstract = entry.find("atom:summary", NAMESPACES).text
        abstract = " ".join(abstract.split()) if abstract else None

        # Extract authors
        authors = []
        for author_elem in entry.findall("atom:author", NAMESPACES):
            name = author_elem.find("atom:name", NAMESPACES).text
            affiliation_elem = author_elem.find("arxiv:affiliation", NAMESPACES)
            affiliation = affiliation_elem.text if affiliation_elem is not None else None
            authors.append(Author(name=name, affiliation=affiliation))

        # Extract publication date
        published = entry.find("atom:published", NAMESPACES).text
        published_at = datetime.fromisoformat(published.replace("Z", "+00:00"))

        # Extract URL
        url = f"https://arxiv.org/abs/{arxiv_id}"

        return Article(
            id=f"arxiv:{arxiv_id}",
            source=ArticleSource.ARXIV,
            content_type=ContentType.RESEARCH,
            title=title,
            abstract=abstract,
            authors=authors,
            published_at=published_at,
            url=url,
            citation_count=0,  # arXiv doesn't provide this, we get it from Semantic Scholar
        )

    async def search(
        self,
        topic: TopicPath,
        keywords: list[str],
        since: Optional[datetime] = None,
        max_results: int = 50,
    ) -> AsyncIterator[Article]:
        """Search arXiv for articles matching the topic and keywords."""
        # Get arXiv categories for this topic
        categories = TOPIC_TO_ARXIV_CATEGORIES.get(topic.path, ["cs.AI"])

        # Build search query
        cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
        keyword_query = " AND ".join(f"all:{kw}" for kw in keywords[:5])  # Limit keywords

        if keyword_query:
            search_query = f"({cat_query}) AND ({keyword_query})"
        else:
            search_query = f"({cat_query})"

        # Build date filter
        if since:
            # arXiv doesn't support date filtering in query, we filter in code
            pass

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        xml_response = await self._fetch(params)
        root = ET.fromstring(xml_response)

        for entry in root.findall("atom:entry", NAMESPACES):
            try:
                article = self._parse_article(entry)

                # Apply date filter
                if since and article.published_at < since:
                    continue

                yield article

            except Exception as e:
                # Log and continue on parse errors
                print(f"Error parsing arXiv entry: {e}")
                continue

    async def get_article(self, article_id: str) -> Optional[Article]:
        """Fetch a specific article by ID."""
        # Extract arXiv ID from our format
        if article_id.startswith("arxiv:"):
            arxiv_id = article_id[6:]
        else:
            arxiv_id = article_id

        params = {
            "id_list": arxiv_id,
        }

        xml_response = await self._fetch(params)
        root = ET.fromstring(xml_response)

        entries = root.findall("atom:entry", NAMESPACES)
        if entries:
            return self._parse_article(entries[0])
        return None

    async def get_citations(self, article_id: str) -> tuple[list[str], list[str]]:
        """
        arXiv doesn't provide citation data.
        We get this from Semantic Scholar instead.
        """
        return [], []

    async def health_check(self) -> bool:
        """Check if arXiv API is available."""
        try:
            params = {"search_query": "cat:cs.AI", "max_results": 1}
            await self._fetch(params)
            return True
        except Exception:
            return False
