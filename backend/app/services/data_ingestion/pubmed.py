"""
PubMed/NCBI E-utilities API integration.

PubMed provides access to biomedical literature including
peer-reviewed articles, clinical studies, and more.

API Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25497/
"""

import asyncio
import re
from datetime import datetime, timedelta
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

# PubMed search terms for each topic domain
DOMAIN_SEARCH_TERMS = {
    TopicDomain.BIOTECH: [
        "biotechnology",
        "gene therapy",
        "CRISPR",
        "genomics",
        "synthetic biology",
        "bioinformatics",
        "drug discovery",
        "immunotherapy",
    ],
    TopicDomain.AI_ML: [
        "machine learning medicine",
        "artificial intelligence healthcare",
        "deep learning radiology",
        "clinical decision support",
    ],
}


def create_pubmed_config(api_key: Optional[str] = None) -> SourceConfig:
    """Create default PubMed source configuration."""
    return SourceConfig(
        name="pubmed",
        source_type=SourceType.ACADEMIC_API,
        base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        api_key=api_key,
        rate_limit_requests=3 if api_key is None else 10,
        rate_limit_period=1,
        topics=[TopicDomain.BIOTECH, TopicDomain.AI_ML],
        priority=5,  # High authority for peer-reviewed content
    )


class PubMedSource(BaseSource):
    """
    PubMed E-utilities API source implementation.

    Uses the E-utilities API to search and fetch articles.
    """

    def __init__(self, config: Optional[SourceConfig] = None):
        super().__init__(config or create_pubmed_config())
        self.rate_limiter = get_rate_limiter()

        # Set rate limit based on API key presence
        if self.config.api_key:
            self.rate_limiter.set_limit("pubmed", 10, 1)
        else:
            self.rate_limiter.set_limit("pubmed", 3, 1)

    async def fetch_recent(
        self,
        max_results: int = 100,
        since: Optional[datetime] = None,
    ) -> list[RawArticle]:
        """
        Fetch recent biotech articles.

        Args:
            max_results: Maximum articles to fetch
            since: Only articles after this date

        Returns:
            List of RawArticle objects
        """
        # Default to last 7 days
        if since is None:
            since = datetime.utcnow() - timedelta(days=7)

        # Build search terms for all biotech topics
        search_terms = DOMAIN_SEARCH_TERMS.get(TopicDomain.BIOTECH, ["biotechnology"])
        query = " OR ".join(f'"{term}"' for term in search_terms[:5])

        return await self._search_and_fetch(query, max_results, since)

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
        search_terms = DOMAIN_SEARCH_TERMS.get(topic, [])
        if not search_terms:
            logger.warning(f"No PubMed search terms for topic: {topic}")
            return []

        query = " OR ".join(f'"{term}"' for term in search_terms[:5])
        since = datetime.utcnow() - timedelta(days=7)

        return await self._search_and_fetch(query, max_results, since)

    async def _search_and_fetch(
        self,
        query: str,
        max_results: int,
        since: datetime,
    ) -> list[RawArticle]:
        """Search for articles and fetch their details."""
        # Step 1: Search for PMIDs
        pmids = await self._esearch(query, max_results, since)
        if not pmids:
            return []

        # Step 2: Fetch article details
        return await self._efetch(pmids)

    async def _esearch(
        self,
        query: str,
        max_results: int,
        since: datetime,
    ) -> list[str]:
        """Search PubMed and return list of PMIDs."""
        await self.rate_limiter.wait_if_needed("pubmed")

        # Format date for PubMed
        date_str = since.strftime("%Y/%m/%d")

        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "date",
            "mindate": date_str,
            "maxdate": "3000",  # No upper limit
            "datetype": "pdat",  # Publication date
        }

        if self.config.api_key:
            params["api_key"] = self.config.api_key

        url = f"{self.config.base_url}/esearch.fcgi"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()

            data = response.json()
            return data.get("esearchresult", {}).get("idlist", [])

        except httpx.HTTPError as e:
            logger.error(f"PubMed search error: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to search PubMed: {e}")
            return []

    async def _efetch(self, pmids: list[str]) -> list[RawArticle]:
        """Fetch article details for given PMIDs."""
        if not pmids:
            return []

        await self.rate_limiter.wait_if_needed("pubmed")

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }

        if self.config.api_key:
            params["api_key"] = self.config.api_key

        url = f"{self.config.base_url}/efetch.fcgi"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()

            return self._parse_efetch_xml(response.text)

        except httpx.HTTPError as e:
            logger.error(f"PubMed fetch error: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch from PubMed: {e}")
            return []

    def _parse_efetch_xml(self, xml_content: str) -> list[RawArticle]:
        """Parse PubMed efetch XML into RawArticle objects."""
        articles = []

        try:
            root = ElementTree.fromstring(xml_content)
        except ElementTree.ParseError as e:
            logger.error(f"Failed to parse PubMed XML: {e}")
            return []

        for article_elem in root.findall(".//PubmedArticle"):
            try:
                article = self._parse_article(article_elem)
                if article:
                    articles.append(article)
            except Exception as e:
                logger.warning(f"Failed to parse PubMed article: {e}")
                continue

        return articles

    def _parse_article(self, elem: ElementTree.Element) -> Optional[RawArticle]:
        """Parse a single PubmedArticle element."""
        # Get PMID
        pmid_elem = elem.find(".//PMID")
        if pmid_elem is None or not pmid_elem.text:
            return None
        pmid = pmid_elem.text

        # Get article data
        medline = elem.find(".//MedlineCitation")
        if medline is None:
            return None

        article_data = medline.find(".//Article")
        if article_data is None:
            return None

        # Title
        title_elem = article_data.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None and title_elem.text else ""
        if not title:
            return None

        # Abstract
        abstract_parts = []
        for abstract_text in article_data.findall(".//AbstractText"):
            label = abstract_text.get("Label", "")
            text = abstract_text.text or ""
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Authors
        authors = []
        for author in article_data.findall(".//Author"):
            last_name = author.findtext("LastName", "")
            fore_name = author.findtext("ForeName", "")
            if last_name:
                authors.append(f"{fore_name} {last_name}".strip())

        # Publication date
        pub_date = self._parse_pub_date(article_data)

        # DOI
        doi = None
        for article_id in elem.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text
                break

        # MeSH terms (for topic mapping)
        mesh_terms = []
        for mesh in medline.findall(".//MeshHeading/DescriptorName"):
            if mesh.text:
                mesh_terms.append(mesh.text)

        # Keywords
        keywords = []
        for keyword in medline.findall(".//Keyword"):
            if keyword.text:
                keywords.append(keyword.text)

        # URL
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        return RawArticle(
            external_id=pmid,
            source_name="pubmed",
            title=title,
            url=url,
            abstract=abstract,
            authors=authors,
            published_at=pub_date,
            topics=mesh_terms + keywords,
            matched_domains=[TopicDomain.BIOTECH],  # Primary domain
            doi=doi,
            pmid=pmid,
            peer_reviewed=True,
            source_authority_score=0.9,  # High authority for peer-reviewed
        )

    def _parse_pub_date(self, article: ElementTree.Element) -> Optional[datetime]:
        """Parse publication date from various PubMed date formats."""
        # Try ArticleDate first (electronic publication)
        article_date = article.find(".//ArticleDate")
        if article_date is not None:
            year = article_date.findtext("Year")
            month = article_date.findtext("Month", "1")
            day = article_date.findtext("Day", "1")
            if year:
                try:
                    return datetime(int(year), int(month), int(day))
                except ValueError:
                    pass

        # Fall back to Journal PubDate
        pub_date = article.find(".//Journal/JournalIssue/PubDate")
        if pub_date is not None:
            year = pub_date.findtext("Year")
            month = pub_date.findtext("Month", "1")
            day = pub_date.findtext("Day", "1")

            # Handle month names
            month_map = {
                "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
                "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
                "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
            }
            if not month.isdigit():
                month = str(month_map.get(month[:3], 1))

            if year:
                try:
                    return datetime(int(year), int(month), int(day))
                except ValueError:
                    pass

        return None

    def map_to_domain(self, source_categories: list[str]) -> list[TopicDomain]:
        """Map MeSH terms to topic domains."""
        domains = set()

        biotech_terms = {
            "biotechnology", "genetic engineering", "crispr",
            "gene therapy", "genomics", "proteomics",
            "synthetic biology", "bioinformatics",
        }

        ai_terms = {
            "machine learning", "artificial intelligence",
            "deep learning", "neural networks",
        }

        for term in source_categories:
            term_lower = term.lower()
            if any(bt in term_lower for bt in biotech_terms):
                domains.add(TopicDomain.BIOTECH)
            if any(at in term_lower for at in ai_terms):
                domains.add(TopicDomain.AI_ML)

        return list(domains) if domains else [TopicDomain.BIOTECH]
