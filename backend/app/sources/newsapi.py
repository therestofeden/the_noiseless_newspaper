"""
NewsAPI adapter for current news articles.
API docs: https://newsapi.org/docs
"""
import asyncio
from datetime import datetime, timedelta
from typing import AsyncIterator, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.models.domain import Article, ArticleSource, Author, ContentType, TopicPath
from app.sources.base import ArticleSourceAdapter

# Keywords and sources for our taxonomy
TOPIC_TO_NEWS_CONFIG = {
    # Economics
    "economics/macro/monetary": {
        "keywords": ["central bank", "federal reserve", "ECB", "interest rates", "monetary policy"],
        "domains": "ft.com,economist.com,bloomberg.com,reuters.com",
    },
    "economics/macro/fiscal": {
        "keywords": ["fiscal policy", "government spending", "national debt", "budget deficit"],
        "domains": "ft.com,economist.com,bloomberg.com",
    },
    "economics/macro/trade": {
        "keywords": ["international trade", "tariffs", "trade war", "WTO", "trade agreement"],
        "domains": "ft.com,economist.com,reuters.com",
    },
    "economics/markets/equities": {
        "keywords": ["stock market", "S&P 500", "earnings", "IPO", "equity"],
        "domains": "bloomberg.com,ft.com,wsj.com,reuters.com",
    },
    "economics/markets/crypto": {
        "keywords": ["bitcoin", "ethereum", "cryptocurrency", "blockchain", "DeFi"],
        "domains": "coindesk.com,bloomberg.com,ft.com",
    },
    "economics/markets/derivatives": {
        "keywords": ["derivatives", "options", "futures", "hedge fund", "risk management"],
        "domains": "bloomberg.com,ft.com,risk.net",
    },
    "economics/behavioral/decision-making": {
        "keywords": ["behavioral economics", "decision making", "cognitive bias"],
        "domains": "economist.com,ft.com",
    },
    "economics/behavioral/nudges": {
        "keywords": ["nudge", "behavioral policy", "choice architecture"],
        "domains": "economist.com,ft.com",
    },
    "economics/development/poverty": {
        "keywords": ["poverty", "inequality", "development economics", "World Bank"],
        "domains": "economist.com,ft.com,devex.com",
    },
    "economics/development/institutions": {
        "keywords": ["economic institutions", "governance", "economic growth", "IMF"],
        "domains": "economist.com,ft.com",
    },

    # Politics
    "politics/geopolitics/us-china": {
        "keywords": ["US China", "Taiwan", "trade war", "semiconductor", "decoupling"],
        "domains": "foreignaffairs.com,economist.com,ft.com,reuters.com",
    },
    "politics/geopolitics/europe": {
        "keywords": ["European Union", "EU", "Brexit", "Eurozone"],
        "domains": "economist.com,ft.com,politico.eu,reuters.com",
    },
    "politics/geopolitics/emerging": {
        "keywords": ["BRICS", "emerging markets", "India", "Brazil", "global south"],
        "domains": "economist.com,ft.com,foreignaffairs.com",
    },
    "politics/domestic/elections": {
        "keywords": ["election", "voting", "polls", "campaign", "democracy"],
        "domains": "economist.com,ft.com,politico.com,reuters.com",
    },
    "politics/domestic/polarization": {
        "keywords": ["polarization", "misinformation", "social media", "political divide"],
        "domains": "economist.com,theatlantic.com",
    },
    "politics/domestic/policy": {
        "keywords": ["policy analysis", "legislation", "regulation", "government"],
        "domains": "economist.com,ft.com,politico.com",
    },
    "politics/governance/democracy": {
        "keywords": ["democracy", "democratic institutions", "rule of law", "authoritarianism"],
        "domains": "economist.com,foreignaffairs.com,ft.com",
    },
    "politics/governance/tech-regulation": {
        "keywords": ["tech regulation", "AI regulation", "antitrust", "data privacy", "GDPR"],
        "domains": "economist.com,ft.com,wired.com,techcrunch.com",
    },
    "politics/governance/international-law": {
        "keywords": ["international law", "UN", "sanctions", "human rights", "ICC"],
        "domains": "economist.com,ft.com,foreignaffairs.com",
    },
    "politics/security/cyber": {
        "keywords": ["cybersecurity", "cyber attack", "hacking", "ransomware"],
        "domains": "wired.com,arstechnica.com,ft.com",
    },
    "politics/security/military": {
        "keywords": ["military", "defense", "NATO", "arms", "warfare"],
        "domains": "economist.com,ft.com,foreignaffairs.com,reuters.com",
    },
}


class NewsAPIAdapter(ArticleSourceAdapter):
    """Adapter for fetching news articles from NewsAPI."""

    BASE_URL = "https://newsapi.org/v2"

    def __init__(self):
        self.settings = get_settings()
        self._rate_limiter = asyncio.Semaphore(self.settings.newsapi_rate_limit)

    @property
    def source_type(self) -> ArticleSource:
        return ArticleSource.NEWSAPI

    @property
    def name(self) -> str:
        return "NewsAPI"

    def _has_api_key(self) -> bool:
        return bool(self.settings.newsapi_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _fetch(self, endpoint: str, params: dict) -> dict:
        """Fetch from NewsAPI with rate limiting."""
        if not self._has_api_key():
            raise ValueError("NewsAPI key not configured")

        async with self._rate_limiter:
            async with httpx.AsyncClient() as client:
                url = f"{self.BASE_URL}/{endpoint}"
                params["apiKey"] = self.settings.newsapi_key
                response = await client.get(url, params=params, timeout=30.0)
                response.raise_for_status()
                data = response.json()

                if data.get("status") != "ok":
                    raise ValueError(f"NewsAPI error: {data.get('message')}")

                return data

    def _parse_article(self, article: dict) -> Optional[Article]:
        """Parse a NewsAPI article into our Article model."""
        title = article.get("title")
        if not title or title == "[Removed]":
            return None

        url = article.get("url")
        if not url:
            return None

        # Parse author
        authors = []
        author_str = article.get("author")
        if author_str:
            # NewsAPI sometimes has comma-separated authors
            for name in author_str.split(",")[:3]:
                name = name.strip()
                if name and name.lower() not in ["unknown", "null", "none"]:
                    authors.append(Author(name=name))

        # Parse publication date
        pub_date_str = article.get("publishedAt")
        if pub_date_str:
            try:
                published_at = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
            except ValueError:
                published_at = datetime.now()
        else:
            published_at = datetime.now()

        # Get source info
        source = article.get("source", {})
        source_name = source.get("name", "Unknown")

        # Generate ID from URL hash
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        article_id = f"news:{url_hash}"

        # Get description as abstract
        abstract = article.get("description")
        if abstract == "[Removed]":
            abstract = None

        return Article(
            id=article_id,
            source=ArticleSource.NEWSAPI,
            content_type=ContentType.NEWS,
            title=title,
            abstract=abstract,
            authors=authors,
            published_at=published_at,
            url=url,
            citation_count=0,  # News doesn't have citations in the traditional sense
        )

    async def search(
        self,
        topic: TopicPath,
        keywords: list[str],
        since: Optional[datetime] = None,
        max_results: int = 50,
    ) -> AsyncIterator[Article]:
        """Search NewsAPI for articles matching the topic."""
        if not self._has_api_key():
            return

        # Get config for this topic
        topic_key = f"{topic.category_id}/{topic.subtopic_id}/{topic.niche_id}"
        config = TOPIC_TO_NEWS_CONFIG.get(topic_key, {})

        # Combine topic keywords with provided keywords
        all_keywords = list(set(config.get("keywords", []) + keywords))

        # Build query - NewsAPI uses boolean operators
        if len(all_keywords) > 1:
            query = " OR ".join(f'"{kw}"' for kw in all_keywords[:5])
        elif all_keywords:
            query = all_keywords[0]
        else:
            query = topic.niche_id.replace("-", " ")

        # Date range (NewsAPI free tier only allows 1 month back)
        if since is None:
            since = datetime.now() - timedelta(days=7)

        params = {
            "q": query,
            "from": since.strftime("%Y-%m-%d"),
            "sortBy": "relevancy",
            "pageSize": min(max_results, 100),
            "language": "en",
        }

        # Add domain filter if configured
        if domains := config.get("domains"):
            params["domains"] = domains

        try:
            data = await self._fetch("everything", params)

            for article_data in data.get("articles", []):
                article = self._parse_article(article_data)
                if article:
                    yield article

        except Exception as e:
            print(f"NewsAPI error: {e}")

    async def get_article(self, article_id: str) -> Optional[Article]:
        """
        NewsAPI doesn't support fetching by ID.
        We'd need to store articles in our DB.
        """
        return None

    async def get_citations(self, article_id: str) -> tuple[list[str], list[str]]:
        """News articles don't have citations in the traditional sense."""
        return [], []

    async def health_check(self) -> bool:
        """Check if NewsAPI is available."""
        if not self._has_api_key():
            return False

        try:
            params = {"q": "test", "pageSize": 1}
            await self._fetch("everything", params)
            return True
        except Exception:
            return False
