"""
RSS Feed aggregation for news sources.

Handles fetching and parsing RSS/Atom feeds from various
news outlets, blogs, and think tanks.
"""

import asyncio
import hashlib
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Optional
from xml.etree import ElementTree
import logging
import re

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

# Comprehensive RSS feed configuration
RSS_FEEDS = {
    TopicDomain.AI_ML: [
        {
            "name": "MIT Technology Review - AI",
            "url": "https://www.technologyreview.com/feed/",
            "authority": 0.8,
        },
        {
            "name": "Google AI Blog",
            "url": "https://blog.google/technology/ai/rss/",
            "authority": 0.85,
        },
        {
            "name": "OpenAI Blog",
            "url": "https://openai.com/blog/rss.xml",
            "authority": 0.85,
        },
        {
            "name": "The Gradient",
            "url": "https://thegradient.pub/rss/",
            "authority": 0.75,
        },
        {
            "name": "Hugging Face Papers",
            "url": "https://huggingface.co/papers/rss",
            "authority": 0.7,
        },
    ],
    TopicDomain.PHYSICS: [
        {
            "name": "APS Physics",
            "url": "https://physics.aps.org/feed",
            "authority": 0.9,
        },
        {
            "name": "Quanta Magazine",
            "url": "https://www.quantamagazine.org/feed/",
            "authority": 0.85,
        },
        {
            "name": "CERN News",
            "url": "https://home.cern/api/news/news/feed.rss",
            "authority": 0.9,
        },
        {
            "name": "Phys.org Physics",
            "url": "https://phys.org/rss-feed/physics-news/",
            "authority": 0.7,
        },
        {
            "name": "NASA Science",
            "url": "https://science.nasa.gov/rss-feeds/",
            "authority": 0.9,
        },
    ],
    TopicDomain.ECONOMICS: [
        {
            "name": "NBER Working Papers",
            "url": "https://www.nber.org/rss/new.xml",
            "authority": 0.9,
        },
        {
            "name": "VoxEU",
            "url": "https://voxeu.org/rss.xml",
            "authority": 0.85,
        },
        {
            "name": "Brookings",
            "url": "https://www.brookings.edu/feed/",
            "authority": 0.85,
        },
        {
            "name": "IMF Blog",
            "url": "https://www.imf.org/en/News/rss",
            "authority": 0.85,
        },
        {
            "name": "Peterson Institute",
            "url": "https://www.piie.com/rss.xml",
            "authority": 0.8,
        },
    ],
    TopicDomain.BIOTECH: [
        {
            "name": "STAT News",
            "url": "https://www.statnews.com/feed/",
            "authority": 0.8,
        },
        {
            "name": "GenomeWeb",
            "url": "https://www.genomeweb.com/rss.xml",
            "authority": 0.75,
        },
        {
            "name": "The Scientist",
            "url": "https://www.the-scientist.com/rss",
            "authority": 0.75,
        },
        {
            "name": "NIH News",
            "url": "https://www.nih.gov/news-events/news-releases/feed",
            "authority": 0.9,
        },
    ],
    TopicDomain.POLITICS: [
        {
            "name": "Council on Foreign Relations",
            "url": "https://www.cfr.org/rss.xml",
            "authority": 0.85,
        },
        {
            "name": "Lawfare",
            "url": "https://www.lawfaremedia.org/feed",
            "authority": 0.8,
        },
        {
            "name": "RAND Corporation",
            "url": "https://www.rand.org/news.xml",
            "authority": 0.85,
        },
        {
            "name": "The Hill",
            "url": "https://thehill.com/feed/",
            "authority": 0.7,
        },
        {
            "name": "Politico",
            "url": "https://www.politico.com/rss/politics.xml",
            "authority": 0.75,
        },
    ],

    # ========== NEW GENERAL TOPICS ==========

    TopicDomain.SPORTS: [
        {
            "name": "ESPN",
            "url": "https://www.espn.com/espn/rss/news",
            "authority": 0.8,
        },
        {
            "name": "ESPN NFL",
            "url": "https://www.espn.com/espn/rss/nfl/news",
            "authority": 0.8,
        },
        {
            "name": "ESPN NBA",
            "url": "https://www.espn.com/espn/rss/nba/news",
            "authority": 0.8,
        },
        {
            "name": "BBC Sport",
            "url": "https://feeds.bbci.co.uk/sport/rss.xml",
            "authority": 0.85,
        },
        {
            "name": "BBC Football",
            "url": "https://feeds.bbci.co.uk/sport/football/rss.xml",
            "authority": 0.85,
        },
        {
            "name": "Sports Illustrated",
            "url": "https://www.si.com/rss/si_topstories.rss",
            "authority": 0.75,
        },
        {
            "name": "The Ringer",
            "url": "https://www.theringer.com/rss/index.xml",
            "authority": 0.7,
        },
    ],

    TopicDomain.ENTERTAINMENT: [
        {
            "name": "Variety",
            "url": "https://variety.com/feed/",
            "authority": 0.85,
        },
        {
            "name": "Hollywood Reporter",
            "url": "https://www.hollywoodreporter.com/feed/",
            "authority": 0.85,
        },
        {
            "name": "Deadline",
            "url": "https://deadline.com/feed/",
            "authority": 0.8,
        },
        {
            "name": "Pitchfork",
            "url": "https://pitchfork.com/feed/feed-news/rss",
            "authority": 0.8,
        },
        {
            "name": "Rolling Stone",
            "url": "https://www.rollingstone.com/feed/",
            "authority": 0.75,
        },
        {
            "name": "The Guardian Culture",
            "url": "https://www.theguardian.com/culture/rss",
            "authority": 0.8,
        },
        {
            "name": "Vulture",
            "url": "https://www.vulture.com/feed/rss/index.xml",
            "authority": 0.75,
        },
        {
            "name": "IndieWire",
            "url": "https://www.indiewire.com/feed/",
            "authority": 0.75,
        },
        {
            "name": "Literary Hub",
            "url": "https://lithub.com/feed/",
            "authority": 0.75,
        },
    ],

    TopicDomain.TECHNOLOGY: [
        {
            "name": "TechCrunch",
            "url": "https://techcrunch.com/feed/",
            "authority": 0.8,
        },
        {
            "name": "The Verge",
            "url": "https://www.theverge.com/rss/index.xml",
            "authority": 0.8,
        },
        {
            "name": "Wired",
            "url": "https://www.wired.com/feed/rss",
            "authority": 0.8,
        },
        {
            "name": "Ars Technica",
            "url": "https://feeds.arstechnica.com/arstechnica/index",
            "authority": 0.85,
        },
        {
            "name": "Engadget",
            "url": "https://www.engadget.com/rss.xml",
            "authority": 0.7,
        },
        {
            "name": "Krebs on Security",
            "url": "https://krebsonsecurity.com/feed/",
            "authority": 0.9,
        },
        {
            "name": "9to5Mac",
            "url": "https://9to5mac.com/feed/",
            "authority": 0.7,
        },
        {
            "name": "9to5Google",
            "url": "https://9to5google.com/feed/",
            "authority": 0.7,
        },
    ],

    TopicDomain.BUSINESS: [
        {
            "name": "Reuters Business",
            "url": "https://www.reuters.com/business/feed/",
            "authority": 0.95,
        },
        {
            "name": "Reuters Markets",
            "url": "https://www.reuters.com/markets/feed/",
            "authority": 0.95,
        },
        {
            "name": "CNBC Top News",
            "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "authority": 0.8,
        },
        {
            "name": "CNBC Markets",
            "url": "https://www.cnbc.com/id/10001147/device/rss/rss.html",
            "authority": 0.8,
        },
        {
            "name": "Fortune",
            "url": "https://fortune.com/feed/",
            "authority": 0.8,
        },
        {
            "name": "Harvard Business Review",
            "url": "https://hbr.org/feed",
            "authority": 0.9,
        },
        {
            "name": "Forbes",
            "url": "https://www.forbes.com/innovation/feed/",
            "authority": 0.7,
        },
        {
            "name": "Axios",
            "url": "https://api.axios.com/feed/",
            "authority": 0.75,
        },
    ],

    TopicDomain.WORLD: [
        {
            "name": "Reuters World",
            "url": "https://www.reuters.com/world/feed/",
            "authority": 0.95,
        },
        {
            "name": "AP World News",
            "url": "https://feeds.apnews.com/rss/world",
            "authority": 0.95,
        },
        {
            "name": "BBC World",
            "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
            "authority": 0.9,
        },
        {
            "name": "BBC Europe",
            "url": "https://feeds.bbci.co.uk/news/world/europe/rss.xml",
            "authority": 0.9,
        },
        {
            "name": "BBC Asia",
            "url": "https://feeds.bbci.co.uk/news/world/asia/rss.xml",
            "authority": 0.9,
        },
        {
            "name": "Al Jazeera",
            "url": "https://www.aljazeera.com/xml/rss/all.xml",
            "authority": 0.8,
        },
        {
            "name": "The Guardian World",
            "url": "https://www.theguardian.com/world/rss",
            "authority": 0.85,
        },
        {
            "name": "Foreign Policy",
            "url": "https://foreignpolicy.com/feed/",
            "authority": 0.85,
        },
        {
            "name": "War on the Rocks",
            "url": "https://warontherocks.com/feed/",
            "authority": 0.8,
        },
    ],

    TopicDomain.ENVIRONMENT: [
        {
            "name": "Carbon Brief",
            "url": "https://www.carbonbrief.org/feed/",
            "authority": 0.9,
        },
        {
            "name": "The Guardian Environment",
            "url": "https://www.theguardian.com/environment/rss",
            "authority": 0.85,
        },
        {
            "name": "Yale Environment 360",
            "url": "https://e360.yale.edu/feed",
            "authority": 0.9,
        },
        {
            "name": "Grist",
            "url": "https://grist.org/feed/",
            "authority": 0.8,
        },
        {
            "name": "Inside Climate News",
            "url": "https://insideclimatenews.org/feed/",
            "authority": 0.85,
        },
        {
            "name": "Nature Climate Change",
            "url": "https://www.nature.com/nclimate.rss",
            "authority": 0.95,
        },
        {
            "name": "NASA Climate",
            "url": "https://climate.nasa.gov/feed/news",
            "authority": 0.9,
        },
        {
            "name": "Mongabay",
            "url": "https://news.mongabay.com/feed/",
            "authority": 0.8,
        },
        {
            "name": "Canary Media",
            "url": "https://www.canarymedia.com/feed/",
            "authority": 0.75,
        },
    ],
}

# XML namespaces for Atom feeds
ATOM_NS = "{http://www.w3.org/2005/Atom}"
DC_NS = "{http://purl.org/dc/elements/1.1/}"
CONTENT_NS = "{http://purl.org/rss/1.0/modules/content/}"


def create_rss_config() -> SourceConfig:
    """Create default RSS source configuration."""
    return SourceConfig(
        name="rss_aggregator",
        source_type=SourceType.RSS_FEED,
        base_url="",  # Multiple URLs
        rate_limit_requests=10,
        rate_limit_period=1,
        topics=list(TopicDomain),
        priority=3,  # Lower than academic sources
    )


class RSSSource(BaseSource):
    """
    RSS/Atom feed aggregator.

    Fetches from multiple RSS feeds and normalizes to RawArticle format.
    """

    def __init__(self, config: Optional[SourceConfig] = None):
        super().__init__(config or create_rss_config())
        self.rate_limiter = get_rate_limiter()

    async def fetch_recent(
        self,
        max_results: int = 100,
        since: Optional[datetime] = None,
    ) -> list[RawArticle]:
        """
        Fetch recent articles from all configured RSS feeds.

        Args:
            max_results: Maximum total articles to return
            since: Only articles after this date

        Returns:
            List of RawArticle objects, sorted by date
        """
        all_articles = []

        # Fetch from all topic feeds concurrently
        tasks = []
        for topic in TopicDomain:
            tasks.append(self._fetch_topic_feeds(topic, max_results // len(TopicDomain)))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"RSS fetch error: {result}")
                continue
            all_articles.extend(result)

        # Filter by date if specified
        if since:
            all_articles = [a for a in all_articles if a.published_at and a.published_at > since]

        # Sort by date and limit
        all_articles.sort(key=lambda a: a.published_at or datetime.min, reverse=True)
        return all_articles[:max_results]

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
        return await self._fetch_topic_feeds(topic, max_results)

    async def _fetch_topic_feeds(
        self,
        topic: TopicDomain,
        max_results: int,
    ) -> list[RawArticle]:
        """Fetch from all feeds for a topic."""
        feeds = RSS_FEEDS.get(topic, [])
        if not feeds:
            return []

        # Fetch all feeds concurrently
        tasks = [self._fetch_feed(feed, topic) for feed in feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Feed fetch error: {result}")
                continue
            all_articles.extend(result)

        # Sort by date and limit per feed to ensure diversity
        all_articles.sort(key=lambda a: a.published_at or datetime.min, reverse=True)
        return all_articles[:max_results]

    async def _fetch_feed(
        self,
        feed_config: dict,
        topic: TopicDomain,
    ) -> list[RawArticle]:
        """Fetch and parse a single RSS feed."""
        await self.rate_limiter.wait_if_needed("rss")

        url = feed_config["url"]
        name = feed_config["name"]
        authority = feed_config.get("authority", 0.5)

        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(url, headers={
                    "User-Agent": "TheNoiselessNewspaper/1.0 (Academic RSS Aggregator)"
                })
                response.raise_for_status()

            # Detect feed type and parse
            content = response.text
            if "<feed" in content[:500]:
                articles = self._parse_atom(content, name, authority, topic)
            else:
                articles = self._parse_rss(content, name, authority, topic)

            logger.debug(f"Fetched {len(articles)} articles from {name}")
            return articles

        except httpx.HTTPError as e:
            logger.warning(f"HTTP error fetching {name}: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error fetching {name}: {e}")
            return []

    def _parse_rss(
        self,
        xml_content: str,
        source_name: str,
        authority: float,
        topic: TopicDomain,
    ) -> list[RawArticle]:
        """Parse RSS 2.0 feed."""
        articles = []

        try:
            root = ElementTree.fromstring(xml_content)
        except ElementTree.ParseError as e:
            logger.error(f"Failed to parse RSS from {source_name}: {e}")
            return []

        for item in root.findall(".//item"):
            try:
                article = self._parse_rss_item(item, source_name, authority, topic)
                if article:
                    articles.append(article)
            except Exception as e:
                logger.warning(f"Failed to parse RSS item from {source_name}: {e}")
                continue

        return articles

    def _parse_rss_item(
        self,
        item: ElementTree.Element,
        source_name: str,
        authority: float,
        topic: TopicDomain,
    ) -> Optional[RawArticle]:
        """Parse a single RSS item."""
        title = item.findtext("title", "").strip()
        if not title:
            return None

        link = item.findtext("link", "").strip()
        if not link:
            return None

        # Generate external ID from URL
        external_id = hashlib.md5(link.encode()).hexdigest()[:16]

        # Description/summary
        description = item.findtext("description", "")
        content = item.findtext(f"{CONTENT_NS}encoded", "")
        abstract = self._clean_html(content or description)

        # Author
        author = item.findtext("author") or item.findtext(f"{DC_NS}creator")
        authors = [author] if author else []

        # Publication date
        pub_date_str = item.findtext("pubDate")
        published_at = self._parse_rss_date(pub_date_str)

        # Categories
        categories = [cat.text for cat in item.findall("category") if cat.text]

        return RawArticle(
            external_id=external_id,
            source_name=source_name,
            title=title,
            url=link,
            abstract=abstract[:2000] if abstract else None,
            authors=authors,
            published_at=published_at,
            topics=categories,
            matched_domains=[topic],
            peer_reviewed=False,
            source_authority_score=authority,
        )

    def _parse_atom(
        self,
        xml_content: str,
        source_name: str,
        authority: float,
        topic: TopicDomain,
    ) -> list[RawArticle]:
        """Parse Atom feed."""
        articles = []

        try:
            root = ElementTree.fromstring(xml_content)
        except ElementTree.ParseError as e:
            logger.error(f"Failed to parse Atom from {source_name}: {e}")
            return []

        for entry in root.findall(f"{ATOM_NS}entry"):
            try:
                article = self._parse_atom_entry(entry, source_name, authority, topic)
                if article:
                    articles.append(article)
            except Exception as e:
                logger.warning(f"Failed to parse Atom entry from {source_name}: {e}")
                continue

        return articles

    def _parse_atom_entry(
        self,
        entry: ElementTree.Element,
        source_name: str,
        authority: float,
        topic: TopicDomain,
    ) -> Optional[RawArticle]:
        """Parse a single Atom entry."""
        title = entry.findtext(f"{ATOM_NS}title", "").strip()
        if not title:
            return None

        # Get link
        link = None
        for link_elem in entry.findall(f"{ATOM_NS}link"):
            rel = link_elem.get("rel", "alternate")
            if rel == "alternate":
                link = link_elem.get("href")
                break
        if not link:
            link = entry.findtext(f"{ATOM_NS}id", "")
        if not link:
            return None

        # Generate external ID
        entry_id = entry.findtext(f"{ATOM_NS}id", link)
        external_id = hashlib.md5(entry_id.encode()).hexdigest()[:16]

        # Summary/content
        summary = entry.findtext(f"{ATOM_NS}summary", "")
        content = entry.findtext(f"{ATOM_NS}content", "")
        abstract = self._clean_html(content or summary)

        # Authors
        authors = []
        for author in entry.findall(f"{ATOM_NS}author"):
            name = author.findtext(f"{ATOM_NS}name")
            if name:
                authors.append(name)

        # Publication date
        published_str = entry.findtext(f"{ATOM_NS}published")
        updated_str = entry.findtext(f"{ATOM_NS}updated")
        published_at = self._parse_atom_date(published_str or updated_str)

        # Categories
        categories = []
        for cat in entry.findall(f"{ATOM_NS}category"):
            term = cat.get("term") or cat.get("label")
            if term:
                categories.append(term)

        return RawArticle(
            external_id=external_id,
            source_name=source_name,
            title=title,
            url=link,
            abstract=abstract[:2000] if abstract else None,
            authors=authors,
            published_at=published_at,
            topics=categories,
            matched_domains=[topic],
            peer_reviewed=False,
            source_authority_score=authority,
        )

    def _clean_html(self, html: str) -> str:
        """Strip HTML tags from content."""
        if not html:
            return ""

        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', ' ', html)
        # Normalize whitespace
        clean = ' '.join(clean.split())
        # Decode common entities
        clean = clean.replace('&amp;', '&')
        clean = clean.replace('&lt;', '<')
        clean = clean.replace('&gt;', '>')
        clean = clean.replace('&quot;', '"')
        clean = clean.replace('&#39;', "'")
        clean = clean.replace('&nbsp;', ' ')

        return clean.strip()

    def _parse_rss_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse RSS date format (RFC 822)."""
        if not date_str:
            return None

        try:
            return parsedate_to_datetime(date_str)
        except (ValueError, TypeError):
            pass

        # Try ISO format as fallback
        return self._parse_atom_date(date_str)

    def _parse_atom_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse Atom/ISO date format."""
        if not date_str:
            return None

        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            pass

        try:
            # Try without timezone
            return datetime.fromisoformat(date_str[:19])
        except ValueError:
            return None

    def map_to_domain(self, source_categories: list[str]) -> list[TopicDomain]:
        """
        RSS feeds are pre-mapped to domains during fetch.
        This method is not typically used.
        """
        return []
