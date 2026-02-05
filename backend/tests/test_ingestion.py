"""
Tests for data ingestion services.

These tests use mocked HTTP responses to verify parsing logic
without requiring network access.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, '.')

from app.services.data_ingestion.base import TopicDomain, RawArticle
from app.services.data_ingestion.arxiv import ArxivSource
from app.services.data_ingestion.rss import RSSSource
from app.services.data_ingestion.aggregator import SourceAggregator


# Sample arXiv API response
SAMPLE_ARXIV_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>Attention Is All You Need: A Comprehensive Survey</title>
    <summary>We present a comprehensive survey of attention mechanisms in deep learning, covering their applications in natural language processing, computer vision, and beyond.</summary>
    <author><name>John Smith</name></author>
    <author><name>Jane Doe</name></author>
    <published>2024-01-15T12:00:00Z</published>
    <category term="cs.LG"/>
    <category term="cs.AI"/>
    <link href="http://arxiv.org/abs/2401.12345v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v1" rel="related" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.67890v1</id>
    <title>Quantum Computing for Machine Learning</title>
    <summary>This paper explores the intersection of quantum computing and machine learning algorithms.</summary>
    <author><name>Alice Quantum</name></author>
    <published>2024-01-14T10:00:00Z</published>
    <category term="quant-ph"/>
    <category term="cs.LG"/>
    <link href="http://arxiv.org/abs/2401.67890v1" rel="alternate" type="text/html"/>
  </entry>
</feed>
"""

# Sample RSS feed response
SAMPLE_RSS_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>MIT Technology Review - AI</title>
    <item>
      <title>The Future of Large Language Models</title>
      <link>https://example.com/article/llm-future</link>
      <description>A deep dive into where LLMs are headed in 2024.</description>
      <pubDate>Mon, 15 Jan 2024 09:00:00 GMT</pubDate>
      <author>tech@example.com</author>
      <category>AI</category>
    </item>
    <item>
      <title>AI Regulation: What to Expect</title>
      <link>https://example.com/article/ai-regulation</link>
      <description>Governments worldwide are grappling with AI policy.</description>
      <pubDate>Sun, 14 Jan 2024 15:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


class TestArxivSource:
    """Tests for arXiv source."""

    def test_parse_atom_feed(self):
        """Test parsing of arXiv Atom feed."""
        source = ArxivSource()
        articles = source._parse_atom_feed(SAMPLE_ARXIV_RESPONSE)

        assert len(articles) == 2

        # Check first article
        article = articles[0]
        assert article.title == "Attention Is All You Need: A Comprehensive Survey"
        assert article.arxiv_id == "2401.12345"
        assert "John Smith" in article.authors
        assert "Jane Doe" in article.authors
        assert article.url == "https://arxiv.org/abs/2401.12345"
        assert TopicDomain.AI_ML in article.matched_domains

        # Check second article
        article2 = articles[1]
        assert article2.arxiv_id == "2401.67890"
        assert TopicDomain.PHYSICS in article2.matched_domains  # quant-ph

    def test_category_mapping(self):
        """Test arXiv category to domain mapping."""
        source = ArxivSource()

        # AI/ML categories
        domains = source.map_to_domain(["cs.AI", "cs.LG"])
        assert TopicDomain.AI_ML in domains

        # Physics categories
        domains = source.map_to_domain(["quant-ph", "hep-th"])
        assert TopicDomain.PHYSICS in domains

        # Mixed categories
        domains = source.map_to_domain(["cs.LG", "quant-ph"])
        assert TopicDomain.AI_ML in domains
        assert TopicDomain.PHYSICS in domains


class TestRSSSource:
    """Tests for RSS feed source."""

    def test_parse_rss(self):
        """Test parsing of RSS feed."""
        source = RSSSource()
        articles = source._parse_rss(
            SAMPLE_RSS_RESPONSE,
            "MIT Technology Review",
            0.8,
            TopicDomain.AI_ML
        )

        assert len(articles) == 2

        # Check first article
        article = articles[0]
        assert article.title == "The Future of Large Language Models"
        assert "example.com" in article.url
        assert article.source_authority_score == 0.8
        assert TopicDomain.AI_ML in article.matched_domains


class TestAggregator:
    """Tests for source aggregator."""

    def test_url_normalization(self):
        """Test URL normalization for deduplication."""
        aggregator = SourceAggregator(
            arxiv_enabled=False,
            pubmed_enabled=False,
            rss_enabled=False,
            semantic_scholar_enabled=False,
        )

        # URLs with tracking params should normalize to same value
        url1 = "https://example.com/article?id=123&utm_source=twitter"
        url2 = "https://example.com/article?id=123&utm_campaign=test"
        url3 = "https://example.com/article?id=123"

        assert aggregator._normalize_url(url1) == aggregator._normalize_url(url3)
        assert aggregator._normalize_url(url2) == aggregator._normalize_url(url3)

    def test_title_normalization(self):
        """Test title normalization for fuzzy matching."""
        aggregator = SourceAggregator(
            arxiv_enabled=False,
            pubmed_enabled=False,
            rss_enabled=False,
            semantic_scholar_enabled=False,
        )

        title1 = "The Future of AI: A Survey"
        title2 = "The Future of AI:  A Survey"  # Extra space
        title3 = "THE FUTURE OF AI: A SURVEY"  # Uppercase

        norm1 = aggregator._normalize_title(title1)
        norm2 = aggregator._normalize_title(title2)
        norm3 = aggregator._normalize_title(title3)

        assert norm1 == norm2 == norm3

    def test_deduplication(self):
        """Test article deduplication."""
        aggregator = SourceAggregator(
            arxiv_enabled=False,
            pubmed_enabled=False,
            rss_enabled=False,
            semantic_scholar_enabled=False,
        )

        # Create duplicate articles
        articles = [
            RawArticle(
                external_id="1",
                source_name="arxiv",
                title="Test Article",
                url="https://arxiv.org/abs/2401.12345",
                doi="10.1234/test",
                source_authority_score=0.9,
            ),
            RawArticle(
                external_id="2",
                source_name="semantic_scholar",
                title="Test Article",
                url="https://semanticscholar.org/paper/abc",
                doi="10.1234/test",  # Same DOI
                source_authority_score=0.7,
            ),
        ]

        deduplicated = aggregator._deduplicate(articles)

        # Should keep only one, preferring higher authority
        assert len(deduplicated) == 1
        assert deduplicated[0].source_name == "arxiv"  # Higher authority


def run_tests():
    """Run all tests."""
    print("Running ingestion tests...")
    print()

    # arXiv tests
    print("=== ArxivSource Tests ===")
    arxiv_tests = TestArxivSource()
    try:
        arxiv_tests.test_parse_atom_feed()
        print("✓ test_parse_atom_feed")
    except AssertionError as e:
        print(f"✗ test_parse_atom_feed: {e}")

    try:
        arxiv_tests.test_category_mapping()
        print("✓ test_category_mapping")
    except AssertionError as e:
        print(f"✗ test_category_mapping: {e}")

    # RSS tests
    print()
    print("=== RSSSource Tests ===")
    rss_tests = TestRSSSource()
    try:
        rss_tests.test_parse_rss()
        print("✓ test_parse_rss")
    except AssertionError as e:
        print(f"✗ test_parse_rss: {e}")

    # Aggregator tests
    print()
    print("=== Aggregator Tests ===")
    agg_tests = TestAggregator()
    try:
        agg_tests.test_url_normalization()
        print("✓ test_url_normalization")
    except AssertionError as e:
        print(f"✗ test_url_normalization: {e}")

    try:
        agg_tests.test_title_normalization()
        print("✓ test_title_normalization")
    except AssertionError as e:
        print(f"✗ test_title_normalization: {e}")

    try:
        agg_tests.test_deduplication()
        print("✓ test_deduplication")
    except AssertionError as e:
        print(f"✗ test_deduplication: {e}")

    print()
    print("All tests completed!")


if __name__ == "__main__":
    run_tests()
