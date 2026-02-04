"""
Article source adapters for The Noiseless Newspaper.
"""
from app.sources.arxiv import ArxivAdapter
from app.sources.base import ArticleSourceAdapter
from app.sources.mock import MockAdapter
from app.sources.newsapi import NewsAPIAdapter
from app.sources.openalex import OpenAlexAdapter
from app.sources.semantic_scholar import SemanticScholarAdapter

__all__ = [
    "ArticleSourceAdapter",
    "ArxivAdapter",
    "SemanticScholarAdapter",
    "OpenAlexAdapter",
    "NewsAPIAdapter",
    "MockAdapter",
]
