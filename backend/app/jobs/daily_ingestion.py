"""
Daily batch job for article ingestion and scoring.

This job runs once per day (configurable) and:
1. Fetches new articles from all sources
2. Classifies articles into topics
3. Generates summaries
4. Computes embeddings
5. Fetches citation data
6. Rebuilds the PageRank graph
7. Updates all article scores
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.taxonomy import TOPIC_TAXONOMY, get_all_topic_paths
from app.models.database import Database, DBArticle
from app.models.domain import Article, TopicPath
from app.services.citation_graph import CitationGraphService
from app.services.embeddings import (
    ArticleIndexService,
    EmbeddingService,
    TopicEmbeddingService,
)
from app.services.ranking import RankingService, VoteAggregationService
from app.services.summarization import SummarizationService
from app.sources import (
    ArxivAdapter,
    MockAdapter,
    NewsAPIAdapter,
    OpenAlexAdapter,
    SemanticScholarAdapter,
)
from app.sources.base import ArticleSourceAdapter

logger = structlog.get_logger()


class DailyIngestionJob:
    """
    Orchestrates the daily article ingestion pipeline.

    Pipeline stages:
    1. Fetch articles from all sources for all topics
    2. Deduplicate (same article from multiple sources)
    3. Classify into topics using embeddings
    4. Generate summaries using LLM
    5. Compute embeddings for all new articles
    6. Fetch and store citation relationships
    7. Rebuild PageRank graph and compute scores
    8. Update final scores for all articles
    9. Aggregate vote statistics
    """

    def __init__(self, database: Database):
        self.database = database
        self.settings = get_settings()

        # Initialize services
        self.embedding_service = EmbeddingService()
        self.topic_embedding_service = TopicEmbeddingService(self.embedding_service)
        self.citation_service = CitationGraphService()
        self.summarization_service = SummarizationService()
        self.vote_aggregation_service = VoteAggregationService()
        self.article_index = ArticleIndexService(self.embedding_service)

        # Initialize source adapters
        self.sources: list[ArticleSourceAdapter] = []

    async def initialize(self):
        """Initialize all services and adapters."""
        logger.info("Initializing daily ingestion job")

        # Initialize embedding service
        await self.embedding_service.initialize()
        logger.info("Embedding service initialized")

        # Initialize topic embeddings
        await self.topic_embedding_service.initialize_topic_embeddings(TOPIC_TAXONOMY)
        logger.info("Topic embeddings initialized", num_topics=len(get_all_topic_paths()))

        # Initialize summarization
        await self.summarization_service.initialize()
        logger.info("Summarization service initialized")

        # Initialize source adapters based on available API keys
        self.sources = []

        # Always include mock adapter for development
        self.sources.append(MockAdapter())

        # Academic sources (free)
        self.sources.append(ArxivAdapter())
        self.sources.append(OpenAlexAdapter())

        # Semantic Scholar (free tier or API key)
        self.sources.append(SemanticScholarAdapter())

        # News sources (require API key)
        if self.settings.newsapi_key:
            self.sources.append(NewsAPIAdapter())

        logger.info("Source adapters initialized", sources=[s.name for s in self.sources])

    async def run(self):
        """Execute the full daily ingestion pipeline."""
        start_time = datetime.utcnow()
        logger.info("Starting daily ingestion job", start_time=start_time.isoformat())

        stats = {
            "articles_fetched": 0,
            "articles_new": 0,
            "articles_updated": 0,
            "summaries_generated": 0,
            "citations_added": 0,
            "errors": [],
        }

        try:
            async with self.database.async_session() as session:
                # Stage 1: Fetch articles
                articles = await self._fetch_all_articles(session, stats)
                logger.info("Articles fetched", count=len(articles))

                # Stage 2: Store new articles
                new_articles = await self._store_articles(session, articles, stats)
                logger.info("New articles stored", count=len(new_articles))

                # Stage 3: Generate summaries for new articles
                await self._generate_summaries(session, new_articles, stats)
                logger.info("Summaries generated", count=stats["summaries_generated"])

                # Stage 4: Compute embeddings
                await self._compute_embeddings(session, new_articles, stats)
                logger.info("Embeddings computed")

                # Stage 5: Fetch citations
                await self._fetch_citations(session, new_articles, stats)
                logger.info("Citations fetched", count=stats["citations_added"])

                # Stage 6: Rebuild PageRank
                await self._rebuild_pagerank(session, stats)
                logger.info("PageRank rebuilt")

                # Stage 7: Update final scores
                await self._update_scores(session, stats)
                logger.info("Scores updated")

                # Stage 8: Aggregate votes
                await self._aggregate_votes(session, stats)
                logger.info("Votes aggregated")

        except Exception as e:
            logger.error("Daily ingestion job failed", error=str(e))
            stats["errors"].append(str(e))
            raise

        elapsed = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            "Daily ingestion job completed",
            elapsed_seconds=elapsed,
            stats=stats,
        )

        return stats

    async def _fetch_all_articles(
        self,
        session: AsyncSession,
        stats: dict,
    ) -> list[Article]:
        """Fetch articles from all sources for all topics."""
        all_articles = []
        seen_ids = set()

        # Get date range (last 7 days for daily job)
        since = datetime.utcnow() - timedelta(days=7)

        for topic_path in get_all_topic_paths():
            topic = TopicPath.from_path(topic_path)

            # Get keywords for this topic from taxonomy
            keywords = self._get_topic_keywords(topic_path)

            for source in self.sources:
                try:
                    async for article in source.search(
                        topic=topic,
                        keywords=keywords,
                        since=since,
                        max_results=self.settings.articles_per_topic_per_source,
                    ):
                        # Deduplicate
                        if article.id not in seen_ids:
                            article.topic_path = topic_path
                            all_articles.append(article)
                            seen_ids.add(article.id)
                            stats["articles_fetched"] += 1

                except Exception as e:
                    logger.warning(
                        "Error fetching from source",
                        source=source.name,
                        topic=topic_path,
                        error=str(e),
                    )
                    stats["errors"].append(f"{source.name}/{topic_path}: {e}")

        return all_articles

    async def _store_articles(
        self,
        session: AsyncSession,
        articles: list[Article],
        stats: dict,
    ) -> list[Article]:
        """Store articles in database, returning only new ones."""
        new_articles = []

        for article in articles:
            # Check if article already exists
            result = await session.execute(
                select(DBArticle).where(DBArticle.id == article.id)
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update if newer data
                if article.citation_count > existing.citation_count:
                    existing.citation_count = article.citation_count
                    stats["articles_updated"] += 1
            else:
                # Create new article
                db_article = DBArticle(
                    id=article.id,
                    source=article.source.value,
                    content_type=article.content_type.value,
                    title=article.title,
                    abstract=article.abstract,
                    authors_json=json.dumps([a.model_dump() for a in article.authors]),
                    published_at=article.published_at,
                    url=article.url,
                    topic_path=article.topic_path,
                    citation_count=article.citation_count,
                )
                session.add(db_article)
                new_articles.append(article)
                stats["articles_new"] += 1

        await session.commit()
        return new_articles

    async def _generate_summaries(
        self,
        session: AsyncSession,
        articles: list[Article],
        stats: dict,
    ):
        """Generate summaries for articles that don't have one."""
        articles_needing_summary = [
            a for a in articles
            if a.abstract and not a.summary
        ]

        if not articles_needing_summary:
            return

        summaries = await self.summarization_service.summarize_batch(
            articles_needing_summary
        )

        for article_id, summary in summaries.items():
            result = await session.execute(
                select(DBArticle).where(DBArticle.id == article_id)
            )
            db_article = result.scalar_one_or_none()
            if db_article:
                db_article.summary = summary
                stats["summaries_generated"] += 1

        await session.commit()

    async def _compute_embeddings(
        self,
        session: AsyncSession,
        articles: list[Article],
        stats: dict,
    ):
        """Compute and store embeddings for new articles."""
        if not articles:
            return

        # Prepare texts for embedding
        article_texts = [
            f"{a.title} {a.abstract or ''}"
            for a in articles
        ]

        # Batch compute embeddings
        embeddings = await self.embedding_service.embed_texts(article_texts)

        # Store embeddings and add to index
        for article, embedding in zip(articles, embeddings):
            result = await session.execute(
                select(DBArticle).where(DBArticle.id == article.id)
            )
            db_article = result.scalar_one_or_none()
            if db_article:
                db_article.embedding_json = json.dumps(embedding.tolist())

            # Also compute topic relevance
            if article.topic_path:
                topic_embedding = self.topic_embedding_service.get_topic_embedding(
                    article.topic_path
                )
                if topic_embedding is not None:
                    relevance = EmbeddingService.cosine_similarity(embedding, topic_embedding)
                    if db_article:
                        db_article.topic_relevance_score = max(0, relevance)

        await session.commit()

        # Add to in-memory index
        await self.article_index.add_articles_batch([
            (a.id, f"{a.title} {a.abstract or ''}")
            for a in articles
        ])

    async def _fetch_citations(
        self,
        session: AsyncSession,
        articles: list[Article],
        stats: dict,
    ):
        """Fetch citation data for new articles."""
        # Use Semantic Scholar for citation data
        s2_adapter = SemanticScholarAdapter()

        for article in articles:
            try:
                citing_ids, cited_ids = await s2_adapter.get_citations(article.id)

                if citing_ids or cited_ids:
                    added = await self.citation_service.add_citations(
                        session, article.id, citing_ids, cited_ids
                    )
                    stats["citations_added"] += added

            except Exception as e:
                logger.debug(
                    "Error fetching citations",
                    article_id=article.id,
                    error=str(e),
                )

            # Rate limiting
            await asyncio.sleep(0.5)

    async def _rebuild_pagerank(
        self,
        session: AsyncSession,
        stats: dict,
    ):
        """Rebuild the citation graph and compute PageRank scores."""
        # Build graph from database
        graph = await self.citation_service.build_graph(session)

        if graph.number_of_nodes() > 0:
            # Compute PageRank
            self.citation_service.compute_pagerank(graph)

            # Update scores in database
            await self.citation_service.update_article_scores(session)

            # Log graph stats
            graph_stats = self.citation_service.analyze_graph()
            logger.info("Citation graph stats", **graph_stats)

    async def _update_scores(
        self,
        session: AsyncSession,
        stats: dict,
    ):
        """Update final scores for all articles."""
        # Get all articles
        result = await session.execute(select(DBArticle))
        db_articles = result.scalars().all()

        for db_article in db_articles:
            # Compute recency score
            ranking_service = RankingService(
                self.citation_service,
                self.embedding_service,
                self.topic_embedding_service,
            )
            recency_score = ranking_service.compute_recency_score(db_article.published_at)
            db_article.recency_score = recency_score

            # Compute simple final score (full ranking done at query time)
            # This is a pre-computed approximation for sorting
            db_article.final_score = (
                0.35 * db_article.pagerank_score
                + 0.20 * recency_score
                + 0.30 * db_article.topic_relevance_score
                + 0.15 * 0  # Vote score computed at query time
            )
            db_article.scored_at = datetime.utcnow()

        await session.commit()

    async def _aggregate_votes(
        self,
        session: AsyncSession,
        stats: dict,
    ):
        """Aggregate vote statistics."""
        updated = await self.vote_aggregation_service.aggregate_votes(session)
        logger.info("Vote statistics aggregated", articles_updated=updated)

    def _get_topic_keywords(self, topic_path: str) -> list[str]:
        """Get keywords for a topic from the taxonomy."""
        parts = topic_path.split("/")
        if len(parts) != 3:
            return []

        cat_id, sub_id, niche_id = parts
        category = TOPIC_TAXONOMY.get(cat_id, {})
        subtopic = category.get("subtopics", {}).get(sub_id, {})
        niche = subtopic.get("niches", {}).get(niche_id, {})

        # Combine names as keywords
        keywords = [
            category.get("name", ""),
            subtopic.get("name", ""),
            niche.get("name", ""),
        ]

        # Add explicit keywords if defined
        keywords.extend(niche.get("keywords", []))

        return [k for k in keywords if k]


async def run_daily_job(database_url: Optional[str] = None):
    """Entry point for running the daily ingestion job."""
    settings = get_settings()
    database = Database(database_url or settings.database_url)

    # Create tables if they don't exist
    await database.create_tables()

    # Initialize and run job
    job = DailyIngestionJob(database)
    await job.initialize()
    stats = await job.run()

    return stats


if __name__ == "__main__":
    # Run job directly for testing
    asyncio.run(run_daily_job())
