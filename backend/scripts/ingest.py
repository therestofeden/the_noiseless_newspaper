#!/usr/bin/env python3
"""
CLI tool for data ingestion.

Usage:
    # Fetch all sources
    python -m scripts.ingest fetch --all

    # Fetch specific topic
    python -m scripts.ingest fetch --topic ai-ml

    # Check source health
    python -m scripts.ingest health

    # Show source stats
    python -m scripts.ingest stats

    # Run scheduler (continuous)
    python -m scripts.ingest serve --interval 6
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from app.services.data_ingestion import (
    SourceAggregator,
    TopicDomain,
    RawArticle,
)
from app.services.data_ingestion.scheduler import IngestionScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_aggregator() -> SourceAggregator:
    """Create source aggregator with environment config."""
    import os

    return SourceAggregator(
        arxiv_enabled=True,
        pubmed_enabled=True,
        rss_enabled=True,
        semantic_scholar_enabled=True,
        semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
        pubmed_api_key=os.getenv("PUBMED_API_KEY"),
    )


async def cmd_fetch(args):
    """Fetch articles from sources."""
    aggregator = create_aggregator()

    if args.topic:
        try:
            topic = TopicDomain(args.topic)
        except ValueError:
            print(f"Invalid topic: {args.topic}")
            print(f"Valid topics: {[t.value for t in TopicDomain]}")
            return 1

        print(f"Fetching articles for topic: {topic.value}")
        articles, results = await aggregator.fetch_by_topic(
            topic, max_per_source=args.limit
        )
    else:
        print("Fetching articles from all sources...")
        articles, results = await aggregator.fetch_all(
            max_per_source=args.limit
        )

    # Print results
    print("\n" + "=" * 60)
    print("INGESTION RESULTS")
    print("=" * 60)

    for result in results:
        print(result)

    print("-" * 60)
    print(f"Total articles: {len(articles)}")

    # Output articles if requested
    if args.output:
        output_data = [
            {
                "external_id": a.external_id,
                "source": a.source_name,
                "title": a.title,
                "url": a.url,
                "abstract": a.abstract[:500] if a.abstract else None,
                "authors": a.authors,
                "published_at": a.published_at.isoformat() if a.published_at else None,
                "topics": a.topics,
                "domains": [d.value for d in a.matched_domains],
                "doi": a.doi,
                "citation_count": a.citation_count,
                "authority_score": a.source_authority_score,
            }
            for a in articles
        ]

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nArticles saved to: {args.output}")

    # Preview articles if verbose
    if args.verbose:
        print("\n" + "=" * 60)
        print("SAMPLE ARTICLES")
        print("=" * 60)

        for article in articles[:10]:
            print(f"\n[{article.source_name}] {article.title}")
            print(f"  URL: {article.url}")
            print(f"  Date: {article.published_at}")
            print(f"  Topics: {', '.join(d.value for d in article.matched_domains)}")
            if article.doi:
                print(f"  DOI: {article.doi}")

    return 0


async def cmd_health(args):
    """Check health of all sources."""
    aggregator = create_aggregator()

    print("Checking source health...")
    health = await aggregator.health_check()

    print("\n" + "=" * 40)
    print("SOURCE HEALTH")
    print("=" * 40)

    all_healthy = True
    for source, is_healthy in health.items():
        status = "✓ OK" if is_healthy else "✗ FAILED"
        print(f"  {source}: {status}")
        if not is_healthy:
            all_healthy = False

    return 0 if all_healthy else 1


async def cmd_stats(args):
    """Show source statistics."""
    aggregator = create_aggregator()
    stats = aggregator.get_source_stats()

    print("\n" + "=" * 50)
    print("SOURCE CONFIGURATION")
    print("=" * 50)
    print(f"Total sources: {stats['total_sources']}")
    print()

    for source in stats["sources"]:
        print(f"  {source['name']}")
        print(f"    Type: {source['type']}")
        print(f"    Topics: {', '.join(source['topics'])}")
        print(f"    Priority: {source['priority']}")
        print()

    return 0


async def cmd_serve(args):
    """Run continuous scheduler."""
    aggregator = create_aggregator()

    async def on_articles(articles: list[RawArticle]):
        """Callback when articles are fetched."""
        logger.info(f"Received {len(articles)} articles")

        # In a real implementation, this would save to database
        # For now, just log
        for article in articles[:5]:
            logger.info(f"  - [{article.source_name}] {article.title[:60]}...")

    scheduler = IngestionScheduler(
        aggregator=aggregator,
        fetch_interval_hours=args.interval,
        on_articles_fetched=on_articles,
    )

    print(f"Starting scheduler (fetch every {args.interval} hours)")
    print("Press Ctrl+C to stop")

    try:
        await scheduler.start()

        # Keep running until interrupted
        while scheduler.is_running:
            await asyncio.sleep(60)

            # Log status periodically
            status = scheduler.get_status()
            if status["last_fetch"]:
                logger.debug(f"Last fetch: {status['last_fetch']}")

    except KeyboardInterrupt:
        print("\nShutting down...")
        await scheduler.stop()

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="The Noiseless Newspaper - Data Ingestion CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch articles")
    fetch_parser.add_argument(
        "--topic", "-t",
        help="Specific topic to fetch (e.g., ai-ml, physics)"
    )
    fetch_parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Fetch from all sources"
    )
    fetch_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=50,
        help="Max articles per source (default: 50)"
    )
    fetch_parser.add_argument(
        "--output", "-o",
        help="Output file for articles (JSON)"
    )
    fetch_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show article previews"
    )

    # Health command
    health_parser = subparsers.add_parser("health", help="Check source health")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show source statistics")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Run continuous scheduler")
    serve_parser.add_argument(
        "--interval", "-i",
        type=int,
        default=6,
        help="Fetch interval in hours (default: 6)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Run command
    if args.command == "fetch":
        return asyncio.run(cmd_fetch(args))
    elif args.command == "health":
        return asyncio.run(cmd_health(args))
    elif args.command == "stats":
        return asyncio.run(cmd_stats(args))
    elif args.command == "serve":
        return asyncio.run(cmd_serve(args))

    return 0


if __name__ == "__main__":
    sys.exit(main())
