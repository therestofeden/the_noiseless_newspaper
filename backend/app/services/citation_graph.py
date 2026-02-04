"""
Citation graph service using NetworkX for PageRank computation.
This is the core of the cold-start ranking algorithm.
"""
import asyncio
from datetime import datetime
from typing import Optional

import networkx as nx
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import DBArticle, DBCitation
from app.models.domain import Article


class CitationGraphService:
    """
    Manages the citation graph and computes PageRank scores.

    The citation graph is a directed graph where:
    - Nodes are articles (identified by their ID)
    - Edges represent citations (A -> B means A cites B)

    PageRank gives higher scores to articles that are:
    1. Cited by many other articles
    2. Cited by influential articles (recursive authority)
    """

    def __init__(self):
        self.settings = get_settings()
        self._graph: Optional[nx.DiGraph] = None
        self._pagerank_scores: dict[str, float] = {}
        self._last_computed: Optional[datetime] = None

    async def build_graph(self, session: AsyncSession) -> nx.DiGraph:
        """
        Build the citation graph from the database.

        This loads all articles and citations into a NetworkX DiGraph.
        For large datasets, we might need to do this incrementally.
        """
        graph = nx.DiGraph()

        # Load all articles as nodes
        result = await session.execute(
            select(DBArticle.id, DBArticle.citation_count)
        )
        articles = result.all()

        for article_id, citation_count in articles:
            graph.add_node(article_id, citation_count=citation_count)

        # Load all citations as edges
        result = await session.execute(
            select(DBCitation.citing_article_id, DBCitation.cited_article_id)
        )
        citations = result.all()

        for citing_id, cited_id in citations:
            # Add edge: citing -> cited
            # PageRank will flow from citing to cited
            graph.add_edge(citing_id, cited_id)

        self._graph = graph
        return graph

    def compute_pagerank(
        self,
        graph: Optional[nx.DiGraph] = None,
        damping: Optional[float] = None,
        max_iter: Optional[int] = None,
    ) -> dict[str, float]:
        """
        Compute PageRank scores for all nodes in the graph.

        The damping factor (typically 0.85) represents the probability
        that a random walker continues following links vs. jumping
        to a random node.

        Higher damping = more weight on link structure
        Lower damping = more uniform distribution

        Returns:
            Dictionary mapping article IDs to PageRank scores
        """
        if graph is None:
            graph = self._graph

        if graph is None or len(graph) == 0:
            return {}

        damping = damping or self.settings.pagerank_damping
        max_iter = max_iter or self.settings.pagerank_iterations

        try:
            # NetworkX's pagerank function handles:
            # - Dangling nodes (no outgoing edges)
            # - Convergence detection
            # - Normalization (scores sum to 1)
            scores = nx.pagerank(
                graph,
                alpha=damping,
                max_iter=max_iter,
                tol=1e-8,
            )

            self._pagerank_scores = scores
            self._last_computed = datetime.utcnow()

            return scores

        except nx.PowerIterationFailedConvergence:
            # Fall back to simpler approach if convergence fails
            # This can happen with certain graph structures
            scores = nx.pagerank(
                graph,
                alpha=0.5,  # Lower damping for better convergence
                max_iter=max_iter * 2,
                tol=1e-6,
            )
            self._pagerank_scores = scores
            self._last_computed = datetime.utcnow()
            return scores

    async def update_article_scores(self, session: AsyncSession) -> int:
        """
        Update PageRank scores for all articles in the database.

        Returns:
            Number of articles updated
        """
        if not self._pagerank_scores:
            return 0

        updated = 0

        # Normalize scores to 0-1 range for easier interpretation
        max_score = max(self._pagerank_scores.values()) if self._pagerank_scores else 1
        min_score = min(self._pagerank_scores.values()) if self._pagerank_scores else 0
        score_range = max_score - min_score if max_score > min_score else 1

        for article_id, raw_score in self._pagerank_scores.items():
            # Normalize to 0-1
            normalized_score = (raw_score - min_score) / score_range

            # Update in database
            result = await session.execute(
                select(DBArticle).where(DBArticle.id == article_id)
            )
            article = result.scalar_one_or_none()

            if article:
                article.pagerank_score = normalized_score
                article.scored_at = datetime.utcnow()
                updated += 1

        await session.commit()
        return updated

    def get_score(self, article_id: str) -> float:
        """Get the PageRank score for a specific article."""
        return self._pagerank_scores.get(article_id, 0.0)

    def get_top_articles(self, n: int = 100) -> list[tuple[str, float]]:
        """Get the top N articles by PageRank score."""
        sorted_scores = sorted(
            self._pagerank_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_scores[:n]

    async def add_citations(
        self,
        session: AsyncSession,
        article_id: str,
        citing_ids: list[str],
        cited_ids: list[str],
    ) -> int:
        """
        Add citation relationships to the database.

        Args:
            article_id: The article we're adding citations for
            citing_ids: Articles that cite this article
            cited_ids: Articles that this article cites

        Returns:
            Number of new citations added
        """
        added = 0

        # Add citations where other articles cite this one
        for citing_id in citing_ids:
            try:
                citation = DBCitation(
                    citing_article_id=citing_id,
                    cited_article_id=article_id,
                )
                session.add(citation)
                added += 1
            except Exception:
                # Likely duplicate, skip
                pass

        # Add citations where this article cites others
        for cited_id in cited_ids:
            try:
                citation = DBCitation(
                    citing_article_id=article_id,
                    cited_article_id=cited_id,
                )
                session.add(citation)
                added += 1
            except Exception:
                pass

        if added > 0:
            await session.commit()

        return added

    def analyze_graph(self) -> dict:
        """
        Analyze the citation graph structure.
        Useful for debugging and understanding the data.
        """
        if not self._graph:
            return {"error": "Graph not built"}

        graph = self._graph

        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_dag": nx.is_directed_acyclic_graph(graph),
            "num_weakly_connected_components": nx.number_weakly_connected_components(graph),
            "num_strongly_connected_components": nx.number_strongly_connected_components(graph),
            "avg_in_degree": np.mean([d for n, d in graph.in_degree()]) if graph.number_of_nodes() > 0 else 0,
            "avg_out_degree": np.mean([d for n, d in graph.out_degree()]) if graph.number_of_nodes() > 0 else 0,
            "max_in_degree": max([d for n, d in graph.in_degree()]) if graph.number_of_nodes() > 0 else 0,
            "max_out_degree": max([d for n, d in graph.out_degree()]) if graph.number_of_nodes() > 0 else 0,
            "last_computed": self._last_computed.isoformat() if self._last_computed else None,
        }


class IncrementalPageRank:
    """
    Incremental PageRank for efficiently updating scores when new articles arrive.

    Instead of recomputing from scratch, we can:
    1. Add new nodes/edges to the existing graph
    2. Run a few PageRank iterations to propagate changes
    3. Only fully recompute periodically (e.g., daily batch)

    This is more efficient for real-time updates.
    """

    def __init__(self, base_service: CitationGraphService):
        self.base = base_service
        self._pending_additions: list[tuple[str, list[str], list[str]]] = []

    def queue_addition(
        self,
        article_id: str,
        citing_ids: list[str],
        cited_ids: list[str],
    ):
        """Queue a new article and its citations for incremental update."""
        self._pending_additions.append((article_id, citing_ids, cited_ids))

    async def apply_pending(self, session: AsyncSession) -> int:
        """
        Apply pending additions and do a quick PageRank update.

        This runs fewer iterations than a full computation,
        trading off accuracy for speed.
        """
        if not self._pending_additions:
            return 0

        graph = self.base._graph
        if graph is None:
            await self.base.build_graph(session)
            graph = self.base._graph

        # Add new nodes and edges
        for article_id, citing_ids, cited_ids in self._pending_additions:
            graph.add_node(article_id)

            for citing_id in citing_ids:
                if citing_id in graph:
                    graph.add_edge(citing_id, article_id)

            for cited_id in cited_ids:
                if cited_id in graph:
                    graph.add_edge(article_id, cited_id)

            # Also add to database
            await self.base.add_citations(session, article_id, citing_ids, cited_ids)

        # Quick PageRank update (fewer iterations)
        self.base.compute_pagerank(graph, max_iter=20)

        count = len(self._pending_additions)
        self._pending_additions = []

        return count
