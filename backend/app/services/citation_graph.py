"""
Citation graph service using NetworkX for PageRank computation.

Builds a directed graph from article citations and computes
PageRank scores to measure article importance in the citation network.
"""

from typing import Sequence
from uuid import UUID

import networkx as nx
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import DBArticle, DBCitation

logger = structlog.get_logger(__name__)


class CitationGraphService:
    """Service for managing the citation graph and computing PageRank."""

    def __init__(self):
        """Initialize the citation graph service."""
        self._graph: nx.DiGraph | None = None
        self._pagerank_scores: dict[str, float] = {}
        self._is_dirty = True

    @property
    def graph(self) -> nx.DiGraph:
        """Get the citation graph, creating if necessary."""
        if self._graph is None:
            self._graph = nx.DiGraph()
        return self._graph

    def clear(self) -> None:
        """Clear the graph and scores."""
        self._graph = None
        self._pagerank_scores = {}
        self._is_dirty = True

    async def build_graph_from_db(self, session: AsyncSession) -> nx.DiGraph:
        """
        Build the citation graph from database records.

        Args:
            session: Async database session

        Returns:
            The built NetworkX DiGraph
        """
        logger.info("Building citation graph from database")

        # Clear existing graph
        self._graph = nx.DiGraph()

        # Fetch all articles
        articles_result = await session.execute(select(DBArticle.id))
        article_ids = [row[0] for row in articles_result.fetchall()]

        # Add all articles as nodes
        for article_id in article_ids:
            self._graph.add_node(article_id)

        logger.info("Added article nodes", count=len(article_ids))

        # Fetch all citations
        citations_result = await session.execute(
            select(DBCitation.citing_article_id, DBCitation.cited_article_id)
        )
        citations = citations_result.fetchall()

        # Add edges (citing -> cited)
        edge_count = 0
        for citing_id, cited_id in citations:
            # Only add edge if both nodes exist
            if self._graph.has_node(citing_id) and self._graph.has_node(cited_id):
                self._graph.add_edge(citing_id, cited_id)
                edge_count += 1

        logger.info(
            "Built citation graph",
            nodes=self._graph.number_of_nodes(),
            edges=edge_count,
        )

        self._is_dirty = True
        return self._graph

    def build_graph_from_articles(
        self,
        articles: Sequence[tuple[str, list[str]]],
    ) -> nx.DiGraph:
        """
        Build the citation graph from article data.

        Args:
            articles: Sequence of (article_id, citation_ids) tuples

        Returns:
            The built NetworkX DiGraph
        """
        self._graph = nx.DiGraph()

        # First pass: add all nodes
        article_ids = set()
        for article_id, _ in articles:
            article_ids.add(article_id)
            self._graph.add_node(article_id)

        # Second pass: add edges
        edge_count = 0
        for article_id, citation_ids in articles:
            for cited_id in citation_ids:
                # Only add edge if cited article exists in our corpus
                if cited_id in article_ids:
                    self._graph.add_edge(article_id, cited_id)
                    edge_count += 1

        logger.info(
            "Built citation graph from articles",
            nodes=len(article_ids),
            edges=edge_count,
        )

        self._is_dirty = True
        return self._graph

    def add_article(self, article_id: str, citation_ids: list[str] | None = None) -> None:
        """
        Add an article to the graph.

        Args:
            article_id: The article's ID
            citation_ids: Optional list of cited article IDs
        """
        self.graph.add_node(article_id)

        if citation_ids:
            for cited_id in citation_ids:
                if self.graph.has_node(cited_id):
                    self.graph.add_edge(article_id, cited_id)

        self._is_dirty = True

    def add_citation(self, citing_id: str, cited_id: str) -> None:
        """
        Add a citation edge to the graph.

        Args:
            citing_id: ID of the citing article
            cited_id: ID of the cited article
        """
        # Ensure both nodes exist
        if not self.graph.has_node(citing_id):
            self.graph.add_node(citing_id)
        if not self.graph.has_node(cited_id):
            self.graph.add_node(cited_id)

        self.graph.add_edge(citing_id, cited_id)
        self._is_dirty = True

    def remove_article(self, article_id: str) -> None:
        """
        Remove an article from the graph.

        Args:
            article_id: The article's ID
        """
        if self.graph.has_node(article_id):
            self.graph.remove_node(article_id)
            self._is_dirty = True

    def compute_pagerank(
        self,
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1.0e-6,
        personalization: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Compute PageRank scores for all articles.

        Uses NetworkX's PageRank implementation with configurable parameters.

        Args:
            alpha: Damping factor (probability of following a link)
            max_iter: Maximum iterations for convergence
            tol: Convergence tolerance
            personalization: Optional dict of node -> weight for personalized PageRank

        Returns:
            Dict mapping article_id to PageRank score
        """
        if not self._is_dirty and self._pagerank_scores:
            logger.debug("Returning cached PageRank scores")
            return self._pagerank_scores

        if self.graph.number_of_nodes() == 0:
            logger.warning("Empty citation graph, returning empty scores")
            self._pagerank_scores = {}
            self._is_dirty = False
            return self._pagerank_scores

        try:
            # Compute PageRank
            raw_scores = nx.pagerank(
                self.graph,
                alpha=alpha,
                max_iter=max_iter,
                tol=tol,
                personalization=personalization,
            )

            # Normalize scores to [0, 1] range
            if raw_scores:
                max_score = max(raw_scores.values())
                min_score = min(raw_scores.values())
                score_range = max_score - min_score

                if score_range > 0:
                    self._pagerank_scores = {
                        article_id: (score - min_score) / score_range
                        for article_id, score in raw_scores.items()
                    }
                else:
                    # All scores equal, normalize to 0.5
                    self._pagerank_scores = {
                        article_id: 0.5 for article_id in raw_scores
                    }
            else:
                self._pagerank_scores = {}

            self._is_dirty = False

            logger.info(
                "Computed PageRank scores",
                articles=len(self._pagerank_scores),
                max_score=max(self._pagerank_scores.values()) if self._pagerank_scores else 0,
            )

            return self._pagerank_scores

        except nx.PowerIterationFailedConvergence:
            logger.warning("PageRank failed to converge, using uniform scores")
            node_count = self.graph.number_of_nodes()
            self._pagerank_scores = {
                node: 1.0 / node_count for node in self.graph.nodes()
            }
            self._is_dirty = False
            return self._pagerank_scores

    def get_score(self, article_id: str | UUID) -> float:
        """
        Get the PageRank score for an article.

        Args:
            article_id: The article's ID

        Returns:
            PageRank score (0.0 if not found or not computed)
        """
        # Convert UUID to string if necessary
        str_id = str(article_id) if isinstance(article_id, UUID) else article_id

        if self._is_dirty:
            self.compute_pagerank()

        return self._pagerank_scores.get(str_id, 0.0)

    def get_scores_batch(self, article_ids: Sequence[str | UUID]) -> dict[UUID, float]:
        """
        Get PageRank scores for multiple articles.

        Args:
            article_ids: Sequence of article IDs

        Returns:
            Dict mapping UUID to PageRank score
        """
        if self._is_dirty:
            self.compute_pagerank()

        result = {}
        for article_id in article_ids:
            uuid_id = article_id if isinstance(article_id, UUID) else UUID(article_id)
            str_id = str(uuid_id)
            result[uuid_id] = self._pagerank_scores.get(str_id, 0.0)

        return result

    def get_citation_count(self, article_id: str) -> int:
        """
        Get the number of citations (in-degree) for an article.

        Args:
            article_id: The article's ID

        Returns:
            Number of articles citing this one
        """
        if self.graph.has_node(article_id):
            return self.graph.in_degree(article_id)
        return 0

    def get_citing_articles(self, article_id: str) -> list[str]:
        """
        Get IDs of articles that cite this article.

        Args:
            article_id: The article's ID

        Returns:
            List of citing article IDs
        """
        if self.graph.has_node(article_id):
            return list(self.graph.predecessors(article_id))
        return []

    def get_cited_articles(self, article_id: str) -> list[str]:
        """
        Get IDs of articles cited by this article.

        Args:
            article_id: The article's ID

        Returns:
            List of cited article IDs
        """
        if self.graph.has_node(article_id):
            return list(self.graph.successors(article_id))
        return []

    def get_stats(self) -> dict:
        """
        Get statistics about the citation graph.

        Returns:
            Dict with graph statistics
        """
        if self.graph.number_of_nodes() == 0:
            return {
                "nodes": 0,
                "edges": 0,
                "density": 0.0,
                "avg_in_degree": 0.0,
                "avg_out_degree": 0.0,
                "is_dirty": self._is_dirty,
            }

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "avg_in_degree": sum(d for _, d in self.graph.in_degree()) / self.graph.number_of_nodes(),
            "avg_out_degree": sum(d for _, d in self.graph.out_degree()) / self.graph.number_of_nodes(),
            "is_dirty": self._is_dirty,
        }


# Global service instance
_citation_graph_service: CitationGraphService | None = None


def get_citation_graph_service() -> CitationGraphService:
    """Get or create the citation graph service singleton."""
    global _citation_graph_service
    if _citation_graph_service is None:
        _citation_graph_service = CitationGraphService()
    return _citation_graph_service
