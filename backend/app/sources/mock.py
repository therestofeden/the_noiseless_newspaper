"""
Mock data source adapter with sample articles for each topic.

Provides realistic sample articles for testing and development
without requiring actual API connections.
"""

from datetime import datetime, timedelta
from typing import AsyncIterator
from uuid import UUID, uuid4

from app.models.domain import Article, Author, TopicPath


class MockSourceAdapter:
    """Mock adapter that generates sample articles for testing."""

    SOURCE_ID = "mock"

    # Sample articles organized by domain
    SAMPLE_ARTICLES: dict[str, list[dict]] = {
        "ai-ml": [
            {
                "title": "Scaling Laws for Neural Language Models Revisited",
                "abstract": "We investigate the scaling behavior of transformer language models, finding that performance continues to improve predictably with model size, data, and compute. Our analysis reveals new insights into optimal training configurations.",
                "authors": [
                    {"name": "Alice Chen", "affiliation": "DeepMind"},
                    {"name": "Bob Zhang", "affiliation": "Stanford University"},
                ],
                "topics": [
                    {"domain": "ai-ml", "subtopic": "llms", "niche": "architectures"},
                ],
                "url": "https://arxiv.org/abs/2401.00001",
                "citation_count": 142,
            },
            {
                "title": "Direct Preference Optimization: A Simpler Alternative to RLHF",
                "abstract": "We propose Direct Preference Optimization (DPO), which directly optimizes language model responses to align with human preferences without explicit reward modeling or reinforcement learning.",
                "authors": [
                    {"name": "Emma Wilson", "affiliation": "Anthropic"},
                ],
                "topics": [
                    {"domain": "ai-ml", "subtopic": "llms", "niche": "training"},
                ],
                "url": "https://arxiv.org/abs/2401.00002",
                "citation_count": 89,
            },
            {
                "title": "Chain-of-Thought Prompting Emerges from Language Model Scale",
                "abstract": "We demonstrate that chain-of-thought reasoning capabilities emerge as language models scale, enabling complex multi-step reasoning without explicit training for such tasks.",
                "authors": [
                    {"name": "David Kim", "affiliation": "Google Research"},
                    {"name": "Sarah Lee", "affiliation": "Google Research"},
                ],
                "topics": [
                    {"domain": "ai-ml", "subtopic": "llms", "niche": "reasoning"},
                ],
                "url": "https://arxiv.org/abs/2401.00003",
                "citation_count": 215,
            },
            {
                "title": "Red Teaming Language Models: Methods and Findings",
                "abstract": "We present a comprehensive framework for red teaming large language models, identifying vulnerabilities and developing mitigation strategies for safer AI deployment.",
                "authors": [
                    {"name": "Michael Brown", "affiliation": "OpenAI"},
                ],
                "topics": [
                    {"domain": "ai-ml", "subtopic": "llms", "niche": "safety"},
                ],
                "url": "https://arxiv.org/abs/2401.00004",
                "citation_count": 67,
            },
            {
                "title": "Diffusion Models Beat GANs on Image Synthesis",
                "abstract": "We show that diffusion models achieve state-of-the-art image generation quality, surpassing GANs on benchmark metrics while offering improved training stability.",
                "authors": [
                    {"name": "Lisa Wang", "affiliation": "MIT"},
                    {"name": "John Smith", "affiliation": "NVIDIA"},
                ],
                "topics": [
                    {"domain": "ai-ml", "subtopic": "computer-vision", "niche": "generation"},
                ],
                "url": "https://arxiv.org/abs/2401.00005",
                "citation_count": 312,
            },
            {
                "title": "YOLO-X: Exceeding YOLO Series in Real-Time Detection",
                "abstract": "We propose YOLO-X, an anchor-free detector that achieves superior performance in real-time object detection across various deployment scenarios.",
                "authors": [
                    {"name": "Wei Liu", "affiliation": "Megvii"},
                ],
                "topics": [
                    {"domain": "ai-ml", "subtopic": "computer-vision", "niche": "recognition"},
                ],
                "url": "https://arxiv.org/abs/2401.00006",
                "citation_count": 156,
            },
            {
                "title": "PPO: Proximal Policy Optimization Algorithms",
                "abstract": "We introduce a new family of policy gradient methods for reinforcement learning that alternate between sampling data and optimizing a surrogate objective function.",
                "authors": [
                    {"name": "James Miller", "affiliation": "OpenAI"},
                ],
                "topics": [
                    {"domain": "ai-ml", "subtopic": "reinforcement", "niche": "deep-rl"},
                ],
                "url": "https://arxiv.org/abs/2401.00007",
                "citation_count": 498,
            },
            {
                "title": "Self-Supervised Representation Learning: A Unified View",
                "abstract": "We provide a unified framework for understanding self-supervised learning methods, showing connections between contrastive, predictive, and generative approaches.",
                "authors": [
                    {"name": "Anna Garcia", "affiliation": "Facebook AI"},
                    {"name": "Peter Johnson", "affiliation": "UC Berkeley"},
                ],
                "topics": [
                    {"domain": "ai-ml", "subtopic": "ml-theory", "niche": "representation"},
                ],
                "url": "https://arxiv.org/abs/2401.00008",
                "citation_count": 178,
            },
        ],
        "physics": [
            {
                "title": "Scale-Free Networks: A Decade Later",
                "abstract": "We review the development of scale-free network theory and its applications across biology, technology, and social systems over the past decade.",
                "authors": [
                    {"name": "Albert Barabasi", "affiliation": "Northeastern University"},
                ],
                "topics": [
                    {"domain": "physics", "subtopic": "complexity", "niche": "networks"},
                ],
                "url": "https://arxiv.org/abs/2401.00009",
                "citation_count": 234,
            },
            {
                "title": "Quantum Error Correction with Surface Codes",
                "abstract": "We demonstrate practical quantum error correction using surface codes, achieving error rates below the threshold for fault-tolerant computation.",
                "authors": [
                    {"name": "Maria Rodriguez", "affiliation": "IBM Quantum"},
                    {"name": "Thomas Anderson", "affiliation": "Google Quantum AI"},
                ],
                "topics": [
                    {"domain": "physics", "subtopic": "quantum", "niche": "computing"},
                ],
                "url": "https://arxiv.org/abs/2401.00010",
                "citation_count": 89,
            },
            {
                "title": "Room Temperature Superconductivity: A Critical Review",
                "abstract": "We critically examine recent claims of room-temperature superconductivity, analyzing experimental evidence and theoretical predictions.",
                "authors": [
                    {"name": "Carlos Martinez", "affiliation": "Max Planck Institute"},
                ],
                "topics": [
                    {"domain": "physics", "subtopic": "condensed", "niche": "superconductivity"},
                ],
                "url": "https://arxiv.org/abs/2401.00011",
                "citation_count": 156,
            },
            {
                "title": "JWST Observations of Early Galaxy Formation",
                "abstract": "James Webb Space Telescope observations reveal unexpectedly massive galaxies in the early universe, challenging current models of galaxy formation.",
                "authors": [
                    {"name": "Elena Petrova", "affiliation": "STScI"},
                    {"name": "James Wright", "affiliation": "Cambridge University"},
                ],
                "topics": [
                    {"domain": "physics", "subtopic": "astro", "niche": "cosmology"},
                ],
                "url": "https://arxiv.org/abs/2401.00012",
                "citation_count": 201,
            },
        ],
        "biotech": [
            {
                "title": "CRISPR-Cas9: Five Years of Therapeutic Progress",
                "abstract": "We review the clinical progress of CRISPR-Cas9 gene editing, highlighting successful trials and remaining challenges for therapeutic applications.",
                "authors": [
                    {"name": "Jennifer Doudna", "affiliation": "UC Berkeley"},
                ],
                "topics": [
                    {"domain": "biotech", "subtopic": "gene-editing", "niche": "crispr"},
                ],
                "url": "https://pubmed.gov/2401.00013",
                "citation_count": 445,
            },
            {
                "title": "AlphaFold2: Highly Accurate Protein Structure Prediction",
                "abstract": "We present AlphaFold2, a deep learning system that predicts protein structures with atomic accuracy, revolutionizing structural biology.",
                "authors": [
                    {"name": "John Jumper", "affiliation": "DeepMind"},
                ],
                "topics": [
                    {"domain": "biotech", "subtopic": "drug-discovery", "niche": "ai-drug"},
                ],
                "url": "https://nature.com/2401.00014",
                "citation_count": 789,
            },
            {
                "title": "Synthetic Gene Circuits for Cellular Computing",
                "abstract": "We engineer genetic circuits capable of performing logic operations within living cells, enabling programmable biological computation.",
                "authors": [
                    {"name": "Drew Endy", "affiliation": "Stanford University"},
                    {"name": "Christina Smolke", "affiliation": "Stanford University"},
                ],
                "topics": [
                    {"domain": "biotech", "subtopic": "synbio", "niche": "genetic-circuits"},
                ],
                "url": "https://cell.com/2401.00015",
                "citation_count": 167,
            },
            {
                "title": "Neuralink: Progress in Brain-Machine Interfaces",
                "abstract": "We report advances in high-bandwidth brain-machine interfaces, demonstrating stable long-term neural recordings and bidirectional communication.",
                "authors": [
                    {"name": "Matthew McDougall", "affiliation": "Neuralink"},
                ],
                "topics": [
                    {"domain": "biotech", "subtopic": "neuro", "niche": "brain-machine"},
                ],
                "url": "https://biorxiv.org/2401.00016",
                "citation_count": 134,
            },
        ],
        "economics": [
            {
                "title": "Monetary Policy in the Age of Inflation",
                "abstract": "We analyze central bank responses to post-pandemic inflation, examining the effectiveness of interest rate policies and quantitative tightening.",
                "authors": [
                    {"name": "Paul Krugman", "affiliation": "CUNY"},
                ],
                "topics": [
                    {"domain": "economics", "subtopic": "macro", "niche": "monetary"},
                ],
                "url": "https://nber.org/2401.00017",
                "citation_count": 89,
            },
            {
                "title": "DeFi: Promise and Peril of Decentralized Finance",
                "abstract": "We examine the growth of decentralized finance protocols, analyzing their mechanisms, risks, and potential for financial innovation.",
                "authors": [
                    {"name": "Vitalik Buterin", "affiliation": "Ethereum Foundation"},
                    {"name": "Andrea Chen", "affiliation": "MIT Sloan"},
                ],
                "topics": [
                    {"domain": "economics", "subtopic": "markets", "niche": "crypto"},
                ],
                "url": "https://ssrn.com/2401.00018",
                "citation_count": 234,
            },
            {
                "title": "Nudging for Good: Evidence from Large-Scale Field Experiments",
                "abstract": "We present results from behavioral interventions across multiple countries, showing how small changes in choice architecture can improve outcomes.",
                "authors": [
                    {"name": "Richard Thaler", "affiliation": "University of Chicago"},
                ],
                "topics": [
                    {"domain": "economics", "subtopic": "behavioral", "niche": "decision"},
                ],
                "url": "https://aeaweb.org/2401.00019",
                "citation_count": 178,
            },
            {
                "title": "Global Supply Chain Resilience After COVID-19",
                "abstract": "We study how firms restructured global supply chains in response to pandemic disruptions, with implications for international trade patterns.",
                "authors": [
                    {"name": "Dani Rodrik", "affiliation": "Harvard Kennedy School"},
                ],
                "topics": [
                    {"domain": "economics", "subtopic": "development", "niche": "trade"},
                ],
                "url": "https://nber.org/2401.00020",
                "citation_count": 145,
            },
        ],
        "politics": [
            {
                "title": "US-China Technology Competition: A New Cold War?",
                "abstract": "We analyze the intensifying technological competition between the US and China, examining semiconductor policies, AI development, and geopolitical implications.",
                "authors": [
                    {"name": "Graham Allison", "affiliation": "Harvard Kennedy School"},
                ],
                "topics": [
                    {"domain": "politics", "subtopic": "geopolitics", "niche": "great-powers"},
                ],
                "url": "https://foreignaffairs.com/2401.00021",
                "citation_count": 112,
            },
            {
                "title": "Political Polarization in the Social Media Age",
                "abstract": "We measure political polarization across multiple countries, finding that social media exposure amplifies partisan divisions through algorithmic curation.",
                "authors": [
                    {"name": "Ezra Klein", "affiliation": "NYU"},
                    {"name": "Yochai Benkler", "affiliation": "Harvard Law School"},
                ],
                "topics": [
                    {"domain": "politics", "subtopic": "domestic", "niche": "polarization"},
                ],
                "url": "https://apsr.org/2401.00022",
                "citation_count": 234,
            },
            {
                "title": "EU AI Act: Regulating Artificial Intelligence",
                "abstract": "We analyze the European Union's AI Act, examining its risk-based regulatory framework and implications for global AI governance.",
                "authors": [
                    {"name": "Margrethe Vestager", "affiliation": "European Commission"},
                ],
                "topics": [
                    {"domain": "politics", "subtopic": "governance", "niche": "tech-regulation"},
                ],
                "url": "https://brookings.edu/2401.00023",
                "citation_count": 89,
            },
            {
                "title": "SolarWinds and the Future of Cyber Defense",
                "abstract": "We examine the SolarWinds cyberattack, analyzing vulnerabilities in software supply chains and proposing frameworks for improved cyber resilience.",
                "authors": [
                    {"name": "Bruce Schneier", "affiliation": "Harvard Kennedy School"},
                ],
                "topics": [
                    {"domain": "politics", "subtopic": "security", "niche": "cyber"},
                ],
                "url": "https://rand.org/2401.00024",
                "citation_count": 156,
            },
        ],
    }

    def __init__(self):
        """Initialize the mock adapter."""
        self._article_cache: dict[UUID, Article] = {}

    def _create_article(
        self,
        data: dict,
        external_id: str,
        published_offset_days: int,
    ) -> Article:
        """Create an Article from raw data."""
        article_id = uuid4()
        now = datetime.utcnow()

        article = Article(
            id=article_id,
            source=self.SOURCE_ID,
            external_id=external_id,
            title=data["title"],
            abstract=data.get("abstract"),
            url=data["url"],
            authors=[
                Author(
                    name=a["name"],
                    affiliation=a.get("affiliation"),
                )
                for a in data.get("authors", [])
            ],
            topics=[
                TopicPath(
                    domain=t["domain"],
                    subtopic=t["subtopic"],
                    niche=t.get("niche"),
                )
                for t in data.get("topics", [])
            ],
            published_at=now - timedelta(days=published_offset_days),
            fetched_at=now,
            citation_count=data.get("citation_count", 0),
            citation_ids=[],  # Mock doesn't have real citations
        )

        self._article_cache[article_id] = article
        return article

    async def fetch_articles(
        self,
        domains: list[str] | None = None,
        limit: int = 50,
    ) -> list[Article]:
        """
        Fetch mock articles, optionally filtered by domain.

        Args:
            domains: Optional list of domain IDs to filter by
            limit: Maximum number of articles to return

        Returns:
            List of Article objects
        """
        articles: list[Article] = []
        external_counter = 1

        target_domains = domains if domains else list(self.SAMPLE_ARTICLES.keys())

        for domain in target_domains:
            if domain not in self.SAMPLE_ARTICLES:
                continue

            domain_articles = self.SAMPLE_ARTICLES[domain]
            for i, data in enumerate(domain_articles):
                if len(articles) >= limit:
                    break

                # Vary publication dates for realistic ranking
                published_offset = (i % 7) + (external_counter % 14)

                article = self._create_article(
                    data=data,
                    external_id=f"mock-{external_counter:06d}",
                    published_offset_days=published_offset,
                )
                articles.append(article)
                external_counter += 1

            if len(articles) >= limit:
                break

        return articles[:limit]

    async def fetch_articles_stream(
        self,
        domains: list[str] | None = None,
        limit: int = 50,
    ) -> AsyncIterator[Article]:
        """
        Stream mock articles asynchronously.

        Args:
            domains: Optional list of domain IDs to filter by
            limit: Maximum number of articles to return

        Yields:
            Article objects
        """
        articles = await self.fetch_articles(domains=domains, limit=limit)
        for article in articles:
            yield article

    def get_cached_article(self, article_id: UUID) -> Article | None:
        """Get a previously fetched article from cache."""
        return self._article_cache.get(article_id)

    def clear_cache(self) -> None:
        """Clear the article cache."""
        self._article_cache.clear()


# Factory function
def get_mock_adapter() -> MockSourceAdapter:
    """Create a new mock adapter instance."""
    return MockSourceAdapter()
