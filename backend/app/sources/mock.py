"""
Mock data source for development and testing.
Generates realistic-looking articles without external API calls.
"""
import hashlib
import random
from datetime import datetime, timedelta
from typing import AsyncIterator, Optional

from app.models.domain import Article, ArticleSource, Author, ContentType, TopicPath
from app.sources.base import ArticleSourceAdapter

# Comprehensive mock data organized by topic
MOCK_DATA = {
    # AI & ML
    "ai-ml/llms/interpretability": [
        {
            "title": "Sparse Autoencoders Reveal Hidden Structure in Large Language Models",
            "abstract": "We demonstrate that sparse autoencoders can decompose neural network activations into interpretable features, revealing how language models organize knowledge internally. The technique identifies monosemantic neurons and polysemantic superposition patterns.",
            "authors": ["Sarah Chen", "Michael Roberts"],
            "source": "arXiv",
            "citations": 847,
        },
        {
            "title": "Mechanistic Interpretability at Scale: Mapping GPT-4's Internal Representations",
            "abstract": "A comprehensive study applying mechanistic interpretability techniques to production-scale language models. We identify circuit patterns responsible for in-context learning, factual recall, and chain-of-thought reasoning.",
            "authors": ["James Liu", "Emily Watson", "David Park"],
            "source": "Anthropic Research",
            "citations": 623,
        },
        {
            "title": "Causal Tracing in Transformer Architectures: A Unified Framework",
            "abstract": "We present a unified framework for causal intervention experiments in transformers, enabling precise localization of factual knowledge storage and retrieval mechanisms.",
            "authors": ["Anna Kowalski", "Thomas Schmidt"],
            "source": "NeurIPS 2025",
            "citations": 412,
        },
    ],
    "ai-ml/llms/scaling": [
        {
            "title": "Chinchilla-2: Optimal Compute Allocation for 10T Parameter Models",
            "abstract": "Revisiting scaling laws for frontier models, we find diminishing returns beyond 5T parameters without architectural innovations. We propose mixture-of-experts as the path forward.",
            "authors": ["DeepMind Team"],
            "source": "DeepMind",
            "citations": 1203,
        },
        {
            "title": "The Bitter Lesson Revisited: Emergent Capabilities at 100T Parameters",
            "abstract": "Analysis of emergent capabilities in hypothetical 100T parameter models suggests phase transitions in reasoning ability that cannot be predicted from smaller scales.",
            "authors": ["Richard Sutton Jr.", "Wei Zhang"],
            "source": "arXiv",
            "citations": 534,
        },
    ],
    "ai-ml/llms/architectures": [
        {
            "title": "Mamba-2: Linear-Time Sequence Modeling Matches Transformer Quality",
            "abstract": "We present Mamba-2, achieving transformer-quality language modeling with O(n) complexity. Key innovations include selective state spaces and hardware-aware algorithm design.",
            "authors": ["Albert Gu", "Tri Dao"],
            "source": "ICML 2025",
            "citations": 892,
        },
    ],

    # Physics - Complexity
    "physics/complexity/networks": [
        {
            "title": "Universal Scaling in Temporal Networks Reveals Hidden Social Dynamics",
            "abstract": "Analysis of 50 million interactions shows temporal networks follow universal scaling laws that predict cascade failures and viral spread with unprecedented accuracy.",
            "authors": ["László Barabási", "Maria Santos"],
            "source": "Nature Physics",
            "citations": 432,
        },
        {
            "title": "Hypergraph Dynamics and Higher-Order Interactions in Complex Systems",
            "abstract": "We extend network theory to hypergraphs, revealing that higher-order interactions fundamentally alter spreading dynamics, synchronization, and collective behavior.",
            "authors": ["Giovanni Petri", "Alain Barrat"],
            "source": "Physical Review X",
            "citations": 287,
        },
    ],
    "physics/complexity/emergence": [
        {
            "title": "Self-Organized Criticality in Neural Networks Explains Generalization",
            "abstract": "Deep networks naturally evolve toward critical states during training, and this criticality directly correlates with generalization performance.",
            "authors": ["Surya Ganguli", "Jascha Sohl-Dickstein"],
            "source": "Physical Review X",
            "citations": 289,
        },
    ],

    # Physics - Quantum
    "physics/quantum/computing": [
        {
            "title": "1000-Qubit Error-Corrected Quantum Computer Achieves Fault Tolerance",
            "abstract": "Milestone demonstration of fault-tolerant quantum computation using surface codes, running Shor's algorithm on a 128-bit integer.",
            "authors": ["IBM Quantum Team"],
            "source": "Nature",
            "citations": 2341,
        },
    ],

    # Biotech
    "biotech/gene-editing/crispr": [
        {
            "title": "CRISPR-Based Gene Therapy Shows 94% Efficacy in Sickle Cell Treatment",
            "abstract": "Landmark clinical trial demonstrates near-complete elimination of sickle cell crises in treated patients over a 3-year follow-up period.",
            "authors": ["Jennifer Doudna", "Fyodor Urnov", "Clinical Team"],
            "source": "Nature Medicine",
            "citations": 1567,
        },
    ],
    "biotech/neuro/brain-computer": [
        {
            "title": "Neuralink Competitor Achieves 10,000 Channel Recording in Human Trials",
            "abstract": "Synchron's new electrode array enables unprecedented resolution in brain-computer interfaces, allowing paralyzed patients to control robotic arms with fine motor precision.",
            "authors": ["Synchron Research Team"],
            "source": "Science",
            "citations": 892,
        },
    ],

    # Economics
    "economics/macro/monetary": [
        {
            "title": "Central Banks Coordinate on Digital Currency Interoperability Standard",
            "abstract": "The Bank for International Settlements publishes framework enabling cross-border CBDC transactions across 47 participating nations.",
            "authors": ["BIS Research Team"],
            "source": "BIS Working Papers",
            "citations": 312,
        },
    ],
    "economics/behavioral/decision-making": [
        {
            "title": "Meta-Analysis of 500 Nudge Interventions Reveals Long-Term Decay Patterns",
            "abstract": "Comprehensive study shows most behavioral interventions lose 60% effectiveness within 6 months, but certain 'habit-forming' nudges persist indefinitely.",
            "authors": ["Cass Sunstein", "Richard Thaler Jr."],
            "source": "Journal of Political Economy",
            "citations": 445,
        },
    ],

    # Politics
    "politics/geopolitics/us-china": [
        {
            "title": "Taiwan Semiconductor Announces Arizona Fab Expansion Amid Tensions",
            "abstract": "TSMC's $40B investment signals strategic decoupling acceleration. Analysis of supply chain implications for global chip markets.",
            "authors": ["Graham Allison", "Kevin Rudd"],
            "source": "Foreign Affairs",
            "citations": 234,
        },
    ],
    "politics/governance/tech-regulation": [
        {
            "title": "EU AI Act Enforcement Begins: First Major Fines Expected Q2",
            "abstract": "Regulatory bodies gear up for enforcement of world's first comprehensive AI legislation. Companies scramble to demonstrate compliance.",
            "authors": ["Margrethe Vestager", "EU Commission"],
            "source": "The Economist",
            "citations": 156,
        },
    ],
}

# Fill in missing topics with generic content
DEFAULT_ARTICLES = [
    {
        "title": "Recent Advances in {topic} Research",
        "abstract": "A comprehensive review of the latest developments in {topic}, highlighting key breakthroughs and future directions for the field.",
        "authors": ["Research Team"],
        "source": "Nature Reviews",
        "citations": 150,
    },
    {
        "title": "New Methodology for Studying {topic}",
        "abstract": "We present a novel approach to investigating {topic} that enables more precise measurements and deeper insights.",
        "authors": ["Academic Consortium"],
        "source": "Science",
        "citations": 89,
    },
]


class MockAdapter(ArticleSourceAdapter):
    """Mock adapter for development without API keys."""

    def __init__(self):
        self._article_cache: dict[str, Article] = {}

    @property
    def source_type(self) -> ArticleSource:
        return ArticleSource.MOCK

    @property
    def name(self) -> str:
        return "Mock Data"

    def _generate_article(self, topic: TopicPath, template: dict, index: int) -> Article:
        """Generate an article from a template."""
        topic_name = topic.niche_id.replace("-", " ").title()

        title = template["title"].format(topic=topic_name)
        abstract = template["abstract"].format(topic=topic_name)

        # Generate deterministic but varied dates
        seed = hash(f"{topic.path}:{title}")
        random.seed(seed)
        days_ago = random.randint(1, 90)
        published_at = datetime.now() - timedelta(days=days_ago)

        # Generate ID
        id_hash = hashlib.md5(f"{topic.path}:{title}".encode()).hexdigest()[:10]
        article_id = f"mock:{id_hash}"

        # Parse authors
        authors = [Author(name=name) for name in template.get("authors", ["Unknown"])]

        # Add some variance to citations
        base_citations = template.get("citations", 100)
        citations = base_citations + random.randint(-20, 50)

        article = Article(
            id=article_id,
            source=ArticleSource.MOCK,
            content_type=ContentType.RESEARCH if "politics" not in topic.path else ContentType.NEWS,
            title=title,
            abstract=abstract,
            authors=authors,
            published_at=published_at,
            url=f"https://example.com/articles/{id_hash}",
            citation_count=max(0, citations),
        )

        # Cache for later retrieval
        self._article_cache[article_id] = article

        return article

    async def search(
        self,
        topic: TopicPath,
        keywords: list[str],
        since: Optional[datetime] = None,
        max_results: int = 50,
    ) -> AsyncIterator[Article]:
        """Generate mock articles for the topic."""
        # Look up topic-specific mock data
        templates = MOCK_DATA.get(topic.path, [])

        # Fall back to default templates if none found
        if not templates:
            # Try parent topic
            parent_path = f"{topic.category_id}/{topic.subtopic_id}"
            for key, value in MOCK_DATA.items():
                if key.startswith(parent_path):
                    templates = value
                    break

        if not templates:
            templates = DEFAULT_ARTICLES

        # Generate articles
        for i, template in enumerate(templates[:max_results]):
            article = self._generate_article(topic, template, i)

            # Apply date filter
            if since and article.published_at < since:
                continue

            yield article

    async def get_article(self, article_id: str) -> Optional[Article]:
        """Retrieve a previously generated article."""
        return self._article_cache.get(article_id)

    async def get_citations(self, article_id: str) -> tuple[list[str], list[str]]:
        """Generate mock citation relationships."""
        # Generate some random mock citations
        random.seed(hash(article_id))

        citing = [f"mock:{hashlib.md5(f'{article_id}:citing:{i}'.encode()).hexdigest()[:10]}"
                  for i in range(random.randint(5, 20))]

        cited = [f"mock:{hashlib.md5(f'{article_id}:cited:{i}'.encode()).hexdigest()[:10]}"
                 for i in range(random.randint(10, 30))]

        return citing, cited

    async def health_check(self) -> bool:
        """Mock is always healthy."""
        return True
