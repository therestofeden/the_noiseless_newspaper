"""
Full 3-level taxonomy for article classification.

Structure:
- Domain (top level): e.g., "ai-ml", "physics"
- Subtopic (mid level): e.g., "llms", "quantum"
- Niche (bottom level): specific areas with keywords for matching
"""

from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class Niche:
    """Lowest level of taxonomy - specific topic area."""

    id: str
    name: str
    keywords: tuple[str, ...]

    def matches(self, text: str) -> float:
        """Return match score (0-1) based on keyword presence."""
        text_lower = text.lower()
        matches = sum(1 for kw in self.keywords if kw.lower() in text_lower)
        return min(matches / max(len(self.keywords), 1), 1.0)


@dataclass(frozen=True)
class Subtopic:
    """Mid level of taxonomy."""

    id: str
    name: str
    niches: tuple[Niche, ...]

    def get_niche(self, niche_id: str) -> Niche | None:
        """Get a niche by ID."""
        for niche in self.niches:
            if niche.id == niche_id:
                return niche
        return None

    def all_keywords(self) -> set[str]:
        """Get all keywords from all niches."""
        keywords: set[str] = set()
        for niche in self.niches:
            keywords.update(niche.keywords)
        return keywords


@dataclass(frozen=True)
class Domain:
    """Top level of taxonomy."""

    id: str
    name: str
    description: str
    subtopics: tuple[Subtopic, ...]

    def get_subtopic(self, subtopic_id: str) -> Subtopic | None:
        """Get a subtopic by ID."""
        for subtopic in self.subtopics:
            if subtopic.id == subtopic_id:
                return subtopic
        return None


@dataclass
class Taxonomy:
    """Full taxonomy structure."""

    domains: dict[str, Domain] = field(default_factory=dict)

    def get_domain(self, domain_id: str) -> Domain | None:
        """Get a domain by ID."""
        return self.domains.get(domain_id)

    def get_path(self, domain_id: str, subtopic_id: str, niche_id: str) -> tuple[Domain, Subtopic, Niche] | None:
        """Get full path by IDs."""
        domain = self.get_domain(domain_id)
        if not domain:
            return None
        subtopic = domain.get_subtopic(subtopic_id)
        if not subtopic:
            return None
        niche = subtopic.get_niche(niche_id)
        if not niche:
            return None
        return (domain, subtopic, niche)

    def iter_all_niches(self) -> Iterator[tuple[str, str, str, Niche]]:
        """Iterate over all niches yielding (domain_id, subtopic_id, niche_id, niche)."""
        for domain in self.domains.values():
            for subtopic in domain.subtopics:
                for niche in subtopic.niches:
                    yield (domain.id, subtopic.id, niche.id, niche)

    def to_dict(self) -> dict:
        """Convert taxonomy to dictionary for API responses."""
        return {
            domain_id: {
                "name": domain.name,
                "description": domain.description,
                "subtopics": {
                    subtopic.id: {
                        "name": subtopic.name,
                        "niches": {
                            niche.id: {
                                "name": niche.name,
                                "keywords": list(niche.keywords),
                            }
                            for niche in subtopic.niches
                        },
                    }
                    for subtopic in domain.subtopics
                },
            }
            for domain_id, domain in self.domains.items()
        }


# ============================================================================
# FULL TAXONOMY DEFINITION
# ============================================================================

AI_ML_DOMAIN = Domain(
    id="ai-ml",
    name="AI & Machine Learning",
    description="Artificial intelligence, machine learning, and related technologies",
    subtopics=(
        Subtopic(
            id="llms",
            name="Large Language Models",
            niches=(
                Niche(
                    id="architectures",
                    name="Model Architectures",
                    keywords=("transformer", "attention mechanism", "mixture of experts", "moe", "sparse models", "scaling laws"),
                ),
                Niche(
                    id="training",
                    name="Training & Fine-tuning",
                    keywords=("rlhf", "dpo", "instruction tuning", "pretraining", "fine-tuning", "lora", "qlora", "peft"),
                ),
                Niche(
                    id="reasoning",
                    name="Reasoning & Agents",
                    keywords=("chain of thought", "cot", "reasoning", "agents", "tool use", "function calling", "planning"),
                ),
                Niche(
                    id="safety",
                    name="Safety & Alignment",
                    keywords=("alignment", "constitutional ai", "red teaming", "jailbreak", "guardrails", "safety", "interpretability"),
                ),
            ),
        ),
        Subtopic(
            id="computer-vision",
            name="Computer Vision",
            niches=(
                Niche(
                    id="generation",
                    name="Image Generation",
                    keywords=("diffusion", "stable diffusion", "dalle", "midjourney", "image synthesis", "text-to-image", "gan"),
                ),
                Niche(
                    id="recognition",
                    name="Recognition & Detection",
                    keywords=("object detection", "image classification", "yolo", "segmentation", "face recognition", "ocr"),
                ),
                Niche(
                    id="video",
                    name="Video Understanding",
                    keywords=("video generation", "video understanding", "temporal modeling", "action recognition", "sora"),
                ),
            ),
        ),
        Subtopic(
            id="reinforcement",
            name="Reinforcement Learning",
            niches=(
                Niche(
                    id="deep-rl",
                    name="Deep RL",
                    keywords=("dqn", "ppo", "sac", "a3c", "policy gradient", "deep reinforcement learning", "actor critic"),
                ),
                Niche(
                    id="robotics",
                    name="Robotics & Control",
                    keywords=("robot learning", "manipulation", "locomotion", "sim-to-real", "imitation learning", "control"),
                ),
                Niche(
                    id="games",
                    name="Game Playing",
                    keywords=("alphago", "alphastar", "game ai", "mcts", "self-play", "atari", "openai five"),
                ),
            ),
        ),
        Subtopic(
            id="ml-theory",
            name="ML Theory & Methods",
            niches=(
                Niche(
                    id="optimization",
                    name="Optimization",
                    keywords=("sgd", "adam", "optimization", "learning rate", "convergence", "gradient descent", "loss landscape"),
                ),
                Niche(
                    id="generalization",
                    name="Generalization & Learning Theory",
                    keywords=("generalization", "overfitting", "regularization", "pac learning", "vc dimension", "double descent"),
                ),
                Niche(
                    id="representation",
                    name="Representation Learning",
                    keywords=("embeddings", "contrastive learning", "self-supervised", "representation learning", "disentanglement"),
                ),
            ),
        ),
    ),
)

PHYSICS_DOMAIN = Domain(
    id="physics",
    name="Physics",
    description="Physical sciences from quantum to cosmological scales",
    subtopics=(
        Subtopic(
            id="complexity",
            name="Complexity Science",
            niches=(
                Niche(
                    id="networks",
                    name="Network Science",
                    keywords=("network theory", "graph theory", "scale-free", "small world", "social networks", "centrality"),
                ),
                Niche(
                    id="emergence",
                    name="Emergence & Self-Organization",
                    keywords=("emergence", "self-organization", "criticality", "phase transitions", "collective behavior"),
                ),
                Niche(
                    id="chaos",
                    name="Chaos & Nonlinear Dynamics",
                    keywords=("chaos theory", "strange attractor", "bifurcation", "nonlinear dynamics", "lyapunov"),
                ),
            ),
        ),
        Subtopic(
            id="quantum",
            name="Quantum Physics",
            niches=(
                Niche(
                    id="computing",
                    name="Quantum Computing",
                    keywords=("qubit", "quantum computer", "quantum supremacy", "quantum error correction", "quantum algorithm"),
                ),
                Niche(
                    id="information",
                    name="Quantum Information",
                    keywords=("entanglement", "quantum cryptography", "qkd", "quantum communication", "teleportation"),
                ),
                Niche(
                    id="foundations",
                    name="Foundations of QM",
                    keywords=("measurement problem", "decoherence", "many worlds", "copenhagen", "bell inequality", "wave function"),
                ),
            ),
        ),
        Subtopic(
            id="condensed",
            name="Condensed Matter",
            niches=(
                Niche(
                    id="superconductivity",
                    name="Superconductivity",
                    keywords=("superconductor", "high-tc", "bcs theory", "cooper pairs", "room temperature superconductor"),
                ),
                Niche(
                    id="topological",
                    name="Topological Materials",
                    keywords=("topological insulator", "weyl semimetal", "majorana", "topological phases", "berry phase"),
                ),
                Niche(
                    id="2d-materials",
                    name="2D Materials",
                    keywords=("graphene", "tmdc", "van der waals", "moire", "twisted bilayer", "2d materials"),
                ),
            ),
        ),
        Subtopic(
            id="astro",
            name="Astrophysics & Cosmology",
            niches=(
                Niche(
                    id="cosmology",
                    name="Cosmology",
                    keywords=("dark matter", "dark energy", "cmb", "inflation", "big bang", "cosmological constant", "hubble"),
                ),
                Niche(
                    id="black-holes",
                    name="Black Holes & Gravity",
                    keywords=("black hole", "gravitational waves", "ligo", "event horizon", "hawking radiation", "singularity"),
                ),
                Niche(
                    id="exoplanets",
                    name="Exoplanets & Astrobiology",
                    keywords=("exoplanet", "habitable zone", "biosignature", "james webb", "kepler", "astrobiology"),
                ),
            ),
        ),
    ),
)

BIOTECH_DOMAIN = Domain(
    id="biotech",
    name="Biotechnology",
    description="Biological sciences and biotechnology applications",
    subtopics=(
        Subtopic(
            id="gene-editing",
            name="Gene Editing",
            niches=(
                Niche(
                    id="crispr",
                    name="CRISPR Technology",
                    keywords=("crispr", "cas9", "cas12", "cas13", "gene editing", "crispr-cas", "guide rna"),
                ),
                Niche(
                    id="gene-therapy",
                    name="Gene Therapy",
                    keywords=("gene therapy", "aav", "viral vector", "gene delivery", "in vivo editing", "ex vivo"),
                ),
                Niche(
                    id="base-editing",
                    name="Base & Prime Editing",
                    keywords=("base editing", "prime editing", "adenine base editor", "cytosine base editor", "precise editing"),
                ),
            ),
        ),
        Subtopic(
            id="drug-discovery",
            name="Drug Discovery",
            niches=(
                Niche(
                    id="ai-drug",
                    name="AI for Drug Discovery",
                    keywords=("alphafold", "protein folding", "drug design", "virtual screening", "molecular generation", "docking"),
                ),
                Niche(
                    id="antibodies",
                    name="Antibody Engineering",
                    keywords=("monoclonal antibody", "bispecific", "antibody-drug conjugate", "adc", "nanobody", "immunotherapy"),
                ),
                Niche(
                    id="clinical-trials",
                    name="Clinical Development",
                    keywords=("clinical trial", "phase 1", "phase 2", "phase 3", "fda approval", "endpoint", "biomarker"),
                ),
            ),
        ),
        Subtopic(
            id="synbio",
            name="Synthetic Biology",
            niches=(
                Niche(
                    id="metabolic",
                    name="Metabolic Engineering",
                    keywords=("metabolic engineering", "biofuel", "biosynthesis", "pathway engineering", "fermentation"),
                ),
                Niche(
                    id="genetic-circuits",
                    name="Genetic Circuits",
                    keywords=("genetic circuit", "synthetic gene", "toggle switch", "oscillator", "logic gate", "optogenetics"),
                ),
                Niche(
                    id="cell-free",
                    name="Cell-Free Systems",
                    keywords=("cell-free", "in vitro transcription", "extract", "protocell", "minimal cell"),
                ),
            ),
        ),
        Subtopic(
            id="neuro",
            name="Neuroscience",
            niches=(
                Niche(
                    id="brain-machine",
                    name="Brain-Machine Interfaces",
                    keywords=("bci", "neuralink", "brain-computer interface", "neural implant", "neuroprosthesis", "eeg"),
                ),
                Niche(
                    id="connectomics",
                    name="Connectomics",
                    keywords=("connectome", "neural circuit", "synapse mapping", "electron microscopy", "brain mapping"),
                ),
                Niche(
                    id="neurodegeneration",
                    name="Neurodegeneration",
                    keywords=("alzheimer", "parkinson", "amyloid", "tau", "neurodegeneration", "dementia", "huntington"),
                ),
            ),
        ),
    ),
)

ECONOMICS_DOMAIN = Domain(
    id="economics",
    name="Economics",
    description="Economic research and financial systems",
    subtopics=(
        Subtopic(
            id="macro",
            name="Macroeconomics",
            niches=(
                Niche(
                    id="monetary",
                    name="Monetary Policy",
                    keywords=("federal reserve", "interest rate", "inflation", "quantitative easing", "central bank", "monetary policy"),
                ),
                Niche(
                    id="fiscal",
                    name="Fiscal Policy",
                    keywords=("fiscal policy", "government spending", "taxation", "deficit", "debt", "stimulus", "budget"),
                ),
                Niche(
                    id="growth",
                    name="Economic Growth",
                    keywords=("gdp", "economic growth", "productivity", "total factor productivity", "convergence", "development"),
                ),
            ),
        ),
        Subtopic(
            id="markets",
            name="Financial Markets",
            niches=(
                Niche(
                    id="crypto",
                    name="Cryptocurrency & DeFi",
                    keywords=("bitcoin", "ethereum", "cryptocurrency", "defi", "blockchain", "smart contract", "nft", "web3"),
                ),
                Niche(
                    id="equities",
                    name="Equities & Trading",
                    keywords=("stock market", "equity", "trading", "hedge fund", "algorithmic trading", "market microstructure"),
                ),
                Niche(
                    id="derivatives",
                    name="Derivatives & Risk",
                    keywords=("options", "futures", "derivatives", "hedging", "risk management", "volatility", "var"),
                ),
            ),
        ),
        Subtopic(
            id="behavioral",
            name="Behavioral Economics",
            niches=(
                Niche(
                    id="decision",
                    name="Decision Making",
                    keywords=("prospect theory", "heuristics", "cognitive bias", "bounded rationality", "nudge", "choice architecture"),
                ),
                Niche(
                    id="experimental",
                    name="Experimental Economics",
                    keywords=("experiment", "lab study", "field experiment", "game theory", "mechanism design", "auction"),
                ),
                Niche(
                    id="neuroeconomics",
                    name="Neuroeconomics",
                    keywords=("neuroeconomics", "fmri", "dopamine", "reward", "neural economics", "brain imaging"),
                ),
            ),
        ),
        Subtopic(
            id="development",
            name="Development Economics",
            niches=(
                Niche(
                    id="poverty",
                    name="Poverty & Inequality",
                    keywords=("poverty", "inequality", "gini", "redistribution", "social mobility", "basic income"),
                ),
                Niche(
                    id="institutions",
                    name="Institutions & Governance",
                    keywords=("institutions", "property rights", "rule of law", "corruption", "governance", "democracy"),
                ),
                Niche(
                    id="trade",
                    name="International Trade",
                    keywords=("trade", "tariff", "globalization", "supply chain", "comparative advantage", "trade war"),
                ),
            ),
        ),
    ),
)

POLITICS_DOMAIN = Domain(
    id="politics",
    name="Politics & Policy",
    description="Political science, governance, and public policy",
    subtopics=(
        Subtopic(
            id="geopolitics",
            name="Geopolitics",
            niches=(
                Niche(
                    id="great-powers",
                    name="Great Power Competition",
                    keywords=("us-china", "russia", "great power", "superpower", "hegemony", "bipolar", "multipolar"),
                ),
                Niche(
                    id="regional",
                    name="Regional Dynamics",
                    keywords=("middle east", "asia-pacific", "europe", "africa", "latin america", "regional conflict"),
                ),
                Niche(
                    id="international-orgs",
                    name="International Organizations",
                    keywords=("united nations", "nato", "eu", "wto", "imf", "world bank", "multilateral"),
                ),
            ),
        ),
        Subtopic(
            id="domestic",
            name="Domestic Politics",
            niches=(
                Niche(
                    id="elections",
                    name="Elections & Voting",
                    keywords=("election", "voting", "poll", "campaign", "primary", "electoral", "turnout", "swing state"),
                ),
                Niche(
                    id="polarization",
                    name="Political Polarization",
                    keywords=("polarization", "partisan", "tribal", "culture war", "media bias", "echo chamber", "misinformation"),
                ),
                Niche(
                    id="movements",
                    name="Social Movements",
                    keywords=("protest", "activism", "grassroots", "civil rights", "social movement", "reform"),
                ),
            ),
        ),
        Subtopic(
            id="governance",
            name="Governance & Regulation",
            niches=(
                Niche(
                    id="tech-regulation",
                    name="Technology Regulation",
                    keywords=("ai regulation", "antitrust", "section 230", "data privacy", "gdpr", "tech policy", "platform"),
                ),
                Niche(
                    id="climate-policy",
                    name="Climate Policy",
                    keywords=("climate policy", "carbon tax", "paris agreement", "net zero", "green new deal", "emissions"),
                ),
                Niche(
                    id="healthcare-policy",
                    name="Healthcare Policy",
                    keywords=("healthcare policy", "medicare", "universal healthcare", "drug pricing", "public health", "pandemic"),
                ),
            ),
        ),
        Subtopic(
            id="security",
            name="Security & Defense",
            niches=(
                Niche(
                    id="cyber",
                    name="Cybersecurity",
                    keywords=("cybersecurity", "cyberattack", "hacking", "ransomware", "cyber warfare", "critical infrastructure"),
                ),
                Niche(
                    id="military",
                    name="Military & Defense",
                    keywords=("military", "defense", "pentagon", "weapons", "deterrence", "arms control", "nuclear"),
                ),
                Niche(
                    id="intelligence",
                    name="Intelligence & Espionage",
                    keywords=("intelligence", "espionage", "cia", "surveillance", "counterintelligence", "covert"),
                ),
            ),
        ),
    ),
)


def build_taxonomy() -> Taxonomy:
    """Build and return the full taxonomy."""
    taxonomy = Taxonomy()
    for domain in [AI_ML_DOMAIN, PHYSICS_DOMAIN, BIOTECH_DOMAIN, ECONOMICS_DOMAIN, POLITICS_DOMAIN]:
        taxonomy.domains[domain.id] = domain
    return taxonomy


# Global taxonomy instance
TAXONOMY = build_taxonomy()


def get_taxonomy() -> Taxonomy:
    """Get the global taxonomy instance."""
    return TAXONOMY


def classify_text(text: str, threshold: float = 0.1) -> list[tuple[str, str, str, float]]:
    """
    Classify text into taxonomy categories.

    Returns list of (domain_id, subtopic_id, niche_id, score) tuples
    sorted by score descending.
    """
    results: list[tuple[str, str, str, float]] = []

    for domain_id, subtopic_id, niche_id, niche in TAXONOMY.iter_all_niches():
        score = niche.matches(text)
        if score >= threshold:
            results.append((domain_id, subtopic_id, niche_id, score))

    results.sort(key=lambda x: x[3], reverse=True)
    return results
