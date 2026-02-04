"""
Topic taxonomy definition for The Noiseless Newspaper.

This is the single source of truth for all topic categories,
subtopics, and niches. It's used by:
- Frontend for displaying topic selection
- Backend for article classification
- Embedding service for topic matching
"""

TOPIC_TAXONOMY = {
    "ai-ml": {
        "id": "ai-ml",
        "name": "AI & Machine Learning",
        "icon": "🤖",
        "description": "Artificial intelligence, deep learning, and computational intelligence",
        "content_type": "research",
        "subtopics": {
            "llms": {
                "id": "llms",
                "name": "Large Language Models",
                "niches": {
                    "interpretability": {
                        "id": "interpretability",
                        "name": "Interpretability & Alignment",
                        "keywords": ["mechanistic interpretability", "alignment", "safety", "explainability", "sparse autoencoders"],
                    },
                    "architectures": {
                        "id": "architectures",
                        "name": "Model Architectures",
                        "keywords": ["transformer", "attention", "mamba", "state space", "architecture"],
                    },
                    "scaling": {
                        "id": "scaling",
                        "name": "Scaling Laws & Training",
                        "keywords": ["scaling laws", "chinchilla", "compute optimal", "training dynamics"],
                    },
                    "applications": {
                        "id": "applications",
                        "name": "Applications & Deployment",
                        "keywords": ["deployment", "inference", "fine-tuning", "RLHF", "applications"],
                    },
                },
            },
            "computer-vision": {
                "id": "computer-vision",
                "name": "Computer Vision",
                "niches": {
                    "generative": {
                        "id": "generative",
                        "name": "Generative Models (Diffusion, GANs)",
                        "keywords": ["diffusion", "stable diffusion", "GAN", "image generation", "DALL-E"],
                    },
                    "detection": {
                        "id": "detection",
                        "name": "Object Detection & Segmentation",
                        "keywords": ["object detection", "segmentation", "YOLO", "SAM", "instance segmentation"],
                    },
                    "video": {
                        "id": "video",
                        "name": "Video Understanding",
                        "keywords": ["video understanding", "action recognition", "video generation", "temporal"],
                    },
                },
            },
            "reinforcement": {
                "id": "reinforcement",
                "name": "Reinforcement Learning",
                "niches": {
                    "robotics": {
                        "id": "robotics",
                        "name": "Robotics & Control",
                        "keywords": ["robotics", "control", "manipulation", "locomotion", "sim2real"],
                    },
                    "games": {
                        "id": "games",
                        "name": "Game Playing & Strategy",
                        "keywords": ["game playing", "AlphaGo", "strategy", "MCTS", "self-play"],
                    },
                    "multi-agent": {
                        "id": "multi-agent",
                        "name": "Multi-Agent Systems",
                        "keywords": ["multi-agent", "cooperation", "competition", "emergent behavior"],
                    },
                },
            },
            "ml-theory": {
                "id": "ml-theory",
                "name": "ML Theory & Foundations",
                "niches": {
                    "optimization": {
                        "id": "optimization",
                        "name": "Optimization & Generalization",
                        "keywords": ["optimization", "generalization", "loss landscape", "SGD", "Adam"],
                    },
                    "causality": {
                        "id": "causality",
                        "name": "Causal Inference",
                        "keywords": ["causal inference", "causality", "do-calculus", "counterfactual"],
                    },
                    "fairness": {
                        "id": "fairness",
                        "name": "Fairness & Robustness",
                        "keywords": ["fairness", "bias", "robustness", "adversarial", "out-of-distribution"],
                    },
                },
            },
        },
    },
    "physics": {
        "id": "physics",
        "name": "Physics",
        "icon": "⚛️",
        "description": "Fundamental physics, condensed matter, and theoretical advances",
        "content_type": "research",
        "subtopics": {
            "complexity": {
                "id": "complexity",
                "name": "Complexity Science",
                "niches": {
                    "networks": {
                        "id": "networks",
                        "name": "Network Theory",
                        "keywords": ["network theory", "graph theory", "scale-free", "small world", "centrality"],
                    },
                    "emergence": {
                        "id": "emergence",
                        "name": "Emergence & Self-Organization",
                        "keywords": ["emergence", "self-organization", "collective behavior", "phase transition"],
                    },
                    "chaos": {
                        "id": "chaos",
                        "name": "Chaos & Dynamical Systems",
                        "keywords": ["chaos", "dynamical systems", "attractor", "bifurcation", "Lyapunov"],
                    },
                    "info-theory": {
                        "id": "info-theory",
                        "name": "Information Theory",
                        "keywords": ["information theory", "entropy", "mutual information", "complexity measures"],
                    },
                },
            },
            "quantum": {
                "id": "quantum",
                "name": "Quantum Physics",
                "niches": {
                    "computing": {
                        "id": "computing",
                        "name": "Quantum Computing",
                        "keywords": ["quantum computing", "qubit", "quantum algorithm", "error correction", "NISQ"],
                    },
                    "foundations": {
                        "id": "foundations",
                        "name": "Quantum Foundations",
                        "keywords": ["quantum foundations", "entanglement", "Bell inequality", "measurement"],
                    },
                    "materials": {
                        "id": "materials",
                        "name": "Quantum Materials",
                        "keywords": ["quantum materials", "topological", "superconductor", "quantum spin"],
                    },
                },
            },
            "condensed": {
                "id": "condensed",
                "name": "Condensed Matter",
                "niches": {
                    "superconductivity": {
                        "id": "superconductivity",
                        "name": "Superconductivity",
                        "keywords": ["superconductivity", "superconductor", "BCS", "cuprate", "room temperature"],
                    },
                    "topological": {
                        "id": "topological",
                        "name": "Topological Phases",
                        "keywords": ["topological", "topological insulator", "Weyl semimetal", "Berry phase"],
                    },
                    "soft-matter": {
                        "id": "soft-matter",
                        "name": "Soft Matter & Polymers",
                        "keywords": ["soft matter", "polymer", "colloid", "liquid crystal", "active matter"],
                    },
                },
            },
            "astro": {
                "id": "astro",
                "name": "Astrophysics & Cosmology",
                "niches": {
                    "exoplanets": {
                        "id": "exoplanets",
                        "name": "Exoplanets",
                        "keywords": ["exoplanet", "habitable zone", "transit", "radial velocity", "biosignature"],
                    },
                    "dark-matter": {
                        "id": "dark-matter",
                        "name": "Dark Matter & Energy",
                        "keywords": ["dark matter", "dark energy", "cosmology", "WIMP", "axion"],
                    },
                    "gravitational": {
                        "id": "gravitational",
                        "name": "Gravitational Waves",
                        "keywords": ["gravitational waves", "LIGO", "black hole merger", "neutron star"],
                    },
                },
            },
        },
    },
    "biotech": {
        "id": "biotech",
        "name": "Biotechnology",
        "icon": "🧬",
        "description": "Genetic engineering, drug discovery, and biological systems",
        "content_type": "research",
        "subtopics": {
            "gene-editing": {
                "id": "gene-editing",
                "name": "Gene Editing & Therapy",
                "niches": {
                    "crispr": {
                        "id": "crispr",
                        "name": "CRISPR & Base Editing",
                        "keywords": ["CRISPR", "Cas9", "base editing", "prime editing", "gene editing"],
                    },
                    "gene-therapy": {
                        "id": "gene-therapy",
                        "name": "Gene Therapy Trials",
                        "keywords": ["gene therapy", "clinical trial", "AAV", "lentivirus", "in vivo"],
                    },
                    "epigenetics": {
                        "id": "epigenetics",
                        "name": "Epigenetics",
                        "keywords": ["epigenetics", "methylation", "histone", "chromatin", "epigenome"],
                    },
                },
            },
            "drug-discovery": {
                "id": "drug-discovery",
                "name": "Drug Discovery",
                "niches": {
                    "ai-pharma": {
                        "id": "ai-pharma",
                        "name": "AI in Pharma",
                        "keywords": ["AI drug discovery", "AlphaFold", "molecular design", "virtual screening"],
                    },
                    "clinical-trials": {
                        "id": "clinical-trials",
                        "name": "Clinical Trials",
                        "keywords": ["clinical trial", "Phase III", "FDA approval", "efficacy", "safety"],
                    },
                    "small-molecules": {
                        "id": "small-molecules",
                        "name": "Small Molecules",
                        "keywords": ["small molecule", "kinase inhibitor", "drug target", "medicinal chemistry"],
                    },
                },
            },
            "synbio": {
                "id": "synbio",
                "name": "Synthetic Biology",
                "niches": {
                    "metabolic": {
                        "id": "metabolic",
                        "name": "Metabolic Engineering",
                        "keywords": ["metabolic engineering", "pathway engineering", "flux", "fermentation"],
                    },
                    "cell-free": {
                        "id": "cell-free",
                        "name": "Cell-Free Systems",
                        "keywords": ["cell-free", "in vitro", "TX-TL", "protein synthesis"],
                    },
                    "biofuels": {
                        "id": "biofuels",
                        "name": "Biofuels & Biomaterials",
                        "keywords": ["biofuel", "biomaterial", "sustainable", "bioeconomy", "algae"],
                    },
                },
            },
            "neuro": {
                "id": "neuro",
                "name": "Neuroscience",
                "niches": {
                    "brain-computer": {
                        "id": "brain-computer",
                        "name": "Brain-Computer Interfaces",
                        "keywords": ["brain-computer interface", "BCI", "neural interface", "Neuralink", "electrode"],
                    },
                    "connectomics": {
                        "id": "connectomics",
                        "name": "Connectomics",
                        "keywords": ["connectome", "connectomics", "neural circuit", "synapse", "wiring diagram"],
                    },
                    "neurodegeneration": {
                        "id": "neurodegeneration",
                        "name": "Neurodegeneration",
                        "keywords": ["neurodegeneration", "Alzheimer", "Parkinson", "amyloid", "tau"],
                    },
                },
            },
        },
    },
    "economics": {
        "id": "economics",
        "name": "Economics & Finance",
        "icon": "📈",
        "description": "Markets, policy, and economic theory",
        "content_type": "research",
        "subtopics": {
            "macro": {
                "id": "macro",
                "name": "Macroeconomics",
                "niches": {
                    "monetary": {
                        "id": "monetary",
                        "name": "Monetary Policy & Central Banks",
                        "keywords": ["monetary policy", "central bank", "interest rate", "inflation", "Federal Reserve"],
                    },
                    "fiscal": {
                        "id": "fiscal",
                        "name": "Fiscal Policy & Debt",
                        "keywords": ["fiscal policy", "government spending", "national debt", "deficit", "stimulus"],
                    },
                    "trade": {
                        "id": "trade",
                        "name": "International Trade",
                        "keywords": ["international trade", "tariff", "trade agreement", "WTO", "globalization"],
                    },
                },
            },
            "markets": {
                "id": "markets",
                "name": "Financial Markets",
                "niches": {
                    "equities": {
                        "id": "equities",
                        "name": "Equity Markets",
                        "keywords": ["stock market", "equity", "S&P 500", "earnings", "valuation"],
                    },
                    "crypto": {
                        "id": "crypto",
                        "name": "Crypto & Digital Assets",
                        "keywords": ["cryptocurrency", "bitcoin", "ethereum", "DeFi", "blockchain"],
                    },
                    "derivatives": {
                        "id": "derivatives",
                        "name": "Derivatives & Risk",
                        "keywords": ["derivatives", "options", "futures", "risk management", "hedging"],
                    },
                },
            },
            "behavioral": {
                "id": "behavioral",
                "name": "Behavioral Economics",
                "niches": {
                    "decision-making": {
                        "id": "decision-making",
                        "name": "Decision Making",
                        "keywords": ["decision making", "cognitive bias", "heuristics", "prospect theory"],
                    },
                    "nudges": {
                        "id": "nudges",
                        "name": "Nudges & Policy Design",
                        "keywords": ["nudge", "choice architecture", "behavioral intervention", "default"],
                    },
                    "experiments": {
                        "id": "experiments",
                        "name": "Economic Experiments",
                        "keywords": ["economic experiment", "lab experiment", "field experiment", "game theory"],
                    },
                },
            },
            "development": {
                "id": "development",
                "name": "Development Economics",
                "niches": {
                    "poverty": {
                        "id": "poverty",
                        "name": "Poverty & Inequality",
                        "keywords": ["poverty", "inequality", "Gini", "social mobility", "wealth distribution"],
                    },
                    "institutions": {
                        "id": "institutions",
                        "name": "Institutions & Growth",
                        "keywords": ["institutions", "economic growth", "governance", "rule of law"],
                    },
                    "rcts": {
                        "id": "rcts",
                        "name": "RCTs & Impact Evaluation",
                        "keywords": ["RCT", "randomized controlled trial", "impact evaluation", "J-PAL"],
                    },
                },
            },
        },
    },
    "politics": {
        "id": "politics",
        "name": "Politics & Geopolitics",
        "icon": "🌍",
        "description": "Political analysis, international relations, and governance",
        "content_type": "news",
        "subtopics": {
            "geopolitics": {
                "id": "geopolitics",
                "name": "Geopolitics",
                "niches": {
                    "us-china": {
                        "id": "us-china",
                        "name": "US-China Relations",
                        "keywords": ["US China", "Taiwan", "decoupling", "semiconductor", "trade war"],
                    },
                    "europe": {
                        "id": "europe",
                        "name": "European Union",
                        "keywords": ["European Union", "EU", "Brexit", "Eurozone", "NATO"],
                    },
                    "emerging": {
                        "id": "emerging",
                        "name": "Emerging Powers",
                        "keywords": ["BRICS", "emerging markets", "India", "Global South", "multipolar"],
                    },
                },
            },
            "domestic": {
                "id": "domestic",
                "name": "Domestic Politics",
                "niches": {
                    "elections": {
                        "id": "elections",
                        "name": "Elections & Voting",
                        "keywords": ["election", "voting", "polls", "campaign", "electoral"],
                    },
                    "polarization": {
                        "id": "polarization",
                        "name": "Polarization & Media",
                        "keywords": ["polarization", "misinformation", "media", "social media", "partisan"],
                    },
                    "policy": {
                        "id": "policy",
                        "name": "Policy Analysis",
                        "keywords": ["policy analysis", "legislation", "regulation", "reform"],
                    },
                },
            },
            "governance": {
                "id": "governance",
                "name": "Governance & Institutions",
                "niches": {
                    "democracy": {
                        "id": "democracy",
                        "name": "Democratic Institutions",
                        "keywords": ["democracy", "democratic institutions", "rule of law", "authoritarianism"],
                    },
                    "tech-regulation": {
                        "id": "tech-regulation",
                        "name": "Tech Regulation",
                        "keywords": ["tech regulation", "AI regulation", "antitrust", "data privacy", "GDPR"],
                    },
                    "international-law": {
                        "id": "international-law",
                        "name": "International Law",
                        "keywords": ["international law", "UN", "sanctions", "human rights", "ICC"],
                    },
                },
            },
            "security": {
                "id": "security",
                "name": "Security & Defense",
                "niches": {
                    "cyber": {
                        "id": "cyber",
                        "name": "Cybersecurity",
                        "keywords": ["cybersecurity", "cyber attack", "hacking", "ransomware", "APT"],
                    },
                    "military": {
                        "id": "military",
                        "name": "Military Strategy",
                        "keywords": ["military", "defense", "warfare", "strategy", "deterrence"],
                    },
                    "terrorism": {
                        "id": "terrorism",
                        "name": "Terrorism & Counterterrorism",
                        "keywords": ["terrorism", "counterterrorism", "extremism", "radicalization"],
                    },
                },
            },
        },
    },
}


def get_all_topic_paths() -> list[str]:
    """Get all topic paths in the taxonomy."""
    paths = []
    for cat_id, category in TOPIC_TAXONOMY.items():
        for sub_id, subtopic in category.get("subtopics", {}).items():
            for niche_id in subtopic.get("niches", {}).keys():
                paths.append(f"{cat_id}/{sub_id}/{niche_id}")
    return paths


def get_topic_info(path: str) -> dict | None:
    """Get information about a specific topic path."""
    parts = path.split("/")
    if len(parts) != 3:
        return None

    cat_id, sub_id, niche_id = parts
    category = TOPIC_TAXONOMY.get(cat_id)
    if not category:
        return None

    subtopic = category.get("subtopics", {}).get(sub_id)
    if not subtopic:
        return None

    niche = subtopic.get("niches", {}).get(niche_id)
    if not niche:
        return None

    return {
        "path": path,
        "category": {
            "id": cat_id,
            "name": category.get("name"),
            "icon": category.get("icon"),
        },
        "subtopic": {
            "id": sub_id,
            "name": subtopic.get("name"),
        },
        "niche": {
            "id": niche_id,
            "name": niche.get("name"),
            "keywords": niche.get("keywords", []),
        },
        "content_type": category.get("content_type", "research"),
    }


def get_taxonomy_for_frontend() -> dict:
    """Get taxonomy formatted for frontend consumption."""
    return TOPIC_TAXONOMY
