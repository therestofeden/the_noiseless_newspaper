# The Noiseless Newspaper

> **Less (noise) is More.** Signal survives time.

One article per day. Chosen by what matters over time, not what trends right now.

## 🎯 The Concept

Most content platforms optimize for engagement, which selects for novelty and outrage. The Noiseless Newspaper inverts this by optimizing for **retrospective relevance** - content that users rate as important long after they first encountered it.

**How it works:**
1. Select your interests from a 3-level topic taxonomy
2. Each day, choose ONE topic to read about
3. Receive THE single best article for that topic
4. Vote on relevance at 1 week, 1 month, and 1 year
5. Later votes count more (signal survives time)

## 🚀 Quick Start

### Frontend Demo (No Setup Required)

Open `frontend/index.html` in any browser to try the interactive prototype.

Or visit the live demo: [https://therestofeden.github.io/the_noiseless_newspaper/](https://therestofeden.github.io/the_noiseless_newspaper/)

### Backend API

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
python -m app.main
```

Visit http://localhost:8000/docs for API documentation.

## 📁 Project Structure

```
the_noiseless_newspaper/
├── frontend/               # Interactive prototype
│   ├── index.html         # Standalone demo (works offline)
│   └── noiseless-newspaper.jsx  # React component
├── backend/               # Python API
│   ├── app/
│   │   ├── api/          # FastAPI routes
│   │   ├── core/         # Topic taxonomy
│   │   ├── jobs/         # Daily ingestion pipeline
│   │   ├── models/       # Database & domain models
│   │   ├── services/     # Ranking algorithm, PageRank, embeddings
│   │   └── sources/      # arXiv, Semantic Scholar, OpenAlex, NewsAPI
│   ├── pyproject.toml
│   └── README.md
└── README.md
```

## 🧠 The Algorithm

### Cold Start (New Articles)
```
InitialScore = PageRank(citation_graph)
```
Articles without votes are ranked by their position in the academic citation network.

### Time-Weighted Voting
| Period | Weight | Why |
|--------|--------|-----|
| 1 week | 15% | Initial impression, may be hype |
| 1 month | 35% | Some perspective gained |
| 1 year | **50%** | True long-term signal |

### Final Ranking
```
λ = sigmoid(total_votes / threshold)
FinalScore = (1-λ) × CitationScore + λ × VoteScore
```

As articles accumulate votes, we trust human judgment over citations.

## 📚 Topic Coverage

- **AI & Machine Learning** - LLMs, Computer Vision, RL, Theory
- **Physics** - Complexity Science, Quantum, Condensed Matter, Astrophysics
- **Biotechnology** - Gene Editing, Drug Discovery, Synthetic Biology, Neuroscience
- **Economics & Finance** - Macro, Markets, Behavioral, Development
- **Politics & Geopolitics** - International Relations, Domestic, Governance, Security

Each category has 4 subtopics, each with 3-4 specific niches (60+ topics total).

## 🛠 Tech Stack

**Frontend:**
- React 18 + Tailwind CSS
- Standalone HTML (no build step for demo)
- localStorage for preferences

**Backend:**
- Python 3.11 + FastAPI
- SQLAlchemy (async) + SQLite/PostgreSQL
- NetworkX for PageRank
- sentence-transformers for embeddings
- APScheduler for daily jobs

**Data Sources:**
- arXiv API (preprints)
- Semantic Scholar (citations)
- OpenAlex (250M papers, free)
- NewsAPI (current events)

## 🤝 Contributing

This is an early-stage project. Ideas, feedback, and contributions welcome!

1. Fork the repo
2. Create a feature branch
3. Submit a PR

## 📜 License

MIT

---

*Built on the insight that what feels important today often isn't.*
