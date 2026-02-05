# The Noiseless Newspaper - Data Sources

This document catalogs primary sources for each topic domain, including API access details, rate limits, and content coverage.

> **Note**: Starting with English-language sources (Global/US/UK focus). As we expand to additional languages, we'll add regional sources for each language market.

---

## 1. Artificial Intelligence & Machine Learning

| # | Source | Type | API Access | Rate Limits | Content Coverage |
|---|--------|------|------------|-------------|------------------|
| 1 | **arXiv** | Academic Preprints | REST API (free, no key) | 1 req/3 sec | cs.AI, cs.LG, cs.CL, stat.ML - cutting-edge ML research |
| 2 | **Semantic Scholar** | Academic Search | REST API (free key) | 100 req/5 min (free) | 200M+ papers, citation graphs, author data |
| 3 | **Papers With Code** | Research + Code | REST API (free) | Reasonable use | Papers with implementations, SOTA benchmarks |
| 4 | **Hugging Face Daily Papers** | Curated Research | RSS + API | No strict limits | Community-curated trending papers |
| 5 | **MIT Technology Review** | Tech Journalism | RSS Feed | N/A | AI/ML analysis, industry implications |
| 6 | **The Gradient** | Long-form Analysis | RSS Feed | N/A | In-depth ML essays, interviews |
| 7 | **Import AI Newsletter** | Industry Newsletter | RSS/Archive | N/A | Weekly AI developments, policy |
| 8 | **Google AI Blog** | Corporate Research | RSS Feed | N/A | Google's ML research announcements |
| 9 | **OpenAI Blog** | Corporate Research | RSS Feed | N/A | OpenAI research, safety updates |
| 10 | **DeepMind Blog** | Corporate Research | RSS Feed | N/A | DeepMind research publications |

### API Implementation Details

```python
# arXiv API
BASE_URL = "http://export.arxiv.org/api/query"
CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE", "stat.ML"]
# Query: search_query=cat:cs.AI&start=0&max_results=100&sortBy=submittedDate

# Semantic Scholar API
BASE_URL = "https://api.semanticscholar.org/graph/v1"
HEADERS = {"x-api-key": "YOUR_KEY"}  # Optional for higher limits
# Endpoint: /paper/search?query=machine+learning&fields=title,abstract,citationCount

# Papers With Code API
BASE_URL = "https://paperswithcode.com/api/v1"
# Endpoint: /papers/?ordering=-published&page=1

# Hugging Face Papers
RSS_URL = "https://huggingface.co/papers/rss"
```

---

## 2. Physics

| # | Source | Type | API Access | Rate Limits | Content Coverage |
|---|--------|------|------------|-------------|------------------|
| 1 | **arXiv** | Academic Preprints | REST API (free) | 1 req/3 sec | physics.*, hep-*, cond-mat, quant-ph |
| 2 | **APS Physics** | Peer-reviewed | RSS Feeds | N/A | Physical Review journals, Physics magazine |
| 3 | **Nature Physics** | Peer-reviewed | RSS Feed | N/A | High-impact physics research |
| 4 | **Physics Today** | News/Analysis | RSS Feed | N/A | Industry news, career, analysis |
| 5 | **Quanta Magazine** | Science Journalism | RSS Feed | N/A | Accessible physics explanations |
| 6 | **CERN News** | Institutional | RSS Feed | N/A | Particle physics updates |
| 7 | **Phys.org** | News Aggregator | RSS Feed | N/A | Physics news from multiple sources |
| 8 | **Science (AAAS)** | Peer-reviewed | RSS Feed | N/A | Multidisciplinary, high-impact |
| 9 | **IOP Physics World** | News/Features | RSS Feed | N/A | Physics news, career content |
| 10 | **NASA Science** | Institutional | RSS + API | Reasonable | Astrophysics, planetary science |

### API Implementation Details

```python
# arXiv Physics Categories
PHYSICS_CATS = [
    "physics.gen-ph", "hep-th", "hep-ph", "hep-ex",
    "cond-mat", "quant-ph", "astro-ph", "gr-qc"
]

# NASA API
BASE_URL = "https://api.nasa.gov"
API_KEY = "DEMO_KEY"  # or register for free key
# Endpoints: /planetary/apod, /neo/rest/v1/feed

# APS Journals RSS
FEEDS = {
    "prl": "https://feeds.aps.org/rss/recent/prl.xml",
    "prx": "https://feeds.aps.org/rss/recent/prx.xml",
    "physics": "https://physics.aps.org/feed"
}
```

---

## 3. Economics & Finance

| # | Source | Type | API Access | Rate Limits | Content Coverage |
|---|--------|------|------------|-------------|------------------|
| 1 | **NBER Working Papers** | Academic Research | RSS Feed | N/A | Pre-publication economics research |
| 2 | **SSRN Economics** | Preprints | RSS + Search | Limited | Economics/finance working papers |
| 3 | **Federal Reserve (FRED)** | Economic Data | REST API (free key) | 120 req/min | US economic indicators, time series |
| 4 | **World Bank Data** | Global Data | REST API (free) | Reasonable | Development indicators, global economics |
| 5 | **IMF Publications** | Policy Research | RSS Feed | N/A | Global economic policy, working papers |
| 6 | **The Economist** | News/Analysis | RSS Feed | N/A | Global economics, politics analysis |
| 7 | **Financial Times** | News | RSS (limited) | N/A | Markets, finance, global business |
| 8 | **Brookings Institution** | Think Tank | RSS Feed | N/A | Economic policy research |
| 9 | **VoxEU** | Research Portal | RSS Feed | N/A | Policy analysis, research summaries |
| 10 | **Peterson Institute** | Think Tank | RSS Feed | N/A | International economics, trade policy |

### API Implementation Details

```python
# FRED API
BASE_URL = "https://api.stlouisfed.org/fred"
API_KEY = "YOUR_FRED_KEY"  # Free registration
# Endpoint: /series/observations?series_id=GDP&api_key=KEY&file_type=json

# World Bank API
BASE_URL = "https://api.worldbank.org/v2"
# Endpoint: /country/all/indicator/NY.GDP.MKTP.CD?format=json

# NBER RSS
RSS_URL = "https://www.nber.org/rss/new.xml"

# SSRN Economics
RSS_URL = "https://papers.ssrn.com/sol3/Jeljour_results.cfm?form_name=journalbrowse&journal_id=212398&Network=no&SortOrder=ab_approval_date&stype=rss"
```

---

## 4. Biotechnology

| # | Source | Type | API Access | Rate Limits | Content Coverage |
|---|--------|------|------------|-------------|------------------|
| 1 | **PubMed/NCBI** | Academic Database | REST API (free) | 3 req/sec (no key) | Biomedical literature, life sciences |
| 2 | **bioRxiv** | Preprints | RSS + API | Reasonable | Biology preprints |
| 3 | **medRxiv** | Preprints | RSS + API | Reasonable | Medical/clinical preprints |
| 4 | **Nature Biotechnology** | Peer-reviewed | RSS Feed | N/A | High-impact biotech research |
| 5 | **Science Translational Medicine** | Peer-reviewed | RSS Feed | N/A | Translational research |
| 6 | **STAT News** | Biotech Journalism | RSS Feed | N/A | Pharma, biotech industry news |
| 7 | **GenomeWeb** | Industry News | RSS Feed | N/A | Genomics, sequencing, diagnostics |
| 8 | **Endpoints News** | Industry News | RSS Feed | N/A | Biopharma business, clinical trials |
| 9 | **The Scientist** | Science News | RSS Feed | N/A | Life sciences news, features |
| 10 | **NIH News** | Institutional | RSS Feed | N/A | NIH research, funding announcements |

### API Implementation Details

```python
# PubMed E-utilities
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# Search: esearch.fcgi?db=pubmed&term=CRISPR&retmax=100&retmode=json
# Fetch: efetch.fcgi?db=pubmed&id=12345,67890&retmode=xml

# bioRxiv API
BASE_URL = "https://api.biorxiv.org"
# Endpoint: /details/biorxiv/2024-01-01/2024-01-31

# medRxiv API
BASE_URL = "https://api.medrxiv.org"
# Endpoint: /details/medrxiv/2024-01-01/2024-01-31

# ClinicalTrials.gov API
BASE_URL = "https://clinicaltrials.gov/api/v2"
# Endpoint: /studies?query.term=gene+therapy&pageSize=100
```

---

## 5. Politics & Policy

| # | Source | Type | API Access | Rate Limits | Content Coverage |
|---|--------|------|------------|-------------|------------------|
| 1 | **Congress.gov** | Legislative Data | REST API (free key) | 5000/day | US legislation, votes, members |
| 2 | **ProPublica Congress** | Legislative Analysis | REST API (free key) | 5000/day | Congressional data, voting records |
| 3 | **GovInfo** | Government Docs | REST API (free) | Reasonable | Federal Register, bills, reports |
| 4 | **UK Parliament** | Legislative Data | REST API (free) | Reasonable | UK bills, debates, votes |
| 5 | **Council on Foreign Relations** | Think Tank | RSS Feed | N/A | Foreign policy analysis |
| 6 | **Foreign Affairs** | Policy Journal | RSS Feed | N/A | International relations |
| 7 | **Politico** | Political News | RSS Feed | N/A | US/EU political coverage |
| 8 | **The Hill** | Political News | RSS Feed | N/A | Congress, White House coverage |
| 9 | **Lawfare** | Legal/Policy Blog | RSS Feed | N/A | National security law, policy |
| 10 | **RAND Corporation** | Think Tank | RSS Feed | N/A | Defense, policy research |

### API Implementation Details

```python
# Congress.gov API
BASE_URL = "https://api.congress.gov/v3"
API_KEY = "YOUR_KEY"  # Free registration
# Endpoint: /bill?api_key=KEY&format=json&limit=100

# ProPublica Congress API
BASE_URL = "https://api.propublica.org/congress/v1"
HEADERS = {"X-API-Key": "YOUR_KEY"}
# Endpoint: /117/senate/votes/recent.json

# UK Parliament API
BASE_URL = "https://bills-api.parliament.uk/api/v1"
# Endpoint: /Bills?CurrentHouse=All&SortOrder=DateUpdatedDesc

# GovInfo API
BASE_URL = "https://api.govinfo.gov"
API_KEY = "YOUR_KEY"
# Endpoint: /collections/FR/2024-01-01?api_key=KEY
```

---

## RSS Feed Aggregation Strategy

For sources without APIs, we use RSS feeds with standardized parsing:

```python
import feedparser
from datetime import datetime

RSS_FEEDS = {
    "ai_ml": [
        "https://huggingface.co/papers/rss",
        "https://www.technologyreview.com/feed/",
        "https://thegradient.pub/rss/",
        "https://blog.google/technology/ai/rss/",
        "https://openai.com/blog/rss.xml",
    ],
    "physics": [
        "https://physics.aps.org/feed",
        "https://www.nature.com/nphys.rss",
        "https://www.quantamagazine.org/feed/",
        "https://home.cern/api/news/news/feed.rss",
        "https://phys.org/rss-feed/physics-news/",
    ],
    "economics": [
        "https://www.nber.org/rss/new.xml",
        "https://www.imf.org/en/News/rss",
        "https://www.brookings.edu/feed/",
        "https://voxeu.org/rss.xml",
    ],
    "biotech": [
        "https://www.statnews.com/feed/",
        "https://www.genomeweb.com/rss.xml",
        "https://www.the-scientist.com/rss",
        "https://www.nih.gov/news-events/news-releases/feed",
    ],
    "politics": [
        "https://www.cfr.org/rss.xml",
        "https://www.foreignaffairs.com/rss.xml",
        "https://www.politico.com/rss/politics.xml",
        "https://thehill.com/feed/",
        "https://www.lawfaremedia.org/feed",
    ],
}

def parse_feed(url: str) -> list[dict]:
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries:
        articles.append({
            "title": entry.get("title"),
            "url": entry.get("link"),
            "published": entry.get("published_parsed"),
            "summary": entry.get("summary"),
            "source": feed.feed.get("title"),
        })
    return articles
```

---

## API Key Management

Required API keys (all free tier):

| Service | Registration URL | Key Type |
|---------|------------------|----------|
| Semantic Scholar | https://www.semanticscholar.org/product/api | API Key |
| FRED | https://fred.stlouisfed.org/docs/api/api_key.html | API Key |
| Congress.gov | https://api.congress.gov/sign-up/ | API Key |
| ProPublica | https://www.propublica.org/datastore/api/propublica-congress-api | API Key |
| NASA | https://api.nasa.gov/ | API Key |
| GovInfo | https://api.govinfo.gov/docs/ | API Key |

---

## Rate Limiting Strategy

```python
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict

class RateLimiter:
    """Per-source rate limiting to respect API constraints."""

    LIMITS = {
        "arxiv": {"requests": 1, "period": 3},  # 1 req per 3 sec
        "semantic_scholar": {"requests": 100, "period": 300},  # 100 per 5 min
        "pubmed": {"requests": 3, "period": 1},  # 3 per second
        "fred": {"requests": 120, "period": 60},  # 120 per minute
        "congress": {"requests": 5000, "period": 86400},  # 5000 per day
    }

    def __init__(self):
        self.request_times = defaultdict(list)

    async def acquire(self, source: str):
        if source not in self.LIMITS:
            return

        limit = self.LIMITS[source]
        now = datetime.now()
        cutoff = now - timedelta(seconds=limit["period"])

        # Clean old requests
        self.request_times[source] = [
            t for t in self.request_times[source] if t > cutoff
        ]

        # Wait if at limit
        while len(self.request_times[source]) >= limit["requests"]:
            await asyncio.sleep(0.1)
            now = datetime.now()
            cutoff = now - timedelta(seconds=limit["period"])
            self.request_times[source] = [
                t for t in self.request_times[source] if t > cutoff
            ]

        self.request_times[source].append(now)
```

---

## Content Deduplication

Articles may appear in multiple sources. We deduplicate using:

1. **URL normalization**: Remove tracking parameters, normalize domains
2. **Title similarity**: Fuzzy matching with >0.9 threshold
3. **DOI matching**: For academic papers with DOIs
4. **Content fingerprinting**: SimHash for near-duplicate detection

```python
from urllib.parse import urlparse, parse_qs, urlencode
import re

def normalize_url(url: str) -> str:
    """Remove tracking params, normalize URL."""
    parsed = urlparse(url)

    # Remove common tracking params
    TRACKING_PARAMS = {'utm_source', 'utm_medium', 'utm_campaign', 'ref', 'source'}
    params = parse_qs(parsed.query)
    clean_params = {k: v for k, v in params.items() if k not in TRACKING_PARAMS}

    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

def extract_doi(text: str) -> str | None:
    """Extract DOI from text."""
    doi_pattern = r'10\.\d{4,}/[^\s]+'
    match = re.search(doi_pattern, text)
    return match.group(0) if match else None
```

---

## Future Language Expansion

When adding new languages, we'll add regional sources:

### German (DE/AT/CH)
- Der Spiegel, FAZ, NZZ (news)
- Max Planck Society (research)
- Deutsche Welle (international)

### French (FR/BE/CH/CA)
- Le Monde, Libération (news)
- CNRS, INSERM (research)
- France 24 (international)

### Spanish (ES/MX/AR)
- El País, La Nación (news)
- CSIC (research)

### Chinese (CN/TW)
- Xinhua, South China Morning Post (news)
- Chinese Academy of Sciences (research)

---

## Implementation Priority

### Phase 1 (MVP)
1. arXiv (AI/ML + Physics)
2. PubMed (Biotech)
3. RSS aggregation for news sources
4. Basic rate limiting

### Phase 2 (Enhanced)
1. Semantic Scholar integration
2. FRED economic data
3. Congress.gov legislative data
4. Deduplication pipeline

### Phase 3 (Scale)
1. Full API key rotation
2. Caching layer (Redis)
3. Multi-language sources
4. Real-time streaming
