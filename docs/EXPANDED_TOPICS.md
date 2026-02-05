# The Noiseless Newspaper - Expanded Topic System

This document outlines the expanded topic taxonomy, adding 6 new general news categories to complement the original 5 deep research topics.

## Philosophy

The Noiseless Newspaper serves **timeless insights** - content that:
- Reveals underlying mechanisms (explains WHY, not just WHAT)
- Survives replication and scrutiny
- Will matter in 5-10 years

Even for "news" topics like Sports or Entertainment, we prioritize stories with lasting significance over ephemeral updates.

---

## Complete Topic Taxonomy

### Original Deep Topics (Research-Focused)

| Domain | Icon | Subtopics |
|--------|------|-----------|
| **AI & Machine Learning** | ◈ | Deep Learning, NLP, Computer Vision, Robotics & Embodied AI |
| **Physics** | ◉ | Quantum, Particle & High-Energy, Condensed Matter, Astrophysics |
| **Economics & Finance** | ◆ | Macro, Behavioral, Development, Financial Markets |
| **Biotechnology** | ◇ | Genomics & CRISPR, Drug Discovery, Synthetic Biology, Neuroscience |
| **Politics & Policy** | ▣ | US Politics, International Relations, Public Policy, Political Economy |

### New General Topics (News-Focused)

| Domain | Icon | Subtopics |
|--------|------|-----------|
| **Sports** | ⚽ | Football/Soccer, American Sports, Tennis & Golf, Olympics & Athletics |
| **Entertainment & Culture** | 🎬 | Film & Television, Music, Books & Literature, Art & Design |
| **Technology** | 📱 | Consumer Tech, Social Media & Internet, Gaming, Cybersecurity |
| **Business & Markets** | 📈 | Corporate News, Startups & VC, Markets & Investing, Management & Strategy |
| **World News** | 🌍 | Geopolitics, Conflicts & Security, Diplomacy, Regional News |
| **Environment & Climate** | 🌱 | Climate Science, Energy Transition, Conservation, Sustainability |

---

## Detailed Subtopic Taxonomy

### 6. Sports ⚽

```
sports/
├── football-soccer/
│   ├── premier-league
│   ├── champions-league
│   ├── world-cup
│   └── transfers-rumors
├── american-sports/
│   ├── nfl
│   ├── nba
│   ├── mlb
│   └── nhl
├── tennis-golf/
│   ├── grand-slams
│   ├── pga-tour
│   └── rankings
└── olympics-athletics/
    ├── track-field
    ├── swimming
    └── winter-sports
```

**Signal over noise**: Focus on significant trades, championship outcomes, records broken, rule changes affecting the sport - not daily game recaps.

### 7. Entertainment & Culture 🎬

```
entertainment/
├── film-television/
│   ├── theatrical-releases
│   ├── streaming
│   ├── awards-festivals
│   └── industry-business
├── music/
│   ├── album-releases
│   ├── tours-festivals
│   ├── industry-trends
│   └── retrospectives
├── books-literature/
│   ├── fiction
│   ├── nonfiction
│   ├── awards
│   └── publishing-industry
└── art-design/
    ├── visual-arts
    ├── architecture
    ├── fashion
    └── exhibitions
```

**Signal over noise**: Focus on works that will be remembered, cultural shifts, industry transformations - not celebrity gossip.

### 8. Technology 📱

```
technology/
├── consumer-tech/
│   ├── smartphones-devices
│   ├── computing
│   ├── smart-home
│   └── wearables
├── social-internet/
│   ├── social-platforms
│   ├── creator-economy
│   ├── digital-culture
│   └── misinformation
├── gaming/
│   ├── console-pc
│   ├── mobile
│   ├── esports
│   └── industry
└── cybersecurity/
    ├── threats-breaches
    ├── privacy
    ├── regulation
    └── enterprise
```

**Signal over noise**: Focus on product launches that shift markets, platform policy changes with lasting impact, security incidents affecting millions - not gadget rumors.

### 9. Business & Markets 📈

```
business/
├── corporate/
│   ├── earnings-reports
│   ├── mergers-acquisitions
│   ├── executive-changes
│   └── strategy
├── startups-vc/
│   ├── funding-rounds
│   ├── unicorns
│   ├── founder-stories
│   └── ecosystem-trends
├── markets/
│   ├── equities
│   ├── fixed-income
│   ├── commodities
│   └── crypto
└── management/
    ├── leadership
    ├── organizational
    ├── future-of-work
    └── case-studies
```

**Signal over noise**: Focus on strategic moves that reshape industries, funding that signals market direction, management insights that compound - not daily stock moves.

### 10. World News 🌍

```
world/
├── geopolitics/
│   ├── great-power-competition
│   ├── alliances-blocs
│   ├── trade-sanctions
│   └── international-law
├── conflicts/
│   ├── active-wars
│   ├── terrorism
│   ├── humanitarian
│   └── peacekeeping
├── diplomacy/
│   ├── summits-treaties
│   ├── un-institutions
│   ├── bilateral-relations
│   └── foreign-policy
└── regions/
    ├── europe
    ├── asia-pacific
    ├── middle-east
    ├── africa
    └── americas
```

**Signal over noise**: Focus on structural shifts in global order, conflicts with lasting consequences, diplomatic breakthroughs - not daily political theater.

### 11. Environment & Climate 🌱

```
environment/
├── climate-science/
│   ├── research-findings
│   ├── ipcc-reports
│   ├── modeling
│   └── attribution
├── energy-transition/
│   ├── renewables
│   ├── nuclear
│   ├── grid-storage
│   └── policy
├── conservation/
│   ├── biodiversity
│   ├── oceans
│   ├── forests
│   └── species
└── sustainability/
    ├── circular-economy
    ├── sustainable-business
    ├── urban-planning
    └── food-agriculture
```

**Signal over noise**: Focus on breakthrough research, major policy shifts, tipping points - not weather events unless they signal larger patterns.

---

## Source Strategy by Topic Type

### Deep Topics (Academic + News)
- Primary: Academic APIs (arXiv, PubMed, Semantic Scholar)
- Secondary: Quality journalism (MIT Tech Review, Nature News)
- Tertiary: RSS from research blogs

### News Topics (Journalism + Wire Services)
- Primary: Major news outlets (Reuters, AP, BBC, NYT)
- Secondary: Specialized publications (ESPN, Variety, TechCrunch)
- Tertiary: Wire services and press releases

---

## Implementation Notes

### TopicDomain Enum Extension

```python
class TopicDomain(str, Enum):
    # Original deep topics
    AI_ML = "ai-ml"
    PHYSICS = "physics"
    ECONOMICS = "economics"
    BIOTECH = "biotech"
    POLITICS = "politics"

    # New general topics
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    WORLD = "world"
    ENVIRONMENT = "environment"
```

### Survival Model Considerations

The survival prediction model may behave differently for news vs. academic content:

| Aspect | Academic Topics | News Topics |
|--------|-----------------|-------------|
| Citation signals | Strong predictor | Weak/absent |
| Source authority | Journal impact factor | Outlet reputation |
| Recency importance | Moderate | Higher |
| Voting patterns | Slower, more stable | Faster, more volatile |

Consider training separate models or adding topic-type features.

---

## Frontend Updates

The taxonomy in `frontend/index.html` should be updated to include all 11 topics with their subtopics and visual styling.

Color palette suggestion for new topics:
- Sports: Athletic blue (#3B82F6)
- Entertainment: Purple (#8B5CF6)
- Technology: Cyan (#06B6D4)
- Business: Gold (#F59E0B)
- World: Deep blue (#1E40AF)
- Environment: Green (#10B981)
