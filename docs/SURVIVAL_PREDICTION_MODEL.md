# The Noiseless Newspaper: Survival Prediction Model

## Executive Summary

The Noiseless Newspaper shows users **one article per day** — the single most valuable piece of content for their interests. But how do we know what's "valuable"?

Our core insight: **Signal survives time.**

An article that still matters to someone 1 year after reading it is fundamentally more valuable than one forgotten after a week. We ask users to vote on relevance at three intervals (1 week, 1 month, 1 year), with later votes carrying more weight. This creates a "ground truth" dataset of what content has lasting value.

The **Survival Prediction Model** learns from this collective wisdom to predict which *new* articles will stand the test of time — before anyone has voted on them.

---

## The Problem

When a new article is published, we face a cold-start problem:

```
New Article → No votes yet → How do we rank it?
```

Traditional approaches use proxies like citation counts or recency. But these don't capture *lasting human value* — they capture academic impact or newsworthiness, which are different things.

**We want to answer:** "If we showed this article to users today, would they still find it valuable in a year?"

---

## The Solution: Learning from Collective Wisdom

### Phase 1: Collect Ground Truth

Users read articles and vote on them at three time intervals:

| Vote Timing | Weight | What It Measures |
|-------------|--------|------------------|
| 1 week | 15% | Initial reaction, immediate utility |
| 1 month | 35% | Sustained relevance, practical value |
| 1 year | 50% | Lasting impact, fundamental insight |

The **Survival Score** is the weighted average:

```
Survival Score = 0.15 × (1-week vote) + 0.35 × (1-month vote) + 0.50 × (1-year vote)
```

Articles with high survival scores are our "survivors" — content that proved valuable over time.

### Phase 2: Extract Predictive Features

For every article (survivors and non-survivors), we extract features that might predict survival:

**Content Features:**
- Title characteristics (length, question vs. statement, hedging language)
- Abstract complexity (reading level, technical density)
- Claim type (discovery, analysis, opinion, synthesis)
- Topic evergreen score (is this topic timeless or trending?)

**Source Features:**
- Publication/journal historical survival rate
- Author historical survival rate
- Source type (academic, mainstream news, specialized outlet)

**Citation Features:**
- Initial citation count
- Citation velocity (citations per day in first week)
- Cross-domain citation ratio
- Citation from high-impact sources

**Engagement Features (from other users):**
- Early vote variance (do people agree or disagree?)
- 1-week to 1-month vote correlation
- Read completion rate (if available)

**Temporal Features:**
- Day of week published
- Time since last "survivor" in this topic
- Topic trending score at publication time

### Phase 3: Train the Prediction Model

With enough data (target: 1000+ articles with 1-year votes), we train a model:

```
Input: Article features at publication time
Output: Predicted Survival Score (0-1)
```

**Model progression:**
1. **Bootstrap (0-100 articles):** Heuristics only (citations + source reputation)
2. **Early Learning (100-1000 articles):** Logistic regression on top features
3. **Mature (1000+ articles):** Gradient boosting (XGBoost/LightGBM) with full feature set
4. **Advanced (10000+ articles):** Neural network with text embeddings

### Phase 4: Integrate into Ranking

The predicted survival score becomes a key signal in our ranking formula:

```
Final Score = λ × Vote-Based Score + (1-λ) × Prediction-Based Score

Where:
- λ = sigmoid(total_votes / threshold)  # Transitions from prediction to actual votes
- Vote-Based Score = weighted average of actual user votes
- Prediction-Based Score = f(citations, recency, topic_relevance, predicted_survival)
```

**The key insight:** For new articles with no votes, we rely on the prediction model. As votes accumulate, we gradually trust actual human feedback more than predictions.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         THE NOISELESS NEWSPAPER                             │
│                      Survival Prediction Pipeline                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   INGEST     │     │   EXTRACT    │     │   PREDICT    │     │    RANK      │
│              │────▶│              │────▶│              │────▶│              │
│ New Articles │     │  Features    │     │  Survival    │     │  Final Score │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                │
                                                │ Model improves
                                                │ over time
                                                ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   COLLECT    │     │   LABEL      │     │   TRAIN      │
│              │◀────│              │◀────│              │
│  User Votes  │     │  Survival    │     │  Prediction  │
│  (delayed)   │     │  Scores      │     │  Model       │
└──────────────┘     └──────────────┘     └──────────────┘

                    ▲                                    │
                    │         Feedback Loop              │
                    └────────────────────────────────────┘
```

### Data Flow

1. **Ingestion:** Articles arrive from sources (arXiv, news APIs, etc.)
2. **Feature Extraction:** Extract ~50 features per article
3. **Prediction:** Model predicts survival score for new articles
4. **Ranking:** Combine prediction with other signals (recency, topic relevance)
5. **Serving:** Show top article to each user
6. **Voting:** Users vote at 1 week, 1 month, 1 year intervals
7. **Labeling:** Compute actual survival scores from votes
8. **Training:** Retrain model with new labeled data (weekly batch)

---

## The Reinforcement Learning Framing

While we implement this as supervised learning (predicting survival scores), the system has RL characteristics:

| RL Concept | Our Implementation |
|------------|-------------------|
| **Agent** | The recommendation system |
| **Environment** | Users + their evolving interests |
| **State** | Article features at serving time |
| **Action** | Decide which article to show |
| **Reward** | Time-weighted vote (delayed reward!) |
| **Policy** | The survival prediction model |

**Key RL insight:** The reward is *delayed*. We don't know if showing an article was the right decision until up to a year later. This is why we frame it as "survival prediction" — we're learning to predict delayed rewards.

**Exploration vs. Exploitation:**
- **Exploitation:** Show articles with highest predicted survival
- **Exploration:** Occasionally show uncertain articles to gather data
- **Implementation:** ε-greedy or Thompson sampling on predicted scores

---

## Metrics & Success Criteria

### Model Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Survival Prediction AUC | > 0.70 | Can we distinguish survivors from non-survivors? |
| Mean Absolute Error | < 0.15 | How close are predicted vs actual survival scores? |
| Feature Importance Stability | > 0.80 | Are the same features predictive over time? |

### User Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| 1-Year Vote Completion Rate | > 30% | Do users return to vote after a year? |
| Average Survival Score | Increasing | Are we getting better at picking survivors? |
| User Retention | > 50% at 1 year | Do users stick with the platform? |

### Business Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| DAU/MAU Ratio | > 0.3 | Daily engagement (remember: 1 article/day!) |
| NPS | > 50 | Would users recommend us? |
| Time to First Survivor | < 30 days | How quickly can new users experience value? |

---

## Cold Start Strategies

### New User Cold Start

A new user has no vote history. We:
1. Ask for topic interests during onboarding
2. Show articles with high *global* survival scores in their topics
3. Personalize as their votes accumulate

### New Article Cold Start

A new article has no votes. We:
1. Extract features immediately
2. Predict survival score using the model
3. Blend with citation-based PageRank score
4. As votes come in, transition to vote-based ranking (λ sigmoid)

### New Topic Cold Start

A new topic has no historical data. We:
1. Find similar existing topics (embedding similarity)
2. Transfer feature weights from similar topics
3. Apply higher exploration rate to gather data faster

---

## Privacy & Ethics

### Data Collection
- We only collect votes, not reading behavior
- Votes are aggregated; individual patterns are not exposed
- Users can delete their vote history

### Algorithmic Fairness
- We monitor survival rates across source types (to avoid bias toward certain outlets)
- We ensure diverse topics are represented in recommendations
- We don't optimize for engagement; we optimize for lasting value

### Transparency
- Users can see why an article was recommended (feature explanation)
- Model weights and feature importance are published
- Survival scores are visible on articles

---

## Implementation Phases

### Phase 1: Foundation (Months 1-2)
- [ ] Implement feature extraction pipeline
- [ ] Build heuristic-based survival prediction (citations + source reputation)
- [ ] Integrate into ranking with configurable weight
- [ ] Set up vote collection infrastructure

### Phase 2: Learning (Months 3-6)
- [ ] Accumulate 500+ articles with 1-month votes
- [ ] Train first logistic regression model
- [ ] A/B test model predictions vs. heuristics
- [ ] Implement model retraining pipeline

### Phase 3: Maturity (Months 6-12)
- [ ] Accumulate 1000+ articles with 1-year votes
- [ ] Graduate to gradient boosting model
- [ ] Add exploration/exploitation strategy
- [ ] Implement personalized survival prediction

### Phase 4: Scale (Year 2+)
- [ ] Neural network with text embeddings
- [ ] Cross-topic transfer learning
- [ ] Real-time feature updates
- [ ] Causal inference for feature importance

---

## FAQ

**Q: Why not just use engagement metrics like clicks or time-on-page?**

A: Engagement measures *interest*, not *value*. Clickbait gets high engagement but low survival. We optimize for what users say matters to them after reflection, not in the moment.

**Q: Won't users forget to vote after a year?**

A: Yes, many will. That's okay — we only need a subset of votes to train the model. We also send gentle reminders and make voting frictionless (one tap).

**Q: What if the model learns to predict what's *popular* rather than what's *valuable*?**

A: Great question. The time-weighting helps here — popularity fades, value persists. We also monitor for this by checking if high-survival articles cluster around certain "viral" features.

**Q: How do you handle topics where a year is too long (e.g., breaking news)?**

A: We adjust time horizons by content type. News might use 1 day / 1 week / 1 month. Academic content uses 1 week / 1 month / 1 year. The principle (later votes matter more) stays the same.

**Q: Can users game the system by voting strategically?**

A: Votes are private and there's no incentive to game (no likes/followers). Strategic voting would require coordinated effort across time periods, which is costly. We also detect anomalies.

---

## Appendix: Feature Dictionary

| Feature Name | Type | Description | Expected Predictive Power |
|--------------|------|-------------|--------------------------|
| `citation_count_initial` | int | Citations at publication | Medium |
| `citation_velocity_7d` | float | Citations per day in first week | High |
| `source_historical_survival` | float | Avg survival score from this source | High |
| `author_historical_survival` | float | Avg survival score from this author | Medium |
| `title_length` | int | Number of words in title | Low |
| `title_has_question` | bool | Title contains "?" | Low |
| `abstract_complexity` | float | Flesch-Kincaid reading level | Medium |
| `topic_evergreen_score` | float | Historical topic survival rate | Medium |
| `cross_domain_citation_ratio` | float | % citations from other fields | High |
| `early_vote_variance` | float | Std dev of 1-week votes | Medium |
| `claim_confidence_score` | float | NLP: hedging vs certainty | Medium |
| `novelty_score` | float | Embedding distance from recent articles | Medium |

---

*Document version: 1.0*
*Last updated: February 2026*
*Author: The Noiseless Newspaper Team*
