import React, { useState, useEffect } from 'react';

// Sample article data across domains
const sampleArticles = [
  {
    id: 1,
    title: "Sparse Autoencoders Reveal Hidden Structure in Large Language Models",
    domain: "AI & Machine Learning",
    source: "arXiv",
    citations: 847,
    date: "2026-01-28",
    summary: "Researchers demonstrate that sparse autoencoders can decompose neural network activations into interpretable features, revealing how language models organize knowledge internally. The technique opens new paths for understanding and steering AI behavior.",
    link: "https://arxiv.org/abs/2401.xxxxx",
    citationTrend: "rising"
  },
  {
    id: 2,
    title: "CRISPR-Based Gene Therapy Shows 94% Efficacy in Sickle Cell Treatment",
    domain: "Biotechnology",
    source: "Nature Medicine",
    citations: 1203,
    date: "2026-01-27",
    summary: "A landmark clinical trial demonstrates near-complete elimination of sickle cell crises in treated patients over a 3-year follow-up period. The one-time treatment modifies patients' own stem cells to produce functional hemoglobin.",
    link: "https://nature.com/articles/xxxxx",
    citationTrend: "stable"
  },
  {
    id: 3,
    title: "Central Banks Coordinate on Digital Currency Interoperability Standard",
    domain: "Finance & Economics",
    source: "BIS Working Papers",
    citations: 312,
    date: "2026-01-26",
    summary: "The Bank for International Settlements publishes a framework enabling cross-border CBDC transactions. The protocol addresses privacy, settlement finality, and regulatory compliance across 47 participating nations.",
    link: "https://bis.org/publ/xxxxx",
    citationTrend: "rising"
  },
  {
    id: 4,
    title: "Room-Temperature Superconductor Claim Independently Replicated",
    domain: "Physics",
    source: "Physical Review Letters",
    citations: 2341,
    date: "2026-01-25",
    summary: "Three independent laboratories confirm superconducting behavior in a modified LK-99 variant at temperatures up to 15°C under ambient pressure. The material's complex synthesis remains a barrier to practical applications.",
    link: "https://journals.aps.org/prl/xxxxx",
    citationTrend: "rising"
  },
  {
    id: 5,
    title: "Meta-Analysis Reveals Optimal Intervention Timing for Climate Adaptation",
    domain: "Climate Science",
    source: "Science",
    citations: 891,
    date: "2026-01-24",
    summary: "Analysis of 2,400 climate adaptation projects shows interventions implemented before tipping point indicators are 4.7x more cost-effective. The study provides a decision framework for prioritizing limited adaptation budgets.",
    link: "https://science.org/doi/xxxxx",
    citationTrend: "stable"
  }
];

const interests = [
  { id: 'ai', name: 'AI & Machine Learning', icon: '🤖' },
  { id: 'bio', name: 'Biotechnology', icon: '🧬' },
  { id: 'finance', name: 'Finance & Economics', icon: '📈' },
  { id: 'physics', name: 'Physics', icon: '⚛️' },
  { id: 'climate', name: 'Climate Science', icon: '🌍' },
  { id: 'neuro', name: 'Neuroscience', icon: '🧠' },
  { id: 'space', name: 'Space & Astronomy', icon: '🚀' },
  { id: 'materials', name: 'Materials Science', icon: '🔬' }
];

// Voting history for timeline demo
const votingHistory = [
  { articleId: 1, title: "Quantum Error Correction Breakthrough", votedAt: '1 week', relevanceScore: 4, currentScore: 3.2 },
  { articleId: 2, title: "mRNA Vaccine Platform Expansion", votedAt: '1 month', relevanceScore: 5, currentScore: 4.8 },
  { articleId: 3, title: "Fusion Net Energy Milestone", votedAt: '1 year', relevanceScore: 5, currentScore: 4.9, landmark: true }
];

export default function NoiselessNewspaper() {
  const [screen, setScreen] = useState('landing');
  const [selectedInterests, setSelectedInterests] = useState([]);
  const [todayChoice, setTodayChoice] = useState(null);
  const [currentArticle, setCurrentArticle] = useState(null);
  const [showAlgorithm, setShowAlgorithm] = useState(false);
  const [algorithmMode, setAlgorithmMode] = useState('simple');
  const [timeSimulation, setTimeSimulation] = useState(null);
  const [votingQueue, setVotingQueue] = useState([]);
  const [votes, setVotes] = useState({});
  const [animatingNodes, setAnimatingNodes] = useState(false);

  // Time simulation effect
  useEffect(() => {
    if (timeSimulation) {
      const timer = setTimeout(() => {
        if (timeSimulation === 'week') {
          setVotingQueue([{ ...sampleArticles[0], period: '1 week' }]);
        } else if (timeSimulation === 'month') {
          setVotingQueue([{ ...sampleArticles[1], period: '1 month' }]);
        } else if (timeSimulation === 'year') {
          setVotingQueue([{ ...sampleArticles[2], period: '1 year' }]);
        }
        setTimeSimulation(null);
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [timeSimulation]);

  const toggleInterest = (id) => {
    setSelectedInterests(prev =>
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    );
  };

  const handleVote = (articleId, score) => {
    setVotes(prev => ({ ...prev, [articleId]: score }));
    setVotingQueue(prev => prev.filter(a => a.id !== articleId));
    setAnimatingNodes(true);
    setTimeout(() => setAnimatingNodes(false), 2000);
  };

  // Landing Page
  if (screen === 'landing') {
    return (
      <div className="min-h-screen bg-stone-50 flex flex-col">
        <nav className="p-6 flex justify-between items-center max-w-5xl mx-auto w-full">
          <div className="text-xl font-light tracking-wide text-stone-800">The Noiseless Newspaper</div>
          <button
            onClick={() => setScreen('onboarding')}
            className="text-sm text-stone-600 hover:text-stone-900 transition-colors"
          >
            Sign in
          </button>
        </nav>

        <main className="flex-1 flex flex-col items-center justify-center px-6 -mt-20">
          <div className="max-w-2xl text-center space-y-8">
            <h1 className="text-5xl md:text-6xl font-light text-stone-900 leading-tight">
              Less <span className="text-stone-400 line-through decoration-1">(noise)</span> is More.
            </h1>

            <p className="text-xl text-stone-600 font-light leading-relaxed max-w-lg mx-auto">
              One article per day. Chosen by what matters over time, not what trends right now.
            </p>

            <div className="pt-4">
              <button
                onClick={() => setScreen('onboarding')}
                className="px-8 py-4 bg-stone-900 text-white rounded-full text-lg font-light hover:bg-stone-800 transition-all hover:shadow-lg"
              >
                Begin your quiet reading
              </button>
            </div>

            <button
              onClick={() => setShowAlgorithm(true)}
              className="text-sm text-stone-500 hover:text-stone-700 underline underline-offset-4 decoration-stone-300"
            >
              How does it work?
            </button>
          </div>
        </main>

        <footer className="p-6 text-center text-stone-400 text-sm">
          Signal survives time.
        </footer>

        {/* Algorithm Explainer Modal */}
        {showAlgorithm && (
          <AlgorithmExplainer
            mode={algorithmMode}
            setMode={setAlgorithmMode}
            onClose={() => setShowAlgorithm(false)}
            animating={animatingNodes}
          />
        )}
      </div>
    );
  }

  // Onboarding - Interest Selection
  if (screen === 'onboarding') {
    return (
      <div className="min-h-screen bg-stone-50 flex flex-col">
        <nav className="p-6 flex justify-between items-center max-w-5xl mx-auto w-full">
          <button onClick={() => setScreen('landing')} className="text-xl font-light tracking-wide text-stone-800">
            The Noiseless Newspaper
          </button>
        </nav>

        <main className="flex-1 flex flex-col items-center justify-center px-6">
          <div className="max-w-xl w-full space-y-8">
            <div className="text-center space-y-3">
              <h2 className="text-3xl font-light text-stone-900">What matters to you?</h2>
              <p className="text-stone-600">Select your areas of interest. You can always change these later.</p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              {interests.map(interest => (
                <button
                  key={interest.id}
                  onClick={() => toggleInterest(interest.id)}
                  className={`p-4 rounded-xl border-2 transition-all text-left ${
                    selectedInterests.includes(interest.id)
                      ? 'border-stone-900 bg-stone-900 text-white'
                      : 'border-stone-200 bg-white hover:border-stone-400'
                  }`}
                >
                  <span className="text-2xl mb-2 block">{interest.icon}</span>
                  <span className="font-medium">{interest.name}</span>
                </button>
              ))}
            </div>

            <button
              onClick={() => setScreen('daily-choice')}
              disabled={selectedInterests.length === 0}
              className={`w-full py-4 rounded-full text-lg font-light transition-all ${
                selectedInterests.length > 0
                  ? 'bg-stone-900 text-white hover:bg-stone-800'
                  : 'bg-stone-200 text-stone-400 cursor-not-allowed'
              }`}
            >
              Continue
            </button>
          </div>
        </main>
      </div>
    );
  }

  // Daily Choice
  if (screen === 'daily-choice') {
    const availableChoices = interests.filter(i => selectedInterests.includes(i.id));

    return (
      <div className="min-h-screen bg-stone-50 flex flex-col">
        <nav className="p-6 flex justify-between items-center max-w-5xl mx-auto w-full">
          <button onClick={() => setScreen('landing')} className="text-xl font-light tracking-wide text-stone-800">
            The Noiseless Newspaper
          </button>
          <button
            onClick={() => setShowAlgorithm(true)}
            className="text-sm text-stone-500 hover:text-stone-700"
          >
            How it works
          </button>
        </nav>

        <main className="flex-1 flex flex-col items-center justify-center px-6">
          <div className="max-w-lg w-full space-y-8 text-center">
            <div className="space-y-2">
              <p className="text-stone-500 text-sm uppercase tracking-wider">January 31, 2026</p>
              <h2 className="text-3xl font-light text-stone-900">What would you like to read about today?</h2>
            </div>

            <div className="space-y-3">
              {availableChoices.map(choice => (
                <button
                  key={choice.id}
                  onClick={() => {
                    setTodayChoice(choice);
                    // Select a random article from the matching domain or nearby
                    const article = sampleArticles.find(a =>
                      a.domain.toLowerCase().includes(choice.name.toLowerCase().split(' ')[0].toLowerCase())
                    ) || sampleArticles[0];
                    setCurrentArticle(article);
                    setScreen('article');
                  }}
                  className="w-full p-5 bg-white rounded-xl border border-stone-200 hover:border-stone-400 transition-all text-left flex items-center gap-4 group"
                >
                  <span className="text-3xl">{choice.icon}</span>
                  <span className="text-lg text-stone-800 group-hover:text-stone-900">{choice.name}</span>
                  <span className="ml-auto text-stone-400 group-hover:text-stone-600">→</span>
                </button>
              ))}
            </div>

            <p className="text-stone-400 text-sm">
              One choice. One article. That's all for today.
            </p>
          </div>
        </main>

        {showAlgorithm && (
          <AlgorithmExplainer
            mode={algorithmMode}
            setMode={setAlgorithmMode}
            onClose={() => setShowAlgorithm(false)}
            animating={animatingNodes}
          />
        )}
      </div>
    );
  }

  // Article View
  if (screen === 'article') {
    return (
      <div className="min-h-screen bg-stone-50">
        <nav className="p-6 flex justify-between items-center max-w-3xl mx-auto">
          <button onClick={() => setScreen('landing')} className="text-xl font-light tracking-wide text-stone-800">
            The Noiseless Newspaper
          </button>
          <div className="flex gap-4">
            <button
              onClick={() => setShowAlgorithm(true)}
              className="text-sm text-stone-500 hover:text-stone-700"
            >
              How it works
            </button>
            <button
              onClick={() => setScreen('timeline')}
              className="text-sm text-stone-500 hover:text-stone-700"
            >
              My Timeline
            </button>
          </div>
        </nav>

        <main className="max-w-3xl mx-auto px-6 py-8">
          <article className="space-y-8">
            <header className="space-y-4">
              <div className="flex items-center gap-3 text-sm text-stone-500">
                <span className="px-3 py-1 bg-stone-200 rounded-full">{currentArticle?.domain}</span>
                <span>{currentArticle?.source}</span>
                <span>•</span>
                <span>{currentArticle?.date}</span>
              </div>

              <h1 className="text-3xl md:text-4xl font-light text-stone-900 leading-snug">
                {currentArticle?.title}
              </h1>

              <div className="flex items-center gap-4 text-sm text-stone-600">
                <div className="flex items-center gap-1">
                  <span>📚</span>
                  <span>{currentArticle?.citations} citations</span>
                  {currentArticle?.citationTrend === 'rising' && (
                    <span className="text-green-600">↑</span>
                  )}
                </div>
              </div>
            </header>

            <div className="prose prose-stone prose-lg max-w-none">
              <p className="text-xl leading-relaxed text-stone-700">
                {currentArticle?.summary}
              </p>
            </div>

            <div className="pt-4 space-y-4">
              <a
                href={currentArticle?.link}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-6 py-3 bg-stone-900 text-white rounded-full hover:bg-stone-800 transition-all"
              >
                Read full article
                <span>↗</span>
              </a>

              <p className="text-stone-500 text-sm">
                We'll ask you about this article's relevance in 1 week, 1 month, and 1 year.
              </p>
            </div>

            {/* Time Simulation Controls */}
            <div className="mt-12 p-6 bg-white rounded-xl border border-stone-200">
              <h3 className="text-lg font-medium text-stone-800 mb-4">⏱ Demo: Simulate Time Passing</h3>
              <p className="text-stone-600 text-sm mb-4">
                In the real app, you'd receive notifications at these intervals. Click to simulate:
              </p>
              <div className="flex gap-3">
                <button
                  onClick={() => setTimeSimulation('week')}
                  disabled={timeSimulation}
                  className="px-4 py-2 bg-stone-100 hover:bg-stone-200 rounded-lg text-sm transition-all disabled:opacity-50"
                >
                  {timeSimulation === 'week' ? '⏳ Loading...' : 'Fast-forward 1 week →'}
                </button>
                <button
                  onClick={() => setTimeSimulation('month')}
                  disabled={timeSimulation}
                  className="px-4 py-2 bg-stone-100 hover:bg-stone-200 rounded-lg text-sm transition-all disabled:opacity-50"
                >
                  {timeSimulation === 'month' ? '⏳ Loading...' : 'Fast-forward 1 month →'}
                </button>
                <button
                  onClick={() => setTimeSimulation('year')}
                  disabled={timeSimulation}
                  className="px-4 py-2 bg-stone-100 hover:bg-stone-200 rounded-lg text-sm transition-all disabled:opacity-50"
                >
                  {timeSimulation === 'year' ? '⏳ Loading...' : 'Fast-forward 1 year →'}
                </button>
              </div>
            </div>
          </article>
        </main>

        {/* Voting Modal */}
        {votingQueue.length > 0 && (
          <VotingModal
            article={votingQueue[0]}
            onVote={handleVote}
            onClose={() => setVotingQueue([])}
          />
        )}

        {showAlgorithm && (
          <AlgorithmExplainer
            mode={algorithmMode}
            setMode={setAlgorithmMode}
            onClose={() => setShowAlgorithm(false)}
            animating={animatingNodes}
          />
        )}
      </div>
    );
  }

  // Timeline View
  if (screen === 'timeline') {
    return (
      <div className="min-h-screen bg-stone-50">
        <nav className="p-6 flex justify-between items-center max-w-3xl mx-auto">
          <button onClick={() => setScreen('landing')} className="text-xl font-light tracking-wide text-stone-800">
            The Noiseless Newspaper
          </button>
          <button
            onClick={() => setScreen('article')}
            className="text-sm text-stone-500 hover:text-stone-700"
          >
            ← Back to article
          </button>
        </nav>

        <main className="max-w-3xl mx-auto px-6 py-8">
          <div className="space-y-8">
            <header className="space-y-2">
              <h2 className="text-3xl font-light text-stone-900">Your Reading Timeline</h2>
              <p className="text-stone-600">See how your past articles have held up over time.</p>
            </header>

            <div className="space-y-4">
              {votingHistory.map((item, idx) => (
                <div
                  key={idx}
                  className={`p-5 bg-white rounded-xl border ${item.landmark ? 'border-amber-300 bg-amber-50' : 'border-stone-200'}`}
                >
                  <div className="flex justify-between items-start">
                    <div className="space-y-1">
                      <h3 className="font-medium text-stone-800">{item.title}</h3>
                      <p className="text-sm text-stone-500">Voted {item.votedAt} ago</p>
                    </div>
                    <div className="text-right">
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-stone-500">Your vote:</span>
                        <span className="font-medium">{item.relevanceScore}/5</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <span className="text-stone-500">Community:</span>
                        <span className={item.currentScore >= 4 ? 'text-green-600' : 'text-stone-600'}>
                          {item.currentScore}/5
                        </span>
                      </div>
                    </div>
                  </div>
                  {item.landmark && (
                    <div className="mt-3 pt-3 border-t border-amber-200">
                      <span className="text-sm text-amber-700">🏆 This article was validated as highly significant by the community over 1 year</span>
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="p-6 bg-stone-100 rounded-xl">
              <h3 className="font-medium text-stone-800 mb-2">Your Signal Score</h3>
              <p className="text-stone-600 text-sm">
                Based on how well your votes predict long-term relevance, your recommendations are weighted.
                Better predictions = more influence on what others see.
              </p>
              <div className="mt-4 flex items-center gap-3">
                <div className="h-2 flex-1 bg-stone-200 rounded-full overflow-hidden">
                  <div className="h-full w-3/4 bg-stone-700 rounded-full"></div>
                </div>
                <span className="text-stone-700 font-medium">75%</span>
              </div>
            </div>
          </div>
        </main>
      </div>
    );
  }

  return null;
}

// Voting Modal Component
function VotingModal({ article, onVote, onClose }) {
  const [hoveredScore, setHoveredScore] = useState(0);
  const [selectedScore, setSelectedScore] = useState(0);

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-6 z-50">
      <div className="bg-white rounded-2xl max-w-md w-full p-8 space-y-6">
        <div className="text-center space-y-2">
          <p className="text-sm text-stone-500 uppercase tracking-wider">
            {article.period} later...
          </p>
          <h3 className="text-2xl font-light text-stone-900">
            How relevant was this article?
          </h3>
        </div>

        <div className="p-4 bg-stone-50 rounded-xl">
          <p className="text-stone-700 font-medium">{article.title}</p>
          <p className="text-sm text-stone-500 mt-1">{article.domain}</p>
        </div>

        <div className="space-y-3">
          <p className="text-sm text-stone-600 text-center">
            Looking back, how important does this feel now?
          </p>
          <div className="flex justify-center gap-2">
            {[1, 2, 3, 4, 5].map(score => (
              <button
                key={score}
                onMouseEnter={() => setHoveredScore(score)}
                onMouseLeave={() => setHoveredScore(0)}
                onClick={() => setSelectedScore(score)}
                className={`w-12 h-12 rounded-full border-2 transition-all text-lg ${
                  (hoveredScore || selectedScore) >= score
                    ? 'border-stone-900 bg-stone-900 text-white'
                    : 'border-stone-300 text-stone-400 hover:border-stone-400'
                }`}
              >
                {score}
              </button>
            ))}
          </div>
          <div className="flex justify-between text-xs text-stone-400 px-1">
            <span>Noise</span>
            <span>Signal</span>
          </div>
        </div>

        <div className="flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 py-3 border border-stone-200 rounded-full text-stone-600 hover:bg-stone-50 transition-all"
          >
            Skip
          </button>
          <button
            onClick={() => selectedScore && onVote(article.id, selectedScore)}
            disabled={!selectedScore}
            className={`flex-1 py-3 rounded-full transition-all ${
              selectedScore
                ? 'bg-stone-900 text-white hover:bg-stone-800'
                : 'bg-stone-200 text-stone-400 cursor-not-allowed'
            }`}
          >
            Submit
          </button>
        </div>
      </div>
    </div>
  );
}

// Algorithm Explainer Component
function AlgorithmExplainer({ mode, setMode, onClose, animating }) {
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-6 z-50 overflow-y-auto">
      <div className="bg-white rounded-2xl max-w-3xl w-full p-8 my-8">
        <div className="flex justify-between items-start mb-6">
          <h2 className="text-2xl font-light text-stone-900">How The Algorithm Works</h2>
          <button onClick={onClose} className="text-stone-400 hover:text-stone-600 text-2xl">×</button>
        </div>

        {/* Mode Toggle */}
        <div className="flex gap-2 mb-8">
          <button
            onClick={() => setMode('simple')}
            className={`px-4 py-2 rounded-full text-sm transition-all ${
              mode === 'simple' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600'
            }`}
          >
            Simple Explanation
          </button>
          <button
            onClick={() => setMode('technical')}
            className={`px-4 py-2 rounded-full text-sm transition-all ${
              mode === 'technical' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600'
            }`}
          >
            Technical Details
          </button>
        </div>

        {mode === 'simple' ? (
          <div className="space-y-8">
            {/* Visual Network Diagram */}
            <div className="relative h-64 bg-stone-50 rounded-xl overflow-hidden">
              <svg viewBox="0 0 400 200" className="w-full h-full">
                {/* Citation Links */}
                <line x1="200" y1="100" x2="80" y2="60" stroke="#d6d3d1" strokeWidth="2" />
                <line x1="200" y1="100" x2="320" y2="60" stroke="#d6d3d1" strokeWidth="2" />
                <line x1="200" y1="100" x2="120" y2="160" stroke="#d6d3d1" strokeWidth="2" />
                <line x1="200" y1="100" x2="280" y2="160" stroke="#d6d3d1" strokeWidth="2" />
                <line x1="80" y1="60" x2="120" y2="160" stroke="#e7e5e4" strokeWidth="1" />
                <line x1="320" y1="60" x2="280" y2="160" stroke="#e7e5e4" strokeWidth="1" />

                {/* Article Nodes */}
                <g className={animating ? 'animate-pulse' : ''}>
                  <circle cx="200" cy="100" r="24" fill="#292524" />
                  <text x="200" y="105" textAnchor="middle" fill="white" fontSize="10">Today</text>
                </g>

                <circle cx="80" cy="60" r="16" fill="#78716c" />
                <circle cx="320" cy="60" r="18" fill="#78716c" />
                <circle cx="120" cy="160" r="14" fill="#a8a29e" />
                <circle cx="280" cy="160" r="20" fill="#57534e" />

                {/* Vote Indicators */}
                {animating && (
                  <>
                    <circle cx="200" cy="100" r="30" fill="none" stroke="#22c55e" strokeWidth="2" opacity="0.6">
                      <animate attributeName="r" from="24" to="40" dur="1s" repeatCount="indefinite" />
                      <animate attributeName="opacity" from="0.6" to="0" dur="1s" repeatCount="indefinite" />
                    </circle>
                  </>
                )}
              </svg>
              <div className="absolute bottom-3 left-3 text-xs text-stone-500">
                Node size = relevance score over time
              </div>
            </div>

            <div className="grid md:grid-cols-3 gap-6">
              <div className="space-y-2">
                <div className="text-3xl">📚</div>
                <h3 className="font-medium text-stone-800">Citations First</h3>
                <p className="text-sm text-stone-600">
                  New articles start with a score based on how other papers cite them.
                  Think of it like academic street cred.
                </p>
              </div>

              <div className="space-y-2">
                <div className="text-3xl">⏳</div>
                <h3 className="font-medium text-stone-800">Time-Tested Votes</h3>
                <p className="text-sm text-stone-600">
                  Your votes at 1 week, 1 month, and 1 year tell us what actually mattered.
                  Later votes count more.
                </p>
              </div>

              <div className="space-y-2">
                <div className="text-3xl">🎯</div>
                <h3 className="font-medium text-stone-800">Signal Finds Signal</h3>
                <p className="text-sm text-stone-600">
                  If your votes predict what others find valuable long-term,
                  your future votes carry more weight.
                </p>
              </div>
            </div>

            <div className="p-4 bg-amber-50 border border-amber-200 rounded-xl">
              <p className="text-amber-800 text-sm">
                <strong>The core insight:</strong> What feels important today often isn't.
                We optimize for what you'll still care about in a year.
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-6 text-sm">
            {/* Technical Mode */}
            <div className="space-y-4">
              <h3 className="font-medium text-stone-800">Cold Start: Citation-Based Ranking</h3>
              <div className="bg-stone-100 p-4 rounded-lg font-mono text-xs overflow-x-auto">
                <pre>{`InitialScore(article) =
  α × CitationCount
  + β × CitationVelocity
  + γ × SourceAuthority

where:
  α = 0.4 (raw citation weight)
  β = 0.35 (recent citation momentum)
  γ = 0.25 (PageRank of citing sources)`}</pre>
              </div>
              <p className="text-stone-600">
                Similar to PageRank, we weight citations by the authority of the citing source.
                A citation from Nature carries more weight than a citation from an unknown preprint.
              </p>
            </div>

            <div className="space-y-4">
              <h3 className="font-medium text-stone-800">Time-Decay Voting Function</h3>
              <div className="bg-stone-100 p-4 rounded-lg font-mono text-xs overflow-x-auto">
                <pre>{`RelevanceScore(article, t) =
  Σ(vote_i × TimeWeight(t_i) × UserCredibility_i)

TimeWeight(t) = {
  t = 1 week:  0.15
  t = 1 month: 0.35
  t = 1 year:  0.50
}

UserCredibility(user) =
  correlation(user_votes, community_long_term_votes)`}</pre>
              </div>
              <p className="text-stone-600">
                Votes cast after longer time periods receive exponentially more weight.
                Users whose past votes correlate with long-term community consensus gain influence.
              </p>
            </div>

            <div className="space-y-4">
              <h3 className="font-medium text-stone-800">Final Ranking</h3>
              <div className="bg-stone-100 p-4 rounded-lg font-mono text-xs overflow-x-auto">
                <pre>{`FinalScore(article) =
  (1 - λ) × InitialScore(article)
  + λ × RelevanceScore(article, now)

λ increases as more votes accumulate:
  λ = sigmoid(vote_count / threshold)`}</pre>
              </div>
              <p className="text-stone-600">
                As an article accumulates votes, the algorithm transitions from citation-based
                ranking to community-validated relevance scoring.
              </p>
            </div>

            <div className="p-4 bg-stone-800 text-stone-100 rounded-xl">
              <h4 className="font-medium mb-2">Key Properties</h4>
              <ul className="space-y-1 text-stone-300">
                <li>• Resistant to initial hype cycles</li>
                <li>• Self-correcting through delayed voting</li>
                <li>• Sybil-resistant via credibility weighting</li>
                <li>• Graceful degradation with sparse data</li>
              </ul>
            </div>
          </div>
        )}

        <button
          onClick={onClose}
          className="w-full mt-8 py-3 bg-stone-900 text-white rounded-full hover:bg-stone-800 transition-all"
        >
          Got it
        </button>
      </div>
    </div>
  );
}
