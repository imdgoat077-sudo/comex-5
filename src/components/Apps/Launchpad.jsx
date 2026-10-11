import React, { useState } from 'react';
import { ArrowUp, Search } from 'lucide-react';
import './Apps.css';

function normalizeWebUrl(value) {
  const trimmed = value.trim();
  if (!trimmed) return null;

  const candidate = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
  try {
    const url = new URL(candidate);
    return ['http:', 'https:'].includes(url.protocol) && url.hostname.includes('.') ? url.href : null;
  } catch {
    return null;
  }
}

export default function Launchpad({ onNavigate, onOpenChat, onSubmitPrompt, history = [] }) {
  const [query, setQuery] = useState('');

  const handleSubmit = (event) => {
    event.preventDefault();
    const value = query.trim();
    if (!value) return;

    const webUrl = normalizeWebUrl(value);
    if (webUrl) {
      onNavigate(webUrl);
    } else if (onSubmitPrompt) {
      onSubmitPrompt(value);
    } else {
      onOpenChat?.();
    }
    setQuery('');
  };

  return (
    <main className="comet-landing">
      <section className="comet-landing-content" aria-labelledby="landing-title">
        <h1 id="landing-title">What do you want to get done?</h1>
        <p className="comet-subtitle">Ask Comex to research or act on the web, or go straight to a site.</p>

        <form className="landing-prompt-form" onSubmit={handleSubmit}>
          <div className="landing-input-leading"><Search size={17} /></div>
          <input
            id="landing-prompt"
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Ask anything, or enter a website"
            aria-label="Ask Comex or enter a website"
            autoFocus
          />
          <button type="submit" className="landing-submit" aria-label="Go">
            <ArrowUp size={17} />
          </button>
        </form>

        <p className="landing-hints">Websites open in this tab. Questions and tasks go to the assistant.</p>

        {history.length > 0 && (
          <section className="landing-history" aria-labelledby="landing-history-title">
            <div className="landing-history-heading">
              <h2 id="landing-history-title">Recent</h2>
            </div>
            <div className="landing-history-list">
              {history.slice(0, 6).map((item) => (
                <div key={item.id} className={`landing-history-item landing-history-item--${item.status || 'done'}`}>
                  <div className="landing-history-item-header">
                    <span className="landing-history-item-title">{item.title}</span>
                    <span className="landing-history-item-time">{item.timestamp}</span>
                  </div>
                  {item.result && (
                    <pre className="landing-history-item-result">{item.result}</pre>
                  )}
                </div>
              ))}
            </div>
          </section>
        )}
      </section>
    </main>
  );
}
