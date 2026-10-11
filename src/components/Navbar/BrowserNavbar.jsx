import React, { useState, useRef, useEffect } from 'react';
import {
  ArrowLeft, ArrowRight, RotateCw, Home, Lock, Star,
  Plus, X, Sparkles, Search, Globe, HelpCircle, Volume2, Bot
} from 'lucide-react';
import './BrowserNavbar.css';

const INTERNAL_URLS = new Set(['comex://launchpad', 'nexus://launchpad', 'about:blank']);

const QUICK_SITES = [
  { url: 'https://www.google.com', title: 'Google', host: 'google.com' },
  { url: 'https://github.com', title: 'GitHub', host: 'github.com' },
  { url: 'https://en.wikipedia.org', title: 'Wikipedia', host: 'wikipedia.org' },
  { url: 'https://news.ycombinator.com', title: 'Hacker News', host: 'news.ycombinator.com' },
];

const AGENT_STATUS_LABEL = {
  working: 'Agent working in this tab',
  completed: 'Agent finished',
  failed: 'Agent failed',
  cancelled: 'Agent stopped',
  blocked: 'Agent blocked',
};

function toNavigableUrl(raw) {
  const value = raw.trim();
  if (/^(https?|comex|nexus):\/\//i.test(value)) return value;
  if (value.includes('.') && !value.includes(' ')) return `https://${value}`;
  return `https://www.google.com/search?q=${encodeURIComponent(value)}`;
}

export default function BrowserNavbar({
  tabs,
  activeTabId,
  onSelectTab,
  onAddTab,
  onCloseTab,
  onNavigate,
  onReload,
  onGoHome,
  onToggleChat,
  isChatOpen,
  unreadCount,
  onOpenShortcuts
}) {
  const activeTab = tabs.find(t => t.id === activeTabId) || tabs[0];
  const displayUrl = (url) => (INTERNAL_URLS.has(url) ? '' : url || '');
  const [urlInput, setUrlInput] = useState(displayUrl(activeTab?.url));
  const [isFocused, setIsFocused] = useState(false);
  const [isBookmarked, setIsBookmarked] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    if (activeTab) {
      setUrlInput(displayUrl(activeTab.url));
      setIsBookmarked(activeTab.isBookmarked || false);
    }
  }, [activeTab?.id, activeTab?.url, activeTab?.isBookmarked]);

  const canGoBack = activeTab ? (activeTab.canGoBack || activeTab.historyIndex > 0) : false;
  const canGoForward = activeTab ? (activeTab.canGoForward || activeTab.historyIndex < activeTab.history.length - 1) : false;
  const isSecure = activeTab?.url?.startsWith('https://');
  const isInternal = INTERNAL_URLS.has(activeTab?.url);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!urlInput.trim()) return;
    onNavigate(toNavigableUrl(urlInput));
    inputRef.current?.blur();
    setIsFocused(false);
  };

  const openQuickSite = (url) => {
    setUrlInput(url);
    onNavigate(url);
    setIsFocused(false);
  };

  const isElectron = typeof window !== 'undefined' && window.nexusNative?.isElectron;

  return (
    <header className="browser-header">
      <div className="browser-topbar">
        {isElectron ? (
          <div className="electron-traffic-spacer" />
        ) : (
          <div className="window-controls" aria-hidden="true">
            <span className="control-btn control-close"></span>
            <span className="control-btn control-minimize"></span>
            <span className="control-btn control-maximize"></span>
          </div>
        )}

        <div className="tabs-container" role="tablist" aria-label="Open tabs">
          {tabs.map((tab) => {
            const isActive = tab.id === activeTabId;
            const agentLabel = tab.agentStatus ? AGENT_STATUS_LABEL[tab.agentStatus] : null;
            return (
              <div
                key={tab.id}
                role="tab"
                aria-selected={isActive}
                tabIndex={0}
                className={`browser-tab ${isActive ? 'active' : ''} ${tab.agentStatus ? `agent-tab agent-${tab.agentStatus}` : ''}`}
                onClick={() => onSelectTab(tab.id)}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelectTab(tab.id); } }}
                title={tab.agentLabel ? `${agentLabel}: ${tab.agentLabel}\n${tab.url}` : tab.url}
              >
                <span className="tab-favicon">
                  {tab.isLoading ? (
                    <RotateCw className="tab-spinner animate-spin" size={12} />
                  ) : tab.favicon ? (
                    <img src={tab.favicon} alt="" className="tab-favicon-img" onError={(e) => { e.target.style.display = 'none'; }} />
                  ) : (
                    <Globe size={12} className="tab-globe" />
                  )}
                </span>
                {tab.agentStatus && (
                  <span className="tab-agent-badge" aria-label={agentLabel}>
                    <Bot size={11} />
                  </span>
                )}
                <span className="tab-title">{tab.title || 'New tab'}</span>
                {tab.isAudible && (
                  <span className="tab-audio-indicator" aria-label="Playing audio">
                    <Volume2 size={12} />
                  </span>
                )}
                {tabs.length > 1 && (
                  <button
                    className="tab-close-btn"
                    onClick={(e) => { e.stopPropagation(); onCloseTab(tab.id); }}
                    aria-label={`Close ${tab.title || 'tab'}`}
                    title="Close tab (⌘W)"
                  >
                    <X size={12} />
                  </button>
                )}
              </div>
            );
          })}

          <button className="add-tab-btn" onClick={onAddTab} aria-label="New tab" title="New tab (⌘T)">
            <Plus size={15} />
          </button>
        </div>

        <div className="topbar-actions">
          <button className="utility-btn" onClick={onOpenShortcuts} aria-label="Keyboard shortcuts" title="Keyboard shortcuts">
            <HelpCircle size={15} />
          </button>
        </div>
      </div>

      <div className="browser-toolbar">
        <div className="nav-group">
          <button className="nav-btn" disabled={!canGoBack} onClick={() => onNavigate('BACK')} aria-label="Back" title="Back">
            <ArrowLeft size={16} />
          </button>
          <button className="nav-btn" disabled={!canGoForward} onClick={() => onNavigate('FORWARD')} aria-label="Forward" title="Forward">
            <ArrowRight size={16} />
          </button>
          <button className="nav-btn" onClick={onReload} aria-label="Reload" title="Reload (⌘R)">
            <RotateCw className={activeTab?.isLoading ? 'animate-spin' : ''} size={15} />
          </button>
          <button className="nav-btn" onClick={onGoHome} aria-label="Home" title="Home">
            <Home size={15} />
          </button>
        </div>

        <div className={`omnibox-container ${isFocused ? 'focused' : ''}`}>
          <form onSubmit={handleSubmit} className="omnibox-form">
            <span className="security-badge" title={isSecure ? 'Secure connection' : ''}>
              {isInternal || !activeTab?.url ? <Search size={13} /> : isSecure ? <Lock size={12} /> : <Globe size={13} />}
            </span>
            <input
              ref={inputRef}
              type="text"
              className="omnibox-input"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              onFocus={(e) => { setIsFocused(true); e.target.select(); }}
              onBlur={() => setTimeout(() => setIsFocused(false), 150)}
              placeholder="Search or enter address"
              aria-label="Address and search bar"
              spellCheck={false}
            />
            {!isInternal && activeTab?.url && (
              <button
                type="button"
                className={`bookmark-btn ${isBookmarked ? 'active' : ''}`}
                onClick={() => setIsBookmarked(!isBookmarked)}
                aria-label={isBookmarked ? 'Remove bookmark' : 'Bookmark this page'}
                aria-pressed={isBookmarked}
              >
                <Star size={14} fill={isBookmarked ? 'currentColor' : 'none'} />
              </button>
            )}
          </form>

          {isFocused && (
            <div className="search-dropdown animate-fade-in" role="listbox">
              {QUICK_SITES.map((site) => (
                <button
                  key={site.url}
                  type="button"
                  className="dropdown-item"
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => openQuickSite(site.url)}
                >
                  <Globe size={14} className="item-icon" />
                  <span className="item-title">{site.title}</span>
                  <span className="item-url">{site.host}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        <button
          className={`chat-toggle-btn ${isChatOpen ? 'active' : ''}`}
          onClick={onToggleChat}
          aria-pressed={isChatOpen}
          title="Assistant (⌘J)"
        >
          <Sparkles size={14} />
          <span className="chat-btn-label">Assistant</span>
          {unreadCount > 0 && <span className="unread-dot" aria-label={`${unreadCount} unread`}>{unreadCount}</span>}
        </button>
      </div>
    </header>
  );
}
