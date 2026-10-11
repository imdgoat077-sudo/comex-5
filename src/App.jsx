import React, { useState, useEffect, useRef } from 'react';
import BrowserNavbar from './components/Navbar/BrowserNavbar';
import BookmarksBar from './components/Bookmarks/BookmarksBar';
import BrowserCanvas from './components/Canvas/BrowserCanvas';
import ChatDrawer from './components/Chat/ChatDrawer';
import ShortcutsModal from './components/Modals/ShortcutsModal';
import { classifyIntent } from './agents/IntentClassifier';
import './App.css';

export default function App() {
  // Master Tab State
  const [tabs, setTabs] = useState([
    {
      id: 'tab-1',
      title: 'Comex Launchpad',
      url: 'comex://launchpad',
      history: ['comex://launchpad'],
      historyIndex: 0,
      isLoading: false,
      isBookmarked: false,
      canGoBack: false,
      canGoForward: false
    }
  ]);

  const [activeTabId, setActiveTabId] = useState('tab-1');
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);
  const [isShortcutsOpen, setIsShortcutsOpen] = useState(false);
  const [landingHistory, setLandingHistory] = useState([]);

  const activeTab = tabs.find(t => t.id === activeTabId) || tabs[0];
  // Agent tabs whose worker is still running; the first one opened is focused so the user can watch it.
  const workingAgentTabsRef = useRef(new Set());

  // Initialize tabs in Electron
  useEffect(() => {
    if (window.nexusNative?.tabs) {
      window.nexusNative.tabs.create('tab-1', 'comex://launchpad');
      window.nexusNative.tabs.setActive('tab-1');

      window.nexusNative.tabs.onUpdate((tabId, data) => {
        // A multi-agent worker opened its own visible tab (Electron already created the view)
        if (tabId === '__agent_tab__' && data.tabId) {
          const shouldFocus = workingAgentTabsRef.current.size === 0;
          workingAgentTabsRef.current.add(data.tabId);
          setTabs(prevTabs => prevTabs.some(t => t.id === data.tabId) ? prevTabs : [...prevTabs, {
            id: data.tabId,
            title: data.label || data.url,
            url: data.url,
            history: [data.url],
            historyIndex: 0,
            isLoading: true,
            isBookmarked: false,
            canGoBack: false,
            canGoForward: false,
            agentStatus: 'working',
            agentLabel: data.label,
          }]);
          if (shouldFocus) setActiveTabId(data.tabId);
          return;
        }

        if (data.agentStatus && data.agentStatus !== 'working') {
          workingAgentTabsRef.current.delete(tabId);
        }

        // Main process signals a new-window request → create a new Comex tab
        if (tabId === '__new_tab__' && data.url) {
          const newTabId = data.newTabId || `tab-${Date.now()}`;
          
          window.nexusNative.tabs.create(newTabId, data.url);
          // Let the useEffect handle sync to main process to avoid race conditions
          // window.nexusNative.tabs.setActive(newTabId);
          
          setTabs(prevTabs => {
            const newTab = {
              id: newTabId,
              title: data.url,
              url: data.url,
              history: [data.url],
              historyIndex: 0,
              isLoading: true,
              isBookmarked: false,
              canGoBack: false,
              canGoForward: false,
            };
            return [...prevTabs, newTab];
          });
          
          setActiveTabId(newTabId);
          return;
        }

        setTabs(prevTabs => prevTabs.map(tab => {
          if (tab.id === tabId) {
            return {
              ...tab,
              title: data.title !== undefined ? data.title : tab.title,
              favicon: data.favicon !== undefined ? data.favicon : tab.favicon,
              url: data.url !== undefined ? data.url : tab.url,
              isLoading: data.isLoading !== undefined ? data.isLoading : tab.isLoading,
              canGoBack: data.canGoBack !== undefined ? data.canGoBack : tab.canGoBack,
              canGoForward: data.canGoForward !== undefined ? data.canGoForward : tab.canGoForward,
              agentStatus: data.agentStatus !== undefined ? data.agentStatus : tab.agentStatus,
            };
          }
          return tab;
        }));
      });
    }
  }, []);

  // Sync active tab to Electron
  useEffect(() => {
    if (window.nexusNative?.tabs) {
      window.nexusNative.tabs.setActive(activeTabId);
    }
  }, [activeTabId]);

  // Dynamic Tab Title, Favicon, URL, Audio status, and Nav state update from Webview / Web page (Fallback for non-electron)
  const handleUpdateTabInfo = (tabId, title, favicon = null, url = null, isAudible = null, navState = null) => {
    if (!title && !favicon && !url && isAudible === null && !navState) return;
    setTabs(prevTabs => prevTabs.map(tab => {
      if (tab.id === tabId) {
        return {
          ...tab,
          title: title || tab.title,
          favicon: favicon || tab.favicon,
          url: url || tab.url,
          isAudible: isAudible !== null ? isAudible : tab.isAudible,
          canGoBack: navState?.canGoBack !== undefined ? navState.canGoBack : tab.canGoBack,
          canGoForward: navState?.canGoForward !== undefined ? navState.canGoForward : tab.canGoForward,
        };
      }
      return tab;
    }));
  };

  // Helper to extract clean page title from URL
  const getTitleFromUrl = (targetUrl) => {
    if (targetUrl === 'comex://launchpad' || targetUrl === 'nexus://launchpad' || !targetUrl) return 'Comex Launchpad';
    try {
      const urlObj = new URL(targetUrl);
      if (urlObj.hostname.includes('google.com')) return `Google Search`;
      if (urlObj.hostname.includes('wikipedia.org')) return `Quantum Teleportation - Wikipedia`;
      if (urlObj.hostname.includes('github.com')) return `comex-browser/core - GitHub`;
      if (urlObj.hostname.includes('ycombinator.com')) return `Hacker News`;
      return urlObj.hostname;
    } catch {
      return targetUrl;
    }
  };

  // Add new tab
  const handleAddTab = (initialUrl = 'comex://launchpad') => {
    const newTabId = `tab-${Date.now()}`;
    if (window.nexusNative?.tabs) {
      window.nexusNative.tabs.create(newTabId, initialUrl);
    }
    const newTab = {
      id: newTabId,
      title: getTitleFromUrl(initialUrl),
      url: initialUrl,
      history: [initialUrl],
      historyIndex: 0,
      isLoading: false,
      isBookmarked: false,
      canGoBack: false,
      canGoForward: false
    };
    setTabs(prev => [...prev, newTab]);
    setActiveTabId(newTabId);
  };

  // Select active tab
  const handleSelectTab = (id) => {
    setActiveTabId(id);
  };

  // Close tab
  const handleCloseTab = (idToClose) => {
    workingAgentTabsRef.current.delete(idToClose);
    if (tabs.length === 1) {
      // Don't close last tab; reset it to launchpad instead
      const remainingId = tabs[0].id;
      if (window.nexusNative?.tabs) {
        window.nexusNative.tabs.navigate(remainingId, 'comex://launchpad');
      }
      setTabs([{
        id: remainingId,
        title: 'Comex Launchpad',
        url: 'comex://launchpad',
        history: ['comex://launchpad'],
        historyIndex: 0,
        isLoading: false,
        isBookmarked: false,
        canGoBack: false,
        canGoForward: false
      }]);
      setActiveTabId(remainingId);
      return;
    }

    if (window.nexusNative?.tabs) {
      window.nexusNative.tabs.destroy(idToClose);
    }

    const closedIndex = tabs.findIndex(t => t.id === idToClose);
    const updatedTabs = tabs.filter(t => t.id !== idToClose);
    setTabs(updatedTabs);

    if (activeTabId === idToClose) {
      const nextActiveIndex = Math.max(0, closedIndex - 1);
      setActiveTabId(updatedTabs[nextActiveIndex].id);
    }
  };

  // Navigate within active tab
  const handleNavigate = (target) => {
    if (target === 'BACK' || target === 'FORWARD') {
      if (window.nexusNative?.tabs) {
        target === 'BACK' ? window.nexusNative.tabs.goBack(activeTabId) : window.nexusNative.tabs.goForward(activeTabId);
      } else {
        window.dispatchEvent(new CustomEvent('comex-webview-nav', {
          detail: { action: target, tabId: activeTabId }
        }));
      }
      return;
    }

    if (window.nexusNative?.tabs) {
      window.nexusNative.tabs.navigate(activeTabId, target);
    }

    setTabs(prevTabs => prevTabs.map(tab => {
      if (tab.id !== activeTabId) return tab;

      const truncatedHistory = tab.history.slice(0, tab.historyIndex + 1);
      const newHistory = [...truncatedHistory, target];
      const newIndex = newHistory.length - 1;

      return {
        ...tab,
        url: target,
        title: getTitleFromUrl(target),
        history: newHistory,
        historyIndex: newIndex,
        isLoading: true
      };
    }));

    if (!window.nexusNative?.tabs) {
      // Fallback reset loading state
      setTimeout(() => {
        setTabs(prevTabs => prevTabs.map(tab => {
          if (tab.id === activeTabId) {
            return { ...tab, isLoading: false };
          }
          return tab;
        }));
      }, 450);
    }
  };

  // Reload current page
  const handleReload = () => {
    if (window.nexusNative?.tabs) {
      window.nexusNative.tabs.reload(activeTabId);
    } else {
      window.dispatchEvent(new CustomEvent('comex-webview-nav', {
        detail: { action: 'RELOAD', tabId: activeTabId }
      }));
    }

    setTabs(prev => prev.map(tab => {
      if (tab.id === activeTabId) {
        return { ...tab, isLoading: true };
      }
      return tab;
    }));

    if (!window.nexusNative?.tabs) {
      setTimeout(() => {
        setTabs(prev => prev.map(tab => {
          if (tab.id === activeTabId) {
            return { ...tab, isLoading: false };
          }
          return tab;
        }));
      }, 500);
    }
  };

  // Go Home
  const handleGoHome = () => {
    handleNavigate('comex://launchpad');
  };

  // Toggle AI Chat Drawer
  const handleToggleChat = () => {
    setIsChatOpen(!isChatOpen);
    if (!isChatOpen) {
      setUnreadCount(0);
    }
  };

  // Keyboard Shortcuts Listener
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.metaKey || e.ctrlKey) {
        if (e.key === 't' || e.key === 'T') {
          e.preventDefault();
          handleAddTab();
        } else if (e.key === 'w' || e.key === 'W') {
          e.preventDefault();
          handleCloseTab(activeTabId);
        } else if (e.key === 'r' || e.key === 'R') {
          e.preventDefault();
          handleReload();
        } else if (e.key === 'j' || e.key === 'J') {
          e.preventDefault();
          handleToggleChat();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [activeTabId, isChatOpen]);


  // Landing AI prompt: route by semantic intent using the LLM classifier.
  // - research -> keep the user on the landing page / sidebar chat flow.
  // - visible_browser_task -> create a visible tab and switch workspaces.
  // Ambiguous requests default safely to research.
  const handleLandingPrompt = async (prompt) => {
    const lowerPrompt = prompt.toLowerCase();
    const isHiddenExamplePoc = lowerPrompt.includes('hidden background tab')
      && lowerPrompt.includes('example.com')
      && lowerPrompt.includes('without showing the browser tab');

    if (isHiddenExamplePoc && window.nexusNative?.tabs?.createHiddenAgentTab) {
      const hiddenTabId = `hidden-agent-${Date.now()}`;
      const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

      setLandingHistory((prev) => [{
        id: `landing_${Date.now()}`,
        title: prompt,
        status: 'working',
        result: 'Working in hidden background tab…',
        timestamp,
      }, ...prev].slice(0, 6));

      try {
        window.nexusNative.tabs.createHiddenAgentTab(hiddenTabId, 'https://example.com');

        let data;
        if (window.nexusNative?.agent?.run) {
          data = await window.nexusNative.agent.run({
            task: 'Open https://example.com, read the page title and the main heading, and return both exactly as: Page Title: <title>\\nMain Heading: <heading>. Do not interact with any visible tab.',
            tabId: hiddenTabId,
            url: 'https://example.com',
            taskId: `task_${Date.now()}`,
          });
        } else {
          const response = await fetch('http://127.0.0.1:8000/agent/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              task: 'Open https://example.com, read the page title and the main heading, and return both exactly as: Page Title: <title>\\nMain Heading: <heading>. Do not interact with any visible tab.',
              tabId: hiddenTabId,
              url: 'https://example.com',
              taskId: `task_${Date.now()}`,
            })
          });
          data = await response.json();
        }
        const resultText = data.success ? data.result : data.error || 'Hidden-tab task failed.';

        setLandingHistory((prev) => prev.map((item, index) => index === 0 ? {
          ...item,
          status: data.success ? 'completed' : 'failed',
          result: resultText,
        } : item));
      } catch (error) {
        setLandingHistory((prev) => prev.map((item, index) => index === 0 ? {
          ...item,
          status: 'failed',
          result: error.message || 'Hidden-tab task failed.',
        } : item));
      }

      return;
    }

    try {
      const result = await classifyIntent(prompt);
      const { intent, confidence, target } = result || {};

      if (intent === 'visible_browser_task' && confidence >= 0.6) {
        const newTabId = `tab-${Date.now()}`;
        const nextUrl = target || 'comex://launchpad';

        // Pre-emptively adjust bounds for chat drawer before tab activation
        if (window.nexusNative?.tabs?.setBounds) {
          const chatWidth = Math.min(440, Math.round(window.innerWidth * 0.92));
          window.nexusNative.tabs.setBounds({
            x: 0,
            y: 138,
            width: Math.max(0, window.innerWidth - chatWidth),
            height: Math.max(0, window.innerHeight - 138),
            isHidden: false
          });
        }

        if (window.nexusNative?.tabs) {
          window.nexusNative.tabs.create(newTabId, nextUrl);
          window.nexusNative.tabs.setActive(newTabId);
        }

        const newTab = {
          id: newTabId,
          title: getTitleFromUrl(nextUrl),
          url: nextUrl,
          history: [nextUrl],
          historyIndex: 0,
          isLoading: true,
          isBookmarked: false,
          canGoBack: false,
          canGoForward: false
        };

        setTabs((prev) => [...prev, newTab]);
        setActiveTabId(newTabId);
        setIsChatOpen(true);

        requestAnimationFrame(() => {
          window.dispatchEvent(new CustomEvent('comex-landing-prompt', {
            detail: { prompt, visibleTabId: newTabId, intent }
          }));
        });
        return;
      }

      setIsChatOpen(true);
      window.dispatchEvent(new CustomEvent('comex-landing-prompt', { detail: { prompt, intent } }));
    } catch (err) {
      console.error('Intent classification failed:', err);
      setIsChatOpen(true);
      window.dispatchEvent(new CustomEvent('comex-landing-prompt', { detail: { prompt } }));
    }
  };

  return (
    <div className="nexus-app-container">
      {/* Master Header Bar */}
      <BrowserNavbar
        tabs={tabs}
        activeTabId={activeTabId}
        onSelectTab={handleSelectTab}
        onAddTab={() => handleAddTab('comex://launchpad')}
        onCloseTab={handleCloseTab}
        onNavigate={handleNavigate}
        onReload={handleReload}
        onGoHome={handleGoHome}
        onToggleChat={handleToggleChat}
        isChatOpen={isChatOpen}
        unreadCount={unreadCount}
        onOpenShortcuts={() => setIsShortcutsOpen(true)}
      />

      {/* Bookmarks Bar */}
      <BookmarksBar onNavigate={handleNavigate} />

      {/* Main Workspace Frame */}
      <div className="browser-body-layout">
        <BrowserCanvas
          tabs={tabs}
          activeTabId={activeTabId}
          activeTab={activeTab}
          onNavigate={handleNavigate}
          onOpenChat={() => setIsChatOpen(true)}
          onSubmitLandingPrompt={handleLandingPrompt}
          onUpdateTabInfo={handleUpdateTabInfo}
          history={landingHistory}
          isChatOpen={isChatOpen}
        />

        {/* AI Co-Pilot Chat Drawer */}
        <ChatDrawer
          isOpen={isChatOpen}
          onClose={() => setIsChatOpen(false)}
          activeTab={activeTab}
          onSelectTab={handleSelectTab}
        />
      </div>

      {/* Shortcuts Modal */}
      <ShortcutsModal
        isOpen={isShortcutsOpen}
        onClose={() => setIsShortcutsOpen(false)}
      />
    </div>
  );
}
