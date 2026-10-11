import React, { useEffect, useRef } from 'react';
import Launchpad from '../Apps/Launchpad';
import GoogleSearchApp from '../Apps/GoogleSearchApp';
import WikipediaApp from '../Apps/WikipediaApp';
import GitHubApp from '../Apps/GitHubApp';
import HackerNewsApp from '../Apps/HackerNewsApp';
import IframeViewer from '../Apps/IframeViewer';

function isInternalUrl(url) {
  return !url || url === 'comex://launchpad' || url === 'nexus://launchpad' || url === 'about:blank';
}

export default function BrowserCanvas({
  tabs = [],
  activeTabId,
  activeTab,
  onNavigate,
  onOpenChat,
  onSubmitLandingPrompt,
  onUpdateTabInfo,
  history = [],
  isChatOpen = false,
}) {
  const isElectron = typeof window !== 'undefined' && window.nexusNative?.isElectron;
  const boundsRef = useRef(null);

  // Element ref used only as a presence/mount check; we compute bounds
  // from window dimensions + known chrome heights, not from getBoundingClientRect,
  // because the React renderer's coords and the BrowserWindow content coords are
  // not the same coordinate space.
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!isElectron || !window.nexusNative?.tabs) return;

    const sendBounds = () => {
      const el = canvasRef.current;
      const bodyLayoutEl = document.querySelector('.browser-body-layout');
      const chatDrawerEl = document.querySelector('.chat-drawer');

      // Determine drawer width directly from DOM if open, otherwise fallback to standard calculated width
      let drawerWidth = 0;
      if (isChatOpen) {
        if (chatDrawerEl) {
          const drawerRect = chatDrawerEl.getBoundingClientRect();
          drawerWidth = Math.round(drawerRect.width);
        }
        if (!drawerWidth) {
          drawerWidth = Math.min(440, Math.round(window.innerWidth * 0.92));
        }
      }

      const hidden = isInternalUrl(activeTab?.url);

      // macOS hiddenInset titlebar adds 28px to the content view coordinate space
      const MACOS_TITLEBAR_OFFSET = 28;

      const bodyRect = bodyLayoutEl
        ? bodyLayoutEl.getBoundingClientRect()
        : (el ? el.getBoundingClientRect() : { top: 0, height: window.innerHeight, width: window.innerWidth });

      // Calculate view top in Electron coordinate space (minimum MACOS_TITLEBAR_OFFSET to prevent covering titlebar/navbar)
      const topOffset = bodyRect.top > 0 ? bodyRect.top : 80;
      const viewStartY = Math.max(MACOS_TITLEBAR_OFFSET, Math.round(topOffset + MACOS_TITLEBAR_OFFSET));

      // Clip the right boundary by the ChatDrawer width: width = containerRect.width - (isDrawerOpen ? drawerWidth : 0)
      const containerWidth = bodyRect.width || window.innerWidth;
      const width = Math.max(0, Math.round(containerWidth - (isChatOpen ? drawerWidth : 0)));

      // Ensure height never extends past window bottom
      const height = Math.max(0, Math.min(Math.round(bodyRect.height), window.innerHeight - Math.round(bodyRect.top)));

      const bounds = {
        x: 0,
        y: viewStartY,
        width,
        height,
        isHidden: hidden,
      };

      console.log('[CANVAS] Sending bounds to main process:', JSON.stringify(bounds));
      window.nexusNative.tabs.setBounds(bounds);
    };

    // Send after mount so the canvas element exists in DOM
    const raf = requestAnimationFrame(sendBounds);
    const observer = new ResizeObserver(sendBounds);
    if (canvasRef.current) observer.observe(canvasRef.current);
    window.addEventListener('resize', sendBounds);

    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      window.removeEventListener('resize', sendBounds);
    };
  }, [isElectron, activeTabId, activeTab?.url, isChatOpen]);

  const renderTabContent = (tab) => {
    const url = tab.url || 'comex://launchpad';

    if (isInternalUrl(url)) {
      return <Launchpad onNavigate={onNavigate} onOpenChat={onOpenChat} onSubmitPrompt={onSubmitLandingPrompt} history={history} />;
    }

    // Real websites are rendered by Electron WebContentsView, not React.
    if (isElectron) {
      return null;
    }

    if (url.includes('google.com')) {
      return <GoogleSearchApp currentUrl={url} onNavigate={onNavigate} />;
    }

    if (url.includes('wikipedia.org')) {
      return <WikipediaApp />;
    }

    if (url.includes('github.com')) {
      return <GitHubApp />;
    }

    if (url.includes('news.ycombinator.com')) {
      return <HackerNewsApp onNavigate={onNavigate} />;
    }

    return (
      <IframeViewer
        url={url}
        title={tab.title}
        tabId={tab.id}
        onUpdateTabInfo={onUpdateTabInfo}
      />
    );
  };

  const tabList = tabs.length > 0 ? tabs : (activeTab ? [activeTab] : []);

  return (
    <main
      ref={canvasRef}
      className={`browser-canvas-area ${isChatOpen ? 'with-drawer' : ''}`}
      style={{ position: 'relative', width: '100%', height: '100%' }}
    >
      {tabList.map((tab) => {
        const isActive = tab.id === activeTabId || (activeTab && tab.id === activeTab.id);
        return (
          <div
            key={tab.id}
            className={`tab-viewport ${isActive ? 'active' : 'hidden'}`}
            style={{
              display: isActive ? 'block' : 'none',
              width: '100%',
              height: '100%',
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
            }}
          >
            {renderTabContent(tab)}
          </div>
        );
      })}
    </main>
  );
}
