import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('nexusNative', {
  isElectron: true,
  platform: process.platform,
  // NOTE: internalToken is strictly omitted from the contextBridge boundary.
  // The renderer NEVER receives the application's reusable internal secret.
  openExternal: (url) => ipcRenderer.send('open-external', url),
  tabs: {
    create: (tabId, url) => ipcRenderer.send('tabs-create', tabId, url),
    destroy: (tabId) => ipcRenderer.send('tabs-destroy', tabId),
    setActive: (tabId) => ipcRenderer.send('tabs-set-active', tabId),
    navigate: (tabId, url) => ipcRenderer.send('tabs-navigate', tabId, url),
    goBack: (tabId) => ipcRenderer.send('tabs-go-back', tabId),
    goForward: (tabId) => ipcRenderer.send('tabs-go-forward', tabId),
    reload: (tabId) => ipcRenderer.send('tabs-reload', tabId),
    setBounds: (bounds) => ipcRenderer.send('tabs-set-bounds', bounds),
    createHiddenAgentTab: (tabId, url) => ipcRenderer.send('tabs-create-hidden-agent', tabId, url),
    destroyHiddenAgentTab: (tabId) => ipcRenderer.send('tabs-destroy-hidden-agent', tabId),
    openAgentTab: (tabId, url, label) => ipcRenderer.invoke('tabs-open-agent', { tabId, url, label }),
    setAgentStatus: (tabId, status) => ipcRenderer.send('tabs-set-agent-status', tabId, status),
    onUpdate: (callback) => {
      ipcRenderer.removeAllListeners('tab-update');
      ipcRenderer.on('tab-update', (event, tabId, data) => callback(tabId, data));
    }
  },
  cursor: {
    animate: (data) => ipcRenderer.send('cursor-animate', data),
    reset: () => ipcRenderer.send('cursor-reset'),
  },
  agent: {
    run: (payload) => ipcRenderer.invoke('agent-run', payload),
    stop: (taskId) => ipcRenderer.invoke('agent-stop', taskId),
    stream: (payload, onChunk, onError, onEnd) => {
      const streamId = 'stream_' + Math.random().toString(36).slice(2) + Date.now().toString(36);
      const chunkChannel = `agent-stream-chunk:${streamId}`;
      const errorChannel = `agent-stream-error:${streamId}`;
      const endChannel = `agent-stream-end:${streamId}`;

      const handleChunk = (_event, chunk) => onChunk?.(chunk);
      const handleError = (_event, err) => { cleanup(); onError?.(err); };
      const handleEnd = () => { cleanup(); onEnd?.(); };

      const cleanup = () => {
        ipcRenderer.removeListener(chunkChannel, handleChunk);
        ipcRenderer.removeListener(errorChannel, handleError);
        ipcRenderer.removeListener(endChannel, handleEnd);
      };

      ipcRenderer.on(chunkChannel, handleChunk);
      ipcRenderer.on(errorChannel, handleError);
      ipcRenderer.on(endChannel, handleEnd);

      ipcRenderer.invoke('agent-stream-start', { streamId, payload }).catch((err) => {
        cleanup();
        onError?.(err?.message || String(err));
      });

      return () => {
        cleanup();
        ipcRenderer.send('agent-stream-abort', streamId);
      };
    },
  }
});
