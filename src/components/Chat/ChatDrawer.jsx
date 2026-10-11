import React, { useState, useRef, useEffect } from 'react';
import {
  Sparkles, X, Send, Bot, User, Trash2, Copy, Check,
  ChevronRight, ChevronDown,
  Globe, Play, Layers, Activity, Crown, Compass, Search, ShieldCheck, Terminal,
  Brain, Cpu, BookOpen, Lightbulb, RefreshCw, Plus, Eye, Zap
} from 'lucide-react';
import MarkdownRenderer from './MarkdownRenderer';
import { runMultiAgentSwarm } from '../../agents/OrchestratorAgent';
import { AGENT_ROLES } from '../../agents/types';
import './ChatDrawer.css';

function getAgentRoleIcon(roleId, size = 12) {
  switch (roleId) {
    case 'orchestrator':
      return <Crown size={size} />;
    case 'navigator':
      return <Compass size={size} />;
    case 'researcher':
      return <Search size={size} />;
    case 'inspector':
      return <ShieldCheck size={size} />;
    default:
      return <Bot size={size} />;
  }
}

function formatDuration(durationMs) {
  const totalSeconds = Math.max(0, durationMs) / 1000;
  if (totalSeconds < 60) return `${totalSeconds.toFixed(1)}s`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.round(totalSeconds % 60);
  return `${minutes}m ${seconds}s`;
}

export default function ChatDrawer({
  isOpen,
  onClose,
  activeTab,
  onNavigate,
  onAddTab
}) {
  const [isMultiAgentMode, setIsMultiAgentMode] = useState(false); // default: browser-use Single Agent path
  const [conversationId, setConversationId] = useState(() => 'conv_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8));
  const [messages, setMessages] = useState([
    {
      id: 'm1',
      sender: 'assistant',
      text: "Welcome to **Comex Multi-Agent Swarm**.\n\nOur specialized sub-agent pipeline (Orchestrator, Navigator, Researcher, Inspector) executes concurrent background searches, inspects webview DOM semantics, and compiles structured research reports.",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    },
    {
      id: 'm2',
      sender: 'assistant',
      text: "### Autonomous Capabilities\n- Search & compare repositories on **GitHub**\n- Execute web queries on **Google**\n- Search & stream videos on **YouTube**\n- Summarize DOM content of active web pages",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);

  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [agentLogs, setAgentLogs] = useState([]);
  const [currentProgress, setCurrentProgress] = useState(null);
  const [expandedActivity, setExpandedActivity] = useState({});
  const [activeSingleAgentMessageId, setActiveSingleAgentMessageId] = useState(null);
  const [copiedId, setCopiedId] = useState(null);
  const [activeDrawerTab, setActiveDrawerTab] = useState('chat'); // 'chat' | 'brain'
  const [brainSection, setBrainSection] = useState('user_memory'); // 'user_memory' | 'skills' | 'reflections' | 'adapters'
  const [userMemories, setUserMemories] = useState([]);
  const [skills, setSkills] = useState([]);
  const [reflections, setReflections] = useState([]);
  const [adapters, setAdapters] = useState([]);
  const [memoryStats, setMemoryStats] = useState(null);
  const [newKey, setNewKey] = useState('');
  const [newValue, setNewValue] = useState('');
  const [newCategory, setNewCategory] = useState('preference');
  const [isRefreshingBrain, setIsRefreshingBrain] = useState(false);
  const activeTaskRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Load Brain memory data from Python Agent service
  const fetchBrainData = async () => {
    setIsRefreshingBrain(true);
    try {
      const [ovRes, memRes, sklRes, refRes, adpRes] = await Promise.all([
        fetch('http://127.0.0.1:8000/agent/memory/overview').then(r => r.ok ? r.json() : null),
        fetch('http://127.0.0.1:8000/agent/user/memory').then(r => r.ok ? r.json() : []),
        fetch('http://127.0.0.1:8000/agent/skills?limit=30').then(r => r.ok ? r.json() : []),
        fetch('http://127.0.0.1:8000/agent/reflections?limit=20').then(r => r.ok ? r.json() : []),
        fetch('http://127.0.0.1:8000/agent/adapters').then(r => r.ok ? r.json() : [])
      ]);
      if (ovRes) setMemoryStats(ovRes.stats);
      if (Array.isArray(memRes)) setUserMemories(memRes);
      if (Array.isArray(sklRes)) setSkills(sklRes);
      if (Array.isArray(refRes)) setReflections(refRes);
      if (Array.isArray(adpRes)) setAdapters(adpRes);
    } catch (e) {
      console.warn('Could not fetch brain data:', e);
    } finally {
      setIsRefreshingBrain(false);
    }
  };

  useEffect(() => {
    if (activeDrawerTab === 'brain') {
      fetchBrainData();
    }
  }, [activeDrawerTab]);

  const handleDeleteMemory = async (key) => {
    try {
      await fetch(`http://127.0.0.1:8000/agent/user/memory/${encodeURIComponent(key)}`, { method: 'DELETE' });
      setUserMemories(prev => prev.filter(m => m.key !== key));
    } catch (e) {
      console.warn('Failed to delete memory:', e);
    }
  };

  const handleDeleteSkill = async (id) => {
    try {
      await fetch(`http://127.0.0.1:8000/agent/skills/${encodeURIComponent(id)}`, { method: 'DELETE' });
      setSkills(prev => prev.filter(s => s.id !== id));
    } catch (e) {
      console.warn('Failed to delete skill:', e);
    }
  };

  const handleDeleteAdapter = async (id) => {
    try {
      await fetch(`http://127.0.0.1:8000/agent/adapters/${encodeURIComponent(id)}`, { method: 'DELETE' });
      setAdapters(prev => prev.filter(a => a.id !== id));
    } catch (e) {
      console.warn('Failed to delete adapter:', e);
    }
  };

  const handleConsolidateSkills = async () => {
    try {
      setIsRefreshingBrain(true);
      await fetch('http://127.0.0.1:8000/agent/skills/consolidate', { method: 'POST' });
      await fetchBrainData();
    } catch (e) {
      console.warn('Failed to consolidate skills:', e);
    } finally {
      setIsRefreshingBrain(false);
    }
  };

  const handleAddMemory = async (e) => {
    e.preventDefault();
    if (!newKey.trim() || !newValue.trim()) return;
    try {
      await fetch('http://127.0.0.1:8000/agent/user/memory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: newKey.trim(), value: newValue.trim(), category: newCategory })
      });
      setNewKey('');
      setNewValue('');
      fetchBrainData();
    } catch (e) {
      console.warn('Failed to add user memory:', e);
    }
  };

  // Auto scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping, agentLogs, activeSingleAgentMessageId]);

  // Track active Single Agent message and append real tab activity details
  useEffect(() => {
    if (!activeSingleAgentMessageId) return;
    if (!activeTab) return;

    setMessages((prev) => prev.map((msg) => {
      if (msg.id !== activeSingleAgentMessageId) return msg;

      const newActivity = [...(msg.activity || [])];
      let stepMessage = '';

      if (activeTab.isLoading) {
        stepMessage = `Loading ${activeTab.url || 'page'}...`;
      } else if (activeTab.title && activeTab.title !== 'Comex Launchpad' && activeTab.title !== 'about:blank') {
        stepMessage = `Opened page: ${activeTab.title}`;
      } else if (activeTab.url && activeTab.url !== 'comex://launchpad' && activeTab.url !== 'about:blank') {
        try {
          const urlObj = new URL(activeTab.url);
          stepMessage = `Navigating to ${urlObj.hostname}`;
        } catch {
          stepMessage = `Navigating to ${activeTab.url}`;
        }
      }

      if (stepMessage && !newActivity.some(log => log.message === stepMessage)) {
        newActivity.push({
          id: `log_${Date.now()}_${Math.random()}`,
          message: stepMessage,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
        });
      }

      const latestMsg = newActivity[newActivity.length - 1]?.message || 'Working...';

      return {
        ...msg,
        activity: newActivity,
        statusLabel: latestMsg
      };
    }));
  }, [activeTab?.url, activeTab?.title, activeTab?.isLoading, activeSingleAgentMessageId]);

  // Receive prompts submitted from the Comet-style landing page.
  useEffect(() => {
    const handleLandingPrompt = (event) => {
      const prompt = event.detail?.prompt?.trim();
      const intent = event.detail?.intent;
      if (prompt) handleSend(prompt, intent);
    };

    window.addEventListener('comex-landing-prompt', handleLandingPrompt);
    return () => window.removeEventListener('comex-landing-prompt', handleLandingPrompt);
  }, [activeTab?.id, activeTab?.url, isMultiAgentMode]);

  // Single Agent Direct Automation with Live Streaming Event Stream
  const executeSingleAgentAction = async (text, taskId, controller, intent, messageId, conversationOptions = {}) => {
    try {
      let buffer = '';
      let outcome = { status: 'completed', text: '', durationMs: null, task_outcome: null };

      const processChunk = (chunkText) => {
        buffer += chunkText;
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep partial line in buffer

        let currentEventType = 'message';

        for (let i = 0; i < lines.length; i++) {
          const line = lines[i].trim();
          if (!line) continue;

          if (line.startsWith('event:')) {
            currentEventType = line.replace(/^event:\s*/, '');
          } else if (line.startsWith('data:')) {
            const dataStr = line.replace(/^data:\s*/, '');
            try {
              const data = JSON.parse(dataStr);

              // Synchronously update outcome outside React's batched state updater
              if (currentEventType === 'done' || currentEventType === 'completed' || data.type === 'result' || data.type === 'chat' || data.done) {
                const isFailed = data.stopped ? false : (data.success === false || Boolean(data.error));
                const isConv = Boolean(!isFailed && (data.is_conversational || data.type === 'chat'));
                const errorText = data.error || data.metadata?.error;
                const rawContent = data.content || data.text || data.result;
                const fallbackSuccessText = isConv ? 'Hello! How can I assist you today?' : 'Task completed.';
                const finalContent = isFailed
                  ? (errorText || rawContent || outcome.content || outcome.text || 'The agent could not complete the task.')
                  : (rawContent || outcome.content || outcome.text || fallbackSuccessText);

                outcome = {
                  status: data.stopped ? 'stopped' : isFailed ? 'failed' : 'completed',
                  text: finalContent,
                  content: finalContent,
                  durationMs: data.durationMs || outcome.durationMs,
                  is_conversational: isConv,
                  fast_path: Boolean(!isFailed && (data.fast_path || outcome.fast_path)),
                  adapter_name: isFailed ? null : (data.adapter_name || outcome.adapter_name),
                  task_outcome: isFailed ? null : (data.metadata?.task_outcome || data.task_outcome || outcome.task_outcome || null)
                };
              }

              setMessages((prev) => prev.map((msg) => {
                if (msg.id !== messageId) return msg;

                const curActivity = [...(msg.activity || [])];

                if (currentEventType === 'perception') {
                  return {
                    ...msg,
                    perception: {
                      url: data.url,
                      title: activeTab?.title || data.url,
                      detail: data.detail || `Observing ${data.url}`
                    }
                  };
                }

                if (currentEventType === 'thinking') {
                  return {
                    ...msg,
                    statusLabel: data.detail || 'Thinking...',
                    thinking: data.detail
                  };
                }

                if (currentEventType === 'planning') {
                  return {
                    ...msg,
                    statusLabel: data.detail || 'Planning next action...'
                  };
                }

                if (currentEventType === 'step') {
                  const existingIdx = curActivity.findIndex(a => a.step === data.step);
                  if (existingIdx >= 0) {
                    curActivity[existingIdx] = { ...curActivity[existingIdx], ...data };
                  } else {
                    curActivity.push({
                      id: `step_${data.step}_${Date.now()}`,
                      step: data.step,
                      tool: data.tool,
                      label: data.label,
                      status: 'running',
                      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
                    });
                  }
                  return {
                    ...msg,
                    activity: curActivity,
                    statusLabel: data.label
                  };
                }

                if (currentEventType === 'step_done') {
                  const existingIdx = curActivity.findIndex(a => a.step === data.step);
                  if (existingIdx >= 0) {
                    curActivity[existingIdx] = {
                      ...curActivity[existingIdx],
                      status: data.success ? 'done' : 'failed'
                    };
                  }
                  return {
                    ...msg,
                    activity: curActivity
                  };
                }

                if (currentEventType === 'done' || currentEventType === 'completed' || data.type === 'result' || data.type === 'chat' || data.done) {
                  const finalContent = outcome.content || outcome.text;
                  const isConv = outcome.is_conversational;
                  const statusLbl = outcome.status === 'failed'
                    ? 'Task failed'
                    : (outcome.status === 'stopped'
                      ? 'Task stopped'
                      : (outcome.fast_path ? `⚡ Fast Path (${outcome.adapter_name || 'Adapter'})` : (isConv ? '' : 'Task completed')));
                  return {
                    ...msg,
                    text: finalContent,
                    content: finalContent,
                    status: outcome.status,
                    statusLabel: statusLbl,
                    is_conversational: isConv,
                    fast_path: outcome.fast_path,
                    adapter_name: outcome.adapter_name,
                    task_outcome: outcome.task_outcome,
                    isStreaming: false
                  };
                }

                return msg;
              }));
            } catch (je) {
              console.warn('Error parsing SSE event:', je);
            }
          }
        }
      };

      if (window.nexusNative?.agent?.stream) {
        await new Promise((resolve, reject) => {
          const unsubscribe = window.nexusNative.agent.stream(
            {
              task: text,
              taskId,
              tabId: 'active_tab',
              url: activeTab?.url || '',
              conversation_id: conversationOptions?.conversation_id,
              messages: conversationOptions?.messages,
              task_context: conversationOptions?.task_context,
              ...(intent ? { intent } : {})
            },
            (chunk) => processChunk(chunk),
            (err) => reject(new Error(err)),
            () => resolve()
          );
          controller.signal.addEventListener('abort', () => {
            unsubscribe();
            resolve();
          });
        });
        return outcome;
      }

      // Non-electron web fallback
      const response = await fetch('http://127.0.0.1:8000/agent/run/stream', {
        method: 'POST',
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          task: text,
          taskId,
          tabId: 'active_tab',
          url: activeTab?.url || '',
          conversation_id: conversationOptions?.conversation_id,
          messages: conversationOptions?.messages,
          task_context: conversationOptions?.task_context,
          ...(intent ? { intent } : {})
        })
      });

      if (!response.ok || !response.body) {
        throw new Error(`Agent service responded with status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        processChunk(decoder.decode(value, { stream: true }));
      }

      return outcome;
    } catch (err) {
      if (err.name === 'AbortError') return { status: 'stopped', text: 'The task was stopped by the user.' };
      return { status: 'failed', text: '### Connection Error\n\nCould not connect to the Python browser-use service (http://127.0.0.1:8000). Ensure `agent_service.py` is running.' };
    }
  };

  const stopSingleAgent = async () => {
    const active = activeTaskRef.current;
    if (!active || active.stopping) return;

    active.stopping = true;
    setMessages((prev) => prev.map((message) => message.id === active.messageId ? {
      ...message,
      statusLabel: 'Stopping...',
    } : message));

    active.controller.abort();
    try {
      if (window.nexusNative?.agent?.stop) {
        await window.nexusNative.agent.stop(active.taskId);
      } else {
        await fetch('http://127.0.0.1:8000/agent/stop', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ taskId: active.taskId }),
        });
      }
    } catch (err) {
      console.warn('Agent cancellation request failed:', err);
    }
  };

  const handleSend = async (textToSend = input, intent = undefined) => {
    const text = textToSend.trim();
    if (!text) return;

    const userMsg = {
      id: `u_${Date.now()}`,
      sender: 'user',
      text: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages((prev) => [...prev, userMsg]);
    if (textToSend === input) setInput('');
    setIsTyping(true);
    setAgentLogs([]);
    setCurrentProgress(null);

    if (isMultiAgentMode) {
      try {
        const finalReport = await runMultiAgentSwarm(text, activeTab, {
          onLog: (logItem) => {
            setAgentLogs((prev) => [...prev, logItem]);
          },
          onProgress: (progress) => {
            setCurrentProgress(progress);
          },
          onTabNavigate: onNavigate,
          onAddTab
        });

        const aiMsg = {
          id: `a_${Date.now()}`,
          sender: 'assistant',
          text: finalReport,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        };

        setMessages((prev) => [...prev, aiMsg]);
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          {
            id: `a_err_${Date.now()}`,
            sender: 'assistant',
            text: `### Multi-Agent Pipeline Warning\n\n${err.message}`,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
          }
        ]);
      } finally {
        setIsTyping(false);
        setCurrentProgress(null);
      }
    } else {
      // Single Agent path: keep one assistant message and replace its status in place.
      const messageId = `a_${Date.now()}`;
      setMessages((prev) => [...prev, {
        id: messageId,
        sender: 'assistant',
        text: '',
        status: 'working',
        statusLabel: 'Working...',
        activity: [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }]);
      setActiveSingleAgentMessageId(messageId);

      try {
        const taskId = `task-${Date.now()}`;
        const controller = new AbortController();
        activeTaskRef.current = { taskId, controller, messageId };

        const historyMessages = messages
          .filter(m => (m.sender === 'user' || m.sender === 'assistant') && (m.text || m.content) && m.status !== 'working')
          .slice(-10)
          .map(m => ({
            role: m.sender === 'user' ? 'user' : 'assistant',
            content: (m.text || m.content || '').slice(0, 2000),
            task_outcome: m.task_outcome || null,
          }));

        let lastTaskContext = null;
        for (let i = messages.length - 1; i >= 0; i--) {
          if (messages[i].task_outcome) {
            lastTaskContext = messages[i].task_outcome;
            break;
          }
        }

        const convOptions = {
          conversation_id: conversationId,
          messages: historyMessages,
          task_context: lastTaskContext,
        };

        const outcome = await executeSingleAgentAction(text, taskId, controller, intent, messageId, convOptions);
        const active = activeTaskRef.current;
        const terminalStatus = active?.stopping || outcome.status === 'stopped' ? 'stopped' : outcome.status;
        const isConv = Boolean(outcome.is_conversational);
        const defaultFallback = terminalStatus === 'failed'
          ? 'The agent could not complete the task.'
          : (isConv ? 'Hello! How can I assist you today?' : 'Task completed.');
        const cleanText = terminalStatus === 'stopped'
          ? 'The task was stopped by the user.'
          : (outcome.content || outcome.text || defaultFallback).replace(/^### (Agent Error|Connection Error|Browser-Use Result)\s*/, '');
        const isFastPath = Boolean(outcome.fast_path);
        const fastPathBadge = isFastPath ? `⚡ Fast Path (${outcome.adapter_name || 'Adapter'})` : '';
        setMessages((prev) => prev.map((message) => message.id === messageId ? {
          ...message,
          text: cleanText,
          content: cleanText,
          status: terminalStatus,
          statusLabel: terminalStatus === 'stopped' ? 'Task stopped' : terminalStatus === 'failed' ? 'Task failed' : (isFastPath ? fastPathBadge : (isConv ? '' : 'Task completed')),
          activity: message.activity || [],
          durationMs: outcome.durationMs,
          is_conversational: isConv,
          fast_path: isFastPath,
          adapter_name: outcome.adapter_name,
          task_outcome: outcome.task_outcome,
          completed: true,
          isStreaming: false,
        } : message));
      } catch (err) {
        const active = activeTaskRef.current;
        const stopped = active?.stopping || err.name === 'AbortError';
        setMessages((prev) => prev.map((message) => message.id === messageId ? {
          ...message,
          text: stopped ? 'The task was stopped by the user.' : err.message || 'The agent could not complete the task.',
          status: stopped ? 'stopped' : 'failed',
          statusLabel: stopped ? 'Task stopped' : 'Task failed',
          activity: message.activity || [],
          completed: true,
        } : message));
      } finally {
        setIsTyping(false);
        setActiveSingleAgentMessageId(null);
        activeTaskRef.current = null;
      }
    }
  };

  const handleCopy = (id, text) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleClearChat = () => {
    setConversationId('conv_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8));
    setMessages([
      {
        id: 'm1',
        sender: 'assistant',
        text: "Chat session cleared. Comex Multi-Agent Swarm pipeline is idle.",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
    ]);
    setAgentLogs([]);
  };

  if (!isOpen) return null;

  return (
    <div className="chat-drawer animate-slide-right">
      {/* Drawer Header Row 1: Brand & View Switcher */}
      <div className="chat-header">
        <div className="chat-title-group">
          <div className="ai-avatar-badge">
            <Sparkles size={15} />
          </div>
          <div className="view-toggle-pills">
            <button
              className={`view-pill ${activeDrawerTab === 'chat' ? 'active' : ''}`}
              onClick={() => setActiveDrawerTab('chat')}
            >
              Chat
            </button>
            <button
              className={`view-pill ${activeDrawerTab === 'brain' ? 'active' : ''}`}
              onClick={() => setActiveDrawerTab('brain')}
            >
              <Brain size={13} />
              <span>Memory & Brain</span>
            </button>
          </div>
        </div>

        <div className="chat-header-actions">
          {activeDrawerTab === 'chat' ? (
            <button
              className="chat-icon-btn"
              onClick={handleClearChat}
              title="Clear Chat History"
            >
              <Trash2 size={15} />
            </button>
          ) : (
            <button
              className="chat-icon-btn"
              onClick={fetchBrainData}
              title="Refresh Brain Data"
            >
              <RefreshCw size={14} className={isRefreshingBrain ? 'animate-spin' : ''} />
            </button>
          )}
          <button
            className="chat-icon-btn"
            onClick={onClose}
            title="Close Drawer"
          >
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Drawer Header Row 2: Active Tab Badge & Swarm Mode Toggle */}
      {activeDrawerTab === 'chat' && (
        <div className="chat-subbar">
          <div className="active-tab-chip" title={activeTab?.title || "Comex Launchpad"}>
            <Globe size={12} className="tab-chip-icon" />
            <span className="active-tab-text">{activeTab?.title || "Comex Launchpad"}</span>
          </div>

          <button
            className={`agent-mode-toggle ${isMultiAgentMode ? 'active' : ''}`}
            onClick={() => setIsMultiAgentMode(!isMultiAgentMode)}
            title={isMultiAgentMode ? "Multi-Agent Swarm Mode Active" : "Single Agent Mode"}
          >
            <Layers size={12} />
            <span>{isMultiAgentMode ? "Swarm Mode" : "Single Agent"}</span>
          </button>
        </div>
      )}

      {/* Messages List Area OR Brain Inspector */}
      {activeDrawerTab === 'chat' ? (
        <>
          <div className="chat-messages">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`chat-bubble-wrapper ${msg.sender === 'user' ? 'user-wrapper' : 'assistant-wrapper'}`}
              >
                {msg.sender === 'assistant' && (
                  <div className="avatar-icon assistant">
                    <Bot size={13} />
                  </div>
                )}
                <div className={`chat-bubble ${msg.sender} ${msg.status ? `message-${msg.status}` : ''}`}>
                  {msg.sender === 'assistant' && msg.status ? (
                    <>
                      {/* Live In-Progress State: Perception, Thinking & Minimalist Linear Steps */}
                      {msg.status === 'working' ? (
                        <div className="agent-live-execution-panel animate-fade-in">
                          {/* Perception Header */}
                          {msg.perception && (
                            <div className="agent-perception-bar">
                              <Eye size={12} className="perception-icon" />
                              <span className="perception-label">Observing:</span>
                              <span className="perception-target" title={msg.perception.url}>
                                {msg.perception.title || msg.perception.url || 'Active Tab'}
                              </span>
                            </div>
                          )}

                          {/* Minimalist Linear Steps */}
                          <div className="agent-linear-steps">
                            {msg.activity && msg.activity.length > 0 ? (
                              msg.activity.map((stepItem, idx) => {
                                const isCurrent = stepItem.status === 'running';
                                const isFailed = stepItem.status === 'failed';
                                return (
                                  <div
                                    key={stepItem.id || idx}
                                    className={`linear-step-row ${isCurrent ? 'active' : isFailed ? 'failed' : 'done'}`}
                                  >
                                    <div className="step-indicator">
                                      {isCurrent ? (
                                        <span className="step-pulse-dot" />
                                      ) : isFailed ? (
                                        <span className="step-fail-cross">✕</span>
                                      ) : (
                                        <span className="step-check-mark">✓</span>
                                      )}
                                    </div>
                                    <div className="step-body">
                                      <span className="step-label-text">{stepItem.label}</span>
                                      {stepItem.tool && (
                                        <span className="step-tool-badge">{stepItem.tool}</span>
                                      )}
                                    </div>
                                  </div>
                                );
                              })
                            ) : (
                              <div className="linear-step-row active">
                                <div className="step-indicator">
                                  <span className="step-pulse-dot" />
                                </div>
                                <div className="step-body">
                                  <span className="step-label-text">{msg.statusLabel || 'Analyzing request...'}</span>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      ) : (
                        /* Completed/Stopped State: Slim Collapsible Summary Bar (only when tools were run) */
                        <>
                          {msg.activity && msg.activity.length > 0 && !msg.is_conversational && (
                            <div className="agent-completed-container">
                              <button
                                className="agent-completion-bar"
                                onClick={() => setExpandedActivity((prev) => ({ ...prev, [msg.id]: !prev[msg.id] }))}
                                aria-expanded={Boolean(expandedActivity[msg.id])}
                                type="button"
                              >
                                <div className="completion-summary-left">
                                  <span className={`completion-status-icon ${msg.status === 'completed' ? 'success' : 'warn'}`}>
                                    {msg.status === 'completed' ? '✓' : '•'}
                                  </span>
                                  <span className="completion-title">
                                    {msg.status === 'stopped' ? 'Task stopped' : msg.status === 'failed' ? 'Task failed' : 'Task completed'}
                                  </span>
                                  <span className="completion-meta-tag">
                                    {msg.activity.length} {msg.activity.length === 1 ? 'step' : 'steps'}
                                  </span>
                                  {msg.durationMs != null && (
                                    <span className="completion-meta-tag">
                                      {formatDuration(msg.durationMs)}
                                    </span>
                                  )}
                                </div>
                                <div className="completion-toggle-btn">
                                  <span>{expandedActivity[msg.id] ? 'Hide steps' : 'View steps'}</span>
                                  {expandedActivity[msg.id] ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                                </div>
                              </button>

                              {expandedActivity[msg.id] && (
                                <div className="agent-expanded-trajectory animate-fade-in">
                                  {msg.activity.map((stepItem, idx) => (
                                    <div key={stepItem.id || idx} className="trajectory-item">
                                      <span className="trajectory-num">{idx + 1}</span>
                                      <span className="trajectory-label">{stepItem.label}</span>
                                      {stepItem.tool && (
                                        <span className="trajectory-tool">{stepItem.tool}</span>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}
                        </>
                      )}
                      {(msg.text || msg.content) && (
                        <div className="chat-message-text">
                          <MarkdownRenderer content={msg.text || msg.content} />
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="bubble-text">
                      {msg.sender === 'user' ? <p>{msg.text || msg.content}</p> : <MarkdownRenderer content={msg.text || msg.content} />}
                    </div>
                  )}
                  <div className="bubble-footer">
                    <span className="timestamp">{msg.timestamp}</span>
                    {msg.durationMs != null && (
                      <span className="agent-duration" title="Time taken by the agent">
                        ⏱ {formatDuration(msg.durationMs)}
                      </span>
                    )}
                    {msg.sender === 'assistant' && (
                      <button className="copy-btn" onClick={() => handleCopy(msg.id, msg.text)} title="Copy message">
                        {copiedId === msg.id ? <Check size={12} /> : <Copy size={12} />}
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}

            {/* Live Multi-Agent Swarm Execution Log Feed */}
            {isTyping && isMultiAgentMode && (
              <div className="agent-activity-box animate-fade-in">
                <div className="agent-activity-header">
                  <Activity size={14} className="animate-spin text-accent" />
                  <span>Multi-Agent Swarm Pipeline</span>
                  {currentProgress && (
                    <span className="step-tag">Step {currentProgress.step}/{currentProgress.totalSteps}</span>
                  )}
                </div>

                <div className="agent-logs-list">
                  {agentLogs.map((log) => (
                    <div key={log.id} className="agent-log-item animate-fade-in">
                      <span className="agent-role-badge" style={{ color: log.agent.color }}>
                        {getAgentRoleIcon(log.agent.id, 12)}
                        <span className="badge-name">{log.agent.name}</span>
                      </span>
                      <span className="agent-log-msg">{log.message}</span>
                      <span className="agent-log-time">{log.timestamp}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick Action Prompt Chips */}
          <div className="prompt-chips-bar">
            <button className="chip-btn" onClick={() => handleSend("Research trending AI frameworks on GitHub")}>
              <Layers size={12} /> GitHub Research Swarm
            </button>
            <button className="chip-btn" onClick={() => handleSend("Open YouTube and search for Lo-Fi Beats")}>
              <Play size={12} /> YouTube Search
            </button>
            <button className="chip-btn" onClick={() => handleSend("Search Google for Quantum Computing 2026")}>
              <Globe size={12} /> Google Query
            </button>
          </div>

          {/* Input Textarea Bar */}
          <div className="chat-input-container">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSend();
              }}
              className="chat-form"
            >
              <input
                type="text"
                className="chat-input"
                placeholder={isMultiAgentMode ? "Ask Multi-Agent Swarm..." : "Ask Comex AI Assistant..."}
                value={input}
                onChange={(e) => setInput(e.target.value)}
              />
              <button
                type={isTyping && !isMultiAgentMode ? 'button' : 'submit'}
                className={`send-btn ${isTyping && !isMultiAgentMode ? 'stop-btn' : ''}`}
                disabled={isTyping && !isMultiAgentMode ? Boolean(activeTaskRef.current?.stopping) : !input.trim() || isTyping}
                onClick={isTyping && !isMultiAgentMode ? stopSingleAgent : undefined}
                aria-label={isTyping && !isMultiAgentMode ? 'Stop agent task' : 'Send message'}
                title={isTyping && !isMultiAgentMode ? 'Stop agent task' : 'Send message'}
              >
                {isTyping && !isMultiAgentMode ? <span className="stop-icon" aria-hidden="true" /> : <Send size={14} />}
              </button>
            </form>
          </div>
        </>
      ) : (
        /* Brain & Memory Inspector View */
        <div className="brain-inspector animate-fade-in">
          {/* Brain Stats Overview Bar */}
          {memoryStats && (
            <div className="brain-stats-bar">
              <div className="stat-pill"><span className="stat-label">User Facts:</span> <span className="stat-val">{memoryStats.usermemory_count || 0}</span></div>
              <div className="stat-pill"><span className="stat-label">Skills:</span> <span className="stat-val">{memoryStats.skill_count || 0}</span></div>
              <div className="stat-pill"><span className="stat-label">Reflections:</span> <span className="stat-val">{memoryStats.reflection_count || 0}</span></div>
              <div className="stat-pill"><span className="stat-label">Artifacts:</span> <span className="stat-val">{memoryStats.artifact_count || 0}</span></div>
            </div>
          )}

          {/* Sub-tabs */}
          <div className="brain-nav-tabs">
            <button
              className={`brain-tab-btn ${brainSection === 'user_memory' ? 'active' : ''}`}
              onClick={() => setBrainSection('user_memory')}
            >
              <User size={13} />
              <span>User Facts ({userMemories.length})</span>
            </button>
            <button
              className={`brain-tab-btn ${brainSection === 'adapters' ? 'active' : ''}`}
              onClick={() => setBrainSection('adapters')}
            >
              <Zap size={13} />
              <span>Fast Adapters ({adapters.length})</span>
            </button>
            <button
              className={`brain-tab-btn ${brainSection === 'skills' ? 'active' : ''}`}
              onClick={() => setBrainSection('skills')}
            >
              <Cpu size={13} />
              <span>Learned Skills ({skills.length})</span>
            </button>
            <button
              className={`brain-tab-btn ${brainSection === 'reflections' ? 'active' : ''}`}
              onClick={() => setBrainSection('reflections')}
            >
              <Lightbulb size={13} />
              <span>Task Reflections ({reflections.length})</span>
            </button>
          </div>

          <div className="brain-tab-content">
            {/* 1. User Memory */}
            {brainSection === 'user_memory' && (
              <div className="brain-memory-panel">
                <form className="add-memory-form" onSubmit={handleAddMemory}>
                  <div className="add-memory-inputs">
                    <input
                      type="text"
                      className="brain-input"
                      placeholder="Key (e.g. preferred_theme)"
                      value={newKey}
                      onChange={e => setNewKey(e.target.value)}
                    />
                    <input
                      type="text"
                      className="brain-input"
                      placeholder="Value (e.g. dark)"
                      value={newValue}
                      onChange={e => setNewValue(e.target.value)}
                    />
                    <select
                      className="brain-select"
                      value={newCategory}
                      onChange={e => setNewCategory(e.target.value)}
                    >
                      <option value="preference">preference</option>
                      <option value="constraint">constraint</option>
                      <option value="profile">profile</option>
                    </select>
                  </div>
                  <button type="submit" className="add-memory-btn" disabled={!newKey.trim() || !newValue.trim()}>
                    <Plus size={13} /> Save Fact
                  </button>
                </form>

                <div className="brain-items-list">
                  {userMemories.length === 0 ? (
                    <div className="brain-empty">No user facts saved yet. The agent automatically learns facts or you can add them above.</div>
                  ) : (
                    userMemories.map(m => (
                      <div key={m.id || m.key} className="brain-card memory-card">
                        <div className="card-header">
                          <span className="card-badge">{m.category}</span>
                          <span className="card-key">{m.key}</span>
                          <button
                            className="card-del-btn"
                            onClick={() => handleDeleteMemory(m.key)}
                            title="Delete Fact"
                          >
                            <Trash2 size={13} />
                          </button>
                        </div>
                        <div className="card-value">{m.value}</div>
                        {m.updated_at && <div className="card-time">{new Date(m.updated_at).toLocaleString()}</div>}
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}

            {/* 2. Skills */}
            {brainSection === 'skills' && (
              <div className="brain-skills-panel">
                <div className="skills-toolbar" style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '10px' }}>
                  <button
                    className="add-memory-btn"
                    style={{ fontSize: '0.72rem', padding: '4px 10px', background: '#252833', border: '1px solid #333644' }}
                    onClick={handleConsolidateSkills}
                    title="Consolidate duplicate and near-identical skills into generalized entries"
                  >
                    <Cpu size={12} /> Consolidate & Deduplicate
                  </button>
                </div>
                <div className="brain-items-list">
                  {skills.length === 0 ? (
                    <div className="brain-empty">No reusable interaction skills distilled yet. The agent self-distills skills after novel tasks.</div>
                  ) : (
                    skills.map(s => (
                      <div key={s.id} className="brain-card skill-card">
                        <div className="card-header">
                          <span className="card-badge skill-badge">{s.domain}</span>
                          <span className="card-key">{s.name}</span>
                          <button
                            className="card-del-btn"
                            onClick={() => handleDeleteSkill(s.id)}
                            title="Delete Skill"
                          >
                            <Trash2 size={13} />
                          </button>
                        </div>
                        <p className="skill-desc">{s.description}</p>
                        {s.code_snippet && (
                          <pre className="skill-code">
                            <code>{s.code_snippet}</code>
                          </pre>
                        )}
                        <div className="skill-metrics">
                          <span>Uses: {s.usage_count || 0}</span>
                          <span>Success: {s.success_count || 0}</span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}

            {/* 3. Reflections */}
            {brainSection === 'reflections' && (
              <div className="brain-reflections-panel">
                <div className="brain-items-list">
                  {reflections.length === 0 ? (
                    <div className="brain-empty">No task self-evaluations recorded yet. Reflections run in the background after each task.</div>
                  ) : (
                    reflections.map(r => (
                      <div key={r.id} className="brain-card reflection-card">
                        <div className="card-header">
                          <span className={`card-badge ${r.is_novel ? 'novel-badge' : 'standard-badge'}`}>
                            {r.is_novel ? '★ Novel Task' : 'Standard Routine'}
                          </span>
                          <span className="card-time">{r.created_at}</span>
                        </div>
                        {r.novelty_reason && (
                          <div className="reflection-section">
                            <strong>Novelty Assessment:</strong>
                            <p>{r.novelty_reason}</p>
                          </div>
                        )}
                        {r.critique && (
                          <div className="reflection-section">
                            <strong>Trajectory Critique:</strong>
                            <p>{r.critique}</p>
                          </div>
                        )}
                        {r.lesson_learned && (
                          <div className="reflection-section highlight">
                            <strong>Lesson Learned:</strong>
                            <p>{r.lesson_learned}</p>
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}

            {/* 4. Fast Adapters */}
            {brainSection === 'adapters' && (
              <div className="brain-skills-panel">
                <div className="brain-items-list">
                  {adapters.length === 0 ? (
                    <div className="brain-empty">
                      No compiled site adapters yet. When you complete routine actions, Comex automatically distills them into sub-second Fast-Path adapters.
                    </div>
                  ) : (
                    adapters.map(a => (
                      <div key={a.id} className="brain-card skill-card">
                        <div className="card-header">
                          <span className={`card-badge ${a.status === 'active' ? 'skill-badge' : 'warn'}`}>
                            {a.domain} ({a.status})
                          </span>
                          <span className="card-key">{a.name}</span>
                          <button
                            className="card-del-btn"
                            onClick={() => handleDeleteAdapter(a.id)}
                            title="Delete Adapter"
                          >
                            <Trash2 size={13} />
                          </button>
                        </div>
                        <p className="skill-desc"><strong>Intent:</strong> {a.action_intent}</p>
                        {a.code_template && (
                          <pre className="skill-code">
                            <code>{a.code_template}</code>
                          </pre>
                        )}
                        <div className="skill-metrics">
                          <span>⚡ Successes: {a.success_count || 0}</span>
                          <span>Failures: {a.failure_count || 0}</span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

    </div>
  );
}
