import React, { useState, useRef, useEffect } from 'react';
import {
  X, ArrowUp, Square, Copy, Check, ChevronRight, ChevronDown, Brain, ArrowLeft,
  SquarePen, RefreshCw, Trash2, Plus, Loader2, Circle, Minus, ExternalLink, Users, User
} from 'lucide-react';
import MarkdownRenderer from './MarkdownRenderer';
import { runSwarm, stopSwarmRun, respondToSwarmApproval } from '../../agents/orchestratorClient';
import './ChatDrawer.css';

function formatDuration(durationMs) {
  const totalSeconds = Math.max(0, durationMs) / 1000;
  if (totalSeconds < 10) return `${totalSeconds.toFixed(1)}s`;
  if (totalSeconds < 60) return `${Math.round(totalSeconds)}s`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.round(totalSeconds % 60);
  return `${minutes}m ${seconds}s`;
}

function nowTime() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Re-renders every second while `active`, for live elapsed timers.
function useNow(active) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!active) return undefined;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [active]);
  return now;
}

const TERMINAL_STATUSES = new Set(['completed', 'failed', 'cancelled', 'blocked']);

function StatusIcon({ status }) {
  if (status === 'running') return <Loader2 size={14} className="status-icon status-running animate-spin" aria-label="Running" />;
  if (status === 'completed') return <Check size={14} className="status-icon status-ok" aria-label="Done" />;
  if (status === 'failed' || status === 'blocked') return <X size={14} className="status-icon status-err" aria-label="Failed" />;
  if (status === 'cancelled') return <Minus size={14} className="status-icon status-muted" aria-label="Stopped" />;
  return <Circle size={12} className="status-icon status-muted" aria-label="Queued" />;
}

const RUN_PHASE_LABEL = {
  planning: 'Planning the work',
  working: 'Agents are working',
  comparing: 'Comparing results',
  approval: 'Waiting for your approval',
  acting: 'Carrying out the approved step',
  done: 'Done',
  failed: 'Run failed',
  stopped: 'Stopped',
};

function RunPanel({ run, onOpenTab }) {
  const isLive = !['done', 'failed', 'stopped'].includes(run.phase);
  const now = useNow(isLive);
  const [expanded, setExpanded] = useState(true);
  const elapsed = (run.endedAt || now) - run.startedAt;
  const working = run.agents.filter((a) => a.status === 'running').length;
  const finished = run.agents.filter((a) => a.status === 'completed').length;
  const n = run.agents.length;
  let headline = RUN_PHASE_LABEL[run.phase];
  if (run.phase === 'working' && n > 0) headline = n === 1 ? 'Agent is working' : `${working} of ${n} agents working`;
  if (run.phase === 'done' && n > 0) {
    headline = n === 1
      ? (finished ? 'Agent finished' : 'Agent stopped')
      : finished === n ? `${n} agents finished` : `${finished} of ${n} agents finished`;
  }

  useEffect(() => {
    if (run.phase === 'done') setExpanded(false);
  }, [run.phase]);

  return (
    <section className={`run-panel run-${run.phase}`} aria-label="Multi-agent run" aria-live="polite">
      {isLive && <div className="run-progress" aria-hidden="true" />}
      <button type="button" className="run-header" onClick={() => setExpanded((v) => !v)} aria-expanded={expanded}>
        <StatusIcon status={isLive ? 'running' : run.phase === 'done' ? 'completed' : run.phase === 'stopped' ? 'cancelled' : 'failed'} />
        <span className="run-headline">{headline}</span>
        <span className="run-elapsed">{formatDuration(elapsed)}</span>
        {expanded ? <ChevronDown size={14} className="run-chevron" /> : <ChevronRight size={14} className="run-chevron" />}
      </button>

      {expanded && run.agents.length > 0 && (
        <ol className="run-agents">
          {run.agents.map((agent) => {
            const agentElapsed = agent.startedAt ? (agent.endedAt || now) - agent.startedAt : null;
            const canOpen = Boolean(agent.tabId && onOpenTab);
            return (
              <li key={agent.taskId} className={`run-agent agent-${agent.status}`}>
                <StatusIcon status={agent.status} />
                <div className="run-agent-body">
                  <span className="run-agent-task">{agent.description}</span>
                  {agent.status === 'running' && agent.currentStep && (
                    <span className="run-agent-step">{agent.currentStep}</span>
                  )}
                  {agent.error && <span className="run-agent-error">{agent.error}</span>}
                </div>
                {agentElapsed != null && <span className="run-agent-time">{formatDuration(agentElapsed)}</span>}
                {canOpen && (
                  <button
                    type="button"
                    className="run-agent-open"
                    onClick={() => onOpenTab(agent.tabId)}
                    aria-label={`Watch this agent's tab: ${agent.description}`}
                    title="Watch this agent's tab"
                  >
                    <ExternalLink size={13} />
                  </button>
                )}
              </li>
            );
          })}
          {run.phase === 'comparing' && (
            <li className="run-agent agent-running">
              <StatusIcon status="running" />
              <div className="run-agent-body"><span className="run-agent-task">Comparing what the agents found</span></div>
            </li>
          )}
        </ol>
      )}
    </section>
  );
}

// Helper agents started by the single agent, shaped like a multi-agent run for RunPanel
function helperRun(helpers) {
  const running = helpers.some((h) => h.status === 'running' || h.status === 'queued');
  return {
    phase: running ? 'working' : 'done',
    startedAt: Math.min(...helpers.map((h) => h.startedAt || Date.now())),
    endedAt: running ? null : Math.max(...helpers.map((h) => h.endedAt || Date.now())),
    agents: helpers,
  };
}

const SUGGESTIONS = {
  single: [
    'Summarize this page',
    'Find the cheapest flight from Bengaluru to Delhi next Friday',
    'Open YouTube and search for lo-fi beats',
  ],
  multi: [
    'Compare the best headphones under ₹3,000 on Amazon and Flipkart',
    'Compare the top stories on Hacker News and Lobsters',
    'Research the pros and cons of Rust vs Go for a web backend',
  ],
};

function EmptyState({ isMultiAgentMode, onPick }) {
  const items = isMultiAgentMode ? SUGGESTIONS.multi : SUGGESTIONS.single;
  return (
    <div className="chat-empty">
      <h2 className="chat-empty-title">{isMultiAgentMode ? 'What should the agents work on?' : 'What can Comex do for you?'}</h2>
      <p className="chat-empty-text">
        {isMultiAgentMode
          ? 'Comex splits the task across up to four agents. Each opens its own tab so you can watch, then the results are compared into one answer.'
          : 'Comex works in the current tab: it can read the page, search, click and fill in forms for you.'}
      </p>
      <ul className="chat-suggestions">
        {items.map((text) => (
          <li key={text}>
            <button type="button" className="chat-suggestion" onClick={() => onPick(text)}>{text}</button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function ChatDrawer({
  isOpen,
  onClose,
  activeTab,
  onSelectTab
}) {
  const [isMultiAgentMode, setIsMultiAgentMode] = useState(false);
  const [conversationId, setConversationId] = useState(() => 'conv_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8));
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
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
  const inputRef = useRef(null);

  // Grow the composer with its content, up to the CSS max-height.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${el.scrollHeight}px`;
  }, [input]);

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
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, isTyping, activeSingleAgentMessageId]);

  // Track active Single Agent message and append real tab activity details
  useEffect(() => {
    if (!activeSingleAgentMessageId) return;
    if (!activeTab) return;

    setMessages((prev) => prev.map((msg) => {
      if (msg.id !== activeSingleAgentMessageId) return msg;

      const newActivity = [...(msg.activity || [])];
      let stepMessage = '';

      if (activeTab.isLoading) {
        stepMessage = `Loading ${activeTab.url || 'page'}`;
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

      const latestMsg = newActivity[newActivity.length - 1]?.message || 'Starting';

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
                    statusLabel: data.detail || 'Thinking',
                    thinking: data.detail
                  };
                }

                if (currentEventType === 'planning') {
                  return {
                    ...msg,
                    statusLabel: data.detail || 'Planning the next step'
                  };
                }

                if (currentEventType.startsWith('helper_')) {
                  const helpers = [...(msg.helpers || [])];
                  const idx = helpers.findIndex((h) => h.taskId === data.agent_id);
                  const base = idx >= 0 ? helpers[idx] : {
                    taskId: data.agent_id, description: data.description, status: 'running', startedAt: Date.now(),
                  };
                  let next = { ...base, tabId: data.tab_id || base.tabId };
                  if (currentEventType === 'helper_progress') next.currentStep = data.label;
                  if (currentEventType === 'helper_finished') {
                    next = { ...next, status: data.status, endedAt: Date.now(), currentStep: null,
                      error: data.status === 'completed' ? null : data.error };
                  }
                  if (idx >= 0) helpers[idx] = next; else helpers.push(next);
                  return { ...msg, helpers };
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
                      : (outcome.fast_path ? `Fast path: ${outcome.adapter_name || 'adapter'}` : (isConv ? '' : 'Task completed')));
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
      return { status: 'failed', text: "Can't reach the Comex agent service at 127.0.0.1:8000. Start it with `python agent_service.py`, then try again." };
    }
  };

  const stopSingleAgent = async () => {
    const active = activeTaskRef.current;
    if (!active || active.stopping) return;

    active.stopping = true;
    setMessages((prev) => prev.map((message) => message.id === active.messageId ? {
      ...message,
      statusLabel: 'Stopping',
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

  const respondToApproval = async (messageId, runId, approved) => {
    setMessages((prev) => prev.map((m) => m.id === messageId && m.approval
      ? { ...m, approval: { ...m.approval, state: approved ? 'approved' : 'declined' } }
      : m));
    try {
      await respondToSwarmApproval(runId, approved);
    } catch (err) {
      console.warn('Approval response failed:', err);
    }
  };

  const stopSwarm = async () => {
    const active = activeTaskRef.current;
    if (!active || active.stopping) return;

    active.stopping = true;
    active.controller.abort();
    try {
      await stopSwarmRun(active.runId);
    } catch (err) {
      console.warn('Swarm cancellation request failed:', err);
    }
  };

  const runMultiAgent = async (text) => {
    const controller = new AbortController();
    activeTaskRef.current = { controller, runId: null, stopping: false };
    const messageId = `a_run_${Date.now()}`;
    setMessages((prev) => [...prev, {
      id: messageId,
      sender: 'assistant',
      kind: 'run',
      text: '',
      timestamp: nowTime(),
      run: { phase: 'planning', startedAt: Date.now(), endedAt: null, agents: [] },
    }]);

    const updateMessage = (fn) => setMessages((prev) => prev.map((m) => (m.id === messageId ? fn(m) : m)));
    const updateRun = (fn) => updateMessage((m) => ({ ...m, run: fn(m.run) }));

    const handleTaskUpdate = (ev) => {
      switch (ev.eventType) {
        case 'run_started':
          if (activeTaskRef.current) activeTaskRef.current.runId = ev.run_id;
          updateRun((run) => ({
            ...run,
            phase: 'working',
            agents: (ev.tasks || []).map((t) => ({ taskId: t.task_id, description: t.description, status: 'queued' })),
          }));
          return;
        case 'task_started':
          updateRun((run) => {
            const started = { taskId: ev.task_id, description: ev.description, status: 'running', tabId: ev.tab_id, startedAt: Date.now() };
            const exists = run.agents.some((a) => a.taskId === ev.task_id);
            return {
              ...run,
              phase: ev.task_id === 'follow_up' ? 'acting' : run.phase,
              agents: exists ? run.agents.map((a) => (a.taskId === ev.task_id ? { ...a, ...started } : a)) : [...run.agents, started],
            };
          });
          return;
        case 'task_progress':
          updateRun((run) => ({
            ...run,
            agents: run.agents.map((a) => (a.taskId === ev.task_id ? { ...a, currentStep: ev.label } : a)),
          }));
          return;
        case 'synthesis_started':
          updateRun((run) => ({ ...run, phase: 'comparing' }));
          return;
        case 'approval_required':
          updateMessage((m) => ({
            ...m,
            text: ev.proposal,
            approval: { runId: ev.run_id, action: ev.action, state: 'pending' },
            run: { ...m.run, phase: 'approval' },
          }));
          return;
        default: {
          const status = ev.eventType?.startsWith('task_') ? ev.eventType.slice(5) : null;
          if (status && TERMINAL_STATUSES.has(status)) {
            updateRun((run) => ({
              ...run,
              agents: run.agents.map((a) => (a.taskId === ev.task_id
                ? { ...a, status, endedAt: Date.now(), error: status === 'completed' ? null : ev.error }
                : a)),
            }));
          }
        }
      }
    };

    const expireApproval = (m) => (m.approval?.state === 'pending' ? { ...m.approval, state: 'expired' } : m.approval);

    try {
      const result = await runSwarm(text, { signal: controller.signal, onTaskUpdate: handleTaskUpdate });
      updateMessage((m) => ({
        ...m,
        text: result.finalAnswer,
        approval: expireApproval(m),
        run: { ...m.run, phase: 'done', endedAt: Date.now() },
      }));
    } catch (err) {
      const stopped = err.name === 'AbortError' || activeTaskRef.current?.stopping;
      updateMessage((m) => ({
        ...m,
        text: stopped
          ? 'Stopped. The agents’ tabs stay open if you want to look at what they found.'
          : `The run failed: ${err.message}`,
        failed: !stopped,
        approval: expireApproval(m),
        run: {
          ...m.run,
          phase: stopped ? 'stopped' : 'failed',
          endedAt: Date.now(),
          agents: m.run.agents.map((a) => (a.status === 'running' || a.status === 'queued' ? { ...a, status: 'cancelled', endedAt: Date.now() } : a)),
        },
      }));
    } finally {
      activeTaskRef.current = null;
      setIsTyping(false);
    }
  };

  const runSingleAgent = async (text, intent) => {
    const messageId = `a_${Date.now()}`;
    setMessages((prev) => [...prev, {
      id: messageId,
      sender: 'assistant',
      text: '',
      status: 'working',
      statusLabel: 'Starting',
      activity: [],
      startedAt: Date.now(),
      timestamp: nowTime()
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

      const outcome = await executeSingleAgentAction(text, taskId, controller, intent, messageId, {
        conversation_id: conversationId,
        messages: historyMessages,
        task_context: lastTaskContext,
      });
      const active = activeTaskRef.current;
      const terminalStatus = active?.stopping || outcome.status === 'stopped' ? 'stopped' : outcome.status;
      const isConv = Boolean(outcome.is_conversational);
      const defaultFallback = terminalStatus === 'failed'
        ? 'The agent could not complete the task.'
        : (isConv ? 'Hello! How can I help?' : 'Task completed.');
      const cleanText = terminalStatus === 'stopped'
        ? 'Stopped.'
        : (outcome.content || outcome.text || defaultFallback).replace(/^### (Agent Error|Connection Error|Browser-Use Result)\s*/, '');
      const isFastPath = Boolean(outcome.fast_path);
      setMessages((prev) => prev.map((message) => message.id === messageId ? {
        ...message,
        text: cleanText,
        content: cleanText,
        status: terminalStatus,
        statusLabel: terminalStatus === 'stopped' ? 'Stopped' : terminalStatus === 'failed' ? 'Failed' : (isConv ? '' : 'Completed'),
        activity: message.activity || [],
        durationMs: outcome.durationMs ?? (Date.now() - message.startedAt),
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
        text: stopped ? 'Stopped.' : err.message || 'The agent could not complete the task.',
        status: stopped ? 'stopped' : 'failed',
        statusLabel: stopped ? 'Stopped' : 'Failed',
        activity: message.activity || [],
        completed: true,
      } : message));
    } finally {
      setIsTyping(false);
      setActiveSingleAgentMessageId(null);
      activeTaskRef.current = null;
    }
  };

  const handleSend = async (textToSend = input, intent = undefined) => {
    const text = textToSend.trim();
    if (!text || isTyping) return;

    setMessages((prev) => [...prev, { id: `u_${Date.now()}`, sender: 'user', text, timestamp: nowTime() }]);
    if (textToSend === input) setInput('');
    setIsTyping(true);

    if (isMultiAgentMode) {
      await runMultiAgent(text);
    } else {
      await runSingleAgent(text, intent);
    }
  };

  const handleComposerKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleCopy = (id, text) => {
    navigator.clipboard.writeText(text || '');
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleNewChat = () => {
    if (isTyping) return;
    setConversationId('conv_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8));
    setMessages([]);
    setExpandedActivity({});
    inputRef.current?.focus();
  };

  if (!isOpen) return null;

  const tabTitle = activeTab?.url && !activeTab.url.startsWith('comex://') ? activeTab.title : null;

  const renderFooter = (msg) => (
    <div className="msg-footer">
      <span>{msg.timestamp}</span>
      {msg.durationMs != null && <span>{formatDuration(msg.durationMs)}</span>}
      {msg.fast_path && <span className="msg-tag">Fast path</span>}
      <button
        type="button"
        className="msg-copy"
        onClick={() => handleCopy(msg.id, msg.text)}
        aria-label={copiedId === msg.id ? 'Copied' : 'Copy reply'}
        title="Copy"
      >
        {copiedId === msg.id ? <Check size={13} /> : <Copy size={13} />}
      </button>
    </div>
  );

  const renderSingleAgentWorking = (msg) => (
    <div className="agent-working-panel" aria-live="polite">
      <div className="agent-working-status">
        <Loader2 size={14} className="status-icon status-running animate-spin" />
        <span className="agent-working-label">{msg.statusLabel || 'Starting'}</span>
        <ElapsedTime since={msg.startedAt} />
      </div>
      {msg.perception && (
        <div className="agent-working-context" title={msg.perception.url}>
          Reading {msg.perception.title || msg.perception.url}
        </div>
      )}
      {msg.activity?.length > 0 && (
        <ol className="agent-steps">
          {msg.activity.map((stepItem, idx) => (
            <li key={stepItem.id || idx} className={`agent-step step-${stepItem.status || 'done'}`}>
              <StatusIcon status={stepItem.status === 'running' ? 'running' : stepItem.status === 'failed' ? 'failed' : 'completed'} />
              <span className="agent-step-label">{stepItem.label || stepItem.message}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );

  const renderStepsToggle = (msg) => {
    const steps = msg.activity || [];
    if (steps.length === 0 || msg.is_conversational) return null;
    const open = Boolean(expandedActivity[msg.id]);
    return (
      <div className="steps-toggle-group">
        <button
          type="button"
          className="steps-toggle"
          onClick={() => setExpandedActivity((prev) => ({ ...prev, [msg.id]: !prev[msg.id] }))}
          aria-expanded={open}
        >
          <StatusIcon status={msg.status === 'completed' ? 'completed' : msg.status === 'stopped' ? 'cancelled' : 'failed'} />
          <span>{msg.statusLabel || 'Completed'} in {steps.length} {steps.length === 1 ? 'step' : 'steps'}</span>
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </button>
        {open && (
          <ol className="agent-steps agent-steps-history">
            {steps.map((stepItem, idx) => (
              <li key={stepItem.id || idx} className={`agent-step step-${stepItem.status || 'done'}`}>
                <StatusIcon status={stepItem.status === 'failed' ? 'failed' : 'completed'} />
                <span className="agent-step-label">{stepItem.label || stepItem.message}</span>
                {stepItem.tool && <span className="agent-step-tool">{stepItem.tool}</span>}
              </li>
            ))}
          </ol>
        )}
      </div>
    );
  };

  const renderApproval = (msg) => {
    const { approval } = msg;
    if (!approval) return null;
    return (
      <div className={`approval approval-${approval.state}`} role="group" aria-label="Approval needed">
        <p className="approval-question">Comex wants to: <strong>{approval.action}</strong></p>
        {approval.state === 'pending' ? (
          <div className="approval-actions">
            <button type="button" className="btn btn-primary" onClick={() => respondToApproval(msg.id, approval.runId, true)}>
              Approve
            </button>
            <button type="button" className="btn btn-secondary" onClick={() => respondToApproval(msg.id, approval.runId, false)}>
              Decline
            </button>
            <span className="approval-note">Comex never checks out or pays.</span>
          </div>
        ) : (
          <p className="approval-state" role="status">
            {{ approved: 'Approved', declined: 'Declined', expired: 'No longer waiting for approval' }[approval.state]}
          </p>
        )}
      </div>
    );
  };

  const renderMessage = (msg) => {
    if (msg.sender === 'user') {
      return (
        <div key={msg.id} className="msg msg-user">
          <div className="msg-user-bubble">{msg.text}</div>
        </div>
      );
    }

    const isWorking = msg.status === 'working';
    return (
      <article key={msg.id} className={`msg msg-assistant ${msg.status === 'failed' || msg.failed ? 'msg-failed' : ''}`}>
        {msg.kind === 'run' && <RunPanel run={msg.run} onOpenTab={onSelectTab} />}
        {msg.helpers?.length > 0 && <RunPanel run={helperRun(msg.helpers)} onOpenTab={onSelectTab} />}
        {isWorking ? renderSingleAgentWorking(msg) : renderStepsToggle(msg)}
        {(msg.text || msg.content) && (
          <div className="msg-body">
            <MarkdownRenderer content={msg.text || msg.content} />
          </div>
        )}
        {renderApproval(msg)}
        {!isWorking && (msg.text || msg.content) && renderFooter(msg)}
      </article>
    );
  };

  return (
    <aside className="chat-drawer animate-slide-right" aria-label="Comex assistant">
      <header className="drawer-header">
        {activeDrawerTab === 'brain' ? (
          <>
            <button type="button" className="icon-btn" onClick={() => setActiveDrawerTab('chat')} aria-label="Back to chat" title="Back to chat">
              <ArrowLeft size={16} />
            </button>
            <h2 className="drawer-title">Memory</h2>
            <div className="drawer-actions">
              <button type="button" className="icon-btn" onClick={fetchBrainData} aria-label="Refresh memory" title="Refresh">
                <RefreshCw size={15} className={isRefreshingBrain ? 'animate-spin' : ''} />
              </button>
              <button type="button" className="icon-btn" onClick={onClose} aria-label="Close assistant" title="Close (⌘J)">
                <X size={16} />
              </button>
            </div>
          </>
        ) : (
          <>
            <h2 className="drawer-title">Assistant</h2>
            <div className="drawer-actions">
              <button type="button" className="icon-btn" onClick={() => setActiveDrawerTab('brain')} aria-label="Memory" title="Memory">
                <Brain size={15} />
              </button>
              <button type="button" className="icon-btn" onClick={handleNewChat} disabled={isTyping} aria-label="New chat" title="New chat">
                <SquarePen size={15} />
              </button>
              <button type="button" className="icon-btn" onClick={onClose} aria-label="Close assistant" title="Close (⌘J)">
                <X size={16} />
              </button>
            </div>
          </>
        )}
      </header>

      {activeDrawerTab === 'chat' ? (
        <>
          <div className="drawer-modebar">
            <div className="mode-switch" role="radiogroup" aria-label="Agent mode">
              <button
                type="button"
                role="radio"
                aria-checked={!isMultiAgentMode}
                className={`mode-option ${!isMultiAgentMode ? 'selected' : ''}`}
                onClick={() => setIsMultiAgentMode(false)}
                disabled={isTyping}
              >
                <User size={13} /> Single agent
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={isMultiAgentMode}
                className={`mode-option ${isMultiAgentMode ? 'selected' : ''}`}
                onClick={() => setIsMultiAgentMode(true)}
                disabled={isTyping}
              >
                <Users size={13} /> Multi-agent
              </button>
            </div>
            <span className="drawer-context" title={tabTitle || ''}>
              {isMultiAgentMode ? 'Agents open their own tabs' : tabTitle ? `In: ${tabTitle}` : 'New tab'}
            </span>
          </div>

          <div className="chat-messages">
            {messages.length === 0
              ? <EmptyState isMultiAgentMode={isMultiAgentMode} onPick={(text) => handleSend(text)} />
              : messages.map(renderMessage)}
            <div ref={messagesEndRef} />
          </div>

          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
          >
            <textarea
              ref={inputRef}
              rows={1}
              className="composer-input"
              placeholder={isMultiAgentMode ? 'Describe a task to split across agents' : 'Ask about this page, or tell Comex what to do'}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleComposerKeyDown}
              aria-label="Message"
            />
            <button
              type={isTyping ? 'button' : 'submit'}
              className={`composer-send ${isTyping ? 'is-stop' : ''}`}
              disabled={isTyping ? Boolean(activeTaskRef.current?.stopping) : !input.trim()}
              onClick={isTyping ? (isMultiAgentMode ? stopSwarm : stopSingleAgent) : undefined}
              aria-label={isTyping ? 'Stop' : 'Send'}
              title={isTyping ? 'Stop' : 'Send (Enter)'}
            >
              {isTyping ? <Square size={11} fill="currentColor" /> : <ArrowUp size={16} />}
            </button>
          </form>
        </>
      ) : (
        <div className="brain-inspector">
          {memoryStats && (
            <dl className="brain-stats">
              <div><dt>Facts</dt><dd>{memoryStats.usermemory_count || 0}</dd></div>
              <div><dt>Skills</dt><dd>{memoryStats.skill_count || 0}</dd></div>
              <div><dt>Reflections</dt><dd>{memoryStats.reflection_count || 0}</dd></div>
              <div><dt>Artifacts</dt><dd>{memoryStats.artifact_count || 0}</dd></div>
            </dl>
          )}

          <div className="brain-nav-tabs" role="tablist" aria-label="Memory sections">
            {[
              ['user_memory', `Facts (${userMemories.length})`],
              ['adapters', `Adapters (${adapters.length})`],
              ['skills', `Skills (${skills.length})`],
              ['reflections', `Reflections (${reflections.length})`],
            ].map(([key, label]) => (
              <button
                key={key}
                type="button"
                role="tab"
                aria-selected={brainSection === key}
                className={`brain-tab-btn ${brainSection === key ? 'active' : ''}`}
                onClick={() => setBrainSection(key)}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="brain-tab-content">
            {brainSection === 'user_memory' && (
              <>
                <form className="add-memory-form" onSubmit={handleAddMemory}>
                  <input type="text" className="brain-input" placeholder="Key, e.g. preferred_airline" value={newKey} onChange={e => setNewKey(e.target.value)} aria-label="Fact key" />
                  <input type="text" className="brain-input" placeholder="Value, e.g. IndiGo" value={newValue} onChange={e => setNewValue(e.target.value)} aria-label="Fact value" />
                  <div className="add-memory-row">
                    <select className="brain-select" value={newCategory} onChange={e => setNewCategory(e.target.value)} aria-label="Fact category">
                      <option value="preference">Preference</option>
                      <option value="constraint">Constraint</option>
                      <option value="profile">Profile</option>
                    </select>
                    <button type="submit" className="btn btn-primary" disabled={!newKey.trim() || !newValue.trim()}>
                      <Plus size={13} /> Save fact
                    </button>
                  </div>
                </form>
                <div className="brain-items-list">
                  {userMemories.length === 0 ? (
                    <p className="brain-empty">No saved facts yet. Comex learns them as you work, or you can add one above.</p>
                  ) : userMemories.map(m => (
                    <div key={m.id || m.key} className="brain-card">
                      <div className="card-header">
                        <span className="card-badge">{m.category}</span>
                        <span className="card-key">{m.key}</span>
                        <button type="button" className="icon-btn icon-btn-sm" onClick={() => handleDeleteMemory(m.key)} aria-label={`Delete ${m.key}`}>
                          <Trash2 size={13} />
                        </button>
                      </div>
                      <div className="card-value">{m.value}</div>
                      {m.updated_at && <div className="card-time">{new Date(m.updated_at).toLocaleString()}</div>}
                    </div>
                  ))}
                </div>
              </>
            )}

            {brainSection === 'skills' && (
              <>
                <div className="brain-toolbar">
                  <button type="button" className="btn btn-secondary" onClick={handleConsolidateSkills} title="Merge duplicate and near-identical skills">
                    Merge duplicates
                  </button>
                </div>
                <div className="brain-items-list">
                  {skills.length === 0 ? (
                    <p className="brain-empty">No skills yet. Comex distills reusable skills after it handles something new.</p>
                  ) : skills.map(s => (
                    <div key={s.id} className="brain-card">
                      <div className="card-header">
                        <span className="card-badge">{s.domain}</span>
                        <span className="card-key">{s.name}</span>
                        <button type="button" className="icon-btn icon-btn-sm" onClick={() => handleDeleteSkill(s.id)} aria-label={`Delete ${s.name}`}>
                          <Trash2 size={13} />
                        </button>
                      </div>
                      <p className="card-text">{s.description}</p>
                      {s.code_snippet && <pre className="card-code"><code>{s.code_snippet}</code></pre>}
                      <div className="card-metrics">
                        <span>Used {s.usage_count || 0} times</span>
                        <span>{s.success_count || 0} successes</span>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}

            {brainSection === 'reflections' && (
              <div className="brain-items-list">
                {reflections.length === 0 ? (
                  <p className="brain-empty">No reflections yet. Comex reviews each task in the background after it finishes.</p>
                ) : reflections.map(r => (
                  <div key={r.id} className="brain-card">
                    <div className="card-header">
                      <span className={`card-badge ${r.is_novel ? 'card-badge-accent' : ''}`}>{r.is_novel ? 'New kind of task' : 'Routine'}</span>
                      <span className="card-time">{r.created_at}</span>
                    </div>
                    {r.critique && <p className="card-text">{r.critique}</p>}
                    {r.lesson_learned && (
                      <p className="card-lesson"><strong>Lesson:</strong> {r.lesson_learned}</p>
                    )}
                  </div>
                ))}
              </div>
            )}

            {brainSection === 'adapters' && (
              <div className="brain-items-list">
                {adapters.length === 0 ? (
                  <p className="brain-empty">No adapters yet. Routine actions you repeat are compiled into fast adapters automatically.</p>
                ) : adapters.map(a => (
                  <div key={a.id} className="brain-card">
                    <div className="card-header">
                      <span className={`card-badge ${a.status === 'active' ? 'card-badge-accent' : ''}`}>{a.domain}</span>
                      <span className="card-key">{a.name}</span>
                      <button type="button" className="icon-btn icon-btn-sm" onClick={() => handleDeleteAdapter(a.id)} aria-label={`Delete ${a.name}`}>
                        <Trash2 size={13} />
                      </button>
                    </div>
                    <p className="card-text">{a.action_intent}</p>
                    {a.code_template && <pre className="card-code"><code>{a.code_template}</code></pre>}
                    <div className="card-metrics">
                      <span>{a.status}</span>
                      <span>{a.success_count || 0} successes</span>
                      <span>{a.failure_count || 0} failures</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </aside>
  );
}

function ElapsedTime({ since }) {
  const now = useNow(Boolean(since));
  if (!since) return null;
  return <span className="elapsed">{formatDuration(now - since)}</span>;
}
