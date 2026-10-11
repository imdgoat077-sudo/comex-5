/**
 * Multi-Agent Orchestrator Client
 *
 * Streams progress from POST /agent/orchestrate/stream (planning, run_started, per-agent
 * task_* events, synthesis_started, approval_required) and resolves with the orchestrator's
 * final answer, which compares the findings of every agent. A follow-up action (e.g. add to
 * cart) only runs after respondToSwarmApproval(runId, true).
 */

const AGENT_SERVICE_URL = 'http://127.0.0.1:8000';

export async function runSwarm(userPrompt, { onTaskUpdate, signal } = {}) {
  const response = await fetch(`${AGENT_SERVICE_URL}/agent/orchestrate/stream`, {
    method: 'POST',
    signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_request: userPrompt })
  });

  if (!response.ok || !response.body) {
    throw new Error(`Orchestrator service responded with status ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalPayload = null;

  const processBuffer = () => {
    const events = buffer.split('\n\n');
    buffer = events.pop() ?? '';

    for (const rawEvent of events) {
      const lines = rawEvent.split('\n');
      let eventType = 'message';
      let dataLine = '';
      for (const line of lines) {
        if (line.startsWith('event:')) eventType = line.slice(6).trim();
        if (line.startsWith('data:')) dataLine = line.slice(5).trim();
      }
      if (!dataLine) continue;

      let data;
      try {
        data = JSON.parse(dataLine);
      } catch {
        continue;
      }

      if (eventType === 'done') {
        finalPayload = data;
      } else if (onTaskUpdate) {
        onTaskUpdate({ eventType, ...data });
      }
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    processBuffer();
  }
  processBuffer();

  if (!finalPayload) {
    throw new Error('Orchestrator stream ended without a result.');
  }
  if (finalPayload.error) {
    throw new Error(finalPayload.error);
  }

  return {
    finalAnswer: finalPayload.final_answer || finalPayload.summary || 'The agents finished without a result.',
    runId: finalPayload.run_id,
    success: finalPayload.success,
    sources: finalPayload.sources || [],
    followUp: finalPayload.follow_up || null,
  };
}

export async function respondToSwarmApproval(runId, approved) {
  const response = await fetch(`${AGENT_SERVICE_URL}/agent/orchestrate/approve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_id: runId, approved }),
  });
  return response.json();
}

export async function stopSwarmRun(runId) {
  if (!runId) return;
  await fetch(`${AGENT_SERVICE_URL}/agent/orchestrate/stop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_id: runId }),
  });
}
