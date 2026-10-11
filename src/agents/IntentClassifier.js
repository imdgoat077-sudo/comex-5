/**
 * Semantic Intent Classifier
 * 
 * Uses the existing AI/LLM infrastructure to classify user prompts into two primary intents:
 * 1. RESEARCH - Information gathering, comparisons, recommendations, links
 * 2. VISIBLE_BROWSER_TASK - Explicit request to open/view/interact with pages
 * 
 * This classifier replaces the brittle keyword-based detection with semantic understanding.
 */

const ROUTER_API_KEY = import.meta.env.VITE_ROUTER_API_KEY || '';
const OPENAI_BASE_URL = import.meta.env.VITE_OPENAI_BASE_URL || 'http://localhost:20128/v1';
const MODEL_NAME = import.meta.env.VITE_MODEL_NAME || 'ag/gemini-3.8-flash';

export async function classifyIntent(prompt) {
  const systemPrompt = `
    You are an AI intent classifier for a web browser that can answer questions and open/view pages.
    
    CLASSIFY the user's COMPLETE request into ONE of two intents:
    
    === RESEARCH ===
    The user wants information, results, recommendations, comparisons, or links.
    Stay on the current page or use background tabs.
    Examples:
      - "Search YouTube for a good LLD video"
      - "Find me the best LLD video and give me the link"
      - "Search Amazon for a 27-inch monitor under ₹20,000"
      - "Find the latest trailer for me"
      - "What's the latest news about NVIDIA?"
      - "Find me three good monitors under ₹20,000 and compare them"
    
    === VISIBLE_BROWSER_TASK ===
    The user explicitly wants to open, view, or interact with a page.
    Create/reuse a visible browser tab, switch workspace, and run tasks in that visible tab.
    Examples:
      - "Open the best monitor you found"
      - "Open this YouTube video for me"
      - "Go to YouTube and search for lo-fi music"
      - "Find the latest trailer and open it"
      - "Can you pull up YouTube and find the latest trailer?"
      - "Go to NVIDIA's website and show me their latest GPUs"
      - "Find me three good monitors under ₹20,000 and open the best one"
      - "Open the latest trailer"
      - "Go to YouTube"
      - "Pull up Amazon and show me something"
    
    === IMPORTANT RULES ===
    1. USE SEMANTIC UNDERSTANDING of the COMPLETE request, NOT just keywords
    2. MIXED requests like "Find best monitor and open it" → VISIBLE_BROWSER_TASK
    3. When ambiguous, classify as RESEARCH
    4. DO NOT fabricate or assume content that isn't in the request
    5. Focus on the user's actual goal/need
    
    Return ONLY this JSON:
    {
      "intent": "research" | "visible_browser_task",
      "confidence": 0.0-1.0,
      "target": string | null
    }
    
    target should be:
    - null for pure research requests
    - the URL or destination for visible-browser tasks
    - null if unsure
  `;

  // Use the existing 9Router proxy
  const response = await fetch(`${OPENAI_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${ROUTER_API_KEY}`
    },
    body: JSON.stringify({
      model: MODEL_NAME,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: prompt }
      ],
      temperature: 0.1,
      max_tokens: 1024,
      response_format: { type: 'json_object' }
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error('Intent classifier error:', errorText);
    throw new Error(`AI classifier failed: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();
  const text = data.choices?.[0]?.message?.content || '{}';

  try {
    const result = JSON.parse(text);
    if (!['research', 'visible_browser_task'].includes(result.intent)) {
      console.warn('Classifier returned invalid intent:', result.intent);
      result.intent = 'research';
    }
    return result;
  } catch (error) {
    console.error('Failed to parse classifier response:', text, error);
    return { intent: 'research', confidence: 0.5, target: null };
  }
}
