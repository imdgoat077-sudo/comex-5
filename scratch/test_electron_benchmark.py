import asyncio
import http.server
import json
import os
import socketserver
import threading
import urllib.request
import websockets
import time

html_dir = os.path.abspath('fixtures/miniwob-plusplus/miniwob/html')

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=html_dir, **kwargs)
    def log_message(self, format, *args):
        pass

socketserver.TCPServer.allow_reuse_address = True
httpd = socketserver.TCPServer(('127.0.0.1', 8890), QuietHandler)
t = threading.Thread(target=httpd.serve_forever, daemon=True)
t.start()

async def run_single():
    # 1. Connect to Electron tab
    req = urllib.request.urlopen('http://127.0.0.1:9222/json/list', timeout=3)
    targets = json.loads(req.read())
    target = next(t for t in targets if t.get('type') == 'page' and 'cursor_overlay' not in t.get('url', '') and 'about:blank' not in t.get('url', ''))
    
    async with websockets.connect(target['webSocketDebuggerUrl']) as ws:
        msg_id = 1
        async def send(method, params=None):
            nonlocal msg_id
            msg_id += 1
            await ws.send(json.dumps({'id': msg_id, 'method': method, 'params': params or {}}))
            while True:
                res = json.loads(await ws.recv())
                if res.get('id') == msg_id:
                    return res.get('result', {})

        await send('Page.navigate', {'url': 'http://127.0.0.1:8890/miniwob/enter-text.html'})
        await asyncio.sleep(0.6)
        await send('Runtime.evaluate', {'expression': 'core.startEpisodeReal()'})
        await asyncio.sleep(0.2)
        
        # 2. Get Goal
        res = await send('Runtime.evaluate', {'expression': 'core.getUtterance()', 'returnByValue': True})
        goal = res.get('result', {}).get('value')
        
        # 3. Get Area DOM elements
        res = await send('Runtime.evaluate', {'expression': '''
            (() => {
                const area = document.querySelector('#area') || document.body;
                const elements = area.querySelectorAll('button, input, select, textarea, a, [role="button"]');
                return Array.from(elements).map((el, i) => {
                    const r = el.getBoundingClientRect();
                    return {
                        tag: el.tagName.toLowerCase(),
                        type: el.type || '',
                        id: el.id || '',
                        text: (el.innerText || el.value || el.placeholder || '').trim(),
                        selector: el.id ? ('#' + el.id) : (el.tagName.toLowerCase() + ':nth-of-type(' + (i+1) + ')'),
                        center: { x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2) }
                    };
                });
            })()
        ''', 'returnByValue': True})
        elements = res.get('result', {}).get('value')
        
        print('GOAL:', goal)
        print('ELEMENTS:', elements)
        
        # 4. Ask Gemini via 9Router
        prompt = f'''You are a browser automation agent running in Comex Browser.
Goal: {goal}
Available Interactive Elements on Page:
{json.dumps(elements, indent=2)}

You can execute a sequence of actions. Format your response strictly as a JSON array of actions:
[
  {{"action": "fill", "selector": "#id", "text": "string"}},
  {{"action": "click", "selector": "#id"}}
]

Output ONLY valid JSON, nothing else.'''

        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        api_key = os.getenv('ROUTER_API_KEY', os.getenv('OPENAI_API_KEY', ''))
        llm = ChatOpenAI(
            model='ag/gemini-3.8-flash',
            base_url='http://localhost:20128/v1',
            api_key=api_key,
            temperature=0.0
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw_text = resp.content.strip()
        if raw_text.startswith('```'):
            raw_text = raw_text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        print('LLM Response:', raw_text)
        actions = json.loads(raw_text)
        
        # 5. Execute actions in Electron
        for act in actions:
            action_type = act.get('action')
            sel = act.get('selector')
            print(f'Executing {action_type} on {sel}...')
            
            box_res = await send('Runtime.evaluate', {'expression': f'''
                (() => {{
                    const el = document.querySelector('{sel}');
                    if (!el) return null;
                    const r = el.getBoundingClientRect();
                    return {{ x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2) }};
                }})()
            ''', 'returnByValue': True})
            center = box_res.get('result', {}).get('value')
            if center:
                try:
                    cursor_req = urllib.request.Request(
                        'http://127.0.0.1:8000/agent/cursor/notify',
                        data=json.dumps({'x': center['x'], 'y': center['y'], 'action': 'click', 'tabId': 'active_tab'}).encode(),
                        headers={'Content-Type': 'application/json'}
                    )
                    urllib.request.urlopen(cursor_req, timeout=1)
                except Exception:
                    pass
                    
            if action_type == 'fill':
                text_to_fill = act.get('text', '')
                await send('Runtime.evaluate', {'expression': f'''
                    (() => {{
                        const el = document.querySelector('{sel}');
                        if (el) {{
                            el.focus();
                            el.value = {json.dumps(text_to_fill)};
                            el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        }}
                    }})()
                '''})
            elif action_type == 'click':
                await send('Runtime.evaluate', {'expression': f'''
                    (() => {{
                        const el = document.querySelector('{sel}');
                        if (el) el.click();
                    }})()
                '''})
            await asyncio.sleep(0.3)
            
        # 6. Read referee reward
        res = await send('Runtime.evaluate', {'expression': '''
            ({ done: window.WOB_DONE_GLOBAL, reward: window.WOB_RAW_REWARD_GLOBAL, reason: window.WOB_REWARD_REASON })
        ''', 'returnByValue': True})
        print('BENCHMARK EVALUATION IN ELECTRON:', res.get('result', {}).get('value'))

if __name__ == '__main__':
    try:
        asyncio.run(run_single())
    finally:
        httpd.shutdown()
