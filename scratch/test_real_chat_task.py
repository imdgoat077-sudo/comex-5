import urllib.request
import json
import time

url = "http://127.0.0.1:8000/agent/run/stream"
payload = {
    "task": "Go to https://news.ycombinator.com and click the first story.",
    "taskId": f"task-real-{int(time.time())}",
    "tabId": "tab-1",
    "url": "about:blank",
    "intent": "visible_browser_task"
}

print(f"Sending real agent chat task: {payload['task']}")
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode('utf-8'),
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req, timeout=120) as resp:
    for line in resp:
        line = line.decode('utf-8').strip()
        if line.startswith("data:"):
            try:
                data = json.loads(line[5:].strip())
                event = data.get("event")
                payload_data = data.get("data", {})
                if event == "step":
                    print(f"[SSE step] tool={payload_data.get('tool')} input={str(payload_data.get('input'))[:120]}")
                elif event == "step_done":
                    print(f"[SSE step_done] success={payload_data.get('success')} output={str(payload_data.get('output'))[:120]}")
                elif event == "done":
                    print(f"[SSE done] result={str(payload_data.get('result'))[:150]}")
                elif event == "thinking":
                    print(f"[SSE thinking] {payload_data.get('thought')}")
            except Exception:
                pass
