"""
Instant Visual Cursor Demo:
Sends a series of coordinates across the screen to demonstrate the 24x24 SVG
fake cursor gliding smoothly and rippling on click.
"""
import asyncio
import json
import urllib.request
import websockets

CDP_URL = "http://127.0.0.1:9222"

async def demo():
    print("Finding Electron React UI target...")
    req = urllib.request.urlopen(f"{CDP_URL}/json/list", timeout=2)
    targets = json.loads(req.read())
    ui_ws = None
    for t in targets:
        url = t.get("url", "")
        if t.get("type") == "page" and ("localhost:5173" in url or "index.html" in url):
            ui_ws = t.get("webSocketDebuggerUrl")
            break

    if not ui_ws:
        print("Error: Could not find Electron main window on CDP port 9222.")
        print("Please ensure Electron is running with remote debugging port 9222.")
        return

    print(f"Connected to main window: {ui_ws}")

    async with websockets.connect(ui_ws) as ws:
        points = [
            (250, 200, "Search bar click"),
            (500, 350, "First result click"),
            (800, 450, "Details button click"),
            (350, 600, "Back button click"),
        ]

        for x, y, label in points:
            print(f"-> Moving fake cursor to ({x}, {y}): {label}")
            expr = f"window.nexusNative && window.nexusNative.cursor && window.nexusNative.cursor.animate({{ x: {x}, y: {y}, action: 'click' }});"
            msg = {
                "id": 1,
                "method": "Runtime.evaluate",
                "params": {"expression": expr}
            }
            await ws.send(json.dumps(msg))
            await ws.recv()
            await asyncio.sleep(1.0)

    print("\n✓ Demo complete! Did you see the fake cursor glide and ripple?")

if __name__ == "__main__":
    asyncio.run(demo())
