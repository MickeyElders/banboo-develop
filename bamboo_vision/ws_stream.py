import asyncio
import logging
import threading
from typing import Set

import websockets


class FrameBroadcaster:
    """Minimal WebSocket broadcaster for JPEG frames."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.loop = asyncio.new_event_loop()
        self._server = None

    def start(self):
        t = threading.Thread(target=self._run_loop, daemon=True, name="ws-frame-broadcaster")
        t.start()
        logging.info("WebSocket frame broadcaster listening on ws://%s:%d", self.host, self.port)

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self._server = websockets.serve(self._handler, self.host, self.port, max_queue=1)
        self.loop.run_until_complete(self._server)
        self.loop.run_forever()

    async def _handler(self, websocket, path):
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)

    async def _broadcast(self, data: bytes):
        stale = []
        for ws in list(self.clients):
            try:
                await ws.send(data)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.clients.discard(ws)

    def push(self, data: bytes):
        if not self.clients:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(data), self.loop)
