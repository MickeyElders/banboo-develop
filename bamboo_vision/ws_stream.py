import asyncio
import logging
import threading
from typing import Set

import websockets


class WebSocketStreamer:
    """Threaded WebSocket binary streamer with its own event loop."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server = None
        self._stopping = threading.Event()
        self._ready = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ws-streamer")
        self._thread.start()
        logging.info("WebSocket streamer listening on ws://%s:%d", self.host, self.port)

    def stop(self):
        self._stopping.set()
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run_loop(self):
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        server_coro = websockets.serve(self._handler, self.host, self.port, max_queue=1)
        try:
            self._server = self.loop.run_until_complete(server_coro)
            self._ready.set()
            self.loop.run_forever()
        except Exception as e:
            logging.error("WebSocket streamer loop error: %s", e)
        finally:
            try:
                if self._server:
                    self._server.close()
                    self.loop.run_until_complete(self._server.wait_closed())
            finally:
                self.loop.close()

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
        if not self.clients or not self.loop or not self.loop.is_running() or not self._ready.is_set():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(data), self.loop)
        except Exception as e:
            logging.debug("WebSocket push failed: %s", e)
