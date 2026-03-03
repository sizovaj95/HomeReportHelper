from __future__ import annotations

import asyncio
import concurrent.futures
import json
import queue
import threading
from pathlib import Path
from typing import Any

from agent import config


class MCPStdioClient:
    """Minimal synchronous MCP client over stdio.

    This client keeps one MCP stdio session open for the whole process/run.
    It exposes synchronous calls while internally using a dedicated async worker.
    """

    def __init__(
        self,
        command: str = "python3",
        args: list[str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self.command = command
        self.args = args or ["-m", "mcp_server.home_report_server"]
        self.cwd = cwd or str(config.BASE_DIR)
        self._thread: threading.Thread | None = None
        self._request_queue: queue.Queue | None = None
        self._ready = threading.Event()
        self._start_error: Exception | None = None
        self._closed = False
        self._stop_sentinel = object()

    def __enter__(self) -> "MCPStdioClient":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._closed = False
        self._start_error = None
        self._request_queue = queue.Queue()
        self._ready.clear()
        self._thread = threading.Thread(target=self._run_worker, name="mcp-stdio-worker", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=30)
        if self._start_error is not None:
            err = self._start_error
            self._start_error = None
            self.close()
            raise RuntimeError(f"Failed to start MCP stdio session: {err}") from err
        if self._thread is None or not self._thread.is_alive():
            self.close()
            raise RuntimeError("MCP stdio worker failed to start.")

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        try:
            if self._request_queue is not None:
                self._request_queue.put(self._stop_sentinel)
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=10)
        finally:
            self._thread = None
            self._request_queue = None

    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        self.open()
        if self._request_queue is None:
            raise RuntimeError("MCP request queue is not available.")

        response_future: concurrent.futures.Future = concurrent.futures.Future()
        self._request_queue.put((tool_name, arguments or {}, response_future))
        return response_future.result()

    def _run_worker(self) -> None:
        try:
            asyncio.run(self._worker_main())
        except Exception as exc:  # pragma: no cover
            self._start_error = exc
            self._ready.set()

    async def _worker_main(self) -> None:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "MCP client requires the 'mcp' package. Install dependencies first."
            ) from exc

        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            cwd=str(Path(self.cwd).resolve()),
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._ready.set()
                await self._serve_requests(session)

    async def _serve_requests(self, session) -> None:
        if self._request_queue is None:
            return

        while True:
            item = await asyncio.to_thread(self._request_queue.get)
            if item is self._stop_sentinel:
                return

            tool_name, arguments, response_future = item
            try:
                result = await session.call_tool(tool_name, arguments)
                decoded = self._decode_tool_result(result)
                response_future.set_result(decoded)
            except Exception as exc:  # pragma: no cover
                response_future.set_exception(exc)

    def _decode_tool_result(self, result: Any) -> Any:
        # 1) structured content variants
        for attr in ("structured_content", "structuredContent"):
            if hasattr(result, attr):
                value = getattr(result, attr)
                if value is not None:
                    return value
            if isinstance(result, dict) and attr in result and result[attr] is not None:
                return result[attr]

        # 2) content list with text payloads
        content = getattr(result, "content", None)
        if content is None and isinstance(result, dict):
            content = result.get("content")

        if isinstance(content, list) and content:
            texts: list[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if text is None and isinstance(item, dict):
                    text = item.get("text")
                if isinstance(text, str):
                    texts.append(text)

            if len(texts) == 1:
                return self._maybe_json(texts[0])
            if texts:
                return [self._maybe_json(t) for t in texts]

        # 3) model dump fallback
        if hasattr(result, "model_dump"):
            try:
                return result.model_dump()
            except Exception:
                pass

        return result

    @staticmethod
    def _maybe_json(value: str) -> Any:
        value = value.strip()
        if not value:
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
