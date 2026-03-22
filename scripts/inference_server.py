#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from engine.mock_provider import RandomMockProvider
from engine.pytorch_provider import PyTorchModelProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_provider():
    provider_name = os.environ.get("ENGINE_PROVIDER", "pytorch").strip().lower()
    if provider_name == "pytorch":
        temperature = float(os.environ.get("ENGINE_TEMPERATURE", "0.0"))
        top_p = float(os.environ.get("ENGINE_TOP_P", "1.0"))
        return provider_name, PyTorchModelProvider(temperature=temperature, top_p=top_p)
    if provider_name == "mock":
        return provider_name, RandomMockProvider()
    raise ValueError(f"Unknown ENGINE_PROVIDER='{provider_name}'. Supported values: mock, pytorch")


PROVIDER_NAME, PROVIDER = create_provider()


class InferenceHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "provider": PROVIDER_NAME})
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self):
        if self.path != "/predict":
            self._send_json(404, {"error": "not_found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            moves = str(payload.get("moves", "")).strip()

            best_move = PROVIDER.get_best_move(moves)
            self._send_json(
                200,
                {
                    "best_move": best_move,
                    "provider": PROVIDER_NAME,
                    "input_moves": moves,
                },
            )
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid_json"})
        except Exception as exc:
            logger.exception("Inference error")
            self._send_json(500, {"error": "inference_failed", "detail": str(exc)})

    def log_message(self, fmt: str, *args):
        logger.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    logger.info("Starting inference server on %s:%s with provider=%s", host, port, PROVIDER_NAME)
    server = HTTPServer((host, port), InferenceHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
