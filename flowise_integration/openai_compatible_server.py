"""OpenAI-compatible HTTP server that proxies requests to Flowise.

Endpoints implemented (minimal):
- POST /v1/chat/completions  (accepts OpenAI ChatCompletion style requests)
- POST /v1/completions       (accepts OpenAI Completion-style requests)

The server uses `FlowiseClient` from `flowise_client.py` and the
`openai_adapter` conversions. Configure the Flowise connection with env vars:
- FLOWISE_BASE_URL (e.g. https://demo.flow.legaleagle-ai.com/api/v1)
- FLOWISE_FLOW_ID (flow id)
- FLOWISE_API_KEY (api key)  -- flowise_client will also read a local .env

This is intentionally minimal and synchronous. For production use add
timeouts, retries, authentication, and streaming support as needed.
"""
from __future__ import annotations

import os
import time
import json
from typing import Any, Dict, List, Optional, Iterable, Callable
from flask import Flask, request, jsonify, Response

from flowise_client import FlowiseClient
from openai_adapter import openai_to_flowise, flowise_to_openai

import logging
from pathlib import Path

app = Flask(__name__)

# Create a global FlowiseClient instance from environment variables.
FLOWISE_BASE_URL = os.getenv("FLOWISE_BASE_URL")
FLOWISE_API_KEY = os.getenv("FLOWISE_API_KEY")

_flowise_clients: Dict[str, FlowiseClient] = {}  # Cache clients by flow_id


def _load_repo_root_env():
    """Also try to load a `.env` file from the repository root if present.

    This is helpful when running the server from the project root rather
    than the integration folder.
    """
    try:
        root_env = Path(__file__).parents[1] / ".env"
        if root_env.exists():
            with root_env.open("r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and os.getenv(k) is None:
                        os.environ[k] = v
    except Exception:
        pass


def get_client(flow_id: str) -> FlowiseClient:
    """Get or create a FlowiseClient for the specified flow_id.
    
    Args:
        flow_id: The Flowise flow ID to use
        
    Returns:
        A FlowiseClient instance configured for the flow
    """
    global _flowise_clients
    
    # Return cached client if available
    if flow_id in _flowise_clients:
        return _flowise_clients[flow_id]

    # Allow loading repo-root .env as a fallback
    _load_repo_root_env()

    # refresh vars after loading root env
    base = os.getenv("FLOWISE_BASE_URL", FLOWISE_BASE_URL)
    key = os.getenv("FLOWISE_API_KEY", FLOWISE_API_KEY)

    if not base:
        raise RuntimeError("FLOWISE_BASE_URL must be set in environment")

    # FlowiseClient will read FLOWISE_API_KEY from env/.env if api_key is None
    client = FlowiseClient(base_url=base, flow_id=flow_id, api_key=key)
    _flowise_clients[flow_id] = client
    return client



@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/openai/deployments/<deployment_name>/chat/completions", methods=["POST"])
def chat_completions(deployment_name=None):
    """Handle OpenAI ChatCompletion-style requests.

    Supports both standard OpenAI and Azure OpenAI URL formats.
    If request body includes "stream": true, returns Server-Sent Events (SSE)
    with OpenAI-compatible incremental chunks (chat.completion.chunk).
    """
    body: Dict[str, Any] = request.get_json(force=True)
    stream_requested = bool(body.get("stream"))

    # Extract flow_id from model field (or deployment_name for Azure)
    flow_id = deployment_name or body.get("model")
    if not flow_id:
        return jsonify({"error": "model field is required (used as Flowise flow_id)"}), 400

    # Extract messages or prompt
    messages: List[Dict[str, Any]] = body.get("messages") or []

    if messages:
        last = messages[-1]
        question = last.get("content") if isinstance(last, dict) else str(last)
        history = messages[:-1]
    else:
        prompt = body.get("prompt") or body.get("input") or body.get("question")
        if isinstance(prompt, list):
            question = "\n".join(str(p) for p in prompt)
        else:
            question = str(prompt or "")
        history = []

    try:
        client = get_client(flow_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    try:
        resp = client.predict(question=question, history=history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not stream_requested:
        return jsonify(resp)

    # Streaming mode: break assistant content into chunks and emit SSE frames
    # OpenAI streaming spec: each line begins with 'data: ' + JSON, terminated by blank line.
    # Final line: data: [DONE]
    assistant_msg = ""
    try:
        # Chat mode response shape ensures choices[0].message.content
        assistant_msg = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        assistant_msg = ""

    model_name = resp.get("model", flow_id)
    base_id = resp.get("id", f"flowise-{int(time.time()*1000)}")

    def sse_chat_generator() -> Iterable[str]:
        created = int(time.time())
        # First chunk: role only (optional content if desired)
        first_chunk = {
            "id": base_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

        # Tokenize content naïvely by splitting on whitespace while preserving it.
        # This keeps things simple; adjust if token granularity needed.
        if assistant_msg:
            # Reconstruct with spaces as separate chunks to preserve formatting
            import re
            pieces = re.findall(r"\S+|\s+", assistant_msg)
            buffer = []
            max_chunk_chars = 50  # group tokens to reduce SSE overhead
            for piece in pieces:
                buffer.append(piece)
                if sum(len(p) for p in buffer) >= max_chunk_chars:
                    content_chunk = "".join(buffer)
                    buffer.clear()
                    chunk = {
                        "id": base_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content_chunk},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            # Flush remaining buffer
            if buffer:
                content_chunk = "".join(buffer)
                chunk = {
                    "id": base_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content_chunk},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # Final chunk with finish_reason
        final_chunk = {
            "id": base_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        sse_chat_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/v1/completions", methods=["POST"])
@app.route("/openai/deployments/<deployment_name>/completions", methods=["POST"])
def completions(deployment_name=None):
    """Handle legacy OpenAI completion-style requests.

    If request body includes "stream": true, emits SSE streaming chunks similar
    to OpenAI completions streaming semantics.
    """
    body: Dict[str, Any] = request.get_json(force=True)
    stream_requested = bool(body.get("stream"))

    flow_id = deployment_name or body.get("model")
    if not flow_id:
        return jsonify({"error": "model field is required (used as Flowise flow_id)"}), 400

    prompt = body.get("prompt") or body.get("input") or body.get("question")
    if isinstance(prompt, list):
        prompt = "\n".join(str(p) for p in prompt)
    prompt = str(prompt or "")

    try:
        client = get_client(flow_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    try:
        resp = client.predict(question=prompt, history=[])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not stream_requested:
        return jsonify(resp)

    # Gather completion text
    completion_text = ""
    try:
        completion_text = resp.get("choices", [{}])[0].get("text", "") or resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        completion_text = ""

    model_name = resp.get("model", flow_id)
    base_id = resp.get("id", f"flowise-{int(time.time()*1000)}")

    def sse_completion_generator() -> Iterable[str]:
        created = int(time.time())
        # Stream text in grouped chunks
        import re
        pieces = re.findall(r"\S+|\s+", completion_text)
        buffer = []
        max_chunk_chars = 50
        for piece in pieces:
            buffer.append(piece)
            if sum(len(p) for p in buffer) >= max_chunk_chars:
                content_chunk = "".join(buffer)
                buffer.clear()
                chunk = {
                    "id": base_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "text": content_chunk,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        if buffer:
            content_chunk = "".join(buffer)
            chunk = {
                "id": base_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "text": content_chunk,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        final_chunk = {
            "id": base_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "text": "",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        sse_completion_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    # Enable debug logging to see requests
    logging.basicConfig(level=logging.DEBUG)
    
    # Default host/port; change via env PORT
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
