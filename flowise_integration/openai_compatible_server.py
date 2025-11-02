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
from typing import Any, Dict, List, Optional
from flask import Flask, request, jsonify

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
    """Handle OpenAI ChatCompletion-style requests and return an OpenAI-compatible response.
    
    Supports both standard OpenAI and Azure OpenAI URL formats.
    The 'model' field in the request body is used as the Flowise flow ID.
    """
    body: Dict[str, Any] = request.get_json(force=True)

    # Extract flow_id from model field (or deployment_name for Azure)
    flow_id = deployment_name or body.get("model")
    if not flow_id:
        return jsonify({"error": "model field is required (used as Flowise flow_id)"}), 400

    # Extract messages or prompt
    messages: List[Dict[str, Any]] = body.get("messages") or []

    # If messages present, pick the last user message as question and keep preceding messages as history
    if messages:
        # Find last message index (use last element)
        last = messages[-1]
        question = last.get("content") if isinstance(last, dict) else str(last)
        history = messages[:-1]
    else:
        # Fallback to prompt or input
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

    # Convert history into OpenAI messages list expected by FlowiseClient
    # FlowiseClient.predict expects a history list in OpenAI format
    try:
        resp = client.predict(question=question, history=history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # resp should already be OpenAI-compatible because flowise_to_openai returns that shape
    return jsonify(resp)


@app.route("/v1/completions", methods=["POST"])
@app.route("/openai/deployments/<deployment_name>/completions", methods=["POST"])
def completions(deployment_name=None):
    """Handle legacy OpenAI completion-style requests.
    
    Supports both standard OpenAI and Azure OpenAI URL formats.
    The 'model' field in the request body is used as the Flowise flow ID.
    """
    body: Dict[str, Any] = request.get_json(force=True)
    
    # Extract flow_id from model field (or deployment_name for Azure)
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
        # For completion endpoint, we pass prompt as a single user message
        resp = client.predict(question=prompt, history=[])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(resp)


if __name__ == "__main__":
    # Enable debug logging to see requests
    logging.basicConfig(level=logging.DEBUG)
    
    # Default host/port; change via env PORT
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
