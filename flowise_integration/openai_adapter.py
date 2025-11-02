"""OpenAI <-> Flowise adapter

Provides conversion helpers to translate an OpenAI-style completion request
into a Flowise prediction request, and to translate a Flowise prediction
response back into an OpenAI-style completion response.

This is intentionally conservative: it maps common fields (prompt -> input,
temperature, max_tokens, top_p, n) and returns a minimal, valid OpenAI
completion response shape built from the Flowise outputs.
"""
from typing import Any, Dict, List, Optional
import time
import json
from pathlib import Path


def openai_to_flowise(openai_req: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an OpenAI completion request dict to a Flowise prediction request.

    Expected openai_req keys (commonly present):
      - model (ignored by Flowise adapter)
      - prompt (str or list of str)
      - max_tokens (int)
      - temperature (float)
      - top_p (float)
      - n (int) number of completions
      - stop (str or list)

    Returns a dict ready to be sent to Flowise's prediction endpoint.
    The exact Flowise request schema varies by Flowise setup; this mapper
    targets a generic 'inputs' + 'parameters' shape commonly used by
    Flowise-style model servers.
    """
    # Normalize prompt to a single string input. If prompt is a list, join with "\n"
    prompt = openai_req.get("prompt")
    if prompt is None:
        prompt = ""
    elif isinstance(prompt, list):
        prompt = "\n".join(str(p) for p in prompt)
    else:
        prompt = str(prompt)

    # Map basic numeric knobs
    temperature = openai_req.get("temperature")
    top_p = openai_req.get("top_p")
    max_tokens = openai_req.get("max_tokens")
    n = openai_req.get("n", 1)
    stop = openai_req.get("stop")

    # Assemble Flowise-style request. Many Flowise setups accept an "input"
    # or "inputs" field. Use `inputs` as a dict with 'text' by default.
    flowise_req: Dict[str, Any] = {
        "inputs": {"text": prompt},
        "parameters": {},
    }

    params: Dict[str, Any] = flowise_req["parameters"]
    if temperature is not None:
        params["temperature"] = float(temperature)
    if top_p is not None:
        params["top_p"] = float(top_p)
    if max_tokens is not None:
        params["max_tokens"] = int(max_tokens)
    # Flowise may use `num_return_sequences` or `n`. Provide both aliases.
    params["n"] = int(n)
    params["num_return_sequences"] = int(n)
    if stop is not None:
        params["stop"] = stop

    # Attach model value if provided (Flowise node or model id). This is optional
    model = openai_req.get("model")
    if model:
        flowise_req["model"] = model

    return flowise_req


def _build_choice(text: str, index: int = 0, finish_reason: Optional[str] = "stop") -> Dict[str, Any]:
    """Helper to build a single OpenAI-style choice entry."""
    return {
        "text": text,
        "index": index,
        # We do not provide logprobs here
        "logprobs": None,
        "finish_reason": finish_reason,
    }


def flowise_to_openai(flowise_resp: Dict[str, Any], openai_req: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convert a Flowise prediction response into an OpenAI completion response.

    The adapter expects flowise_resp to contain either:
      - a top-level `outputs` list of strings OR
      - `outputs` list of dicts with a `text` key OR
      - a top-level `output`/`result` string

    Returns a dict matching the OpenAI completion response shape with keys:
      - id, object, created, model, choices (list), usage (minimal if available)
    """
    # Minimal metadata
    created = int(time.time())
    model = None
    if openai_req:
        model = openai_req.get("model")
    if not model:
        # Try to read model from flowise response if present
        model = flowise_resp.get("model")
    if not model:
        model = "flowise-model"

    # Extract outputs robustly
    raw_outputs: List[str] = []
    if isinstance(flowise_resp, dict):
        # Common patterns
        if "outputs" in flowise_resp:
            outs = flowise_resp["outputs"]
            if isinstance(outs, list):
                for o in outs:
                    if isinstance(o, str):
                        raw_outputs.append(o)
                    elif isinstance(o, dict) and "text" in o:
                        raw_outputs.append(str(o.get("text", "")))
                    else:
                        # Fallback to stringifying the element
                        raw_outputs.append(str(o))
            else:
                # single output
                raw_outputs.append(str(outs))
        elif "output" in flowise_resp:
            raw_outputs.append(str(flowise_resp["output"]))
        elif "result" in flowise_resp:
            raw_outputs.append(str(flowise_resp["result"]))
        else:
            # As last resort, if there's a 'data' or 'text' key
            if "text" in flowise_resp:
                raw_outputs.append(str(flowise_resp["text"]))
            elif "data" in flowise_resp:
                raw_outputs.append(str(flowise_resp["data"]))
    else:
        # If it's a plain string or list
        if isinstance(flowise_resp, list):
            for item in flowise_resp:
                raw_outputs.append(str(item))
        else:
            raw_outputs.append(str(flowise_resp))

    # If no outputs found, ensure at least an empty string choice
    if not raw_outputs:
        raw_outputs = [""]

    # Build choices according to requested `n` if available
    n = 1
    if openai_req:
        try:
            n = int(openai_req.get("n", 1))
        except Exception:
            n = 1

    choices: List[Dict[str, Any]] = []
    # Take up to n outputs, or repeat the first one if less available
    for i in range(n):
        if i < len(raw_outputs):
            text = raw_outputs[i]
        else:
            text = raw_outputs[0]
        choices.append(_build_choice(text, index=i, finish_reason="stop"))

    # Minimal usage: tokens not known from Flowise; set to None or estimates
    usage = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }

    response = {
        "id": f"flowise-{int(time.time()*1000)}",
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": choices,
        "usage": usage,
    }

    return response


if __name__ == "__main__":
    # Quick example usage
    example_openai = {
        "model": "gpt-test",
        "prompt": "Say hello",
        "max_tokens": 16,
        "temperature": 0.7,
        "n": 2,
    }

    req = openai_to_flowise(example_openai)
    print("Flowise request:\n", req)

    # Try loading a saved Flowise response for testing
    json_path = Path(__file__).parent / "flowise_raw.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                flowise_resp = json.load(f)
            print(f"Loaded test Flowise response from {json_path}")
        except Exception as e:
            print(f"Failed to load {json_path}: {e}")
            flowise_resp = {"outputs": ["Hello!", "Hi there!"]}
    else:
        # Fallback sample
        flowise_resp = {"outputs": ["Hello!", "Hi there!"]}

    # Ensure example_openai exists (some editors may have commented it out).
    if "example_openai" not in globals():
        example_openai = {
            "model": "gpt-test",
            "prompt": "Say hello",
            "max_tokens": 16,
            "temperature": 0.7,
            "n": 1,
        }

    oa = flowise_to_openai(flowise_resp, example_openai)
    print("OpenAI-style response:\n", oa)
