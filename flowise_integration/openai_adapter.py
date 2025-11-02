"""OpenAI <-> Flowise adapter

Provides conversion helpers to translate an OpenAI-style completion request
into a Flowise prediction request, and to translate a Flowise prediction
response back into an OpenAI-style completion response.

This also provides a FlowiseClient class to interact with a Flowise server.

Example:
    client = FlowiseClient(
        base_url="https://demo.flow.legaleagle-ai.com/api/v1",
        flow_id="5518949f-3ebc-4082-af01-fa2a18623da6",
        api_key="12345"
    )
    response = client.predict("What is the price for Apple?")
"""
from typing import Any, Dict, List, Optional
import time
import json
from pathlib import Path
import requests
from urllib.parse import urljoin


def _merge_adjacent_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge adjacent messages with the same role into a single message.

    Concatenates `content` with a newline between messages. Preserves an
    optional `name` field: if merged entry lacks name and a subsequent
    message has a name, that name will be added.
    """
    if not messages:
        return []
    merged: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            # normalize non-dict entries
            m = {"role": None, "content": str(m)}
        role = m.get("role")
        content = m.get("content", "")
        name = m.get("name")

        if merged and merged[-1].get("role") == role:
            # join contents
            prev = merged[-1]
            prev_content = prev.get("content", "")
            if prev_content:
                prev["content"] = prev_content + "\n" + content
            else:
                prev["content"] = content
            # prefer existing name, but fill if missing
            if not prev.get("name") and name:
                prev["name"] = name
        else:
            entry: Dict[str, Any] = {"role": role, "content": content}
            if name:
                entry["name"] = name
            merged.append(entry)
    return merged


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

    # Assemble Flowise-style request based on the documented format
    flowise_req: Dict[str, Any] = {
        "question": prompt,  # Main input field is 'question'
        "history": [],      # Initialize empty history
        "streaming": False, # Default to non-streaming mode
        "overrideConfig": {}  # Initialize overrideConfig for parameters
    }

    # Add chat history if present (either from messages or history field)
    if "messages" in openai_req:
        hist = _merge_adjacent_history(openai_req["messages"])
        # Convert OpenAI roles to Flowise roles (user -> userMessage, assistant -> apiMessage)
        flowise_req["history"] = [
            {"role": "userMessage" if m["role"] == "user" else "apiMessage", 
             "content": m["content"]} 
            for m in hist
        ]
    elif "history" in openai_req:
        hist = _merge_adjacent_history(openai_req["history"])
        flowise_req["history"] = [
            {"role": "userMessage" if m["role"] == "user" else "apiMessage", 
             "content": m["content"]} 
            for m in hist
        ]

    # Map parameters to overrideConfig
    # Map parameters to overrideConfig with correct names
    params: Dict[str, Any] = flowise_req["overrideConfig"]
    if temperature is not None:
        params["temperature"] = float(temperature)
    if max_tokens is not None:
        params["maxTokens"] = int(max_tokens)
    
    # Session ID is required - generate one if not provided
    params["sessionId"] = openai_req.get("sessionId", f"session-{int(time.time()*1000)}")
    
    # Optional form data
    if "form" in openai_req:
        flowise_req["form"] = openai_req["form"]
    else:
        flowise_req["form"] = {"count": n}  # Use n as count in form
        
    # Uploads field (if present)
    if "uploads" in openai_req:
        flowise_req["uploads"] = openai_req["uploads"]
        
    # Human input field (if present)
    if "humanInput" in openai_req:
        flowise_req["humanInput"] = openai_req["humanInput"]

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

    # Determine if caller expects chat-style messages (OpenAI Chat API)
    chat_mode = False
    if openai_req and ("messages" in openai_req or "history" in openai_req):
        chat_mode = True

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

        if chat_mode:
            # ChatCompletion-style choice with a message object
            choice = {
                "index": i,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
            # keep compatibility: also include text field
            choice["text"] = text
            choices.append(choice)
        else:
            choices.append(_build_choice(text, index=i, finish_reason="stop"))

    # Minimal usage: tokens not known from Flowise; set to None or estimates
    usage = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }

    response = {
        "id": f"flowise-{int(time.time()*1000)}",
        "object": "text_completion" if not chat_mode else "chat.completion",
        "created": created,
        "model": model,
        "choices": choices,
        "usage": usage,
    }

    # If chat_mode, include the conversation history for convenience
    if chat_mode:
        conv: List[Dict[str, Any]] = []
        # Prefer explicit messages passed in the original request
        if openai_req and "messages" in openai_req:
            conv = _merge_adjacent_history(list(openai_req["messages"]))
        else:
            # Try to extract chatHistory from flowise response nodes
            if isinstance(flowise_resp, dict):
                # Search agentFlowExecutedData for chatHistory arrays
                afd = flowise_resp.get("agentFlowExecutedData") or []
                for node in afd:
                    data = node.get("data", {})
                    ch = data.get("chatHistory")
                    if isinstance(ch, list):
                        # merge node-level chatHistory entries before appending
                        node_msgs: List[Dict[str, Any]] = []
                        for m in ch:
                            if isinstance(m, dict) and "role" in m and "content" in m:
                                node_msgs.append({"role": m["role"], "content": m["content"], **({"name": m.get("name")} if m.get("name") else {})})
                        node_msgs = _merge_adjacent_history(node_msgs)
                        for nm in node_msgs:
                            conv.append(nm)
        # Append assistant final message(s) from choices
        if choices:
            # take first choice's text/message
            first = choices[0]
            assistant_entry = None
            if "message" in first:
                assistant_entry = {"role": "assistant", "content": first["message"]["content"]}
            elif "text" in first:
                assistant_entry = {"role": "assistant", "content": first["text"]}
            if assistant_entry:
                conv.append(assistant_entry)

        # Merge adjacent entries in the assembled conversation
        conv = _merge_adjacent_history(conv)
        response["conversation"] = conv

    return response


class FlowiseClient:
    """Client for interacting with a Flowise server."""
    
    def __init__(self, base_url: str, flow_id: str, api_key: str):
        """Initialize the client.
        
        Args:
            base_url: Base URL of the Flowise server (e.g., https://demo.flow.legaleagle-ai.com/api/v1)
            flow_id: ID of the flow to use
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.flow_id = flow_id
        self.api_key = api_key
        
    def predict(self, question: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Send a prediction request to the Flowise server.
        
        Args:
            question: The question to ask
            history: Optional chat history in OpenAI format
            
        Returns:
            The OpenAI-formatted response
        """
        # Convert simple question into OpenAI request format
        openai_req = {
            "messages": [{"role": "user", "content": question}]
        }
        if history:
            openai_req["messages"] = history + openai_req["messages"]
            
        # Convert to Flowise format
        flowise_req = openai_to_flowise(openai_req)
        
        # Make request to Flowise server
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        url = f"{self.base_url}/prediction/{self.flow_id}"
        response = requests.post(url, json=flowise_req, headers=headers)
        response.raise_for_status()  # Raise exception for failed requests
        
        # Convert Flowise response back to OpenAI format
        flowise_resp = response.json()
        return flowise_to_openai(flowise_resp, openai_req)


if __name__ == "__main__":
    # Test the client if credentials are provided
    test_client = False
    if test_client:
        client = FlowiseClient(
            base_url="https://demo.flow.legaleagle-ai.com/api/v1",
            flow_id="5518949f-3ebc-4082-af01-fa2a18623da6",
            api_key="12345"  # Replace with actual API key
        )
        response = client.predict("What is the price for Apple?")
        print("\nFlowise API Response:")
        print(json.dumps(response, indent=2))
    
    # Load test inputs from JSON files for format conversion testing
    flowise_path = Path(__file__).parent / "flowise_raw.json"
    openai_path = Path(__file__).parent / "openai_raw.json"
    
    try:
        with open(flowise_path, "r", encoding="utf-8") as f:
            flowise_resp = json.load(f)
        with open(openai_path, "r", encoding="utf-8") as f:
            openai_req = json.load(f)
    except Exception as e:
        print(f"Failed to load test files: {e}")
        exit(1)

    # Convert OpenAI request → Flowise request
    flowise_req = openai_to_flowise(openai_req)
    print("\nOpenAI → Flowise request:")
    print(json.dumps(flowise_req, indent=2))

    # Convert Flowise response → OpenAI response
    openai_resp = flowise_to_openai(flowise_resp, openai_req)
    print("\nFlowise → OpenAI response:")
    print(json.dumps(openai_resp, indent=2))
