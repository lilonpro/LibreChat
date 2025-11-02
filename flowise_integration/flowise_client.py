"""Flowise API Client

A client for interacting with Flowise API endpoints. Uses the OpenAI adapter
for request/response format conversions.

Example:
    client = FlowiseClient(
        base_url="https://demo.flow.legaleagle-ai.com/api/v1",
        flow_id="5518949f-3ebc-4082-af01-fa2a18623da6",
        api_key="your_api_key"
    )
    response = client.predict("What is the price for Apple?")
"""
from typing import Any, Dict, List, Optional
import os
from pathlib import Path
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urljoin
import json

from openai_adapter import openai_to_flowise, flowise_to_openai

class FlowiseClient:
    """Client for interacting with a Flowise server.

    If `api_key` is not provided it will attempt to read `FLOWISE_API_KEY`
    from the environment. If a `.env` file exists next to this module it
    will be loaded (simple KEY=VALUE parser) to populate environment
    variables.
    """
    
    def __init__(self, base_url: str, flow_id: str, api_key: Optional[str] = None):
        """Initialize the client.
        
        Args:
            base_url: Base URL of the Flowise server (e.g., https://demo.flow.legaleagle-ai.com/api/v1)
            flow_id: ID of the flow to use
            api_key: API key for authentication. If omitted, uses FLOWISE_API_KEY from env/.env
        """
        self.base_url = base_url.rstrip('/')
        self.flow_id = flow_id

        # Load .env file next to this module if present to populate FLOWISE_API_KEY
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            try:
                with env_path.open("r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        # Only set if not already in environment
                        if k and os.getenv(k) is None:
                            os.environ[k] = v
            except Exception:
                # Fail silently; environment may already have necessary vars
                pass

        # Final api_key resolution
        self.api_key = api_key or os.getenv("FLOWISE_API_KEY")
        if not self.api_key:
            raise ValueError("Flowise API key not provided and FLOWISE_API_KEY not set in environment")

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Setup a requests Session with retries to be resilient to transient errors
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=(502, 503, 504))
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Default request timeout (seconds)
        self.timeout = int(os.getenv("FLOWISE_REQUEST_TIMEOUT", "30"))
        
    def predict(self, question: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Send a prediction request to the Flowise server.
        
        Args:
            question: The question to ask
            history: Optional chat history in OpenAI format
            
        Returns:
            The OpenAI-formatted response
            
        Raises:
            requests.exceptions.RequestException: If the API request fails
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
        try:
            self.logger.debug("POST %s payload=%s", url, flowise_req)
            response = self.session.post(url, json=flowise_req, headers=headers, timeout=self.timeout)
            response.raise_for_status()  # Raise exception for failed requests
        except requests.exceptions.RequestException as e:
            self.logger.exception("Flowise request failed: %s", e)
            raise

        # Convert Flowise response back to OpenAI format
        flowise_resp = response.json()
        return flowise_to_openai(flowise_resp, openai_req)

if __name__ == "__main__":
    pass  # See main.py for usage examples