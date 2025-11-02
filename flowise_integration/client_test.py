"""Test script for FlowiseClient

Example usage of the FlowiseClient class for interacting with Flowise API.
"""
import json
import os
import requests
from flowise_client import FlowiseClient

def main():
    # Initialize client with API key read from environment (or .env)
    api_key = os.getenv("FLOWISE_API_KEY")
    if not api_key:
        print("FLOWISE_API_KEY not set in environment; please set it in .env or env vars")
        return

    client = FlowiseClient(
        base_url="https://demo.flow.legaleagle-ai.com/api/v1",
        flow_id="5518949f-3ebc-4082-af01-fa2a18623da6",
        api_key=api_key
    )
    
    try:
        # Simple question
        response = client.predict("What is the price for Apple?")
        print("\nResponse for simple question:")
        print(json.dumps(response, indent=2))
        
        # # With history
        # history = [
        #     {"role": "user", "content": "Hi"},
        #     {"role": "assistant", "content": "Hello! How can I help?"}
        # ]
        # response = client.predict("What is the price for Apple?", history=history)
        # print("\nResponse with history:")
        # print(json.dumps(response, indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()