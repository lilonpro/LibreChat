"""Test the OpenAI-compatible server using the Azure OpenAI SDK.

This demonstrates that our server implementation is compatible with
the OpenAI SDK by pointing it to our local endpoint.

Usage:
    # Start the server first:
    python openai_compatible_server.py

    # Then run this test:
    python test.py
"""
from openai import AzureOpenAI

def test_chat_completion(client: AzureOpenAI, flow_id: str):
    """Test the chat completions endpoint."""
    chat_completion = client.chat.completions.create(
        model=flow_id,
        messages=[
            {"role": "user", "content": "What is the price for Apple?"}
        ]
    )
    
    print("\nChat Completion Response:")
    print(f"Content: {chat_completion.choices[0].message.content}")
    print(f"Role: {chat_completion.choices[0].message.role}")
    print(f"Model: {chat_completion.model}")


def test_completion(client: AzureOpenAI, flow_id: str):
    """Test the legacy completions endpoint."""
    completion = client.completions.create(
        model=flow_id,
        prompt="What is the price for Apple?"
    )
    
    print("\nCompletion Response:")
    print(f"Text: {completion.choices[0].text}")
    print(f"Model: {completion.model}")


def test_with_history(client: AzureOpenAI, flow_id: str):
    """Test chat completion with conversation history."""
    messages = [
        {"role": "user", "content": "Hi, I want to know about stocks."},
        {"role": "assistant", "content": "Hello! I'd be happy to help you with stock information. What would you like to know?"},
        {"role": "user", "content": "What is the price for Apple?"}
    ]
    
    chat_completion = client.chat.completions.create(
        model=flow_id,
        messages=messages
    )
    
    print("\nChat Completion with History Response:")
    print(f"Content: {chat_completion.choices[0].message.content}")
    print(f"Role: {chat_completion.choices[0].message.role}")
    print(f"Model: {chat_completion.model}")


if __name__ == "__main__":
    # Hardcoded values for quick local testing
    SERVER_URL = "http://localhost:8000"
    FLOW_ID = "5518949f-3ebc-4082-af01-fa2a18623da6"
    
    client = AzureOpenAI(
        api_key="dummy-key",      # Not checked by our server
        api_version="2023-05-15", # Required by SDK but not used
        azure_endpoint=SERVER_URL
    )
    
    print(f"Testing OpenAI-compatible server at {SERVER_URL}")
    print(f"Using Flowise flow ID: {FLOW_ID}")
    try:
        test_chat_completion(client, FLOW_ID)
        test_completion(client, FLOW_ID)
        test_with_history(client, FLOW_ID)
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nError during tests: {e}")
        raise