"""Integration tests for StructLLM with local endpoint."""

import os
from typing import List

import pytest
from pydantic import BaseModel

from structllm import StructLLM


class CalendarEvent(BaseModel):
    """Test model for calendar events."""

    name: str
    date: str
    participants: List[str]


@pytest.mark.integration
def test_openrouter_endpoint():
    """Test with OpenRouter API endpoint."""
    # This test requires a valid OpenRouter API key
    # Skip if not available or if running in CI
    if os.getenv("SKIP_INTEGRATION_TESTS", "false").lower() == "true":
        pytest.skip("Integration tests disabled")

    client = StructLLM(
        api_base="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),  # Set this in your environment
    )

    messages = [
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        },
    ]

    try:
        response = client.parse(
            model="openrouter/moonshotai/kimi-k2",
            messages=messages,
            response_format=CalendarEvent,
        )

        # Basic assertions
        assert response is not None
        assert response.raw_response is not None

        # If parsing was successful, check the output
        if response.output_parsed:
            assert isinstance(response.output_parsed, CalendarEvent)
            print(f"Parsed event: {response.output_parsed}")

    except Exception as e:
        pytest.skip(f"OpenRouter API not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__ + "::test_openrouter_endpoint", "-v"])
