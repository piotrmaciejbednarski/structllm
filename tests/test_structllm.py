"""Tests for the StructLLM library."""

import json
from typing import List
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from structllm import StructLLM
from structllm.client import StructuredResponse


class CalendarEvent(BaseModel):
    """Test model for calendar events."""

    name: str
    date: str
    participants: List[str]


class TestStructLLM:
    """Test cases for StructLLM client."""

    @staticmethod
    def test_init_default():
        """Test StructLLM initialization with default parameters."""
        client = StructLLM()
        assert client is not None

    @staticmethod
    def test_init_with_params():
        """Test StructLLM initialization with custom parameters."""
        with patch("structllm.client.litellm") as mock_litellm:
            client = StructLLM(api_base="http://localhost:1234/v1", api_key="test-key")
            assert client is not None
            assert mock_litellm.api_base == "http://localhost:1234/v1"
            assert mock_litellm.api_key == "test-key"

    @patch("structllm.client.litellm.completion")
    def test_parse_successful(self, mock_completion):
        """Test successful parsing of structured output."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {"name": "Science Fair", "date": "Friday", "participants": ["Alice", "Bob"]}
        )
        mock_completion.return_value = mock_response

        client = StructLLM()
        messages = [
            {"role": "system", "content": "Extract the event information."},
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday.",
            },
        ]

        response = client.parse(
            model="openrouter/moonshotai/kimi-k2",
            messages=messages,
            response_format=CalendarEvent,
        )

        assert isinstance(response, StructuredResponse)
        assert response.output_parsed is not None
        assert isinstance(response.output_parsed, CalendarEvent)
        assert response.output_parsed.name == "Science Fair"
        assert response.output_parsed.date == "Friday"
        assert response.output_parsed.participants == ["Alice", "Bob"]

        # Verify completion was called with correct parameters
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        assert call_args[1]["model"] == "openrouter/moonshotai/kimi-k2"
        assert call_args[1]["temperature"] == 0.1
        assert call_args[1]["top_p"] == 0.1

    @patch("structllm.client.litellm.completion")
    def test_parse_with_json_extraction(self, mock_completion):
        """Test parsing when JSON is embedded in text."""
        # Mock response with extra text
        mock_response = Mock()
        mock_response.choices = [Mock()]
        json_obj = {
            "name": "Science Fair",
            "date": "Friday",
            "participants": ["Alice", "Bob"],
        }
        json_data = json.dumps(json_obj)
        mock_response.choices[
            0
        ].message.content = (
            f"Here is the extracted information: {json_data} Hope this helps!"
        )
        mock_completion.return_value = mock_response

        client = StructLLM()
        messages = [{"role": "user", "content": "Extract event info."}]

        response = client.parse(
            model="openrouter/moonshotai/kimi-k2",
            messages=messages,
            response_format=CalendarEvent,
        )

        assert isinstance(response, StructuredResponse)
        assert response.output_parsed is not None
        assert isinstance(response.output_parsed, CalendarEvent)
        assert response.output_parsed.name == "Science Fair"

    @patch("structllm.client.litellm.completion")
    def test_parse_failure_returns_raw(self, mock_completion):
        """Test that parsing failure returns raw response."""
        # Mock response with invalid JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is not JSON at all"
        mock_completion.return_value = mock_response

        client = StructLLM()
        messages = [{"role": "user", "content": "Extract event info."}]

        response = client.parse(
            model="openrouter/moonshotai/kimi-k2",
            messages=messages,
            response_format=CalendarEvent,
        )

        assert isinstance(response, StructuredResponse)
        assert response.output_parsed is None
        assert response.raw_response == mock_response

    @staticmethod
    def test_prepare_messages_with_system():
        """Test message preparation with existing system message."""
        client = StructLLM()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Extract info."},
        ]
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        result = client._prepare_messages(messages, schema)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "You are a helpful assistant." in result[0]["content"]
        assert "JSON Schema:" in result[0]["content"]
        assert result[1]["role"] == "user"

    @staticmethod
    def test_prepare_messages_without_system():
        """Test message preparation without system message."""
        client = StructLLM()
        messages = [{"role": "user", "content": "Extract info."}]
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        result = client._prepare_messages(messages, schema)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "JSON Schema:" in result[0]["content"]
        assert result[1]["role"] == "user"

    @staticmethod
    def test_extract_json_valid():
        """Test JSON extraction from text."""
        client = StructLLM()
        content = 'Here is the result: {"name": "test", "value": 123} and more text'

        result = client._extract_json(content)

        assert result is not None
        assert result["name"] == "test"
        assert result["value"] == 123

    @staticmethod
    def test_extract_json_invalid():
        """Test JSON extraction with no valid JSON."""
        client = StructLLM()
        content = "This contains no JSON at all"

        result = client._extract_json(content)

        assert result is None

    @staticmethod
    def test_extract_json_malformed():
        """Test JSON extraction with malformed JSON."""
        client = StructLLM()
        content = 'Here is broken JSON: {"name": "test", "value":} incomplete'

        result = client._extract_json(content)

        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])
