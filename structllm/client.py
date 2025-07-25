"""StructLLM client implementation."""

import json
from typing import Any, Dict, List, Optional, Type, TypeVar

import litellm
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class StructuredResponse:
    """Response object for structured outputs."""

    def __init__(self, raw_response: Any, output_parsed: Optional[BaseModel] = None):
        self.raw_response = raw_response
        self.output_parsed = output_parsed


class StructLLM:
    """Universal client for structured outputs with any LLM provider."""

    def __init__(self, api_base: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize the StructLLM client.

        Args:
            api_base: Base URL for the API (optional)
            api_key: API key for authentication (optional)
        """
        self.api_base = api_base
        self.api_key = api_key

        if api_base:
            litellm.api_base = api_base
        if api_key:
            litellm.api_key = api_key

    def parse(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Type[T],
        temperature: float = 0.1,
        top_p: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> StructuredResponse:
        """Parse structured output from LLM.

        Args:
            model: Model name (e.g., "gpt-4o-2024-08-06", "claude-3-5-sonnet-20240620")
            messages: List of messages for the conversation
            response_format: Pydantic model class for structured output
            temperature: Sampling temperature (default: 0.1 for consistency)
            top_p: Top-p sampling parameter (default: 0.1 for consistency)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for litellm.completion

        Returns:
            StructuredResponse containing the parsed output
        """
        # Get JSON schema from Pydantic model
        schema = response_format.model_json_schema()

        # Create system message with structured output instructions
        structured_messages = self._prepare_messages(messages, schema)

        # Call LiteLLM with structured output parameters
        completion_kwargs = {
            "model": model,
            "messages": structured_messages,
            "temperature": temperature,
            "top_p": top_p,
            **kwargs,
        }

        if max_tokens is not None:
            completion_kwargs["max_tokens"] = max_tokens

        # Add api_base and api_key if configured
        if self.api_base:
            completion_kwargs["api_base"] = self.api_base
        if self.api_key:
            completion_kwargs["api_key"] = self.api_key

        response = litellm.completion(**completion_kwargs)

        # Extract and parse the response
        content = response.choices[0].message.content

        try:
            # Try to parse as JSON and validate with Pydantic
            json_data = json.loads(content)
            parsed_output = response_format(**json_data)

            return StructuredResponse(
                raw_response=response, output_parsed=parsed_output
            )
        except (json.JSONDecodeError, ValidationError):
            # If parsing fails, try to extract JSON from the content
            extracted_json = self._extract_json(content)
            if extracted_json:
                try:
                    parsed_output = response_format(**extracted_json)
                    return StructuredResponse(
                        raw_response=response, output_parsed=parsed_output
                    )
                except ValidationError:
                    pass

            # If all parsing attempts fail, return raw response
            return StructuredResponse(raw_response=response)

    @staticmethod
    def _prepare_messages(
        messages: List[Dict[str, str]], schema: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Prepare messages with structured output instructions."""

        # Create structured output system message
        structured_instruction = (
            "You must respond with a valid JSON object that conforms exactly "
            "to the following schema. "
            "IMPORTANT: Do not use Markdown formatting, just return the JSON "
            "object directly.\n\n"
            f"JSON Schema:\n{json.dumps(schema, indent=2)}"
        )

        # Modify or add system message
        structured_messages = []
        has_system_message = False

        for message in messages:
            if message["role"] == "system":
                # Append structured output instructions to existing system message
                enhanced_content = f"{message['content']}\n\n{structured_instruction}"
                structured_messages.append(
                    {"role": "system", "content": enhanced_content}
                )
                has_system_message = True
            else:
                structured_messages.append(message)

        # If no system message exists, add one
        if not has_system_message:
            structured_messages.insert(
                0, {"role": "system", "content": structured_instruction}
            )

        return structured_messages

    @staticmethod
    def _extract_json(content: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from content that might contain additional text."""
        # Try to find JSON within the content
        content = content.strip()

        # Look for JSON object boundaries
        start_idx = content.find("{")
        if start_idx == -1:
            return None

        # Find the matching closing brace
        brace_count = 0
        end_idx = -1

        for i in range(start_idx, len(content)):
            if content[i] == "{":
                brace_count += 1
            elif content[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break

        if end_idx == -1:
            return None

        json_str = content[start_idx : end_idx + 1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
