# structllm

<div style="text-align: center;">
  <img width="100%" src="structllm.svg" alt="Logo">
</div>

[![PyPI version](https://badge.fury.io/py/structllm.svg)](https://badge.fury.io/py/structllm)
[![Python Support](https://img.shields.io/pypi/pyversions/structllm.svg)](https://pypi.org/project/structllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**structllm** is a universal and lightweight Python library that provides [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses) functionality for any LLM provider (OpenAI, Anthropic, Mistral, local models, etc.), not just OpenAI. It guarantees that LLM responses conform to your provided JSON schema using Pydantic models.

If your LLM model has 7B parameters or more, it can be used with structllm.

## Installation

```bash
pip install structllm
```

Or using uv (recommended):

```bash
uv add structllm
```

## Quick Start

```python
from pydantic import BaseModel
from structllm import StructLLM
from typing import List

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: List[str]

client = StructLLM(
    api_base="https://openrouter.ai/api/v1",
    api_key="sk-or-v1-...",
)

messages = [
    {"role": "system", "content": "Extract the event information."},
    {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
]

response = client.parse(
    model="openrouter/moonshotai/kimi-k2",
    messages=messages,
    response_format=CalendarEvent,
)

if response.output_parsed:
    print(response.output_parsed)
    # {"name": "science fair", "date": "Friday", "participants": ["Alice", "Bob"]}
else:
    print("Failed to parse structured output")
```

## Provider Support

StructLLM works with **100+ LLM providers** through LiteLLM. Check the [LiteLLM documentation](https://docs.litellm.ai/docs/providers) for the full list of supported providers.

## Advanced Usage

### Complex Data Structures

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Task(BaseModel):
    title: str = Field(description="The task title")
    description: Optional[str] = Field(default=None, description="Task description")
    priority: Priority = Field(description="Task priority level")
    assignees: List[str] = Field(description="List of assigned people")
    due_date: Optional[str] = Field(default=None, description="Due date in YYYY-MM-DD format")

client = StructLLM(
    api_base="https://openrouter.ai/api/v1",
    api_key="sk-or-v1-...",
)

response = client.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {
            "role": "user",
            "content": "Create a high-priority task for John and Sarah to review the quarterly report by next Friday."
        }
    ],
    response_format=Task,
)

task = response.output_parsed
print(f"Task: {task.title}")
print(f"Priority: {task.priority}")
print(f"Assignees: {task.assignees}")
```

### Error Handling

```python
response = client.parse(
    model="gpt-4o-2024-08-06",
    messages=messages,
    response_format=CalendarEvent,
)

if response.output_parsed:
    # Successfully parsed
    event = response.output_parsed
    print(f"Parsed event: {event}")
else:
    # Parsing failed, but raw response is available
    print("Failed to parse structured output")
    print(f"Raw response: {response.raw_response.choices[0].message.content}")
```

### Custom Configuration

```python
client = StructLLM(
    api_base="https://api.custom-provider.com/v1",
    api_key="your-api-key"
)

response = client.parse(
    model="custom/model-name",
    messages=messages,
    response_format=YourModel,
    temperature=0.1,
    top_p=0.1,
    max_tokens=1000,
    # Any additional parameters supported by the LiteLLM interface
    custom_parameter="value"
)
```

## How It Works

StructLLM uses prompt engineering to ensure structured outputs:

1. **Schema Injection**: Automatically injects your Pydantic model's JSON schema into the system prompt
2. **Format Instructions**: Adds specific instructions for JSON-only responses
3. **Intelligent Parsing**: Extracts JSON from responses even when wrapped in additional text
4. **Validation**: Uses Pydantic for robust type checking and validation
5. **Fallback Handling**: Gracefully handles parsing failures while preserving raw responses

By default it uses low `temperature` and `top_p` settings to ensure consistent outputs, but you can customize these parameters as needed.

## Testing

Run the test suite:

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest
uv run pytest -m "not integration"

# Run integration tests (requires external services)
uv run pytest -m "integration"

# Run linting
uv run ruff check .
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes with tests
4. Run the test suite: `uv run pytest`
5. Run linting: `uv run ruff check .`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LiteLLM](https://github.com/BerriAI/litellm) for providing the universal LLM interface
- [Pydantic](https://github.com/pydantic/pydantic) for structured data validation