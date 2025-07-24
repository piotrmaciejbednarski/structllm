"""Example usage of StructLLM library."""

from typing import List

from pydantic import BaseModel

from structllm import StructLLM


class CalendarEvent(BaseModel):
    """A calendar event with name, date, and participants."""

    name: str
    date: str
    participants: List[str]


def main():
    """Demonstrate StructLLM usage."""
    # Initialize the client for OpenRouter
    client = StructLLM(
        api_base="https://openrouter.ai/api/v1",
        api_key="sk-or-v1-...",
    )

    # Define the conversation
    messages = [
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        },
    ]

    try:
        # Parse structured output
        response = client.parse(
            model="openrouter/moonshotai/kimi-k2",
            messages=messages,
            response_format=CalendarEvent,
        )

        print("Raw response content:")
        print(f"{response.raw_response.choices[0].message.content}\n")
        # {"name": "science fair", "date": "Friday", "participants": ["Alice", "Bob"]}

        if response.output_parsed:
            print("Parsed structured output:")
            print(f"Name: {response.output_parsed.name}")  # science fair
            print(f"Date: {response.output_parsed.date}")  # Friday
            print(
                f"Participants: {response.output_parsed.participants}"
            )  # ['Alice', 'Bob']
        else:
            print("Failed to parse structured output")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a valid OpenRouter API key and internet connection")


if __name__ == "__main__":
    main()
