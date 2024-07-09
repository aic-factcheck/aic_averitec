"""
Chat module: provides a simple interface for interacting with OpenAI's chat API. It includes the SimpleJSONChat class, which facilitates sending messages and receiving responses from the API in a structured JSON format. Additionally, the module contains a utility function, pretty_print, for formatting long strings of text with line breaks for improved readability.

Classes:
    SimpleJSONChat: Simplistic class to handle communication with OpenAI's chat API, abstracting away the sending of prompts and receiving of responses.

Functions:
    pretty_print(text: str, break_line_at: int = 90) -> str: Pretty prints a given text with line breaks at specified intervals.
"""

import openai
import json
from typing import Any, Dict, List, Optional, Union


class SimpleJSONChat:
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        client: Optional[openai.OpenAI] = None,
        model: str = "gpt-4-1106-preview",
        temperature: float = 0.0,
        parse_output: bool = True,
        **openai_kwargs: Any,
    ) -> None:
        """
        Initialize a SimpleJSONChat object to interact with OpenAI's chat API.

        Args:
            system_prompt (Optional[str]): The initial prompt that sets the context for the conversation.
            client (Optional[openai.OpenAI]): An instance of the OpenAI API client.
            model (str): The model to be used for generating responses.
            temperature (float): Controls the randomness of the response. Lower values mean less random responses.
            parse_output (bool): If True, parse the output as JSON, otherwise return the raw string.
            **openai_kwargs: Additional keyword arguments to pass to the OpenAI API.
        """
        if client is None:
            client = openai.OpenAI()
        self.system_prompt = system_prompt
        self.client = client
        self.model = model
        self.temperature = temperature
        self.openai_kwargs = openai_kwargs
        self.parse_output = parse_output

    def __call__(
        self, user_prompts: Union[str, List[str]], system_prompt: Optional[str] = None
    ) -> Union[Dict[str, Any], str, List]:
        """
        Send a message to the OpenAI chat API and receive a response.

        Args:
            user_prompts (Union[str, List[str]]): The user's message or messages to send to the API.
            system_prompt (Optional[str]): An optional override for the initial system prompt.

        Returns:
            Union[Dict[str, Any], str, List]: The parsed JSON response if parse_output is True,
            the raw string response otherwise, or an empty list in case of an error.
        """
        if not isinstance(user_prompts, list):
            user_prompts = [user_prompts]
        if system_prompt is None:
            system_prompt = self.system_prompt

        messages = [{"role": "system", "content": system_prompt}]
        for user_prompt in user_prompts:
            messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, messages=messages, **self.openai_kwargs
        )
        try:
            result = response.choices[0].message.content.replace("```json", "").replace("```", "")
            return json.loads(result) if self.parse_output else result
        except:
            print("Error parsing response from OpenAI chat API.\n", response.choices[0].message.content)
            return []

    def __repr__(self) -> str:
        """
        Return a string representation of the SimpleJSONChat object.

        Returns:
            str: A string representation of the object.
        """
        return f"SimpleJSONChat(system_prompt={self.system_prompt}, model={self.model}, temperature={self.temperature}, kwargs={self.openai_kwargs})"

    def __str__(self) -> str:
        """
        Return a string representation of the SimpleJSONChat object.

        Returns:
            str: A string representation of the object.
        """
        return self.__repr__()


def pretty_print(text: str, break_line_at: int = 90) -> str:
    """
    Pretty print a given text by inserting line breaks at specified intervals.

    Args:
        text (str): The text to be pretty printed.
        break_line_at (int): The maximum line length before inserting a line break.

    Returns:
        str: The pretty printed text.
    """
    words = text.split(" ")
    lines = []
    for word in words:
        if len(lines) == 0:
            lines.append(word)
        elif len(lines[-1]) + len(word) + 1 <= break_line_at:
            lines[-1] += " " + word
        else:
            lines.append(word)

    return "\n".join(lines)
