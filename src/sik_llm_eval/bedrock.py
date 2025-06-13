"""Provides a simple interface for using AWS Bedrock via the OpenAI API."""
from collections.abc import Callable
from sik_llm_eval.openai import (
    AsyncOpenAICompletion,
    OpenAIChatResponse,
    OpenAICompletion,
    OpenAICompletionResponse,
)


# https://aws.amazon.com/bedrock/pricing/
# https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
# https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-supported.html
MODEL_COST_PER_TOKEN = {
   'anthropic.claude-3-5-sonnet-20240620-v1:0': {'input': 0.003 / 1_000, 'output': 0.015 / 1_000},
   'anthropic.claude-3-5-sonnet-20241022-v2:0': {'input': 0.003 / 1_000, 'output': 0.015 / 1_000},
   'anthropic.claude-3-5-haiku-20241022-v1:0': {'input': 0.0008 / 1_000, 'output': 0.004 / 1_000},
   'anthropic.claude-3-haiku-20240307-v1:0': {'input': 0.00025 / 1_000, 'output': 0.00125 / 1_000},
}

class BedrockCompletion(OpenAICompletion):
    """
    Non-Async wrapper for OpenAI API which provides a simple interface for calling the
    chat.completions.create method and parsing the response.
    """

    def __call__(
            self,
            messages: list[str],
            model: str | None = None,
            stream_callback: Callable | None = None,
            user: str = 'sik-llm-eval',
            **model_kwargs: dict,
        ) -> OpenAIChatResponse | OpenAICompletionResponse:
        """
        Call the chat.completions.create method and parse the response.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            model: The model to use.
            stream_callback: A callback function to call with each chunk of the response.
            user: The user. This is required for telemetry in bedrock.
            **model_kwargs: Additional model parameters.
        """
        return super().__call__(
            messages=messages,
            model=model,
            stream_callback=stream_callback,
            user=user,
            **model_kwargs,
        )


class AsyncBedrockCompletion(AsyncOpenAICompletion):
    """
    Async wrapper for OpenAI API which provides a simple interface for calling the
    chat.completions.create method and parsing the response.
    """

    def __call__(
            self,
            messages: list[str],
            model: str | None = None,
            stream_callback: Callable | None = None,
            user: str = 'sik-llm-eval',
            **model_kwargs: dict,
        ) -> OpenAIChatResponse | OpenAICompletionResponse:
        """
        Call the chat.completions.create method and parse the response.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            model: The model to use.
            stream_callback: A callback function to call with each chunk of the response.
            user: The user. This is required for telemetry in bedrock.
            **model_kwargs: Additional model parameters.
        """
        return super().__call__(
            messages=messages,
            model=model,
            stream_callback=stream_callback,
            user=user,
            **model_kwargs,
        )


