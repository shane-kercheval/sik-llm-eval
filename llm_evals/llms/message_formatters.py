"""Contains helper functions for formatting LLM messages based on various types of models."""

from llm_evals.llms.base import ExchangeRecord


def mistral_message_formatter(
    system_message: str | None,
    history: list[ExchangeRecord] | None,
    prompt: str | None) -> str:
    """
    A message formatter takes a list of messages (ExchangeRecord objects) and formats them
    according to the best practices for interacting with the model.

    For example, for a Mistral, the messages should be formatted as follows:
        You are a helpful assistant.[INST]Hello, how are you?[/INST]
        I am doing well. How are you?
        [INST]I am doing well. How's the weather?[/INST]
        It is sunny today.

    Args:
        system_message:
            The content of the message associated with the "system" `role`.
        history:
            A list of ExchangeRecord objects, containing the prompt/response pairs.
        prompt:
            The next prompt to be sent to the model.
    """
    formatted_messages = ['<s>']
    if system_message:
        formatted_messages.append(f"{system_message}")
    if history:
        for message in history:
            formatted_messages.append(
                f"[INST]{message.prompt}[/INST] " + f"{message.response}\n\n",
            )
    if prompt:
        formatted_messages.append(f"[INST]{prompt}[/INST] ")
    return ''.join(formatted_messages)
