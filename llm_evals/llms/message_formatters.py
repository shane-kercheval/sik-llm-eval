"""Contains helper functions for formatting LLM messages based on various types of models."""

from llm_evals.llms.base import ExchangeRecord


def openai_message_formatter(
        system_message: str | None,
        history: list[ExchangeRecord] | None,
        prompt: str | None) -> list[dict]:
    """
    A message formatter takes a system_message, list of messages (ExchangeRecord objects), and a
    prompt, and formats them according to the best practices for interacting with the model.
    """
    # initial message; always keep system message regardless of memory_manager
    messages = []
    if system_message:
        messages += [{'role': 'system', 'content': system_message}]
    if history:
        for message in history:
            messages += [
                {'role': 'user', 'content': message.prompt},
                {'role': 'assistant', 'content': message.response},
            ]
    if prompt:
        messages += [{'role': 'user', 'content': prompt}]
    return messages


# For example, for Lamma-2-7b, the messages should be formatted as follows:
#     [INST] <<SYS>> You are a helpful assistant. <</SYS>> [/INST]
#     [INST] Hello, how are you? [/INST]
#     I am doing well. How are you?
#     [INST] I am doing well. How's the weather? [/INST]
#     It is sunny today.
SYSTEM_FORMAT_LLAMA = "[INST] <<SYS>> {system_message} <</SYS>> [/INST]\n"
PROMPT_FORMAT_LLAMA = "[INST] {prompt} [/INST]\n"
RESPONSE_FORMAT_LLAMA = "{response}\n"

# For example, for a Mistral, the messages should be formatted as follows:
#     You are a helpful assistant.[INST]Hello, how are you?[/INST]
#     I am doing well. How are you?
#     [INST]I am doing well. How's the weather?[/INST]
#     It is sunny today.
SYSTEM_FORMAT_MISTRAL = None
PROMPT_FORMAT_MISTRAL = "[INST]{prompt}[/INST]"
RESPONSE_FORMAT_MISTRAL = "{response}\n"


def _format_template(template: str, content: str) -> str:
    """Formats a single message using a template."""
    return template.format(**{'system_message': content, 'prompt': content, 'response': content})


def create_message_formatter(
        system_format: str | None = '{system_message}',
        prompt_format: str | None = '{prompt}',
        response_format: str | None = '{response}') -> callable:
    """Returns a function that formats messages for interacting with an LLM."""

    def message_formatter(
            system_message: str | None,
            history: list[ExchangeRecord | tuple] | None,
            prompt: str | None) -> str:
        """Formats messages for interacting with an LLM."""
        formatted_messages = []
        if system_message:
            formatted_messages.append(_format_template(system_format, system_message))
        if history:
            for message in history:
                prompt_text = message.prompt if isinstance(message, ExchangeRecord) else message[0]
                response_text = message.response if isinstance(message, ExchangeRecord) else message[1]  # noqa
                formatted_messages.append(
                    _format_template(prompt_format, prompt_text) +
                    _format_template(response_format, response_text),
                )
        if prompt:
            formatted_messages.append(_format_template(prompt_format, prompt))
        return ''.join(formatted_messages)

    return message_formatter


llama_message_formatter = create_message_formatter(
    system_format=SYSTEM_FORMAT_LLAMA,
    prompt_format=PROMPT_FORMAT_LLAMA,
    response_format=RESPONSE_FORMAT_LLAMA,
)

mistral_message_formatter = create_message_formatter(
    system_format=SYSTEM_FORMAT_MISTRAL,
    prompt_format=PROMPT_FORMAT_MISTRAL,
    response_format=RESPONSE_FORMAT_MISTRAL,
)

# def _format_template(template: str, content: str) -> str:
#     """Formats a single message using a template."""
#     return template.format(**{'system_message': content, 'prompt': content, 'response': content})


# class MessageFormatter:
#     """Formats messages for interacting with an LLM."""

#     def __init__(
#             self,
#             system_format: str | None,
#             prompt_format: str | None,
#             response_format: str | None) -> None:
#         """
#         Args:
#             system_format:
#                 A string template for formatting the system message.
#             prompt_format:
#                 A string template for formatting the prompt.
#             response_format:
#                 A string template for formatting the response.
#         """
#         self.system_format = system_format or '{system_message}'
#         self.prompt_format = prompt_format or '{prompt}'
#         self.response_format = response_format or '{response}'

#     def __call__(
#             self,
#             system_message: str | None,
#             history: list[ExchangeRecord | tuple] | None,
#             prompt: str | None) -> str:
#         """Formats messages for interacting with an LLM."""
#         formatted_messages = []
#         if system_message:
#             formatted_messages.append(_format_template(self.system_format, system_message))
#         if history:
#             for message in history:
#               prompt_text = message.prompt if isinstance(message, ExchangeRecord) else message[0]
#                 response_text = message.response if isinstance(message, ExchangeRecord) else message[1]  # noqa
#                 formatted_messages.append(
#                     _format_template(self.prompt_format, prompt_text) +
#                     _format_template(self.response_format, response_text),
#                 )
#         if prompt:
#             formatted_messages.append(_format_template(self.prompt_format, prompt))
#         return ''.join(formatted_messages)


# def llama_message_formatter(
#         system_message: str | None,
#         history: list[ExchangeRecord] | None,
#         prompt: str | None) -> str:
#     """
#     A message formatter takes a list of messages (ExchangeRecord objects) and formats them
#     according to the best practices for interacting with the model.

#     For example, for Lamma-2-7b, the messages should be formatted as follows:
#         [INST] <<SYS>> You are a helpful assistant. <</SYS>> [/INST]
#         [INST] Hello, how are you? [/INST]
#         I am doing well. How are you?
#         [INST] I am doing well. How's the weather? [/INST]
#         It is sunny today.

#     https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

#     Args:
#         system_message:
#             The content of the message associated with the "system" `role`.
#         history:
#             A list of ExchangeRecord objects, containing the prompt/response pairs.
#         prompt:
#             The next prompt to be sent to the model.
#     """
#     formatted_messages = []
#     if system_message:
#         formatted_messages.append(f"[INST] <<SYS>> {system_message} <</SYS>> [/INST]\n")
#     if history:
#         for message in history:
#             formatted_messages.append(
#                 f"[INST] {message.prompt} [/INST]\n" + f"{message.response}\n",
#             )
#     if prompt:
#         formatted_messages.append(f"[INST] {prompt} [/INST]\n")
#     return ''.join(formatted_messages)


# def mistral_message_formatter(
#     system_message: str | None,
#     history: list[ExchangeRecord] | None,
#     prompt: str | None) -> str:
#     """
#     A message formatter takes a list of messages (ExchangeRecord objects) and formats them
#     according to the best practices for interacting with the model.

#     For example, for a Mistral, the messages should be formatted as follows:
#         You are a helpful assistant.[INST]Hello, how are you?[/INST]
#         I am doing well. How are you?
#         [INST]I am doing well. How's the weather?[/INST]
#         It is sunny today.

#     Args:
#         system_message:
#             The content of the message associated with the "system" `role`.
#         history:
#             A list of ExchangeRecord objects, containing the prompt/response pairs.
#         prompt:
#             The next prompt to be sent to the model.
#     """
#     formatted_messages = ['<s>']
#     if system_message:
#         formatted_messages.append(f"{system_message}")
#     if history:
#         for message in history:
#             formatted_messages.append(
#                 f"[INST]{message.prompt}[/INST] " + f"{message.response}\n\n",
#             )
#     if prompt:
#         formatted_messages.append(f"[INST]{prompt}[/INST] ")
#     return ''.join(formatted_messages)
