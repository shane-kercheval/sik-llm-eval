"""Tests the message formatters module."""
from llm_evals.llms.base import ExchangeRecord
from llm_evals.llms.message_formatters import (
    PROMPT_FORMAT_LLAMA,
    RESPONSE_FORMAT_LLAMA,
    SYSTEM_FORMAT_LLAMA,
    MessageFormatter,
    LlamaMessageFormatter,
)


def test__MessageFormatter__empty():  # noqa
    message_formatter = MessageFormatter(
        system_format=None,
        prompt_format=None,
        response_format=None,
    )
    assert message_formatter(None, None, None) == ''
    message_formatter = MessageFormatter(
        system_format=SYSTEM_FORMAT_LLAMA,
        prompt_format=PROMPT_FORMAT_LLAMA,
        response_format=RESPONSE_FORMAT_LLAMA,
    )
    assert message_formatter(None, None, None) == ''

def test_MessageFormatter():  # noqa
    message_formatter = MessageFormatter(
        system_format=SYSTEM_FORMAT_LLAMA,
        prompt_format=PROMPT_FORMAT_LLAMA,
        response_format=RESPONSE_FORMAT_LLAMA,
    )
    _system = 'test system'
    _message_prompt = 'test message prompt'
    _message_response = 'test message response'
    _prompt = 'test prompt'

    expected_value = SYSTEM_FORMAT_LLAMA.format(system_message=_system)
    assert message_formatter(_system, [], None) == expected_value
    assert message_formatter(_system, None, None) == expected_value
    assert message_formatter(_system, None, None) == expected_value

    expected_value = SYSTEM_FORMAT_LLAMA.format(system_message=_system) \
        + PROMPT_FORMAT_LLAMA.format(prompt=_prompt)
    assert message_formatter(_system, [], _prompt) == expected_value
    assert message_formatter(_system, None, _prompt) == expected_value

    expected_value = PROMPT_FORMAT_LLAMA.format(prompt=_prompt)
    assert message_formatter(None, [], _prompt) == expected_value
    assert message_formatter(None, None, _prompt) == expected_value
    assert message_formatter('', None, _prompt) == expected_value

    expected_value = SYSTEM_FORMAT_LLAMA.format(system_message=_system) \
        + PROMPT_FORMAT_LLAMA.format(prompt=_message_prompt) \
        + RESPONSE_FORMAT_LLAMA.format(response=_message_response)
    actual_value = message_formatter(
        _system,
        [ExchangeRecord(prompt=_message_prompt, response=_message_response)],
        None,
    )
    assert actual_value == expected_value
    actual_value = message_formatter(
        _system,
        [(_message_prompt, _message_response)],
        None,
    )
    assert actual_value == expected_value

    expected_value = PROMPT_FORMAT_LLAMA.format(prompt=_message_prompt) \
        + RESPONSE_FORMAT_LLAMA.format(response=_message_response)
    actual_value = message_formatter(
        None,
        [ExchangeRecord(prompt=_message_prompt, response=_message_response)],
        None,
    )
    assert actual_value == expected_value
    actual_value = message_formatter(
        '',
        [(_message_prompt, _message_response)],
        None,
    )
    assert actual_value == expected_value

    expected_value = SYSTEM_FORMAT_LLAMA.format(system_message=_system) \
        + PROMPT_FORMAT_LLAMA.format(prompt=_message_prompt) \
        + RESPONSE_FORMAT_LLAMA.format(response=_message_response) \
        + PROMPT_FORMAT_LLAMA.format(prompt=_message_prompt + '2') \
        + RESPONSE_FORMAT_LLAMA.format(response=_message_response + '2') \
        + PROMPT_FORMAT_LLAMA.format(prompt=_prompt)
    actual_value = message_formatter(
        _system,
        [
            ExchangeRecord(prompt=_message_prompt, response=_message_response),
            ExchangeRecord(prompt=_message_prompt + '2', response=_message_response + '2'),
        ],
        _prompt,
    )
    assert actual_value == expected_value
    actual_value = message_formatter(
        _system,
        [
            (_message_prompt, _message_response),
            (_message_prompt + '2', _message_response + '2'),
        ],
        _prompt,
    )
    assert actual_value == expected_value

def test_LlamaMessageFormatter():  # noqa
    assert LlamaMessageFormatter()(system_message=None, history=[], prompt=None) == ''
    assert LlamaMessageFormatter()(system_message=None, history=None, prompt=None) == ''

    messages = LlamaMessageFormatter()(
        system_message=None,
        history=[
            ExchangeRecord(prompt='a', response='b'),
            ExchangeRecord(prompt='c', response='d'),
        ],
        prompt=None,
    )
    expected_value = '[INST] a [/INST]\nb\n[INST] c [/INST]\nd\n'
    assert messages == '[INST] a [/INST]\nb\n[INST] c [/INST]\nd\n'

    messages = LlamaMessageFormatter()(
        system_message='system',
        history=[
            ExchangeRecord(prompt='a', response='b'),
            ExchangeRecord(prompt='c', response='d'),
        ],
        prompt=None,
    )
    assert messages == '[INST] <<SYS>> system <</SYS>> [/INST]\n' + expected_value

    messages = LlamaMessageFormatter()(
        system_message='system',
        history=[
            ExchangeRecord(prompt='a', response='b'),
            ExchangeRecord(prompt='c', response='d'),
        ],
        prompt='e',
    )
    assert messages == f'[INST] <<SYS>> system <</SYS>> [/INST]\n{expected_value}[INST] e [/INST]\n'  # noqa
