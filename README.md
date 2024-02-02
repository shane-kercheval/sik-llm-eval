# llm-eval

`llm-eval` is a framework for evaluating the quality of responses from LLMs (e.g. the number of code blocks generated that successfully run), as well as the performance (e.g. characters generated per second).

There are two key concepts in this framework.

- `Eval`: An Eval is single scenario (i.e. one or more prompts) that the user is interested in evaluating. Multiple prompts can be used to test conversations (i.e. sequential prompt/response exchanges where the assumption is the LLM client maintains conversational history). Additionally, each Eval is associated with custom "checks". Examples of checks are: if the response matches an exact value, if it contain a particular value, if it contain code blocks, if those code blocks run, if the variables/functions/etc created by those code blocks contain the expected values/behavior.
- `Candidate`: A Candidate encapsulates the underlying LLM and corresponding client the user is interested in evaluating the prompts (Evals) against. Examples of candidates are ChatGPT 4.0 (LLM & client/API are synonymous), Llama-2-7b-Chat (LLM) running on Hugging Face Endpoints with Nvidia 10G (client), Llama-2-7b-Chat Q6_K.gguf (LLM) running locally on LM Studio (client). The latter two are examples of the same underlying model running on different hardware. They are likely to have very similar quality of responses (but this is also determined by the quantization) but may have very different performance (e.g. characters per second).

# Using `llm-eval`

## Examples

### Loading Evals and Candidates from yaml files

The following yaml file (found in `examples/candidates/openai_3.5_1106.yaml`) defines a ChatGPT Candidate. This file specifies that we want to use `gpt-3.5-turbo-1106` as well as other model parameters such as the system message and temperature. The `candidate_type: OPENAI` entry allows the Candidate registry to create an instance of the OpenAICandidate class, and forwards the model parameters to OpenAI. A similar file for ChatGPT 4 can be found in `examples/candidates/openai_4.0_1106.yaml`.

```yaml
metadata:
  name: OpenAI GPT-3.5-Turbo (1106)
candidate_type: OPENAI
model_parameters:
  model_name: gpt-3.5-turbo-1106
  system_message: You are a helpful AI assistant.
  temperature: 0.01
  max_tokens: 4096
  seed: 42
```

The following yaml file (found in `examples/evals/simple_example.yaml`) defines an Eval. For this Eval, the goal is to have the LLM create a function that generates the Fibonacci sequence, and subsequentally generate assertion statements that test the function. Each prompt is associated with multiple checks that are ran against the corresponding responses from the LLM. 

```yaml
metadata:
  name: Fibonacci Sequence
test_sequence:
  - prompt: Create a python function called `fib` that takes an integer `n` and returns the `n`th number in the Fibonacci sequence. Use type hints and docstrings.
    checks:
      - check_type: REGEX
        pattern: "def fib\\([a-zA-Z_]+\\: int\\) -> int\\:"
      - check_type: PYTHON_CODE_BLOCKS_PRESENT
  - prompt: Create a set of assertion statements that test the function.
    checks:
      - check_type: CONTAINS
        value: assert fib(
      - check_type: PYTHON_CODE_BLOCKS_PRESENT
      - check_type: PYTHON_CODE_BLOCKS_RUN
        code_setup: |
          import re
        functions:
          - |
            def verify_mask_emails_with_no_email_returns_original_string(code_blocks: list[str]) -> bool:
                value = 'This is a string with no email addresses'
                return mask_emails(value) == value
```

Additionally, the Eval above defines a `PYTHON_CODE_BLOCKS_RUN` check, which runs the code blocks (generated from the LLM) in an isolated environment and tracks the number of code blocks that run successfully. Additionally, the user can define one or more functions (which return boolean values indicating success/failure) that are ran in the same environment and can test variables or functions that are created from the code blocks. In the example above, the function tests that `mask_emails` function (expected to be defined in the code block in the LLM response and created in the isolated enviroment) returns the original value if no emails are present.

The following code loads in the Candidate and Eval above (along with a ChatGPT 4 Candidate and additional Eval) and runs all of the Evals against each Candidate.

```python
eval_harness = EvalHarness(callback=print_result)
eval_harness.add_eval_from_yaml('examples/evals/simple_example.yaml')
eval_harness.add_eval_from_yaml('examples/evals/mask_emails.yaml')
eval_harness.add_candidate_from_yaml('examples/candidates/openai_3.5_1106.yaml')
eval_harness.add_candidate_from_yaml('examples/candidates/openai_4.0_1106.yaml')
results = eval_harness()
print(results[0][0])
```

Also note that we could have loaded in all of the yaml files within a directory via `add_evals_from_yamls` and `add_candidates_from_yamls`.

```python
eval_harness = EvalHarness(callback=print_result)
eval_harness.add_evals_from_yamls('examples/evals/*.yaml')
eval_harness.add_candidate_from_yamls('examples/candidates/*.yaml')
results = eval_harness()
print(results[0][0])
```

`results` contains a list of lists of EvalResults. Each item in the outer list corresponds to a single candidate and contains a list of EvalResults for all Evals ran against the Candidate. In our example, `results` is `[[EvalResult, EvalResult], [EvalResult, EvalResult]]` where the first list corresponds to results of the Evals associated with the first Candidate (ChatGPT 3.5) and the second list corresponds to results of the Evals associated with the second Candidate (ChatGPT 4.0).

`print(results[0][0])` will give:


```
EvalResult:
    Candidate:                  OpenAI GPT-3.5-Turbo (1106)
    Eval:                       Fibonacci Sequence
    # of Prompts Tested:        2
    Cost:                       $0.0011
    Total Response Time:        14.7 seconds
    # of Response Characters:   1,336
    # of Code Blocks Generated: 2
    Characters per Second:      90.9
    # of Checks:                4
    # of Successful Checks:     4
    % of Successful Checks:     100.0%
```

This is merely the string representation of the EvalResult object. The object itself will contain additional information associated with each individual check, allowing the user to understand the results at a deeper level.


### From python objects

The same example from above could be ran using dictionary (or Candidate/Eval objects) already loadded into memory.

```python
candidate_chatgpt_35 = {"metadata": {"name": "OpenAI GPT-3.5-Turbo (1106)" ... }
candidate_chatgpt_40 = {"metadata": {"name": "OpenAI GPT-4.0-Turbo (1106)" ... }
eval_simple = {metadata: ..., test_sequence: ...}
eval_mask_email = {metadata: ..., test_sequence: ...}
eval_harness = EvalHarness(callback=print_result)
eval_harness.add_eval(eval_simple)
eval_harness.add_eval(eval_mask_email)
eval_harness.add_candidate(candidate_chatgpt_35)
eval_harness.add_candidate(candidate_chatgpt_40)
results = eval_harness()
print(results[0][0])
```

### Executing a single Eval against a single Candidate

If you are only interested in evaluating a particular Eval against a particular Candidate, you can create both objects and pass the Candidate object to the Eval object (which is callable).

```python
candidate = OpenAICandidate({'model_parameters': {'model_name': 'gpt-3.5-turbo-1106'}})
eval_obj = Eval(test_sequence={'prompt': "Create a python function called `fib` that takes an integer `n` and returns the `n`th number in the Fibonacci sequence. Use type hints and docstrings."})
result = eval_obj(candidate)
print(result)
```

which gives:

```
EvalResult:
    # of Prompts Tested:        1
    Cost:                       $0.0005
    Total Response Time:        4.4 seconds
    # of Response Characters:   734
    # of Code Blocks Generated: 1
    Characters per Second:      168.3
    # of Checks:                0
    # of Successful Checks:     0
```

This minimal example does not give much insight into the quality of the response, but could still be used to compare the response time and cost of, for example, ChatGPT 3.5 vs 4.0, as well as visually compare the responses.

## Installing

Currently, the easiest way to install the `llm-eval` package is 

## Environment Variables

- `OPENAI_API_KEY` environment variable and api key is required to use `OpenAIChat`
- `HUGGING_FACE_API_KEY` environment variable and api key is required to use `HuggingFaceEndpointChat`

# Examples

TODO

# Development Setup

## Environment Variables / API Keys

The following environment variables need to be set (e.g. via `.env` file) if the corresponding services are used: 

- `HUGGING_FACE_API_KEY`
- `OPENAI_API_KEY`

In order to test `HuggingFaceEndpointChat` in `llm_evals/llms/hugging_face.py` (via `tests/test_hugging_face.py`) set the `HUGGING_FACE_ENDPOINT_UNIT_TESTS` environment variable (e.g. via `.env` file) to a deployed model on [Hugging Face Endpoints](https://huggingface.co/inference-endpoints)
