# llm-eval

`llm-eval` is a framework for evaluating the performance (e.g. characters generated per second) and quality (e.g. number of code blocks generated that successfully run) of responses from various LLMs.

There are two key concepts in this framework.

- `Eval`: A Eval is single scenario (i.e. one or more prompts) that the user is interested in evaluating. Multiple prompts can be used to test conversations (i.e. multiple prompt/response exchanges where the assumption is the LLM client maintains conversational history). Additionally, each Eval is associated with custom "checks". Examples of checks are: if the response matches an exact value, if it contain a particular value, if it contain code blocks, if those code blocks run, if the variables/functions/etc created by those code blocks contain the expected values/behavior.
- `Candidate`: A Candidate encapsulates the underlying LLM and corresponding client the user is interested in evaluating the prompts (Evals) against. Examples of candidates are ChatGPT 4.0 (LLM & client/API are synonymous), Llama-2-7b-Chat (LLM) running on Hugging Face Endpoints with Nvidia 10G (client), Llama-2-7b-Chat Q6_K.gguf (LLM) running locally on LM Studio (client). The latter two are examples of the same underlying model running on different hardware. They are likely to have very similar quality of responses (but this is also determined by the quantization) but may have very different performance (e.g. characters per second).

# Using `llm-eval`

## Examples

### From a yaml files 

```yaml

```

### From python object

```python

```

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
