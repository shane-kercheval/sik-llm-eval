# llm-eval

Framework for evaluating performance and quality of responses from various LLMs.


# Setup

# Environment Variables / API Keys

The following environment variables need to be set (e.g. via `.env` file) if the corresponding services are used: 

- `HUGGING_FACE_API_KEY`
- `OPENAI_API_KEY`

In order to test `HuggingFaceEndpointChat` in `llm_evals/llms/hugging_face.py` (via `tests/test_hugging_face.py`) set the `HUGGING_FACE_ENDPOINT_UNIT_TESTS` environment variable (e.g. via `.env` file) to a deployed model on [Hugging Face Endpoints](https://huggingface.co/inference-endpoints)




# NOTES

- perhaps `Scenario` can have system_message which would override system_message provided by Candidate?




- Botht the eval yaml and the candidate yaml are duplicated in results yaml. This seems useful for
comparing versions/archiving/etc. You don't have to retain the exact eval to know what was tested for a particular result