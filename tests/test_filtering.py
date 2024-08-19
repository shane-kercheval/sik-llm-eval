"""Unit tests for the filtering module."""
from llm_eval.candidates import Candidate, CandidateResponse
from llm_eval.eval import Eval
from llm_eval.filtering import (
    eval_contains_code_block_tests,
    eval_expects_code_blocks,
    filter_contains_code_block_tests,
    filter_expects_code_blocks,
    filter_tags,
    matches_tags,
    result_contains_code_block_tests,
    result_expects_code_blocks,
)


class MockLMM:
    """Mock class representing an LLM."""

    def __init__(self, **kwargs: dict):
        self.llm_parameters = kwargs
        self.prompts = []

    def __call__(self, prompt: str) -> str:
        """Caches prompts for unit tests."""
        self.prompts.append(prompt)
        return prompt[-1]['content']


@Candidate.register('MOCK_MODEL_FILTERING')
class MockCandidate(Candidate):
    """Mock class representing a Candidate."""

    def __init__(self, **kwargs: dict) -> None:
        """Initialize a MockCandidate object."""
        metadata = kwargs.pop('metadata', None)
        parameters = kwargs.pop('parameters', None)
        super().__init__(parameters=parameters, metadata=metadata)
        self.model = None
        if parameters is not None:
            self.model = MockLMM(**parameters)
        else:
            self.model = MockLMM()

    def __call__(self, input: str) -> CandidateResponse:  # noqa: A002
        """Invokes the underlying model with the input and returns the response."""
        return CandidateResponse(content=self.model(input))

def test_filter_tags(  # noqa: PLR0915
        fake_eval_8f9fbf37: dict,
        fake_eval_subtract_two_numbers: dict,
        fake_eval_sum_two_numbers: dict,
        fake_eval_sum_two_numbers_code_blocks_run: dict) -> None:
    """Test the filtering function."""
    candidate = MockCandidate()
    results = []
    results.append(Eval(**fake_eval_8f9fbf37)(candidate))
    results.append(Eval(**fake_eval_subtract_two_numbers)(candidate))
    results.append(Eval(**fake_eval_sum_two_numbers)(candidate))
    results.append(Eval(**fake_eval_sum_two_numbers_code_blocks_run)(candidate))

    assert 'eval_8f9' in results[0].eval_obj.metadata['tags']
    assert 'subtract_two_numbers' in results[1].eval_obj.metadata['tags']
    assert 'sum_two_numbers' in results[2].eval_obj.metadata['tags']
    assert 'sum_two_numbers' in results[3].eval_obj.metadata['tags']
    assert 'code_block' in results[3].eval_obj.metadata['tags']

    # using `is True` and `is False` to ensure that the return value is a boolean
    assert matches_tags(results[0]) is True
    assert matches_tags(results[0], include='eval_8f9') is True
    assert matches_tags(results[0], exclude='eval_8f9') is False
    assert matches_tags(results[3]) is True
    assert matches_tags(results[3], include='sum_two_numbers') is True
    assert matches_tags(results[3], exclude='sum_two_numbers') is False
    assert matches_tags(results[3], include='sum_two_numbers', exclude='sum_two_numbers') is False
    assert matches_tags(results[3], include=['sum_two_numbers', 'code_block']) is True
    assert matches_tags(results[3], include='sum_two_numbers', exclude='code_block') is False
    assert matches_tags(results[3], include=['sum_two_numbers', 'code_block'], exclude='python') is False  # noqa
    assert matches_tags(results[3], include=['sum_two_numbers', 'code_block'], exclude=['python']) is False  # noqa

    filtered_results = filter_tags(results)
    assert len(filtered_results) == 4
    assert filtered_results == results

    filtered_results = filter_tags(results, exclude='eval_8f9')
    assert len(filtered_results) == 3
    assert results[0] not in filtered_results
    assert results[1] in filtered_results
    assert results[2] in filtered_results
    assert results[3] in filtered_results

    filtered_results = filter_tags(results, include='eval_8f9')
    assert len(filtered_results) == 1
    assert results[0] in filtered_results
    assert results[1] not in filtered_results
    assert results[2] not in filtered_results
    assert results[3] not in filtered_results

    filtered_results = filter_tags(results, include='sum_two_numbers')
    assert len(filtered_results) == 2
    assert results[0] not in filtered_results
    assert results[1] not in filtered_results
    assert results[2] in filtered_results
    assert results[3] in filtered_results

    filtered_results = filter_tags(results, exclude='sum_two_numbers')
    assert len(filtered_results) == 2
    assert results[0] in filtered_results
    assert results[1] in filtered_results
    assert results[2] not in filtered_results
    assert results[3] not in filtered_results

    filtered_results = filter_tags(results, include='sum_two_numbers', exclude='sum_two_numbers')
    assert len(filtered_results) == 0

    filtered_results = filter_tags(results, include=['eval_8f9', 'subtract_two_numbers'])
    assert len(filtered_results) == 2
    assert results[0] in filtered_results
    assert results[1] in filtered_results
    assert results[2] not in filtered_results
    assert results[3] not in filtered_results

    filtered_results = filter_tags(results, include='sum_two_numbers', exclude='code_block')
    assert len(filtered_results) == 1
    assert results[0] not in filtered_results
    assert results[1] not in filtered_results
    assert results[2] in filtered_results
    assert results[3] not in filtered_results

def test__expects_code_blocks(
        fake_eval_8f9fbf37: dict,
        fake_eval_subtract_two_numbers: dict,
        fake_eval_sum_two_numbers: dict,
        fake_eval_sum_two_numbers_code_blocks_run: dict,
        fake_eval_no_code_blocks: dict) -> None:
    """Test the xxx_expects_code_blocks functions."""
    assert eval_expects_code_blocks(Eval(**fake_eval_8f9fbf37))
    assert eval_expects_code_blocks(Eval(**fake_eval_subtract_two_numbers))
    assert eval_expects_code_blocks(Eval(**fake_eval_sum_two_numbers))
    assert eval_expects_code_blocks(Eval(**fake_eval_sum_two_numbers_code_blocks_run))
    assert not eval_expects_code_blocks(Eval(**fake_eval_no_code_blocks))

    candidate = MockCandidate()
    results = []
    results.append(Eval(**fake_eval_8f9fbf37)(candidate))
    results.append(Eval(**fake_eval_subtract_two_numbers)(candidate))
    results.append(Eval(**fake_eval_sum_two_numbers)(candidate))
    results.append(Eval(**fake_eval_sum_two_numbers_code_blocks_run)(candidate))
    results.append(Eval(**fake_eval_no_code_blocks)(candidate))

    assert result_expects_code_blocks(results[0])
    assert result_expects_code_blocks(results[1])
    assert result_expects_code_blocks(results[2])
    assert result_expects_code_blocks(results[3])
    assert not result_expects_code_blocks(results[4])

    filtered_results = filter_expects_code_blocks(results)
    assert len(filtered_results) == 4
    assert results[0] in filtered_results
    assert results[1] in filtered_results
    assert results[2] in filtered_results
    assert results[3] in filtered_results
    assert results[4] not in filtered_results

def test__contains_code_block_tests(
        fake_eval_8f9fbf37: dict,
        fake_eval_subtract_two_numbers: dict,
        fake_eval_sum_two_numbers: dict,
        fake_eval_sum_two_numbers_code_blocks_run: dict,
        fake_eval_no_code_blocks: dict) -> None:
    """Test the xxx_contains_code_block_tests functions."""
    assert not eval_contains_code_block_tests(Eval(**fake_eval_8f9fbf37))
    assert not eval_contains_code_block_tests(Eval(**fake_eval_subtract_two_numbers))
    assert not eval_contains_code_block_tests(Eval(**fake_eval_sum_two_numbers))
    assert eval_contains_code_block_tests(Eval(**fake_eval_sum_two_numbers_code_blocks_run))
    assert not eval_contains_code_block_tests(Eval(**fake_eval_no_code_blocks))

    candidate = MockCandidate()
    results = []
    results.append(Eval(**fake_eval_8f9fbf37)(candidate))
    results.append(Eval(**fake_eval_subtract_two_numbers)(candidate))
    results.append(Eval(**fake_eval_sum_two_numbers)(candidate))
    results.append(Eval(**fake_eval_sum_two_numbers_code_blocks_run)(candidate))
    results.append(Eval(**fake_eval_no_code_blocks)(candidate))

    assert not result_contains_code_block_tests(results[0])
    assert not result_contains_code_block_tests(results[1])
    assert not result_contains_code_block_tests(results[2])
    assert result_contains_code_block_tests(results[3])
    assert not result_contains_code_block_tests(results[4])

    filtered_results = filter_contains_code_block_tests(results)
    assert len(filtered_results) == 1
    assert results[0] not in filtered_results
    assert results[1] not in filtered_results
    assert results[2] not in filtered_results
    assert results[3] in filtered_results
    assert results[4] not in filtered_results
