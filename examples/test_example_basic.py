import pytest

TEST_CASES = [
    {"input": {"text": "I need to debug this Python code"}, "expected": True},
    {"input": {"text": "The cat jumped over the lazy dog"}, "expected": False},
    {"input": {"text": "My monitor keeps flickering"}, "expected": True},
]


def is_computer_related(input):
    # Fake -- this would call an LLM and return the result.
    return True


@pytest.mark.llmeval()
@pytest.mark.parametrize("test_case", TEST_CASES)
def test_something(llmeval_result, test_case):
    # `llmeval_result` is provided by `@pytest.mark.llmeval()`
    # `test_case` is provided by `@pytest.mark.parametrize("test_case", TEST_CASES)`
    # https://docs.pytest.org/en/stable/how-to/parametrize.html

    # Run the actual test
    actual_result = is_computer_related(test_case["input"]["text"])

    # Record the result
    llmeval_result.set_result(
        expected=test_case["expected"],
        actual=actual_result,
        input_data=test_case["input"],
    )

    # Regular pytest assertion
    assert llmeval_result.is_correct()
