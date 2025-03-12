import pytest

TEST_CASES = [
    {"input": {"text": "I need to debug this Python code"}, "expected": True},
    {"input": {"text": "The cat jumped over the lazy dog"}, "expected": False},
    {"input": {"text": "My monitor keeps flickering"}, "expected": True},
]


def is_computer_related(input):
    # Fake -- this would call an LLM and return the result.
    return True


@pytest.mark.llmeval(output_file='./output/test_save_to_file.csv')
@pytest.mark.parametrize("test_case", TEST_CASES)
def test_save_to_file_csv(llmeval_result, test_case):
    llmeval_result.set_result(
        expected=test_case["expected"],
        actual=is_computer_related(test_case["input"]["text"]),
        input_data=test_case["input"],
    )
    assert llmeval_result.is_correct()

@pytest.mark.llmeval(output_file='./output/test_save_to_file.txt')
@pytest.mark.parametrize("test_case", TEST_CASES)
def test_save_to_file_txt(llmeval_result, test_case):
    llmeval_result.set_result(
        expected=test_case["expected"],
        actual=is_computer_related(test_case["input"]["text"]),
        input_data=test_case["input"],
    )
    assert llmeval_result.is_correct()
