import pytest

PROMPT_TEMPLATES = [
    f"Is this computer related? Say True or False",
    f"Say True or False: Is this computer related?",
]

TEST_CASES = [
    {"input": {"text": "I need to debug this Python code"}, "expected": True},
    {"input": {"text": "The cat jumped over the lazy dog"}, "expected": False},
    {"input": {"text": "My monitor keeps flickering"}, "expected": True},
]


def is_computer_related(prompt_template, input):
    # Fake -- this would call an LLM and return the result.
    return True


# Example using the group parameter in set_result with output file
@pytest.mark.llmeval(output_file="./output/prompt_comparison.json")
@pytest.mark.parametrize("prompt_template", PROMPT_TEMPLATES)
@pytest.mark.parametrize("test_case", TEST_CASES)
def test_parametrized_prompt_template(llmeval_result, prompt_template, test_case):
    llmeval_result.set_result(
        expected=test_case["expected"],
        actual=is_computer_related(prompt_template, test_case["input"]["text"]),
        input_data=test_case["input"],
        group=prompt_template  # Use the group parameter to group results by prompt template
    )
    llmeval_result.add_metadata(prompt_template=prompt_template)
    assert llmeval_result.is_correct()
