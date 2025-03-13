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


def llm_is_computer_related(prompt_template, input):
    # Fake -- this would call an LLM and return the result.
    return prompt_template.startswith('Is this')


@pytest.mark.llmeval(output_file="./output/test_example_prompts.txt")
@pytest.mark.parametrize("prompt_template", PROMPT_TEMPLATES)
@pytest.mark.parametrize("test_case", TEST_CASES)
def test_prompts(llmeval_result, prompt_template, test_case):
    llmeval_result.set_result(
        expected=test_case["expected"],
        actual=llm_is_computer_related(prompt_template, test_case["input"]["text"]),
        input_data=test_case["input"],
        group=prompt_template,
    )
    assert llmeval_result.is_correct()
