# pytest-llmeval

[![PyPI version](https://img.shields.io/pypi/v/pytest-llmeval.svg)](https://pypi.org/project/pytest-llmeval)
[![Python versions](https://img.shields.io/pypi/pyversions/pytest-llmeval.svg)](https://pypi.org/project/pytest-llmeval)
[![See Build Status on GitHub Actions](https://github.com/kevinschaul/pytest-llmeval/actions/workflows/main.yml/badge.svg)](https://github.com/kevinschaul/pytest-llmeval/actions/workflows/main.yml)

A pytest plugin to evaluate/benchmark LLM prompts

---

This [pytest](https://github.com/pytest-dev/pytest) plugin was generated with [Cookiecutter](https://github.com/audreyr/cookiecutter) along with [@hackebrot](https://github.com/hackebrot)'s [cookiecutter-pytest-plugin](https://github.com/pytest-dev/cookiecutter-pytest-plugin) template.

## Usage

See full usage examples in [examples/](examples/).

The main interface for this plugin is the `@pytest.mark.llmeval()` decorator, which injects an `llmeval_result` parameter into your test function.

### Basic Usage

You can run the same code cross multiple test cases by using pytest's [parametrize](https://docs.pytest.org/en/stable/example/parametrize.html) functionality.

```python
TEST_CASES = [
    {"input": "I need to debug this Python code", "expected": True},
    {"input": "The cat jumped over the lazy dog", "expected": False},
    {"input": "My monitor keeps flickering", "expected": True},
]

@pytest.mark.llmeval()
@pytest.mark.parametrize("test_case", TEST_CASES)
def test_computer_related(llmeval_result, test_case):

    # Run your llm code that returns a result for this test case
    result = llm_is_computer_related(test_case["input"])

    # Store the details on `llmeval_result`
    llmeval_result.set_result(
        input_data=test_case["input"],
        expected=test_case["expected"],
        actual=result,
    )

    # `assert` whether the actual result was the expected result
    assert llmeval_result.is_correct()
```

Run test like normal (with `uv run pytest` or similar) When the tests complete, a [classification report](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report) will be printed to stdout, in a format like:

```
# LLM Eval: test_computer_related

## Group: overall
              precision    recall  f1-score   support

        True       0.00      0.00      0.00         1
       False       0.67      1.00      0.80         2

    accuracy                           0.67         3
   macro avg       0.33      0.50      0.40         3
weighted avg       0.44      0.67      0.53         3
```

### Comparing different prompts

You can run compare different prompts or other variables by specifying `llmeval.set_result()`'s `group=` parameter:

```python
PROMPT_TEMPLATES = [
    f"Is this computer related? Say True or False",
    f"Say True or False: Is this computer related?",
]

TEST_CASES = [
    {"input": "I need to debug this Python code", "expected": True},
    {"input": "The cat jumped over the lazy dog", "expected": False},
    {"input": "My monitor keeps flickering", "expected": True},
]

@pytest.mark.llmeval()
@pytest.mark.parametrize("prompt_template", PROMPT_TEMPLATES)
@pytest.mark.parametrize("test_case", TEST_CASES)
def test_prompts(llmeval_result, prompt_template, test_case):
    result = llm_is_computer_related(test_case["input"])

    llmeval_result.set_result(
        input_data=test_case["input"],
        expected=test_case["expected"],
        actual=result,
        group=prompt_template,
    )
    assert llmeval_result.is_correct()
```

```
# LLM Eval: test_prompts

## Group: Is this computer related? Say True or False
              precision    recall  f1-score   support

       False       0.00      0.00      0.00         1
        True       0.67      1.00      0.80         2

    accuracy                           0.67         3
   macro avg       0.33      0.50      0.40         3
weighted avg       0.44      0.67      0.53         3


## Group: Say True or False: Is this computer related?
              precision    recall  f1-score   support

       False       0.33      1.00      0.50         1
        True       0.00      0.00      0.00         2

    accuracy                           0.33         3
   macro avg       0.17      0.50      0.25         3
weighted avg       0.11      0.33      0.17         3
```

### Saving reports

You can save evaluation results to a file by providing the `@pytest.mark.llmeval()` the `file_path` parameter:

```python
@pytest.mark.llmeval(file_path="results/test_prompts.txt")
def test_prompts(llmeval_result, prompt_template, test_case):
    # Your test code here
    pass
```

## API

TODO

### `@pytest.mark.llmeval()`

Marks a test function for evaluation. The test function will be passed the parameter `llmeval_result`.

## Installation

You can install "pytest-llmeval" via [pipx](https://pipx.pypa.io/stable/):

```
pipx install pytest-llmeval
```

## Contributing

Contributions are very welcome. Tests can be run with `uv run pytest`, please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT](https://opensource.org/licenses/MIT) license, "pytest-llmeval" is free and open source software

## Issues

If you encounter any problems, please [file an issue](https://github.com/kevinschaul/pytest-llmeval/issues) along with a detailed description.
