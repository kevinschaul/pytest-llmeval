# pytest-llmeval

[![PyPI version](https://img.shields.io/pypi/v/pytest-llmeval.svg)](https://pypi.org/project/pytest-llmeval)
[![Python versions](https://img.shields.io/pypi/pyversions/pytest-llmeval.svg)](https://pypi.org/project/pytest-llmeval)
[![See Build Status on GitHub Actions](https://github.com/kevinschaul/pytest-llmeval/actions/workflows/main.yml/badge.svg)](https://github.com/kevinschaul/pytest-llmeval/actions/workflows/main.yml)

A pytest plugin to evaluate/benchmark LLM prompts

## Features

- **Simple interface**: Just mark which tests are LLM evals and store the results
- **Evaluation metrics**: Get comprehensive classification metrics including precision, recall, and F1 scores
- **Grouped evaluations**: Compare how different prompts or models perform acorss your test cases
- **File export**: Save evaluation reports to file for monitoring performance changes over time
- **Custom analysis function**: Write your own analysis function if you prefer
- **Pytest integration**: Evaluations fit right in with your project's other tests

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

### Comparing across variables like different prompts or models

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

The test report would be saved to "results/test_prompts.txt".

### Custom analysis functions

If you prefer to do a different analysis across the results, pass a function with the `analysis_func` parameter:

```python
def my_analysis(test_id, results):
    print(f"My custom analysis function processed {len(results)} results")

@pytest.mark.llmeval(analysis_func=my_analyis)
def test_prompts(llmeval_result, prompt_template, test_case):
    # Your test code here
    pass
```

## API

### `@pytest.mark.llmeval()`

Marks a test function for evaluation. The test function will be passed the parameter `llmeval_result`.

**Parameters**:

- `file_path` (str, optional): Path where the evaluation report will be saved. If not provided, the report will only be displayed in the test output.

- `analysis_func(test_id: str, results: ClassificationResult[]) -> str[]` (function, optional): A custom analysis function to run across all results. Do whatever calculations you want in here. Optionally return a list of strings to be printed to stdout.

**Injected parameters**:

- `llmeval_result`: An object to track test evaluation results with the following methods:

  - `set_result(expected: str, actual: str, input_data: str | dict, group?: str)`: Record the details of this test result
  - `is_correct() -> bool`: Returns whether the expected result equals the actual result

### `ClassificationResult`

- `expected`: The expected result
- `actual`: The actual result
- `input`: Input data used for this test case
- `group` (optional): An optional variable to group by before running analyses. E.g. pass a prompt to group results by prompt

## Installation

You can install "pytest-llmeval" via [pipx](https://pipx.pypa.io/stable/):

```
pipx install pytest-llmeval
```

## Contributing

Contributions are very welcome. Tests can be run with `uv run pytest`, please ensure
the coverage at least stays the same before you submit a pull request.

This [pytest](https://github.com/pytest-dev/pytest) plugin was generated with [Cookiecutter](https://github.com/audreyr/cookiecutter) along with [@hackebrot](https://github.com/hackebrot)'s [cookiecutter-pytest-plugin](https://github.com/pytest-dev/cookiecutter-pytest-plugin) template.

## License

Distributed under the terms of the [MIT](https://opensource.org/licenses/MIT) license, "pytest-llmeval" is free and open source software

## Issues

If you encounter any problems, please [file an issue](https://github.com/kevinschaul/pytest-llmeval/issues) along with a detailed description.
