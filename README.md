# pytest-llmeval

[![PyPI version](https://img.shields.io/pypi/v/pytest-llmeval.svg)](https://pypi.org/project/pytest-llmeval)
[![Python versions](https://img.shields.io/pypi/pyversions/pytest-llmeval.svg)](https://pypi.org/project/pytest-llmeval)
[![See Build Status on GitHub Actions](https://github.com/kevinschaul/pytest-llmeval/actions/workflows/main.yml/badge.svg)](https://github.com/kevinschaul/pytest-llmeval/actions/workflows/main.yml)

A pytest plugin to evaluate/benchmark LLM prompts

---

This [pytest](https://github.com/pytest-dev/pytest) plugin was generated with [Cookiecutter](https://github.com/audreyr/cookiecutter) along with [@hackebrot](https://github.com/hackebrot)'s [cookiecutter-pytest-plugin](https://github.com/pytest-dev/cookiecutter-pytest-plugin) template.

## Usage

See full usage examples in [example/](example/).

The basic usage is:

1. Mark test with the decorator `@pytest.mark.llmeval`, storing the test case details on `llmeval_result`:

```
@pytest.mark.llmeval
def test_llm_dog_or_cat(llmeval_result):
    # Basic example, but these can be parametrized
    test_case = {
        "input": "Tony the Tiger",
        "expected": "cat",
    }

    # Call the llm
    result = llm_dog_or_cat(test_case["input"])

    # Store the details on `llmeval_result`
    llmeval_result.set_result(
        input_data=test_case["input"]
        expected=test_case["expected"],
        actual=result,
    )
    assert llmeval_result.is_correct()
```

2. Run test like normal (with `uv run pytest` or similar)

3. When the tests complete, a [classification report](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report) will be printed to stdout, in a format like:

```
======================= LLM Evaluation Results ========================

Test: test_llm_dog_or_cat
Number of test cases: 400

Classification Report:
              precision    recall  f1-score   support

       False       0.74      0.81      0.78       200
        True       0.79      0.72      0.75       200

    accuracy                           0.77       400
   macro avg       0.77      0.77      0.76       400
weighted avg       0.77      0.77      0.76       400

==================== End of LLM Evaluation Results ====================
```

## Installation

You can install "pytest-llmeval" via [pip](https://pypi.org/project/pip/) from [PyPI](https://pypi.org/project):

```
$ pip install pytest-llmeval
```

## Contributing

Contributions are very welcome. Tests can be run with `uv run pytest`, please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT](https://opensource.org/licenses/MIT) license, "pytest-llmeval" is free and open source software

## Issues

If you encounter any problems, please [file an issue](https://github.com/kevinschaul/pytest-llmeval/issues) along with a detailed description.
