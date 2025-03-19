import textwrap
import os
import subprocess

from pytest import LineMatcher


def test_marker_registered(pytester):
    pytester.makepyfile(
        """
        import pytest
        
        @pytest.mark.llmeval
        def test_with_marker():
            assert True
    """
    )

    result = pytester.runpytest("--markers")
    result.stdout.fnmatch_lines(
        [
            "*@pytest.mark.llmeval*",
        ]
    )
    assert result.ret == 0


def test_result_fixture(pytester):
    pytester.makepyfile(
        """
        import pytest
        
        @pytest.mark.llmeval
        def test_with_result(llmeval_result):
            assert llmeval_result is not None
            llmeval_result.set_result("expected", "expected")
            assert llmeval_result.is_correct()
    """
    )

    result = pytester.runpytest("-v")
    result.stdout.fnmatch_lines(
        [
            "*::test_with_result LLMEVAL*",
        ]
    )
    assert result.ret == 0


def test_example_basic(pytester):
    pytester.copy_example("examples/test_example_basic.py")
    # https://docs.pytest.org/en/stable/reference/reference.html#pytest.RunResult
    run_result = pytester.runpytest("-k", "test_basic")

    # Check that the table is printed
    run_result.stdout.fnmatch_lines(
        [
            "# LLM Eval: test_example_basic.py::test_basic",
            "",
            "## Group: overall",
            "*precision*recall*f1-score*support",
        ],
        consecutive=True,
    )


EXAMPLES_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "../", "examples")
)


def test_example_prompts():
    """Should output the table and the files examples/output/test_example_prompts.txt"""
    example = "test_example_prompts.py"
    run_result = subprocess.run(
        ["pytest", example], cwd=EXAMPLES_DIR, capture_output=True, text=True
    )
    LineMatcher(run_result.stdout.split("\n")).fnmatch_lines(
        [
            "# LLM Eval: examples/test_example_prompts.py::test_prompts",
            "",
            "## Group: Is this computer related? Say True or False",
            "*precision*recall*f1-score*support",
        ],
        consecutive=True,
    )
    LineMatcher(run_result.stdout.split("\n")).fnmatch_lines(
        [
            "## Group: Say True or False: Is this computer related?",
            "*precision*recall*f1-score*support",
        ],
        consecutive=True,
    )

    output_file = os.path.join(EXAMPLES_DIR, "output", "test_example_prompts.txt")

    expected_contents = textwrap.dedent(
        """
        # LLM Eval: examples/test_example_prompts.py::test_prompts

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
    """
    ).strip()

    with open(output_file) as f:
        actual_contents = f.read().strip()
    assert actual_contents == expected_contents


def test_output_file(testdir, tmp_path):
    output_file = tmp_path / "results.txt"

    testdir.makepyfile(
        f"""
        import pytest
        import os
        
        @pytest.mark.llmeval(output_file="{output_file}")
        def test_with_output_file(llmeval_result):
            llmeval_result.set_result(True, True)
            assert True

        @pytest.mark.llmeval()
        def test_without_output_file(llmeval_result):
            llmeval_result.set_result(True, True)
            assert True
    """
    )

    result = testdir.runpytest("-v")
    result.stdout.fnmatch_lines(["*::test_with_output_file LLMEVAL*"])

    assert output_file.exists()

    expected_contents = textwrap.dedent(
        """
        # LLM Eval: test_output_file.py::test_with_output_file

        ## Group: overall
                      precision    recall  f1-score   support

                True       1.00      1.00      1.00         1

            accuracy                           1.00         1
           macro avg       1.00      1.00      1.00         1
        weighted avg       1.00      1.00      1.00         1
    """
    ).strip()

    with open(output_file) as f:
        actual_contents = f.read().strip()
    assert actual_contents == expected_contents

    if output_file.exists():
        output_file.unlink()


def test_analysis_func(pytester):
    pytester.makepyfile(
        """
        import pytest

        def custom_analysis_func(test_id, test_grouped):
            return ['hello from custom_analysis_func', str(len(test_grouped))]
        
        TEST_CASES = [
            {"input": "I need to debug this Python code", "expected": True},
            {"input": "The cat jumped over the lazy dog", "expected": False},
            {"input": "My monitor keeps flickering", "expected": True},
        ]

        @pytest.mark.llmeval(analysis_func=custom_analysis_func)
        @pytest.mark.parametrize("test_case", TEST_CASES)
        def test_with_result(llmeval_result, test_case):
            llmeval_result.set_result(
                input_data=test_case["input"],
                expected=test_case["expected"],
                actual=True,
            )
    """
    )

    result = pytester.runpytest("-v")
    result.stdout.fnmatch_lines(["hello from custom_analysis_func", "3"])
    assert result.ret == 0


def test_invalid_args(pytester):
    """Should raise error if output_file and analysis_func are both specified"""
    pytester.makepyfile(
        """
        import pytest
        @pytest.mark.llmeval(output_file="out.txt", analysis_func=lambda x, y: x)
        def test_with_result(llmeval_result):
            pass
    """
    )

    result = pytester.runpytest("-v")
    result.stdout.fnmatch_lines(["*LLMEvalException*"])
    assert result.ret != 0
