import pytest
import csv
import os
import io
import textwrap

from pytest_llmeval.plugin import (
    format_report_as_text,
    format_report_as_csv,
)


# Sample data for testing
@pytest.fixture
def sample_report_data():
    return {
        "test_name": "test_llm_function",
        "num_cases": 3,
        "report_dict": {
            "False": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 1.0},
            "True": {"precision": 0.67, "recall": 1.0, "f1-score": 0.8, "support": 2.0},
            "accuracy": 0.67,
            "macro avg": {
                "precision": 0.33,
                "recall": 0.5,
                "f1-score": 0.4,
                "support": 3.0,
            },
            "weighted avg": {
                "precision": 0.44,
                "recall": 0.67,
                "f1-score": 0.53,
                "support": 3.0,
            },
        },
    }


class TestLLMEvalBase:
    def test_llmeval_marker_registered(self, pytester):
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

    def test_llmeval_result_fixture(self, pytester):
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

    def test_llmeval_add_metadata(self, pytester):
        pytester.makepyfile(
            """
            import pytest
            
            @pytest.mark.llmeval()
            def test_with_metadata(llmeval_result):
                llmeval_result.add_metadata(difficulty="easy")
                assert llmeval_result.metadata["difficulty"] == "easy"
        """
        )

        result = pytester.runpytest("-v")
        result.stdout.fnmatch_lines(
            [
                "*::test_with_metadata LLMEVAL*",
            ]
        )
        assert result.ret == 0


class TestLLMEvalFormat:
    def test_format_report_as_text(self, sample_report_data):
        text_data = format_report_as_text(
            sample_report_data["test_name"],
            sample_report_data["num_cases"],
            sample_report_data["report_dict"],
        )

        expected = textwrap.dedent(
            """
            Test: test_llm_function
            Class        |  Precision |     Recall |   F1-Score |    Support
            ----------------------------------------------------------------
            False        |       0.00 |       0.00 |       0.00 |          1
            True         |       0.67 |       1.00 |       0.80 |          2
            accuracy     |            |            |       0.67 |          3
            macro avg    |       0.33 |       0.50 |       0.40 |          3
            weighted avg |       0.44 |       0.67 |       0.53 |          3
            """
        ).strip()

        assert text_data == expected

    def test_format_report_as_csv(self, sample_report_data):
        csv_rows = format_report_as_csv(
            sample_report_data["test_name"],
            sample_report_data["num_cases"],
            sample_report_data["report_dict"],
        )

        expected = textwrap.dedent(
            """
            test_name,group,class,precision,recall,f1-score,support
            test_llm_function,overall,False,0.0,0.0,0.0,1.0
            test_llm_function,overall,True,0.67,1.0,0.8,2.0
            test_llm_function,overall,accuracy,,,0.67,3
            test_llm_function,overall,macro avg,0.33,0.5,0.4,3.0
            test_llm_function,overall,weighted avg,0.44,0.67,0.53,3.0
            """
        ).strip()

        output = io.StringIO()
        writer = csv.writer(output)
        for row in csv_rows:
            writer.writerow(row)
        csv_text = output.getvalue().strip()

        # Normalize line endings
        normalized_expected = expected.replace('\r\n', '\n').replace('\r', '\n')
        normalized_actual = csv_text.replace('\r\n', '\n').replace('\r', '\n')
        assert normalized_actual == normalized_expected


class TestLLMEvalSave:
    def test_marker_output_file(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            
            @pytest.mark.llmeval(output_file="results.csv")
            def test_with_output_file(llmeval_result):
                llmeval_result.set_result(True, True)
                assert True
            
            def test_without_marker():
                assert True
        """
        )

        testdir.runpytest("-v")

        assert os.path.exists(os.path.join(testdir.tmpdir, "results.csv"))
