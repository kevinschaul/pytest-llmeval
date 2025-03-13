import pytest
import csv
import io
import textwrap

from pytest_llmeval.plugin import (
    ClassificationResult,
    format_report_as_text,
    format_report_as_csv,
    generate_classification_reports,
)

prompt1 = "Is this computer related? Say True or False"
prompt2 = "Say True or False: Is this computer related?"

@pytest.fixture
def test_results_groups_none():
    return {
        "test_filename.py::test_one[test_case0]": ClassificationResult(
            expected=True,
            actual=True,
            input_data={"text": "I need to debug this Python code"},
        ),
        "test_filename.py::test_one[test_case1]": ClassificationResult(
            expected=False,
            actual=True,
            input_data={"text": "The cat jumped over the lazy dog"},
        ),
        "test_filename.py::test_one[test_case2]": ClassificationResult(
            expected=True,
            actual=True,
            input_data={"text": "My monitor keeps flickering"},
        ),
    }


@pytest.fixture
def test_results_groups_two():
    return {
        f"test_filename.py::test_one[test_case0-{prompt1}]": ClassificationResult(
            expected=True,
            actual=True,
            input_data={"text": "I need to debug this Python code"},
            group=prompt1,
        ),
        f"test_filename.py::test_one[test_case1-{prompt1}]": ClassificationResult(
            expected=False,
            actual=True,
            input_data={"text": "The cat jumped over the lazy dog"},
            group=prompt1,
        ),
        f"test_filename.py::test_one[test_case2-{prompt1}]": ClassificationResult(
            expected=True,
            actual=True,
            input_data={"text": "My monitor keeps flickering"},
            group=prompt1,
        ),
        f"test_filename.py::test_one[test_case0-{prompt2}]": ClassificationResult(
            expected=True,
            actual=True,
            input_data={"text": "I need to debug this Python code"},
            group=prompt2,
        ),
        f"test_filename.py::test_one[test_case1-{prompt2}]": ClassificationResult(
            expected=False,
            actual=True,
            input_data={"text": "The cat jumped over the lazy dog"},
            group=prompt2,
        ),
        f"test_filename.py::test_one[test_case2-{prompt2}]": ClassificationResult(
            expected=True,
            actual=True,
            input_data={"text": "My monitor keeps flickering"},
            group=prompt2,
        ),
    }


@pytest.fixture
def sample_groups_one():
    return [
        {
            "test_name": "test_name",
            "group": "overall",
            "class": "False",
            "sample_size": 1,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
        {
            "test_name": "test_name",
            "group": "overall",
            "class": "True",
            "sample_size": 2,
            "accuracy": 0.67,
            "precision": 0.67,
            "recall": 1.0,
            "f1": 0.8,
        },
    ]


@pytest.fixture
def sample_groups_two():
    """Fixture with two different prompt groups for comparison."""
    return [
        {
            "test_name": "test_name",
            "group": prompt1,
            "class": "False",
            "sample_size": 1,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
        {
            "test_name": "test_name",
            "group": prompt1,
            "class": "True",
            "sample_size": 2,
            "accuracy": 0.67,
            "precision": 0.67,
            "recall": 1.0,
            "f1": 0.8,
        },
        {
            "test_name": "test_name",
            "group": prompt2,
            "class": "False",
            "sample_size": 1,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
        {
            "test_name": "test_name",
            "group": prompt2,
            "class": "True",
            "sample_size": 2,
            "accuracy": 0.67,
            "precision": 0.67,
            "recall": 1.0,
            "f1": 0.8,
        },
    ]


class TestLLMEvalPytest:
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


class TestLLMEvalAnalyze:
    def test_generate_classification_reports_groups_none(
        self, test_results_groups_none
    ):
        actual = generate_classification_reports(test_results_groups_none)
        test_one_results = actual.get("test_one", [])

        assert len(test_one_results) == 2

        true_report = next((r for r in test_one_results if r["class"] == "True"), None)
        assert true_report is not None
        assert true_report["test_name"] == "test_one"
        assert true_report["group"] == "overall"
        assert true_report["sample_size"] == 2
        assert true_report["accuracy"] == 0.67
        assert true_report["precision"] == 0.67
        assert true_report["recall"] == 1.0
        assert true_report["f1"] == 0.80

        false_report = next(
            (r for r in test_one_results if r["class"] == "False"), None
        )
        assert false_report is not None
        assert false_report["test_name"] == "test_one"
        assert false_report["group"] == "overall"
        assert false_report["sample_size"] == 1
        assert false_report["accuracy"] == 0.0
        assert false_report["precision"] == 0.0
        assert false_report["recall"] == 0.0
        assert false_report["f1"] == 0.0

    def test_sklearn_report_comparison(self, test_results_groups_none):
        """Test that our metrics match sklearn.metrics.classification_report."""
        from sklearn.metrics import classification_report

        y_true = []
        y_pred = []

        for result in test_results_groups_none.values():
            y_true.append(str(result.expected))
            y_pred.append(str(result.actual))

        sklearn_report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )

        our_report = generate_classification_reports(test_results_groups_none)
        test_one_results = our_report.get("test_one", [])

        true_sklearn = sklearn_report.get("True", {})
        true_ours = next((r for r in test_one_results if r["class"] == "True"), None)

        assert true_ours is not None
        assert true_ours["precision"] == round(true_sklearn["precision"], 2)
        assert true_ours["recall"] == round(true_sklearn["recall"], 2)
        assert true_ours["f1"] == round(true_sklearn["f1-score"], 2)
        assert true_ours["sample_size"] == true_sklearn["support"]

        false_sklearn = sklearn_report.get("False", {})
        false_ours = next((r for r in test_one_results if r["class"] == "False"), None)

        assert false_ours is not None
        assert false_ours["precision"] == round(false_sklearn["precision"], 2)
        assert false_ours["recall"] == round(false_sklearn["recall"], 2)
        assert false_ours["f1"] == round(false_sklearn["f1-score"], 2)
        assert false_ours["sample_size"] == false_sklearn["support"]

        assert round(sklearn_report["accuracy"], 2) == 0.67

    def test_sklearn_report_comparison_grouped(self, test_results_groups_two):
        """Test that our metrics match sklearn.metrics.classification_report for grouped results."""
        from sklearn.metrics import classification_report

        prompt1_results = {
            k: v for k, v in test_results_groups_two.items() if v.group == prompt1
        }
        prompt2_results = {
            k: v for k, v in test_results_groups_two.items() if v.group == prompt2
        }

        y_true_prompt1 = []
        y_pred_prompt1 = []

        for result in prompt1_results.values():
            y_true_prompt1.append(str(result.expected))
            y_pred_prompt1.append(str(result.actual))

        sklearn_report_prompt1 = classification_report(
            y_true_prompt1, y_pred_prompt1, output_dict=True, zero_division=0
        )

        y_true_prompt2 = []
        y_pred_prompt2 = []

        for result in prompt2_results.values():
            y_true_prompt2.append(str(result.expected))
            y_pred_prompt2.append(str(result.actual))

        sklearn_report_prompt2 = classification_report(
            y_true_prompt2, y_pred_prompt2, output_dict=True, zero_division=0
        )

        our_report = generate_classification_reports(test_results_groups_two)
        test_one_results = our_report.get("test_one", [])

        prompt1_true_ours = next(
            (
                r
                for r in test_one_results
                if r["group"] == prompt1 and r["class"] == "True"
            ),
            None,
        )
        prompt1_false_ours = next(
            (
                r
                for r in test_one_results
                if r["group"] == prompt1 and r["class"] == "False"
            ),
            None,
        )

        prompt1_true_sklearn = sklearn_report_prompt1.get("True", {})
        prompt1_false_sklearn = sklearn_report_prompt1.get("False", {})

        assert prompt1_true_ours is not None
        assert prompt1_true_ours["precision"] == round(
            prompt1_true_sklearn["precision"], 2
        )
        assert prompt1_true_ours["recall"] == round(prompt1_true_sklearn["recall"], 2)
        assert prompt1_true_ours["f1"] == round(prompt1_true_sklearn["f1-score"], 2)
        assert prompt1_true_ours["sample_size"] == prompt1_true_sklearn["support"]

        assert prompt1_false_ours is not None
        assert prompt1_false_ours["precision"] == round(
            prompt1_false_sklearn["precision"], 2
        )
        assert prompt1_false_ours["recall"] == round(prompt1_false_sklearn["recall"], 2)
        assert prompt1_false_ours["f1"] == round(prompt1_false_sklearn["f1-score"], 2)
        assert prompt1_false_ours["sample_size"] == prompt1_false_sklearn["support"]

        assert round(sklearn_report_prompt1["accuracy"], 2) == 0.67

        prompt2_true_ours = next(
            (
                r
                for r in test_one_results
                if r["group"] == prompt2 and r["class"] == "True"
            ),
            None,
        )
        prompt2_false_ours = next(
            (
                r
                for r in test_one_results
                if r["group"] == prompt2 and r["class"] == "False"
            ),
            None,
        )

        prompt2_true_sklearn = sklearn_report_prompt2.get("True", {})
        prompt2_false_sklearn = sklearn_report_prompt2.get("False", {})

        assert prompt2_true_ours is not None
        assert prompt2_true_ours["precision"] == round(
            prompt2_true_sklearn["precision"], 2
        )
        assert prompt2_true_ours["recall"] == round(prompt2_true_sklearn["recall"], 2)
        assert prompt2_true_ours["f1"] == round(prompt2_true_sklearn["f1-score"], 2)
        assert prompt2_true_ours["sample_size"] == prompt2_true_sklearn["support"]

        assert prompt2_false_ours is not None
        assert prompt2_false_ours["precision"] == round(
            prompt2_false_sklearn["precision"], 2
        )
        assert prompt2_false_ours["recall"] == round(prompt2_false_sklearn["recall"], 2)
        assert prompt2_false_ours["f1"] == round(prompt2_false_sklearn["f1-score"], 2)
        assert prompt2_false_ours["sample_size"] == prompt2_false_sklearn["support"]

        assert round(sklearn_report_prompt2["accuracy"], 2) == 0.67

    def test_generate_classification_reports_groups_two(self, test_results_groups_two):
        actual = generate_classification_reports(test_results_groups_two)
        test_one_results = actual.get("test_one", [])

        assert len(test_one_results) == 4

        prompt1_true = next(
            (
                r
                for r in test_one_results
                if r["group"] == prompt1 and r["class"] == "True"
            ),
            None,
        )
        assert prompt1_true is not None
        assert prompt1_true["test_name"] == "test_one"
        assert prompt1_true["sample_size"] == 2
        assert prompt1_true["accuracy"] == 0.67
        assert prompt1_true["precision"] == 0.67
        assert prompt1_true["recall"] == 1.0
        assert prompt1_true["f1"] == 0.80

        prompt1_false = next(
            (
                r
                for r in test_one_results
                if r["group"] == prompt1 and r["class"] == "False"
            ),
            None,
        )
        assert prompt1_false is not None
        assert prompt1_false["test_name"] == "test_one"
        assert prompt1_false["sample_size"] == 1
        assert prompt1_false["accuracy"] == 0.0
        assert prompt1_false["precision"] == 0.0
        assert prompt1_false["recall"] == 0.0
        assert prompt1_false["f1"] == 0.0

        prompt2_true = next(
            (
                r
                for r in test_one_results
                if r["group"] == prompt2 and r["class"] == "True"
            ),
            None,
        )
        assert prompt2_true is not None
        assert prompt2_true["test_name"] == "test_one"
        assert prompt2_true["sample_size"] == 2
        assert prompt2_true["accuracy"] == 0.67
        assert prompt2_true["precision"] == 0.67
        assert prompt2_true["recall"] == 1.0
        assert prompt2_true["f1"] == 0.80

        prompt2_false = next(
            (
                r
                for r in test_one_results
                if r["group"] == prompt2 and r["class"] == "False"
            ),
            None,
        )
        assert prompt2_false is not None
        assert prompt2_false["test_name"] == "test_one"
        assert prompt2_false["sample_size"] == 1
        assert prompt2_false["accuracy"] == 0.0
        assert prompt2_false["precision"] == 0.0
        assert prompt2_false["recall"] == 0.0
        assert prompt2_false["f1"] == 0.0


class TestLLMEvalFormat:
    def test_format_report_as_text_one(self, sample_groups_one):
        text_lines = format_report_as_text(
            "test_name",
            sample_groups_one,
        )
        text_data = "\n".join(text_lines)

        expected = textwrap.dedent(
            """
            Test: test_name

            Group Legend:
            G1: overall

            Metrics Comparison:
            Group | Class | Sample Size | Accuracy | Precision | Recall |   F1
            ------------------------------------------------------------------
            G1    | False |           1 |     0.00 |      0.00 |   0.00 | 0.00
            G1    | True  |           2 |     0.00 |      0.67 |   1.00 | 0.80
            G1    |       |           3 |     0.00 |      0.34 |   0.50 | 0.40
            """
        ).strip()
        assert text_data == expected

    def test_format_report_as_text_two(self, sample_groups_two):
        text_lines = format_report_as_text(
            "test_name",
            sample_groups_two,
        )
        text_data = "\n".join(text_lines)

        prompt1 = "Is this computer related? Say True or False"
        prompt2 = "Say True or False: Is this computer related?"

        expected = textwrap.dedent(
            f"""
            Test: test_name

            Group Legend:
            G1: {prompt1}
            G2: {prompt2}

            Metrics Comparison:
            Group | Class | Sample Size | Accuracy | Precision | Recall |   F1
            ------------------------------------------------------------------
            G1    | False |           1 |     0.00 |      0.00 |   0.00 | 0.00
            G1    | True  |           2 |     0.00 |      0.67 |   1.00 | 0.80
            G1    |       |           3 |     0.00 |      0.34 |   0.50 | 0.40
            G2    | False |           1 |     0.00 |      0.00 |   0.00 | 0.00
            G2    | True  |           2 |     0.00 |      0.67 |   1.00 | 0.80
            G2    |       |           3 |     0.00 |      0.34 |   0.50 | 0.40
            """
        ).strip()
        assert text_data == expected

    def test_format_report_as_csv_one(self, sample_groups_one):
        csv_rows = format_report_as_csv(
            "test_name",
            sample_groups_one,
        )

        output = io.StringIO()
        writer = csv.writer(output)
        for row in csv_rows:
            writer.writerow(row)
        csv_text = output.getvalue().strip()

        expected = textwrap.dedent(
            """
            test_name,group,class,sample_size,precision,recall,f1
            test_name,overall,False,1,0.0,0.0,0.0
            test_name,overall,True,2,0.67,1.0,0.8
            test_name,overall,,3,0.335,0.5,0.4
            """
        ).strip()

        output = io.StringIO()
        writer = csv.writer(output)
        for row in csv_rows:
            writer.writerow(row)
        csv_text = output.getvalue().strip()

        normalized_expected = expected.replace("\r\n", "\n").replace("\r", "\n")
        normalized_actual = csv_text.replace("\r\n", "\n").replace("\r", "\n")
        assert normalized_actual == normalized_expected

    def test_format_report_as_csv_two(self, sample_groups_two):
        csv_rows = format_report_as_csv(
            "test_name",
            sample_groups_two,
        )

        expected = textwrap.dedent(
            f"""
            test_name,group,class,sample_size,precision,recall,f1
            test_name,{prompt1},False,1,0.0,0.0,0.0
            test_name,{prompt1},True,2,0.67,1.0,0.8
            test_name,{prompt1},,3,0.335,0.5,0.4
            test_name,{prompt2},False,1,0.0,0.0,0.0
            test_name,{prompt2},True,2,0.67,1.0,0.8
            test_name,{prompt2},,3,0.335,0.5,0.4
            """
        ).strip()

        output = io.StringIO()
        writer = csv.writer(output)
        for row in csv_rows:
            writer.writerow(row)
        csv_text = output.getvalue().strip()

        normalized_expected = expected.replace("\r\n", "\n").replace("\r", "\n")
        normalized_actual = csv_text.replace("\r\n", "\n").replace("\r", "\n")
        assert normalized_actual == normalized_expected


class TestLLMEvalSave:
    def test_marker_output_file(self, testdir, tmp_path):
        output_file = tmp_path / "results.csv"

        testdir.makepyfile(
            f"""
            import pytest
            import os
            
            @pytest.mark.llmeval(output_file="{output_file}")
            def test_with_output_file(llmeval_result):
                llmeval_result.set_result(True, True)
                assert True
            
            def test_without_marker():
                assert True
        """
        )

        result = testdir.runpytest("-v")
        result.stdout.fnmatch_lines(["*::test_with_output_file LLMEVAL*"])

        assert output_file.exists()

        if output_file.exists():
            output_file.unlink()
