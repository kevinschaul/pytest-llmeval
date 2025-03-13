import pytest
from pathlib import Path
from collections import defaultdict
from typing import cast, TypedDict, Dict
from sklearn.metrics import classification_report

# Store all results from decorated tests
test_results = {}
# Store output file configurations
output_file_configs = {}

DEFAULT_GROUP = "overall"


ClassificationReport = TypedDict(
    "ClassificationReport",
    {
        "test_name": str,
        "group": str,
        "class": str,
        "sample_size": int,
        "precision": float,
        "recall": float,
        "f1": float,
    },
)
ClassificationReportsByTestName = Dict[str, ClassificationReport]


class ClassificationResult:
    def __init__(self, expected=None, actual=None, input_data=None, group=None):
        self.expected = expected
        self.actual = actual
        self.input = input_data
        self.group = group

    def set_result(self, expected, actual, input_data=None, group=DEFAULT_GROUP):
        """Set the main result data."""
        self.expected = expected
        self.actual = actual
        self.input = input_data
        self.group = group

    def is_correct(self):
        """Check if the prediction was correct."""
        return self.expected == self.actual


class LLMEvalReportPlugin:
    def __init__(self, config):
        self.config = config
        self.llmeval_nodes = set()

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item):
        """Identify and track tests with llmeval marker."""
        marker = item.get_closest_marker("llmeval")
        if marker:
            self.llmeval_nodes.add(item.nodeid)

            if marker.kwargs.get("output_file"):
                output_file_configs[item.nodeid] = marker.kwargs.get("output_file")

    @pytest.hookimpl(tryfirst=True)
    def pytest_report_teststatus(self, report):
        """Intercept test reports for llmeval tests."""
        if hasattr(report, "nodeid") and report.nodeid in self.llmeval_nodes:
            if report.when == "call":
                return "llmeval", "L", "LLMEVAL"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "llmeval(output_file=None, group_by=None): mark test as an LLM evaluation test with optional output file and grouping parameter",
    )
    test_results.clear()
    output_file_configs.clear()
    config.pluginmanager.register(LLMEvalReportPlugin(config), "llmeval_reporter")


@pytest.fixture
def llmeval_result(request):
    """Fixture to provide a result object for tests with the llmeval marker."""
    marker = request.node.get_closest_marker("llmeval")
    if marker:
        # Create a new classification result
        result_obj = ClassificationResult()

        # Store in our global results dictionary
        test_results[request.node.nodeid] = result_obj

        return result_obj
    return None


def get_grouped_results(test_results):
    """Group all results by test function rather than by individual parametrized runs."""
    results_by_test_func = defaultdict(list)
    results_by_group = defaultdict(lambda: defaultdict(list))

    for nodeid, result in test_results.items():
        # Extract the test function name (remove parameter part)
        test_func = nodeid.split("[")[0]
        results_by_test_func[test_func].append(result)

        group = result.group or DEFAULT_GROUP
        results_by_group[test_func][group].append(result)

    return results_by_test_func, results_by_group


def calculate_macro_averages(classes):
    """
    Calculate macro averages for precision, recall, and F1 score.
    
    Args:
        classes: Dictionary of class names to metrics dictionaries
        
    Returns:
        tuple: (avg_precision, avg_recall, avg_f1)
    """
    # Get class names, excluding any special keys like "accuracy"
    class_names = [c for c in classes.keys() if c != "accuracy"]
    
    if not class_names:
        return 0, 0, 0
        
    # Calculate macro averages (simple average across classes)
    avg_precision = sum(classes[c]["precision"] for c in class_names) / len(class_names)
    avg_recall = sum(classes[c]["recall"] for c in class_names) / len(class_names)
    avg_f1 = sum(classes[c]["f1-score"] for c in class_names) / len(class_names)
    
    return avg_precision, avg_recall, avg_f1


def format_report_as_text(test_name, groups):
    """
    Format the classification report as plain text with group comparison.

    Args:
        test_name: Name of the test function
        groups: List of classification reports

    Returns:
        list: List of text lines for the formatted report
    """
    text_lines = []
    text_lines.append(f"Test: {test_name}")

    group_data = {}
    for report in groups:
        group_name = report["group"]
        class_name = report["class"]

        if group_name not in group_data:
            group_data[group_name] = {}

        group_data[group_name][class_name] = {
            "precision": report["precision"],
            "recall": report["recall"],
            "f1-score": report["f1"],
            "support": report["sample_size"],
        }

        if "accuracy" not in group_data[group_name]:
            group_data[group_name]["accuracy"] = report["accuracy"]

    text_lines.append("\nGroup Legend:")
    group_ids = {group: f"G{i+1}" for i, group in enumerate(group_data.keys())}
    for group, group_id in group_ids.items():
        text_lines.append(f"{group_id}: {group}")

    text_lines.append("\nMetrics Comparison:")
    text_lines.append(
        f"Group | Class | Sample Size | Accuracy | Precision | Recall |   F1"
    )
    text_lines.append("-" * 66)

    # Add rows for each group and class
    for group_name, classes in group_data.items():
        group_id = group_ids[group_name]

        # Sort classes to ensure consistent order
        class_names = sorted([c for c in classes.keys() if c != "accuracy"])

        # First add class-specific rows
        for class_name in class_names:
            metrics = classes[class_name]
            # Use class's own accuracy
            class_accuracy = accuracy = classes.get("accuracy", 0)
            row = f"{group_id:<5} | {class_name:<5} | {metrics['support']:>11} | {class_accuracy:>8.2f} | {metrics['precision']:>9.2f} | {metrics['recall']:>6.2f} | {metrics['f1-score']:>3.2f}"
            text_lines.append(row)

        # Add the group summary row
        total_samples = sum(classes[c]["support"] for c in classes if c != "accuracy")
        accuracy = classes.get("accuracy", 0)
        # Calculate macro average for precision, recall, f1
        avg_precision, avg_recall, avg_f1 = calculate_macro_averages(classes)
        row = f"{group_id:<5} | {'':<5} | {total_samples:>11} | {accuracy:>8.2f} | {avg_precision:>9.2f} | {avg_recall:>6.2f} | {avg_f1:>3.2f}"
        text_lines.append(row)

    return text_lines


def format_report_as_csv(test_name, groups):
    """
    Format the classification report as CSV rows.

    Args:
        test_name: Name of the test function
        groups: List of classification reports

    Returns:
        list: List of CSV rows (each row is a list of values)
    """
    rows = []

    # Header row
    rows.append(
        [
            "test_name",
            "group",
            "class",
            "sample_size",
            "precision",
            "recall",
            "f1",
        ]
    )

    # Convert list of reports to a group-based dictionary
    group_data = {}
    for report in groups:
        group_name = report["group"]
        class_name = report["class"]

        if group_name not in group_data:
            group_data[group_name] = {}

        group_data[group_name][class_name] = {
            "precision": report["precision"],
            "recall": report["recall"],
            "f1-score": report["f1"],
            "support": report["sample_size"],
            "accuracy": report["accuracy"],
        }

    # Add a row for each group and class
    for group_name, classes in group_data.items():
        for class_name, metrics in classes.items():
            if class_name != "accuracy":
                rows.append(
                    [
                        test_name,
                        group_name,
                        class_name,
                        metrics["support"],
                        metrics["precision"],
                        metrics["recall"],
                        metrics["f1-score"],
                    ]
                )

        # Add a summary row for the group
        total_samples = sum(classes[c]["support"] for c in classes if c != "accuracy")
        accuracy = next(iter(classes.values())).get("accuracy", 0)

        # Calculate macro averages for the metrics (like sklearn does)
        avg_precision, avg_recall, avg_f1 = calculate_macro_averages(classes)

        rows.append(
            [
                test_name,
                group_name,
                "",  # No class for summary row
                total_samples,
                avg_precision,
                avg_recall,
                avg_f1,
            ]
        )

    return rows


def save_classification_report(test_name, groups, output_path):
    """
    Save the classification report to a file.

    Args:
        test_name: Name of the test function
        output_path: Path to save the report to
        groups: Dictionary of classification reports
    """
    output_path = Path(output_path)

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine file format based on extension
    ext = output_path.suffix.lower()

    if ext == ".csv":
        import csv

        csv_rows = format_report_as_csv(test_name, groups)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in csv_rows:
                writer.writerow(row)
    else:
        # Default to plain text for any other extension
        text_lines = format_report_as_text(test_name, groups)
        with open(output_path, "w") as f:
            f.write("\n".join(text_lines))


def generate_classification_reports(test_results) -> ClassificationReportsByTestName:
    results_by_test_func, results_by_group = get_grouped_results(test_results)
    report_data = {}

    # Process each test function's results
    for test_func, results in results_by_test_func.items():
        test_name = test_func.split("::")[-1]

        group_data = []
        for group_name, group_results in results_by_group[test_func].items():
            group_y_true = []
            group_y_pred = []

            for result in group_results:
                if result.expected is not None and result.actual is not None:
                    group_y_true.append(str(result.expected))
                    group_y_pred.append(str(result.actual))

            if group_y_true and group_y_pred:
                report_raw = classification_report(
                    group_y_true,
                    group_y_pred,
                    zero_division=0,  # type: ignore
                    output_dict=True,
                )
                if isinstance(report_raw, Dict):
                    for key, value in report_raw.items():
                        if isinstance(value, Dict) and key not in [
                            "macro avg",
                            "weighted avg",
                        ]:
                            # Round all values to 2 decimal places for consistency with tests
                            overall_accuracy = round(report_raw.get("accuracy", 0.0), 2)
                            accuracy = overall_accuracy if key == "True" else 0.0

                            report = cast(
                                ClassificationReport,
                                {
                                    "test_name": test_name,
                                    "group": group_name,
                                    "class": key,
                                    "sample_size": int(value.get("support", 0)),
                                    "accuracy": accuracy,
                                    "precision": round(value.get("precision", 0.0), 2),
                                    "recall": round(value.get("recall", 0.0), 2),
                                    "f1": round(value.get("f1-score", 0.0), 2),
                                },
                            )
                            group_data.append(report)

        report_data[test_name] = group_data
    return report_data


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add a final section to the terminal summary with sklearn's classification report."""
    if not test_results:
        return

    terminalreporter.write_sep("=", "LLM Evaluation Results")

    grouped_reports = generate_classification_reports(test_results)

    for test_name, groups in grouped_reports.items():
        text_lines = format_report_as_text(test_name, groups)

        for line in text_lines:
            terminalreporter.write_line(line)

        # Check if we need to save to a file
        first_nodeid = next(
            (
                nodeid
                for nodeid in test_results.keys()
                if test_name in nodeid.split("::")[-1]
            ),
            None,
        )
        if first_nodeid and first_nodeid in output_file_configs:
            output_file = output_file_configs[first_nodeid]
            if output_file:
                # Save the report with group comparison
                save_classification_report(
                    test_name,
                    groups,
                    output_file,
                )
                terminalreporter.write_line(
                    f"\nClassification report saved to: {output_file}"
                )

    terminalreporter.write_sep("=", "End of LLM Evaluation Results")
