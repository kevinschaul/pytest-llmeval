import pytest
import json
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import classification_report

# Store all results from decorated tests
test_results = {}
# Store output file configurations
output_file_configs = {}


class ClassificationResult:
    def __init__(self):
        self.expected = None
        self.actual = None
        self.input = None
        self.metadata = {}

    def set_result(self, expected, actual, input_data=None):
        """Set the main result data."""
        self.expected = expected
        self.actual = actual
        self.input = input_data

    def add_metadata(self, **kwargs):
        """Add metadata about the test."""
        self.metadata.update(kwargs)

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

            # Extract output_file from marker kwargs if present
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
        "llmeval(output_file=None): mark test as an LLM evaluation test with optional output file",
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


def get_grouped_results():
    """Group all results by test function rather than by individual parametrized runs."""
    grouped_results = defaultdict(list)

    for nodeid, result in test_results.items():
        # Extract the test function name (remove parameter part)
        test_func = nodeid.split("[")[0]
        grouped_results[test_func].append(result)

    return grouped_results


def format_report_as_text(test_name, num_cases, report_dict):
    """
    Format the classification report as plain text with properly aligned columns.

    Args:
        test_name: Name of the test function
        num_cases: Number of test cases
        report_dict: Dictionary representation of the classification report

    Returns:
        str: Formatted text report
    """
    lines = []
    lines.append(f"Test: {test_name}")

    # Calculate column widths for proper alignment
    class_width = max(len(str(k)) for k in report_dict.keys())
    class_width = max(class_width, len("Class"))  # Ensure header fits

    # Define column widths
    precision_width = 10
    recall_width = 10
    f1_width = 10
    support_width = 10

    # Create the header with proper alignment
    header = (
        f"{'Class':<{class_width}} | "
        f"{'Precision':>{precision_width}} | "
        f"{'Recall':>{recall_width}} | "
        f"{'F1-Score':>{f1_width}} | "
        f"{'Support':>{support_width}}"
    )
    lines.append(header)

    # Add separator line
    separator = "-" * (
        class_width + precision_width + recall_width + f1_width + support_width + 12
    )
    lines.append(separator)

    # Add data rows with proper alignment
    for class_name, metrics in report_dict.items():
        if isinstance(metrics, dict):
            row = (
                f"{class_name:<{class_width}} | "
                f"{metrics.get('precision', 0.0):>{precision_width}.2f} | "
                f"{metrics.get('recall', 0.0):>{recall_width}.2f} | "
                f"{metrics.get('f1-score', 0.0):>{f1_width}.2f} | "
                f"{metrics.get('support', 0):>{support_width}.0f}"
            )
            lines.append(row)
        else:
            # Handle special cases like 'accuracy'
            row = (
                f"{class_name:<{class_width}} | "
                f"{'':{precision_width}} | "
                f"{'':{recall_width}} | "
                f"{metrics:{f1_width}.2f} | "
                f"{num_cases:>{support_width}}"
            )
            lines.append(row)

    return "\n".join(lines)


def format_report_as_json(test_name, num_cases, report_dict):
    """
    Format the classification report as JSON.

    Args:
        test_name: Name of the test function
        num_cases: Number of test cases
        report_dict: Dictionary representation of the classification report

    Returns:
        dict: Structured report as a dictionary ready for JSON serialization
    """
    return {
        "test_name": test_name,
        "num_test_cases": num_cases,
        "classification_report": report_dict,
    }


def format_report_as_csv(test_name, num_cases, report_dict):
    """
    Format the classification report as CSV rows.

    Args:
        test_name: Name of the test function
        num_cases: Number of test cases
        report_dict: Dictionary representation of the classification report

    Returns:
        list: List of CSV rows (each row is a list of values)
    """
    rows = []
    # Header row
    rows.append(["test_name", "class", "precision", "recall", "f1-score", "support"])

    # Data rows
    for class_name, metrics in report_dict.items():
        if isinstance(metrics, dict):
            rows.append(
                [
                    test_name,
                    class_name,
                    metrics.get("precision", ""),
                    metrics.get("recall", ""),
                    metrics.get("f1-score", ""),
                    metrics.get("support", ""),
                ]
            )
        else:
            # Handle special cases like 'accuracy'
            rows.append([test_name, class_name, "", "", metrics, num_cases])

    return rows


def save_classification_report(test_name, num_cases, report_dict, output_path):
    """
    Save the classification report to a file.

    Args:
        test_name: Name of the test function
        num_cases: Number of test cases
        report_dict: Dictionary representation of the classification report
        output_path: Path to save the report to
    """
    output_path = Path(output_path)

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine file format based on extension
    ext = output_path.suffix.lower()

    if ext == ".json":
        json_data = format_report_as_json(test_name, num_cases, report_dict)
        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2)
    elif ext == ".csv":
        import csv

        csv_rows = format_report_as_csv(test_name, num_cases, report_dict)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in csv_rows:
                writer.writerow(row)
    else:
        # Default to plain text for any other extension
        text_report = format_report_as_text(test_name, num_cases, report_dict)
        with open(output_path, "w") as f:
            f.write(text_report)


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add a final section to the terminal summary with sklearn's classification report."""
    if not test_results:
        return

    terminalreporter.write_sep("=", "LLM Evaluation Results")

    # Group results by test function (for parameterized tests)
    grouped_results = get_grouped_results()

    # Process each test function's results
    for test_func, results in grouped_results.items():
        test_name = test_func.split("::")[-1]

        y_true = []
        y_pred = []

        for result in results:
            if result.expected is not None and result.actual is not None:
                y_true.append(str(result.expected))
                y_pred.append(str(result.actual))

        # Generate metrics report if we have true/predicted values
        if y_true and y_pred:
            # Generate report dictionary
            report_dict = classification_report(
                y_true, y_pred, zero_division=0.0, output_dict=True
            )

            # Format the text report using our utility function
            text_report = format_report_as_text(test_name, len(results), report_dict)

            # Write to terminal
            for line in text_report.split("\n"):
                terminalreporter.write_line(line)

            # Check if we need to save to a file
            # Get the first nodeid for this test function to check for output_file
            first_nodeid = next(
                (
                    nodeid
                    for nodeid in test_results.keys()
                    if nodeid.startswith(test_func)
                ),
                None,
            )
            if first_nodeid and first_nodeid in output_file_configs:
                output_file = output_file_configs[first_nodeid]
                if output_file:
                    save_classification_report(
                        test_name, len(results), report_dict, output_file
                    )
                    terminalreporter.write_line(
                        f"\nClassification report saved to: {output_file}"
                    )

    terminalreporter.write_sep("=", "End of LLM Evaluation Results")
