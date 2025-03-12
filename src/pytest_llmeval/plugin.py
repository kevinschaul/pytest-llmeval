import pytest
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
        self.group = None

    def set_result(self, expected, actual, input_data=None, group=None):
        """Set the main result data."""
        self.expected = expected
        self.actual = actual
        self.input = input_data
        if group:
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


def get_grouped_results():
    """Group all results by test function rather than by individual parametrized runs."""
    grouped_results = defaultdict(list)
    group_by_metadata = defaultdict(lambda: defaultdict(list))

    for nodeid, result in test_results.items():
        # Extract the test function name (remove parameter part)
        test_func = nodeid.split("[")[0]
        grouped_results[test_func].append(result)

        # If group is specified for this result, also group by that
        if result.group:
            group_by_metadata[test_func][result.group].append(result)

    return grouped_results, group_by_metadata


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


def format_report_as_csv(test_name, num_cases, report_dict, groups=None):
    """
    Format the classification report as CSV rows.

    Args:
        test_name: Name of the test function
        num_cases: Number of test cases
        report_dict: Dictionary representation of the classification report
        groups: Optional dictionary of group metrics

    Returns:
        list: List of CSV rows (each row is a list of values)
    """
    rows = []

    if not groups:
        # Standard format without groups
        # Header row
        rows.append(
            [
                "test_name",
                "group",
                "class",
                "precision",
                "recall",
                "f1-score",
                "support",
            ]
        )

        # Data rows for overall results
        for class_name, metrics in report_dict.items():
            if isinstance(metrics, dict):
                rows.append(
                    [
                        test_name,
                        "overall",  # No group for overall results
                        class_name,
                        metrics.get("precision", ""),
                        metrics.get("recall", ""),
                        metrics.get("f1-score", ""),
                        metrics.get("support", ""),
                    ]
                )
            else:
                # Handle special cases like 'accuracy'
                rows.append(
                    [test_name, "overall", class_name, "", "", metrics, num_cases]
                )
    else:
        # Format with groups in a flat structure
        # Add a header row
        rows.append(
            [
                "test_name",
                "group",
                "class",
                "precision",
                "recall",
                "f1-score",
                "support",
            ]
        )

        # First add rows for overall results
        for class_name, metrics in report_dict.items():
            if isinstance(metrics, dict):
                rows.append(
                    [
                        test_name,
                        "overall",  # Mark as overall results
                        class_name,
                        metrics.get("precision", ""),
                        metrics.get("recall", ""),
                        metrics.get("f1-score", ""),
                        metrics.get("support", ""),
                    ]
                )
            else:
                # Handle special cases like 'accuracy'
                rows.append(
                    [test_name, "overall", class_name, "", "", metrics, num_cases]
                )

        # Then add rows for each group's results
        for group_name, group_data in groups.items():
            for class_name, metrics in group_data["report"].items():
                if isinstance(metrics, dict):
                    rows.append(
                        [
                            test_name,
                            group_name,  # Use the full group name
                            class_name,
                            metrics.get("precision", ""),
                            metrics.get("recall", ""),
                            metrics.get("f1-score", ""),
                            metrics.get("support", ""),
                        ]
                    )
                else:
                    # Handle special cases like 'accuracy'
                    rows.append(
                        [
                            test_name,
                            group_name,
                            class_name,
                            "",
                            "",
                            metrics,
                            group_data["count"],
                        ]
                    )

    return rows


def save_classification_report(
    test_name, num_cases, report_dict, output_path, groups=None
):
    """
    Save the classification report to a file.

    Args:
        test_name: Name of the test function
        num_cases: Number of test cases
        report_dict: Dictionary representation of the classification report
        output_path: Path to save the report to
        groups: Optional dictionary of group metrics
    """
    output_path = Path(output_path)

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine file format based on extension
    ext = output_path.suffix.lower()

    if ext == ".csv":
        import csv

        csv_rows = format_report_as_csv(test_name, num_cases, report_dict, groups)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in csv_rows:
                writer.writerow(row)
    else:
        # Default to plain text for any other extension
        # For text format, we'll add a simple comparison table if groups are present
        text_lines = []
        text_lines.append(format_report_as_text(test_name, num_cases, report_dict))

        if groups:
            # Add group legend
            text_lines.append("\nGroup Legend:")
            text_lines.append("-" * 80)
            group_ids = {group: f"G{i+1}" for i, group in enumerate(groups.keys())}
            for group, group_id in group_ids.items():
                text_lines.append(f"{group_id}: {group}")

            # Add comparison table
            text_lines.append("\nMetrics Comparison:")
            text_lines.append("-" * 80)

            # Collect all labels
            all_labels = set()
            for group_name, group_data in groups.items():
                for label in group_data["report"].keys():
                    if (
                        not label.startswith("macro")
                        and not label.startswith("weighted")
                        and label != "accuracy"
                    ):
                        all_labels.add(label)

            # Create header
            header = f"{'Metric':<15} | {'Class':<10} |"
            for group in groups:
                group_id = group_ids[group]
                header += f" {group_id:<3} |"
            text_lines.append(header)
            text_lines.append("-" * (17 + 12 + (5 * len(groups))))

            # Add rows for each metric and class
            for metric in ["precision", "recall", "f1-score"]:
                for label in sorted(all_labels):
                    row = f"{metric:<15} | {str(label):<10} |"
                    for group in groups:
                        if label in groups[group]["report"]:
                            value = groups[group]["report"][label].get(metric, 0.0)
                            row += f" {value:>3.2f} |"
                        else:
                            row += f" {'':<3} |"
                    text_lines.append(row)

            # Add accuracy row
            row = f"{'accuracy':<15} | {'':<10} |"
            for group in groups:
                value = groups[group]["report"].get("accuracy", 0.0)
                row += f" {value:>3.2f} |"
            text_lines.append(row)

        # Write to file
        with open(output_path, "w") as f:
            f.write("\n".join(text_lines))


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add a final section to the terminal summary with sklearn's classification report."""
    if not test_results:
        return

    terminalreporter.write_sep("=", "LLM Evaluation Results")

    # Group results by test function (for parameterized tests)
    grouped_results, group_by_metadata = get_grouped_results()

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
                    # If we have groups, save a report with group comparison
                    if (
                        test_func in group_by_metadata
                        and group_by_metadata[test_func]
                        and len(group_by_metadata[test_func]) > 1
                    ):
                        # Extract the group metrics in the format needed for saving
                        groups_for_saving = {}
                        for group_name, group_results in group_by_metadata[
                            test_func
                        ].items():
                            group_y_true = []
                            group_y_pred = []

                            for result in group_results:
                                if (
                                    result.expected is not None
                                    and result.actual is not None
                                ):
                                    group_y_true.append(str(result.expected))
                                    group_y_pred.append(str(result.actual))

                            if group_y_true and group_y_pred:
                                group_report_dict = classification_report(
                                    group_y_true,
                                    group_y_pred,
                                    zero_division=0.0,
                                    output_dict=True,
                                )
                                groups_for_saving[group_name] = {
                                    "report": group_report_dict,
                                    "count": len(group_results),
                                }

                        # Save the report with group comparison
                        save_classification_report(
                            test_name,
                            len(results),
                            report_dict,
                            output_file,
                            groups=groups_for_saving,
                        )
                    else:
                        # Standard save without groups
                        save_classification_report(
                            test_name, len(results), report_dict, output_file
                        )

                    terminalreporter.write_line(
                        f"\nClassification report saved to: {output_file}"
                    )

        # If we have grouped results by metadata for this test function, create a comparison table
        if test_func in group_by_metadata and group_by_metadata[test_func]:
            groups = list(group_by_metadata[test_func].keys())
            if len(groups) > 1:  # Only create comparison if there are multiple groups
                terminalreporter.write_line("\nPrompt Template Comparison:")

                # Calculate metrics for each group
                group_metrics = {}
                labels = set()  # Collect all unique labels across groups
                for group_name, group_results in group_by_metadata[test_func].items():
                    group_y_true = []
                    group_y_pred = []

                    for result in group_results:
                        if result.expected is not None and result.actual is not None:
                            group_y_true.append(str(result.expected))
                            group_y_pred.append(str(result.actual))
                            labels.add(str(result.expected))

                    if group_y_true and group_y_pred:
                        # Generate report dictionary for this group
                        group_report_dict = classification_report(
                            group_y_true,
                            group_y_pred,
                            zero_division=0.0,
                            output_dict=True,
                        )
                        group_metrics[group_name] = {
                            "report": group_report_dict,
                            "count": len(group_results),
                        }

                        # We'll save the group reports together at the end
                        # No need to save individual reports here

                # Create group IDs for better table formatting with long prompts
                group_ids = {group: f"G{i+1}" for i, group in enumerate(groups)}

                # First show a legend table mapping IDs to actual group values
                terminalreporter.write_line("\nGroup Legend:")
                terminalreporter.write_line("-" * 80)
                for group, group_id in group_ids.items():
                    terminalreporter.write_line(f"{group_id}: {group}")
                terminalreporter.write_line("-" * 80)

                # Create the comparison table with short IDs
                terminalreporter.write_line("\nMetrics Comparison:")
                terminalreporter.write_line("-" * 80)

                # Determine column widths for IDs (much shorter)
                group_width = max(len(group_id) for group_id in group_ids.values())
                group_width = max(group_width, len("Group"))
                metric_width = 12

                # Create the header row
                header = f"{'Metric':<15} | {'Class':<10} |"
                for group in groups:
                    group_id = group_ids[group]
                    header += f" {group_id:<{group_width}} |"
                terminalreporter.write_line(header)
                terminalreporter.write_line(
                    "-" * (17 + 12 + (group_width + 3) * len(groups))
                )

                # Add rows for precision, recall, f1 for each class
                for metric in ["precision", "recall", "f1-score"]:
                    for label in sorted(labels):
                        row = f"{metric:<15} | {str(label):<10} |"
                        for group in groups:
                            group_id = group_ids[group]
                            if group in group_metrics:
                                report = group_metrics[group]["report"]
                                if str(label) in report:
                                    value = report[str(label)].get(metric, 0.0)
                                    row += f" {value:>{group_width}.2f} |"
                                else:
                                    row += f" {'':<{group_width}} |"
                            else:
                                row += f" {'':<{group_width}} |"
                        terminalreporter.write_line(row)

                # Add accuracy row
                row = f"{'accuracy':<15} | {'':<10} |"
                for group in groups:
                    group_id = group_ids[group]
                    if group in group_metrics:
                        report = group_metrics[group]["report"]
                        value = report.get("accuracy", 0.0)
                        row += f" {value:>{group_width}.2f} |"
                    else:
                        row += f" {'':<{group_width}} |"
                terminalreporter.write_line(row)

                # Add support row
                row = f"{'support':<15} | {'':<10} |"
                for group in groups:
                    group_id = group_ids[group]
                    if group in group_metrics:
                        row += f" {group_metrics[group]['count']:>{group_width}} |"
                    else:
                        row += f" {'':<{group_width}} |"
                terminalreporter.write_line(row)

                terminalreporter.write_line("-" * 80)

                # Also show individual reports if user wants detailed information
                for group_name, group_results in group_by_metadata[test_func].items():
                    group_y_true = []
                    group_y_pred = []

                    for result in group_results:
                        if result.expected is not None and result.actual is not None:
                            group_y_true.append(str(result.expected))
                            group_y_pred.append(str(result.actual))

                    if group_y_true and group_y_pred:
                        # Generate report dictionary for this group
                        group_report_dict = classification_report(
                            group_y_true,
                            group_y_pred,
                            zero_division=0.0,
                            output_dict=True,
                        )

                        # Format the text report using our utility function
                        group_text_report = format_report_as_text(
                            f"{test_name} (Group: {group_name})",
                            len(group_results),
                            group_report_dict,
                        )

                        # Write to terminal with a separator
                        terminalreporter.write_line("-" * 40)
                        for line in group_text_report.split("\n"):
                            terminalreporter.write_line(line)

    terminalreporter.write_sep("=", "End of LLM Evaluation Results")
