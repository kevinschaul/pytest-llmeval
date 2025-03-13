import pytest
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import classification_report

# Store all results from decorated tests
test_results = {}
# Store output file configurations
output_file_configs = {}

DEFAULT_GROUP = "overall"


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
        "llmeval(output_file=None): mark test as an LLM evaluation test with optional output file parameter",
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
    """Group `test_results` by test function and group"""
    grouped = defaultdict(lambda: defaultdict(list))

    for nodeid, result in test_results.items():
        # Extract the test function name (remove parameter part)
        test_func = nodeid.split("[")[0]
        group = result.group or DEFAULT_GROUP
        grouped[test_func][group].append(result)

    return grouped


def format_report_as_text(test_func, grouped):
    """Generate a metrics table for each grouped result"""
    lines = []
    test_name = test_func.split("::")[-1]
    lines.append(f"# LLM Eval: {test_name}")

    for group_name, group_results in grouped.items():
        group_y_true = []
        group_y_pred = []

        for result in group_results:
            if result.expected is not None and result.actual is not None:
                group_y_true.append(str(result.expected))
                group_y_pred.append(str(result.actual))

        if group_y_true and group_y_pred:
            report = classification_report(
                group_y_true,
                group_y_pred,
                zero_division=0,  # type: ignore
            )

            lines.append(f"\n## Group: {group_name}")
            lines.append(report)
    return lines


def save_report(output_path, lines):
    output_path = Path(output_path)

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add a final section to the terminal summary with sklearn's classification report."""
    if not test_results:
        return

    grouped = get_grouped_results(test_results)
    for test_func, test_grouped in grouped.items():
        lines = format_report_as_text(test_func, test_grouped)

        for line in lines:
            terminalreporter.write_line(line)

        test_name = test_func.split("::")[-1]

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
                save_report(output_file, lines)
                terminalreporter.write_line(
                    f"\nClassification report saved to: {output_file}"
                )
