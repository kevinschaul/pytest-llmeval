import pytest
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import classification_report


DEFAULT_GROUP = "overall"


class LLMEvalException(Exception):
    pass


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
        self.llmeval_results = {}
        self.llmeval_kwargs = {}

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item):
        """Identify and track tests with llmeval marker."""
        marker = item.get_closest_marker("llmeval")
        if not marker:
            return

        self.llmeval_nodes.add(item.nodeid)

        test_id = self.get_test_id(item.nodeid)
        config = self.llmeval_kwargs.setdefault(test_id, {})

        analysis_func = marker.kwargs.get("analysis_func")
        output_file = marker.kwargs.get("output_file")

        if analysis_func and output_file:
            raise LLMEvalException(
                "`output_file` does nothing when `analysis_func` is specified"
            )
        elif output_file:
            config["output_file"] = output_file

        if analysis_func:
            config["analysis_func"] = analysis_func

    @pytest.hookimpl(tryfirst=True)
    def pytest_report_teststatus(self, report):
        """Intercept test reports for llmeval tests."""
        if hasattr(report, "nodeid") and report.nodeid in self.llmeval_nodes:
            if report.when == "call":
                return "llmeval", "L", "LLMEVAL"

    def get_result_for_test(self, nodeid):
        """Create or retrieve a result object for a test."""
        if nodeid not in self.llmeval_results:
            self.llmeval_results[nodeid] = ClassificationResult()
        return self.llmeval_results[nodeid]

    def save_report(self, output_path, lines):
        """Save report to a file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

    def analysis_func_classification_report(self, test_id, results):
        """Default analysis function that generates a classification report for results.

        Args:
            test_id: The name of the test function
            results: List of ClassificationResult objects

        Returns:
            List of strings to be printed in the terminal
        """
        lines = []
        lines.append(f"# LLM Eval: {test_id}")

        # Group results by their group attribute
        grouped_results = defaultdict(list)
        for result in results:
            group = result.group or DEFAULT_GROUP
            grouped_results[group].append(result)

        for group_name, group_results in grouped_results.items():
            # Extract expected and actual values
            pairs = [
                (r.expected, r.actual)
                for r in group_results
                if r.expected is not None and r.actual is not None
            ]

            if not pairs:
                continue

            group_y_true, group_y_pred = zip(*pairs)
            group_y_true = [str(y) for y in group_y_true]
            group_y_pred = [str(y) for y in group_y_pred]

            report = classification_report(
                group_y_true,
                group_y_pred,
                zero_division=0,  # type: ignore
            )

            lines.append(f"\n## Group: {group_name}")
            lines.append(report)

        # Save report if configured
        config = self.llmeval_kwargs.get(test_id, {})
        output_file = config.get("output_file")

        if output_file:
            self.save_report(output_file, lines)
            lines.append(f"\nClassification report saved to: {output_file}")

        return lines

    def get_analysis_func(self, test_id):
        """Get the appropriate analysis function for a test function."""
        config = self.llmeval_kwargs.get(test_id, {})

        if config.get("analysis_func"):
            return config["analysis_func"]
        else:
            return self.analysis_func_classification_report

    def get_test_id(self, nodeid):
        return nodeid.split("[")[0]

    @pytest.hookimpl(trylast=True)
    def pytest_terminal_summary(self, terminalreporter, exitstatus):
        """Add a final section to the terminal summary."""
        if not self.llmeval_results:
            return

        # First collect all results by test id
        results_by_test_id = defaultdict(list)
        for nodeid, result in self.llmeval_results.items():
            test_id = self.get_test_id(nodeid)
            results_by_test_id[test_id].append(result)

        # Process each test id with all its results
        for test_id, results in results_by_test_id.items():
            analysis_func = self.get_analysis_func(test_id)
            output = analysis_func(test_id, results)

            if not output:
                continue

            lines = output if isinstance(output, list) else [output]
            for line in lines:
                terminalreporter.write_line(line)


# Plugin instance used to store state
_plugin_instance = None


def pytest_configure(config):
    """Register the plugin."""
    global _plugin_instance

    config.addinivalue_line(
        "markers",
        "llmeval(output_file=None, analysis_func=None): mark test as an LLM evaluation test with optional parameters. Note: when analysis_func is provided, output_file is ignored as the custom function handles its own output. The analysis_func should accept (test_id, grouped_results) parameters and can optionally return strings to display in the terminal.",
    )

    _plugin_instance = LLMEvalReportPlugin(config)
    config.pluginmanager.register(_plugin_instance, "llmeval_reporter")


def pytest_unconfigure(config):
    """Unregister the plugin."""
    global _plugin_instance

    if _plugin_instance:
        config.pluginmanager.unregister(_plugin_instance)
        _plugin_instance = None


@pytest.fixture
def llmeval_result(request):
    """Fixture to provide a result object for tests with the llmeval marker."""
    global _plugin_instance

    if not _plugin_instance:
        return None

    marker = request.node.get_closest_marker("llmeval")
    if marker:
        return _plugin_instance.get_result_for_test(request.node.nodeid)

    return None
