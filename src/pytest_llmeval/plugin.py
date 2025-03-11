import pytest
from collections import defaultdict
from sklearn.metrics import classification_report

# Store all results from decorated tests
test_results = {}


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
        if item.get_closest_marker("llmeval"):
            self.llmeval_nodes.add(item.nodeid)

    @pytest.hookimpl(tryfirst=True)
    def pytest_report_teststatus(self, report):
        """Intercept test reports for llmeval tests."""
        if hasattr(report, "nodeid") and report.nodeid in self.llmeval_nodes:
            if report.when == "call":
                return "llmeval", "L", "LLMEVAL"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "llmeval: mark test as an LLM evaluation test",
    )
    test_results.clear()
    config.pluginmanager.register(LLMEvalReportPlugin(config), "llmeval_reporter")


@pytest.fixture
def llmeval_result(request):
    """Fixture to provide a result object for tests with the llmeval marker."""
    marker = request.node.get_closest_marker("llmeval")
    if marker:
        # Create a new classification result
        result_obj = ClassificationResult()

        # Add any keyword arguments from the marker to the metadata
        if marker.kwargs:
            result_obj.add_metadata(**marker.kwargs)

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
        terminalreporter.write_line(f"\nTest: {test_name}")
        terminalreporter.write_line(f"Number of test cases: {len(results)}")

        y_true = []
        y_pred = []

        for result in results:
            if result.expected is not None and result.actual is not None:
                y_true.append(str(result.expected))
                y_pred.append(str(result.actual))

        # Generate metrics report if we have true/predicted values
        if y_true and y_pred:
            # Use sklearn's classification_report directly
            report = classification_report(y_true, y_pred)

            terminalreporter.write_line("\nClassification Report:")
            terminalreporter.write_line(report)

    terminalreporter.write_sep("=", "End of LLM Evaluation Results")
