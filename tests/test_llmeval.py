import pytest


def test_llmeval_marker_registered(pytester):
    """Test that the llmeval marker is properly registered."""
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
            "*@pytest.mark.llmeval:*",
        ]
    )
    assert result.ret == 0


def test_llmeval_result_fixture(pytester):
    """Test that the llmeval_result fixture works properly."""
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


def test_llmeval_metadata(pytester):
    """Test that metadata can be added to llmeval results."""
    pytester.makepyfile(
        """
        import pytest
        
        @pytest.mark.llmeval(category="sentiment")
        def test_with_metadata(llmeval_result):
            assert llmeval_result.metadata["category"] == "sentiment"
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


def test_llmeval_classification_report(pytester):
    """Test that the classification report is generated correctly."""
    pytester.makepyfile(
        """
        import pytest
        
        @pytest.mark.llmeval
        @pytest.mark.parametrize("expected,actual", [
            ("positive", "positive"),
            ("positive", "negative"),
            ("negative", "negative"),
            ("neutral", "neutral"),
        ])
        def test_sentiment(llmeval_result, expected, actual):
            llmeval_result.set_result(expected, actual)
            # The test itself always passes - we're just collecting results
            assert True
    """
    )

    result = pytester.runpytest("-v")
    result.stdout.fnmatch_lines(
        [
            "*precision*recall*f1-score*support*",
            "*negative*",
            "*neutral*",
            "*positive*",
            "*accuracy*",
        ]
    )
    assert result.ret == 0


def test_multiple_llmeval_tests(pytester):
    """Test that multiple llmeval tests are reported separately."""
    pytester.makepyfile(
        """
        import pytest
        
        @pytest.mark.llmeval
        def test_binary(llmeval_result):
            llmeval_result.set_result("yes", "yes")
            assert True
        
        @pytest.mark.llmeval
        def test_multiclass(llmeval_result):
            llmeval_result.set_result("class_a", "class_b")
            assert True
    """
    )

    result = pytester.runpytest("-v")
    result.stdout.re_match_lines(
        [
            r".*Test: test_binary.*",
            r".*Test: test_multiclass.*",
        ]
    )
    assert result.ret == 0
