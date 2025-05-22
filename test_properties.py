from hypothesis import given, strategies as st
from model import extract_binary_features, contains_phishing_keywords

@given(st.text())
def test_extract_binary_features_returns_three_binaries(text):
    features = extract_binary_features(text)
    assert isinstance(features, list), "Output should be a list"
    assert len(features) == 3, "Should return 3 features"
    assert all(isinstance(x, int) for x in features), "All features must be integers"
    assert all(x in [0, 1] for x in features), "All values must be binary (0 or 1)"

@given(st.text())
def test_contains_phishing_keywords_always_returns_bool(text):
    result = contains_phishing_keywords(text)
    assert isinstance(result, bool), "Should always return a boolean"
