from project_paths import EVENTS_FILE
from analysis.analysis import calculate_pass_accuracy


def test_calculate_pass_accuracy_from_barcelona_match():
    result = calculate_pass_accuracy(EVENTS_FILE, "Barcelona")

    assert result["total_passes"] == 917
    assert result["successful_passes"] == 820
    assert result["unsuccessful_passes"] == 97
    assert round(result["pass_accuracy_pct"], 2) == 89.42
