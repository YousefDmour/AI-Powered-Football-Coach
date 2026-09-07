from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_layer.player_stats import load_player_stats
from recommendation.recommendation import recommendation_metrics, recommend


def test_recommend_flags_weak_player_and_team_metrics():
    tips = recommend(
        {
            "player": {
				"position": "Center Forward",
                "pass_accuracy": 0.70,
                "shots_on_target_pct": 0.20,
                "dribbles_attempted": 6,
                "dribbles_completed": 2,
                "duels": 6,
                "duels_won": 2,
            },
            "team": {
                "pass_accuracy": 0.70,
                "shots_on_target_pct": 0.20,
                "defensive_actions": 10,
                "defensive_actions_successful": 4,
            },
        }
    )

    assert len(tips) == 7


def test_shooting_recommendation_requires_forward_position():
    forward_tips = recommend(
        {"player": {"position": "Left Center Forward", "shots_on_target_pct": 0.20}}
    )
    non_forward_tips = recommend(
        {"player": {"position": "Left Center Midfield", "shots_on_target_pct": 0.20}}
    )

    assert "Shooting-at-goal drill from various angles" in forward_tips
    assert "Shooting-at-goal drill from various angles" not in non_forward_tips


def test_recommendation_metrics_aggregates_match_dataframe():
    dataframe = load_player_stats("data_layer/15946.json")

    metrics = recommendation_metrics(dataframe, "Barcelona")

    assert round(metrics["team"]["pass_accuracy"], 4) == 0.8942
    assert metrics["team"]["defensive_actions"] >= metrics["team"]["defensive_actions_successful"]