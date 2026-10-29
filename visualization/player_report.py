from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.analysis import get_player_analysis
from data_layer.player_stats import load_player_stats
from project_paths import EVENTS_FILE
from recommendation.recommendation import recommendation_metrics, recommend


def build_player_report(dataframe: pd.DataFrame, team_name: str) -> list[dict[str, Any]]:
    """Build player, metric, and recommendation rows for a team."""
    team = dataframe.loc[dataframe["team"].eq(team_name)]
    if team.empty:
        raise KeyError(f"Team not found: {team_name}")

    report = []
    team_stats = recommendation_metrics(dataframe, team_name)["team"]
    for player_name in team["player_name"]:
        player_metrics = get_player_analysis(dataframe, player_name)
        report.append(
            {
                "player": player_name,
                "metrics": player_metrics,
                "recommendations": recommend(
                    {"player": player_metrics, "team": team_stats}
                ),
            }
        )
    return report


def print_player_report(dataframe: pd.DataFrame, team_name: str) -> None:
    """Print a compact console report for every player in a team."""
    print(f"\nPlayers' Report - {team_name}")
    print("=" * 72)
    for row in build_player_report(dataframe, team_name):
        metrics = row["metrics"]
        print(f"\nPlayer: {row['player']} | Position: {metrics['position']}")
        print(
            "Metrics: "
            f"Pass Accuracy {metrics['pass_accuracy']:.1%}، "
            f"Shots on Target {metrics['shots_on_target_pct']:.1%}، "
            f"Shots {metrics['shots']}، "
            f"Tackles Won {metrics['tackles_won']}، "
            f"Interceptions Won {metrics['interceptions_won']}"
        )
        tips = row["recommendations"]
        print("Recommendations:")
        if tips:
            for tip in tips:
                print(f"  - {tip}")
        else:
            print("There are no Urgent Recommendations.")


def main() -> None:
    dataframe = load_player_stats(EVENTS_FILE)
    print_player_report(dataframe, "Barcelona")


if __name__ == "__main__":
    main()
