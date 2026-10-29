import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_layer.player_stats import load_player_stats
from project_paths import EVENTS_FILE


def calculate_pass_accuracy(file_path: str | Path, team_name: str) -> dict[str, Any]:
    """Calculate team passing statistics from the standardized player DataFrame."""
    dataframe = load_player_stats(file_path)
    team_stats = dataframe.loc[dataframe["team"].eq(team_name)]
    total_passes = int(team_stats["passes_attempted"].sum())
    successful_passes = int(team_stats["passes_completed"].sum())
    unsuccessful_passes = int(team_stats["miss_passes"].sum())
    pass_accuracy_pct = (successful_passes / total_passes * 100) if total_passes else 0.0

    player_accuracy = team_stats[
        ["player_name", "passes_attempted", "passes_completed", "miss_passes", "pass_accuracy_pct"]
    ].rename(
        columns={
            "player_name": "player",
            "passes_attempted": "total_passes",
            "passes_completed": "successful_passes",
            "miss_passes": "unsuccessful_passes",
        }
    ).sort_values("successful_passes", ascending=False).to_dict("records")

    return {
        "team": team_name,
        "total_passes": total_passes,
        "successful_passes": successful_passes,
        "unsuccessful_passes": unsuccessful_passes,
        "pass_accuracy_pct": round(pass_accuracy_pct, 2),
        "player_accuracy": player_accuracy,
    }


_METRIC_ALIASES = {
    "passes": "passes_attempted",
    "shots": "shots",
    "shots_on_target": "shots_on_target",
    "tackles": "tackles",
    "tackles_won": "tackles_won",
    "interceptions": "interceptions",
    "interceptions_won": "interceptions_won",
    "touches": "touches",
    "carry_distance": "carry_distance",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def get_player_metric(
    dataframe: pd.DataFrame, player_name: str, event_type: str
) -> int | float:
    """Return one metric for a player from the standardized player DataFrame."""
    matching = dataframe.loc[dataframe["player_name"].eq(player_name)]
    if matching.empty:
        raise KeyError(f"Player not found: {player_name}")

    metric = _METRIC_ALIASES.get(event_type.lower(), event_type)
    if metric not in dataframe.columns:
        event_column = f"event_{_slug(event_type)}_count"
        metric = event_column if event_column in dataframe.columns else metric
    if metric not in dataframe.columns:
        raise ValueError(f"Unknown player metric: {event_type}")
    value = matching.iloc[0][metric]
    return value.item() if hasattr(value, "item") else value


def get_player_analysis(dataframe: pd.DataFrame, player_name: str) -> dict[str, Any]:
    """Return the stable player-analysis schema for recommendations or UI."""
    matching = dataframe.loc[dataframe["player_name"].eq(player_name)]
    if matching.empty:
        raise KeyError(f"Player not found: {player_name}")
    row = matching.iloc[0]
    return {
        "player": row["player_name"],
        "team": row["team"],
        "position": row["position"],
        "pass_accuracy": float(
            round(row["passes_completed"] / row["passes_attempted"], 4)
        ) if row["passes_attempted"] else 0.0,
        "shots": int(row["shots"]),
        "shots_on_target": int(row["shots_on_target"]),
        "shots_on_target_pct": float(row["shots_on_target_pct"]),
        "tackles_won": int(row["tackles_won"]),
        "interceptions_won": int(row["interceptions_won"]),
        "defensive_actions_successful": int(row["defensive_actions_successful"]),
        "touches": int(row["touches"]),
        "carry_distance": float(row["carry_distance"]),
    }


if __name__ == "__main__":
    result = calculate_pass_accuracy(EVENTS_FILE, "Barcelona")
    print(f"Team: {result['team']}")
    print(f"Total passes: {result['total_passes']}")
    print(f"Successful passes: {result['successful_passes']}")
    print(f"Unsuccessful passes: {result['unsuccessful_passes']}")
    print(f"Pass accuracy: {result['pass_accuracy_pct']}%")
    print("Top players by pass accuracy:")
    for player in result["player_accuracy"][:5]:
        print(
            f"- {player['player']}: {player['pass_accuracy_pct']}% "
            f"({player['successful_passes']}/{player['total_passes']})"
        )
