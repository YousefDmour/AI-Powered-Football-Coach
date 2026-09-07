from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def _ratio(value: Any, default: float = 0.0) -> float:
	"""Return a metric as a ratio, accepting either 0..1 or percentage values."""
	try:
		number = float(value)
	except (TypeError, ValueError):
		return default
	return number / 100 if number > 1 else number


def recommend(metrics: Mapping[str, Any]) -> list[str]:
	"""Return coaching tips for player and team metrics.

	The mapping may contain flat metrics or ``player`` and ``team`` mappings.
	Ratios such as ``pass_accuracy`` and ``shots_on_target_pct`` can be given
	either as values between 0 and 1 or as percentages between 0 and 100.
	"""
	player = metrics.get("player", metrics)
	team = metrics.get("team", {})
	tips: list[str] = []
	position = str(player.get("position", "")).lower()
	is_forward = "forward" in position

	if _ratio(player.get("pass_accuracy")) < 0.75:
		tips.append("Focused short-passing drill, 15 minutes daily")
	if is_forward and _ratio(player.get("shots_on_target_pct")) < 0.30:
		tips.append("Shooting-at-goal drill from various angles")

	dribbles = float(player.get("dribbles_attempted", 0) or 0)
	completed_dribbles = float(player.get("dribbles_completed", 0) or 0)
	if dribbles >= 5 and completed_dribbles / dribbles < 0.5:
		tips.append("Dribbling and change-of-direction drills under pressure")

	duels = float(player.get("duels", 0) or 0)
	if duels >= 5 and float(player.get("duels_won", 0) or 0) / duels < 0.5:
		tips.append("Contact and balance recovery drills")

	if _ratio(team.get("pass_accuracy")) < 0.80:
		tips.append("Improving build-up play quality through a ball-circulation drill under pressure")
	if _ratio(team.get("shots_on_target_pct")) < 0.35:
		tips.append("Increasing team-based finishing drills inside the penalty area")
	if team.get("defensive_actions_successful", 0) < team.get("defensive_actions", 0) * 0.5:
		tips.append("Organizing collective pressing and defensive coverage after losing possession.")

	return tips


def team_metrics(dataframe: pd.DataFrame, team_name: str) -> dict[str, float | int]:
	"""Aggregate the metrics needed by :func:`recommend` for one team."""
	team = dataframe.loc[dataframe["team"].eq(team_name)]
	passes = int(team["passes_attempted"].sum())
	shots = int(team["shots"].sum())
	duels = int(team["duels"].sum())
	defensive_actions = int(team["interceptions"].sum() + team["tackles"].sum())
	successful_defensive_actions = int(team["interceptions_won"].sum() + team["tackles_won"].sum())
	return {
		"pass_accuracy": float(team["passes_completed"].sum() / passes) if passes else 0.0,
		"shots_on_target_pct": float(team["shots_on_target"].sum() / shots) if shots else 0.0,
		"defensive_actions": defensive_actions,
		"defensive_actions_successful": successful_defensive_actions,
		"duels": duels,
		"duels_won": int(team["duels_won"].sum()),
	}


def recommendation_metrics(
	dataframe: pd.DataFrame, team_name: str, player_name: str | None = None
) -> dict[str, dict[str, Any]]:
	"""Build the nested metrics mapping consumed by :func:`recommend`."""
	team = dataframe.loc[dataframe["team"].eq(team_name)]
	if team.empty:
		raise KeyError(f"Team not found: {team_name}")

	metrics: dict[str, dict[str, Any]] = {"team": team_metrics(dataframe, team_name)}
	if player_name is not None:
		player = team.loc[team["player_name"].eq(player_name)]
		if player.empty:
			raise KeyError(f"Player not found: {player_name}")
		player_metrics = player.iloc[0].to_dict()
		player_metrics["pass_accuracy"] = (
			float(player_metrics["passes_completed"] / player_metrics["passes_attempted"])
			if player_metrics["passes_attempted"]
			else 0.0
		)
		metrics["player"] = player_metrics
	return metrics
