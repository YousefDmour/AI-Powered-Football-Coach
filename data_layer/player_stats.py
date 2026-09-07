"""Build player-level statistics from a StatsBomb/Opta-style event file."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def _name(value: Any) -> str | None:
    if isinstance(value, dict):
        return value.get("name")
    return None


def _player(event: dict[str, Any]) -> tuple[int | str, dict[str, Any]] | None:
    player = event.get("player")
    if not isinstance(player, dict):
        return None
    player_id = player.get("id")
    player_name = player.get("name")
    if player_id is None or player_name is None:
        return None
    return player_id, player


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _successful_pass(event: dict[str, Any]) -> bool:
    outcome = event.get("pass", {}).get("outcome")
    if not isinstance(outcome, dict) or outcome.get("name") is None:
        return True
    return outcome["name"] in {"Complete", "Completed", "Successful"}


def _successful_dribble(event: dict[str, Any]) -> bool:
    outcome = event.get("dribble", {}).get("outcome")
    return _name(outcome) in {"Complete", "Won"}


def _successful_duel(event: dict[str, Any]) -> bool:
    duel_type = _name(event.get("duel", {}).get("type"))
    return duel_type not in {"Aerial Lost", "Tackle", "Challenge Lost"}


def _add_event_count(stats: dict[str, Any], event_type: str) -> None:
    column = f"event_{_slug(event_type)}_count"
    stats[column] = stats.get(column, 0) + 1


def _distance(start: Any, end: Any) -> float:
    if not isinstance(start, list) or not isinstance(end, list):
        return 0.0
    if len(start) < 2 or len(end) < 2:
        return 0.0
    return math.hypot(end[0] - start[0], end[1] - start[1])


def load_player_stats(file_path: str | Path) -> pd.DataFrame:
    """Return one row per player with match metadata and event statistics.

    The source is an event list containing ``Starting XI`` and action events.
    The ``foot_played_with`` column is inferred from recorded action body parts;
    the source does not provide a preferred-foot player profile field.
    """
    with Path(file_path).open("r", encoding="utf-8") as source:
        events = json.load(source)

    if not isinstance(events, list):
        raise ValueError("Expected the JSON file to contain a list of events")

    players: dict[int | str, dict[str, Any]] = {}
    body_parts: defaultdict[int | str, Counter[str]] = defaultdict(Counter)
    event_players = {
        event.get("id"): event.get("player", {}).get("id")
        for event in events
        if isinstance(event, dict)
        and event.get("id") is not None
        and isinstance(event.get("player"), dict)
    }

    def ensure_player(
        player_id: int | str,
        player_name: str,
        team: str | None = None,
        position: str | None = None,
    ) -> dict[str, Any]:
        stats = players.setdefault(
            player_id,
            {
                "player_id": player_id,
                "player_name": player_name,
                "team": team,
                "position": position,
                "jersey_number": None,
                "foot_played_with": None,
                "left_foot_actions": 0,
                "right_foot_actions": 0,
                "head_actions": 0,
                "other_body_part_actions": 0,
                "passes_attempted": 0,
                "passes_completed": 0,
                "miss_passes": 0,
                "pass_accuracy_pct": 0.0,
                "goals": 0,
                "assists": 0,
                "shots": 0,
                "shots_on_target": 0,
                "shots_off_target": 0,
                "shots_blocked": 0,
                "shots_on_target_pct": 0.0,
                "expected_goals": 0.0,
                "dribbles_attempted": 0,
                "dribbles_completed": 0,
                "duels": 0,
                "duels_won": 0,
                "duels_lost": 0,
                "tackles": 0,
                "tackles_won": 0,
                "interceptions": 0,
                "interceptions_won": 0,
                "defensive_actions_successful": 0,
                "clearances": 0,
                "blocks": 0,
                "ball_recoveries": 0,
                "pressures": 0,
                "ball_receipts": 0,
                "carries": 0,
                "carry_distance": 0.0,
                "touches": 0,
                "dispossessed": 0,
                "miscontrols": 0,
                "fouls_committed": 0,
                "fouls_won": 0,
                "yellow_cards": 0,
                "red_cards": 0,
                "errors": 0,
                "goalkeeper_actions": 0,
                "substitutions_in": 0,
                "substitutions_out": 0,
            },
        )
        if team and not stats["team"]:
            stats["team"] = team
        if position and not stats["position"]:
            stats["position"] = position
        return stats

    for event in events:
        if not isinstance(event, dict):
            continue

        event_type = _name(event.get("type"))
        event_team = _name(event.get("team"))

        if event_type == "Starting XI":
            tactics = event.get("tactics", {})
            lineup = tactics.get("lineup", []) if isinstance(tactics, dict) else []
            for lineup_player in lineup:
                if not isinstance(lineup_player, dict):
                    continue
                player = lineup_player.get("player", {})
                if not isinstance(player, dict) or player.get("id") is None:
                    continue
                position = _name(lineup_player.get("position"))
                stats = ensure_player(
                    player["id"], player.get("name", "Unknown"), event_team, position
                )
                stats["jersey_number"] = lineup_player.get("jersey_number")
            continue

        player_info = _player(event)
        if player_info is None:
            continue
        player_id, player = player_info
        stats = ensure_player(
            player_id,
            player["name"],
            event_team,
            _name(event.get("position")),
        )
        if event_type:
            _add_event_count(stats, event_type)

        detail = event.get(_slug(event_type)) if event_type else None
        if isinstance(detail, dict):
            body_part = _name(detail.get("body_part"))
            if body_part:
                body_parts[player_id][body_part] += 1

        if event.get("location") is not None and event_type in {
            "Ball Receipt*",
            "Carry",
            "Pass",
            "Shot",
            "Dribble",
            "Duel",
            "Ball Recovery",
            "Dispossessed",
            "Miscontrol",
        }:
            stats["touches"] += 1

        if event_type == "Pass":
            stats["passes_attempted"] += 1
            if _successful_pass(event):
                stats["passes_completed"] += 1
            else:
                stats["miss_passes"] += 1
        elif event_type == "Shot":
            stats["shots"] += 1
            shot = event.get("shot", {})
            outcome = _name(shot.get("outcome")) if isinstance(shot, dict) else None
            if outcome == "Goal":
                stats["goals"] += 1
                key_pass_id = shot.get("key_pass_id") if isinstance(shot, dict) else None
                assister_id = event_players.get(key_pass_id)
                if assister_id in players:
                    players[assister_id]["assists"] += 1
            elif outcome in {"Saved", "Saved To Post"}:
                stats["shots_on_target"] += 1
            elif outcome == "Blocked":
                stats["shots_blocked"] += 1
            else:
                stats["shots_off_target"] += 1
            if isinstance(shot, dict) and isinstance(shot.get("statsbomb_xg"), (int, float)):
                stats["expected_goals"] += shot["statsbomb_xg"]
        elif event_type == "Dribble":
            stats["dribbles_attempted"] += 1
            stats["dribbles_completed"] += _successful_dribble(event)
        elif event_type == "Duel":
            stats["duels"] += 1
            if _successful_duel(event):
                stats["duels_won"] += 1
            else:
                stats["duels_lost"] += 1
        elif event_type == "Tackle":
            stats["tackles"] += 1
        elif event_type == "Interception":
            stats["interceptions"] += 1
            outcome = _name(event.get("interception", {}).get("outcome"))
            if outcome not in {"Lost Out", "Lost"}:
                stats["interceptions_won"] += 1
                stats["defensive_actions_successful"] += 1
        elif event_type == "Clearance":
            stats["clearances"] += 1
        elif event_type == "Block":
            stats["blocks"] += 1
        elif event_type == "Ball Recovery":
            stats["ball_recoveries"] += 1
        elif event_type == "Pressure":
            stats["pressures"] += 1
        elif event_type == "Ball Receipt*":
            stats["ball_receipts"] += 1
        elif event_type == "Carry":
            stats["carries"] += 1
            carry = event.get("carry", {})
            if isinstance(carry, dict):
                stats["carry_distance"] += _distance(
                    event.get("location"), carry.get("end_location")
                )
        elif event_type == "Dispossessed":
            stats["dispossessed"] += 1
        elif event_type == "Miscontrol":
            stats["miscontrols"] += 1
        elif event_type == "Foul Committed":
            stats["fouls_committed"] += 1
            foul = event.get("foul_committed", {})
            card = _name(foul.get("card")) if isinstance(foul, dict) else None
            stats["yellow_cards"] += card == "Yellow Card"
            stats["red_cards"] += card in {"Red Card", "Second Yellow"}
        elif event_type == "Foul Won":
            stats["fouls_won"] += 1
        elif event_type == "Error":
            stats["errors"] += 1
        elif event_type == "Goal Keeper":
            stats["goalkeeper_actions"] += 1
        elif event_type == "Substitution":
            substitution = event.get("substitution", {})
            replacement = substitution.get("replacement", {}) if isinstance(substitution, dict) else {}
            stats["substitutions_out"] += 1
            if isinstance(replacement, dict) and replacement.get("id") is not None:
                replacement_stats = ensure_player(
                    replacement["id"], replacement.get("name", "Unknown"), event_team
                )
                replacement_stats["substitutions_in"] += 1

    for player_id, stats in players.items():
        total_passes = stats["passes_attempted"]
        stats["pass_accuracy_pct"] = round(
            stats["passes_completed"] / total_passes * 100, 2
        ) if total_passes else 0.0
        stats["shots_on_target_pct"] = round(
            stats["shots_on_target"] / stats["shots"], 4
        ) if stats["shots"] else 0.0
        parts = body_parts[player_id]
        stats["foot_played_with"] = parts.most_common(1)[0][0] if parts else None
        stats["left_foot_actions"] = parts["Left Foot"]
        stats["right_foot_actions"] = parts["Right Foot"]
        stats["head_actions"] = parts["Head"]
        stats["other_body_part_actions"] = sum(parts.values()) - sum(
            parts[name] for name in ("Left Foot", "Right Foot", "Head")
        )
        stats["expected_goals"] = round(stats["expected_goals"], 4)
        stats["carry_distance"] = round(stats["carry_distance"], 2)

    dataframe = pd.DataFrame(players.values())
    if dataframe.empty:
        return pd.DataFrame()
    return dataframe.sort_values(["team", "player_name"], na_position="last").reset_index(drop=True)


if __name__ == "__main__":
    df = load_player_stats(Path(__file__).parent / "15946.json")
    print(df)