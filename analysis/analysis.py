import json
from pathlib import Path
from typing import Any, Dict, List


def _is_successful_pass(event: Dict[str, Any]) -> bool:
    """Return True for successful passes in the Opta-style match event JSON."""
    outcome = event.get("pass", {}).get("outcome")
    if not isinstance(outcome, dict):
        return True

    outcome_name = outcome.get("name")
    if outcome_name is None:
        return True

    successful_outcomes = {"Complete", "Completed", "Successful"}
    return outcome_name in successful_outcomes


def calculate_pass_accuracy(file_path: str | Path, team_name: str) -> Dict[str, Any]:
    """Calculate total, successful, unsuccessful and accuracy for a team."""
    with open(file_path, "r", encoding="utf-8") as source:
        data = json.load(source)

    team_passes: List[Dict[str, Any]] = []
    for event in data:
        if not isinstance(event, dict):
            continue
        if event.get("type", {}).get("name") != "Pass":
            continue
        if event.get("team", {}).get("name") != team_name:
            continue
        team_passes.append(event)

    total_passes = len(team_passes)
    successful_passes = sum(1 for event in team_passes if _is_successful_pass(event))
    unsuccessful_passes = total_passes - successful_passes
    pass_accuracy_pct = (successful_passes / total_passes * 100) if total_passes else 0.0

    player_stats: Dict[str, Dict[str, int]] = {}
    for event in team_passes:
        player_name = event.get("player", {}).get("name", "Unknown")
        stats = player_stats.setdefault(
            player_name,
            {"total": 0, "successful": 0, "unsuccessful": 0},
        )
        stats["total"] += 1
        if _is_successful_pass(event):
            stats["successful"] += 1
        else:
            stats["unsuccessful"] += 1

    player_accuracy = []
    for player_name, stats in sorted(
        player_stats.items(),
        key=lambda item: item[1]["successful"],
        reverse=True,
    ):
        total = stats["total"]
        accuracy = (stats["successful"] / total * 100) if total else 0.0
        player_accuracy.append(
            {
                "player": player_name,
                "total_passes": total,
                "successful_passes": stats["successful"],
                "unsuccessful_passes": stats["unsuccessful"],
                "pass_accuracy_pct": round(accuracy, 2),
            }
        )

    return {
        "team": team_name,
        "total_passes": total_passes,
        "successful_passes": successful_passes,
        "unsuccessful_passes": unsuccessful_passes,
        "pass_accuracy_pct": round(pass_accuracy_pct, 2),
        "player_accuracy": player_accuracy,
    }


if __name__ == "__main__":
    result = calculate_pass_accuracy("data_layer/15946.json", "Barcelona")
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

