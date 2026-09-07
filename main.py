from analysis.analysis import calculate_pass_accuracy


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
