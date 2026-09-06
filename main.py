from analysis.analysis import calculate_pass_accuracy


if __name__ == "__main__":
    result = calculate_pass_accuracy("data_layer/15946.json", "Barcelona")
    print(f"Barcelona pass accuracy: {result['pass_accuracy_pct']}%")
    print(f"Successful passes: {result['successful_passes']} / {result['total_passes']}")
