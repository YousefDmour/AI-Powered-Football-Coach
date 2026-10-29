"""Run an offline report or simulated live event analysis with --live."""

import sys


if __name__ == "__main__":
    if "--dashboard" in sys.argv[1:]:
        from live.dashboard import main

        main([argument for argument in sys.argv[1:] if argument != "--dashboard"])
    elif "--live" in sys.argv[1:]:
        from live.cli import main

        main([argument for argument in sys.argv[1:] if argument != "--live"])
    else:
        from visualization.player_report import main

        main()
