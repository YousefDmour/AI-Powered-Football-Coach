"""Terminal scoreboard for simulated live event playback."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

from live.replay import Replay, load_events
from project_paths import EVENTS_FILE


def show(replay, alerts, stopped=False):
    snapshot = replay.state.snapshot()
    minute, second = divmod(snapshot["match_second"], 60)
    status = "PAUSED" if replay.paused else "PLAYING"
    if replay.finished:
        status = "FINISHED"
    elif stopped:
        status = "STOPPED"
    speed = "instant" if replay.speed == 0 else f"{replay.speed:g}x"
    print(f"\n[SIMULATED LIVE] {minute:02}:{second:02} | Period {snapshot['period']} | "
          f"{status} | {speed} | Events {snapshot['events_processed']}")
    for team, stats in snapshot["teams"].items():
        accuracy = (f"{stats['pass_accuracy']:.1%}"
                    if stats["pass_accuracy"] is not None else "N/A")
        print(f"  {team}: goals {stats['goals']} | "
              f"passes {stats['passes_completed']}/{stats['passes_attempted']} ({accuracy}) | "
              f"shots {stats['shots']} | on target {stats['shots_on_target']}")
    for alert in alerts:
        alert_minute, alert_second = divmod(alert["match_second"], 60)
        print(f"  COACHING HYPOTHESIS [{alert_minute:02}:{alert_second:02} | "
              f"{alert['team']}]: {alert['advice']} {alert['evidence']}")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Simulated live match-event analysis")
    parser.add_argument("--events", type=Path, default=EVENTS_FILE)
    parser.add_argument("--speed", type=float, default=120, help="Match seconds per real second; 0 = instant")
    parser.add_argument("--refresh", type=float, default=1, help="Display interval in real seconds")
    parser.add_argument("--export", type=Path, help="Save final/partial statistics and alert history as JSON")
    args = parser.parse_args(argv)
    if not math.isfinite(args.refresh) or args.refresh <= 0:
        parser.error("--refresh must be finite and positive")
    try:
        replay = Replay(load_events(args.events), speed=args.speed)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    keyboard = None
    if sys.stdin.isatty():
        try:
            import msvcrt
            keyboard = msvcrt
        except ImportError:
            pass
    print("Source: recorded event JSON (not live video). Thresholds are experimental.")
    print("Windows terminal: Space = pause/resume | + / - = speed | q = stop. Ctrl+C also stops.")
    previous = time.monotonic()
    next_display = previous
    history, pending = [], []
    try:
        while not replay.finished:
            now = time.monotonic()
            # Deliver the interval under the previous control state first.
            alerts = replay.advance(now - previous)
            previous = now
            history.extend(alerts)
            pending.extend(alerts)
            if keyboard and keyboard.kbhit():
                key = keyboard.getwch()
                if key.lower() == "q":
                    break
                if key == " ":
                    replay.paused = not replay.paused
                elif key == "+":
                    replay.speed = min((replay.speed or 1) * 2, 4096)
                elif key == "-":
                    replay.speed = max((replay.speed or 1) / 2, 0.25)
                next_display = now
            if now >= next_display or replay.finished:
                show(replay, pending)
                pending.clear()
                next_display = now + args.refresh
            if not replay.finished:
                time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    if not replay.finished:
        show(replay, pending, stopped=True)
        print("Replay stopped; statistics cover only delivered events.")
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps({"source": "simulated_event_replay",
            "completed": replay.finished, "statistics": replay.state.snapshot(),
            "alerts": history}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {args.export}")


if __name__ == "__main__":
    main()
