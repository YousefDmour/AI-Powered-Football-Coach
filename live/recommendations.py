"""Evidence-gated coaching hypotheses; thresholds are configurable heuristics."""


class LiveRecommendations:
    def __init__(self, cooldown_seconds=300):
        self.cooldown_seconds = cooldown_seconds
        self.last_sent = {}

    def evaluate(self, snapshot):
        alerts = []
        for team, stats in snapshot["recent"].items():
            rules = (
                ("passing", "passes_completed", "passes_attempted", 20, 0.80,
                 "Consider shorter passing options and review support around the ball."),
                ("shooting", "shots_on_target", "shots", 5, 0.35,
                 "Review shot selection and opportunities to create clearer chances."),
                ("defending", "defensive_actions_successful", "defensive_actions", 10, 0.50,
                 "Review defensive support and coordination before changing the press."),
            )
            for rule, numerator, denominator, minimum, threshold, advice in rules:
                attempted = stats[denominator]
                successful = stats[numerator]
                if attempted < minimum or successful / attempted >= threshold:
                    continue
                key = (team, rule)
                elapsed = snapshot["elapsed"]
                if elapsed - self.last_sent.get(key, float("-inf")) < self.cooldown_seconds:
                    continue
                self.last_sent[key] = elapsed
                alerts.append({
                    "team": team, "rule": rule, "advice": advice,
                    "successful": successful, "attempted": attempted,
                    "threshold": threshold, "window_seconds": snapshot["window_seconds"],
                    "match_second": snapshot["match_second"],
                    "evidence": f"{successful}/{attempted} ({successful / attempted:.1%}) "
                                f"in the last {snapshot['window_seconds'] / 60:g} minutes; "
                                f"configured threshold {threshold:.0%}.",
                })
        return alerts
