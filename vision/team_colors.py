"""Conservative shirt-color classification for a calibrated clip, not player identity."""

from collections import Counter, defaultdict, deque

import cv2
import numpy as np


PALETTE = {"team_a": (230, 130, 50), "team_b": (230, 230, 230),
           "official_candidate": (20, 220, 240), "other_kit": (180, 100, 200),
           "unknown": (140, 140, 140)}


def shirt_features(frame, bbox):
    """Use the central upper torso; avoid the head, shorts and box background."""
    x1, y1, x2, y2 = map(float, bbox)
    height, width = frame.shape[:2]
    left, right = max(0, int(x1 + (x2-x1)*.2)), min(width, int(x2-(x2-x1)*.2))
    top, bottom = max(0, int(y1+(y2-y1)*.18)), min(height, int(y1+(y2-y1)*.55))
    crop = frame[top:bottom, left:right]
    if right <= left or bottom <= top or crop.size < 16*3:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = hsv.T
    # Remove turf but reject boxes with too little remaining shirt evidence.
    foreground = ~((h >= 35) & (h <= 90) & (s >= 45)) & (v >= 25)
    if int(foreground.sum()) < 16 or float(foreground.mean()) < .20:
        return None
    h, s, v = hsv[foreground].T
    return {
        "blue": float(np.mean((h >= 95) & (h <= 179) & (s >= 40) & (v >= 25))),
        "white": float(np.mean((s <= 65) & (v >= 155))),
        "yellow": float(np.mean((h >= 18) & (h <= 38) & (s >= 100) & (v >= 120))),
        "red": float(np.mean(((h <= 12) | (h >= 170)) & (s >= 110) & (v >= 65))),
    }


def classify_shirt(features):
    """Clip-specific thresholds. Support is a pixel fraction, not a probability."""
    if features is None:
        return "unknown", 0.0
    if features["yellow"] >= .30:
        return "official_candidate", features["yellow"]
    if features["red"] >= .45 and features["blue"] < .12:
        return "other_kit", features["red"]
    scores = sorted(((features["blue"], "team_a"), (features["white"], "team_b")), reverse=True)
    support, label = scores[0]
    if support < .30 or support - scores[1][0] < .12:
        return "unknown", support
    return label, support


class TeamSmoother:
    """Vote over recent observations of the same tracker ID; never merge identities."""

    def __init__(self, history=5):
        self.votes = defaultdict(lambda: deque(maxlen=history))

    def update(self, track_id, raw_label):
        votes = self.votes[track_id]
        votes.append(raw_label)
        if raw_label not in {"team_a", "team_b"}:
            return raw_label
        best, count = Counter(votes).most_common(1)[0]
        if best in {"team_a", "team_b"} and count >= 2 and count / len(votes) >= .6:
            return best
        return "unknown"

    def reset(self):
        self.votes.clear()
