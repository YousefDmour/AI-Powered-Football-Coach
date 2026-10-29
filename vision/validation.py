"""Source-window checks and an explicit gate before experimental possession work."""

from pathlib import Path


def validate_cache(metadata, video, start_frame, max_frames):
    cached_start = metadata.get('source_start_frame',0)
    cached_end = cached_start+metadata['processed_frames']
    if (Path(metadata['source']).resolve()!=Path(video).resolve()
            or start_frame<cached_start or start_frame+max_frames>cached_end):
        raise ValueError('Cached player tracks must match the source and requested original-frame interval')


def validation_gate(ball, players):
    """Prototype thresholds fixed before viewing predictions; inadequate evidence fails."""
    visible=ball['visible_annotated_frames']
    invisible=ball['reviewed_not_visible_frames']
    transitions=players['player_identity_transitions_checked']
    checks={
        'at_least_10_visible_ball_frames':visible>=10,
        'at_least_5_not_visible_ball_frames':invisible>=5,
        'visible_ball_recall_at_least_80_percent':visible>0 and ball['correct_centers']/visible>=.8,
        'wrong_ball_centers_at_most_10_percent':visible>0 and ball['wrong_centers_on_visible_frames']/visible<=.1,
        'false_visibility_at_most_10_percent':invisible>0 and ball['false_visible_on_not_visible_frames']/invisible<=.1,
        'at_least_10_player_transitions':transitions>=10,
        'player_id_changes_at_most_10_percent':transitions>0 and players['observed_player_id_changes']/transitions<=.1,
    }
    return {'passed':all(checks.values()),'checks':checks,
            'decision':'eligible_for_experimental_possession' if all(checks.values()) else 'improve_or_extend_validation_before_possession',
            'limits':'Same-video, assistant-reviewed sample; passing is a prototype gate, not independently validated real-world accuracy.'}
