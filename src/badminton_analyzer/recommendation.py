from __future__ import annotations

from typing import Dict, List


def generate_recommendations(predicted_label: str, feature_values: Dict[str, float]) -> List[str]:
    recommendations: List[str] = []

    hip_speed_mean = feature_values.get("hip_speed_mean", 0.0)
    court_coverage_x = feature_values.get("court_coverage_x", 0.0)
    court_coverage_y = feature_values.get("court_coverage_y", 0.0)
    stance_ratio_mean = feature_values.get("stance_ratio_mean", 0.0)
    lunge_rate = feature_values.get("lunge_rate", 0.0)
    direction_change_rate = feature_values.get("direction_change_rate", 0.0)

    if predicted_label == "beginner":
        recommendations.append("Practice split-step timing before every opponent shot.")
        recommendations.append("Use 4-corner footwork drill for 15-20 minutes daily.")
        recommendations.append("Record and correct racket preparation to reduce slow responses.")
    elif predicted_label == "intermediate":
        recommendations.append("Improve transition speed from rear-court to net with shadow drills.")
        recommendations.append("Add multi-shuttle drills to improve endurance under fast rallies.")
    elif predicted_label == "pro":
        recommendations.append("Use opponent-pattern analysis and deception drills for match advantage.")
        recommendations.append("Track rally-level fatigue and optimize pace variation.")
    else:
        recommendations.append("Stabilize baseline movement with repeated shadow footwork patterns.")

    if hip_speed_mean < 0.01:
        recommendations.append("Increase explosive movement using resisted lateral shuffle training.")

    if court_coverage_x < 0.15 or court_coverage_y < 0.15:
        recommendations.append("Expand court coverage with full-court ghosting sessions.")

    if stance_ratio_mean < 1.1:
        recommendations.append("Maintain wider ready stance to improve balance during direction changes.")

    if lunge_rate < 0.05:
        recommendations.append("Practice controlled lunge and recover drills to strengthen net play.")

    if direction_change_rate < 0.1:
        recommendations.append("Add agility ladder and reaction cue drills to increase re-direction speed.")

    return recommendations
