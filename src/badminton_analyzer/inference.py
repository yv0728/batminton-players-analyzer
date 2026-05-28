from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from .pose_features import PoseFeatureExtractor
from .recommendation import generate_recommendations


class SkillInference:
    def __init__(self, model_path: str | Path):
        loaded = joblib.load(model_path)
        self.pipeline = loaded["pipeline"]
        self.label_encoder = loaded["label_encoder"]
        self.feature_columns = loaded["feature_columns"]
        self.extractor = PoseFeatureExtractor()

    @staticmethod
    def _clip01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    def _calculate_action_efficiency(self, features: Dict[str, float], confidence: float) -> Dict[str, float]:
        footwork_speed = self._clip01(features.get("hip_speed_mean", 0.0) / 0.02)
        court_coverage = self._clip01(
            (features.get("court_coverage_x", 0.0) + features.get("court_coverage_y", 0.0)) / 0.5
        )
        agility = self._clip01(features.get("direction_change_rate", 0.0) / 0.6)
        stance_control = self._clip01(1.0 - abs(features.get("stance_ratio_mean", 1.25) - 1.25) / 0.5)
        lunge_efficiency = self._clip01(features.get("lunge_rate", 0.0) / 0.25)
        wrist_speed = self._clip01(
            (
                features.get("left_wrist_speed_mean", 0.0)
                + features.get("right_wrist_speed_mean", 0.0)
            )
            / 0.05
        )

        # Weighted action efficiency score emphasizing movement quality and model certainty.
        weighted = (
            0.24 * footwork_speed
            + 0.2 * court_coverage
            + 0.16 * agility
            + 0.14 * stance_control
            + 0.12 * lunge_efficiency
            + 0.14 * wrist_speed
        )
        calibrated = self._clip01(0.8 * weighted + 0.2 * confidence)

        return {
            "efficiency_score": round(calibrated * 100.0, 2),
            "footwork_speed": round(footwork_speed * 100.0, 2),
            "court_coverage": round(court_coverage * 100.0, 2),
            "agility": round(agility * 100.0, 2),
            "stance_control": round(stance_control * 100.0, 2),
            "lunge_efficiency": round(lunge_efficiency * 100.0, 2),
            "stroke_readiness": round(wrist_speed * 100.0, 2),
        }

    def analyze_video(self, video_path: str | Path) -> Dict:
        features = self.extractor.extract_video_features(video_path)

        row = {k: features.get(k, 0.0) for k in self.feature_columns}
        X = pd.DataFrame([row])

        proba = self.pipeline.predict_proba(X)[0]
        pred_idx = int(proba.argmax())
        pred_label = str(self.label_encoder.inverse_transform([pred_idx])[0])
        confidence = float(proba[pred_idx])
        action_efficiency = self._calculate_action_efficiency(features, confidence)

        recommendations = generate_recommendations(pred_label, features)

        return {
            "video_path": str(video_path),
            "predicted_skill_level": pred_label,
            "confidence": confidence,
            "efficiency_score": action_efficiency["efficiency_score"],
            "action_efficiency_breakdown": {
                k: v for k, v in action_efficiency.items() if k != "efficiency_score"
            },
            "class_probabilities": {
                str(cls): float(prob)
                for cls, prob in zip(self.label_encoder.classes_, proba)
            },
            "movement_features": features,
            "recommendations": recommendations,
        }

    def save_report(self, report: Dict, output_path: str | Path) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
