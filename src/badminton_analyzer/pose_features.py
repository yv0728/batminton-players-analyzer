from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:
    from mediapipe.python import solutions as mp_solutions
except Exception:
    mp_solutions = None


@dataclass
class PoseExtractionConfig:
    visibility_threshold: float = 0.5
    max_frames: int = 2500
    frame_stride: int = 2


class PoseFeatureExtractor:
    def __init__(self, config: PoseExtractionConfig | None = None):
        self.config = config or PoseExtractionConfig()
        pose_api = mp.solutions.pose if hasattr(mp, "solutions") else mp_solutions.pose
        self.pose = pose_api.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _safe_point(self, landmarks, idx: int) -> np.ndarray | None:
        point = landmarks[idx]
        if point.visibility < self.config.visibility_threshold:
            return None
        return np.array([point.x, point.y], dtype=np.float32)

    @staticmethod
    def _distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return np.nan
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _angle(a: np.ndarray | None, b: np.ndarray | None, c: np.ndarray | None) -> float:
        if a is None or b is None or c is None:
            return np.nan
        ba = a - b
        bc = c - b
        norm = (np.linalg.norm(ba) * np.linalg.norm(bc))
        if norm < 1e-6:
            return np.nan
        cosine = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    def extract_video_features(self, video_path: str | Path) -> Dict[str, float]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        left_ankle_positions: List[np.ndarray] = []
        right_ankle_positions: List[np.ndarray] = []
        hip_center_positions: List[np.ndarray] = []
        left_wrist_positions: List[np.ndarray] = []
        right_wrist_positions: List[np.ndarray] = []
        knee_angles: List[float] = []
        stance_widths: List[float] = []
        shoulder_widths: List[float] = []

        frame_idx = 0
        processed = 0

        while capture.isOpened() and processed < self.config.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % self.config.frame_stride != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb)
            if not result.pose_landmarks:
                continue

            lm = result.pose_landmarks.landmark
            idmap = mp.solutions.pose.PoseLandmark if hasattr(mp, "solutions") else mp_solutions.pose.PoseLandmark

            l_ankle = self._safe_point(lm, idmap.LEFT_ANKLE.value)
            r_ankle = self._safe_point(lm, idmap.RIGHT_ANKLE.value)
            l_hip = self._safe_point(lm, idmap.LEFT_HIP.value)
            r_hip = self._safe_point(lm, idmap.RIGHT_HIP.value)
            l_shoulder = self._safe_point(lm, idmap.LEFT_SHOULDER.value)
            r_shoulder = self._safe_point(lm, idmap.RIGHT_SHOULDER.value)
            l_wrist = self._safe_point(lm, idmap.LEFT_WRIST.value)
            r_wrist = self._safe_point(lm, idmap.RIGHT_WRIST.value)
            l_knee = self._safe_point(lm, idmap.LEFT_KNEE.value)
            r_knee = self._safe_point(lm, idmap.RIGHT_KNEE.value)

            if l_ankle is not None:
                left_ankle_positions.append(l_ankle)
            if r_ankle is not None:
                right_ankle_positions.append(r_ankle)
            if l_wrist is not None:
                left_wrist_positions.append(l_wrist)
            if r_wrist is not None:
                right_wrist_positions.append(r_wrist)

            if l_hip is not None and r_hip is not None:
                hip_center_positions.append((l_hip + r_hip) / 2.0)

            shoulder_w = self._distance(l_shoulder, r_shoulder)
            stance_w = self._distance(l_ankle, r_ankle)
            if not np.isnan(shoulder_w):
                shoulder_widths.append(shoulder_w)
            if not np.isnan(stance_w):
                stance_widths.append(stance_w)

            left_knee_angle = self._angle(l_hip, l_knee, l_ankle)
            right_knee_angle = self._angle(r_hip, r_knee, r_ankle)
            if not np.isnan(left_knee_angle):
                knee_angles.append(left_knee_angle)
            if not np.isnan(right_knee_angle):
                knee_angles.append(right_knee_angle)

            processed += 1

        capture.release()

        if len(hip_center_positions) < 10:
            raise ValueError(
                f"Insufficient pose data extracted from {video_path}. Try a clearer single-player clip."
            )

        features = self._aggregate_features(
            left_ankle_positions,
            right_ankle_positions,
            hip_center_positions,
            left_wrist_positions,
            right_wrist_positions,
            knee_angles,
            stance_widths,
            shoulder_widths,
        )
        return features

    @staticmethod
    def _velocity(positions: List[np.ndarray]) -> np.ndarray:
        if len(positions) < 3:
            return np.array([], dtype=np.float32)
        pts = np.stack(positions)
        return np.linalg.norm(np.diff(pts, axis=0), axis=1)

    @staticmethod
    def _stat_block(prefix: str, values: np.ndarray) -> Dict[str, float]:
        if values.size == 0:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_std": 0.0,
                f"{prefix}_max": 0.0,
                f"{prefix}_p90": 0.0,
            }
        return {
            f"{prefix}_mean": float(np.mean(values)),
            f"{prefix}_std": float(np.std(values)),
            f"{prefix}_max": float(np.max(values)),
            f"{prefix}_p90": float(np.percentile(values, 90)),
        }

    def _aggregate_features(
        self,
        left_ankle_positions: List[np.ndarray],
        right_ankle_positions: List[np.ndarray],
        hip_center_positions: List[np.ndarray],
        left_wrist_positions: List[np.ndarray],
        right_wrist_positions: List[np.ndarray],
        knee_angles: List[float],
        stance_widths: List[float],
        shoulder_widths: List[float],
    ) -> Dict[str, float]:
        left_ankle_speed = self._velocity(left_ankle_positions)
        right_ankle_speed = self._velocity(right_ankle_positions)
        hip_speed = self._velocity(hip_center_positions)
        left_wrist_speed = self._velocity(left_wrist_positions)
        right_wrist_speed = self._velocity(right_wrist_positions)

        stance_width_arr = np.array(stance_widths, dtype=np.float32)
        shoulder_width_arr = np.array(shoulder_widths, dtype=np.float32)
        knee_angles_arr = np.array(knee_angles, dtype=np.float32)

        stance_ratio = np.array([], dtype=np.float32)
        if stance_width_arr.size and shoulder_width_arr.size:
            base = float(np.mean(shoulder_width_arr))
            if base > 1e-6:
                stance_ratio = stance_width_arr / base

        features: Dict[str, float] = {}
        features.update(self._stat_block("left_ankle_speed", left_ankle_speed))
        features.update(self._stat_block("right_ankle_speed", right_ankle_speed))
        features.update(self._stat_block("hip_speed", hip_speed))
        features.update(self._stat_block("left_wrist_speed", left_wrist_speed))
        features.update(self._stat_block("right_wrist_speed", right_wrist_speed))
        features.update(self._stat_block("knee_angle", knee_angles_arr))
        features.update(self._stat_block("stance_ratio", stance_ratio))

        if len(hip_center_positions) >= 5:
            hips = np.stack(hip_center_positions)
            features["court_coverage_x"] = float(np.max(hips[:, 0]) - np.min(hips[:, 0]))
            features["court_coverage_y"] = float(np.max(hips[:, 1]) - np.min(hips[:, 1]))
        else:
            features["court_coverage_x"] = 0.0
            features["court_coverage_y"] = 0.0

        lunge_rate = 0.0
        if knee_angles_arr.size:
            lunge_rate = float(np.mean(knee_angles_arr < 135.0))
        features["lunge_rate"] = lunge_rate

        if hip_speed.size >= 3:
            acceleration = np.diff(hip_speed)
            features.update(self._stat_block("hip_acceleration", acceleration))
            direction_change = np.diff(np.sign(acceleration))
            features["direction_change_rate"] = float(np.mean(np.abs(direction_change) > 0))
        else:
            features.update(self._stat_block("hip_acceleration", np.array([], dtype=np.float32)))
            features["direction_change_rate"] = 0.0

        return features
