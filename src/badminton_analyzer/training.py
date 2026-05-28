from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .pose_features import PoseFeatureExtractor


try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


class SkillTrainer:
    def __init__(self):
        self.extractor = PoseFeatureExtractor()

    @staticmethod
    def _normalize_skill_label(raw_label: str) -> str:
        value = raw_label.strip().lower()
        aliases = {
            "bignner": "beginner",
            "begginer": "beginner",
            "novice": "beginner",
            "intermusit": "intermediate",
            "intermediate": "intermediate",
            "advanced": "pro",
            "professional": "pro",
            "elite": "pro",
            "pro": "pro",
        }
        return aliases.get(value, value)

    def build_feature_table(
        self,
        labels_csv_path: str | Path,
        output_feature_csv: str | Path | None = None,
    ) -> pd.DataFrame:
        labels_df = pd.read_csv(labels_csv_path)
        required_cols = {"video_path", "skill_level"}
        if not required_cols.issubset(labels_df.columns):
            raise ValueError("labels CSV must contain columns: video_path, skill_level")

        rows: List[Dict[str, float]] = []
        failures: List[Tuple[str, str]] = []

        for _, row in labels_df.iterrows():
            video_path = row["video_path"]
            skill_level = self._normalize_skill_label(str(row["skill_level"]))
            try:
                features = self.extractor.extract_video_features(video_path)
                features["video_path"] = video_path
                features["skill_level"] = skill_level
                rows.append(features)
                print(f"Processed: {video_path}")
            except Exception as ex:
                failures.append((video_path, str(ex)))
                print(f"Failed: {video_path} -> {ex}")

        if not rows:
            raise RuntimeError("No videos were processed successfully. Check input data quality.")

        feature_df = pd.DataFrame(rows)
        if output_feature_csv:
            Path(output_feature_csv).parent.mkdir(parents=True, exist_ok=True)
            feature_df.to_csv(output_feature_csv, index=False)

        if failures:
            print("\\nSome videos failed processing:")
            for video_path, reason in failures:
                print(f"- {video_path}: {reason}")

        return feature_df

    def train_model(
        self,
        feature_df: pd.DataFrame,
        model_output_path: str | Path,
        metadata_output_path: str | Path,
    ) -> Dict[str, float]:
        label_col = "skill_level"
        ignore_cols = ["video_path", label_col]

        X = feature_df.drop(columns=[c for c in ignore_cols if c in feature_df.columns])
        y = feature_df[label_col].astype(str)

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        numeric_cols = list(X.columns)
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
            ],
            remainder="drop",
        )

        if HAS_XGBOOST:
            model = XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
            )
        else:
            model = RandomForestClassifier(
                n_estimators=500,
                max_depth=16,
                min_samples_leaf=2,
                random_state=42,
                class_weight="balanced",
            )

        pipeline = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", model),
            ]
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring="accuracy")

        pipeline.fit(X, y_encoded)
        y_pred = pipeline.predict(X)
        report = classification_report(y_encoded, y_pred, target_names=le.classes_, output_dict=True)

        Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": pipeline, "label_encoder": le, "feature_columns": numeric_cols}, model_output_path)

        metadata = {
            "mean_cv_accuracy": float(np.mean(cv_scores)),
            "std_cv_accuracy": float(np.std(cv_scores)),
            "train_accuracy": float(np.mean(y_pred == y_encoded)),
            "classification_report": report,
            "uses_xgboost": HAS_XGBOOST,
            "num_samples": int(len(feature_df)),
        }

        Path(metadata_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_output_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "mean_cv_accuracy": metadata["mean_cv_accuracy"],
            "std_cv_accuracy": metadata["std_cv_accuracy"],
            "train_accuracy": metadata["train_accuracy"],
        }
