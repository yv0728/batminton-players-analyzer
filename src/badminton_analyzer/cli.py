from __future__ import annotations

import argparse
from pathlib import Path

from .inference import SkillInference
from .training import SkillTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Badminton player movement and skill-level analyzer"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train model from labeled videos")
    train_parser.add_argument(
        "--labels_csv",
        type=str,
        required=True,
        help="CSV with columns: video_path, skill_level",
    )
    train_parser.add_argument(
        "--feature_csv",
        type=str,
        default="data/extracted_features.csv",
        help="Where extracted video features are saved",
    )
    train_parser.add_argument(
        "--model_out",
        type=str,
        default="artifacts/skill_model.joblib",
        help="Output model path",
    )
    train_parser.add_argument(
        "--metadata_out",
        type=str,
        default="artifacts/model_metrics.json",
        help="Output metrics path",
    )

    infer_parser = subparsers.add_parser("analyze", help="Analyze one new player video")
    infer_parser.add_argument("--video", type=str, required=True, help="Path to player video")
    infer_parser.add_argument(
        "--model",
        type=str,
        default="artifacts/skill_model.joblib",
        help="Trained model path",
    )
    infer_parser.add_argument(
        "--report_out",
        type=str,
        default="reports/latest_report.json",
        help="Output JSON report path",
    )

    args = parser.parse_args()

    if args.command == "train":
        trainer = SkillTrainer()
        feature_df = trainer.build_feature_table(args.labels_csv, args.feature_csv)
        metrics = trainer.train_model(feature_df, args.model_out, args.metadata_out)
        print("Training complete")
        print(
            f"Mean CV Accuracy: {metrics['mean_cv_accuracy']:.4f} +/- {metrics['std_cv_accuracy']:.4f}"
        )
        print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Model saved to: {Path(args.model_out).resolve()}")
        print(f"Metrics saved to: {Path(args.metadata_out).resolve()}")

    elif args.command == "analyze":
        infer = SkillInference(args.model)
        report = infer.analyze_video(args.video)
        infer.save_report(report, args.report_out)

        print("Analysis complete")
        print(f"Predicted Skill: {report['predicted_skill_level']}")
        print(f"Confidence: {report['confidence']:.4f}")
        print(f"Report saved to: {Path(args.report_out).resolve()}")


if __name__ == "__main__":
    main()
