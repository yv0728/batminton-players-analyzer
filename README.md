# Badminton Skill Analyzer

This project analyzes badminton player movement from videos and predicts a player skill level:
- beginner
- intermediate
- pro

It also generates movement insights and improvement recommendations.

## What This Project Does

- Extracts movement features from player videos using MediaPipe pose landmarks.
- Trains a classifier to predict skill level from labeled videos.
- Analyzes a new video and outputs:
  - predicted skill level
  - confidence score
  - efficiency score (0-100)
  - action efficiency breakdown (footwork, agility, coverage, stability)
  - class probabilities
  - movement feature summary
  - personalized practice recommendations

## High-Accuracy Guidance

Model accuracy depends mostly on your dataset quality. To maximize accuracy:

- Use at least 80-100 videos per class (beginner/intermediate/pro).
- Keep camera angle consistent (side or back baseline view).
- Ensure one clear player per clip.
- Include different rallies, speeds, and fatigue states.
- Balance classes evenly.
- Avoid mixing very low frame-rate clips with high frame-rate clips.

## Project Structure

- `src/badminton_analyzer/pose_features.py`: movement feature extraction
- `src/badminton_analyzer/training.py`: training and validation
- `src/badminton_analyzer/inference.py`: video inference and report generation
- `src/badminton_analyzer/recommendation.py`: personalized improvement suggestions
- `src/badminton_analyzer/cli.py`: command-line entry point
- `data/sample_labels.csv`: example label file format

## Setup

```powershell
cd D:\batmittor analyzer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Prepare Labels File

Create CSV with columns:
- `video_path`
- `skill_level`

Example is included at `data/sample_labels.csv`.

## Train

```powershell
cd D:\batmittor analyzer
$env:PYTHONPATH = "src"
python -m badminton_analyzer.cli train --labels_csv data/sample_labels.csv
```

Outputs:
- model: `artifacts/skill_model.joblib`
- metrics: `artifacts/model_metrics.json`
- extracted features: `data/extracted_features.csv`

## Analyze New Player Video

```powershell
cd D:\batmittor analyzer
$env:PYTHONPATH = "src"
python -m badminton_analyzer.cli analyze --video path\\to\\new_player.mp4
```

Output report:
- `reports/latest_report.json`

## Run Local Server UI

```powershell
cd D:\batmittor analyzer
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

Then open:
- `http://127.0.0.1:8502`

UI workflow:
- In **Train Model** tab: upload clips by class label (beginner/intermediate/pro), add them to dataset, then train from UI.
- In **Analyze Video** tab: upload one player video, watch it loop in the UI, then run analysis.
- The app writes model files into `artifacts/` and reports into `reports/`.

## Notes

- This model is intended for coaching assistance, not medical diagnosis.
- For better shot-level analysis (smash/drop/clear quality), extend with shuttle tracking and temporal models.
