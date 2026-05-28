from __future__ import annotations

import base64
import json
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import quote_plus

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# Ensure local package imports work when running from project root.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from badminton_analyzer.inference import SkillInference
from badminton_analyzer.training import SkillTrainer


st.set_page_config(page_title="Badminton Analyzer", layout="wide")

st.markdown(
        """
        <style>
        :root {
            --bg-soft: #f5f7fb;
            --card: #ffffff;
            --ink: #101828;
            --accent: #0f766e;
            --accent-2: #eab308;
            --btn-ink: #ffffff;
        }
        @keyframes heroIn {
            0% { transform: translateY(18px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
        }
        @keyframes tabIn {
            0% { transform: translateY(8px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
        }
        @keyframes glowPulse {
            0%, 100% { box-shadow: 0 8px 18px rgba(15, 118, 110, 0.22); }
            50% { box-shadow: 0 10px 26px rgba(15, 118, 110, 0.32); }
        }
        .stApp {
            background:
                radial-gradient(circle at 90% 10%, rgba(15, 118, 110, 0.06), transparent 30%),
                radial-gradient(circle at 10% 0%, rgba(234, 179, 8, 0.06), transparent 24%),
                var(--bg-soft);
        }
        .stMarkdown, .stText, .stCaption, .stSubheader, .stHeader, .stMetric,
        div[data-testid="stFileUploader"], div[data-testid="stSelectbox"],
        div[data-testid="stCheckbox"], div[data-testid="stExpander"],
        label, p, h1, h2, h3 {
            color: var(--ink) !important;
        }
        div[data-testid="stTabs"] button {
            color: #334155 !important;
            border-radius: 10px 10px 0 0;
            transition: all 180ms ease;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #0f766e !important;
            border-bottom-color: #0f766e !important;
            font-weight: 700;
            background: rgba(15, 118, 110, 0.08) !important;
        }
        div[data-testid="stTabContent"] {
            animation: tabIn 260ms ease;
        }
        .hero {
            padding: 16px 18px;
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(15,118,110,0.97), rgba(22,101,52,0.93));
            color: #ffffff !important;
            border: 1px solid rgba(255,255,255,0.18);
            box-shadow: 0 10px 22px rgba(15, 118, 110, 0.18);
            margin-bottom: 12px;
            animation: heroIn 420ms ease;
        }
        .hero h1 {
            margin: 0;
            font-size: 2rem;
            letter-spacing: 0.2px;
        }
        .hero p {
            margin: 6px 0 0 0;
            opacity: 0.95;
        }
        .badge-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin: 10px 0 4px 0;
        }
        .badge {
            border: 1px solid rgba(255,255,255,0.26);
            background: rgba(255,255,255,0.11);
            color: #fff;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.82rem;
        }
        .note-card {
            border: 1px solid #d0d5dd;
            background: var(--card);
            border-radius: 14px;
            padding: 12px 14px;
            margin-top: 10px;
            transition: transform 180ms ease, box-shadow 180ms ease;
        }
        .note-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(16, 24, 40, 0.09);
        }
        .stButton > button,
        .stDownloadButton > button,
        button[data-testid="baseButton-primary"],
        button[data-testid="baseButton-secondary"] {
            background: linear-gradient(135deg, #0f766e, #0a5f59) !important;
            color: var(--btn-ink) !important;
            border: 1px solid #0a5f59 !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            letter-spacing: 0.2px;
            transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
            animation: glowPulse 2.4s ease-in-out infinite;
        }
        div[data-testid="stFileUploaderDropzone"] {
            border: 1px dashed #0f766e !important;
            background: #ffffff !important;
        }
        div[data-testid="stFileUploaderDropzone"] * {
            color: #0f172a !important;
            opacity: 1 !important;
        }
        div[data-testid="stFileUploaderDropzone"] button {
            background: linear-gradient(135deg, #0f766e, #0a5f59) !important;
            color: #ffffff !important;
            border: 1px solid #0a5f59 !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        button[data-testid="baseButton-primary"]:hover,
        button[data-testid="baseButton-secondary"]:hover {
            transform: translateY(-1px);
            filter: brightness(1.05);
        }
        .stButton > button:focus,
        .stDownloadButton > button:focus,
        button[data-testid="baseButton-primary"]:focus,
        button[data-testid="baseButton-secondary"]:focus {
            outline: 3px solid rgba(234, 179, 8, 0.45) !important;
            outline-offset: 1px;
        }
        [data-testid="stDeployButton"] { display: none !important; }
        header[data-testid="stHeader"] .stDeployButton { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
)

st.markdown(
        """
        <div class="hero">
            <h1>Batminton Players Analyzer</h1>
            <p>Track movement efficiency, classify skill level, and generate report-ready coaching insights from uploaded match clips.</p>
            <div class="badge-row">
                <span class="badge">Pose Tracking</span>
                <span class="badge">Efficiency Score</span>
                <span class="badge">Skill Classifier</span>
                <span class="badge">Action Overlay</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
)
st.caption("Built for batminton player performance tracking and development analysis.")

default_model = ROOT / "artifacts" / "skill_model.joblib"
default_report = ROOT / "reports" / "latest_report.json"
uploads_train_dir = ROOT / "data" / "uploads" / "train"
uploads_infer_dir = ROOT / "data" / "uploads" / "analyze"
tracked_preview_dir = ROOT / "reports" / "tracked_preview"

PUBLIC_DATASET_SOURCES = [
    {
        "name": "Kaggle: Badminton Dataset",
        "url": "https://www.kaggle.com/datasets/ardyhealthy/badminton-dataset",
        "note": "General badminton video and image source.",
    },
    {
        "name": "Kaggle: Badminton Shot Dataset",
        "url": "https://www.kaggle.com/datasets/phucthaiv02/badminton-shot-dataset",
        "note": "Adds action diversity for shot and movement analysis.",
    },
    {
        "name": "Kaggle: AI Badminton Dataset",
        "url": "https://www.kaggle.com/datasets/sayeedulhaq/ai-badminton-dataset",
        "note": "Helpful for improving robustness across match conditions.",
    },
]


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)
    return cleaned.strip("._") or "video.mp4"


def _save_uploaded_file(uploaded_file, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = target_dir / f"{timestamp}_{_safe_name(uploaded_file.name)}"
    out_path.write_bytes(uploaded_file.getbuffer())
    return out_path


def _render_video_preview(video_file: Path, title: str) -> None:
    video_bytes = video_file.read_bytes()
    st.caption(title)
    st.video(video_bytes, autoplay=True, muted=True, loop=True)


def _render_gif_preview(gif_file: Path, title: str) -> None:
    st.caption(title)
    st.image(gif_file.read_bytes())


def _chart_png_from_mapping(title: str, values: dict[str, float]) -> bytes:
    labels = list(values.keys())
    data = [float(values[k]) for k in labels]

    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=160)
    bars = ax.bar(labels, data, color=["#0f766e", "#14b8a6", "#22c55e", "#eab308", "#f97316", "#ef4444"][: len(labels)])
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(data + [1.0]) * 1.2)
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    for bar, value in zip(bars, data):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def _build_pdf_report(report: dict, video_stem: str) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    y = height - 40
    c.setFont("Helvetica-Bold", 18)
    c.drawString(36, y, "Batminton Players Analyzer - Performance Report")
    y -= 24

    c.setFont("Helvetica", 10)
    c.drawString(36, y, f"Video: {video_stem}")
    y -= 14
    c.drawString(36, y, f"Predicted Skill: {str(report.get('predicted_skill_level', '')).upper()}")
    y -= 14
    c.drawString(36, y, f"Confidence: {float(report.get('confidence', 0.0)) * 100:.2f}%")
    y -= 14
    c.drawString(36, y, f"Efficiency Score: {float(report.get('efficiency_score', 0.0)):.2f}/100")

    y -= 24
    c.setFont("Helvetica-Bold", 11)
    c.drawString(36, y, "Action Efficiency Breakdown")

    action_map = {
        str(k): float(v) for k, v in report.get("action_efficiency_breakdown", {}).items()
    }
    action_png = _chart_png_from_mapping("Action Efficiency", action_map) if action_map else None
    y -= 210
    if action_png:
        c.drawImage(ImageReader(BytesIO(action_png)), 36, y, width=520, height=180, preserveAspectRatio=True, mask="auto")

    y -= 24
    c.setFont("Helvetica-Bold", 11)
    c.drawString(36, y, "Class Probabilities")
    prob_map = {
        str(k): float(v) * 100.0 for k, v in report.get("class_probabilities", {}).items()
    }
    probs_png = _chart_png_from_mapping("Class Probabilities (%)", prob_map) if prob_map else None
    y -= 210
    if probs_png:
        c.drawImage(ImageReader(BytesIO(probs_png)), 36, y, width=520, height=180, preserveAspectRatio=True, mask="auto")

    y -= 12
    c.setFont("Helvetica-Bold", 11)
    c.drawString(36, y, "Recommendations")
    c.setFont("Helvetica", 9)
    y -= 14
    for rec in report.get("recommendations", []):
        if y < 40:
            c.showPage()
            y = height - 40
            c.setFont("Helvetica", 9)
        c.drawString(42, y, f"- {rec[:110]}")
        y -= 12

    c.save()
    buf.seek(0)
    return buf.read()


# Preload the MediaPipe drawing utilities once at module level
_MP_POSE = mp.solutions.pose
_MP_DRAW = mp.solutions.drawing_utils
_MP_DRAW_STYLES = mp.solutions.drawing_styles

_LANDMARK_STYLE = _MP_DRAW.DrawingSpec(color=(0, 255, 180), thickness=2, circle_radius=3)
_CONNECTION_STYLE = _MP_DRAW.DrawingSpec(color=(0, 210, 255), thickness=2)


def _lm_px(lm, w: int, h: int) -> tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


def _build_tracking_preview(
    video_file: Path,
    frame_stride: int = 2,           # every other frame — smooth but 2× faster
    max_preview_frames: int = 220,   # lower cap for faster completion
    timeout_seconds: float = 45.0,   # hard safety cap
    progress_callback=None,
) -> Path:
    tracked_preview_dir.mkdir(parents=True, exist_ok=True)
    out_file_mp4 = tracked_preview_dir / f"tracked_{video_file.stem}.mp4"
    out_file_gif = tracked_preview_dir / f"tracked_{video_file.stem}.gif"
    if out_file_mp4.exists():
        return out_file_mp4
    if out_file_gif.exists():
        return out_file_gif

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise ValueError(f"Cannot open uploaded video: {video_file.name}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps    = float(cap.get(cv2.CAP_PROP_FPS)        or 24.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_time = time.perf_counter()

    # model_complexity=1 → full BlazePose model, ~3× faster than complexity=2
    pose = _MP_POSE.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Downsample input to 640px wide before pose inference — landmark coords
    # are normalized (0-1) so accuracy is preserved; just much faster.
    proc_scale = min(1.0, 576.0 / max(1.0, width))
    proc_w = max(1, int(width  * proc_scale))
    proc_h = max(1, int(height * proc_scale))

    hip_trail: list[tuple[int, int]] = []
    raw_count = 0
    sampled_count = 0
    out_w = proc_w
    out_h = proc_h
    out_fps = max(8, int(min(fps, 15)))
    writer = None
    use_gif_fallback = False
    gif_frames: list[np.ndarray] = []

    # Prefer browser-friendly MP4 codecs. If unavailable, fall back to GIF.
    for codec in ("avc1", "H264", "X264", "mp4v"):
        candidate = cv2.VideoWriter(
            str(out_file_mp4),
            cv2.VideoWriter_fourcc(*codec),
            out_fps,
            (out_w, out_h),
        )
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()

    if writer is None:
        use_gif_fallback = True

    try:
        while cap.isOpened() and sampled_count < max_preview_frames:
            if timeout_seconds > 0 and (time.perf_counter() - start_time) > timeout_seconds:
                break

            ok, frame = cap.read()
            if not ok:
                break
            raw_count += 1

            if frame_stride > 1 and (raw_count % frame_stride != 0):
                continue

            # Downsample once — reuse proc_frame for both inference and drawing
            proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
            rgb    = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark

                # ── draw full skeleton on the downsampled frame ──
                _MP_DRAW.draw_landmarks(
                    proc_frame,
                    result.pose_landmarks,
                    _MP_POSE.POSE_CONNECTIONS,
                    landmark_drawing_spec=_LANDMARK_STYLE,
                    connection_drawing_spec=_CONNECTION_STYLE,
                )

                # ── tight bounding box around all visible joints ──
                visible = [
                    _lm_px(p, proc_w, proc_h)
                    for p in lm if p.visibility > 0.4
                ]
                if visible:
                    pts = np.array(visible, dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(pts)
                    pad = 8
                    x1, y1 = max(0, x - pad), max(0, y - pad)
                    x2, y2 = min(proc_w, x + w + pad), min(proc_h, y + h + pad)
                    cv2.rectangle(proc_frame, (x1, y1), (x2, y2), (0, 210, 255), 2)
                    cv2.putText(
                        proc_frame, "Player",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 210, 255), 2, cv2.LINE_AA,
                    )

                # ── hip-centre movement trail ──
                lh = lm[_MP_POSE.PoseLandmark.LEFT_HIP.value]
                rh = lm[_MP_POSE.PoseLandmark.RIGHT_HIP.value]
                if lh.visibility > 0.4 and rh.visibility > 0.4:
                    lhx, lhy = _lm_px(lh, proc_w, proc_h)
                    rhx, rhy = _lm_px(rh, proc_w, proc_h)
                    hip_trail.append(((lhx + rhx) // 2, (lhy + rhy) // 2))
                    if len(hip_trail) > 80:
                        hip_trail.pop(0)

            if len(hip_trail) > 1:
                cv2.polylines(
                    proc_frame,
                    [np.array(hip_trail, dtype=np.int32)],
                    isClosed=False,
                    color=(30, 255, 30),
                    thickness=2,
                )
                cv2.circle(proc_frame, hip_trail[-1], 6, (30, 255, 30), -1)

            if use_gif_fallback:
                gif_frames.append(cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB))
            else:
                writer.write(proc_frame)
            sampled_count += 1

            if progress_callback is not None:
                if total_frames > 0:
                    # Cap at 0.98 so 100% shows only when writer finalization is done
                    progress_callback(min(raw_count / total_frames * 0.98, 0.98))
                else:
                    progress_callback(min(sampled_count / 300 * 0.98, 0.98))
    finally:
        cap.release()
        if writer is not None and not use_gif_fallback:
            writer.release()
        pose.close()

    if sampled_count == 0:
        raise RuntimeError("Failed to generate tracked preview frames")

    out_file = out_file_mp4
    if use_gif_fallback:
        imageio.mimsave(out_file_gif, gif_frames, fps=max(6, min(out_fps, 10)), loop=0)
        out_file = out_file_gif

    if not out_file.exists():
        raise RuntimeError("Failed to generate tracked preview video")

    return out_file


if "show_train_panel" not in st.session_state:
    st.session_state.show_train_panel = False

train_toggle_label = "Hide Train Model" if st.session_state.show_train_panel else "Train Model"
if st.button(train_toggle_label, type="secondary"):
    st.session_state.show_train_panel = not st.session_state.show_train_panel
    st.rerun()

if st.session_state.show_train_panel:
    st.subheader("Train With Uploaded Videos")
    st.write("Upload labeled clips and train a model directly from the UI.")

    st.markdown(
        """
        <div class="note-card">
        <strong>Model note:</strong> This model is purpose-built for batminton player analysis in this app.
        It learns from your uploaded player clips and predicts beginner, intermediate, or pro with movement efficiency scoring.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Internet datasets to improve accuracy"):
        st.write("Use these public badminton datasets to increase sample count and balance.")
        for source in PUBLIC_DATASET_SOURCES:
            st.markdown(f"- [{source['name']}]({source['url']}) - {source['note']}")
        search_query = quote_plus("badminton player dataset video")
        st.markdown(
            f"- [Search more datasets](https://www.google.com/search?q={search_query})"
        )

    label_choice = st.selectbox("Training label", ["beginner", "intermediate", "pro"], index=0)
    train_uploads = st.file_uploader(
        "Upload training videos (mp4/mov/avi)",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        accept_multiple_files=True,
        key="train_upload",
    )

    if "train_manifest" not in st.session_state:
        st.session_state.train_manifest = []

    add_to_dataset = st.button("Add Uploaded Clips To Training Set")
    if add_to_dataset and train_uploads:
        added = 0
        for uploaded in train_uploads:
            saved_path = _save_uploaded_file(uploaded, uploads_train_dir / label_choice)
            st.session_state.train_manifest.append(
                {"video_path": str(saved_path), "skill_level": label_choice}
            )
            added += 1
        st.success(f"Added {added} clips under label '{label_choice}'.")

    if st.session_state.train_manifest:
        st.write(f"Training clips in session: {len(st.session_state.train_manifest)}")
        preview_rows = [
            {"clip": Path(item["video_path"]).name, "skill_level": item["skill_level"]}
            for item in st.session_state.train_manifest
        ]
        st.dataframe(preview_rows, width="stretch")

    train_btn = st.button("Train Model From Uploaded Dataset", type="primary")
    if train_btn:
        if len(st.session_state.train_manifest) < 6:
            st.error("Please add more clips. Minimum recommended: at least 2 per class.")
        else:
            labels_csv_path = ROOT / "data" / "generated_labels.csv"
            labels_csv_path.parent.mkdir(parents=True, exist_ok=True)

            import pandas as pd

            labels_df = pd.DataFrame(st.session_state.train_manifest)
            labels_df.to_csv(labels_csv_path, index=False)

            with st.spinner("Training model from uploaded clips. This may take a while..."):
                trainer = SkillTrainer()
                feature_df = trainer.build_feature_table(
                    labels_csv_path,
                    ROOT / "data" / "extracted_features.csv",
                )
                metrics = trainer.train_model(
                    feature_df,
                    ROOT / "artifacts" / "skill_model.joblib",
                    ROOT / "artifacts" / "model_metrics.json",
                )

            st.success("Training complete.")
            c1, c2 = st.columns(2)
            c1.metric("Mean CV Accuracy", f"{metrics['mean_cv_accuracy'] * 100:.2f}%")
            c2.metric("Train Accuracy", f"{metrics['train_accuracy'] * 100:.2f}%")
            st.info("Model updated successfully.")

if not st.session_state.show_train_panel:
    st.subheader("Analyze One Uploaded Video")
    st.write("Upload a player clip, preview it, and run skill plus efficiency analysis.")

    auto_save = st.checkbox("Auto-save JSON report", value=True)

    analysis_upload = st.file_uploader(
        "Upload player video for analysis",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        accept_multiple_files=False,
        key="analysis_upload",
    )

    uploaded_analysis_path: Path | None = None
    tracked_preview_path: Path | None = None

    if "analysis_file_name" not in st.session_state:
        st.session_state.analysis_file_name = None
    if "analysis_uploaded_path" not in st.session_state:
        st.session_state.analysis_uploaded_path = None
    if "analysis_tracked_preview_path" not in st.session_state:
        st.session_state.analysis_tracked_preview_path = None

    if analysis_upload is not None:
        if st.session_state.analysis_file_name != analysis_upload.name:
            uploaded_analysis_path = _save_uploaded_file(analysis_upload, uploads_infer_dir)
            st.session_state.analysis_file_name = analysis_upload.name
            st.session_state.analysis_uploaded_path = str(uploaded_analysis_path)
            st.session_state.analysis_tracked_preview_path = None

            progress = st.progress(0.0, text="Generating tracking overlay...")
            try:
                tracked_preview_path = _build_tracking_preview(
                    uploaded_analysis_path,
                    progress_callback=lambda ratio: progress.progress(
                        float(ratio),
                        text=f"MediaPipe BlazePose tracking... {int(ratio / 0.98 * 100)}%"
                        if ratio < 0.98
                        else "Finalizing preview, almost done...",
                    ),
                )
                st.session_state.analysis_tracked_preview_path = str(tracked_preview_path)
                progress.progress(1.0, text="Tracking preview ready")
                progress.empty()
            except Exception as preview_ex:
                st.session_state.analysis_tracked_preview_path = None
                progress.empty()
                st.warning(f"Tracked overlay preview failed: {preview_ex}")
        else:
            if st.session_state.analysis_uploaded_path:
                uploaded_analysis_path = Path(st.session_state.analysis_uploaded_path)
            if st.session_state.analysis_tracked_preview_path:
                tracked_preview_path = Path(st.session_state.analysis_tracked_preview_path)

        st.success(f"Uploaded: {uploaded_analysis_path.name}")

        if tracked_preview_path and tracked_preview_path.exists():
            st.caption("Tracked Overlay (loop)")
            if tracked_preview_path.suffix.lower() == ".gif":
                gif_b64 = base64.b64encode(tracked_preview_path.read_bytes()).decode()
                st.markdown(
                    f'<img src="data:image/gif;base64,{gif_b64}" style="width:420px;border-radius:10px;display:block;" />',
                    unsafe_allow_html=True,
                )
            else:
                st.video(str(tracked_preview_path), autoplay=True, muted=True, loop=True)
        else:
            st.info("Tracking preview is preparing automatically. Please wait a moment.")

    analyze_btn = st.button("Analyze Uploaded Video", type="primary")
    if analyze_btn:
        try:
            model_file = default_model
            report_path = default_report
            if uploaded_analysis_path is None and st.session_state.analysis_uploaded_path:
                uploaded_analysis_path = Path(st.session_state.analysis_uploaded_path)
            if not model_file.exists():
                st.error("Model file not found. Train in the Train Model tab first.")
            elif uploaded_analysis_path is None:
                st.error("Please upload a video before running analysis.")
            else:
                with st.spinner("Running video analysis..."):
                    infer = SkillInference(model_file)
                    report = infer.analyze_video(uploaded_analysis_path)

                    if auto_save:
                        infer.save_report(report, report_path)

                st.success("Analysis complete")

                k1, k2, k3 = st.columns(3)
                with k1:
                    st.metric("Predicted Skill", report["predicted_skill_level"].upper())
                with k2:
                    st.metric("Confidence", f"{report['confidence'] * 100:.2f}%")
                with k3:
                    st.metric("Efficiency Score", f"{report['efficiency_score']:.2f}/100")

                st.subheader("Action Efficiency Breakdown")
                st.bar_chart(report["action_efficiency_breakdown"])

                st.subheader("Class Probabilities")
                st.json(report["class_probabilities"])

                st.subheader("Recommendations")
                for rec in report["recommendations"]:
                    st.write(f"- {rec}")

                with st.expander("Movement Features"):
                    st.json(report["movement_features"])

                with st.expander("Full Report JSON"):
                    st.code(json.dumps(report, indent=2), language="json")

                report_pdf = _build_pdf_report(report, uploaded_analysis_path.stem)
                st.download_button(
                    label="Download Analysis Report (PDF)",
                    data=report_pdf,
                    file_name=f"analysis_report_{uploaded_analysis_path.stem}.pdf",
                    mime="application/pdf",
                    type="primary",
                )

                if auto_save:
                    st.info("Report saved successfully.")

        except Exception as ex:
            st.exception(ex)
else:
    st.info("Training panel is open. Hide Train Model to access Analyze section.")
