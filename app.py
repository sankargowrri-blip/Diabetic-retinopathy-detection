"""Accessible Streamlit dashboard for the educational IDRiD DR project."""

from __future__ import annotations

import json
import html
import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "dr_model.keras"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative Diabetic Retinopathy",
]
NAVIGATION = ("Home", "Prediction", "About Project", "Model Performance")

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light;
            --ink: #172b4d;
            --body: #334155;
            --muted: #526174;
            --teal: #0f766e;
            --teal-dark: #115e59;
            --line: #d8e2ec;
            --surface: #ffffff;
            --canvas: #f3f7fa;
        }

        .stApp {
            background: var(--canvas);
            color: var(--body);
        }
        .stApp, .stApp p, .stApp li, .stApp label,
        .stApp [data-testid="stMarkdownContainer"] {
            color: var(--body);
        }
        h1, h2, h3, h4, h5, h6,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
            color: var(--ink) !important;
            letter-spacing: -0.02em;
        }
        .block-container {
            max-width: 1220px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            background: #eaf1f6;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label {
            color: #24364b !important;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
        }
        [data-testid="stSidebar"] [data-testid="stButton"] button {
            min-height: 2.8rem;
            border-radius: 0.7rem;
            font-weight: 650;
            text-align: left;
        }
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] {
            color: #20334a;
            background: transparent;
            border: 1px solid transparent;
        }
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]:hover {
            color: #0b4f4a;
            background: #dce9ef;
            border-color: #bfd4df;
        }
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
            color: #ffffff;
            background: #0f766e;
            border: 1px solid #0f766e;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: #45576b !important;
        }
        .brand-block {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0.8rem 0.45rem 1.1rem;
            margin-bottom: 0.9rem;
            border-bottom: 1px solid #cad8e3;
        }
        .brand-icon {
            display: grid;
            place-items: center;
            width: 2.65rem;
            height: 2.65rem;
            flex: 0 0 2.65rem;
            color: #ffffff;
            background: #0f766e;
            border-radius: 0.8rem;
            font-size: 1.4rem;
        }
        .brand-name {
            color: #172b4d !important;
            font-weight: 750;
            font-size: 0.98rem;
            line-height: 1.25;
        }
        .brand-caption {
            margin-top: 0.2rem;
            color: #526174 !important;
            font-size: 0.76rem;
        }
        .sidebar-footer {
            margin-top: 1.4rem;
            padding: 0.85rem;
            border: 1px solid #c9d8e3;
            border-radius: 0.75rem;
            background: #f8fbfd;
            color: #334155 !important;
            font-size: 0.78rem;
            line-height: 1.45;
        }
        .hero {
            padding: clamp(1.35rem, 4vw, 2.6rem);
            margin: 0.2rem 0 1.25rem;
            border: 1px solid #c8dce2;
            border-radius: 1.15rem;
            background: linear-gradient(115deg, #ffffff 0%, #eef8f7 100%);
            box-shadow: 0 8px 26px rgba(23, 43, 77, 0.06);
        }
        .eyebrow {
            margin-bottom: 0.55rem;
            color: #0f766e !important;
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0.11em;
            text-transform: uppercase;
        }
        .hero-title {
            margin: 0;
            color: #172b4d !important;
            font-size: clamp(2rem, 4.6vw, 3.15rem);
            font-weight: 800;
            line-height: 1.08;
        }
        .hero-subtitle {
            margin: 0.8rem 0 0.65rem;
            color: #115e59 !important;
            font-size: clamp(1rem, 2.2vw, 1.25rem);
            font-weight: 650;
        }
        .hero-copy {
            max-width: 760px;
            margin: 0;
            color: #40536a !important;
            font-size: 1rem;
            line-height: 1.65;
        }
        .section-kicker {
            margin: 1.15rem 0 0.65rem;
            color: #0f766e !important;
            font-size: 0.75rem;
            font-weight: 750;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }
        .info-card, .content-card, .step-card {
            height: 100%;
            padding: 1rem 1.05rem;
            border: 1px solid #d4e0e9;
            border-radius: 0.9rem;
            background: var(--surface);
            box-shadow: 0 4px 14px rgba(23, 43, 77, 0.035);
        }
        .info-card { min-height: 118px; }
        .info-icon {
            margin-bottom: 0.45rem;
            color: #0f766e !important;
            font-size: 1.25rem;
        }
        .card-title {
            margin: 0 0 0.28rem;
            color: #172b4d !important;
            font-size: 1rem;
            font-weight: 730;
            line-height: 1.35;
        }
        .card-copy {
            margin: 0;
            color: #4b5d72 !important;
            font-size: 0.89rem;
            line-height: 1.5;
        }
        .step-number {
            display: inline-grid;
            place-items: center;
            width: 1.8rem;
            height: 1.8rem;
            margin-bottom: 0.65rem;
            border-radius: 50%;
            background: #d9f1ed;
            color: #115e59 !important;
            font-weight: 800;
        }
        .page-intro {
            margin: -0.25rem 0 1.2rem;
            color: #526174 !important;
            font-size: 1rem;
            line-height: 1.55;
        }
        .grade-result {
            padding: 1.15rem 1.25rem;
            margin: 0.35rem 0 0.9rem;
            border: 1px solid #94d5c9;
            border-left: 5px solid #0f766e;
            border-radius: 0.85rem;
            background: #effaf7;
        }
        .grade-result-label {
            margin-bottom: 0.3rem;
            color: #376259 !important;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .grade-result-value {
            color: #123b38 !important;
            font-size: 1.3rem;
            font-weight: 780;
            line-height: 1.35;
        }
        .probability-row {
            margin: 0.7rem 0 0.9rem;
        }
        .probability-label {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.35rem;
            color: #24364b !important;
            font-size: 0.88rem;
            line-height: 1.35;
        }
        .probability-track {
            width: 100%;
            height: 0.62rem;
            overflow: hidden;
            border-radius: 99px;
            background: #e2eaf0;
        }
        .probability-fill {
            height: 100%;
            border-radius: inherit;
            background: #14857b;
        }
        .disclaimer {
            padding: 0.95rem 1.05rem;
            margin: 1rem 0;
            border: 1px solid #e4c66c;
            border-left: 5px solid #ae7100;
            border-radius: 0.75rem;
            background: #fff8df;
            color: #493400 !important;
            line-height: 1.55;
        }
        .disclaimer strong, .disclaimer span {
            color: #493400 !important;
        }
        .stApp [data-testid="stAlert"] {
            border-radius: 0.75rem;
            color: #652b25 !important;
            background: #fff0ed !important;
            border: 1px solid #efc0b8 !important;
        }
        .stApp [data-testid="stAlert"] p,
        .stApp [data-testid="stAlert"] svg {
            color: #652b25 !important;
            fill: #652b25 !important;
        }
        .notice {
            padding: 0.85rem 1rem;
            margin: 0.65rem 0;
            border: 1px solid;
            border-radius: 0.75rem;
            line-height: 1.5;
        }
        .notice-info {
            color: #173a5e !important;
            background: #eaf4ff;
            border-color: #b8d5f2;
        }
        .notice-info strong, .notice-info span {
            color: #173a5e !important;
        }
        .notice-success {
            color: #174b35 !important;
            background: #eaf8ef;
            border-color: #b5dfc2;
        }
        .notice-success strong, .notice-success span {
            color: #174b35 !important;
        }
        .notice-error {
            color: #652b25 !important;
            background: #fff0ed;
            border-color: #efc0b8;
        }
        .notice-error strong, .notice-error span {
            color: #652b25 !important;
        }
        .stApp [data-testid="stMetric"] {
            padding: 0.9rem 1rem;
            border: 1px solid #d4e0e9;
            border-radius: 0.85rem;
            background: #ffffff;
            box-shadow: 0 4px 14px rgba(23, 43, 77, 0.035);
        }
        .stApp [data-testid="stMetricLabel"] {
            color: #526174 !important;
        }
        .stApp [data-testid="stMetricValue"] {
            color: #123b4a !important;
        }
        .stApp [data-testid="stFileUploader"] {
            padding: 0.8rem;
            border: 1px dashed #7da9b3;
            border-radius: 0.9rem;
            background: #ffffff;
        }
        .stApp [data-testid="stFileUploader"] section {
            background: #f8fbfd;
            border-color: #b9cbd9;
        }
        .stApp [data-testid="stFileUploader"] small,
        .stApp [data-testid="stFileUploader"] span {
            color: #40536a !important;
        }
        .stApp [data-testid="stImage"] img {
            border: 1px solid #d4e0e9;
            border-radius: 0.85rem;
        }
        .stApp [data-testid="stDataFrame"],
        .stApp [data-testid="stTable"] {
            border: 1px solid #d4e0e9;
            border-radius: 0.8rem;
            overflow: hidden;
            background: #ffffff;
        }
        .stApp button, .stApp input, .stApp textarea {
            font-size: 0.96rem;
        }
        .stApp button[kind="primary"] {
            font-weight: 700;
        }
        .muted-caption {
            color: #526174 !important;
            font-size: 0.84rem;
            line-height: 1.5;
        }
        @media (max-width: 700px) {
            .block-container {
                padding: 1rem 1rem 2rem;
            }
            .hero {
                padding: 1.25rem;
                border-radius: 0.9rem;
            }
            .info-card, .content-card, .step-card {
                min-height: auto;
                margin-bottom: 0.55rem;
            }
            .grade-result-value {
                font-size: 1.12rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def configured_model_url() -> str:
    url = os.environ.get("DR_MODEL_URL", "").strip()
    if not url:
        try:
            url = str(st.secrets.get("DR_MODEL_URL", "")).strip()
        except (FileNotFoundError, KeyError):
            url = ""
    return url


@st.cache_resource(show_spinner="Loading the trained model...")
def load_model():
    """Load the trained model locally, or from a configured direct URL."""
    model_path = MODEL_PATH
    if not model_path.is_file():
        model_url = configured_model_url()
        if not model_url:
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Train the model locally "
                "and copy it into models/, or configure DR_MODEL_URL for deployment."
            )
        model_path = Path(tempfile.gettempdir()) / "dr_model.keras"
        if not model_path.is_file():
            temporary_path = model_path.with_suffix(".download")
            try:
                urllib.request.urlretrieve(model_url, temporary_path)
                temporary_path.replace(model_path)
            except (OSError, ValueError) as error:
                temporary_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Could not download the model from DR_MODEL_URL: {error}"
                ) from error

    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def preprocess_image(image_bytes: bytes, image_size: int) -> np.ndarray:
    """Keep inference preprocessing aligned with train.py."""
    import tensorflow as tf

    image = tf.io.decode_image(
        image_bytes, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0
    return np.expand_dims(image.numpy(), axis=0)


def render_disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer">
            <strong>Educational project only — not for clinical use.</strong>
            <span> This model is not clinically validated. Do not use its output
            for diagnosis, treatment, or healthcare decisions.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_notice(message: str, kind: str = "info") -> None:
    if kind not in {"info", "success", "error"}:
        raise ValueError(f"Unsupported notice type: {kind}")
    safe_message = html.escape(message)
    st.markdown(
        f'<div class="notice notice-{kind}">{safe_message}</div>',
        unsafe_allow_html=True,
    )


def navigate_to(page: str) -> None:
    st.session_state["page"] = page
    st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-block">
                <div class="brand-icon" aria-hidden="true">👁</div>
                <div>
                    <div class="brand-name">Retina AI Project</div>
                    <div class="brand-caption">Diabetic retinopathy study</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-kicker">Main menu</div>', unsafe_allow_html=True)
        current_page = st.session_state.get("page", "Home")
        for page in NAVIGATION:
            if st.button(
                page,
                key=f"nav_{page.lower().replace(' ', '_')}",
                type="primary" if page == current_page else "secondary",
                use_container_width=True,
            ):
                st.session_state["page"] = page
                st.rerun()
        st.markdown(
            """
            <div class="sidebar-footer">
                <strong>Educational project only</strong><br>
                Not for clinical use or healthcare decisions.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_card(icon: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-icon">{icon}</div>
            <div class="card-title">{title}</div>
            <p class="card-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home() -> None:
    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">AI &amp; Data Science · College mini project</div>
            <h1 class="hero-title">Diabetic Retinopathy Detection</h1>
            <div class="hero-subtitle">AI-powered retinal image analysis using CNN and MobileNetV2</div>
            <p class="hero-copy">
                Explore how deep learning can classify retinal fundus images into
                five diabetic-retinopathy severity grades. This project combines a
                student-friendly training workflow with an interactive Streamlit app.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Get started  →", type="primary", key="home_get_started"):
        navigate_to("Prediction")

    st.markdown('<div class="section-kicker">Project at a glance</div>', unsafe_allow_html=True)
    cards = st.columns(4, gap="medium")
    with cards[0]:
        render_card("◈", "Transfer learning", "MobileNetV2 with an ImageNet-pretrained backbone.")
    with cards[1]:
        render_card("Ⅴ", "Five severity grades", "A probability output for each labelled grade.")
    with cards[2]:
        render_card("◎", "IDRiD dataset", "Retinal fundus photographs with expert grading labels.")
    with cards[3]:
        render_card("↗", "Streamlit ready", "An interactive interface designed for local and cloud demos.")

    st.markdown('<div class="section-kicker">How it works</div>', unsafe_allow_html=True)
    steps = st.columns(4, gap="medium")
    workflow = (
        ("Upload", "Choose a JPG, JPEG, or PNG fundus image."),
        ("Prepare", "The app converts, resizes, and scales the image."),
        ("Analyze", "MobileNetV2 produces five grade probabilities."),
        ("Review", "See the predicted grade and model confidence."),
    )
    for number, (column, (title, copy)) in enumerate(zip(steps, workflow), start=1):
        with column:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-number">{number}</div>
                    <div class="card-title">{title}</div>
                    <p class="card-copy">{copy}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    render_disclaimer()


def show_prediction(probabilities: np.ndarray) -> None:
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index]) * 100
    st.markdown(
        f"""
        <div class="grade-result">
            <div class="grade-result-label">Predicted severity grade · {predicted_index}</div>
            <div class="grade-result-value">{CLASS_NAMES[predicted_index]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.metric("Model confidence", f"{confidence:.1f}%")
    st.markdown("#### Probability by severity grade")
    for grade, (class_name, probability) in enumerate(
        zip(CLASS_NAMES, probabilities)
    ):
        value = float(np.clip(probability, 0.0, 1.0))
        st.markdown(
            f"""
            <div class="probability-row">
                <div class="probability-label">
                    <span>{grade} — {class_name}</span>
                    <strong>{value * 100:.1f}%</strong>
                </div>
                <div class="probability-track">
                    <div class="probability-fill" style="width:{value * 100:.2f}%"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        "Model confidence is the model's output probability. It is not the same as "
        "diagnostic certainty or clinical accuracy."
    )


def render_prediction() -> None:
    st.title("Retinal Image Prediction")
    st.markdown(
        '<p class="page-intro">Upload a retinal fundus image, preview it, then select '
        '<strong>Analyze image</strong> to see the model output.</p>',
        unsafe_allow_html=True,
    )
    render_disclaimer()

    if st.button("Clear / reset prediction", key="clear_upload"):
        st.session_state["upload_version"] = st.session_state.get("upload_version", 0) + 1
        st.session_state.pop("prediction_result", None)
        st.rerun()

    upload_key = f"fundus_upload_{st.session_state.get('upload_version', 0)}"
    upload_col, preview_col = st.columns([1, 1.05], gap="large")
    with upload_col:
        st.markdown("#### 1. Choose an image")
        st.write("Supported formats: JPG, JPEG, PNG.")
        uploaded_file = st.file_uploader(
            "Upload a retinal fundus image",
            type=["jpg", "jpeg", "png"],
            key=upload_key,
            help="Select a clear retinal fundus photograph in JPG, JPEG, or PNG format.",
        )
    image_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
    digest = str(hash(image_bytes)) if image_bytes is not None else None
    with preview_col:
        st.markdown("#### 2. Preview")
        if uploaded_file is None:
            render_notice("Your image preview will appear here after upload.")
        else:
            try:
                preview = Image.open(uploaded_file).convert("RGB")
                st.image(
                    preview,
                    caption="Uploaded fundus image",
                    use_column_width=True,
                )
            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
                EOFError,
                Image.DecompressionBombError,
            ):
                st.error("Could not open this image. Please choose a valid JPG, JPEG, or PNG.")
                return

    if uploaded_file is None:
        render_notice(
            "Upload an image to enable analysis. No prediction is shown before analysis."
        )
        return

    previous_result = st.session_state.get("prediction_result")
    if previous_result and previous_result["digest"] != digest:
        st.session_state.pop("prediction_result", None)

    if st.button("Analyze Image", type="primary", key="analyze_image"):
        try:
            import tensorflow as tf
        except ImportError as error:
            st.error(f"TensorFlow is not installed or could not be imported: {error}")
            return
        try:
            with st.spinner("Loading the model and analyzing your image..."):
                model = load_model()
                input_shape = model.input_shape
                if isinstance(input_shape, list):
                    input_shape = input_shape[0]
                if input_shape[1] is None or input_shape[2] is None:
                    raise ValueError("The model must have fixed image height and width.")
                model_input = preprocess_image(
                    image_bytes, image_size=int(input_shape[1])
                )
                probabilities = np.asarray(model.predict(model_input, verbose=0))[0]
                if probabilities.shape != (len(CLASS_NAMES),):
                    raise ValueError(
                        "The loaded model does not return the expected five grade probabilities."
                    )
                st.session_state["prediction_result"] = {
                    "digest": digest,
                    "probabilities": probabilities.tolist(),
                }
        except (
            FileNotFoundError,
            RuntimeError,
            ValueError,
            TypeError,
            OSError,
            tf.errors.OpError,
        ) as error:
            st.error(f"Prediction could not be completed: {error}")

    result = st.session_state.get("prediction_result")
    if result and result["digest"] == digest:
        st.markdown("### Analysis result")
        show_prediction(np.asarray(result["probabilities"], dtype=np.float32))


def render_about() -> None:
    st.title("About the Project")
    st.markdown(
        '<p class="page-intro">A student mini project that demonstrates image '
        'classification and transfer learning for retinal image research.</p>',
        unsafe_allow_html=True,
    )
    render_disclaimer()

    st.markdown("### Abstract")
    st.markdown(
        """
        Diabetic retinopathy is an eye condition associated with diabetes. This
        project explores a deep-learning workflow that classifies labelled retinal
        fundus images into five severity grades. An ImageNet-pretrained MobileNetV2
        model is adapted to the IDRiD grades and presented through a Streamlit
        interface. It is a learning demonstration, not a clinically validated tool.
        """
    )

    st.markdown("### Project overview")
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(
            """
            <div class="content-card">
                <div class="card-title">Problem statement</div>
                <p class="card-copy">Manual review of retinal images requires expertise.
                This project studies how a CNN can learn image patterns for a
                five-grade classification task.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Objectives")
        st.markdown(
            """
            - Prepare and validate labelled retinal images.
            - Train a transfer-learning classifier with validation controls.
            - Evaluate performance using a held-out split and class-aware metrics.
            - Demonstrate image upload and model output in Streamlit.
            """
        )
    with right:
        st.markdown(
            """
            <div class="content-card">
                <div class="card-title">Dataset</div>
                <p class="card-copy">IDRiD Disease Grading training images and
                retinopathy-grade labels. This workspace has 413 labelled training
                images; the official testing partition is kept separate.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Severity labels")
        for grade, name in enumerate(CLASS_NAMES):
            st.markdown(f"**{grade} — {name}**")

    st.markdown("### Model and learning approach")
    model_col, transfer_col = st.columns(2, gap="medium")
    with model_col:
        st.markdown(
            """
            <div class="content-card">
                <div class="card-title">CNN baseline</div>
                <p class="card-copy">The original CNN uses convolution, batch
                normalization, pooling, dropout, and a five-unit softmax output.
                It is retained so its result can be compared on the same test images.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with transfer_col:
        st.markdown(
            """
            <div class="content-card">
                <div class="card-title">MobileNetV2 transfer learning</div>
                <p class="card-copy">A lightweight ImageNet-pretrained feature
                extractor is first frozen while a grade classifier is learned.
                The final backbone layers are then fine-tuned at a lower learning rate.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Project workflow")
    workflow = st.columns(4, gap="medium")
    workflow_items = (
        ("1 · Validate", "Match image IDs and check diagnosis labels."),
        ("2 · Prepare", "Resize, RGB-convert, normalize, and augment training data."),
        ("3 · Train", "Use class weights and validation-based callbacks."),
        ("4 · Evaluate", "Compare models on fixed held-out test images."),
    )
    for column, (title, copy) in zip(workflow, workflow_items):
        with column:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="card-title">{title}</div>
                    <p class="card-copy">{copy}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Technologies")
    tech_columns = st.columns(4)
    for column, title, copy in zip(
        tech_columns,
        ("Python & TensorFlow", "MobileNetV2", "Scikit-learn", "Streamlit"),
        (
            "Training pipeline and Keras models.",
            "ImageNet-pretrained transfer learning.",
            "Stratified splits and evaluation metrics.",
            "Interactive local and cloud interface.",
        ),
    ):
        with column:
            render_card("◆", title, copy)

    st.markdown("### Applications and limitations")
    st.markdown(
        """
        **Learning applications:** practice image classification, transfer
        learning, class-imbalance handling, model evaluation, and web-app
        deployment.

        **Limitations:** IDRiD's training set is small and class-imbalanced.
        Performance on one held-out split is uncertain, and model confidence is
        not clinical certainty. The model is not validated for other datasets,
        cameras, patient populations, diagnosis, or treatment.
        """
    )


def show_performance_image(title: str, filename: str) -> None:
    st.markdown(f"#### {title}")
    image_path = OUTPUT_DIR / filename
    if image_path.is_file():
        st.image(str(image_path), use_column_width=True)
    else:
        render_notice(
            f"{filename} is not available yet. Run python train.py to generate "
            "the evaluation outputs."
        )


def render_performance() -> None:
    st.title("Model Performance")
    st.markdown(
        '<p class="page-intro">Evaluation results saved by the training script. '
        'These metrics describe a held-out split, not clinical performance.</p>',
        unsafe_allow_html=True,
    )
    render_disclaimer()
    metrics_path = OUTPUT_DIR / "metrics.json"
    if not metrics_path.is_file():
        render_notice(
            "No evaluation metrics are available yet. Train and evaluate the model "
            "with python train.py; this page will not show placeholder results."
        )
        return

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        required = (
            "test_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        )
        if not all(key in metrics for key in required):
            raise ValueError("The evaluation file is missing required metric fields.")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        st.error(f"Could not read saved evaluation metrics: {error}")
        return

    st.caption(
        f"Held-out test split: {metrics.get('test_image_count', 'unknown')} images"
    )
    metric_columns = st.columns(4, gap="medium")
    for column, label, key in zip(
        metric_columns,
        ("Test accuracy", "Macro precision", "Macro recall", "Macro F1-score"),
        required,
    ):
        column.metric(label, f"{float(metrics[key]) * 100:.1f}%")

    comparison_path = OUTPUT_DIR / "model_comparison.json"
    if comparison_path.is_file():
        try:
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            baseline = comparison["original_cnn"]
            transfer = comparison["mobilenetv2_transfer_learning"]
            comparison_table = pd.DataFrame(
                {
                    "Metric": [
                        "Accuracy",
                        "Macro precision",
                        "Macro recall",
                        "Macro F1-score",
                    ],
                    "Original CNN": [
                        f"{baseline['accuracy']:.1%}",
                        f"{baseline['macro_precision']:.1%}",
                        f"{baseline['macro_recall']:.1%}",
                        f"{baseline['macro_f1']:.1%}",
                    ],
                    "MobileNetV2": [
                        f"{transfer['accuracy']:.1%}",
                        f"{transfer['macro_precision']:.1%}",
                        f"{transfer['macro_recall']:.1%}",
                        f"{transfer['macro_f1']:.1%}",
                    ],
                }
            ).set_index("Metric")
            st.markdown("### CNN vs MobileNetV2")
            st.caption(
                comparison.get(
                    "evaluation",
                    "Model comparison results saved during training.",
                )
            )
            st.table(comparison_table)
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            st.error(f"Could not read the saved model comparison: {error}")
    else:
        render_notice(
            "model_comparison.json is not available. Run training with the "
            "baseline model present to produce a same-split comparison."
        )

    st.markdown("### Training and evaluation charts")
    left, right = st.columns(2, gap="large")
    with left:
        show_performance_image("Confusion matrix · MobileNetV2", "confusion_matrix.png")
    with right:
        show_performance_image("Confusion matrix · original CNN", "baseline_confusion_matrix.png")
    left, right = st.columns(2, gap="large")
    with left:
        show_performance_image("Training and validation accuracy", "accuracy.png")
    with right:
        show_performance_image("Training and validation loss", "loss.png")

    report_path = OUTPUT_DIR / "classification_report.txt"
    if report_path.is_file():
        st.markdown("### Per-grade results")
        with st.expander("View MobileNetV2 classification report"):
            try:
                st.code(report_path.read_text(encoding="utf-8"), language="text")
            except OSError as error:
                st.error(f"Could not read the classification report: {error}")
    else:
        render_notice(
            "classification_report.txt is missing. Run python train.py to "
            "generate per-grade results."
        )
    st.markdown(
        '<p class="muted-caption">IDRiD has limited and imbalanced class counts. '
        'Metrics from this small held-out split are preliminary; they do not '
        'establish clinical performance.</p>',
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_styles()
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"
    render_sidebar()

    page = st.session_state.get("page", "Home")
    if page == "Home":
        render_home()
    elif page == "Prediction":
        render_prediction()
    elif page == "About Project":
        render_about()
    else:
        render_performance()


if __name__ == "__main__":
    main()
