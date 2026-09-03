"""English Streamlit interface for the three EuroLepi ID workflows."""

from __future__ import annotations

from html import escape
import os
from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st

from eurolepi.batch import identify_batch
from eurolepi.dataset_package import inspect_dataset_zip
from eurolepi.prediction import ButterflyIdentifier
from eurolepi.registry import ModelRegistry
from eurolepi.training import TrainingOptions, train_from_zip


WORKSPACE = Path(os.environ.get("EUROLEPI_WORKSPACE", "workspace"))
MODELS_ROOT = WORKSPACE / "models"
PAGE_NAMES = ("Train Model", "Single Identification", "Batch Identification")


st.set_page_config(page_title="EuroLepi ID", page_icon="🦋", layout="wide")
st.markdown(
    """
    <style>
    :root {
      --night: #102A36;
      --lake: #286F7B;
      --sky: #86B8C0;
      --mist: #EEF4F3;
      --paper: #FAFCFB;
      --line: #C7D7D5;
      --amber: #D99B42;
      --ink-soft: #587078;
    }
    .stApp { background: var(--mist); color: var(--night); }
    [data-testid="stSidebar"] {
      background: var(--night);
      border-right: 1px solid #274651;
    }
    [data-testid="stSidebar"] * { color: #EFF7F6; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
      padding: .65rem .7rem;
      border-left: 3px solid transparent;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
      background: #183B48;
      border-left-color: var(--amber);
    }
    h1, h2, h3 { color: var(--night); letter-spacing: -.035em; }
    h1 { font-family: Georgia, 'Times New Roman', serif; font-weight: 500; }
    .brand {
      color: #F6FBFA;
      font-family: Georgia, 'Times New Roman', serif;
      font-size: 1.45rem;
      line-height: 1;
      margin: .75rem 0 .25rem;
    }
    .brand-note {
      color: #A8C5C8;
      font: 600 .67rem/1.5 ui-monospace, SFMono-Regular, Consolas, monospace;
      letter-spacing: .12em;
      text-transform: uppercase;
      margin-bottom: 2rem;
    }
    .eyebrow {
      color: var(--lake);
      font: 700 .72rem/1.2 ui-monospace, SFMono-Regular, Consolas, monospace;
      letter-spacing: .14em;
      text-transform: uppercase;
      margin: .4rem 0 .55rem;
    }
    .lede { color: var(--ink-soft); font-size: 1.02rem; max-width: 52rem; margin-bottom: 1.5rem; }
    .contract {
      background: var(--paper);
      border: 1px solid var(--line);
      border-top: 5px solid var(--lake);
      padding: 1rem 1.15rem;
      margin: .6rem 0 1.25rem;
      box-shadow: 7px 7px 0 rgba(40,111,123,.08);
    }
    .contract strong { color: var(--night); }
    .contract code { color: var(--lake); }
    .candidate-row {
      display: grid;
      grid-template-columns: 2.4rem minmax(12rem, 1fr) 5rem;
      gap: .8rem;
      align-items: center;
      border-bottom: 1px solid var(--line);
      padding: .72rem 0 .45rem;
    }
    .rank {
      color: var(--lake);
      font: 700 .78rem/1 ui-monospace, SFMono-Regular, Consolas, monospace;
    }
    .species { font: italic 1.08rem/1.2 Georgia, 'Times New Roman', serif; }
    .probability { font: 700 .82rem/1 ui-monospace, SFMono-Regular, Consolas, monospace; text-align:right; }
    .confidence-track { grid-column: 2 / 4; height: 5px; background: #DCE8E7; }
    .confidence-fill { height: 100%; background: var(--lake); }
    .reference-caption {
      color: var(--ink-soft);
      font: 600 .68rem/1.3 ui-monospace, SFMono-Regular, Consolas, monospace;
      letter-spacing: .06em;
      text-transform: uppercase;
    }
    div[data-testid="stMetric"] {
      background: var(--paper);
      border: 1px solid var(--line);
      padding: .65rem .8rem;
    }
    .stButton > button, .stDownloadButton > button {
      border-radius: 2px;
      border: 1px solid var(--lake);
      background: var(--lake);
      color: white;
      font-weight: 700;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
      background: var(--night);
      border-color: var(--night);
      color: white;
    }
    button:focus, input:focus, [tabindex]:focus {
      outline: 3px solid var(--amber) !important;
      outline-offset: 2px;
    }
    @media (max-width: 700px) {
      .candidate-row { grid-template-columns: 2rem 1fr 4.5rem; }
    }
    @media (prefers-reduced-motion: reduce) {
      * { scroll-behavior: auto !important; transition: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(f'<div class="eyebrow">{escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="lede">{escape(description)}</div>', unsafe_allow_html=True)


def model_selector(label: str, key: str):
    records = ModelRegistry(MODELS_ROOT).list_models()
    if not records:
        st.info("No trained model is available. Complete Train Model first.")
        return None
    by_id = {record.model_id: record for record in records}
    selected_id = st.selectbox(
        label,
        options=list(by_id),
        format_func=lambda model_id: by_id[model_id].display_name,
        key=key,
    )
    return by_id[selected_id]


@st.cache_resource(show_spinner=False)
def load_identifier(model_id: str, models_root: str):
    record = ModelRegistry(Path(models_root)).get(model_id)
    return ButterflyIdentifier(record)


def render_candidates(result) -> None:
    st.subheader("Top-5 candidates")
    for candidate in result.candidates:
        width = max(0.0, min(100.0, candidate.probability * 100))
        reference_column, result_column = st.columns([1, 3.2], gap="medium")
        with reference_column:
            if candidate.reference_image:
                st.image(str(candidate.reference_image), width="stretch")
                st.markdown(
                    '<div class="reference-caption">Training reference</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Reference image unavailable")
        with result_column:
            st.markdown(
                "<div class='candidate-row'>"
                f"<span class='rank'>#{candidate.rank:02d}</span>"
                f"<span class='species'>{escape(candidate.scientific_name)}</span>"
                f"<span class='probability'>{candidate.probability:.1%}</span>"
                "<div class='confidence-track'>"
                f"<div class='confidence-fill' style='width:{width:.2f}%'></div></div>"
                "</div>",
                unsafe_allow_html=True,
            )


with st.sidebar:
    st.markdown('<div class="brand">EuroLepi ID</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-note">European butterfly classifier</div>', unsafe_allow_html=True)
    page = st.radio("Workflow", PAGE_NAMES, label_visibility="collapsed")
    st.divider()
    st.caption("Models and uploaded datasets remain on this machine and are not committed to Git.")


if page == "Train Model":
    page_header(
        "Workflow 01 · Build",
        "Train Model",
        "Create a reusable European butterfly classifier from one controlled dataset package.",
    )
    st.markdown(
        """
        <div class="contract">
          <strong>Upload one ZIP dataset.</strong><br>
          The ZIP root must contain <code>manifest.csv</code> and an <code>images/</code> folder.
          Only JPG, JPEG, and PNG images are accepted. The manifest must contain
          <code>image_path</code>, <code>scientific_name</code>, and <code>specimen_id</code>.
          Prepare at least 5 species and 3 distinct specimens per species. Crop museum labels,
          QR codes, catalogue numbers, rulers, and colour cards out of every training image.
        </div>
        """,
        unsafe_allow_html=True,
    )
    name_column, upload_column = st.columns([1, 1.45], gap="large")
    with name_column:
        dataset_name = st.text_input(
            "Training dataset name",
            placeholder="European butterflies — field survey 2027",
            help="This name appears in the model selector after training.",
        )
    with upload_column:
        archive = st.file_uploader(
            "Training dataset (.zip)",
            type=["zip"],
            accept_multiple_files=False,
            max_upload_size=2048,
        )

    inspection = None
    archive_bytes = b""
    if archive is not None:
        archive_bytes = archive.getvalue()
        with st.spinner("Checking archive structure, labels, and image files…"):
            inspection = inspect_dataset_zip(archive_bytes)
        metric_columns = st.columns(3)
        metric_columns[0].metric("Images", inspection.image_count)
        metric_columns[1].metric("Specimens", inspection.specimen_count)
        metric_columns[2].metric("Species", inspection.species_count)
        for error in inspection.errors:
            st.error(error)
        for warning in inspection.warnings:
            st.warning(warning)
        if inspection.valid and inspection.manifest is not None:
            st.success("Dataset package passed all blocking checks.")
            summary = (
                inspection.manifest.groupby("scientific_name")
                .agg(images=("image_path", "count"), specimens=("specimen_id", "nunique"))
                .reset_index()
            )
            st.dataframe(summary, hide_index=True, width="stretch")

    with st.expander("Training settings"):
        settings = st.columns(3)
        epochs = settings[0].number_input("Epochs", 1, 200, 20)
        batch_size = settings[1].selectbox("Batch size", [4, 8, 16, 32, 64], index=2)
        learning_rate = settings[2].number_input(
            "Learning rate", min_value=0.000001, max_value=0.01, value=0.0003, format="%.6f"
        )

    ready = bool(dataset_name.strip()) and inspection is not None and inspection.valid
    if st.button("Train model", type="primary", disabled=not ready, width="stretch"):
        progress_bar = st.progress(0.0, text="Preparing training run…")
        status = st.empty()

        def update_progress(epoch, total, metrics):
            progress_bar.progress(
                epoch / total,
                text=f"Epoch {epoch}/{total} · validation accuracy {metrics['validation_accuracy']:.1%}",
            )
            status.caption(
                f"Training loss {metrics['train_loss']:.4f} · "
                f"training accuracy {metrics['train_accuracy']:.1%}"
            )

        try:
            record = train_from_zip(
                archive_bytes,
                dataset_name.strip(),
                WORKSPACE,
                TrainingOptions(
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    learning_rate=float(learning_rate),
                ),
                update_progress,
            )
            load_identifier.clear()
            st.success(f"Model trained and registered: {record.display_name}")
        except (RuntimeError, ValueError, OSError) as exc:
            st.error(str(exc))


elif page == "Single Identification":
    page_header(
        "Workflow 02 · Compare",
        "Single Identification",
        "Choose a training dataset, upload one butterfly photograph, and inspect five visual matches.",
    )
    record = model_selector("Training dataset / model", "single_model")
    image_file = st.file_uploader(
        "Butterfly image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="single_image",
    )
    if record and image_file:
        try:
            image = Image.open(image_file).convert("RGB")
        except (OSError, ValueError) as exc:
            st.error(f"The uploaded image cannot be read: {exc}")
            st.stop()
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            st.image(image, caption=image_file.name, width="stretch")
        with right:
            st.markdown(
                '<div class="contract"><strong>Selected training dataset</strong><br>'
                f"{escape(record.dataset_name)}<br><small>{len(record.species)} species · "
                f"{escape(record.backbone)}</small></div>",
                unsafe_allow_html=True,
            )
            identify = st.button("Identify butterfly", type="primary", width="stretch")
        if identify:
            try:
                with st.spinner("Comparing the image with the selected species set…"):
                    identifier = load_identifier(record.model_id, str(MODELS_ROOT))
                    result = identifier.predict(image, top_k=5)
                render_candidates(result)
            except (RuntimeError, ValueError, OSError, KeyError) as exc:
                st.error(str(exc))
    elif record:
        st.info("Upload one JPG, JPEG, or PNG butterfly photograph to begin.")


else:
    page_header(
        "Workflow 03 · Process",
        "Batch Identification",
        "Select a trained dataset, upload an image folder, and export one Top-5 row per image.",
    )
    record = model_selector("Training dataset / model", "batch_model")
    folder_files = st.file_uploader(
        "Butterfly image folder",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files="directory",
        key="batch_folder",
        help="All supported images in the selected folder and its subfolders will be uploaded.",
    )
    if folder_files:
        st.caption(f"{len(folder_files)} image files selected.")
    if record and folder_files and st.button(
        "Run batch identification", type="primary", width="stretch"
    ):
        try:
            with st.spinner(f"Identifying {len(folder_files)} images…"):
                identifier = load_identifier(record.model_id, str(MODELS_ROOT))
                file_payloads = [(item.name, item.getvalue()) for item in folder_files]
                st.session_state["batch_results"] = identify_batch(file_payloads, identifier)
                st.session_state["batch_model_id"] = record.model_id
        except (RuntimeError, ValueError, OSError, KeyError) as exc:
            st.error(str(exc))

    batch_results: pd.DataFrame | None = st.session_state.get("batch_results")
    if batch_results is not None and record is not None:
        if st.session_state.get("batch_model_id") == record.model_id:
            failed = int((batch_results["error"] != "").sum())
            metric_columns = st.columns(3)
            metric_columns[0].metric("Processed", len(batch_results))
            metric_columns[1].metric("Completed", len(batch_results) - failed)
            metric_columns[2].metric("Errors", failed)
            st.dataframe(batch_results, hide_index=True, width="stretch")
            st.download_button(
                "Download identification_results.csv",
                batch_results.to_csv(index=False).encode("utf-8-sig"),
                file_name="identification_results.csv",
                mime="text/csv",
                type="primary",
                width="stretch",
            )
        else:
            st.info("Run the folder again after changing the selected training dataset.")
