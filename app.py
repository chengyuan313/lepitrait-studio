"""Streamlit GUI for LepiTrait Studio."""

from __future__ import annotations

from hashlib import sha1
import json

import pandas as pd
import streamlit as st
from PIL import Image

from lepitrait.export import flatten_record
from lepitrait.image import load_rgb, overlay_mask
from lepitrait.metadata import (
    OcrUnavailableError,
    ocr_label,
    parse_label,
    suggested_label_crop,
)
from lepitrait.pipeline import AnalysisPipeline, PipelineConfig
from lepitrait.schema import ViewSide


st.set_page_config(page_title="LepiTrait Studio", page_icon="🦋", layout="wide")


def stateful_text_input(label: str, key: str, default: str = "") -> str:
    if key not in st.session_state:
        st.session_state[key] = default
    return st.text_input(label, key=key)


def set_parsed_field_state(image_id: str, metadata) -> None:
    st.session_state[f"verbatim_name_{image_id}"] = metadata.verbatim_scientific_name or ""
    st.session_state[f"catalog_number_{image_id}"] = metadata.catalog_number or ""
    st.session_state[f"event_date_{image_id}"] = (
        metadata.event_date.isoformat() if metadata.event_date else ""
    )
    st.session_state[f"type_status_{image_id}"] = metadata.type_status or ""

st.markdown(
    """
    <style>
    :root {
      --ink: #18252b;
      --paper: #f4f7f5;
      --drawer: #dfe8e5;
      --teal: #1f8f87;
      --oxide: #df7136;
      --blue: #385a7c;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #dfe8e5; border-right: 1px solid #bdccc7; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
    .specimen-tab {
      display: inline-block; padding: .35rem .75rem; margin-bottom: .6rem;
      border-left: 4px solid var(--oxide); background: #edf2f0;
      font-size: .76rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase;
    }
    .calibration-rule {
      height: 7px; margin: .5rem 0 1.25rem;
      background: repeating-linear-gradient(90deg, var(--teal) 0 1px, transparent 1px 18px);
      border-bottom: 1px solid var(--teal);
    }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #cedbd7; padding: .75rem; }
    .status-note { border-left: 3px solid var(--blue); padding: .55rem .8rem; background: #e9eef2; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="specimen-tab">Standardized specimen workstation · v0.2</div>', unsafe_allow_html=True)
st.title("LepiTrait Studio")
st.caption("Measure the specimen. Preserve the evidence. Review every inference.")
st.markdown('<div class="calibration-rule"></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Specimen card")
    specimen_id = st.text_input("Specimen ID", value="MZH-0001")
    view_side = st.selectbox("View", options=[side.value for side in ViewSide])
    pixels_per_mm = st.number_input("Scale (pixels/mm)", min_value=0.0, value=0.0, step=0.1, help="Enter 0 when the ruler has not been calibrated.")
    colour_calibrated = st.checkbox("Colour-card calibrated")
    st.divider()
    background_threshold = st.slider("Background threshold", 180, 254, 238)
    min_saturation = st.slider("Minimum chroma", 0, 60, 10)
    st.caption("These controls tune the transparent baseline mask. Production measurements should use a validated LEPY runner.")

uploaded = st.file_uploader("Drop one standardized dorsal or ventral photograph", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded is None:
    st.info("Upload a specimen image to open the calibrated review workspace.")
    st.markdown("""
    **Required frame:** centered specimen, neutral background, millimetre ruler and colour card.  
    **Classifier rule:** the model receives only the segmented specimen crop—never the label or ruler.
    """)
    st.stop()

uploaded_bytes = uploaded.getvalue()
rgb = load_rgb(uploaded_bytes)
image_state_id = sha1(uploaded_bytes).hexdigest()[:12]
pipeline = AnalysisPipeline(PipelineConfig(background_threshold=background_threshold, min_saturation=min_saturation))
result = pipeline.analyse(
    rgb,
    specimen_id=specimen_id,
    image_name=uploaded.name,
    pixels_per_mm=pixels_per_mm or None,
    colour_calibrated=colour_calibrated,
    view_side=ViewSide(view_side),
)

image_tab, metadata_tab, data_tab, protocol_tab = st.tabs(["Image review", "Label record", "Data & export", "Capture protocol"])

with image_tab:
    left, right = st.columns([1.65, 1], gap="large")
    with left:
        original_col, mask_col = st.columns(2)
        original_col.image(rgb, caption="Original frame", width="stretch")
        mask_col.image(overlay_mask(rgb, result.segmentation.mask), caption="Baseline mask", width="stretch")
    with right:
        st.subheader("Automated checks")
        metric_a, metric_b = st.columns(2)
        metric_a.metric("Foreground", f"{result.segmentation.foreground_fraction:.1%}")
        metric_b.metric("Components", result.segmentation.component_count)
        for flag in result.record.quality_flags:
            if flag.severity.value == "error":
                st.error(f"{flag.code}: {flag.message}")
            elif flag.severity.value == "warning":
                st.warning(f"{flag.code}: {flag.message}")
            else:
                st.success(flag.message)

        st.subheader("Trait preview")
        display = [m.model_dump() for m in result.record.measurements if m.unit in {"mm", "mm2"}]
        if not display:
            display = [m.model_dump() for m in result.record.measurements[:6]]
        st.dataframe(pd.DataFrame(display), hide_index=True, width="stretch")
        if result.record.colour:
            colour = result.record.colour[0]
            st.caption(f"CIELAB: L* {colour.l_mean:.1f} · a* {colour.a_mean:.1f} · b* {colour.b_mean:.1f} · dark area {colour.dark_fraction:.1%}")

        st.subheader("Species candidates")
        if result.record.predictions:
            st.dataframe(
                pd.DataFrame([prediction.model_dump() for prediction in result.record.predictions]),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No species model is configured. The specimen-only crop contract is active, so model weights can be added without changing this review screen.")

with metadata_tab:
    st.subheader("Read, structure and verify the specimen labels")
    st.caption(
        "OCR never enters the species classifier. Its output is evidence to review, not an "
        "automatic taxonomic decision."
    )

    label_upload = st.file_uploader(
        "Optional label close-up",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        key=f"label_upload_{image_state_id}",
        help="Use a close-up when the text in the full specimen frame is too small.",
    )
    specimen_pil = Image.fromarray(rgb)
    crop = None
    if label_upload is not None:
        label_image = Image.open(label_upload).convert("RGB")
        st.caption("Using the separately uploaded label photograph.")
    else:
        with st.expander("Adjust suggested label panel", expanded=False):
            st.caption(
                "The current capture layout assumes labels are on the right. Adjust these "
                "bounds when a museum batch uses another layout."
            )
            bound_a, bound_b, bound_c, bound_d = st.columns(4)
            left_percent = bound_a.number_input("Left %", 0, 95, 56, 1)
            top_percent = bound_b.number_input("Top %", 0, 95, 0, 1)
            right_percent = bound_c.number_input("Right %", 5, 100, 100, 1)
            bottom_percent = bound_d.number_input("Bottom %", 5, 100, 82, 1)
        try:
            crop = suggested_label_crop(
                specimen_pil,
                left_fraction=left_percent / 100,
                top_fraction=top_percent / 100,
                right_fraction=right_percent / 100,
                bottom_fraction=bottom_percent / 100,
            )
            label_image = specimen_pil.crop(crop.box)
        except ValueError as exc:
            st.error(str(exc))
            label_image = specimen_pil

    preview_col, action_col = st.columns([1.55, 1], gap="large")
    with preview_col:
        st.image(label_image, caption="OCR input · verify that ruler and specimen are excluded", width="stretch")
    with action_col:
        st.markdown("**OCR status**")
        st.write("Printed labels and catalogue numbers should be checked first. Historical handwriting will usually require manual correction.")
        run_ocr = st.button("Run label OCR", type="primary", width="stretch")
        if run_ocr:
            try:
                with st.spinner("Reading the label panel…"):
                    recognized = ocr_label(label_image, crop=crop)
                recognized_metadata = parse_label(recognized.text)
                st.session_state[f"ocr_text_{image_state_id}"] = recognized.text
                st.session_state[f"ocr_engine_{image_state_id}"] = recognized.engine
                set_parsed_field_state(image_state_id, recognized_metadata)
                st.success(f"OCR finished with {recognized.engine}. Review every field below.")
            except OcrUnavailableError as exc:
                st.error(str(exc))
                st.code(
                    'winget install --id UB-Mannheim.TesseractOCR\n'
                    '$env:TESSERACT_CMD="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"',
                    language="powershell",
                )
            except RuntimeError as exc:
                st.error(str(exc))

    raw_label = st.text_area(
        "Verbatim OCR text",
        height=220,
        key=f"ocr_text_{image_state_id}",
        placeholder="Run OCR, paste OCR text, or transcribe the labels exactly as printed.",
    )
    parsed = parse_label(raw_label)
    if st.button("Parse edited text"):
        set_parsed_field_state(image_state_id, parsed)
        st.rerun()
    field_a, field_b = st.columns(2)
    with field_a:
        verbatim_name = stateful_text_input(
            "Scientific name on label",
            f"verbatim_name_{image_state_id}",
            parsed.verbatim_scientific_name or "",
        )
        catalog_number = stateful_text_input(
            "Catalog number",
            f"catalog_number_{image_state_id}",
            parsed.catalog_number or "",
        )
        event_date = stateful_text_input(
            "Event date (YYYY-MM-DD)",
            f"event_date_{image_state_id}",
            parsed.event_date.isoformat() if parsed.event_date else "",
        )
        type_status = stateful_text_input(
            "Type status", f"type_status_{image_state_id}", parsed.type_status or ""
        )
    with field_b:
        accepted_name = stateful_text_input(
            "Accepted scientific name", f"accepted_name_{image_state_id}"
        )
        locality = stateful_text_input("Verbatim locality", f"locality_{image_state_id}")
        recorded_by = stateful_text_input("Recorded by", f"recorded_by_{image_state_id}")
        sex_key = f"sex_{image_state_id}"
        if sex_key not in st.session_state:
            st.session_state[sex_key] = parsed.sex or ""
        sex = st.selectbox(
            "Sex",
            options=["", "male", "female", "unknown"],
            key=sex_key,
        )

    result.record.metadata = parsed
    result.record.metadata.verbatim_scientific_name = verbatim_name or None
    result.record.metadata.accepted_scientific_name = accepted_name or None
    result.record.metadata.catalog_number = catalog_number or None
    result.record.metadata.verbatim_locality = locality or None
    result.record.metadata.recorded_by = recorded_by or None
    result.record.metadata.type_status = type_status or None
    result.record.metadata.sex = sex or None
    ocr_engine = st.session_state.get(f"ocr_engine_{image_state_id}")
    if ocr_engine:
        result.record.provenance.parameters["ocr_engine"] = ocr_engine
        if crop:
            result.record.provenance.parameters["label_crop"] = crop.box
    if event_date:
        try:
            result.record.metadata.event_date = pd.Timestamp(event_date).date()
        except ValueError:
            st.warning("Event date could not be parsed. Use YYYY-MM-DD or leave it blank.")
    metadata_display = result.record.metadata.model_dump(mode="json")
    metadata_display = {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        for key, value in metadata_display.items()
    }
    st.dataframe(
        pd.DataFrame([metadata_display]).T.rename(columns={0: "value"}), width="stretch"
    )

with data_tab:
    st.subheader("Research-ready record")
    row = flatten_record(result.record)
    frame = pd.DataFrame([row])
    st.dataframe(frame, width="stretch")
    st.download_button("Download specimen JSON", data=result.record.model_dump_json(indent=2), file_name=f"{specimen_id}.json", mime="application/json")
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    st.download_button("Download flat CSV", data=csv_bytes, file_name=f"{specimen_id}.csv", mime="text/csv")

with protocol_tab:
    st.subheader("Capture gate")
    st.markdown("""
    - Fixed camera, lens, distance, light geometry and exposure protocol.
    - Neutral, texture-free background; specimen long axis aligned consistently.
    - Millimetre ruler and colour target in every frame.
    - RAW capture retained; calibrated TIFF used for analysis.
    - Dorsal and ventral views recorded when identification requires both.
    - Device ID, imaging batch and calibration profile stored with every image.
    """)
    st.markdown('<div class="status-note">Colour values from an uncalibrated image are relative. Do not compare them across museums or imaging batches.</div>', unsafe_allow_html=True)
