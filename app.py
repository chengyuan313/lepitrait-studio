"""Streamlit GUI for LepiTrait Studio."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

from lepitrait.export import flatten_record
from lepitrait.image import load_rgb, overlay_mask
from lepitrait.metadata import parse_label
from lepitrait.pipeline import AnalysisPipeline, PipelineConfig
from lepitrait.schema import ViewSide


st.set_page_config(page_title="LepiTrait Studio", page_icon="🦋", layout="wide")

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

st.markdown('<div class="specimen-tab">Standardized specimen workstation · v0.1</div>', unsafe_allow_html=True)
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

rgb = load_rgb(uploaded.getvalue())
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
        original_col.image(rgb, caption="Original frame", use_container_width=True)
        mask_col.image(overlay_mask(rgb, result.segmentation.mask), caption="Baseline mask", use_container_width=True)
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
        st.dataframe(pd.DataFrame(display), hide_index=True, use_container_width=True)
        if result.record.colour:
            colour = result.record.colour[0]
            st.caption(f"CIELAB: L* {colour.l_mean:.1f} · a* {colour.a_mean:.1f} · b* {colour.b_mean:.1f} · dark area {colour.dark_fraction:.1%}")

        st.subheader("Species candidates")
        if result.record.predictions:
            st.dataframe(
                pd.DataFrame([prediction.model_dump() for prediction in result.record.predictions]),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No species model is configured. The specimen-only crop contract is active, so model weights can be added without changing this review screen.")

with metadata_tab:
    st.subheader("Transcribe the label without contaminating the classifier")
    raw_label = st.text_area("Verbatim label", height=180, placeholder="Paste OCR text or type the label exactly as printed.")
    parsed = parse_label(raw_label)
    accepted_name = st.text_input("Accepted scientific name", value=parsed.verbatim_scientific_name or "")
    catalog_number = st.text_input("Catalog number")
    locality = st.text_input("Verbatim locality")
    if raw_label:
        result.record.metadata = parsed
        result.record.metadata.accepted_scientific_name = accepted_name or None
        result.record.metadata.catalog_number = catalog_number or None
        result.record.metadata.verbatim_locality = locality or None
    st.dataframe(pd.DataFrame([result.record.metadata.model_dump(mode="json")]).T.rename(columns={0: "value"}), use_container_width=True)

with data_tab:
    st.subheader("Research-ready record")
    row = flatten_record(result.record)
    frame = pd.DataFrame([row])
    st.dataframe(frame, use_container_width=True)
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

