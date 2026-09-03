"""Streamlit interface for EuroLepi ID inference and dataset readiness checks."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st

from eurolepi.manifest import summarize_counts, validate_manifest
from eurolepi.schemas import ImageDomain


st.set_page_config(page_title="EuroLepi ID", page_icon="🦋", layout="wide")

st.markdown(
    """
    <style>
    :root {
      --ink: #142a33;
      --mist: #edf3f4;
      --panel: #f8fbfb;
      --line: #c6d6d8;
      --wing: #176b72;
      --signal: #e49b25;
      --deep: #173f5f;
    }
    .stApp { background: var(--mist); color: var(--ink); }
    [data-testid="stSidebar"] { background: #dbe7e8; border-right: 1px solid var(--line); }
    h1 { color: var(--deep); letter-spacing: -.045em; font-weight: 780; }
    h2, h3 { color: var(--ink); letter-spacing: -.018em; }
    .taxon-strip {
      display: flex; align-items: center; gap: .65rem; margin-bottom: .4rem;
      color: var(--wing); font: 700 .74rem/1.2 ui-monospace, SFMono-Regular, Consolas, monospace;
      letter-spacing: .12em; text-transform: uppercase;
    }
    .taxon-strip:before, .taxon-strip:after { content: ""; height: 1px; background: var(--wing); }
    .taxon-strip:before { width: 2.2rem; }
    .taxon-strip:after { flex: 1; opacity: .25; }
    .decision {
      padding: 1rem 1.1rem; border: 1px solid var(--line); background: var(--panel);
      box-shadow: 5px 5px 0 rgba(23,107,114,.10); margin: .4rem 0 1rem;
    }
    .decision.review { border-left: 6px solid var(--signal); }
    .decision.accepted { border-left: 6px solid var(--wing); }
    .candidate { margin: .55rem 0 1rem; }
    .candidate-line { display:flex; justify-content:space-between; gap:1rem; font-size:.95rem; }
    .candidate-line em { font-family: Georgia, Cambria, serif; font-size:1.02rem; }
    .confidence-track { height: 7px; margin-top:.35rem; background:#dce7e8; overflow:hidden; }
    .confidence-fill { height:100%; background:linear-gradient(90deg,var(--wing),#51a59e); }
    .protocol-note { padding:.8rem 1rem; background:#e4ecee; border-left:3px solid var(--deep); }
    button:focus, input:focus, [tabindex]:focus { outline: 3px solid #72aeba !important; outline-offset: 2px; }
    @media (prefers-reduced-motion: reduce) { * { scroll-behavior:auto !important; transition:none !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_identifier(checkpoint_path: str, threshold: float):
    from eurolepi.inference import ButterflyIdentifier

    return ButterflyIdentifier(checkpoint_path, threshold=threshold)


def render_result(result) -> None:
    state = "review" if result.rejected else "accepted"
    headline = "需要专家复核" if result.rejected else escape(result.decision)
    reason = (
        f"最高置信度未达到 {result.threshold:.0%} 的拒识阈值。"
        if result.rejected
        else f"模型置信度达到设定阈值 {result.threshold:.0%}。"
    )
    st.markdown(
        f'<div class="decision {state}"><strong>{headline}</strong><br><small>{reason}</small></div>',
        unsafe_allow_html=True,
    )
    for candidate in result.predictions:
        width = max(0.0, min(100.0, candidate.probability * 100))
        st.markdown(
            "<div class='candidate'>"
            f"<div class='candidate-line'><span>{candidate.rank}. "
            f"<em>{escape(candidate.scientific_name)}</em></span>"
            f"<strong>{candidate.probability:.1%}</strong></div>"
            f"<div class='confidence-track'><div class='confidence-fill' style='width:{width:.2f}%'></div></div>"
            "</div>",
            unsafe_allow_html=True,
        )
    st.download_button(
        "下载鉴定结果",
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        file_name="identification.json",
        mime="application/json",
    )


st.markdown('<div class="taxon-strip">European butterfly identification</div>', unsafe_allow_html=True)
st.title("EuroLepi ID")
st.caption("面向欧洲蝴蝶的可训练图像鉴定框架 · MaxViT-T · Top-5 · 未知种拒识")

with st.sidebar:
    st.header("模型设置")
    checkpoint = st.text_input("本地模型路径", value="models/eurolepi_maxvit_tiny/best.pt")
    threshold = st.slider("拒识阈值", 0.0, 1.0, 0.65, 0.01)
    top_k = st.slider("候选数量", 1, 10, 5)
    st.divider()
    st.caption("模型文件不会提交到Git。训练完成后将best.pt放在models目录即可。")

identify_tab, dataset_tab, train_tab = st.tabs(["鉴定", "数据检查", "训练说明"])

with identify_tab:
    domain = st.selectbox(
        "图像类型",
        options=[item.value for item in ImageDomain],
        format_func={
            "museum_standardized": "博物馆标准标本照",
            "field_standardized": "野外采集后的标准照",
            "field_in_situ": "自然状态野外照",
        }.get,
    )
    label_free = True
    if domain == ImageDomain.MUSEUM_STANDARDIZED.value:
        label_free = st.checkbox("确认分类图像中不含标签、二维码和馆藏号", value=False)
    uploaded = st.file_uploader(
        "上传一张蝴蝶图片", type=["jpg", "jpeg", "png", "tif", "tiff"]
    )
    if uploaded is None:
        st.info("上传图片后，系统将显示Top-5候选；低于阈值时返回“需要专家复核”。")
    else:
        image = Image.open(uploaded).convert("RGB")
        image_col, result_col = st.columns([1.45, 1], gap="large")
        with image_col:
            st.image(image, caption=f"输入图像 · {domain}", width="stretch")
        with result_col:
            if not label_free:
                st.warning("请先裁掉标签、二维码、馆藏号和比例尺，避免模型利用文字作弊。")
            elif not Path(checkpoint).is_file():
                st.markdown(
                    '<div class="protocol-note"><strong>框架已就绪，尚未加载模型。</strong><br>'
                    "取得欧洲数据集并完成训练后，将生成的best.pt放入models目录。</div>",
                    unsafe_allow_html=True,
                )
            else:
                try:
                    with st.spinner("正在比较欧洲蝴蝶候选种…"):
                        identifier = load_identifier(checkpoint, threshold)
                        render_result(identifier.predict(image, top_k=top_k))
                except (RuntimeError, ValueError, KeyError) as exc:
                    st.error(str(exc))

with dataset_tab:
    st.subheader("训练数据是否可以直接使用？")
    manifest_upload = st.file_uploader("上传CSV清单", type=["csv"], key="manifest")
    if manifest_upload is None:
        st.markdown(
            "上传按照 `data/manifest.example.csv` 整理的清单。这里检查字段、物种标签、"
            "标本跨集合泄漏和标签像素是否已经移除。"
        )
    else:
        frame = pd.read_csv(manifest_upload)
        report = validate_manifest(frame, require_files=False, require_split="split" in frame.columns)
        a, b, c = st.columns(3)
        a.metric("图片", report.images)
        b.metric("标本", report.specimens)
        c.metric("物种", report.species)
        if report.valid:
            st.success("清单结构通过检查。训练前请在本机再次执行文件存在性检查。")
        for error in report.errors:
            st.error(error)
        for warning in report.warnings:
            st.warning(warning)
        if report.valid:
            st.dataframe(
                summarize_counts(frame, ["domain", "scientific_name"]),
                hide_index=True,
                width="stretch",
            )

with train_tab:
    st.subheader("拿到欧洲数据后的四条命令")
    st.code(
        """python -m pip install -e \".[ml]\"
eurolepi validate data/manifest_unsplit.csv --before-split
eurolepi split data/manifest_unsplit.csv --output data/manifest.csv
eurolepi train configs/maxvit_tiny.yaml""",
        language="powershell",
    )
    st.markdown(
        "训练生成 `best.pt`、类别顺序、训练历史和模型卡。随后执行："
    )
    st.code(
        "eurolepi evaluate models/eurolepi_maxvit_tiny/best.pt data/manifest.csv",
        language="powershell",
    )
    st.caption("野外识别能力必须在field_in_situ测试集上单独报告，不能由博物馆标本准确率代替。")
