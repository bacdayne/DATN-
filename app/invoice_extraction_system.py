import streamlit as st
import json
import os
import sys
import ast
import tempfile
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.agentic_prompts import classify_template_agent, get_prompt_by_agent, BASE_SCHEMA

st.set_page_config(
    page_title="Invoice Extraction System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PREDICTION_FILE   = "pipeline/predictions/predictions_test_finetuned.json"
SUMMARY_FILE      = "evaluation/evaluation_summary.json"
FIELD_METRICS_FILE = "evaluation/field_metrics.csv"
INSTANCE_ACC_FILE  = "evaluation/instance_accuracy.csv"


@st.cache_data
def load_predictions():
    if not os.path.exists(PREDICTION_FILE):
        return []
    with open(PREDICTION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_summary():
    if not os.path.exists(SUMMARY_FILE):
        return {}
    with open(SUMMARY_FILE) as f:
        return json.load(f)


@st.cache_data
def load_field_metrics():
    if not os.path.exists(FIELD_METRICS_FILE):
        return pd.DataFrame()
    return pd.read_csv(FIELD_METRICS_FILE)

@st.cache_data
def load_instance_acc():
    if not os.path.exists(INSTANCE_ACC_FILE):
        return pd.DataFrame()
    return pd.read_csv(INSTANCE_ACC_FILE)

PREDICTIONS   = load_predictions()
SUMMARY       = load_summary()
FIELD_METRICS = load_field_metrics()
INSTANCE_ACC  = load_instance_acc()


@st.cache_resource
def load_ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)


@st.cache_resource
def load_vllm():
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(
            model="finetuning/qwen2-cord-merged",
            dtype="float16",
            max_model_len=1024,
            max_num_seqs=1,
            gpu_memory_utilization=0.35,
            enforce_eager=True,
        )
        sp = SamplingParams(
            temperature=0,
            max_tokens=600,
            repetition_penalty=1.03,
            stop=["```", "<|im_end|>", "\nOCR:"],
        )
        return llm, sp
    except Exception as e:
        return None, str(e)

KEYS_TO_EVALUATE = [
    "menu.nm", "menu.cnt", "menu.unitprice", "menu.price",
    "sub_total.subtotal_price", "sub_total.discount_price", "sub_total.tax_price",
    "total.total_price", "total.cashprice", "total.changeprice", "total.creditcardprice",
]

def normalize_value(v):
    if v is None:
        return ""
    if isinstance(v, list):
        v = " ".join(str(x) for x in v)
    v = str(v).strip().lower()
    for s in ["rp.", "rp", "idr", " ", "@", "[", "]", "(", ")", ",", "."]:
        v = v.replace(s, "")
    return v

def extract_values(data, level, key):
    values = []
    if not isinstance(data, dict):
        return values
    if level == "menu":
        menu = data.get("menu", [])
        if isinstance(menu, dict):
            menu = [menu]
        if not isinstance(menu, list):
            return values
        for item in menu:
            if isinstance(item, dict) and key in item:
                v = str(item[key]).strip()
                if v:
                    values.append(v)
    else:
        section = data.get(level, {})
        if isinstance(section, dict) and key in section:
            v = str(section[key]).strip()
            if v:
                values.append(v)
    return values


def pretty_json(value):
    try:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                return value
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def normalize_path(path):
    return path.replace("ocr_output/", "ocr/") if path else path


def get_debug_image(index, image_path):
    candidates = [
        f"ocr/debug/test_{index}_debug.png",
        f"ocr/debug/test_{index}.png",
        f"ocr/debug/{index}.png",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return image_path


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: Inter, sans-serif !important; }

#MainMenu, footer, header              { visibility: hidden; }
[data-testid="collapsedControl"]       { display: none !important; }
section[data-testid="stSidebar"]       { display: none !important; }
[data-testid="stDecoration"]           { display: none !important; }

.stApp { background: #f0f4f9 !important; }

/* Topbar card */
.topbar {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
    padding: 14px 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
    flex-wrap: wrap;
    gap: 12px;
}
.brand { display: flex; align-items: center; gap: 10px; }
.brand-line {
    width: 4px; height: 30px; border-radius: 4px;
    background: linear-gradient(180deg,#06b6d4,#6366f1); flex-shrink: 0;
}
.brand-title { font-size: 17px; font-weight: 700; color: #0f172a; }
.model-tag {
    padding: 3px 10px; border-radius: 999px;
    background: #f1f5f9; border: 1px solid #e2e8f0;
    font-size: 11px; font-weight: 600; color: #64748b;
}
.badges { display: flex; gap: 8px; flex-wrap: wrap; }
.badge {
    padding: 4px 11px; border-radius: 999px;
    font-size: 11px; font-weight: 600;
    display: flex; align-items: center; gap: 6px;
}
.badge span { width: 6px; height: 6px; border-radius: 50%; }
.badge.green { background:#f0fdf4; border:1px solid #bbf7d0; color:#15803d; }
.badge.green span { background:#22c55e; }
.badge.cyan  { background:#ecfeff; border:1px solid #a5f3fc; color:#0e7490; }
.badge.cyan  span { background:#06b6d4; }

/* Metrics row */
.metrics-row { display:flex; gap:12px; margin-bottom:14px; flex-wrap:wrap; }
.metric-card {
    flex:1; min-width:110px; background:white;
    border:1px solid #e2e8f0; border-radius:12px;
    padding:14px 16px; box-shadow:0 1px 3px rgba(0,0,0,.05);
}
.metric-card.hl {
    background: linear-gradient(135deg,#0e7490,#06b6d4);
    border-color: transparent;
}
.metric-val  { font-size:20px; font-weight:700; color:#0f172a; letter-spacing:-.4px; line-height:1; margin-bottom:4px; }
.metric-lbl  { font-size:10px; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:.3px; }
.metric-card.hl .metric-val,
.metric-card.hl .metric-lbl { color:white; }

/* Control bar */
.ctrl-bar {
    background:white; border:1px solid #e2e8f0; border-radius:14px;
    box-shadow:0 1px 4px rgba(0,0,0,.05); padding:14px 18px;
    display:flex; align-items:center; gap:14px; margin-bottom:14px; flex-wrap:wrap;
}

/* Section label */
.slabel {
    font-size:10px; font-weight:700; color:#94a3b8;
    text-transform:uppercase; letter-spacing:.5px; margin-bottom:5px;
}

/* Card wrapper */
.card {
    background:white; border:1px solid #e2e8f0; border-radius:14px;
    box-shadow:0 1px 4px rgba(0,0,0,.05); overflow:hidden; margin-bottom:14px;
}
.card-head {
    padding:10px 16px; font-size:12px; font-weight:700;
    color:#64748b; text-transform:uppercase; letter-spacing:.4px;
    border-bottom:1px solid #f1f5f9;
}

/* Code headers */
.code-hd {
    padding:10px 16px; border-radius:10px 10px 0 0;
    font-size:12px; font-weight:700; letter-spacing:.2px;
}
.code-hd.pred { background:#0f172a; color:#7dd3fc; }
.code-hd.gt   { background:#1e1b4b; color:#c4b5fd; }

/* Chips */
.chip { display:inline-block; padding:4px 12px; border-radius:999px; font-size:12px; font-weight:600; }
.chip-agent { background:#f0f9ff; border:1px solid #bae6fd; color:#0369a1; }
.chip-ok    { background:#f0fdf4; border:1px solid #bbf7d0; color:#15803d; }
.chip-fail  { background:#fff1f2; border:1px solid #fecdd3; color:#be123c; }

.meta-row { display:flex; gap:20px; align-items:center; padding:10px 0 6px 0; }
.meta-item { display:flex; align-items:center; gap:8px; }
.meta-lbl  { font-size:10px; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:.4px; }

/* Streamlit overrides */
.stCode { border-radius: 0 0 10px 10px !important; }
[data-testid="stTextArea"] textarea {
    font-size: 12.5px !important;
    line-height: 1.65 !important;
    border: none !important;
}
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#0891b2,#06b6d4) !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(6,182,212,.35) !important;
}
::-webkit-scrollbar       { width:6px; height:6px; }
::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:20px; }
button[data-baseweb="tab"] svg { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── TOPBAR ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="brand">
        <div class="brand-line"></div>
        <div class="brand-title">Invoice Extraction System</div>
        <span class="model-tag">Qwen2-1.5B · LoRA</span>
    </div>
    <div class="badges">
        <div class="badge green"><span></span>OCR Ready</div>
        <div class="badge cyan"><span></span>JSON Pipeline</div>
        <div class="badge green"><span></span>vLLM Connected</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_main, tab_history, tab_eval = st.tabs(["Extraction", f"History ({len(st.session_state.history)})", "Evaluation"])

# ── TAB 1: EXTRACTION ─────────────────────────────────────────────────────────
with tab_main:

    choices = [
        f"{i} — {os.path.basename(item.get('image_path', f'invoice_{i}.png'))}"
        for i, item in enumerate(PREDICTIONS)
    ]

    c1, c2, c3 = st.columns([6, 1, 1])
    with c1:
        selected = st.selectbox("sample", choices, label_visibility="collapsed")
    with c2:
        run = st.button("▶  RUN", type="primary", use_container_width=True)
    with c3:
        if st.button("↺  RESET", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    index = int(selected.split(" — ")[0])
    item  = PREDICTIONS[index]

    st.markdown("<hr style='margin:6px 0 14px 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)

    if run:
        image_path = normalize_path(item.get("image_path", ""))
        ocr_text   = item.get("ocr_text", "") or "No OCR text."
        prediction = pretty_json(item.get("prediction", "{}"))
        agent      = item.get("agent_used", item.get("template_id", "general_agent"))
        parse_ok   = item.get("parse_ok", False)
        debug_path = get_debug_image(index, image_path)

        existing = [h["index"] for h in st.session_state.history]
        if index not in existing:
            st.session_state.history.append({
                "index":      index,
                "image_name": os.path.basename(image_path),
                "image_path": image_path,
                "agent":      agent,
                "parse_ok":   parse_ok,
                "prediction": prediction,
            })

        col_img, col_ocr, col_txt = st.columns([3, 4, 3])
        with col_img:
            st.markdown('<div class="slabel">Input Image</div>', unsafe_allow_html=True)
            if image_path and os.path.exists(image_path):
                st.image(image_path, use_container_width=True)
        with col_ocr:
            st.markdown('<div class="slabel">OCR Detection</div>', unsafe_allow_html=True)
            if debug_path and os.path.exists(debug_path):
                st.image(debug_path, use_container_width=True)
        with col_txt:
            st.markdown('<div class="slabel">Raw OCR Stream</div>', unsafe_allow_html=True)
            st.text_area("ocr", ocr_text, height=320, disabled=True, label_visibility="collapsed")

        parse_chip = '<span class="chip chip-ok">✓ Parse OK</span>' if parse_ok else \
                     '<span class="chip chip-fail">✗ Parse Fail</span>'

        acc_row = INSTANCE_ACC[INSTANCE_ACC["invoice_no"] == index] if not INSTANCE_ACC.empty else pd.DataFrame()
        acc_val = acc_row["accuracy"].values[0] if not acc_row.empty else None
        acc_color = "#15803d" if acc_val and acc_val >= 80 else "#b45309" if acc_val and acc_val >= 50 else "#be123c"
        acc_chip  = f'<span class="chip" style="background:#f8fafc;border:1px solid #e2e8f0;color:{acc_color};font-weight:700;">{acc_val:.1f}%</span>' if acc_val is not None else ""

        st.markdown(f"""
        <div class="meta-row">
            <div class="meta-item"><span class="meta-lbl">Agent</span><span class="chip chip-agent">{agent}</span></div>
            <div class="meta-item"><span class="meta-lbl">Status</span>{parse_chip}</div>
            {"<div class='meta-item'><span class='meta-lbl'>Accuracy</span>" + acc_chip + "</div>" if acc_chip else ""}
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="code-hd pred">&lt;&gt; Prediction</div>', unsafe_allow_html=True)
        st.code(prediction, language="json")

    else:
        col_img, col_ph = st.columns([3, 7])
        with col_img:
            st.markdown('<div class="slabel">Input Image</div>', unsafe_allow_html=True)
            image_path = normalize_path(item.get("image_path", ""))
            if image_path and os.path.exists(image_path):
                st.image(image_path, use_container_width=True)
        with col_ph:
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;height:320px;
                        background:white;border-radius:14px;border:1px solid #e2e8f0;
                        color:#94a3b8;font-size:14px;font-weight:500;gap:6px;">
                Chọn sample và bấm <strong style="color:#06b6d4;">▶ RUN</strong>
            </div>
            """, unsafe_allow_html=True)


# ── TAB 2: HISTORY ────────────────────────────────────────────────────────────
with tab_history:
    if not st.session_state.history:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:center;height:200px;
                    background:white;border-radius:14px;border:1px solid #e2e8f0;
                    color:#94a3b8;font-size:14px;font-weight:500;">
            Chưa có sample nào được xử lý — bấm ▶ RUN ở tab Extraction
        </div>
        """, unsafe_allow_html=True)
    else:
        for h in reversed(st.session_state.history):
            parse_chip = '<span class="chip chip-ok">✓ Parse OK</span>' if h["parse_ok"] else \
                         '<span class="chip chip-fail">✗ Parse Fail</span>'
            with st.expander(f"#{h['index']}  —  {h['image_name']}", expanded=False):
                col_a, col_b = st.columns([2, 8])
                with col_a:
                    if h["image_path"] and os.path.exists(h["image_path"]):
                        st.image(h["image_path"], use_container_width=True)
                with col_b:
                    st.markdown(f"""
                    <div class="meta-row" style="margin-bottom:10px;">
                        <div class="meta-item">
                            <span class="meta-lbl">Agent</span>
                            <span class="chip chip-agent">{h['agent']}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-lbl">Status</span>
                            {parse_chip}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('<div class="code-hd pred">&lt;&gt; Prediction</div>', unsafe_allow_html=True)
                    st.code(h["prediction"], language="json")


# ── TAB 3: EVALUATION ─────────────────────────────────────────────────────────
# with tab_eval:
#     if SUMMARY:
#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("Invoices",       int(SUMMARY.get("num_invoices", 0)))
#         c2.metric("Mean Accuracy",  f"{SUMMARY.get('mean_instance_accuracy', 0):.2f}%")
#         c3.metric("Mean F1",        f"{SUMMARY.get('mean_field_f1', 0)*100:.2f}%")
#         c4.metric("Mean Precision", f"{SUMMARY.get('mean_field_precision', 0)*100:.2f}%")

#     st.markdown("<hr style='margin:10px 0 16px 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)

#     # ── AGENT DISTRIBUTION & FIELD METRICS ───────────────────────────────────
#     col_agent, col_fm = st.columns([5, 5])

#     with col_agent:
#         st.markdown('<div class="slabel">Phân bố Agent</div>', unsafe_allow_html=True)
#         if PREDICTIONS:
#             agent_counts = Counter(item.get("agent_used", "unknown") for item in PREDICTIONS)
#             df_agent = pd.DataFrame(list(agent_counts.items()), columns=["Agent", "Số hóa đơn"])
#             df_agent = df_agent.sort_values("Số hóa đơn", ascending=True)
#             st.bar_chart(df_agent.set_index("Agent"), color="#06b6d4", height=300)

#     with col_fm:
#         st.markdown('<div class="slabel">Field Metrics (Precision / Recall / F1)</div>', unsafe_allow_html=True)
#         if not FIELD_METRICS.empty:
#             df_show = FIELD_METRICS[["key", "precision", "recall", "f1_score"]].copy()
#             df_show[["precision", "recall", "f1_score"]] = (df_show[["precision", "recall", "f1_score"]] * 100).round(1)
#             df_show.columns = ["Field", "P (%)", "R (%)", "F1 (%)"]
#             st.dataframe(df_show, use_container_width=True, hide_index=True, height=300)

#     st.markdown("<hr style='margin:10px 0 16px 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)

#     # ── TOP 5 BEST / WORST ────────────────────────────────────────────────────
#     if not INSTANCE_ACC.empty:
#         def render_top5(df_slice, label, color):
#             st.markdown(f'<div class="slabel">{label}</div>', unsafe_allow_html=True)
#             for _, row in df_slice.iterrows():
#                 idx = int(row["invoice_no"])
#                 acc = row["accuracy"]
#                 inv_item = PREDICTIONS[idx] if idx < len(PREDICTIONS) else {}
#                 image_path = normalize_path(inv_item.get("image_path", ""))
#                 image_name = os.path.basename(image_path) if image_path else f"invoice_{idx}"
#                 agent = inv_item.get("agent_used", "")
#                 c_img, c_info = st.columns([1, 3])
#                 with c_img:
#                     if image_path and os.path.exists(image_path):
#                         st.image(image_path, use_container_width=True)
#                 with c_info:
#                     st.markdown(f"""
#                     <div style="padding:6px 0">
#                         <div style="font-weight:600;font-size:13px;color:#1e293b;">#{idx} — {image_name}</div>
#                         <div style="font-size:12px;color:#64748b;">Agent: {agent}</div>
#                         <div style="font-size:18px;font-weight:700;color:{color};">{acc:.1f}%</div>
#                     </div>
#                     """, unsafe_allow_html=True)

#         col_best, col_worst = st.columns(2)
#         with col_best:
#             render_top5(INSTANCE_ACC.nlargest(5, "accuracy"), "Top 5 — Accuracy cao nhất", "#15803d")
#         with col_worst:
#             render_top5(INSTANCE_ACC.nsmallest(5, "accuracy"), "Top 5 — Accuracy thấp nhất", "#be123c")

#     st.markdown("<hr style='margin:10px 0 16px 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)
#     st.markdown("### Ground Truth vs Prediction — từng hóa đơn")

#     # Per-invoice comparison
#     for item in PREDICTIONS:
#         i          = item.get("image_index", 0)
#         image_name = os.path.basename(item.get("image_path", f"test_{i}.png"))
#         agent      = item.get("agent_used", "")

#         pred = item.get("prediction", {})
#         if isinstance(pred, str):
#             try:
#                 pred = json.loads(pred)
#             except Exception:
#                 pred = {}
#         gt = item.get("ground_truth", {})

#         # instance accuracy
#         acc_row = INSTANCE_ACC[INSTANCE_ACC["invoice_no"] == i] if not INSTANCE_ACC.empty else pd.DataFrame()
#         acc_str = f"{acc_row['accuracy'].values[0]:.1f}%" if not acc_row.empty else "—"

#         with st.expander(f"#{i}  —  {image_name}  |  Agent: {agent}  |  Accuracy: {acc_str}", expanded=False):
#             rows = []
#             for full_key in KEYS_TO_EVALUATE:
#                 level, key = full_key.split(".")
#                 gt_vals   = extract_values(gt,   level, key)
#                 pred_vals = extract_values(pred, level, key)
#                 max_len   = max(len(gt_vals), len(pred_vals), 1) if (gt_vals or pred_vals) else 0
#                 for j in range(max_len):
#                     gt_v   = gt_vals[j]   if j < len(gt_vals)   else ""
#                     pred_v = pred_vals[j] if j < len(pred_vals) else ""
#                     match  = "✓" if normalize_value(gt_v) == normalize_value(pred_v) and gt_v != "" else ("—" if gt_v == "" else "✗")
#                     rows.append({"Field": full_key, "Ground Truth": gt_v, "Prediction": pred_v, "": match})

#             if rows:
#                 df = pd.DataFrame(rows)
#                 st.dataframe(
#                     df.style.apply(
#                         lambda col: [
#                             "background-color:#f0fdf4; color:#15803d" if v == "✓"
#                             else "background-color:#fff1f2; color:#be123c" if v == "✗"
#                             else ""
#                             for v in col
#                         ] if col.name == "" else [""] * len(col),
#                         axis=0
#                     ),
#                     use_container_width=True,
#                     hide_index=True,
#                 )


# ── TAB 3: EVALUATION ─────────────────────────────────────────────────────────
with tab_eval:
    st.markdown("## Evaluation Dashboard")

    # ── 1. SUMMARY METRICS ───────────────────────────────────────────────────
    if SUMMARY:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Invoices",
            int(SUMMARY.get("num_invoices", 0))
        )

        c2.metric(
            "Mean Accuracy",
            f"{SUMMARY.get('mean_instance_accuracy', 0):.2f}%"
        )

        c3.metric(
            "Mean Precision",
            f"{SUMMARY.get('mean_field_precision', 0) * 100:.2f}%"
        )

        c4.metric(
            "Mean F1-score",
            f"{SUMMARY.get('mean_field_f1', 0) * 100:.2f}%"
        )

        st.markdown("<hr style='margin:12px 0 18px 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)

        # ── 2. SUMMARY BAR CHART ─────────────────────────────────────────────
        st.markdown("### Biểu đồ tổng quan kết quả đánh giá")

        summary_chart = pd.DataFrame({
            "Chỉ số": [
                "Mean Accuracy",
                "Mean Precision",
                "Mean Recall",
                "Mean F1-score"
            ],
            "Giá trị (%)": [
                SUMMARY.get("mean_instance_accuracy", 0),
                SUMMARY.get("mean_field_precision", 0) * 100,
                SUMMARY.get("mean_field_recall", 0) * 100,
                SUMMARY.get("mean_field_f1", 0) * 100,
            ]
        })

        st.bar_chart(
            summary_chart.set_index("Chỉ số"),
            y="Giá trị (%)",
            color="#06b6d4",
            height=330
        )

    else:
        st.warning("Chưa có file evaluation_summary.json để hiển thị kết quả tổng quan.")

    st.markdown("<hr style='margin:14px 0 18px 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)

    # ── 3. AGENT DISTRIBUTION + FIELD METRICS TABLE ──────────────────────────
    col_agent, col_fm = st.columns([5, 5])

    with col_agent:
        st.markdown("### Phân bố Agent")

        if PREDICTIONS:
            agent_counts = Counter(
                item.get("agent_used", item.get("template_id", "unknown"))
                for item in PREDICTIONS
            )

            df_agent = pd.DataFrame(
                list(agent_counts.items()),
                columns=["Agent", "Số hóa đơn"]
            ).sort_values("Số hóa đơn", ascending=False)

            st.bar_chart(
                df_agent.set_index("Agent"),
                y="Số hóa đơn",
                color="#22c55e",
                height=320
            )

            st.dataframe(
                df_agent,
                use_container_width=True,
                hide_index=True,
                height=180
            )
        else:
            st.info("Chưa có dữ liệu prediction để thống kê Agent.")

    with col_fm:
        st.markdown("### Bảng Field Metrics")

        if not FIELD_METRICS.empty:
            df_show = FIELD_METRICS[["key", "precision", "recall", "f1_score", "accuracy"]].copy()

            for col in ["precision", "recall", "f1_score", "accuracy"]:
                if df_show[col].max() <= 1:
                    df_show[col] = df_show[col] * 100

            df_show[["precision", "recall", "f1_score", "accuracy"]] = (
                df_show[["precision", "recall", "f1_score", "accuracy"]].round(2)
            )

            df_show.columns = [
                "Field",
                "Precision (%)",
                "Recall (%)",
                "F1-score (%)",
                "Accuracy (%)"
            ]

            st.dataframe(
                df_show,
                use_container_width=True,
                hide_index=True,
                height=320
            )
        else:
            st.info("Chưa có file field_metrics.csv.")

    st.markdown("<hr style='margin:14px 0 18px 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)

    # ── 4. FIELD METRICS BAR CHART ───────────────────────────────────────────
    st.markdown("### Biểu đồ Precision / Recall / F1-score theo từng Field")

    if not FIELD_METRICS.empty:
        df_field_chart = FIELD_METRICS[["key", "precision", "recall", "f1_score"]].copy()

        for col in ["precision", "recall", "f1_score"]:
            if df_field_chart[col].max() <= 1:
                df_field_chart[col] = df_field_chart[col] * 100

        df_field_chart = df_field_chart.round(2)

        df_field_chart = df_field_chart.rename(columns={
            "key": "Field",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-score"
        })

        st.bar_chart(
            df_field_chart.set_index("Field"),
            height=430
        )
    else:
        st.info("Chưa có dữ liệu Field Metrics để vẽ biểu đồ.")

    st.markdown("<hr style='margin:14px 0 18px 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)

    # ── 5. ACCURACY DISTRIBUTION ─────────────────────────────────────────────
    st.markdown("### Biểu đồ Accuracy theo từng hóa đơn")

    if not INSTANCE_ACC.empty:
        df_acc = INSTANCE_ACC.copy()

        if "accuracy" in df_acc.columns:
            if df_acc["accuracy"].max() <= 1:
                df_acc["accuracy"] = df_acc["accuracy"] * 100

            acc_chart = pd.DataFrame({
                "Hóa đơn": df_acc["invoice_no"].astype(str),
                "Accuracy (%)": df_acc["accuracy"].round(2)
            })

            st.bar_chart(
                acc_chart.set_index("Hóa đơn"),
                y="Accuracy (%)",
                color="#6366f1",
                height=360
            )
        else:
            st.warning("File instance_accuracy.csv chưa có cột accuracy.")
    else:
        st.info("Chưa có file instance_accuracy.csv.")

    st.markdown("<hr style='margin:14px 0 18px 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)

    # ── 6. TOP 5 BEST / WORST ────────────────────────────────────────────────
    if not INSTANCE_ACC.empty and "accuracy" in INSTANCE_ACC.columns:
        st.markdown("### Top hóa đơn theo Accuracy")

        def render_top5(df_slice, label, color):
            st.markdown(f'<div class="slabel">{label}</div>', unsafe_allow_html=True)

            for _, row in df_slice.iterrows():
                idx = int(row["invoice_no"])
                acc = row["accuracy"]

                inv_item = PREDICTIONS[idx] if idx < len(PREDICTIONS) else {}
                image_path = normalize_path(inv_item.get("image_path", ""))
                image_name = os.path.basename(image_path) if image_path else f"invoice_{idx}"
                agent = inv_item.get("agent_used", inv_item.get("template_id", ""))

                c_img, c_info = st.columns([1, 3])

                with c_img:
                    if image_path and os.path.exists(image_path):
                        st.image(image_path, use_container_width=True)

                with c_info:
                    st.markdown(f"""
                    <div style="padding:6px 0">
                        <div style="font-weight:600;font-size:13px;color:#1e293b;">
                            #{idx} — {image_name}
                        </div>
                        <div style="font-size:12px;color:#64748b;">
                            Agent: {agent}
                        </div>
                        <div style="font-size:18px;font-weight:700;color:{color};">
                            {acc:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        col_best, col_worst = st.columns(2)

        with col_best:
            render_top5(
                INSTANCE_ACC.nlargest(5, "accuracy"),
                "Top 5 — Accuracy cao nhất",
                "#15803d"
            )

        with col_worst:
            render_top5(
                INSTANCE_ACC.nsmallest(5, "accuracy"),
                "Top 5 — Accuracy thấp nhất",
                "#be123c"
            )

    st.markdown("<hr style='margin:14px 0 18px 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)

    # ── 7. GROUND TRUTH VS PREDICTION ────────────────────────────────────────
    st.markdown("### Ground Truth vs Prediction — từng hóa đơn")

    for item in PREDICTIONS:
        i = item.get("image_index", 0)
        image_name = os.path.basename(item.get("image_path", f"test_{i}.png"))
        agent = item.get("agent_used", item.get("template_id", ""))

        pred = item.get("prediction", {})
        if isinstance(pred, str):
            try:
                pred = json.loads(pred)
            except Exception:
                pred = {}

        gt = item.get("ground_truth", {})

        acc_row = INSTANCE_ACC[INSTANCE_ACC["invoice_no"] == i] if not INSTANCE_ACC.empty else pd.DataFrame()
        acc_str = f"{acc_row['accuracy'].values[0]:.1f}%" if not acc_row.empty else "—"

        with st.expander(
            f"#{i}  —  {image_name}  |  Agent: {agent}  |  Accuracy: {acc_str}",
            expanded=False
        ):
            rows = []

            for full_key in KEYS_TO_EVALUATE:
                level, key = full_key.split(".")

                gt_vals = extract_values(gt, level, key)
                pred_vals = extract_values(pred, level, key)

                max_len = max(len(gt_vals), len(pred_vals), 1) if (gt_vals or pred_vals) else 0

                for j in range(max_len):
                    gt_v = gt_vals[j] if j < len(gt_vals) else ""
                    pred_v = pred_vals[j] if j < len(pred_vals) else ""

                    if gt_v == "":
                        match = "—"
                    elif normalize_value(gt_v) == normalize_value(pred_v):
                        match = "✓"
                    else:
                        match = "✗"

                    rows.append({
                        "Field": full_key,
                        "Ground Truth": gt_v,
                        "Prediction": pred_v,
                        "": match
                    })

            if rows:
                df = pd.DataFrame(rows)

                st.dataframe(
                    df.style.apply(
                        lambda col: [
                            "background-color:#f0fdf4; color:#15803d"
                            if v == "✓"
                            else "background-color:#fff1f2; color:#be123c"
                            if v == "✗"
                            else ""
                            for v in col
                        ] if col.name == "" else [""] * len(col),
                        axis=0
                    ),
                    use_container_width=True,
                    hide_index=True,
                )