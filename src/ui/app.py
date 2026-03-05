import streamlit as st
import torch
import numpy as np
import json
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = "models/bert"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CATEGORIES = {
    0: {
        "label": "Access",
        "icon": "🔐",
        "color": "#00D4FF",
        "desc": "Login, VPN, permissions",
    },
    1: {
        "label": "Administrative Rights",
        "icon": "👑",
        "color": "#FF6B35",
        "desc": "Elevated privileges, sudo",
    },
    2: {
        "label": "HR Support",
        "icon": "🧑‍💼",
        "color": "#A78BFA",
        "desc": "Onboarding, offboarding",
    },
    3: {
        "label": "Hardware",
        "icon": "🖥️",
        "color": "#34D399",
        "desc": "Devices, peripherals",
    },
    4: {
        "label": "Internal Project",
        "icon": "📁",
        "color": "#FCD34D",
        "desc": "Project tools, collab",
    },
    5: {
        "label": "Miscellaneous",
        "icon": "🔧",
        "color": "#94A3B8",
        "desc": "General / uncategorized",
    },
    6: {
        "label": "Purchase",
        "icon": "💳",
        "color": "#F472B6",
        "desc": "Licenses, procurement",
    },
    7: {
        "label": "Storage",
        "icon": "🗄️",
        "color": "#6EE7B7",
        "desc": "Cloud, drives, capacity",
    },
}

SAMPLE_TICKETS = [
    "I can't log into the VPN after last night's password reset.",
    "Please grant admin rights to sarah.jones@company.com for the dev environment.",
    "We need to order 3 new MacBook Pros for the design team.",
    "My external monitor stopped working after the Windows update.",
    "Need 2TB of extra cloud storage for the Q4 reporting project.",
    "New employee starting Monday — please set up their accounts.",
    "Adobe Creative Suite license expired for the marketing team.",
]

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Ticket Routing · AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background-color: #060911;
    color: #e2e8f0;
}

/* Main background */
.stApp { background: #060911; }
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1e293b; }

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Text area */
textarea {
    background: #0d1117 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 14px !important;
}
textarea:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 3px #3b82f620 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 12px 0 !important;
    width: 100% !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 24px #3b82f630 !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 32px #3b82f640 !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* Metrics */
[data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px !important;
}
[data-testid="metric-container"] label { color: #475569 !important; font-size: 11px !important; letter-spacing: 2px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 800 !important; }

/* Expander */
.streamlit-expanderHeader {
    background: #0d1117 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #0d1117 !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* Progress bar override */
.stProgress > div > div { background: #1e293b !important; border-radius: 4px !important; }
.stProgress > div > div > div { border-radius: 4px !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #0d1117; border-bottom: 1px solid #1e293b; gap: 8px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #475569; border-radius: 6px 6px 0 0; font-family: 'IBM Plex Mono', monospace; font-size: 12px; letter-spacing: 1px; }
.stTabs [aria-selected="true"] { background: #1e293b !important; color: #e2e8f0 !important; }

/* Divider */
hr { border-color: #1e293b !important; }

/* Info / warning / error boxes */
.stAlert { border-radius: 10px !important; font-family: 'IBM Plex Mono', monospace !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict(text, tokenizer, model):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()
    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    return predicted_class, confidence, probs


def routing_decision(confidence):
    if confidence > 0.85:
        return "AUTO-ROUTE", "⚡", "success"
    elif confidence > 0.60:
        return "MANUAL REVIEW", "👁", "warning"
    else:
        return "FALLBACK QUEUE", "⚠️", "error"


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "stats" not in st.session_state:
    st.session_state.stats = {"total": 0, "auto": 0, "manual": 0, "fallback": 0}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 TICKET ROUTER")
    st.markdown(
        "<span style='color:#475569;font-size:11px;letter-spacing:2px'>AI-POWERED · REAL-TIME</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        "<span style='color:#475569;font-size:10px;letter-spacing:2px'>DEVICE</span>",
        unsafe_allow_html=True,
    )
    device_color = "#22c55e" if str(DEVICE) == "cuda" else "#f59e0b"
    st.markdown(
        f"<span style='color:{device_color};font-weight:700'>● {str(DEVICE).upper()}</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        "<span style='color:#475569;font-size:10px;letter-spacing:2px'>CATEGORY GUIDE</span>",
        unsafe_allow_html=True,
    )
    for cid, cat in CATEGORIES.items():
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid #0f172a'>"
            f"<span style='width:3px;height:24px;background:{cat['color']};border-radius:2px;display:inline-block;flex-shrink:0'></span>"
            f"<div><div style='color:#cbd5e1;font-size:12px;font-weight:600'>{cat['icon']} {cat['label']}</div>"
            f"<div style='color:#475569;font-size:10px'>{cat['desc']}</div></div></div>",
            unsafe_allow_html=True,
        )
    st.divider()

    # Live stats
    s = st.session_state.stats
    st.markdown(
        "<span style='color:#475569;font-size:10px;letter-spacing:2px'>SESSION STATS</span>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.metric("Total", s["total"])
    c2.metric("Auto-Routed", s["auto"])
    c3, c4 = st.columns(2)
    c3.metric("Manual", s["manual"])
    c4.metric("Fallback", s["fallback"])

    if st.button("🗑 Clear History"):
        st.session_state.history = []
        st.session_state.stats = {"total": 0, "auto": 0, "manual": 0, "fallback": 0}
        st.rerun()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    """
<div style='padding:32px 0 20px'>
  <h1 style='font-family:Space Grotesk,sans-serif;font-size:28px;font-weight:700;color:#f8fafc;margin:0;letter-spacing:-0.5px'>
    INTELLIGENT TICKET ROUTING SYSTEM
  </h1>
  <p style='color:#475569;font-size:12px;letter-spacing:2px;margin:6px 0 0'>
    DISTILBERT NLP · ENTERPRISE HELPDESK · 8 CATEGORIES
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["⬡  CLASSIFY", "📋  HISTORY"])

with tab1:
    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown(
            "<span style='color:#475569;font-size:10px;letter-spacing:2px'>QUICK SAMPLES</span>",
            unsafe_allow_html=True,
        )
        selected_sample = st.selectbox(
            "",
            ["— select a sample ticket —"] + SAMPLE_TICKETS,
            label_visibility="collapsed",
        )

        st.markdown(
            "<span style='color:#475569;font-size:10px;letter-spacing:2px;margin-top:16px;display:block'>TICKET DESCRIPTION</span>",
            unsafe_allow_html=True,
        )
        default_text = (
            selected_sample if selected_sample != "— select a sample ticket —" else ""
        )
        ticket_input = st.text_area(
            "",
            value=default_text,
            height=150,
            placeholder="Describe the support ticket in detail…\n\ne.g. 'I cannot access the shared drive since the network migration.'",
            label_visibility="collapsed",
        )
        char_count = len(ticket_input)
        st.markdown(
            f"<div style='text-align:right;color:#334155;font-size:11px;margin-top:-8px'>{char_count} chars</div>",
            unsafe_allow_html=True,
        )

        classify_btn = st.button("⬡  CLASSIFY TICKET", use_container_width=True)

    with col_result:
        if classify_btn:
            if not ticket_input.strip():
                st.error("Please enter a ticket description.")
            else:
                with st.spinner(""):
                    try:
                        tokenizer, model = load_model()
                    except Exception:
                        st.error(
                            "⚠️ Model not found at `models/bert`. Ensure the model directory exists."
                        )
                        st.stop()

                    pred_class, confidence, all_probs = predict(
                        ticket_input, tokenizer, model
                    )
                    cat = CATEGORIES[pred_class]
                    routing, routing_icon, routing_type = routing_decision(confidence)

                    # Update stats
                    st.session_state.stats["total"] += 1
                    if routing_type == "success":
                        st.session_state.stats["auto"] += 1
                    elif routing_type == "warning":
                        st.session_state.stats["manual"] += 1
                    else:
                        st.session_state.stats["fallback"] += 1

                    # Save history
                    st.session_state.history.insert(
                        0,
                        {
                            "ticket": ticket_input[:80]
                            + ("…" if len(ticket_input) > 80 else ""),
                            "label": cat["label"],
                            "icon": cat["icon"],
                            "color": cat["color"],
                            "confidence": confidence,
                            "routing": routing,
                            "routing_type": routing_type,
                            "ts": datetime.now().strftime("%H:%M:%S"),
                        },
                    )

                    # ── RESULT CARD ──
                    st.markdown(
                        f"""<div style='background:#0d1117;border:2px solid {cat["color"]};border-radius:14px;padding:24px;
                        box-shadow:0 0 40px {cat["color"]}18;margin-bottom:16px'>
                        <div style='display:flex;justify-content:space-between;align-items:center'>
                          <div>
                            <div style='color:#64748b;font-size:10px;letter-spacing:2px;margin-bottom:8px'>CLASSIFICATION</div>
                            <div style='font-size:28px'>{cat["icon"]}</div>
                            <div style='color:{cat["color"]};font-size:22px;font-weight:800;
                            font-family:Space Grotesk,sans-serif;margin-top:4px'>{cat["label"]}</div>
                          </div>
                          <div style='text-align:right'>
                            <div style='color:#64748b;font-size:10px;letter-spacing:2px;margin-bottom:4px'>CONFIDENCE</div>
                            <div style='color:{cat["color"]};font-size:40px;font-weight:800;line-height:1'>
                              {confidence:.1%}
                            </div>
                          </div>
                        </div></div>""",
                        unsafe_allow_html=True,
                    )

                    # Routing badge
                    badge_styles = {
                        "success": ("#052e1a", "#22c55e"),
                        "warning": ("#2d1f07", "#f59e0b"),
                        "error": ("#2d0f0f", "#ef4444"),
                    }
                    bg, tc = badge_styles[routing_type]
                    routing_desc = {
                        "AUTO-ROUTE": "High confidence — dispatching automatically.",
                        "MANUAL REVIEW": "Medium confidence — agent review recommended.",
                        "FALLBACK QUEUE": "Low confidence — placed in general triage queue.",
                    }
                    st.markdown(
                        f"""<div style='background:{bg};border:1px solid {tc};border-radius:10px;
                        padding:14px 18px;margin-bottom:16px;display:flex;align-items:center;gap:12px'>
                        <span style='font-size:22px'>{routing_icon}</span>
                        <div>
                          <div style='color:{tc};font-weight:800;font-size:14px;letter-spacing:1px'>{routing}</div>
                          <div style='color:#64748b;font-size:11px;margin-top:3px'>{routing_desc[routing]}</div>
                        </div></div>""",
                        unsafe_allow_html=True,
                    )

                    # All category scores
                    with st.expander("ALL CATEGORY SCORES", expanded=True):
                        sorted_cats = sorted(
                            enumerate(all_probs), key=lambda x: x[1], reverse=True
                        )
                        for idx, prob in sorted_cats:
                            c = CATEGORIES[idx]
                            is_top = idx == pred_class
                            label_html = f"<span style='color:{'white' if is_top else '#475569'};font-weight:{'700' if is_top else '400'};font-size:12px'>{c['icon']} {c['label']}</span>"
                            pct_html = f"<span style='color:{c['color'] if is_top else '#334155'};font-size:12px;font-family:monospace'>{prob:.1%}</span>"
                            cols = st.columns([3, 1])
                            cols[0].markdown(label_html, unsafe_allow_html=True)
                            cols[1].markdown(pct_html, unsafe_allow_html=True)
                            bar_color = c["color"] if is_top else "#1e293b"
                            st.markdown(
                                f"""<div style='background:#1a1f2e;border-radius:4px;height:6px;margin:-8px 0 8px;overflow:hidden'>
                                <div style='height:100%;width:{prob*100:.1f}%;background:{bar_color};
                                border-radius:4px;box-shadow:0 0 8px {bar_color}88;
                                transition:width 0.8s cubic-bezier(.16,1,.3,1)'></div></div>""",
                                unsafe_allow_html=True,
                            )

        else:
            st.markdown(
                """
            <div style='background:#0d1117;border:1px dashed #1e293b;border-radius:14px;
            padding:48px 24px;text-align:center;margin-top:8px'>
              <div style='font-size:40px;margin-bottom:12px'>⬡</div>
              <div style='color:#334155;font-size:13px'>Classification results will appear here</div>
              <div style='color:#1e293b;font-size:11px;margin-top:8px;letter-spacing:1px'>ENTER A TICKET AND CLICK CLASSIFY</div>
            </div>""",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────
# HISTORY TAB
# ─────────────────────────────────────────────
with tab2:
    if not st.session_state.history:
        st.markdown(
            """<div style='text-align:center;padding:60px 0;color:#334155;font-size:13px'>
        No classifications yet. Go to the Classify tab to get started.</div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<span style='color:#475569;font-size:10px;letter-spacing:2px'>{len(st.session_state.history)} TICKETS CLASSIFIED THIS SESSION</span>",
            unsafe_allow_html=True,
        )
        st.divider()
        for item in st.session_state.history:
            badge_colors = {
                "success": "#22c55e",
                "warning": "#f59e0b",
                "error": "#ef4444",
            }
            tc = badge_colors[item["routing_type"]]
            st.markdown(
                f"""<div style='background:#0d1117;border:1px solid {item["color"]}33;
                border-radius:12px;padding:16px 20px;margin-bottom:10px'>
                <div style='display:flex;justify-content:space-between;align-items:flex-start;gap:12px'>
                  <div style='flex:1;min-width:0'>
                    <div style='color:#64748b;font-size:12px;margin-bottom:8px;
                    overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{item["ticket"]}</div>
                    <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap'>
                      <span style='font-size:16px'>{item["icon"]}</span>
                      <span style='color:{item["color"]};font-weight:700;font-size:12px'>{item["label"]}</span>
                      <span style='background:#0a0f1a;border:1px solid {tc};color:{tc};
                      font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px;
                      letter-spacing:1px'>{item["routing"]}</span>
                      <span style='color:#334155;font-size:10px'>{item["ts"]}</span>
                    </div>
                  </div>
                  <div style='text-align:right;flex-shrink:0'>
                    <div style='color:{item["color"]};font-size:24px;font-weight:800;
                    font-family:monospace;line-height:1'>{item["confidence"]:.1%}</div>
                    <div style='color:#334155;font-size:10px;margin-top:2px'>confidence</div>
                  </div>
                </div></div>""",
                unsafe_allow_html=True,
            )
