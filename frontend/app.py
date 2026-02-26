import os
import base64
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ------------------ Config ------------------
st.set_page_config(
    page_title="Titanic Dataset Chat Agent",
    page_icon="🚢",
    layout="wide",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

# ------------------ Styling ------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      .title-wrap { display:flex; align-items:center; gap: 12px; }
      .badge { font-size: 12px; padding: 4px 10px; border-radius: 999px;
               background: rgba(99,102,241,.12); border: 1px solid rgba(99,102,241,.25); }
      .subtle { color: rgba(255,255,255,.65); }

      /* Chat bubbles */
      .chat-bubble { padding: 12px 14px; border-radius: 14px; margin: 8px 0; max-width: 900px; }
      .user { background: rgba(59,130,246,.15); border: 1px solid rgba(59,130,246,.25); margin-left: auto; }
      .bot  { background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.25); }

      /* Small label */
      .who { font-size: 12px; opacity: .75; margin-bottom: 6px; }

      /* Cards */
      .card { padding: 14px 14px; border-radius: 16px; border: 1px solid rgba(255,255,255,.10);
              background: rgba(255,255,255,.04); }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }

      /* Buttons */
      .stButton button { border-radius: 12px; padding: 0.6rem 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ Helpers ------------------
def api_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def ask_backend(question: str):
    r = requests.post(
        f"{BACKEND_URL}/chat",
        json={"question": question},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()

def render_bubble(role: str, text: str):
    css = "user" if role == "user" else "bot"
    who = "You" if role == "user" else "TitanicBot"
    st.markdown(
        f"""
        <div class="chat-bubble {css}">
            <div class="who">{who}</div>
            <div>{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_image_b64(img_b64: str):
    if not img_b64:
        return
    try:
        data = base64.b64decode(img_b64)
        st.image(data, use_column_width=True)
    except Exception:
        st.warning("Could not render the chart image.")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown(f"**Backend URL:** `{BACKEND_URL}`")
    ok = api_health()
    st.markdown(f"**Status:** {'🟢 Connected' if ok else '🔴 Not connected'}")

    st.markdown("---")
    st.markdown("### 💡 Try these")
    examples = [
        "What percentage of passengers were male?",
        "Show me a histogram of passenger ages",
        "What was the average ticket fare?",
        "How many passengers embarked from each port?",
        "Show survival count by passenger class (Pclass)",
        "Plot fare distribution for survivors vs non-survivors",
    ]

    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["pending_question"] = ex

    st.markdown("---")
    st.markdown("### 🧹 Chat")
    if st.button("Clear chat", use_container_width=True):
        st.session_state["messages"] = []

# ------------------ Header ------------------
col1, col2 = st.columns([0.75, 0.25], vertical_alignment="center")
with col1:
    st.markdown(
        """
        <div class="title-wrap">
          <h1 style="margin:0;">🚢 Titanic Dataset Chat Agent</h1>
          <span class="badge">Answers + Charts</span>
        </div>
        <div class="subtle">Ask about Titanic passengers in plain English. Get accurate stats and visual insights.</div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Quick Tips**")
    st.markdown("- Ask for plots: *histogram*, *bar chart*, *distribution*")
    st.markdown("- Ask for stats: *average fare*, *survival rate*, *counts*")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")

# ------------------ Session State ------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "pending_question" not in st.session_state:
    st.session_state["pending_question"] = None

# ------------------ Chat Area ------------------
chat_container = st.container()

with chat_container:
    if len(st.session_state["messages"]) == 0:
        st.markdown(
            """
            <div class="card">
              <div style="font-size: 18px; font-weight: 600;">Start with a question 👇</div>
              <div class="subtle">Example: “Show me a histogram of passenger ages”</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for msg in st.session_state["messages"]:
        render_bubble(msg["role"], msg["content"])
        if msg.get("image_b64"):
            render_image_b64(msg["image_b64"])

st.markdown("")

# ------------------ Input Row ------------------
q_default = st.session_state["pending_question"] or ""
question = st.chat_input("Type your Titanic question…")
if q_default and question is None:
    # If user clicked sidebar example, auto-send on next rerun
    question = q_default
    st.session_state["pending_question"] = None

if question:
    # User message
    st.session_state["messages"].append({"role": "user", "content": question})

    # Bot response
    with st.spinner("Thinking…"):
        try:
            data = ask_backend(question)
            answer = data.get("answer", "")
            img_b64 = data.get("image_b64")

            st.session_state["messages"].append(
                {"role": "assistant", "content": answer, "image_b64": img_b64}
            )
        except requests.exceptions.RequestException as e:
            st.session_state["messages"].append(
                {"role": "assistant", "content": f"Backend error: {e}", "image_b64": None}
            )

    st.rerun()