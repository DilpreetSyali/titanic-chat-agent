import os
import base64
import requests
import streamlit as st

st.set_page_config(page_title="Titanic Chat Agent", page_icon="🚢", layout="centered")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.title("🚢 Titanic Dataset Chat Agent")
st.caption("Ask questions in plain English. Get answers + charts.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Backend")
    st.write("Using:")
    st.code(BACKEND_URL)
    st.markdown("**Examples:**")
    st.markdown('- What percentage of passengers were male?')
    st.markdown('- Show me a histogram of passenger ages')
    st.markdown('- What was the average ticket fare?')
    st.markdown('- How many passengers embarked from each port?')

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("image_b64"):
            img_bytes = base64.b64decode(m["image_b64"])
            st.image(img_bytes)

prompt = st.chat_input("Ask something about the Titanic dataset...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            r = requests.post(f"{BACKEND_URL}/chat", json={"question": prompt}, timeout=60)
            r.raise_for_status()
            data = r.json()

            st.write(data["answer"])
            if data.get("image_b64"):
                img_bytes = base64.b64decode(data["image_b64"])
                st.image(img_bytes)

            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"],
                "image_b64": data.get("image_b64"),
            })

        except Exception as e:
            st.error(f"Backend error: {e}")
