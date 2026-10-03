import streamlit as st
import requests
import time
import os

st.set_page_config(page_title="MicroPEFT Engine", page_icon="⚡", layout="wide")

st.title("⚡ MicroPEFT Inference Engine")
st.markdown("Multi-domain LoRA Adapter Switching Interface")

# Dynamic Backend URL for Docker Network
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000/v1/generate")

# Sidebar
st.sidebar.header("Configuration")
domain = st.sidebar.selectbox("Select Domain / LoRA Adapter", ["medical", "code", "finance"])
max_tokens = st.sidebar.slider("Max Tokens", min_value=10, max_value=512, value=100)

# Main Input
prompt = st.text_area("Enter your prompt:", value="What are the symptoms of Acute Pancreatitis?", height=120)

if st.button("Generate Response"):
    if not prompt.strip():
        st.warning("Please enter a valid prompt.")
    else:
        with st.spinner("Generating response..."):
            start_time = time.time()
            try:
                payload = {
                    "prompt": prompt,
                    "domain": domain,
                    "max_tokens": max_tokens
                }
                response = requests.post(BACKEND_URL, json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    res_text = data.get("response", "No response key found.")
                    latency = data.get("latency_seconds", round(time.time() - start_time, 2))
                    
                    st.success("Response Generated!")
                    st.write(res_text)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Latency", f"{latency} sec")
                    col2.metric("Domain", domain.capitalize())
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")