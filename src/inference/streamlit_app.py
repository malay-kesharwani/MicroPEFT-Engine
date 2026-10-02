import streamlit as st, requests, time

st.set_page_config(page_title="MicroPEFT AI Suite", layout="wide")
st.title("⚡ MicroPEFT: Multi-Domain LLM Suite")

domain = st.sidebar.selectbox("Select Domain", ["medical", "code", "financial"])
temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)
max_tok = st.sidebar.slider("Max Tokens", 50, 512, 150)
api_url = st.sidebar.text_input("FastAPI Endpoint", "http://localhost:8000/v1/generate")

prompt = st.text_area("Prompt:", value="What are the clinical signs of Acute Pancreatitis?")

if st.button("Generate"):
    res = requests.post(api_url, json={"prompt": prompt, "domain": domain, "max_tokens": max_tok, "temperature": temp})
    if res.status_code == 200:
        data = res.json()
        st.write(data["response"])
        st.caption(f"Latency: {data['latency_seconds']}s | Speed: {data['tokens_per_second']} tok/s")