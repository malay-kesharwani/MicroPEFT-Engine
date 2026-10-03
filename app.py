import subprocess
import time
import os

# 1. Background me FastAPI Backend launch karo
backend_process = subprocess.Popen([
    "python", "-m", "uvicorn", "src.inference.app:app", "--host", "127.0.0.1", "--port", "8000"
])

# Server startup ke liye 3 sec wait
time.sleep(3)

# 2. Streamlit Frontend launch karo (Port 7860 Hugging Face ka default Port hai)
os.system("streamlit run src/inference/streamlit_app.py --server.port 7860 --server.address 0.0.0.0")