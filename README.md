# ⚡ MicroPEFT Engine: Multi-Domain LoRA Adapter Inference Stack

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker Support](https://img.shields.io/badge/Docker-Multi--Container-blue?logo=docker)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red?logo=streamlit)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org/)

**MicroPEFT Engine** is a lightweight, production-grade microservice architecture designed for dynamic Low-Rank Adaptation (LoRA) adapter switching across fine-tuned domain-specific LLMs (Medical, Code, Finance). Built with **FastAPI** for dynamic tensor serving and **Streamlit** for interactive evaluation, containerized via **Docker** & **Kubernetes**.

---

## 🏗️ System Architecture

+-------------------------------------------------------+
|                Streamlit Frontend UI                  |
|                     (Port 8501)                       |
+---------------------------+---------------------------+
                            |
                        REST API
                            |
                            v
+-------------------------------------------------------+
|               FastAPI Inference Engine                |
|                     (Port 8000)                       |
+---------------------------+---------------------------+
                            |
               Dynamic LoRA Adapter Switch
                            |
      +---------------------+---------------------+
      |                     |                     |
      v                     v                     v
[ Medical Adapter ]   [ Code Adapter ]    [ Finance Adapter ]

---

## ✨ Key Features

- **Dynamic Adapter Switching:** Hot-swap LoRA adapters at inference time across **Medical**, **Code**, and **Finance** domains without reloading the base LLM (`Qwen2.5-0.5B-Instruct`).
- **Microservice Orchestration:** Fully containerized multi-service stack using `docker-compose` and production-ready `k8s-deployment.yaml`.
- **Parameter-Efficient Fine-Tuning (PEFT):** Custom LoRA implementation targeting linear projection layers with minimal trainable parameters (~0.1% - 0.2%).
- **Interactive UI & API Engine:** Real-time human evaluation dashboard powered by Streamlit and OpenAPI/Swagger documented FastAPI backend.

---

## 🧪 Model Evaluation & Hyperparameter Experiments

Fine-tuning experiments conducted on T4 GPU instance using `medalpaca/medical_meadow_medical_flashcards` dataset.

| Experiment ID | LoRA Rank (r) | LoRA Alpha | Target Modules | Trainable Ratio | Final Loss | Peak VRAM |
|---|---|---|---|---|---|---|
| `exp_01_baseline` | 8 | 16.0 | `q_proj`, `v_proj` | 0.1093% | **2.7840** | 6.79 GB |
| `exp_02_high_rank` | 16 | 32.0 | `q_proj`, `v_proj` | 0.2184% | **2.4820** | 7.70 GB |
| `exp_03_all_linear` *(Winning)* | 8 | 16.0 | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 0.2184% | **1.7120** | 7.84 GB |

---

## 🛠️ Project Structure

MicroPEFT-Engine/
├── configs/                 # Hyperparameter & training configuration files
│   └── train_config.json
├── notebooks/               # Verified Colab training & benchmark notebook
│   └── micropeftengine.ipynb
├── src/
│   ├── data/                # Dataset tokenization & dataloaders
│   │   └── dataset.py
│   ├── lora/                # Custom LoRA layer wrappers & injection logic
│   │   ├── lora_linear.py
│   │   └── injection.py
│   ├── training/            # Custom PEFT training loop & loss trackers
│   │   └── trainer.py
│   └── inference/           # FastAPI backend & Streamlit frontend
│       ├── app.py           # Engine backend server
│       └── streamlit_app.py # Dashboard frontend
├── tests/                   # Pytest test suite for layers & trainers
│   ├── test_dataset.py
│   ├── test_lora.py
│   ├── test_injection.py
│   └── test_trainer.py
├── docker-compose.yml       # Multi-container service stack
├── Dockerfile               # Backend container instructions
├── k8s-deployment.yaml      # Kubernetes manifest deployment specs
├── domain_registry.json     # Adapter metadata & system prompts
├── requirements.txt         # Core dependencies
├── LICENSE                  # MIT License
└── README.md

---

## 🚀 Quickstart Guide

### Prerequisites

- [Docker Engine](https://docs.docker.com/get-docker/) installed.
- [Docker Compose](https://docs.docker.com/compose/install/) CLI plugin.

### Running with Docker Compose

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/malay-kesharwani/MicroPEFT-Engine.git](https://github.com/malay-kesharwani/MicroPEFT-Engine.git)
   cd MicroPEFT-Engine

1. Spin Up the Microservice Stack:
docker compose up --build
2. Access the Web Services:
Streamlit Frontend: http://localhost:8501
FastAPI OpenAPI Specs: http://localhost:8000/docs

  Local Python Setup
1. Create Virtual Environment & Install Dependencies:
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
2. Run Inference API:
python -m uvicorn src.inference.app:app --host 0.0.0.0 --port 8000 --reload
3. Run Streamlit App:
streamlit run src/inference/streamlit_app.py
📜 License
Distributed under the MIT License. See LICENSE for details.   