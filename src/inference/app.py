import os
import time
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = FastAPI(title="MicroPEFT Engine API", version="1.0")

# Dynamic Device Allocation (Local CPU + Docker/K8s GPU Compatible)
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./outputs/exp_03_all_linear"

print(f"Loading base model on device: {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
)

if os.path.exists(ADAPTER_PATH):
    print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
else:
    print("Warning: LoRA adapter path not found, using base model.")
    model = base_model

model = model.to(DEVICE)
model.eval()

class GenerateRequest(BaseModel):
    prompt: str = Field(..., example="What are the symptoms of Diabetes?")
    domain: str = Field("medical", example="medical")
    max_tokens: int = Field(100, ge=1, le=512)

@app.get("/")
def read_root():
    return {"status": "healthy", "device": DEVICE, "adapter_loaded": os.path.exists(ADAPTER_PATH)}

@app.post("/v1/generate")
def generate_response(request: GenerateRequest):
    try:
        start_time = time.time()
        formatted_prompt = f"<|im_start|>system\nYou are a helpful {request.domain} expert.<|im_end|>\n<|im_start|>user\n{request.prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        latency = time.time() - start_time
        
        return {
            "response": generated_text,
            "domain_used": request.domain,
            "latency_seconds": round(latency, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))