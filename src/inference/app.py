import os, time, torch, uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI(title="MicroPEFT Serving Engine", version="2.0.0")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

DOMAIN_ADAPTERS = {
    "medical": "./adapters/exp_03_all_linear",
    "code": "./adapters/domain_code_adapter",
    "financial": "./adapters/domain_finance_adapter"
}

tokenizer = None
base_model = None
loaded_adapters = {}

class MultiDomainQueryRequest(BaseModel):
    prompt: str
    domain: str = "medical"
    max_tokens: int = 150
    temperature: float = 0.3

class MultiDomainQueryResponse(BaseModel):
    domain_used: str
    response: str
    tokens_generated: int
    latency_seconds: float
    tokens_per_second: float

@app.on_event("startup")
def load_base_engine():
    global tokenizer, base_model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")

@app.get("/health")
def health_check():
    return {"status": "healthy", "base_model": MODEL_ID, "supported_domains": list(DOMAIN_ADAPTERS.keys())}

@app.post("/v1/generate", response_model=MultiDomainQueryResponse)
def generate_response(payload: MultiDomainQueryRequest):
    domain = payload.domain.lower()
    adapter_path = DOMAIN_ADAPTERS.get(domain, DOMAIN_ADAPTERS["medical"])
    
    if os.path.exists(adapter_path):
        if domain not in loaded_adapters:
            loaded_adapters[domain] = PeftModel.from_pretrained(base_model, adapter_path)
            loaded_adapters[domain].eval()
        active_model = loaded_adapters[domain]
    else:
        active_model = base_model

    formatted_prompt = f"<|im_start|>user\n{payload.prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    start_time = time.time()
    with torch.no_grad():
        output_ids = active_model.generate(**inputs, max_new_tokens=payload.max_tokens, temperature=payload.temperature, pad_token_id=tokenizer.pad_token_id)
    latency = time.time() - start_time
    
    generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    num_tokens = len(generated_tokens)
    tps = num_tokens / latency if latency > 0 else 0
    
    return MultiDomainQueryResponse(
        domain_used=domain, response=text, tokens_generated=num_tokens,
        latency_seconds=round(latency, 3), tokens_per_second=round(tps, 2)
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)