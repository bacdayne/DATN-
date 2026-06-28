import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pipeline.agentic_prompts import classify_template_agent, get_prompt_by_agent

BASE_MODEL  = "Qwen/Qwen2-1.5B-Instruct"
ADAPTER_DIR = "finetuning/qwen2-cord-lora"
VAL_FILE    = "finetuning/data/val.jsonl"
N_SAMPLES   = 5


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    return tokenizer, model


def infer(tokenizer, model, ocr_text: str) -> str:
    messages = [{"role": "user", "content": ocr_text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    print("Loading model...")
    tokenizer, model = load_model()
    print("Model ready.\n")

    with open(VAL_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()][:N_SAMPLES]

    for i, s in enumerate(samples):
        print(f"{'='*60}")
        print(f"SAMPLE {i+1}")
        print(f"{'='*60}")

        # Extract OCR text từ prompt (phần sau "OCR:\n")
        raw_prompt = s["prompt"]
        ocr_text   = raw_prompt.split("OCR:\n", 1)[-1].rstrip("\nJSON:")

        # Chạy đúng agentic pipeline như run_extraction_finetuned.py
        agent_name   = classify_template_agent(ocr_text)
        agent_prompt = get_prompt_by_agent(agent_name=agent_name, ocr_text=ocr_text)

        print(f"AGENT: {agent_name}")

        pred = infer(tokenizer, model, agent_prompt)
        gt   = s["completion"]

        print("PREDICTED:")
        try:
            print(json.dumps(json.loads(pred), indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print(pred)

        print("\nGROUND TRUTH:")
        print(json.dumps(json.loads(gt), indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()
