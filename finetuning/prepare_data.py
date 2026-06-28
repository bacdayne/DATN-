import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.agentic_prompts import (
    classify_template_agent,
    get_prompt_by_agent,
)

TRAIN_OCR_FILE = "ocr/ocr_results_train.json"
OUTPUT_DIR = "finetuning/data"
VAL_RATIO = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)


def fix_to_schema(data):
    if not isinstance(data, dict):
        data = {}

    menu = data.get("menu", [])
    if isinstance(menu, dict):
        menu = [menu]
    if not isinstance(menu, list):
        menu = []

    fixed_menu = []
    for item in menu:
        if not isinstance(item, dict):
            continue
        fixed_menu.append({
            "nm":        str(item.get("nm", "")),
            "cnt":       str(item.get("cnt", "")),
            "unitprice": str(item.get("unitprice", "")),
            "price":     str(item.get("price", "")),
        })

    if not fixed_menu:
        fixed_menu = [{"nm": "", "cnt": "", "unitprice": "", "price": ""}]

    sub = data.get("sub_total", {})
    if not isinstance(sub, dict):
        sub = {}

    total = data.get("total", {})
    if not isinstance(total, dict):
        total = {}

    return {
        "menu": fixed_menu,
        "sub_total": {
            "subtotal_price": str(sub.get("subtotal_price", "")),
            "discount_price": str(sub.get("discount_price", "")),
            "tax_price":      str(sub.get("tax_price", "")),
            "service_price":  str(sub.get("service_price", "")),
        },
        "total": {
            "total_price":     str(total.get("total_price", "")),
            "cashprice":       str(total.get("cashprice", "")),
            "changeprice":     str(total.get("changeprice", "")),
            "creditcardprice": str(total.get("creditcardprice", "")),
        },
    }


def build_sample(item):
    ocr_text     = item.get("ocr_text", "")
    ground_truth = item.get("ground_truth", {})

    agent      = classify_template_agent(ocr_text)
    prompt     = get_prompt_by_agent(agent, ocr_text).strip()
    completion = json.dumps(fix_to_schema(ground_truth), ensure_ascii=False)

    return {
        "prompt":     prompt,
        "completion": completion,
        "agent":      agent,
    }


def save_jsonl(samples, path):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} samples → {path}")


def main():
    if not os.path.exists(TRAIN_OCR_FILE):
        print(f"ERROR: {TRAIN_OCR_FILE} not found.")
        print("Run first: python run_ocr.py --split train --limit 800")
        return

    with open(TRAIN_OCR_FILE, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    print(f"Loaded {len(train_data)} samples from {TRAIN_OCR_FILE}")

    samples = [build_sample(item) for item in train_data]

    n_val         = max(1, int(len(samples) * VAL_RATIO))
    val_samples   = samples[:n_val]
    train_samples = samples[n_val:]

    save_jsonl(train_samples, os.path.join(OUTPUT_DIR, "train.jsonl"))
    save_jsonl(val_samples,   os.path.join(OUTPUT_DIR, "val.jsonl"))

    agent_counts = Counter(s["agent"] for s in samples)
    print("Agent distribution:", dict(agent_counts))
    print("\nSample prompt (first 200 chars):")
    print(samples[0]["prompt"][:200])
    print("\nSample completion:")
    print(samples[0]["completion"][:200])


if __name__ == "__main__":
    main()
