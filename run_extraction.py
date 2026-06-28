from vllm import LLM, SamplingParams
import json
import os
import ast

from pipeline.agentic_prompts import (
    classify_template_agent,
    get_prompt_by_agent,
    BASE_SCHEMA
)


INPUT_OCR_FILE = "ocr/ocr_results_test.json"
OUTPUT_DIR = "pipeline/predictions"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "predictions_test.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_first_json(text):
    text = text.strip()
    text = text.replace("```json", "")
    text = text.replace("```", "")

    start = text.find("{")
    if start == -1:
        return ""

    brace_count = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if not in_string:
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1

                if brace_count == 0:
                    return text[start:i + 1]

    return ""


def try_parse_json(text):
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return None


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
            "nm": str(item.get("nm", "")),
            "cnt": str(item.get("cnt", "")),
            "unitprice": str(item.get("unitprice", "")),
            "price": str(item.get("price", ""))
        })

    if len(fixed_menu) == 0:
        fixed_menu = [
            {
                "nm": "",
                "cnt": "",
                "unitprice": "",
                "price": ""
            }
        ]

    sub_total = data.get("sub_total", {})
    if not isinstance(sub_total, dict):
        sub_total = {}

    total = data.get("total", {})
    if not isinstance(total, dict):
        total = {}

    fixed = {
        "menu": fixed_menu,
        "sub_total": {
            "subtotal_price": str(sub_total.get("subtotal_price", "")),
            "discount_price": str(sub_total.get("discount_price", "")),
            "tax_price": str(sub_total.get("tax_price", "")),
            "service_price": str(sub_total.get("service_price", ""))
        },
        "total": {
            "total_price": str(total.get("total_price", "")),
            "cashprice": str(total.get("cashprice", "")),
            "changeprice": str(total.get("changeprice", "")),
            "creditcardprice": str(total.get("creditcardprice", ""))
        }
    }

    return fixed


def generate_prediction(llm, sampling_params, prompt):
    outputs = llm.generate(
        [prompt],
        sampling_params
    )

    raw_prediction = outputs[0].outputs[0].text.strip()

    json_text = extract_first_json(raw_prediction)
    parsed = try_parse_json(json_text)

    if parsed is None:
        return "", raw_prediction, False

    fixed = fix_to_schema(parsed)
    prediction_text = json.dumps(fixed, ensure_ascii=False)

    return prediction_text, raw_prediction, True


def build_retry_prompt(ocr_text):
    return f"""
Convert OCR receipt text to JSON.

JSON format:
{BASE_SCHEMA}

Mapping:
menu name -> menu.nm
quantity -> menu.cnt
unit price -> menu.unitprice
item price -> menu.price
subtotal -> sub_total.subtotal_price
discount -> sub_total.discount_price
tax / vat / pb1 -> sub_total.tax_price
service -> sub_total.service_price
total / grand total -> total.total_price
cash / tunai -> total.cashprice
change / kembalian -> total.changeprice
card / debit / visa / edc -> total.creditcardprice

OCR:
{ocr_text}

JSON:
"""


def main():
    with open(INPUT_OCR_FILE, "r", encoding="utf-8") as f:
        ocr_results = json.load(f)

    llm = LLM(
        model="Qwen/Qwen2-1.5B-Instruct",
        dtype="float16",
        max_model_len=1024,
        max_num_seqs=1,
        gpu_memory_utilization=0.35,
        swap_space=0,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=600,
        repetition_penalty=1.03,
        stop=[
            "```",
            "\nOCR:",
            "\nOCR text:",
            "\nExplanation:",
            "\nNote:",
            "\nPlease note",
            "\nPython",
            "\nimport ",
            "\nfrom ",
            "\ndef ",
            "\nclass "
        ]
    )

    predictions = []
    parse_ok = 0
    parse_fail = 0

    for item in ocr_results:
        ocr_text = item["ocr_text"]

        agent_used = classify_template_agent(ocr_text)

        prompt = get_prompt_by_agent(
            agent_name=agent_used,
            ocr_text=ocr_text
        )

        prediction_text, raw_prediction, ok = generate_prediction(
            llm,
            sampling_params,
            prompt
        )

        if not ok:
            retry_prompt = build_retry_prompt(ocr_text)

            prediction_text, raw_prediction, ok = generate_prediction(
                llm,
                sampling_params,
                retry_prompt
            )

        if ok:
            parse_ok += 1
        else:
            parse_fail += 1

        result = {
            "image_index": item.get("image_index", ""),
            "image_path": item.get("image_path", ""),
            "ocr_text": ocr_text,

            "template_id": agent_used,
            "agent_used": agent_used,

            "prediction": prediction_text,
            "raw_prediction": raw_prediction,
            "parse_ok": ok,
            "ground_truth": item.get("ground_truth", {})
        }

        predictions.append(result)

        print("=" * 60)
        print("IMAGE:", item.get("image_path", ""))
        print("AGENT:", agent_used)
        print("PARSE_OK:", ok)
        print("=" * 60)
        print(prediction_text)
        print()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print("Saved:", OUTPUT_FILE)
    print("parse_ok:", parse_ok)
    print("parse_fail:", parse_fail)


if __name__ == "__main__":
    main()