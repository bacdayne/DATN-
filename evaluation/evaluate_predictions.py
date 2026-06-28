import json
import ast
import re
from collections import Counter
import pandas as pd
import os


PREDICTION_FILE = "pipeline/predictions/predictions_test_finetuned.json"
OUTPUT_DIR = "evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# JSON PARSE
# =========================

def clean_json_text(text):
    if isinstance(text, dict):
        return text

    text = str(text).strip()
    text = text.replace("```json", "")
    text = text.replace("```", "")

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return {}

    text = text[start:end + 1]

    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return {}


# =========================
# NORMALIZE STRUCTURE
# =========================

def normalize_menu(data):
    if not isinstance(data, dict):
        return {}

    if "menu" in data and isinstance(data["menu"], dict):
        data["menu"] = [data["menu"]]

    if "menu" not in data:
        data["menu"] = []

    if "sub_total" not in data:
        data["sub_total"] = {}

    if "total" not in data:
        data["total"] = {}

    return data


# =========================
# NORMALIZE VALUE
# =========================

def normalize_value(v):
    if v is None:
        return ""

    if isinstance(v, list):
        v = " ".join(str(x) for x in v)

    v = str(v).strip().lower()

    # currency
    v = v.replace("rp.", "")
    v = v.replace("rp", "")
    v = v.replace("idr", "")

    # remove common spaces/symbols
    v = v.replace(" ", "")
    v = v.replace("@", "")
    v = v.replace("[", "")
    v = v.replace("]", "")
    v = v.replace("(", "")
    v = v.replace(")", "")

    # normalize number separators
    v = v.replace(",", "")
    v = v.replace(".", "")

    # normalize x quantity
    v = v.replace("xitems", "xitem")
    v = v.replace("items", "item")

    return v


# =========================
# EXTRACT VALUES
# =========================

def extract_values(data, level, key):
    values = []

    if not isinstance(data, dict):
        return values

    if level == "menu":
        menu = data.get("menu", [])

        if isinstance(menu, dict):
            menu = [menu]

        if not isinstance(menu, list):
            return values

        for item in menu:
            if isinstance(item, dict) and key in item:
                values.append(normalize_value(item[key]))

        return [v for v in values if v != ""]

    elif level == "sub":
        menu = data.get("menu", [])

        if isinstance(menu, dict):
            menu = [menu]

        if not isinstance(menu, list):
            return values

        for item in menu:
            if not isinstance(item, dict):
                continue

            sub = item.get("sub", [])

            if isinstance(sub, dict):
                sub = [sub]

            if not isinstance(sub, list):
                continue

            for s in sub:
                if isinstance(s, dict) and key in s:
                    values.append(normalize_value(s[key]))

        return [v for v in values if v != ""]

    else:
        section = data.get(level, {})

        if isinstance(section, dict):
            if key in section:
                values.append(normalize_value(section[key]))

        elif isinstance(section, list):
            for item in section:
                if isinstance(item, dict) and key in item:
                    values.append(normalize_value(item[key]))

        return [v for v in values if v != ""]


# =========================
# METRICS
# =========================

def compute_metrics(gt_values, pred_values):
    gt_counter = Counter(gt_values)
    pred_counter = Counter(pred_values)

    tp = 0

    for val in gt_counter:
        if val in pred_counter:
            tp += min(gt_counter[val], pred_counter[val])

    fp = sum(pred_counter.values()) - tp
    fn = sum(gt_counter.values()) - tp

    return tp, fp, fn


keys_to_evaluate = [
    "menu.nm",
    "menu.cnt",
    "menu.unitprice",
    "menu.price",

    "sub_total.subtotal_price",
    "sub_total.discount_price",
    "sub_total.tax_price",

    "total.total_price",
    "total.cashprice",
    "total.changeprice",
    "total.creditcardprice",
]


# =========================
# LOAD DATA
# =========================

with open(PREDICTION_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)
    results= results[:100]
    parse_ok = 0
    parse_fail = 0

for i, item in enumerate(results):
    pred = clean_json_text(item.get("prediction", ""))

    if pred == {}:
        parse_fail += 1
        print("PARSE FAIL:", i, item.get("prediction", "")[:150])
    else:
        parse_ok += 1

print("parse_ok:", parse_ok)
print("parse_fail:", parse_fail)


prediction_list = []
ground_truth_list = []

for item in results:
    pred = clean_json_text(item.get("prediction", {}))
    gt = item.get("ground_truth", {})

    pred = normalize_menu(pred)
    gt = normalize_menu(gt)

    prediction_list.append(pred)
    ground_truth_list.append(gt)


# =========================
# INSTANCE ACCURACY
# =========================

metrics_list = []

for i in range(len(prediction_list)):
    gt = ground_truth_list[i]
    pred = prediction_list[i]

    total_keys = 0
    matched_keys = 0

    for full_key in keys_to_evaluate:
        level, key = full_key.split(".")

        gt_values = extract_values(gt, level, key)
        pred_values = extract_values(pred, level, key)

        total_keys += len(gt_values)

        tp, fp, fn = compute_metrics(gt_values, pred_values)
        matched_keys += tp

    accuracy = matched_keys / total_keys * 100 if total_keys > 0 else 0

    metrics_list.append({
        "invoice_no": i,
        "total_keys": total_keys,
        "matched_keys": matched_keys,
        "accuracy": accuracy
    })


metrics_df = pd.DataFrame(metrics_list)

metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "instance_accuracy.csv"),
    index=False,
    encoding="utf-8-sig"
)


# =========================
# FIELD METRICS
# =========================

global_metrics = {
    key: {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "total_gt": 0
    }
    for key in keys_to_evaluate
}


for gt, pred in zip(ground_truth_list, prediction_list):
    for full_key in keys_to_evaluate:
        level, key = full_key.split(".")

        gt_values = extract_values(gt, level, key)
        pred_values = extract_values(pred, level, key)

        tp, fp, fn = compute_metrics(gt_values, pred_values)

        global_metrics[full_key]["tp"] += tp
        global_metrics[full_key]["fp"] += fp
        global_metrics[full_key]["fn"] += fn
        global_metrics[full_key]["total_gt"] += len(gt_values)


final_metrics = []

for key, value in global_metrics.items():
    tp = value["tp"]
    fp = value["fp"]
    fn = value["fn"]
    total_gt = value["total_gt"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    accuracy = tp / total_gt if total_gt > 0 else 0

    final_metrics.append({
        "key": key,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_gt": total_gt,
        "accuracy": accuracy
    })


final_metrics_df = pd.DataFrame(final_metrics)
final_metrics_df = final_metrics_df[final_metrics_df["total_gt"] > 0]

final_metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "field_metrics.csv"),
    index=False,
    encoding="utf-8-sig"
)


# =========================
# SUMMARY
# =========================

summary = {
    "num_invoices": len(results),
    "mean_instance_accuracy": float(metrics_df["accuracy"].mean()),
    "mean_field_precision": float(final_metrics_df["precision"].mean()),
    "mean_field_recall": float(final_metrics_df["recall"].mean()),
    "mean_field_f1": float(final_metrics_df["f1_score"].mean()),
}


with open(
    os.path.join(OUTPUT_DIR, "evaluation_summary.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)


print("===== INSTANCE ACCURACY =====")
print(metrics_df.round(2))

print("\nMean accuracy:", metrics_df["accuracy"].mean())

print("\n===== FIELD METRICS =====")
print(final_metrics_df.round(4))

print("\n===== SUMMARY =====")
print(json.dumps(summary, ensure_ascii=False, indent=2))

print("\nSaved:")
print("evaluation/instance_accuracy.csv")
print("evaluation/field_metrics.csv")
print("evaluation/evaluation_summary.json")