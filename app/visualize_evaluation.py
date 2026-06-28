import os
import json
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
PRED_FILE = os.path.join(
    BASE_DIR,
    "pipeline",
    "predictions",
    "predictions_test_finetuned.json"
)


OUT_DIR = os.path.join(BASE_DIR, "outputs", "charts")
os.makedirs(OUT_DIR, exist_ok=True)


def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")


def to_percent(value):
    return value * 100 if value <= 1 else value


def add_bar_labels(ax, fmt="{:.2f}%"):
    for bar in ax.patches:
        value = bar.get_height()
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 4),
            textcoords="offset points",
        )


def plot_summary_metrics():
    summary_path = os.path.join(EVAL_DIR, "evaluation_summary.json")
    check_file(summary_path)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    metrics = {
        "Instance Accuracy": to_percent(summary.get("mean_instance_accuracy", 0)),
        "Precision": to_percent(summary.get("mean_field_precision", 0)),
        "Recall": to_percent(summary.get("mean_field_recall", 0)),
        "F1-score": to_percent(summary.get("mean_field_f1", 0)),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(metrics.keys(), metrics.values())

    ax.set_title("Kết quả đánh giá tổng quan sau Fine-tuning", fontsize=14, fontweight="bold")
    ax.set_ylabel("Giá trị (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    add_bar_labels(ax)

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "01_summary_metrics.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", save_path)


def plot_field_metrics():
    csv_path = os.path.join(EVAL_DIR, "field_metrics.csv")
    check_file(csv_path)

    df = pd.read_csv(csv_path)

    required_cols = ["key", "precision", "recall", "f1_score"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Thiếu cột '{col}' trong field_metrics.csv")

    df["precision"] = df["precision"].apply(to_percent)
    df["recall"] = df["recall"].apply(to_percent)
    df["f1_score"] = df["f1_score"].apply(to_percent)

    df = df.sort_values("f1_score", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.5)))

    y = list(range(len(df)))
    height = 0.25

    ax.barh([i - height for i in y], df["precision"], height=height, label="Precision")
    ax.barh(y, df["recall"], height=height, label="Recall")
    ax.barh([i + height for i in y], df["f1_score"], height=height, label="F1-score")

    ax.set_yticks(y)
    ax.set_yticklabels(df["key"])
    ax.set_xlabel("Giá trị (%)")
    ax.set_title("Precision, Recall và F1-score theo từng trường thông tin", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "02_field_metrics.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", save_path)


def plot_accuracy_distribution():
    csv_path = os.path.join(EVAL_DIR, "instance_accuracy.csv")
    check_file(csv_path)

    df = pd.read_csv(csv_path)

    if "accuracy" not in df.columns:
        raise ValueError("Không tìm thấy cột 'accuracy' trong instance_accuracy.csv")

    acc = df["accuracy"].apply(to_percent)

    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

    groups = pd.cut(
        acc,
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    counts = groups.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(counts.index.astype(str), counts.values)

    ax.set_title("Phân bố độ chính xác theo từng hóa đơn", fontsize=14, fontweight="bold")
    ax.set_xlabel("Khoảng Accuracy")
    ax.set_ylabel("Số lượng hóa đơn")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for i, value in enumerate(counts.values):
        ax.text(i, value + 0.3, str(value), ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "03_accuracy_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", save_path)


def plot_agent_distribution():
    check_file(PRED_FILE)

    with open(PRED_FILE, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    agents = [
        item.get("agent_used") or item.get("template_id") or "unknown"
        for item in predictions
    ]

    counts = pd.Series(agents).value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(counts.index, counts.values)

    ax.set_title("Số lượng hóa đơn theo từng Agent", fontsize=14, fontweight="bold")
    ax.set_xlabel("Số lượng hóa đơn")
    ax.set_ylabel("Agent")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for i, value in enumerate(counts.values):
        ax.text(value + 0.3, i, str(value), va="center", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "04_agent_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", save_path)


def main():
    print("START VISUALIZATION")
    print("BASE_DIR:", BASE_DIR)
    print("EVAL_DIR:", EVAL_DIR)
    print("PRED_FILE:", PRED_FILE)
    print("OUT_DIR:", OUT_DIR)

    plot_summary_metrics()
    plot_field_metrics()
    plot_accuracy_distribution()
    plot_agent_distribution()

    print("DONE")
    print("Charts saved in:", OUT_DIR)


if __name__ == "__main__":
    main()