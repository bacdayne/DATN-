import os
import json
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


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


# ---------------------------------------------------------------------------
# Bảng màu & phong cách dùng chung cho toàn bộ biểu đồ
# ---------------------------------------------------------------------------
# Bảng màu thân thiện với người mù màu (Okabe-Ito), dùng nhất quán ở mọi biểu đồ
COLOR_PRECISION = "#0072B2"   # xanh dương
COLOR_RECALL    = "#E69F00"   # cam
COLOR_F1        = "#009E73"   # xanh lá
COLOR_ACCURACY  = "#2A4D69"   # xanh slate đậm (nhấn mạnh)
COLOR_PRIMARY   = "#2A6F97"   # màu chủ đạo cho biểu đồ đơn chuỗi
COLOR_HIGHLIGHT = "#E63946"   # đỏ nhấn cho giá trị đặc biệt

# Dải màu tuần tự (thấp -> cao) cho phân bố accuracy
SEQ_BLUES = ["#CBDCEC", "#9DBFDD", "#6C9FCB", "#3E7CBE", "#1F5A96"]

GRID_COLOR = "#D6D6D6"
TEXT_COLOR = "#222222"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "text.color": TEXT_COLOR,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.titlecolor": TEXT_COLOR,
    "axes.titlepad": 16,
    "axes.labelsize": 13,
    "axes.labelcolor": TEXT_COLOR,
    "axes.edgecolor": "#8A8A8A",
    "axes.linewidth": 0.8,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")


def to_percent(value):
    return value * 100 if value <= 1 else value


def style_axes(ax, grid_axis="y"):
    """Bỏ viền thừa, thêm lưới nhẹ để biểu đồ thoáng và dễ đọc."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8A8A8A")
    ax.spines["bottom"].set_color("#8A8A8A")
    if grid_axis:
        ax.grid(axis=grid_axis, linestyle="--", linewidth=0.7,
                color=GRID_COLOR, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    save_path = os.path.join(OUT_DIR, name)
    fig.savefig(save_path)
    plt.close(fig)
    print("Saved:", save_path)


# ---------------------------------------------------------------------------
# 01 - Tổng quan Precision / Recall / F1 / Instance Accuracy
# ---------------------------------------------------------------------------
def plot_summary_metrics():
    summary_path = os.path.join(EVAL_DIR, "evaluation_summary.json")
    check_file(summary_path)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    labels = ["Instance\nAccuracy", "Precision", "Recall", "F1-score"]
    values = [
        to_percent(summary.get("mean_instance_accuracy", 0)),
        to_percent(summary.get("mean_field_precision", 0)),
        to_percent(summary.get("mean_field_recall", 0)),
        to_percent(summary.get("mean_field_f1", 0)),
    ]
    colors = [COLOR_ACCURACY, COLOR_PRECISION, COLOR_RECALL, COLOR_F1]

    fig, ax = plt.subplots(figsize=(10, 6.2))
    bars = ax.bar(labels, values, color=colors, width=0.62,
                  edgecolor="white", linewidth=1.2, zorder=3)

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.2f}%",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center", va="bottom",
            fontsize=13, fontweight="bold",
            xytext=(0, 5), textcoords="offset points",
        )

    ax.set_title("Kết quả đánh giá tổng quan sau Fine-tuning")
    ax.set_ylabel("Giá trị (%)")
    ax.set_ylim(0, 100)
    style_axes(ax, grid_axis="y")

    save(fig, "01_summary_metrics.png")


# ---------------------------------------------------------------------------
# 02 - Precision / Recall / F1 theo từng trường
# ---------------------------------------------------------------------------
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

    fig, ax = plt.subplots(figsize=(12.5, max(6.5, len(df) * 0.62)))

    y = list(range(len(df)))
    height = 0.26

    series = [
        (df["precision"], COLOR_PRECISION, "Precision", -height),
        (df["recall"], COLOR_RECALL, "Recall", 0),
        (df["f1_score"], COLOR_F1, "F1-score", height),
    ]

    for data, color, label, offset in series:
        positions = [i + offset for i in y]
        ax.barh(positions, data, height=height, label=label,
                color=color, edgecolor="white", linewidth=0.6, zorder=3)
        for pos, val in zip(positions, data):
            if val <= 0:
                continue
            ax.annotate(
                f"{val:.1f}",
                (val, pos), ha="left", va="center",
                fontsize=9, color="#444444",
                xytext=(3, 0), textcoords="offset points",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(df["key"])
    ax.set_xlabel("Giá trị (%)")
    ax.set_title("Precision, Recall và F1-score theo từng trường thông tin")
    ax.set_xlim(0, 105)
    style_axes(ax, grid_axis="x")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95,
              edgecolor="#D6D6D6", fontsize=12)

    save(fig, "02_field_metrics.png")


# ---------------------------------------------------------------------------
# 03 - Phân bố accuracy theo từng hóa đơn
# ---------------------------------------------------------------------------
def plot_accuracy_distribution():
    csv_path = os.path.join(EVAL_DIR, "instance_accuracy.csv")
    check_file(csv_path)

    df = pd.read_csv(csv_path)

    if "accuracy" not in df.columns:
        raise ValueError("Không tìm thấy cột 'accuracy' trong instance_accuracy.csv")

    acc = df["accuracy"].apply(to_percent)
    total = len(acc)

    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

    groups = pd.cut(acc, bins=bins, labels=labels,
                    include_lowest=True, right=True)
    counts = groups.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=SEQ_BLUES, width=0.68,
                  edgecolor="white", linewidth=1.2, zorder=3)

    for bar, value in zip(bars, counts.values):
        pct = value / total * 100 if total else 0
        ax.annotate(
            f"{value}\n({pct:.0f}%)",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center", va="bottom",
            fontsize=12, fontweight="bold",
            xytext=(0, 4), textcoords="offset points",
        )

    ax.set_title("Phân bố độ chính xác theo từng hóa đơn")
    ax.set_xlabel("Khoảng Accuracy")
    ax.set_ylabel("Số lượng hóa đơn")
    ax.set_ylim(0, max(counts.values) * 1.18)
    style_axes(ax, grid_axis="y")

    save(fig, "03_accuracy_distribution.png")


# ---------------------------------------------------------------------------
# 04 - Số lượng hóa đơn theo từng Agent
# ---------------------------------------------------------------------------
def plot_agent_distribution():
    check_file(PRED_FILE)

    with open(PRED_FILE, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    agents = [
        item.get("agent_used") or item.get("template_id") or "unknown"
        for item in predictions
    ]

    counts = pd.Series(agents).value_counts().sort_values(ascending=True)
    top_value = counts.max()
    colors = [COLOR_HIGHLIGHT if v == top_value else COLOR_PRIMARY
              for v in counts.values]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars = ax.barh(counts.index, counts.values, color=colors,
                   edgecolor="white", linewidth=1.0, height=0.66, zorder=3)

    for bar, value in zip(bars, counts.values):
        ax.annotate(
            str(value),
            (value, bar.get_y() + bar.get_height() / 2),
            ha="left", va="center",
            fontsize=12, fontweight="bold",
            xytext=(5, 0), textcoords="offset points",
        )

    ax.set_title("Số lượng hóa đơn theo từng Agent")
    ax.set_xlabel("Số lượng hóa đơn")
    ax.set_ylabel("Agent")
    ax.set_xlim(0, top_value * 1.12)
    style_axes(ax, grid_axis="x")

    save(fig, "04_agent_distribution.png")


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
