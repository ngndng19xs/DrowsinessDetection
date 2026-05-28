# -*- coding: utf-8 -*-
"""
MODULE: ĐÁNH GIÁ & SO SÁNH CÁC MÔ HÌNH MACHINE LEARNING

Mục đích:
    - Tải các mô hình đã huấn luyện (RF, SVM, XGBoost) từ file .pkl
    - Đọc dữ liệu TEST THỰC TẾ từ data/test_features.csv
      (được tạo bởi extract_features_from_videos.py)
    - Tính toán các chỉ số đánh giá: Accuracy, Precision, Recall, F1-Score
    - Đo Latency: thời gian dự đoán trung bình trên 1 frame (sample)
    - Trực quan hóa kết quả bằng biểu đồ matplotlib

Cách chạy:
    python src/models/evaluate_models.py
"""

import os
import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Cấu hình output encoding (Windows)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# Đường dẫn
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.abspath(os.path.join(MODELS_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(MODELS_DIR, "evaluation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_FILES = {
    "Random Forest": os.path.join(MODELS_DIR, "rf_model.pkl"),
    "SVM":           os.path.join(MODELS_DIR, "svm_model.pkl"),
    "XGBoost":       os.path.join(MODELS_DIR, "xgboost_model.pkl"),
}

# File CSV dữ liệu test thực tế
TEST_CSV = os.path.join(ROOT_DIR, "data", "test_features.csv")

CLASS_NAMES = ["Normal", "Drowsy", "Distracted"]
N_LATENCY   = 500   # Số frame dùng để đo latency

# Màu sắc
PALETTE = {
    "Random Forest": "#4CAF50",   # Xanh lá
    "SVM":           "#2196F3",   # Xanh dương
    "XGBoost":       "#FF5722",   # Cam đỏ
}

BG_COLOR   = "#0F1117"
CARD_COLOR = "#1A1D27"
TEXT_COLOR = "#E8EAF0"
GRID_COLOR = "#2A2D3A"


# ══════════════════════════════════════════════════════════════════════════════
#  1. ĐỌC DỮ LIỆU TEST THỰC TẾ
# ══════════════════════════════════════════════════════════════════════════════
def load_test_data(csv_path: str = TEST_CSV) -> tuple:
    """
    Đọc dữ liệu test THỰC TẾ từ file CSV được tạo bởi extract_features_from_videos.py.

    CSV format: EAR_f0..EAR_f14 | MAR_f0..MAR_f14 | Pitch_f0..Pitch_f14 | Yaw_f0..Yaw_f14 | label

    Trả về:
        (X_test, y_test) dạng numpy array
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Khong tim thay file CSV test: {csv_path}\n"
            "Hay chay truoc: python src/models/extract_features_from_videos.py"
        )

    df = pd.read_csv(csv_path)
    X  = df.drop(columns=["label"]).values.astype(np.float32)
    y  = df["label"].values.astype(int)

    print(f"  >> Doc test CSV: {csv_path}")
    print(f"  >> Kich thuoc : {X.shape[0]} mau x {X.shape[1]} features")

    # In phân phối nhãn
    unique, counts = np.unique(y, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        label_name = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else f"Label_{lbl}"
        print(f"     Lop {lbl} ({label_name}): {cnt} mau")

    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  2. TẢI MÔ HÌNH
# ══════════════════════════════════════════════════════════════════════════════
def load_models() -> dict:
    """Tải tất cả file .pkl, bỏ qua model không tồn tại."""
    models = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
            size_kb = os.path.getsize(path) / 1024
            print(f"  [OK] {name:<16s} <- {os.path.basename(path)}  ({size_kb:.1f} KB)")
        else:
            print(f"  [!!] {name:<16s} KHONG TIM THAY: {path}")
    return models


# ══════════════════════════════════════════════════════════════════════════════
#  3. ĐO LATENCY (ms / frame)
# ══════════════════════════════════════════════════════════════════════════════
def measure_latency(model, X_latency: np.ndarray, n_runs: int = 3) -> dict:
    """
    Đo thời gian dự đoán trung bình cho 1 frame (sample).

    Tham số:
        model      : mô hình sklearn / xgboost đã huấn luyện
        X_latency  : ma trận feature shape (N, 60) dùng để đo
        n_runs     : số lần lặp để tính trung bình (giảm jitter)

    Trả về:
        dict với mean_ms, std_ms, min_ms, max_ms (đơn vị millisecond / frame)
    """
    n_samples = len(X_latency)
    times_per_frame = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(X_latency)
        elapsed = (time.perf_counter() - t0) / n_samples * 1000  # ms / frame
        times_per_frame.append(elapsed)

    return {
        "mean_ms": float(np.mean(times_per_frame)),
        "std_ms":  float(np.std(times_per_frame)),
        "min_ms":  float(np.min(times_per_frame)),
        "max_ms":  float(np.max(times_per_frame)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  4. TÍNH METRICS
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Tính Accuracy, Precision, Recall, F1-Score (macro + per-class) và Confusion Matrix."""
    y_pred = model.predict(X_test)
    return {
        "accuracy":          accuracy_score(y_test, y_pred),
        "precision_macro":   precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro":      recall_score(   y_test, y_pred, average="macro", zero_division=0),
        "f1_macro":          f1_score(       y_test, y_pred, average="macro", zero_division=0),
        "precision_per_class": precision_score(y_test, y_pred, average=None, zero_division=0),
        "recall_per_class":    recall_score(   y_test, y_pred, average=None, zero_division=0),
        "f1_per_class":        f1_score(       y_test, y_pred, average=None, zero_division=0),
        "confusion_matrix":    confusion_matrix(y_test, y_pred),
        "y_pred":              y_pred,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  5. IN BÁO CÁO CONSOLE
# ══════════════════════════════════════════════════════════════════════════════
def print_report(name: str, metrics: dict, latency: dict) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  MODEL: {name}")
    print(sep)
    print(f"  Accuracy  : {metrics['accuracy']*100:6.2f}%")
    print(f"  Precision : {metrics['precision_macro']*100:6.2f}%  (macro)")
    print(f"  Recall    : {metrics['recall_macro']*100:6.2f}%  (macro)")
    print(f"  F1-Score  : {metrics['f1_macro']*100:6.2f}%  (macro)")
    print(f"  Latency   : {latency['mean_ms']:.4f} ms/frame  "
          f"(+/- {latency['std_ms']:.4f} ms)")
    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':<14s}  {'Precision':>9s}  {'Recall':>8s}  {'F1':>8s}")
    print(f"  {'-'*46}")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:<14s}  "
              f"{metrics['precision_per_class'][i]*100:8.2f}%  "
              f"{metrics['recall_per_class'][i]*100:7.2f}%  "
              f"{metrics['f1_per_class'][i]*100:7.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  6. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def _apply_dark_style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Áp dụng theme tối đồng nhất cho một Axes."""
    ax.set_facecolor(CARD_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID_COLOR)
    ax.yaxis.set_tick_params(color=GRID_COLOR)
    ax.xaxis.set_tick_params(color=GRID_COLOR)
    ax.set_title(title,  color=TEXT_COLOR, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=9)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.8)


def plot_bar_comparison(all_results: dict, save_path: str) -> None:
    """
    Biểu đồ thanh nhóm so sánh 4 chỉ số (Accuracy, Precision, Recall, F1)
    giữa tất cả mô hình.
    """
    model_names = list(all_results.keys())
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1-Score\n(macro)"]

    n_models  = len(model_names)
    n_metrics = len(metric_keys)
    x = np.arange(n_metrics)
    bar_w = 0.22
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_w

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG_COLOR)
    _apply_dark_style(ax,
                      title="So sanh chi so danh gia giua cac mo hinh",
                      ylabel="Gia tri (%)")

    for i, (name, results) in enumerate(all_results.items()):
        values = [results["metrics"][k] * 100 for k in metric_keys]
        bars = ax.bar(x + offsets[i], values, width=bar_w,
                      color=PALETTE[name], alpha=0.88,
                      label=name, zorder=3,
                      edgecolor=BG_COLOR, linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.1f}", ha="center", va="bottom",
                    color=TEXT_COLOR, fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, color=TEXT_COLOR, fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
              fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  >> Luu bieu do: {save_path}")


def plot_latency_comparison(all_results: dict, save_path: str) -> None:
    """
    Biểu đồ thanh nằm ngang so sánh Latency (ms/frame) giữa các mô hình.
    Thêm annotation thông lượng (frames/s).
    """
    model_names = list(all_results.keys())
    means = [all_results[n]["latency"]["mean_ms"] for n in model_names]
    stds  = [all_results[n]["latency"]["std_ms"]  for n in model_names]
    fps   = [1000 / m for m in means]
    colors = [PALETTE[n] for n in model_names]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(BG_COLOR)
    _apply_dark_style(ax,
                      title="So sanh Latency du doan (ms / frame)",
                      xlabel="Thoi gian trung binh (ms / frame)")

    y_pos = np.arange(len(model_names))
    bars = ax.barh(y_pos, means, xerr=stds,
                   color=colors, alpha=0.88,
                   edgecolor=BG_COLOR, linewidth=0.5,
                   capsize=5, error_kw={"elinewidth": 1.5, "ecolor": TEXT_COLOR},
                   zorder=3, height=0.5)

    for bar, mean_v, fps_v in zip(bars, means, fps):
        ax.text(mean_v + max(means) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{mean_v:.3f} ms  ({fps_v:.0f} fps)",
                va="center", color=TEXT_COLOR, fontsize=9, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names, color=TEXT_COLOR, fontsize=10)
    ax.set_xlim(0, max(means) * 1.45)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.8)
    ax.grid(axis="y", visible=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  >> Luu bieu do: {save_path}")


def plot_confusion_matrices(all_results: dict, save_path: str) -> None:
    """
    Vẽ Confusion Matrix cho tất cả mô hình cạnh nhau.
    Dùng colormap xanh - từ thấp đến cao.
    """
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5))
    fig.patch.set_facecolor(BG_COLOR)
    if n_models == 1:
        axes = [axes]

    cmap = LinearSegmentedColormap.from_list(
        "drowsy", ["#0F1117", "#1565C0", "#42A5F5", "#E3F2FD"], N=256
    )

    for ax, (name, results) in zip(axes, all_results.items()):
        cm   = results["metrics"]["confusion_matrix"]
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # Normalize

        im = ax.imshow(cm_n, cmap=cmap, vmin=0, vmax=1)
        ax.set_facecolor(BG_COLOR)

        # Giá trị trong ô
        for r in range(len(CLASS_NAMES)):
            for c in range(len(CLASS_NAMES)):
                val_pct = cm_n[r, c] * 100
                val_abs = cm[r, c]
                color_txt = "#0F1117" if cm_n[r, c] > 0.55 else TEXT_COLOR
                ax.text(c, r, f"{val_pct:.1f}%\n({val_abs})",
                        ha="center", va="center",
                        color=color_txt, fontsize=10, fontweight="bold")

        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, color=TEXT_COLOR, fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, color=TEXT_COLOR, fontsize=9, rotation=45)
        ax.set_title(f"Confusion Matrix\n{name}",
                     color=TEXT_COLOR, fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted", color=TEXT_COLOR, fontsize=9)
        ax.set_ylabel("Actual",    color=TEXT_COLOR, fontsize=9)
        ax.tick_params(colors=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors=TEXT_COLOR)

    fig.suptitle("Ma tran nham lan (Confusion Matrix) - Normalized",
                 color=TEXT_COLOR, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  >> Luu bieu do: {save_path}")


def plot_per_class_f1(all_results: dict, save_path: str) -> None:
    """
    Biểu đồ radar / grouped bar F1-Score theo từng lớp cho mỗi mô hình.
    """
    model_names  = list(all_results.keys())
    n_models     = len(model_names)
    n_classes    = len(CLASS_NAMES)
    x = np.arange(n_classes)
    bar_w = 0.22
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_w

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG_COLOR)
    _apply_dark_style(ax,
                      title="F1-Score theo tung lop phan loai",
                      ylabel="F1-Score (%)")

    for i, (name, results) in enumerate(all_results.items()):
        f1_vals = results["metrics"]["f1_per_class"] * 100
        bars = ax.bar(x + offsets[i], f1_vals, width=bar_w,
                      color=PALETTE[name], alpha=0.88,
                      label=name, zorder=3,
                      edgecolor=BG_COLOR, linewidth=0.5)
        for bar, val in zip(bars, f1_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.1f}", ha="center", va="bottom",
                    color=TEXT_COLOR, fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, color=TEXT_COLOR, fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
              fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  >> Luu bieu do: {save_path}")


def plot_summary_dashboard(all_results: dict, y_test: np.ndarray, save_path: str) -> None:
    """
    Dashboard tổng hợp: 4 metrics + latency + confusion matrices trên 1 hình.
    """
    model_names = list(all_results.keys())
    n_models    = len(model_names)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        "DROWSINESS DETECTION - Model Evaluation Dashboard",
        color=TEXT_COLOR, fontsize=16, fontweight="bold", y=0.98
    )

    gs = GridSpec(3, n_models + 1, figure=fig,
                  hspace=0.50, wspace=0.38,
                  height_ratios=[1, 1, 1.4])

    # ── Hàng 1: Summary cards (Accuracy + F1) ──────────────────────────────
    metric_pairs = [
        ("accuracy",       "Accuracy"),
        ("f1_macro",       "F1-Score (macro)"),
    ]

    for col, (name, results) in enumerate(all_results.items()):
        for row, (key, label) in enumerate(metric_pairs):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor(CARD_COLOR)
            for spine in ax.spines.values():
                spine.set_edgecolor(PALETTE[name])
                spine.set_linewidth(2)
            val = results["metrics"][key] * 100
            ax.text(0.5, 0.62, f"{val:.2f}%",
                    ha="center", va="center", transform=ax.transAxes,
                    color=PALETTE[name], fontsize=22, fontweight="bold")
            ax.text(0.5, 0.22, f"{name}\n{label}",
                    ha="center", va="center", transform=ax.transAxes,
                    color=TEXT_COLOR, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    # ── Cột cuối hàng 0: Latency card ──────────────────────────────────────
    ax_lat_card = fig.add_subplot(gs[0, n_models])
    ax_lat_card.set_facecolor(CARD_COLOR)
    for spine in ax_lat_card.spines.values():
        spine.set_edgecolor("#FFC107")
        spine.set_linewidth(2)
    lat_lines = "\n".join(
        f"{name}: {all_results[name]['latency']['mean_ms']:.3f} ms"
        for name in model_names
    )
    ax_lat_card.text(0.5, 0.65, "Latency / frame",
                     ha="center", va="center", transform=ax_lat_card.transAxes,
                     color="#FFC107", fontsize=10, fontweight="bold")
    ax_lat_card.text(0.5, 0.30, lat_lines,
                     ha="center", va="center", transform=ax_lat_card.transAxes,
                     color=TEXT_COLOR, fontsize=9, linespacing=1.8)
    ax_lat_card.set_xticks([])
    ax_lat_card.set_yticks([])

    # ── Hàng 1 cột cuối: grouped bar Precision / Recall / F1 ────────────────
    ax_bar = fig.add_subplot(gs[1, n_models])
    ax_bar.set_facecolor(CARD_COLOR)
    metric_keys   = ["precision_macro", "recall_macro", "f1_macro"]
    metric_labels2 = ["Prec", "Recall", "F1"]
    x = np.arange(len(metric_keys))
    w = 0.22
    offs = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * w
    for i, name in enumerate(model_names):
        vals = [all_results[name]["metrics"][k] * 100 for k in metric_keys]
        ax_bar.bar(x + offs[i], vals, width=w, color=PALETTE[name],
                   alpha=0.88, label=name, zorder=3, edgecolor=BG_COLOR, linewidth=0.4)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_labels2, color=TEXT_COLOR, fontsize=8)
    ax_bar.set_ylim(0, 115)
    ax_bar.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax_bar.set_title("Precision / Recall / F1", color=TEXT_COLOR,
                     fontsize=8, fontweight="bold")
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.spines[["left", "bottom"]].set_color(GRID_COLOR)
    ax_bar.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.7)
    ax_bar.legend(facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
                  fontsize=7, loc="upper right")

    # ── Hàng 2: Confusion Matrices ──────────────────────────────────────────
    cmap = LinearSegmentedColormap.from_list(
        "drowsy", ["#0F1117", "#1565C0", "#42A5F5", "#E3F2FD"], N=256
    )
    for col, (name, results) in enumerate(all_results.items()):
        ax_cm = fig.add_subplot(gs[2, col])
        cm   = results["metrics"]["confusion_matrix"]
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax_cm.imshow(cm_n, cmap=cmap, vmin=0, vmax=1)
        ax_cm.set_facecolor(BG_COLOR)
        for r in range(len(CLASS_NAMES)):
            for c in range(len(CLASS_NAMES)):
                color_txt = "#0F1117" if cm_n[r, c] > 0.55 else TEXT_COLOR
                ax_cm.text(c, r,
                           f"{cm_n[r, c]*100:.1f}%\n({cm[r, c]})",
                           ha="center", va="center",
                           color=color_txt, fontsize=8, fontweight="bold")
        ax_cm.set_xticks(range(len(CLASS_NAMES)))
        ax_cm.set_yticks(range(len(CLASS_NAMES)))
        ax_cm.set_xticklabels(CLASS_NAMES, color=TEXT_COLOR, fontsize=8)
        ax_cm.set_yticklabels(CLASS_NAMES, color=TEXT_COLOR, fontsize=8, rotation=45)
        ax_cm.set_title(f"Confusion Matrix\n{name}",
                        color=TEXT_COLOR, fontsize=9, fontweight="bold")
        ax_cm.set_xlabel("Predicted", color=TEXT_COLOR, fontsize=8)
        ax_cm.set_ylabel("Actual",    color=TEXT_COLOR, fontsize=8)
        ax_cm.tick_params(colors=TEXT_COLOR)

    # ── Hàng 2 cột cuối: Latency bar chart ─────────────────────────────────
    ax_latency = fig.add_subplot(gs[2, n_models])
    ax_latency.set_facecolor(CARD_COLOR)
    lat_vals = [all_results[n]["latency"]["mean_ms"] for n in model_names]
    lat_stds = [all_results[n]["latency"]["std_ms"]  for n in model_names]
    colors_list = [PALETTE[n] for n in model_names]
    y_pos = np.arange(len(model_names))
    ax_latency.barh(y_pos, lat_vals, xerr=lat_stds,
                    color=colors_list, alpha=0.88,
                    capsize=4, error_kw={"elinewidth": 1, "ecolor": TEXT_COLOR},
                    edgecolor=BG_COLOR, linewidth=0.4, zorder=3, height=0.5)
    for j, (v, s) in enumerate(zip(lat_vals, lat_stds)):
        ax_latency.text(v + max(lat_vals) * 0.04, j,
                        f"{v:.3f}ms", va="center",
                        color=TEXT_COLOR, fontsize=8, fontweight="bold")
    ax_latency.set_yticks(y_pos)
    ax_latency.set_yticklabels(model_names, color=TEXT_COLOR, fontsize=8)
    ax_latency.set_xlim(0, max(lat_vals) * 1.55)
    ax_latency.invert_yaxis()
    ax_latency.set_title("Latency (ms/frame)", color=TEXT_COLOR,
                         fontsize=9, fontweight="bold")
    ax_latency.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax_latency.spines[["top", "right"]].set_visible(False)
    ax_latency.spines[["left", "bottom"]].set_color(GRID_COLOR)
    ax_latency.grid(axis="x", color=GRID_COLOR, linewidth=0.5, alpha=0.7)
    ax_latency.grid(axis="y", visible=False)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  >> Luu Dashboard: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  7. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print("  DANH GIA MO HINH - DROWSINESS DETECTION")
    print("  (Du lieu test THUC TE tu data/test_features.csv)")
    print(sep)

    # 1. Đọc dữ liệu test thực tế
    print("\n[1/4] Doc du lieu test thuc te tu CSV...")
    X_test, y_test = load_test_data(TEST_CSV)
    X_latency = X_test[:min(N_LATENCY, len(X_test))]
    print(f"  Test set: {len(X_test)} mau | Latency set: {len(X_latency)} mau")

    # 2. Tải mô hình
    print("\n[2/4] Tai cac mo hinh da huan luyen...")
    models = load_models()

    if not models:
        print("\n[LOI] Khong tim thay bat ky mo hinh nao. Hay chay cac script train truoc!")
        print("  python src/models/extract_features_from_videos.py")
        print("  python src/models/train_rf.py")
        print("  python src/models/train_svm.py")
        print("  python src/models/train_xgboost.py")
        return

    # 3. Tính metrics & latency cho từng model
    print("\n[3/4] Tinh metrics va do latency...")
    all_results = {}
    for name, model in models.items():
        print(f"\n  --- {name} ---")
        metrics = compute_metrics(model, X_test, y_test)
        latency = measure_latency(model, X_latency, n_runs=5)
        all_results[name] = {"metrics": metrics, "latency": latency}

        # In classification report đầy đủ
        print_report(name, metrics, latency)
        print(f"\n  Classification Report (sklearn):")
        print(classification_report(y_test, metrics["y_pred"],
                                    target_names=CLASS_NAMES, digits=4))

    # 4. Vẽ biểu đồ
    print(f"\n[4/4] Ve bieu do - luu vao: {OUTPUT_DIR}")
    plot_bar_comparison(
        all_results,
        os.path.join(OUTPUT_DIR, "01_metric_comparison.png")
    )
    plot_latency_comparison(
        all_results,
        os.path.join(OUTPUT_DIR, "02_latency_comparison.png")
    )
    plot_confusion_matrices(
        all_results,
        os.path.join(OUTPUT_DIR, "03_confusion_matrices.png")
    )
    plot_per_class_f1(
        all_results,
        os.path.join(OUTPUT_DIR, "04_per_class_f1.png")
    )
    plot_summary_dashboard(
        all_results, y_test,
        os.path.join(OUTPUT_DIR, "05_dashboard.png")
    )

    # Tóm tắt bảng so sánh cuối cùng
    print(f"\n{sep}")
    print("  BANG SO SANH TONG HOP")
    print(sep)
    header = f"  {'Model':<18s}  {'Acc':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Latency':>12s}"
    print(header)
    print(f"  {'-'*60}")
    for name, res in all_results.items():
        m = res["metrics"]
        l = res["latency"]
        print(f"  {name:<18s}  "
              f"{m['accuracy']*100:5.2f}%  "
              f"{m['precision_macro']*100:5.2f}%  "
              f"{m['recall_macro']*100:5.2f}%  "
              f"{m['f1_macro']*100:5.2f}%  "
              f"{l['mean_ms']:>8.4f} ms/fr")
    print(f"{sep}\n")
    print(f"  Ket qua luu tai: {OUTPUT_DIR}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
