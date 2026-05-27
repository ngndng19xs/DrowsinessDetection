# -*- coding: utf-8 -*-
import os
import sys

# Đảm bảo terminal Windows hiển thị đúng tiếng Việt UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pickle
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def generate_dummy_data(n_samples=2000):
    """
    Tạo dữ liệu giả lập mô phỏng ma trận không gian - thời gian 4x15.
    Mỗi mẫu gồm 60 features: (EAR, MAR, Pitch, Yaw) x 15 frames liên tiếp.

    Labels:
        0 - Normal
        1 - Drowsy (nhắm mắt / ngáp)
        2 - Distracted (quay đầu)
    """
    X = np.zeros((n_samples, 60))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        label = np.random.choice([0, 1, 2])
        y[i] = label

        if label == 0:  # Normal
            # EAR cao (>0.25), MAR thấp (<0.3), Pose nhỏ
            ear   = np.random.uniform(0.26, 0.35, 15)
            mar   = np.random.uniform(0.10, 0.25, 15)
            pitch = np.random.uniform(-10,  10,  15)
            yaw   = np.random.uniform(-15,  15,  15)

        elif label == 1:  # Drowsy (buồn ngủ)
            is_yawn = np.random.choice([True, False])
            if is_yawn:
                ear   = np.random.uniform(0.25, 0.35, 15)
                mar   = np.random.uniform(0.40, 0.80, 15)   # Ngáp rộng
            else:
                ear   = np.random.uniform(0.10, 0.18, 15)   # Mắt nhắm
                mar   = np.random.uniform(0.10, 0.25, 15)
            pitch = np.random.uniform(-10, 25, 15)           # Cúi gật đầu
            yaw   = np.random.uniform(-15, 15, 15)

        else:  # Distracted
            ear   = np.random.uniform(0.26, 0.35, 15)
            mar   = np.random.uniform(0.10, 0.25, 15)
            # Quay đầu ngang dọc rất lớn
            if np.random.rand() > 0.5:
                pitch = np.random.uniform( 25,  40, 15)
                yaw   = np.random.uniform( 35,  60, 15)
            else:
                pitch = np.random.uniform(-40, -25, 15)
                yaw   = np.random.uniform(-60, -35, 15)

        # Nối 4 mảng 15 phần tử thành mảng 60 phần tử
        sample = np.concatenate([ear, mar, pitch, yaw])
        X[i] = sample

    return X, y


def train_xgboost_model():
    """
    Script huấn luyện mô hình XGBoost cho bài toán
    phát hiện buồn ngủ (Drowsiness Detection).

    Lý do chọn XGBoost:
    - XGBoost (eXtreme Gradient Boosting) là thuật toán Ensemble Learning dựa trên
      kỹ thuật Gradient Boosting, xây dựng nhiều cây quyết định (Decision Trees)
      nối tiếp nhau, mỗi cây học từ sai sót của cây trước.
    - Hiệu suất vượt trội trên dữ liệu dạng bảng (tabular data) như vector 60 chiều
      (4 features x 15 frames) của bài toán này.
    - Không yêu cầu chuẩn hoá features như SVM (khác biệt quan trọng).
    - Tích hợp sẵn xử lý dữ liệu mất cân bằng qua tham số `scale_pos_weight`.
    - Hỗ trợ tính năng `predict_proba` và `feature_importances_` để phân tích.

    Siêu tham số quan trọng:
    - n_estimators   : Số lượng cây boosting (nhiều hơn → chính xác hơn nhưng chậm hơn).
    - max_depth      : Độ sâu tối đa của mỗi cây (tránh overfitting).
    - learning_rate  : Tốc độ học (shrinkage) sau mỗi bước boosting.
    - subsample      : Tỷ lệ mẫu lấy ngẫu nhiên mỗi cây (giảm overfitting).
    - colsample_bytree: Tỷ lệ features lấy ngẫu nhiên mỗi cây.
    - use_label_encoder: Tắt vì XGBoost >= 1.6 không cần encoder nội bộ.
    - eval_metric    : Sử dụng 'mlogloss' cho bài toán đa lớp (multiclass).
    """
    print("=" * 60)
    print("  HUẤN LUYỆN MÔ HÌNH XGBOOST - DROWSINESS DETECTION")
    print("=" * 60)

    # --- 1. Tạo dữ liệu giả lập ---
    print("\n[1/5] Tạo dữ liệu giả lập (2000 mẫu, 60 features)...")
    X, y = generate_dummy_data(n_samples=2000)
    print(f"       Tổng số mẫu: {X.shape[0]} | Số features: {X.shape[1]}")
    unique, counts = np.unique(y, return_counts=True)
    label_names = {0: "Normal", 1: "Drowsy", 2: "Distracted"}
    for lbl, cnt in zip(unique, counts):
        print(f"       Lớp {int(lbl)} ({label_names[int(lbl)]}): {cnt} mẫu")

    # --- 2. Chia tập train / test ---
    print("\n[2/5] Chia dữ liệu: 80% train - 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"       Train: {X_train.shape[0]} mẫu | Test: {X_test.shape[0]} mẫu")

    # --- 3. Khởi tạo mô hình XGBoost ---
    print("\n[3/5] Khởi tạo mô hình XGBoost (đa lớp - softmax)...")
    xgb_model = XGBClassifier(
        n_estimators=300,           # 300 cây boosting
        max_depth=6,                # Độ sâu tối đa mỗi cây
        learning_rate=0.1,          # Shrinkage rate
        subsample=0.8,              # 80% mẫu cho mỗi cây
        colsample_bytree=0.8,       # 80% features cho mỗi cây
        objective='multi:softprob', # Phân loại đa lớp → trả về xác suất
        num_class=3,                # Số lớp: Normal / Drowsy / Distracted
        eval_metric='mlogloss',     # Metric đánh giá: Multiclass Log Loss
        use_label_encoder=False,    # Tắt encoder nội bộ (deprecated)
        random_state=42,
        n_jobs=-1                   # Dùng toàn bộ CPU cores
    )
    print(f"       Cấu hình: n_estimators={xgb_model.n_estimators}, "
          f"max_depth={xgb_model.max_depth}, "
          f"learning_rate={xgb_model.learning_rate}")

    # --- 4. Huấn luyện với Early Stopping ---
    print("\n[4/5] Đang huấn luyện XGBoost...")
    # Tạo tập validation nội bộ từ tập train để dùng early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False               # Tắt log từng vòng lặp
    )
    print("       Hoàn thành huấn luyện!")

    # --- 5. Đánh giá ---
    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[5/5] Kết quả đánh giá trên tập Test:")
    print(f"       Accuracy: {acc * 100:.2f}%")
    print("\n       Classification Report:")
    target_names = ['Normal (0)', 'Drowsy (1)', 'Distracted (2)']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # --- Phân tích Feature Importance (Top 10) ---
    print("       Top 10 Features quan trọng nhất:")
    feature_names = (
        [f"EAR_f{i}"   for i in range(15)] +
        [f"MAR_f{i}"   for i in range(15)] +
        [f"Pitch_f{i}" for i in range(15)] +
        [f"Yaw_f{i}"   for i in range(15)]
    )
    importances = xgb_model.feature_importances_
    top10_idx = np.argsort(importances)[::-1][:10]
    for rank, idx in enumerate(top10_idx, 1):
        print(f"       {rank:2d}. {feature_names[idx]:<12s}: {importances[idx]:.4f}")

    # --- 6. Lưu mô hình ---
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "xgboost_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(xgb_model, f)

    print(f"\nLưu thành công mô hình XGBoost tại: {model_path}")
    print(f"Kích thước file: {os.path.getsize(model_path) / 1024:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    train_xgboost_model()
