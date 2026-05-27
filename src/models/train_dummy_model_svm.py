import os
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
    y = np.zeros(n_samples)

    for i in range(n_samples):
        label = np.random.choice([0, 1, 2])
        y[i] = label

        if label == 0:  # Normal
            # EAR cao (>0.25), MAR thấp (<0.3), Pose nhỏ
            ear = np.random.uniform(0.26, 0.35, 15)
            mar = np.random.uniform(0.1, 0.25, 15)
            pitch = np.random.uniform(-10, 10, 15)
            yaw = np.random.uniform(-15, 15, 15)

        elif label == 1:  # Drowsy (buồn ngủ)
            is_yawn = np.random.choice([True, False])
            if is_yawn:
                ear = np.random.uniform(0.25, 0.35, 15)
                mar = np.random.uniform(0.4, 0.8, 15)   # Ngáp rộng
            else:
                ear = np.random.uniform(0.1, 0.18, 15)   # Mắt nhắm
                mar = np.random.uniform(0.1, 0.25, 15)
            pitch = np.random.uniform(-10, 25, 15)       # Cúi gật đầu
            yaw = np.random.uniform(-15, 15, 15)

        else:  # Distracted
            ear = np.random.uniform(0.26, 0.35, 15)
            mar = np.random.uniform(0.1, 0.25, 15)
            # Quay đầu ngang dọc rất lớn
            if np.random.rand() > 0.5:
                pitch = np.random.uniform(25, 40, 15)
                yaw = np.random.uniform(35, 60, 15)
            else:
                pitch = np.random.uniform(-40, -25, 15)
                yaw = np.random.uniform(-60, -35, 15)

        # Nối 4 mảng 15 phần tử thành mảng 60 phần tử
        sample = np.concatenate([ear, mar, pitch, yaw])
        X[i] = sample

    return X, y


def train_svm_model():
    """
    Script huấn luyện mô hình SVM (Support Vector Machine) cho bài toán
    phát hiện buồn ngủ (Drowsiness Detection).

    Lý do chọn SVM:
    - SVM tìm siêu phẳng (hyperplane) tối ưu để phân tách các lớp trong không gian
      đặc trưng nhiều chiều → phù hợp với vector 60 chiều (4 features x 15 frames).
    - Kernel RBF cho phép xử lý tốt dữ liệu phi tuyến tính (non-linear boundary).

    Lưu ý quan trọng:
    - SVM nhạy cảm với scale của features (khác với Random Forest).
      → Bắt buộc dùng StandardScaler để chuẩn hoá features trước khi train.
    - Pipeline (StandardScaler + SVC) được lưu chung 1 file .pkl,
      nên khi load model ra predict KHÔNG cần chuẩn hoá lại thủ công.
    """
    print("=" * 60)
    print("  HUẤN LUYỆN MÔ HÌNH SVM - DROWSINESS DETECTION")
    print("=" * 60)

    # --- 1. Tạo dữ liệu giả lập ---
    print("\n[1/4] Tạo dữ liệu giả lập (2000 mẫu, 60 features)...")
    X, y = generate_dummy_data(n_samples=2000)

    # --- 2. Chia tập train / test ---
    print("[2/4] Chia dữ liệu: 80% train - 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"       Train: {X_train.shape[0]} mẫu | Test: {X_test.shape[0]} mẫu")

    # --- 3. Xây dựng Pipeline: StandardScaler + SVM ---
    print("[3/4] Xây dựng Pipeline (StandardScaler → SVC với kernel RBF)...")
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),           # Chuẩn hoá features về mean=0, std=1
        ('svm', SVC(
            kernel='rbf',                       # Kernel phi tuyến
            C=10.0,                             # Regularization: cân bằng giữa margin rộng và lỗi
            gamma='scale',                      # gamma = 1 / (n_features * X.var())
            probability=True,                   # Bật xác suất predict_proba
            random_state=42,
            class_weight='balanced'             # Cân bằng class weight cho dữ liệu không đều
        ))
    ])

    # --- 4. Huấn luyện ---
    print("       Đang huấn luyện SVM...")
    svm_pipeline.fit(X_train, y_train)

    # --- 5. Đánh giá ---
    y_pred = svm_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[4/4] Kết quả đánh giá trên tập Test:")
    print(f"       Accuracy: {acc * 100:.2f}%")
    print("\n       Classification Report:")
    target_names = ['Normal (0)', 'Drowsy (1)', 'Distracted (2)']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # --- 6. Lưu mô hình ---
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "svm_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(svm_pipeline, f)

    print(f"Lưu thành công Pipeline (Scaler + SVM) tại: {model_path}")
    print(f"Kích thước file: {os.path.getsize(model_path) / 1024:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    train_svm_model()
