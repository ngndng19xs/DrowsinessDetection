import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

def train_dummy_model():
    """
    Script tạo mô hình Random Forest ảo (Dummy Model) cho mục đích minh họa kiến trúc phần mềm.
    Mô hình nhận đầu vào là ma trận không gian - thời gian 4x15 (flattened -> 60 features) 
    gồm các thông số (EAR, MAR, Pitch, Yaw) qua 15 frames liên tiếp.
    """
    print("Bắt đầu tạo dữ liệu giả lập...")
    
    # Số mẫu (samples)
    n_samples = 2000
    
    # 4 thông số x 15 frames = 60 features
    X = np.zeros((n_samples, 60))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        # 0: Normal, 1: Drowsy (nhắm mắt/ngáp), 2: Distracted (quay đầu)
        label = np.random.choice([0, 1, 2])
        y[i] = label
        
        if label == 0: # Normal
            # EAR cao (>0.25), MAR thấp (<0.3), Pose nhỏ
            ear = np.random.uniform(0.26, 0.35, 15)
            mar = np.random.uniform(0.1, 0.25, 15)
            pitch = np.random.uniform(-10, 10, 15)
            yaw = np.random.uniform(-15, 15, 15)
        elif label == 1: # Drowsy (buồn ngủ)
            # EAR thấp hoặc MAR cao
            is_yawn = np.random.choice([True, False])
            if is_yawn:
                ear = np.random.uniform(0.25, 0.35, 15)
                mar = np.random.uniform(0.4, 0.8, 15) # Ngáp rộng
            else:
                ear = np.random.uniform(0.1, 0.18, 15) # Mắt nhắm
                mar = np.random.uniform(0.1, 0.25, 15)
            pitch = np.random.uniform(-10, 25, 15) # Cúi gật đầu
            yaw = np.random.uniform(-15, 15, 15)
        else: # Distracted
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
        
    print("Khởi tạo và huấn luyện Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    
    # Kiểm tra độ chính xác trên dữ liệu train
    acc = model.score(X, y)
    print(f"Độ chính xác mô hình giả lập: {acc*100:.2f}%")
    
    # Lưu mô hình
    save_dir = os.path.join(os.path.dirname(__file__), "src", "models")
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, "rf_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    print(f"Lưu thành công mô hình tại: {model_path}")

if __name__ == "__main__":
    train_dummy_model()
