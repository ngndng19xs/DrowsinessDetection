import os

def test_system():
    """
    Script kiểm thử nhanh hệ thống bằng cách gọi main.py
    Hoặc có thể mở rộng chạy unittest ở đây.
    """
    print("Bắt đầu chạy kịch bản kiểm thử...")
    
    model_path = os.path.join("src", "models", "rf_model.pkl")
    if not os.path.exists(model_path):
        print("[WARNING] Chưa có mô hình Random Forest. Vui lòng chạy `python train_dummy_model.py` trước.")
    else:
        print("[OK] Đã tìm thấy file mô hình Random Forest.")
        
    print("Khởi chạy ứng dụng DMS...")
    os.system("python main.py")

if __name__ == "__main__":
    test_system()
