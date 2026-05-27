import threading
import queue
import cv2
import mediapipe as mp
import sys
import os
import collections
import numpy as np
import pickle

# Đảm bảo đường dẫn import tương đối
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config.settings import EAR_THRESHOLD, MAR_THRESHOLD, PITCH_THRESHOLD, YAW_THRESHOLD
from src.utils.geometry import calculate_ear, calculate_mar
from src.utils.pose import get_head_pose
from src.utils.preprocessing import preprocess_frame

class AIThread(threading.Thread):
    """
    Luồng AI đảm nhận tính toán nặng nhất của hệ thống: Phát hiện khuôn mặt và đánh giá chỉ số sinh trắc.
    Sử dụng Sliding Temporal Window 4x15 kết hợp Random Forest Classifier.
    """
    def __init__(self, frame_queue, result_queue, shared_state, stop_event):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.shared_state = shared_state
        self.stop_event = stop_event
        
        # Khởi tạo MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Cơ chế Sliding Temporal Window (15 frames cho 4 đặc trưng)
        # Sẽ lưu dạng tuple (ear, mar, pitch, yaw)
        self.sliding_window = collections.deque(maxlen=15)
        
        # Số lần liên tiếp dự đoán Drowsy
        self.consecutive_drowsy = 0
        self.DROWSY_CONSECUTIVE_TH = 15
        
        # Tải mô hình Random Forest
        self.model = None
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("[INFO] Đã tải mô hình Random Forest thành công.")
        else:
            print("[WARNING] Không tìm thấy mô hình rf_model.pkl. Sẽ dùng Logic Ngưỡng Tạm Thời!")

    def run(self):
        while not self.stop_event.is_set():
            # 1. Lấy frame từ queue
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            display_frame = frame.copy()
            h, w, _ = frame.shape
            
            # 2. Tiền xử lý
            rgb_frame, _ = preprocess_frame(frame)
            
            # 3. Chạy MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            state = "Normal"
            ear, mar, pitch, yaw = 0.0, 0.0, 0.0, 0.0
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # EAR
                left_eye_indices = [362, 385, 387, 263, 373, 380]
                right_eye_indices = [33, 160, 158, 133, 153, 144]
                ear = (calculate_ear([landmarks[i] for i in left_eye_indices], w, h) + 
                       calculate_ear([landmarks[i] for i in right_eye_indices], w, h)) / 2.0
                
                # MAR
                mouth_indices = [61, 81, 13, 311, 291, 402, 14, 178]
                mar = calculate_mar([landmarks[i] for i in mouth_indices], w, h)
                
                # Pose
                pitch, yaw, _ = get_head_pose(landmarks, w, h)
                
                # Đẩy vào cửa sổ trượt (Sliding Window)
                self.sliding_window.append((ear, mar, pitch, yaw))
                
                # Đợi đủ 15 frames để bắt đầu phân loại
                if len(self.sliding_window) == 15:
                    if self.model is not None:
                        # Rút trích theo đúng thứ tự lúc train: [ear_15, mar_15, pitch_15, yaw_15]
                        ears = [item[0] for item in self.sliding_window]
                        mars = [item[1] for item in self.sliding_window]
                        pitches = [item[2] for item in self.sliding_window]
                        yaws = [item[3] for item in self.sliding_window]
                        
                        features = np.concatenate([ears, mars, pitches, yaws]).reshape(1, -1)
                        pred = self.model.predict(features)[0]
                        # 0: Normal, 1: Drowsy, 2: Distracted
                        if pred == 1:
                            raw_state = "Drowsy"
                        elif pred == 2:
                            raw_state = "Distracted"
                        else:
                            raw_state = "Normal"
                    else:
                        # Fallback threshold logic
                        if ear < EAR_THRESHOLD or mar > MAR_THRESHOLD:
                            raw_state = "Drowsy"
                        elif abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD:
                            raw_state = "Distracted"
                        else:
                            raw_state = "Normal"
                    
                    # Logic bộ lọc chống nhiễu (Consecutive Counter)
                    if raw_state == "Drowsy":
                        self.consecutive_drowsy += 1
                        if self.consecutive_drowsy >= self.DROWSY_CONSECUTIVE_TH:
                            state = "Drowsy"
                    elif raw_state == "Distracted":
                        state = "Distracted"
                        self.consecutive_drowsy = 0
                    else:
                        state = "Normal"
                        self.consecutive_drowsy = 0
            
            # Cập nhật vào Shared State
            self.shared_state.update(
                status=state.upper(),
                ear=ear,
                mar=mar,
                pitch=pitch,
                yaw=yaw,
                landmarks=results.multi_face_landmarks[0] if results.multi_face_landmarks else None
            )
            
            # Đẩy frame vào result_queue để UI Thread render
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.result_queue.put_nowait(display_frame)
            except queue.Full:
                pass
                
        self.face_mesh.close()
