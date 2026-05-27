import threading
import queue
import time
import sys
import os

"""
MODULE CHÍNH: KHỞI CHẠY HỆ THỐNG (Main Entry Point)
---------------------------------------------------
Kiến trúc luồng 4-Thread Architecture (Dựa trên báo cáo Buổi 5):
1. Camera Thread: Đọc I/O từ camera, đẩy frame vào `frame_queue`.
2. AI Thread: Đọc từ `frame_queue`, tính toán MediaPipe, Tracking Sliding Window, 
   chạy Random Forest, đẩy kết quả vào `SharedState` và frame đã vẽ vào `result_queue`.
3. Alert Thread: Đọc trạng thái từ `SharedState`, phát âm thanh non-blocking nếu DROWSY/DISTRACTED.
4. UI Thread (Main): Đọc frame từ `result_queue` và render chỉ số từ `SharedState` bằng cv2.imshow.
"""

from src.threads.camera_thread import CameraThread
from src.threads.ai_thread import AIThread
from src.threads.alert_thread import AlertThread
from src.ui.main_window import run_main_ui
from src.utils.shared_state import SharedState

def main():
    print("==================================================")
    print(" KHỞI ĐỘNG HỆ THỐNG CẢNH BÁO LÁI XE (DMS)")
    print("==================================================")

    # 1. Khởi tạo Event điều phối luồng
    stop_event = threading.Event()
    
    # 2. Khởi tạo Shared State và các Queues
    shared_state = SharedState()
    frame_queue = queue.Queue(maxsize=2)
    result_queue = queue.Queue(maxsize=2)
    
    # 3. Khởi tạo các Luồng (Threads)
    # Source=0 là Webcam, đổi thành đường dẫn file mp4 nếu cần test video
    camera_thread = CameraThread(frame_queue=frame_queue, stop_event=stop_event, source=0)
    
    ai_thread = AIThread(frame_queue=frame_queue, result_queue=result_queue, 
                         shared_state=shared_state, stop_event=stop_event)
    
    alert_thread = AlertThread(shared_state=shared_state, stop_event=stop_event, sound_path="assets/sounds/alarm.wav")
    
    # 4. Bắt đầu chạy các luồng
    print("[INFO] Đang khởi động Camera Thread...")
    camera_thread.start()
    
    print("[INFO] Đang khởi động AI Thread...")
    ai_thread.start()
    
    print("[INFO] Đang khởi động Alert Thread...")
    alert_thread.start()
    
    time.sleep(1.0) # Đợi một chút để camera khởi động
    print("\n[SUCCESS] Hệ thống đang chạy! Nhấn phím 'q' trên cửa sổ video để THOÁT.")

    # 5. Chạy Luồng UI (Main Thread chặn ở đây)
    try:
        run_main_ui(result_queue=result_queue, stop_event=stop_event, shared_state=shared_state)
    except KeyboardInterrupt:
        stop_event.set()
        
    print("\n[INFO] Tín hiệu thoát (stop_event) đã được kích hoạt. Đang đóng hệ thống...")
    
    # 6. Dọn dẹp tài nguyên
    stop_event.set()
    
    camera_thread.stop()
    alert_thread.stop()
    
    camera_thread.join(timeout=2.0)
    ai_thread.join(timeout=2.0)
    alert_thread.join(timeout=2.0)
    
    print("[SUCCESS] Tất cả các luồng đã được dọn dẹp sạch sẽ. Tạm biệt!")

if __name__ == "__main__":
    main()
