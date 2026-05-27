import threading
import time
import pygame
import os
import sys

# Đảm bảo có thể import cấu hình settings khi chạy file riêng lẻ
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config.settings import ALERT_COOLDOWN

class AlertThread(threading.Thread):
    """
    Luồng độc lập đảm nhiệm việc phát âm thanh cảnh báo khi tài xế có dấu hiệu Buồn ngủ hoặc Mất tập trung.
    Luồng này liên tục đọc trạng thái tổng thể từ Shared State.
    """
    def __init__(self, shared_state, stop_event, sound_path="assets/sounds/alarm.wav"):
        super().__init__()
        self.shared_state = shared_state
        self.stop_event = stop_event
        self.sound_path = sound_path
        self.last_alert_time = 0.0

        # Khởi tạo pygame mixer để xử lý âm thanh không chặn
        pygame.mixer.init()
        try:
            self.sound = pygame.mixer.Sound(self.sound_path)
        except Exception as e:
            print(f"[ERROR] AlertThread: Không thể tải âm thanh từ {self.sound_path}. {e}")
            self.sound = None

    def run(self):
        while not self.stop_event.is_set():
            # Đọc trạng thái từ Shared State
            state = self.shared_state.get("status")
            
            if state in ["DROWSY", "DISTRACTED"]:
                current_time = time.time()
                # Logic Cooldown: Chỉ phát nếu khoảng cách từ lần cuối cảnh báo > ALERT_COOLDOWN (3 giây)
                if current_time - self.last_alert_time >= ALERT_COOLDOWN:
                    if self.sound:
                        self.sound.play()
                    print(f"[{time.strftime('%H:%M:%S')}] 🚨 PHÁT ÂM THANH CẢNH BÁO: {state}!")
                    self.last_alert_time = current_time
            
            # Ngủ 1 chút để tránh quá tải CPU ( polling 100ms 1 lần )
            time.sleep(0.1)

    def stop(self):
        pygame.mixer.quit()
