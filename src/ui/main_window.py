import cv2
import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk
import queue
import time
import numpy as np
import mediapipe as mp
import threading
import winsound

def run_main_ui(result_queue, stop_event, shared_state):
    """
    Main Thread: Giao diện hiển thị sử dụng Tkinter để mô phỏng giao diện chuẩn.
    """
    root = tk.Tk()
    root.title("Hệ Thống Giám Sát Lái Xe (DMS)")
    
    # Kích thước cố định để thoải mái hiển thị UI như ảnh
    root.geometry("1050x650")
    
    # Bảng màu chủ đạo từ ảnh
    BG_COLOR = "#2B3E50"      # Xanh navy tối (nền)
    GREEN_COLOR = "#27AE60"   # Xanh lá (Normal)
    RED_COLOR = "#E74C3C"     # Đỏ (Cảnh báo)
    CYAN_BORDER = "#1ABC9C"   # Xanh ngọc (viền hộp)
    TEXT_COLOR = "white"      # Màu chữ mặc định
    FPS_COLOR = "#F1C40F"     # Vàng (FPS)
    
    root.configure(bg=BG_COLOR)
    
    # Xử lý sự kiện đóng cửa sổ
    def on_closing():
        stop_event.set()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # --- BỐ CỤC CHÍNH (Dùng grid để đảm bảo không bị co giãn lấn chiếm) ---
    root.columnconfigure(0, weight=1) # Cột chứa video giãn nở
    root.columnconfigure(1, weight=0, minsize=350) # Cột chứa thông tin cố định 350px
    root.rowconfigure(0, weight=1)
    
    # --- KHUNG BÊN TRÁI: VIDEO ---
    video_frame = tk.Frame(root, bg="black", bd=2)
    video_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
    
    video_label = tk.Label(video_frame, bg="black")
    video_label.pack(fill=tk.BOTH, expand=True)
    
    # --- KHUNG BÊN PHẢI: THÔNG TIN ---
    info_frame = tk.Frame(root, bg=BG_COLOR, width=350)
    info_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 20), pady=20)
    info_frame.pack_propagate(False) # Ngăn info_frame bị co giãn theo nội dung
    
    # 1. Tiêu đề "TRẠNG THÁI TÀI XẾ"
    lbl_title = tk.Label(info_frame, text="TRẠNG THÁI TÀI XẾ", font=("Arial", 16, "bold"), bg=BG_COLOR, fg=TEXT_COLOR)
    lbl_title.pack(pady=(10, 20))
    
    # 2. Hộp trạng thái
    lbl_status = tk.Label(info_frame, text="NORMAL", font=("Arial", 28, "bold"), bg=GREEN_COLOR, fg=TEXT_COLOR, width=12, pady=15)
    lbl_status.pack(pady=(0, 30))
    
    # 3. Hộp chứa thông số (Border cyan)
    stats_border = tk.Frame(info_frame, bg=CYAN_BORDER, bd=1)
    stats_border.pack(fill=tk.X)
    
    stats_inner = tk.Frame(stats_border, bg=BG_COLOR, padx=20, pady=20)
    stats_inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1) # Dùng padding 1px để lộ nền cyan làm viền
    
    stats_font = ("Consolas", 13)
    
    lbl_ear = tk.Label(stats_inner, text="EAR   : 0.000", font=stats_font, bg=BG_COLOR, fg=TEXT_COLOR, anchor="w")
    lbl_ear.pack(fill=tk.X, pady=8)
    
    lbl_mar = tk.Label(stats_inner, text="MAR   : 0.000", font=stats_font, bg=BG_COLOR, fg=TEXT_COLOR, anchor="w")
    lbl_mar.pack(fill=tk.X, pady=8)
    
    lbl_pitch = tk.Label(stats_inner, text="Pitch : 0.0°", font=stats_font, bg=BG_COLOR, fg=TEXT_COLOR, anchor="w")
    lbl_pitch.pack(fill=tk.X, pady=8)
    
    lbl_yaw = tk.Label(stats_inner, text="Yaw   : 0.0°", font=stats_font, bg=BG_COLOR, fg=TEXT_COLOR, anchor="w")
    lbl_yaw.pack(fill=tk.X, pady=8)
    
    lbl_fps = tk.Label(stats_inner, text="FPS   : 0.0", font=stats_font, bg=BG_COLOR, fg=FPS_COLOR, anchor="w")
    lbl_fps.pack(fill=tk.X, pady=8)
    
    # Người dùng yêu cầu KHÔNG CODE 2 nút (Webcam và Video) nên bỏ qua phần đó.

    # --- KHỞI TẠO BIẾN UPDATE ---
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    custom_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 255))
    
    last_time = time.time()
    frame_count = 0
    current_fps = 0.0
    last_alert_time = 0
    
    def update_ui():
        nonlocal last_time, frame_count, current_fps, last_alert_time
        
        if stop_event.is_set():
            root.destroy()
            return
            
        try:
            # Lấy frame từ queue (không block)
            frame = result_queue.get_nowait()
            
            # Tính FPS
            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                current_fps = frame_count / (now - last_time)
                frame_count = 0
                last_time = now
            
            # Lấy dữ liệu state
            state_data = shared_state.get_all()
            status = state_data.get("status", "NORMAL")
            ear = state_data.get("ear", 0.0) or 0.0
            mar = state_data.get("mar", 0.0) or 0.0
            pitch = state_data.get("pitch", 0.0) or 0.0
            yaw = state_data.get("yaw", 0.0) or 0.0
            landmarks = state_data.get("landmarks", None)
            
            # Vẽ face mesh nếu có
            if landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=custom_spec
                )
            
            # Chuyển đổi hệ màu để hiển thị trên Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Lấy kích thước khung chứa video để resize cho phù hợp
            vw = video_label.winfo_width()
            vh = video_label.winfo_height()
            
            if vw > 10 and vh > 10:
                h, w = frame_rgb.shape[:2]
                scale = min(vw / w, vh / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Resize giữ nguyên tỉ lệ khung hình
                resized_frame = cv2.resize(frame_rgb, (new_w, new_h))
                
                # Tạo canvas nền đen và dán video vào giữa
                canvas = np.zeros((vh, vw, 3), dtype=np.uint8)
                y_offset = (vh - new_h) // 2
                x_offset = (vw - new_w) // 2
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
                frame_rgb = canvas
                
            # Đổi mảng numpy thành đối tượng Image của PIL
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Gắn vào label (giữ reference để không bị garbage collection)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
            
            # Cập nhật giao diện text
            if status in ["DROWSY", "DISTRACTED"]:
                lbl_status.configure(text=status, bg=RED_COLOR)
                
                # Báo động âm thanh (tiếng Beep) mỗi giây
                current_time = time.time()
                if current_time - last_alert_time > 1.0:
                    threading.Thread(target=lambda: winsound.Beep(2500, 500), daemon=True).start()
                    last_alert_time = current_time
            else:
                lbl_status.configure(text="NORMAL", bg=GREEN_COLOR)
                
            lbl_ear.configure(text=f"EAR   : {ear:.3f}")
            lbl_mar.configure(text=f"MAR   : {mar:.3f}")
            lbl_pitch.configure(text=f"Pitch : {pitch:.1f}°")
            lbl_yaw.configure(text=f"Yaw   : {yaw:.1f}°")
            lbl_fps.configure(text=f"FPS   : {current_fps:.1f}")
            
        except queue.Empty:
            pass
        except Exception as e:
            print(f"[UI ERROR] {e}")
            
        # Lặp lại sau 15ms (tương đương ~60FPS giới hạn cập nhật giao diện)
        root.after(15, update_ui)

    # Khởi động vòng lặp update và giao diện
    update_ui()
    root.mainloop()
