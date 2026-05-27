import threading

class SharedState:
    """
    Shared State Object
    Bộ nhớ dùng chung để đồng bộ dữ liệu giữa các luồng.
    Sử dụng Lock để đảm bảo an toàn luồng (thread-safe) khi đọc/ghi đồng thời.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._state = {
            "status": "NORMAL",   # NORMAL, DROWSY, DISTRACTED
            "ear": 0.0,
            "mar": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "fps": 0.0,
            "landmarks": None
        }

    def update(self, **kwargs):
        """Cập nhật các trạng thái. Ví dụ: state.update(status='DROWSY', ear=0.2)"""
        with self._lock:
            for key, value in kwargs.items():
                if key in self._state:
                    self._state[key] = value

    def get(self, key, default=None):
        """Lấy một trạng thái cụ thể"""
        with self._lock:
            return self._state.get(key, default)

    def get_all(self):
        """Lấy toàn bộ trạng thái (copy) để tránh race condition khi xử lý lâu"""
        with self._lock:
            return self._state.copy()
