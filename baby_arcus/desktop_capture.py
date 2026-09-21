"""Bounded, one-shot Windows Graphics Capture. Cursor capture is explicitly disabled."""
import threading
from io import BytesIO


class DesktopCapture:
    def __init__(self, monitor_index=1):
        self.monitor_index = monitor_index
        self.lock = threading.Lock()
        self.cancelled = threading.Event()
        self.serial = threading.Lock()

    def cancel(self):
        # No callback waits while the simulation lock is held.
        with self.lock:
            self.cancelled.set()

    def __call__(self):
        from windows_capture import WindowsCapture
        from PIL import Image
        if not self.serial.acquire(blocking=False):
            raise RuntimeError("Capture already in progress")
        control = None
        try:
            with self.lock:
                cancelled = threading.Event()
                self.cancelled = cancelled
            ready = threading.Event()
            result = []
            capture = WindowsCapture(cursor_capture=False, monitor_index=self.monitor_index,
                                     minimum_update_interval=100)

            @capture.event
            def on_frame_arrived(frame, capture_control):
                try:
                    if not cancelled.is_set() and not result:
                        rgb = frame.frame_buffer[:, :, [2, 1, 0]].copy()
                        image = Image.fromarray(rgb)
                        image.thumbnail((960, 600))
                        output = BytesIO()
                        image.save(output, "JPEG", quality=70)
                        if not cancelled.is_set():
                            result.append({"bytes": output.getvalue(), "mime_type": "image/jpeg",
                                           "width": image.width, "height": image.height,"gaze_enabled":True})
                finally:
                    ready.set()
                    capture_control.stop()

            @capture.event
            def on_closed():
                ready.set()

            control = capture.start_free_threaded()
            if not ready.wait(2.5) or cancelled.is_set() or not result:
                raise RuntimeError("Desktop capture cancelled or unavailable")
            return result[0]
        except Exception as exc:
            raise RuntimeError("Desktop capture unavailable") from exc
        finally:
            try:
                if control is not None:
                    control.stop()
            finally:
                self.serial.release()
