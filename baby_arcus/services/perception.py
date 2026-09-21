"""Source-gated visual observation service; no frames on disk or in session logs."""
import base64
import time
from baby_arcus.contracts import fields
from baby_arcus.gaze import crop_frame


class PerceptionApplication:
    def __init__(self, read_state, playpen_capture, desktop_capture):
        self.read_state = read_state
        self.playpen_capture = playpen_capture
        self.desktop_capture = desktop_capture

    def __call__(self, method, path, body):
        if method != "POST" or path != "/v1/observe":
            raise KeyError(path)
        fields(body, ())
        before = self.read_state()
        if before["arcus"].get("sleep_state", "awake") != "awake" or before["arcus"].get("eyelid_openness",1) <= 0:
            return 409, {"error": "Arcus must be awake with open eyes to observe", "sleep_state": before["arcus"]["sleep_state"]}
        view = before["view"]
        stamp = (view["scope_id"], view["epoch"])
        started = time.monotonic()
        try:
            frame = (self.desktop_capture() if view["source"] == "desktop"
                     else self.playpen_capture(before))
        except (OSError, RuntimeError, TimeoutError) as exc:
            return 503, {"error": "Visual source unavailable", "source": view["source"]}
        after = self.read_state()["view"]
        if stamp != (after["scope_id"], after["epoch"]):
            return 409, {"error": "View changed during capture; discard and observe again"}
        if time.monotonic()-started > 3:
            return 503, {"error": "Capture expired"}
        if not frame or len(frame["bytes"]) > 500_000:
            return 503, {"error": "Capture unavailable or too large"}
        # Adapters explicitly mark encoded image frames for gaze processing.
        if frame.get("gaze_enabled"):
            frame=crop_frame(frame,before["arcus"])
        final=self.read_state()
        if stamp != (final["view"]["scope_id"],final["view"]["epoch"]):
            return 409, {"error":"View changed during image processing"}
        return 200, {"scope_id": stamp[0], "epoch": stamp[1], "source": view["source"],
                     "cursor_included": False, "captured_at": time.time(),
                     "mime_type": frame["mime_type"], "width": frame["width"], "height": frame["height"],
                     "camera":frame.get("camera"),
                     "image_base64": base64.b64encode(frame["bytes"]).decode("ascii")}
