"""Exercise the real desktop runtime using temporary identity state and ephemeral ports.

Runs application-level integration checks, not OS mouse automation. --live-capture
uses the current selected display but retains only dimensions/counts, never pixels.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-capture", action="store_true")
    parser.add_argument("--output", default="runs/arcus_desktop/qualification.json")
    args = parser.parse_args()
    from PySide6.QtWidgets import QApplication
    from baby_arcus.desktop import Host, Runtime
    from baby_arcus.transport import Client, RemoteError
    os.environ["ARCUS_BODY_TOOL_TOKEN"] = uuid.uuid4().hex
    app = QApplication.instance() or QApplication([])
    report = {}
    with tempfile.TemporaryDirectory() as root:
        runtime = Runtime(root, 0, 0, 0, 0, 0)
        host = Host(runtime)
        try:
            # Show only the application's own render surface; no synthetic mouse events.
            host.show()
            deadline = time.monotonic()+20
            while time.monotonic()<deadline and host.canvas_rect is None:
                app.processEvents()
                time.sleep(.02)
            assert host.canvas_rect is not None, "Native WebChannel geometry did not arrive"
            report["webchannel_connected"] = True
            tools = Client(runtime.tool_url, os.environ["ARCUS_BODY_TOOL_TOKEN"], timeout=5, attempts=1)
            def observe():
                return tools.request("POST", "/v1/tools/call", {
                    "request_id":uuid.uuid4().hex, "name":"observe_view", "arguments":{}})
            def event(kind, **extra):
                return runtime.desktop.request("POST", "/v1/desktop/event", {
                    "request_id":uuid.uuid4().hex, "kind":kind, **extra})
            pen = observe()
            assert pen["source"] == "playpen" and pen["mime_type"] == "image/png"
            report["playpen_image"] = {key:pen[key] for key in ("source","width","height","cursor_included")}
            identity = runtime.application.world.body.entity_id
            event("pickup")
            assert runtime.application.world.view.held
            event("carry", inside=False)
            outside = event("drop", inside=False)
            host.accept_state(outside)
            app.processEvents()
            assert host.overlay.isVisible()
            report["overlay_shown_outside"] = True
            if args.live_capture:
                frame = observe()
                assert frame["source"] == "desktop" and not frame["cursor_included"]
                report["desktop_image"] = {key:frame[key] for key in ("source","width","height","cursor_included")}
                del frame
            returned = event("return")
            host.accept_state(returned)
            app.processEvents()
            assert not host.overlay.isVisible()
            restricted = observe()
            assert restricted["source"] == "playpen"
            assert restricted["epoch"] > pen["epoch"]
            assert runtime.application.world.body.entity_id == identity
            report["return_restricted_and_identity_preserved"] = True
            with_invalid_source = {"request_id":"forbidden", "name":"observe_view", "arguments":{"source":"desktop"}}
            try:
                tools.request("POST", "/v1/tools/call", with_invalid_source)
                raise AssertionError("Source override was accepted")
            except RemoteError as exc:
                assert exc.status == 400
            report["source_override_rejected"] = True
            def tool_action(action):
                return tools.request("POST","/v1/tools/call",{"request_id":uuid.uuid4().hex,"name":"body_action","arguments":action})
            tool_action({"kind":"eyelids","openness":0})
            try:
                observe()
                raise AssertionError("Closed eyes captured pixels")
            except RemoteError as exc:
                assert exc.status==409
            tool_action({"kind":"joint","joint":"front_left.knee","delta":-.1})
            senses=tools.request("POST","/v1/tools/call",{"request_id":uuid.uuid4().hex,"name":"observe_senses","arguments":{}})
            assert senses["eyelid_openness"]==0 and senses["joint_positions"]["front_left.knee"]<1
            tool_action({"kind":"sleep"})
            runtime.backend.request("POST","/v1/messages",{"request_id":"native-message","sender":"you","text":"Hello Arcus"})
            queued=runtime.backend.request("GET","/v1/messages")["messages"]
            assert queued[0]["status"]=="queued"
            tool_action({"kind":"wake_up"})
            deadline=time.monotonic()+2
            while time.monotonic()<deadline and runtime.application.world.body.sleep_state!="awake":
                app.processEvents();time.sleep(.02)
            assert runtime.backend.request("GET","/v1/messages")["messages"][0]["status"]=="available"
            assert runtime.application.world.body.eyelid_openness==0
            report["closed_eye_senses_and_sleep_message_queue"]=True
            report["physical_mouse_drag_tested"] = False
            report["pixels_saved"] = False
            report["passed"] = True
        finally:
            host.close()
            app.processEvents()
            runtime.close()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
