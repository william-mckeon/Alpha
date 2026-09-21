"""Windows playpen host with a movable body and separately gated perception service."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import secrets
import sys
import threading
import uuid

from PySide6.QtCore import QObject, Signal, Slot, QTimer, QUrl, QFile, QIODevice, Qt, QPoint
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineScript
from PySide6.QtWebChannel import QWebChannel

from baby_arcus.contracts import fields, ContractError
from baby_arcus.desktop_capture import DesktopCapture
from baby_arcus.desktop_overlay import ArcusOverlay
from baby_arcus.desktop_placement import Rect, canvas_pen_rect, clamp_position
from baby_arcus.playpen_capture import capture_playpen
from baby_arcus.services.perception import PerceptionApplication
from baby_arcus.services.playroom import PlayroomApplication, PlayroomViewer, viewer_server
from baby_arcus.body_tools import BodyToolsApplication
from baby_arcus.transport import Client, RemoteError, serve


class DesktopEvents:
    def __init__(self, application):
        self.application = application

    def __call__(self, method, path, body):
        if method == "POST" and path == "/v1/desktop/event":
            return self.application.desktop_event(body)
        raise KeyError(path)


class Runtime:
    def __init__(self, root, port=8890, simulation_port=8891, tool_port=8892,
                 desktop_port=8893, perception_port=8894, monitor=1):
        self.application = PlayroomApplication(root)
        from baby_arcus.live_interaction_policy import LiveInteractionPolicy as LiveBodyPolicy
        from baby_arcus.body_controller_config import configuration
        project=Path(__file__).resolve().parent.parent
        self.application.policy=LiveBodyPolicy(self.application,project/".venv/Scripts/python.exe",
                                               workdir=project,**configuration(project))
        from baby_arcus.language_runtime import LanguageRuntime
        self.application.language=LanguageRuntime(self.application,project/".venv/Scripts/python.exe",project)
        from baby_arcus.visual_runtime import VisualRuntime
        self.application.visual=VisualRuntime(self.application,project/".venv/Scripts/python.exe",project)
        from baby_arcus.rest_runtime import RestRuntime
        self.application.rest=RestRuntime(self.application,project)
        from baby_arcus.curiosity_runtime import CuriosityRuntime
        self.application.curiosity=CuriosityRuntime(self.application,project)
        from baby_arcus.shared_runtime import SharedRuntime
        self.application.shared=SharedRuntime(self.application,project/'.venv/Scripts/python.exe',project)
        self.capture = DesktopCapture(monitor)
        self.application.on_view_change = self.capture.cancel
        self.servers = []
        self.started = []
        self.pool = ThreadPoolExecutor(max_workers=1)
        try:
            backend_token, desktop_token, perception_token = [secrets.token_urlsafe(32) for _ in range(3)]
            tool_token = os.environ.get("ARCUS_BODY_TOOL_TOKEN") or secrets.token_urlsafe(32)
            backend = serve("127.0.0.1", simulation_port, self.application, backend_token,audit=self.application.audit)
            self.servers.append(backend)
            self.backend = Client(f"http://127.0.0.1:{backend.server_port}", backend_token, timeout=2)
            perception = PerceptionApplication(lambda: self.application("GET", "/v1/state", None)[1],
                                               capture_playpen, self.capture)
            vision = serve("127.0.0.1", perception_port, perception, perception_token,audit=self.application.audit)
            self.servers.append(vision)
            vision_client = Client(f"http://127.0.0.1:{vision.server_port}", perception_token, timeout=4, attempts=1)

            def observe():
                try:
                    return 200, vision_client.request("POST", "/v1/observe", {})
                except RemoteError as exc:
                    return exc.status, {"error": "View unavailable or changed; observe again"}

            tools_server = serve("127.0.0.1", tool_port, BodyToolsApplication(self.application, observe), tool_token,audit=self.application.audit)
            self.servers.append(tools_server)
            desktop = serve("127.0.0.1", desktop_port, DesktopEvents(self.application), desktop_token,audit=self.application.audit)
            self.servers.append(desktop)
            self.desktop = Client(f"http://127.0.0.1:{desktop.server_port}", desktop_token, timeout=2)
            viewer = viewer_server(port, PlayroomViewer(self.backend.base_url, backend_token),audit=self.application.audit)
            self.servers.append(viewer)
            self.url = f"http://127.0.0.1:{viewer.server_port}"
            self.tool_url = f"http://127.0.0.1:{tools_server.server_port}"
            for server in self.servers:
                threading.Thread(target=server.serve_forever, daemon=True).start()
                self.started.append(server)
            threading.Thread(target=self.application.run_clock, daemon=True).start()
        except Exception:
            self.close()
            raise

    def close(self):
        self.application.policy.close()
        self.capture.cancel()
        self.application.stop.set()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.application.desktop_event({"request_id": uuid.uuid4().hex, "kind": "return"})
        for server in self.started:
            server.shutdown()
        for server in self.servers:
            server.server_close()
        self.application.close()


class LocalPage(QWebEnginePage):
    def __init__(self, url, parent):
        super().__init__(parent)
        self.allowed = QUrl(url)

    def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
        return (url.scheme(), url.host(), url.port()) == (
            self.allowed.scheme(), self.allowed.host(), self.allowed.port())


class Bridge(QObject):
    def __init__(self, host):
        super().__init__(host)
        self.host = host

    @Slot(str)
    def geometry(self, raw):
        try:
            value = json.loads(raw)
            fields(value, ("x", "y", "width", "height"))
            if not all(type(v) in (float, int) and math.isfinite(v) and abs(v) < 20000 for v in value.values()):
                return
            if value["width"] <= 0 or value["height"] <= 0:
                return
            self.host.canvas_rect = Rect(**value)
        except (ValueError, TypeError, ContractError):
            return

    @Slot()
    def pickup(self):
        if QGuiApplication.mouseButtons() & Qt.MouseButton.LeftButton:
            self.host.pickup()

    @Slot()
    def returnToPen(self):
        self.host.return_to_pen()


class Host(QMainWindow):
    received = Signal(object)
    failed = Signal(str)

    def __init__(self, runtime, monitor=1):
        super().__init__()
        self.runtime = runtime
        self.state = None
        self.dragging = False
        self.inside = True
        self.pending = False
        self.closed = False
        self.canvas_rect = None
        screens = QGuiApplication.screens()
        if not 1 <= monitor <= len(screens):
            raise ValueError("Monitor index is outside the available screen list")
        self.screen = screens[monitor-1]
        self.setWindowTitle("Arcus Alpha · Playpen")
        self.resize(1100, 850)
        self.web = QWebEngineView(self)
        self.page = LocalPage(runtime.url, self.web)
        self.web.setPage(self.page)
        self.setCentralWidget(self.web)
        self.bridge = Bridge(self)
        channel = QWebChannel(self.page)
        channel.registerObject("arcusDesktop", self.bridge)
        self.page.setWebChannel(channel)
        self.channel = channel
        source = QFile(":/qtwebchannel/qwebchannel.js")
        if not source.open(QIODevice.OpenModeFlag.ReadOnly):
            raise RuntimeError("Qt WebChannel resource unavailable")
        script = QWebEngineScript()
        script.setName("arcus-local-desktop-channel")
        script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
        script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        script.setSourceCode(bytes(source.readAll()).decode("utf-8"))
        self.page.scripts().insert(script)
        self.overlay = ArcusOverlay(self.pickup, self.return_to_pen, self.body_action)
        self.received.connect(self.accept_state)
        self.failed.connect(self.fail)
        self.web.setUrl(QUrl(runtime.url))
        self.clock = QTimer(self)
        self.clock.timeout.connect(self.drag_tick)
        self.clock.start(16)
        self.poller = QTimer(self)
        self.poller.timeout.connect(self.poll)
        self.poller.start(300)

    def submit(self, work):
        if self.closed:
            return
        future = self.runtime.pool.submit(work)
        def done(result):
            try:
                self.received.emit(result.result())
            except Exception as exc:
                self.failed.emit(str(exc))
        future.add_done_callback(done)

    def send_event(self, kind, **extra):
        command = {"request_id": uuid.uuid4().hex, "kind": kind, **extra}
        self.submit(lambda: self.runtime.desktop.request("POST", "/v1/desktop/event", command))

    def body_action(self, kind):
        command = {"request_id": uuid.uuid4().hex, "source": "human", "action": kind if isinstance(kind,dict) else {"kind": kind}}
        self.submit(lambda: self.runtime.backend.request("POST", "/v1/action", command))

    def poll(self):
        if self.pending or self.closed:
            return
        self.pending = True
        def read():
            self.runtime.desktop.request("POST", "/v1/desktop/event", {
                "request_id": uuid.uuid4().hex, "kind": "heartbeat"})
            return {"state": self.runtime.backend.request("GET", "/v1/state")}
        self.submit(read)

    def accept_state(self, result):
        self.pending = False
        state = result["state"]
        if self.state is not None:
            old, new = self.state["view"], state["view"]
            if old["scope_id"] == new["scope_id"] and (new["epoch"], new["sequence"]) < (old["epoch"], old["sequence"]):
                return
        self.state = state
        self.overlay.body = state["arcus"]
        self.overlay.update()
        view = state["view"]
        self.statusBar().showMessage("View: " + view["source"] + " · " + view["event"].replace("_", " ") +
                                     " · cursor excluded · model " + state.get("policy",{}).get("status","idle"))
        if view["event"] == "desktop_connection_lost":
            self.dragging = False
        if not self.dragging:
            self.overlay.setVisible(view["region"] == "desktop" or view["held"])

    def fail(self, message):
        if self.runtime.application.audit:
            self.runtime.application.audit.emit("desktop.disconnected",{"reason":"request_failed"},durable=True)
        self.pending = False
        self.dragging = False
        self.overlay.hide()
        self.runtime.capture.cancel()
        self.statusBar().showMessage("Disconnected: desktop view expires within 3 seconds. " + message)

    def pen_rect(self):
        if self.canvas_rect is None or not self.isVisible() or self.isMinimized():
            return None
        origin = self.web.mapToGlobal(QPoint(0, 0))
        r = self.canvas_rect
        return canvas_pen_rect(Rect(r.x+origin.x(), r.y+origin.y(), r.width, r.height))

    def pickup(self):
        if self.dragging or self.state is None:
            return
        self.dragging = True
        self.inside = self.state["view"]["region"] == "playpen"
        self.overlay.show()
        self.send_event("pickup")
        self.drag_tick()

    def drag_tick(self):
        if not self.dragging:
            return
        cursor = QCursor.pos()
        screen = self.screen.availableGeometry()
        x, y = clamp_position(cursor.x()-90, cursor.y()-75, 180, 155,
                              Rect(screen.x(), screen.y(), screen.width(), screen.height()))
        self.overlay.move(round(x), round(y))
        pen = self.pen_rect()
        inside = bool(pen and pen.contains(cursor.x(), cursor.y()))
        if inside != self.inside:
            self.inside = inside
            self.send_event("carry", inside=inside)
        if not QGuiApplication.mouseButtons() & Qt.MouseButton.LeftButton:
            self.dragging = False
            self.send_event("drop", inside=inside)
            if inside:
                self.overlay.hide()

    def return_to_pen(self):
        self.dragging = False
        self.overlay.hide()
        self.runtime.capture.cancel()
        self.send_event("return")

    def closeEvent(self, event):
        if self.runtime.application.audit:
            self.runtime.application.audit.emit("desktop.window_closed",{},durable=True)
        self.dragging = False
        self.clock.stop()
        self.poller.stop()
        self.overlay.close()
        self.runtime.capture.cancel()
        self.closed = True
        event.accept()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", default="runs/arcus_playroom/entity-state")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--simulation-port", type=int, default=8891)
    parser.add_argument("--tool-port", type=int, default=8892)
    parser.add_argument("--desktop-port", type=int, default=8893)
    parser.add_argument("--perception-port", type=int, default=8894)
    parser.add_argument("--monitor", type=int, default=1)
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    runtime = None
    try:
        runtime = Runtime(args.state_root, args.port, args.simulation_port, args.tool_port,
                          args.desktop_port, args.perception_port, args.monitor)
        host = Host(runtime, args.monitor)
        host.show()
        print(json.dumps({"url": runtime.url, "tools": runtime.tool_url, "model_connected": False}), flush=True)
        return app.exec()
    except Exception as exc:
        QMessageBox.critical(None, "Arcus could not start", str(exc))
        return 1
    finally:
        if runtime:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
