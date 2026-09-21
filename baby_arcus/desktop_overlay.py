"""Transparent body window. Pointer position is never part of model observations."""
from pathlib import Path
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QWidget, QMenu


class ArcusOverlay(QWidget):
    def __init__(self, pickup, return_to_pen, body_action):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool |
                         Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowTitle("Arcus Alpha body")
        self.resize(180, 155)
        self.sprite = QPixmap(str(Path(__file__).parent/"web"/"arcus-body.png"))
        self.lying_sprite = QPixmap(str(Path(__file__).parent/"web"/"arcus-lying.png"))
        self.sitting_sprite = QPixmap(str(Path(__file__).parent/"web"/"arcus-sitting.png"))
        self.body = {"height": 1, "facing": "right"}
        self.pickup, self.return_to_pen, self.body_action = pickup, return_to_pen, body_action
        # Transparent margins do not intercept clicks over the user's applications.
        self.setMask(self.sprite.scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation).mask())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        pose=self.body.get("visual_pose",{"standing_weight":1,"height":129,"width":160})
        height=pose["height"];width=pose["width"];weight=pose["standing_weight"]
        frame=QPixmap(self.size());frame.fill(Qt.GlobalColor.transparent)
        layer=QPainter(frame)
        if self.body["facing"] == "left":
            layer.translate(self.width(), 0);layer.scale(-1, 1)
        rect=QRectF((self.width()-width)/2,self.height()-height,width,height)
        layer.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        for sprite,opacity in ((self.lying_sprite,pose.get("lying_weight",1-weight)),(self.sprite,weight),(self.sitting_sprite,pose.get("sitting_weight",0))):
            if opacity>0:
                layer.setOpacity(opacity);layer.drawPixmap(rect,sprite,QRectF(sprite.rect()))
        layer.end();painter.drawPixmap(0,0,frame);painter.end()
        self.setMask(frame.mask())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.pickup()
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            menu = QMenu()
            menu.addAction("Return to playpen", self.return_to_pen)
            menu.addAction("Stand up", lambda: self.body_action("stand"))
            menu.addAction("Lie down", lambda: self.body_action("lie"))
            menu.addAction("Sleep", lambda: self.body_action("sleep"))
            menu.addAction("Wake up", lambda: self.body_action("wake_up"))
            menu.addAction("Open eyes", lambda: self.body_action({"kind":"eyelids","openness":1}))
            menu.addAction("Close eyes", lambda: self.body_action({"kind":"eyelids","openness":0}))
            menu.exec(event.globalPosition().toPoint())
