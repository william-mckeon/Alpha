"""Geometry in Qt device-independent screen coordinates."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    def contains(self, x, y):
        return self.x <= x < self.x+self.width and self.y <= y < self.y+self.height


def clamp_position(x, y, width, height, screen):
    return (max(screen.x, min(screen.x+max(0, screen.width-width), x)),
            max(screen.y, min(screen.y+max(0, screen.height-height), y)))


def canvas_pen_rect(canvas, logical_width=1100, logical_height=740):
    return Rect(canvas.x+100/logical_width*canvas.width,
                canvas.y+170/logical_height*canvas.height,
                900/logical_width*canvas.width, 490/logical_height*canvas.height)
