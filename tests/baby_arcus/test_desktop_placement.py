import unittest
from baby_arcus.desktop_placement import Rect, clamp_position, canvas_pen_rect


class PlacementTests(unittest.TestCase):
    def test_negative_monitor_and_offscreen_recovery(self):
        screen = Rect(-1920, 0, 1920, 1080)
        self.assertEqual(clamp_position(-3000, -10, 180, 155, screen), (-1920, 0))
        self.assertEqual(clamp_position(100, 1200, 180, 155, screen), (-180, 925))

    def test_canvas_mapping_in_logical_pixels(self):
        pen = canvas_pen_rect(Rect(50, 30, 550, 370))
        self.assertEqual(pen, Rect(100, 115, 450, 245))
        self.assertTrue(pen.contains(101, 116))
        self.assertFalse(pen.contains(551, 200))
