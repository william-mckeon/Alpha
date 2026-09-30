import unittest
from arcus3.config import read
from arcus3.campaign import validate

class ContextTests(unittest.TestCase):
    def test_backbone_context_bound(self):
        cfg=read('configs/arcus3/backbone_adaptation.json')
        self.assertEqual(validate({**cfg,'max_length':8192})['max_length'],8192)
        with self.assertRaises(ValueError):validate({**cfg,'max_length':8193})
