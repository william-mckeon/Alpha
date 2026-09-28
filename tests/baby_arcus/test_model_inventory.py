import unittest
import torch
from baby_arcus.model_inventory import inventory

class InventoryTests(unittest.TestCase):
    def test_adapter_adds_no_parameters(self):
        from tests.baby_arcus.foundation_fixtures import tiny_model
        from baby_arcus.nanotron_adapter import ArcusForTraining
        from baby_arcus.model_inventory import assert_inventory
        model=tiny_model();count=sum(p.numel() for p in model.parameters())
        assert_inventory(model,count)
        self.assertEqual(inventory(ArcusForTraining(model))['unique_parameters'],count)
        with self.assertRaises(ValueError):assert_inventory(model,count+1)

    def test_shared_weights_count_once(self):
        model=torch.nn.Module();model.first=torch.nn.Linear(3,4);model.second=model.first
        result=inventory(model)
        self.assertEqual(result['unique_parameters'],16)
        self.assertEqual(len(result['parameters'][0]['names']),2)
        self.assertEqual(result['unique_weight_bytes'],64)
        self.assertTrue(any(e['kind']=='shared_parameter_alias' for e in result['connections']))
        self.assertEqual({e['target'] for e in result['connections'] if e['kind']=='registered_child'},{'first','second'})
