import unittest
from scripts.run_alpha_phase2_39000 import next_count


class SupervisorTests(unittest.TestCase):
    def test_exact_target_and_every_evaluation_boundary(self):
        current=37000; visited=[]
        while current<39000:
            count=next_count(current)
            self.assertTrue(1<=count<=64)
            current+=count;visited.append(current)
        self.assertEqual(current,39000)
        for value in (37013,37130,37650,38300,39000):self.assertIn(value,visited)
        self.assertEqual(next_count(39000),0)
        for value in (36999,39001,37000.0):
            with self.assertRaises(ValueError):next_count(value)
