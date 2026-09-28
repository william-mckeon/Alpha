import unittest
from baby_arcus.foundation_schedule import accumulation,learning_rate,estimate

class ScheduleTests(unittest.TestCase):
    def test_reference_budget_and_boundaries(self):
        self.assertEqual(accumulation(),64)
        self.assertEqual(accumulation(2048,8,64),1)
        with self.assertRaises(ValueError):accumulation(16383)
        c={'updates':2000000,'learning_rate':.003,'warmup_steps':2000,'decay_start':1600000,'decay_steps':400000}
        self.assertAlmostEqual(learning_rate(2000,c),.003)
        self.assertAlmostEqual(learning_rate(1600000,c),.003)
        self.assertEqual(learning_rate(2000000,c),0)
        self.assertGreater(estimate(10,100)['training_only_days'],1000)
