import math
import unittest
from baby_arcus.evaluation_metrics import language_metrics


class MetricsTests(unittest.TestCase):
    def test_weighted_not_mean_of_windows(self):
        result=language_metrics([{'nll':1.,'target_tokens':1},{'nll':3.,'target_tokens':3}])
        self.assertEqual(result['nll'],2.5)
        self.assertAlmostEqual(result['perplexity'],math.exp(2.5))

    def test_empty_and_nonfinite_rejected(self):
        for rows in ([],[{'nll':float('nan'),'target_tokens':1}],[{'nll':1,'target_tokens':0}]):
            with self.assertRaises(ValueError):language_metrics(rows)
