import unittest
from baby_arcus.shared_identity_context import identity_confidence
from baby_arcus.shared_object_memory import assignments


class IdentityConfidenceTests(unittest.TestCase):
    def test_ambiguity_abstains_without_double_penalizing_distinct_identity(self):
        # Independent ambiguity decision cannot silently raise association's gate.
        self.assertEqual(assignments([[identity_confidence(.92, .15)]], 1, 1), [0])
        self.assertEqual(assignments([[identity_confidence(.99, .5)]], 1, 1), [None])
        self.assertEqual(assignments([[identity_confidence(.99, 1.)]], 1, 1), [None])
        self.assertEqual(assignments([[identity_confidence(.89, 0.)]], 1, 1), [None])
        # Competing identities still require the original margin.
        self.assertEqual(assignments([[identity_confidence(.99, .1), .95]], 1, 2), [None])
        for invalid in (float('nan'), float('inf'), -1., 2.):
            with self.assertRaises(ValueError):identity_confidence(.99, invalid)
