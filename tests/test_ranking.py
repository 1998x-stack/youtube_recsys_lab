import unittest
import numpy as np
from recsys_lab.metrics.ranking import weighted_pairwise_loss

class PairwiseLossTests(unittest.TestCase):
    def test_regular_margins(self):
        result = weighted_pairwise_loss(np.array([2., 0.]), np.array([0., 0.]), np.array([1., 2.]))
        expected = (np.logaddexp(0., -2.) + 2 * np.log(2.)) / 2
        self.assertAlmostEqual(result, expected)

    def test_extreme_negative_margin_not_clipped(self):
        self.assertAlmostEqual(weighted_pairwise_loss(np.array([-1000.]), np.array([0.]), np.array([1.])), 1000.)

    def test_extreme_positive_margin_is_finite(self):
        value = weighted_pairwise_loss(np.array([1000.]), np.array([0.]), np.array([1.]))
        self.assertTrue(np.isfinite(value))
        self.assertGreaterEqual(value, 0.)

    def test_zero_weights(self):
        self.assertEqual(weighted_pairwise_loss(np.array([-1000.]), np.array([0.]), np.array([0.])), 0.)

if __name__ == '__main__':
    unittest.main()
