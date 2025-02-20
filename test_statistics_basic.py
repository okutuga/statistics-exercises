#  py -m unittest test_statistics_basic.py 

import unittest
import numpy as np
from statistics_basic import NormalSample, NormalDistribution  # Import your NormalSample and NormalDistribution classes
from scipy import stats

class TestNormalDistribution(unittest.TestCase):

    def setUp(self):
        self.mean = 68.3
        self.std = 1.8
        self.dist = NormalDistribution(self.mean, self.std)

    def test_get_zscore_from_value(self):
        value = 67.4
        expected_zscore = -0.5
        self.assertAlmostEqual(self.dist.get_zscore_from_value(value), expected_zscore, delta=0.05)

    def test_get_value_from_zscore(self):
        zscore = 2.0
        expected_value = 71.9
        self.assertAlmostEqual(self.dist.get_value_from_zscore(zscore), expected_value, delta=0.05)

    def test_get_percentile_from_value(self):
        value = 71.9
        expected_percentile = 0.977
        self.assertAlmostEqual(self.dist.get_percentile_from_value(value), expected_percentile, delta=0.0005)

    def test_get_value_from_percentile(self):
        percentile = 0.309
        expected_value = 67.4
        self.assertAlmostEqual(self.dist.get_value_from_percentile(percentile), expected_value, delta=0.05)

    def test_get_area_between_two_values(self):
        value1 = 67.4
        value2 = 71.9
        expected_area = 0.669
        self.assertAlmostEqual(self.dist.get_area_between_two_values(value1, value2), expected_area, delta=0.0005)

class TestNormalSample(unittest.TestCase):
    """
    Unit tests for the NormalSample class.
    """

    def setUp(self):
        self.data = [1.23, 1.18, 1.22, 1.22]
        self.sample = NormalSample(self.data)

    def test_mean(self):
        self.assertAlmostEqual(self.sample.mean, np.mean(self.data))

    def test_std(self):
        self.assertAlmostEqual(self.sample.std, np.std(self.data, ddof=1))

    def test_standard_error(self):
        self.assertAlmostEqual(self.sample.standard_error, np.std(self.data, ddof=1) / np.sqrt(len(self.data)))

    def test_get_zscore_from_value(self):
        expected_standardized = (np.array(self.data) - np.mean(self.data)) / np.std(self.data, ddof=1)
        np.testing.assert_allclose(self.sample.zscore, expected_standardized)

    def test_get_value_from_zscore(self):
        zscore = 1.0
        expected_value = zscore * np.std(self.data, ddof=1) + np.mean(self.data)
        self.assertAlmostEqual(self.sample.get_value_from_zscore(zscore), expected_value)

    def test_standard_error_monte_carlo(self):
        se = self.sample.standard_error_monte_carlo(num_simulations=1000)
        self.assertIsNotNone(se)
        self.assertGreater(se, 0)

    def test_get_percentile_from_value(self):
        value = 1.20
        zscore = (value - np.mean(self.data)) / np.std(self.data, ddof=1)
        expected_percentile = stats.norm.cdf(zscore)
        self.assertAlmostEqual(self.sample.get_percentile_from_value(value), expected_percentile)

if __name__ == '__main__':
    unittest.main()
