#  py -m unittest test_statistics_basic.py 

import unittest
import numpy as np
from statistics_basic import Sample, DistributionNormal, DistributionBinomial, DistributionStudent  # Import your classes
from scipy import stats

class TestDistributionNormal(unittest.TestCase):

    def setUp(self):
        """
        Fathers’ heights follow the normal curve with x_mean = 68.3 in and s = 1.8 in.
        What percentage of fathers have heights between 67.4 in and 71.9 in?
        What is the 30th percentile of the fathers’ heights?
        """
        self.mean = 68.3
        self.std = 1.8
        self.dist = DistributionNormal(self.mean, self.std)

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

class TestSample(unittest.TestCase):
    """
    Unit tests for the Sample class.
    """

    def setUp(self):
        self.data = [1.23, 1.18, 1.22, 1.22]
        self.sample = Sample(self.data)

    def test_mean(self):
        self.assertAlmostEqual(self.sample.mean, np.mean(self.data))

    def test_std(self):
        self.assertAlmostEqual(self.sample.std, np.std(self.data, ddof=1))

    def test_standard_error(self):
        self.assertAlmostEqual(self.sample.standard_error, np.std(self.data, ddof=1) / np.sqrt(len(self.data)))

    def test_standard_error_monte_carlo(self):
        se = self.sample.standard_error_monte_carlo(num_simulations=1000)
        self.assertIsNotNone(se)
        self.assertGreater(se, 0)

class TestDistributionBinomial(unittest.TestCase):
    """
    Unit tests for the DistributionBinomial class.
    """

    def test_probability(self):
        # You play an online game 10 times. Each time there are three possible outcomes:
        # P(win a big prize) = 10%, P(win a small prize) = 20%, P(win nothing) = 70%.
        # What is P(win two small prizes) ?
        
        # Number of trials
        n = 10
        # Probability of success
        p = 0.2
        # Number of successes
        k = 2
        dist = DistributionBinomial(n, p)
        expected_probability = 0.302 # 45 * 0.2 ** 2 * 0.8 ** 8
        self.assertAlmostEqual(dist.binomial_probability(k), expected_probability, delta=0.0005)

    def test_area_to_the_left(self):
        """
        In the previous example, we had p = P(win a small prize) = 0.2.
        Play n = 50 times. What is P(at most 12 small prizes) ?
        """
        n = 50
        p = 0.2
        k = 12
        dist = DistributionBinomial(n, p)
        # Calculate the area to the left of k
        expected_area = sum(dist.binomial_probability(i) for i in range(0, k + 1))
        actual_area = dist.get_percentile_from_sum(k)
        self.assertAlmostEqual(actual_area, expected_area, delta=0.1)

    def test_area_to_the_left_precise(self):
        """
        In the previous example, we had p = P(win a small prize) = 0.2.
        Play n = 50 times. What is P(at most 12 small prizes) ?
        """
        n = 500
        p = 0.2
        k = 120
        dist = DistributionBinomial(n, p)
        # Calculate the area to the left of k
        expected_area = sum(dist.binomial_probability(i) for i in range(0, k + 1))
        actual_area = dist.get_percentile_from_sum(k)
        self.assertAlmostEqual(actual_area, expected_area, delta=0.0005)

class TestDistributionStudent(unittest.TestCase):
    """
    Unit tests for the DistributionStudent class.
    """

    def setUp(self):
        """
        Example: A sample of 10 students' test scores with a mean of 75 and a standard deviation of 10.
        """
        self.mean = 75
        self.std = 10
        self.n = 10
        self.dist = DistributionStudent(self.mean, self.std, self.n)

    def test_get_tscore_from_value(self):
        value = 80
        expected_tscore = 1.5811
        self.assertAlmostEqual(self.dist.get_tscore_from_value(value), expected_tscore, delta=0.05)

    def test_get_value_from_tscore(self):
        tscore = 1.5811
        expected_value = 80
        self.assertAlmostEqual(self.dist.get_value_from_tscore(tscore), expected_value, delta=0.05)

    def test_get_percentile_from_tscore(self):
        tscore = 1.5811
        expected_percentile = 0.925
        self.assertAlmostEqual(self.dist.get_percentile_from_tscore(tscore), expected_percentile, delta=0.005)

    def test_get_tscore_from_percentile(self):
        percentile = 0.925
        expected_tscore = 1.5811
        self.assertAlmostEqual(self.dist.get_tscore_from_percentile(percentile), expected_tscore, delta=0.05)

if __name__ == '__main__':
    unittest.main()
