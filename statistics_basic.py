import numpy as np
from scipy import stats, special
import matplotlib.pyplot as plt 

class DistributionNormal:
    """
    Represents a normal distribution and provides methods to standardize values.
    """
    def __init__(self, mean, std):
        """
        Initializes a StandardDistribution object.

        Args:
            mean: The mean of the distribution.
            std: The standard deviation of the distribution.
        """
        self.mean = mean
        self.std = std

    def get_zscore_from_value(self, value):
        """
        Standardizes a given value using the distribution's mean and standard deviation.

        Args:
            value: The value to be standardized.

        Returns:
            The standardized value (z-score) as a float, or None if the standard deviation is None.
        """
        if self.std is None:
            return None  # Cannot standardize if stdev is undefined
        return (value - self.mean) / self.std

    def get_value_from_zscore(self, zscore):
        """
        Converts a z-score back to the original value using the provided mean and standard deviation.

        Args:
            zscore: The z-score to be converted.
            mean: The mean of the data.
            std: The standard deviation of the data.

        Returns:
            The original value as a float.
        """
        return zscore * self.std + self.mean

    def get_percentile_from_zscore(self, zscore):
        """
        Returns the percentile rank for a given z-score.

        Args:
            zscore: The z-score to be converted.

        Returns:
            The percentile rank as a float, dimensionless.
        """
        return stats.norm.cdf(zscore)

    def get_percentile_from_value(self, value):
        """
        Returns the percentile rank for a given value using the provided mean and standard deviation.

        Args:
            value: The value to be converted.

        Returns:
            The percentile rank as a float.
        """
        zscore = self.get_zscore_from_value(value)
        return self.get_percentile_from_zscore(zscore)
    
    def get_zscore_from_percentile(self, percentile):
        """
        Returns the z-score corresponding to a given percentile rank.

        Args:
            percentile: The percentile rank to be converted.

        Returns:
            The z-score as a float.
        """
        return stats.norm.ppf(percentile)
    
    def get_value_from_percentile(self, percentile):
        """
        Returns the value corresponding to a given percentile rank.

        Args:
            percentile: The percentile rank to be converted.

        Returns:
            The value as a float.
        """
        zscore = self.get_zscore_from_percentile(percentile)
        return self.get_value_from_zscore(zscore)
    
    def get_area_between_two_zscores(self, zscore1, zscore2):
        """
        Returns the percentage of values between two given z-scores.

        Args:
            zscore1: The lower z-score.
            zscore2: The upper z-score.

        Returns:
            The percentage of values between the two z-scores as a float.
        """
        return self.get_percentile_from_zscore(zscore2) - self.get_percentile_from_zscore(zscore1)
    
    def get_area_between_two_values(self, value1, value2):
        """
        Returns the percentage of values between two given values.

        Args:
            value1: The lower value.
            value2: The upper value.

        Returns:
            The percentage of values between the two values as a float.
        """
        zscore1 = self.get_zscore_from_value(value1)
        zscore2 = self.get_zscore_from_value(value2)
        return self.get_area_between_two_zscores(zscore1, zscore2)

class Sample:
    """
    Represents a sample of numerical data that follows the normal distribution
    and provides methods for basic statistical calculations using NumPy and SciPy.
    """

    def __init__(self, data):
        """
        Initializes a SampleBinomial object.

        Args:
            data: A list, tuple, or NumPy array of numerical data.
                  Raises TypeError if input is not a list, tuple, or NumPy
                  array, or if elements are not numeric.
        """
        if not isinstance(data, (list, tuple, np.ndarray)):
            raise TypeError("Input data must be a list, tuple, or NumPy array.")

        # Check for numeric types *before* converting to a NumPy array
        if isinstance(data, (list, tuple)):
            if not all(isinstance(x, (int, float)) for x in data):
                raise TypeError("All data elements must be numeric.")
        elif isinstance(data, np.ndarray):
             if not np.issubdtype(data.dtype, np.number):
                raise TypeError("All data elements must be numeric.")

        # Convert to NumPy array for efficiency
        self.data = np.array(data, dtype=np.float64)
        self.n = len(self.data)

        # Handles edge cases (empty or single-element arrays)
        if self.n < 2:
            raise TypeError("Sample must have at least 2 elements.")
        
        # The arithmetic mean (average) using NumPy.
        self.mean = np.mean(self.data)

        # The sample standard deviation using NumPy with Bessel's correction.
        # ddof=1 for Bessel's correction.
        self.std = np.std(self.data, ddof=1)

        # Create a distribution object for the sample.
        if set(self.data) <= {0, 1}:
            self.approx = DistributionBinomial(self.n, self.mean)
        elif self.n < 500:
            self.approx = DistributionNormal(self.mean, self.std) # DistributionStudent(self.mean, self.std, self.n)
        else:
            self.approx = DistributionNormal(self.mean, self.std)

    def __str__(self):
        """
        Returns string representation of the SampleBinomial data.
        """
        return f"SampleBinomial Data: {self.data}"

    @property
    def standard_error(self):
        """
        Calculates the standard error of the mean using SciPy.

        Returns:
           The standard error as a float, or None if the sample has
           fewer than 2 elements.
        """
        if self.n < 2:
            return None  # Standard error undefined for n < 2
        return stats.sem(self.data)

    def standard_error_monte_carlo(self, num_simulations=1000):
        """
        Estimates the standard error of the mean using Monte Carlo simulation.

        Args:
            num_simulations: The number of Monte Carlo simulations to run.
                             Defaults to 10000. Must be a positive integer.

        Returns:
            The estimated standard error as a float, or None if the sample
            has fewer than 2 elements.
        """
        if self.n < 2:
            return None  # Standard error is undefined for n < 2

        if not isinstance(num_simulations, int) or num_simulations <= 0:
            raise ValueError("num_simulations must be a positive integer.")


        sample_means = np.zeros(num_simulations)
        for i in range(num_simulations):
            # Resample with replacement
            resampled_data = np.random.choice(self.data, size=self.n, replace=True)
            sample_means[i] = np.mean(resampled_data)

        # Standard deviation of the sample means is the Monte Carlo SE
        return np.std(sample_means)

    def histogram_probability_data(self):
        """
        Generates a histogram of the distribution of the sample.
        """
        plt.hist(self.data, bins='auto', alpha=0.7, rwidth=0.85)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Empirical Histogram of the Observed Data')
        plt.show()

class DistributionBinomial(DistributionNormal):
    """
    Represents a binomial distribution and provides methods to calculate binomial probabilities.
    """
    def __init__(self, n, p):
        """
        Initializes a DistributionBinomial object.

        Args:
            n: The number of experiments.
            p: The probability of success in each experiment.
        """
        self.n = n
        self.p = p
        # Calculate mean and standard deviation
        self.mean = p
        self.std = np.sqrt(1 / n * p * (1 - p))

    @property
    def probability_success(self):
        """
        Calculates the probability of success in a single experiment.

        Returns:
            The probability of success as a float.
        """
        return self.p
    
    @property
    def probability_failure(self):
        """
        Calculates the probability of failure in a single experiment.

        Returns:
            The probability of failure as a float.
        """
        return 1 - self.p

    def binomial_coefficient(self, k):
        """
        Calculates the binomial coefficient for k successes in n experiments.

        Args:
            k: The number of successes.

        Returns:
            The binomial coefficient as an integer.
        """
        return special.comb(self.n, k)

    def binomial_probability(self, k):
        """
        Calculates the binomial probability for k successes in n experiments.

        Args:
            k: The number of successes.

        Returns:
            The binomial probability as a float.
        """
        coeff = self.binomial_coefficient(k)
        return coeff * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def binomial_probability_all(self):
        """
        Generates a histogram of the binomial probabilities for all possible
        numbers of successes (from 0 to n).

        Returns:
            probabilities: The binomial probability as a float array.
            k_values: The number of successes as an integer array.
        """
        k_values = np.arange(0, self.n + 1)
        probabilities = [self.binomial_probability(k) for k in k_values]
        return probabilities, k_values

    def get_percentile_from_sum(self, sum):
        """
        Returns the percentile rank for a given value using the provided mean and standard deviation.

        Args:
            sum: The value to be converted.

        Returns:
            The percentile rank as a float.
        """
        value = sum / self.n
        zscore = self.get_zscore_from_value(value)
        return self.get_percentile_from_zscore(zscore)
    
    def get_sum_from_percentile(self, percentile):
        """
        Returns the value corresponding to a given percentile rank.

        Args:
            percentile: The percentile rank to be converted.

        Returns:
            The value as a float.
        """
        zscore = self.get_zscore_from_percentile(percentile)
        value = self.get_value_from_zscore(zscore)
        return round(value * self.n)

    def histogram_probability_statistic(self):
        """
        Generates a histogram of the binomial probabilities for all possible
        numbers of successes (from 0 to n).

        Displays the histogram using matplotlib.
        """
        [probabilities, k_values] = self.binomial_probability_all()

        plt.bar(k_values, probabilities)
        plt.xlabel('Number of Successes')
        plt.ylabel('Probability')
        plt.title('Probability Histogram For The Statistic')
        plt.show()

    def histogram_probability_data(self):
        """
        Generates a histogram of the binomial probabilities for a single
        trial.
        """
        x_axis_values = [0, 1]
        y_axis_values = [self.probability_failure, self.probability_success]
        plt.bar(x_axis_values, y_axis_values)
        plt.xlabel('Outcome')
        plt.xticks([0, 1], ['Failure', 'Success'])
        plt.ylabel('Probability')
        plt.title('Probability Histogram of the Data')
        plt.show()

