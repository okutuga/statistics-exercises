import numpy as np
from statistics_basic import SampleBinomial, SampleNormal

import matplotlib.pyplot as plt

# Parameters for the binomial distribution
n = 10  # number of trials
p = 0.5  # probability of success

# Generate binomial distribution data
data = np.random.binomial(n, p, 1000)

# Plot histogram
plt.hist(data, bins=range(n+2), edgecolor='black', align='left')
plt.title('Binomial Distribution Histogram')
plt.xlabel('Number of Successes')
plt.ylabel('Frequency')
plt.xticks(range(n+1))
# plt.show()

# Generate binomial distribution data
data = np.random.choice([0, 1], size=1000)

# Create an instance of SampleBinomial with the generated data
sample = SampleBinomial(data)
# sample.histogram_probability_data()

print(sample.mean)
print(sample.std)
print(sample.standard_error)
# print(sample.standard_error_monte_carlo(num_simulations=1000))

sample = SampleNormal(data)
# sample.histogram_probability_data()

print(sample.mean)
print(sample.std)
print(sample.standard_error)