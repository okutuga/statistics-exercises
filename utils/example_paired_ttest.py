import numpy as np
from statistics_basic import Sample

def lap_time_to_seconds(lap_time_str):
    minutes, seconds = map(float, lap_time_str.split(':'))
    return minutes * 60 + seconds

# Best lap time for every day of winter testing for Marc Marquez
data1_str = ['1:57.606', '1:58.447', '1:57.042', '1:29.184', '1:28.855']
data1 = [lap_time_to_seconds(time) for time in data1_str]

# Best lap time for every day of winter testing for Pecco Bagnaia
data2_str = ['1:58.947', '1:57.652', '1:56.500', '1:30.028', '1:29.378']
data2 = [lap_time_to_seconds(time) for time in data2_str]

# Perform a paired t-test
sample1 = Sample(data1)
sample2 = Sample(data2)
t_statistic, p_value = sample1.paired_ttest(sample2)

# Print the results
print(f't-statistic: {t_statistic:.3f}')
print(f'p-value: {p_value:.3f}')
if p_value < 0.05:
    print('The difference in lap times is statistically significant, and the social media manager of the motogp page is right.')
else:
    print('The difference in lap times is not statistically significant, in spite of what the social media manager of the MotoGP page concludes.')