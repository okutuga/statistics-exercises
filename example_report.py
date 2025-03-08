import numpy as np
import pandas as pd
from business_intelligence_viewer import BikeKPIReport

# Generate random data
np.random.seed(42)
bikes = np.arange(1, 9)
runs = np.arange(1, 11)
sessions = ['Session A', 'Session B', 'Session C']
data = []

for bike in bikes:
    for run in runs:
        for session in sessions:
            kpi = np.random.uniform(100, 130)
            data.append([bike, f'Run {run} - {session}', kpi])

df = pd.DataFrame(data, columns=['Bike', 'Run', 'KPI'])

# Set threshold
threshold = 120

# Create the report
report = BikeKPIReport(df, threshold)
report.generate_html_report('bike_kpi_report.html')
