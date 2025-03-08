import pandas as pd
import xml.etree.ElementTree as ET
from business_intelligence_viewer import BikeKPIReport
import numpy as np

def xml_to_parquet(xml_file, parquet_file):
    """
    Converts an XML file to a Parquet file.

    Args:
        xml_file: The path to the input XML file.
        parquet_file: The path to the output Parquet file.
    """
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract data from XML
    data = []
    bike_colors = {
        "Ducati": "red",
        "KTM": "orange",
        "Yamaha": "blue",
        "Honda": "yellow",
        "Aprilia": "green"
    }
    for rider_info in root.findall(".//rider_info"):
        rider_data = {
            "rider_number": rider_info.get("rider_number"),
            "rider_name": rider_info.get("rider_name"),
            "rider_surname": rider_info.get("rider_surname"),
            "team_name": rider_info.get("team_name"),
            "bike_name": rider_info.get("bike_name"),
            "tyre_name": rider_info.get("tyre_name"),
            "bike_color": bike_colors.get(rider_info.get("bike_name"), "black")
        }
        for lap in rider_info.findall(".//lap"):
            lap_time_dms = int(lap.get("time"))
            lap_speed_mph = int(lap.get("speed"))  # meters per hour
            lap_data = {
                "run_lap": lap.get("run_lap"),
                "lap_time": lap_time_dms / 10000 if lap_time_dms >= 0 else np.nan,  # Convert decimilliseconds to seconds, set to NaN if negative
                "lap_speed": lap_speed_mph / 3600,  # Convert meters per hour to meters per second
            }
            combined_data = {**rider_data, **lap_data}
            data.append(combined_data)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Convert the DataFrame to a Parquet file
    df.to_parquet(parquet_file, engine='pyarrow')

def read_parquet(parquet_file):
    """
    Reads a Parquet file and returns its contents as a DataFrame.

    Args:
        parquet_file: The path to the Parquet file.

    Returns:
        A DataFrame containing the data from the Parquet file.
    """
    df = pd.read_parquet(parquet_file, engine='pyarrow')
    return df

# Example usage
xml_file = './Analysis for THA MotoGP RAC.xml'
parquet_file = './TimingDB.parquet'
xml_to_parquet(xml_file, parquet_file)

# Read the Parquet file
parquet_file = './TimingDB.parquet'
df = read_parquet(parquet_file)

# Define plots
plots = [
    {
        "x_data": "run_lap",
        "y_data": "lap_time",
        "title": "Lap Time vs Lap Number",
        "x_label": "Lap Number",
        "y_label": "Lap Time (seconds)"
    },
    {
        "x_data": "run_lap",
        "y_data": "lap_speed",
        "title": "Maximum Speed vs Lap Number",
        "x_label": "Lap Number",
        "y_label": "Maximum Speed (m/s)"
    }
]

# Generate the report
report = BikeKPIReport(df)
report.generate_html_report('bike_kpi_report.html', plots)

# Display the first few rows of the DataFrame
print(df.head())
print(df.columns)