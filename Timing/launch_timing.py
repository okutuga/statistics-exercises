import pandas as pd  # Importing pandas library for data manipulation
import xml.etree.ElementTree as ET  # Importing ElementTree for parsing XML files
from business_intelligence_viewer import TimingReport  # Importing custom report generation module
import numpy as np  # Importing numpy for numerical operations
import os  # Importing os for operating system dependent functionality
import glob  # Importing glob for file pattern matching

def main():
    # xml_files = glob.glob('G:\\My Drive\\Istruzione\\Coursera\\Statistics\\Excercises\\MotoGP Timing Archive\\XMLs\\RR01 DOHA\\*.xml', recursive=True)  # Get all XML files in the directory
    xml_files = ['./timing/Analysis for THA MotoGP RAC.xml']
    parquet_file = './timing/TimingDB.parquet'
    html_file = './timing/timing_report.html'
    
    all_data = []  # Initialize a list to store all data
    for file in xml_files:  # Iterate over each XML file
        all_data.extend(parse_xml(file))  # Parse the XML file and extend all_data with the parsed data
    
    write_parquet(all_data, parquet_file)
    df = pd.DataFrame(all_data)  # Convert all_data to a DataFrame
    df.to_parquet()  # Convert the DataFrame to a Parquet file
    
    # Read the Parquet file
    df = read_parquet(parquet_file)  # Read the Parquet file into a DataFrame

    # Define plots
    plots = [
        {
            "x_data": "lap_run_lap",  # Define x-axis data
            "y_data": "lap_time",  # Define y-axis data
            "title": "Lap Time vs Lap Number",  # Define plot title
            "x_label": "Lap Number",  # Define x-axis label
            "y_label": "Lap Time (seconds)"  # Define y-axis label
        },
        {
            "x_data": "lap_run_lap",  # Define x-axis data
            "y_data": "lap_speed",  # Define y-axis data
            "title": "Maximum Speed vs Lap Number",  # Define plot title
            "x_label": "Lap Number",  # Define x-axis label
            "y_label": "Maximum Speed (m/s)"  # Define y-axis label
        }
    ]
    report = TimingReport(df)  # Create a TimingReport object
    report.generate_html_report(html_file, plots)  # Generate an HTML report with the plots

def parse_xml(xml_file):
    """
    Converts an XML file to a Parquet file.

    Args:
        xml_file: The path to the input XML file.
    """
    tree = ET.parse(xml_file)  # Parse the XML file
    root = tree.getroot()  # Get the root element of the XML
    
    # Find the elements in the XML
    """
    for child in root:
        for grandson in child:
            print(grandson.tag, grandson.attrib)
        print(child.tag, child.attrib)
    """
    header = root.find("header")
    championship = header.find("championship")
    event = header.find("event")
    session = header.find("session")
    circuit = header.find("circuit")

    # Rename some keys for unambiguity
    file_data = extract_attributes(root, "file_")
    championship_data = extract_attributes(championship, "champ_")
    event_data = extract_attributes(event, "event_")
    session_data = extract_attributes(session, "session_")
    circuit_data = extract_attributes(circuit, "circuit_")

    # Combine all championship data into a dictionary
    meta_data = {**file_data, **championship_data, **event_data, **session_data, **circuit_data}

    data = []  # Initialize a list to store data

    for rider_info in root.findall(".//rider_info"):  # Find all rider_info elements in the XML
        # Extract rider data
        rider_data = extract_attributes(rider_info, "rider_")
        runs_info = rider_info.find("runs_info")  # Find runs_info element for the rider
        runs_data = extract_attributes(runs_info, "rider_")  # Extract runs data for the rider
        rider_data = {**rider_data, **runs_data} 

        for run in rider_info.findall(".//run"):  # Find all run elements for the rider
            run_data = extract_attributes(run, "run_")  # Extract runs data for the rider

            for lap in run.findall(".//lap"):  # Find all lap elements for the rider
                
                """
                lap_time_dms = int(lap.get("time"))  # Get lap time in decimilliseconds
                lap_speed_mph = int(lap.get("speed"))  # Get lap speed in meters per hour
                lap_data = {  # Create a dictionary for lap data
                    "run_lap": lap.get("run_lap"),  # Get run lap number
                    "lap_time": lap_time_dms / 10000 if lap_time_dms >= 0 else np.nan,  # Convert lap time to seconds, set to NaN if negative
                    "lap_speed": lap_speed_mph / 3600,  # Convert lap speed to meters per second
                }
                """
                lap_data = extract_attributes(lap, "lap_")  # Extract lap data
                # Substitute the prefix of the key with "sector_" if it is a digit
                lap_data = {k if not k[-1].isdigit() else "sector_" + k[4:]: v for k, v in lap_data.items()}
                
                # Unit conversion
                lap_data["lap_time"] = int(lap_data["lap_time"]) / 10000 if int(lap_data["lap_time"]) >= 0 else np.nan  # Convert lap time to seconds, set to NaN if negative
                lap_data["lap_speed"] = int(lap_data["lap_speed"]) / 3600  # Convert lap speed to meters per second

                # Combine rider data and lap data
                combined_data = {**meta_data, **rider_data, **run_data, **lap_data}  # Combine rider data and lap data
                data.append(combined_data)  # Append combined data to the data list
    return data  # Return the data list

def extract_attributes(data_origin, prefix):
    """
    Renames keys in a dictionary to make them unambiguous.
    """
    data_buffer = {}  # Initialize a dictionary to store the renamed keys
    for key in data_origin.attrib:
        if key.startswith(prefix):
            data_buffer[key] = data_origin.attrib[key]
        else:
            data_buffer[prefix + key] = data_origin.attrib[key]
    return data_buffer

def write_parquet(data, parquet_file):
    """
    Converts a list of dictionaries to a Parquet file.
    """
    df = pd.DataFrame(data)  # Convert data list to a DataFrame
    df.to_parquet(parquet_file, engine='pyarrow')  # Convert the DataFrame to a Parquet file

def read_parquet(parquet_file):
    """
    Reads a Parquet file and returns its contents as a DataFrame.

    Args:
        parquet_file: The path to the Parquet file.

    Returns:
        A DataFrame containing the data from the Parquet file.
    """
    df = pd.read_parquet(parquet_file, engine='pyarrow')  # Read the Parquet file into a DataFrame
    return df  # Return the DataFrame

"""
Script entry point.
"""
if __name__ == "__main__":
    main()  # Call the main function