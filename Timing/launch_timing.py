# Import necessary modules
import os  # Provides functions to interact with the operating system
import glob  # Used to find all the pathnames matching a specified pattern
import xml.etree.ElementTree as ET  # Used to parse XML files
import pandas as pd  # Used for data manipulation and analysis

# Function to parse an XML file and extract relevant data
def parse_xml(file_path):
    tree = ET.parse(file_path)  # Parse the XML file
    root = tree.getroot()  # Get the root element of the XML tree
    
    data = []  # Initialize an empty list to store the data
    # Loop through each rider_info element in the XML
    for rider in root.findall('.//rider_info'):
        # Extract data for each rider
        rider_data = {
            'champ_id': root.find('.//championship').attrib['champ_id'],  # Championship ID
            'event_id': root.find('.//event').attrib['event_id'],  # Event ID
            'session_id': root.find('.//session').attrib['session_id'],  # Session ID
            'circuit_id': root.find('.//circuit').attrib['id'],  # Circuit ID
            'rider_id': rider.attrib['rider_id'],  # Rider ID
            'rider_name': rider.attrib['rider_name'],  # Rider name
            'rider_surname': rider.attrib['rider_surname'],  # Rider surname
            'team_name': rider.attrib['team_name'],  # Team name
            'bike_name': rider.attrib['bike_name'],  # Bike name
            'tyre_name': rider.attrib['tyre_name'],  # Tyre name
        }
        # Loop through each lap element for the rider
        for lap in rider.findall('.//lap'):
            lap_data = rider_data.copy()  # Copy rider data
            # Update lap data
            lap_data.update({
                'lap_num': lap.attrib['num'],  # Lap number
                'lap_time': lap.attrib['time'],  # Lap time
                'lap_speed': lap.attrib['speed'],  # Lap speed
                # Add more fields as necessary
            })
            data.append(lap_data)  # Append the lap data to the list
    
    return data  # Return the collected data

# Main function to execute the script
def main():
    # Find all XML files in the specified directory
    xml_files = glob.glob('G:\\My Drive\\Istruzione\\Coursera\\Statistics\\Excercises\\MotoGP Timing Archive\\XMLs\\RR01 DOHA\\*.xml', recursive=True)
    all_data = []  # Initialize an empty list to store all data
    # Loop through each XML file
    for file in xml_files:
        all_data.extend(parse_xml(file))  # Parse the XML file and extend the data list
    
    # Create a DataFrame from the collected data
    df = pd.DataFrame(all_data)
    # Save the DataFrame to a Parquet file
    df.to_parquet('.\\timing\\output.parquet')

# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()
