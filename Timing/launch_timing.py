import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    session_info = {
        'champ_id': root.find('.//championship').attrib['champ_id'],
        'event_id': root.find('.//event').attrib['event_id'],
        'session_id': root.find('.//session').attrib['session_id'],
        'circuit_id': root.find('.//circuit').attrib['id'],
        'analysis_info': ET.tostring(root.find('.//analysis_info'), encoding='unicode')
    }
    
    return session_info

def main():
    # xml_files = glob.glob('g:/My Drive/Istruzione/Coursera/Statistics/statistics-exercises/Timing/XMLs/**/*.xml', recursive=True)
    xml_files = glob.glob('g:/My Drive/Istruzione/Coursera/Statistics/statistics-exercises/Timing/XMLs/RR01 DOHA/*.xml', recursive=True)
    sessions = []

    for xml_file in xml_files:
        session_info = parse_xml(xml_file)
        sessions.append(session_info)
    
    df = pd.DataFrame(sessions)
    print(df)
    # df.to_parquet('g:/My Drive/Istruzione/Coursera/Statistics/statistics-exercises/sessions.parquet')

if __name__ == "__main__":
    main()
