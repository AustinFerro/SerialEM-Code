# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 21:27:36 2024

@author: Austin Ferro
"""

def main():
    
    from pathlib import Path
    import pandas as pd
    import json
    import uuid
    
    # Load CSV file
    
    
    baseDirectory = 'D:/MicronsEMdata'
    
    csv_file = 'D:/MicronsEMdata/OPC12_RanPoints.csv'
    opc = '12'
    
    df = pd.read_csv(csv_file, header=None, names=['x', 'y', 'z'])
    
    # Define the JSON structure
    json_structure = {
        "type": "annotation",
        "source": {
            "url": "local://annotations",
            "transform": {
                "outputDimensions": {
                    "x": [4e-9, "m"],
                    "y": [4e-9, "m"],
                    "z": [4e-8, "m"]
                }
            }
        },
        "tool": "annotatePoint",
        "tab": "annotations",
        "annotations": [],
        "name": f"{opc}_ranPoints"
    }
    
    # Add annotations to the JSON structure
    for index, row in df.iterrows():
        annotation = {
            "point": [row['x'], row['y'], row['z']],
            "type": "point",
            "id": str(uuid.uuid4()).replace("-", "")
        }
        json_structure['annotations'].append(annotation)
    
    # Save to JSON file
    json_file = Path(f"{baseDirectory}/{opc}_ranPoints.json")
    with open(json_file, 'w') as f:
        json.dump(json_structure, f, indent=2)
    
    print(f"Converted CSV data has been saved to {json_file}")
  
        
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
