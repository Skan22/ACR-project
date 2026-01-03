import json 
import pandas as pd
import numpy as np
from pathlib import Path
import os
SAMPLE_RATE = 22050  # Standard for audio analysis
HOP_LENGTH = 2048      # ~100ms per frame at 22050 Hz
FRAME_DURATION = HOP_LENGTH / SAMPLE_RATE  # Duration of each frame in seconds



#Cleaning : Only keeping Comp tracks for ACR 
def delete_solo_tracks(folder_path):
    for folder in folder_path.iterdir():
        for jams in folder.iterdir():
            if jams.name.find("solo")!=-1:
                print(f"deleting jams{jams.name}")
                jams.unlink(missing_ok=True)
delete_solo_tracks(Path("./annotation"))
delete_solo_tracks(Path("./audio_mono-mic"))

#Framing time Stamps
def parse_file(annot_file_path,name):
    with open(annot_file_path ,"r") as f : 
        jams_data = json.load(f)

    chord_data =[]

    
    for annotation in jams_data["annotations"]:
        if annotation["namespace"] == "chord":
            for data in annotation["data"]:
                for i in range(int(data['time']*SAMPLE_RATE),int(((data['time']+data['duration'])*SAMPLE_RATE)-HOP_LENGTH),HOP_LENGTH):
                    chord_data.append({
                        'name' : name,
                        'start': i/SAMPLE_RATE,
                        'end': i/SAMPLE_RATE + FRAME_DURATION,
                        'chord': data['value']
                    })
                
            break

    return chord_data 
for folder in Path("./annotation").iterdir():
    Total_data =[]
    for file in folder.iterdir():
        name = file.stem +"_mic.wav"
        data = parse_file(file,name)
        Total_data+=data
    df = pd.DataFrame(Total_data)
    os.makedirs(os.path.dirname(f"./Data/{folder.name}/"), exist_ok=True)
    df.to_csv(f"./Data/{folder.name}/{folder.name}_data.csv")
