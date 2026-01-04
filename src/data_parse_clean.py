import json 
import pandas as pd
import numpy as np
from pathlib import Path
import os
SAMPLE_RATE = 22050  # Standard for audio analysis
HOP_LENGTH = 2048      # ~100ms per frame at 22050 Hz
FRAME_DURATION = HOP_LENGTH / SAMPLE_RATE  # Duration of each frame in seconds
ROOTS = ["A","A#","B","C","C#","D","D#","E","F","F#","G","G#"] # Roots for chords giving a total of 24 classes of maj/min


#Cleaning : Only keeping Comp tracks for ACR 
def delete_solo_tracks(folder_path):
    for folder in folder_path.iterdir():
        for jams in folder.iterdir():
            if jams.name.find("solo")!=-1:
                print(f"deleting jams{jams.name}")
                jams.unlink(missing_ok=True)

#Chord Mapping into Major/Minor 
def simplify_chord(chord):
    try:
        root, suffix = chord.split(":")
        if suffix in ["maj", "7", "maj7"]:
            return f"{root}:maj"
        elif suffix in ["min", "hdim7", "dim", "dim7", "min7"]:
            return f"{root}:min"
        #to modify in the future in case of additions
        else:
            return f"{root}:{suffix}" 
            
    except ValueError:
        print("error")
        # Handles cases where there is no colon in the string
        return chord

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
                        'chord': simplify_chord(data['value'])
                    })
                
            break

    return chord_data 





#-------- Main Program Exec ---------------

print("Cleaning Solo tracks")
delete_solo_tracks(Path("./annotation"))
delete_solo_tracks(Path("./audio_mono-mic"))

print("Framing TimeStamps")
for folder in Path("./annotation").iterdir():
    Total_data =[]
    for file in folder.iterdir():
        name = file.stem +"_mic.wav"
        data = parse_file(file,name)
        Total_data+=data
    df = pd.DataFrame(Total_data)
    os.makedirs(os.path.dirname(f"./Data/{folder.name}/"), exist_ok=True)
    df.to_csv(f"./Data/{folder.name}/Framed_{folder.name}_data.csv")