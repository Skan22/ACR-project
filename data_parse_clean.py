import json 
import pandas as pd
import numpy as np
from pathlib import Path



#Cleaning : Only keeping Comp tracks for ACR 
def delete_solo_tracks(folder_path):
    for folder in folder_path.iterdir():
        for jams in folder.iterdir():
            if jams.name.find("solo")!=-1:
                print(f"deleting jams{jams.name}")
                jams.unlink(missing_ok=True)
delete_solo_tracks(Path("./annotation"))
delete_solo_tracks(Path("./audio_mono-mic"))




