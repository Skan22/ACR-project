import pandas as pd
import numpy as np
from pathlib import Path
import os
import librosa
import soundfile as sf

SAMPLE_RATE = 22050  # Standard for audio analysis
HOP_LENGTH = 2048      # ~100ms per frame at 22050 Hz
FRAME_DURATION = HOP_LENGTH / SAMPLE_RATE  # Duration of each frame in seconds
ROOTS = ["A","Bb","B","C","C#","D","D#","E","F","F#","G","G#"] # Roots for chords giving a total of 24 classes of maj/min
MAJOR_CHORDS = [f"{r}" for r in ROOTS]
MINOR_CHORDS = [f"{r}m" for r in ROOTS]
ALL_CHORDS = MAJOR_CHORDS+MINOR_CHORDS
ROOT_TO_IDX = {r: i for i, r in enumerate(ROOTS)}
IDX_TO_ROOT = {i: r for i, r in enumerate(ROOTS)}

MAJOR_CHORD_TO_IDX = {r: i for i, r in enumerate(MAJOR_CHORDS)}
IDX_TO_MAJOR_CHORD = {i: r for i, r in enumerate(MAJOR_CHORDS)}

MINOR_CHORD_TO_IDX = {r: i for i, r in enumerate(MINOR_CHORDS)}
IDX_TO_MINOR_CHORD = {i: r for i, r in enumerate(MINOR_CHORDS)}

SAMPLE_RATE = 22050

DATA_PATH = Path("./Data")
SEMITONE_SHIFT = [-2 ,-1, 1, 2]
def check_folder_exists(dir,foldername):
    for folder in Path(dir).iterdir():
        if folder.name == foldername:
            return True
    return False 

#Data Augmentation 
def calc_transposed_chord(chord,n_semitones):
    if chord in MAJOR_CHORDS:
        new_chord = IDX_TO_MAJOR_CHORD[(MAJOR_CHORD_TO_IDX[chord]+n_semitones)%12]
    elif chord in MINOR_CHORDS:
        new_chord = IDX_TO_MINOR_CHORD[(MINOR_CHORD_TO_IDX[chord]+n_semitones)%12]
    else :
        print(chord)
        raise(ValueError)
    return new_chord


def transpose_sound_file(directory :Path ,wav_file_name : str,n_semitones: int) :
    #Load initial File
    chord = wav_file_name.split("_")[0]
    y, _ = librosa.load(directory/chord/wav_file_name, sr=SAMPLE_RATE, mono=False)
    y_shift = librosa.effects.pitch_shift(y=y, sr=SAMPLE_RATE, n_steps=n_semitones)

    new_chord = calc_transposed_chord(chord,n_semitones)
    #Save new chord .wav file
    print(f"Saving augmented {new_chord}")
    sf.write(f"./{directory}/{new_chord}/{new_chord}"+"_aug_"+wav_file_name, y_shift.T, SAMPLE_RATE)


train_dir= Path("./Data/Training")
for folder in train_dir.iterdir():
    for shift in SEMITONE_SHIFT:
        new_chord= calc_transposed_chord(folder.name,shift)
        if check_folder_exists(train_dir,new_chord)==True:
            continue
        else :
            new_folder_path= train_dir/new_chord
            print("Creating folder ")
            new_folder_path.mkdir(parents=True, exist_ok=True)
            print(f"Created folder: {new_folder_path}")
            count=0
            for sound_file in folder.iterdir():
                count+=1
                transpose_sound_file(train_dir,sound_file.name,shift)
                print(count)

