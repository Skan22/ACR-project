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

def find_shortest_audio_file(data_dir: Path = DATA_PATH):
    """Find the audio file with the shortest duration in the Data directory."""
    shortest_duration = float('inf')
    shortest_file = None
    
    for subset in data_dir.iterdir():  # Training, Test
        if not subset.is_dir():
            continue
        for chord_folder in subset.iterdir():  # A, Am, B, etc.
            if not chord_folder.is_dir():
                continue
            for audio_file in chord_folder.iterdir():
                if audio_file.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']:
                    try:
                        duration = librosa.get_duration(path=audio_file)
                        if duration < shortest_duration:
                            shortest_duration = duration
                            shortest_file = audio_file
                            print(f"New shortest: {audio_file} - {duration:.3f}s")
                    except Exception as e:
                        print(f"Error loading {audio_file}: {e}")
    
    print(f"\nShortest file: {shortest_file}")
    print(f"Duration: {shortest_duration:.3f} seconds")
    return shortest_file, shortest_duration


def normalize_audio_files(data_dir: Path = DATA_PATH, target_duration: float = 2.0, overwrite: bool = True):
    """
    Trim or pad all audio files to a target duration, normalize amplitude, and apply windowing.
    
    Args:
        data_dir: Path to the Data directory
        target_duration: Target duration in seconds (default 2.0)
        overwrite: If True, overwrite original files; if False, save with '_normalized' suffix
    """
    target_samples = int(target_duration * SAMPLE_RATE)
    processed_count = 0
    trimmed_count = 0
    padded_count = 0
    
    # Create a Hann window for smooth fade-in/fade-out
    window = np.hanning(target_samples)
    
    for subset in data_dir.iterdir():  # Training, Test
        if not subset.is_dir():
            continue
        for chord_folder in subset.iterdir():  # A, Am, B, etc.
            if not chord_folder.is_dir():
                continue
            for audio_file in chord_folder.iterdir():
                if audio_file.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']:
                    try:
                        # Load audio file
                        y, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)
                        original_samples = len(y)
                        
                        if original_samples > target_samples:
                            # Trim to target duration
                            y_adjusted = y[:target_samples]
                            trimmed_count += 1
                        elif original_samples < target_samples:
                            # Pad with zeros to target duration
                            padding = target_samples - original_samples
                            y_adjusted = np.pad(y, (0, padding), mode='constant', constant_values=0)
                            padded_count += 1
                        else:
                            # Already at target duration
                            y_adjusted = y
                        
                        # Normalize amplitude (peak normalization to -1 to 1 range)
                        max_amplitude = np.max(np.abs(y_adjusted))
                        if max_amplitude > 0:
                            y_normalized = y_adjusted / max_amplitude
                        else:
                            y_normalized = y_adjusted
                        
                        # Apply windowing function (Hann window)
                        y_windowed = y_normalized * window
                        
                        # Save the processed audio
                        if overwrite:
                            output_path = audio_file.with_suffix('.wav')
                        else:
                            output_path = audio_file.with_stem(audio_file.stem + '_normalized').with_suffix('.wav')
                        
                        sf.write(output_path, y_windowed, SAMPLE_RATE)
                        processed_count += 1
                        
                        if processed_count % 100 == 0:
                            print(f"Processed {processed_count} files...")
                            
                    except Exception as e:
                        print(f"Error processing {audio_file}: {e}")
    
    print(f"\nNormalization complete!")
    print(f"Total processed: {processed_count}")
    print(f"Trimmed (>{target_duration}s): {trimmed_count}")
    print(f"Padded (<{target_duration}s): {padded_count}")
    print(f"Unchanged: {processed_count - trimmed_count - padded_count}")
    return processed_count, trimmed_count, padded_count



if __name__ == "__main__":
    train_dir= Path("./Data/Training")
    test_dir= Path("./Data/Test")
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
                for sound_file in folder.iterdir():
                    transpose_sound_file(train_dir,sound_file.name,shift)
        
    for folder in test_dir.iterdir():
        for shift in SEMITONE_SHIFT:
            new_chord= calc_transposed_chord(folder.name,shift)
            if check_folder_exists(test_dir,new_chord)==True:
                continue
            else :
                new_folder_path= test_dir/new_chord
                print("Creating folder ")
                new_folder_path.mkdir(parents=True, exist_ok=True)
                print(f"Created folder: {new_folder_path}")
                for sound_file in folder.iterdir():
                    transpose_sound_file(test_dir,sound_file.name,shift)

    shortest_file , shortest_duration = find_shortest_audio_file()
    normalize_audio_files(target_duration=2, overwrite=True)  # Custom duration, keeps originals

    


