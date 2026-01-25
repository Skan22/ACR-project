import json 
import numpy as np
from pathlib import Path
import os
import librosa
import soundfile as sf
from collections import defaultdict

# Config
SAMPLE_RATE = 22050  # Standard for audio analysis

# Roots for chords giving a total of 24 classes of maj/min
ROOTS = ["A", "Bb", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
# Enharmonic mapping (flats to sharps)
ENHARMONIC_MAP = {
    "A#": "Bb", "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#",
    "a#": "Bb", "db": "C#", "eb": "D#", "gb": "F#", "ab": "G#"
}

ROOT_TO_IDX = {r: i for i, r in enumerate(ROOTS)}
IDX_TO_ROOT = {i: r for i, r in enumerate(ROOTS)}

# Base paths
BASE_DIR = Path(__file__).parent.parent / "GuitarSet"
ANNOTATION_DIR = BASE_DIR / "annotation"
AUDIO_DIR = BASE_DIR / "audio_mono-mic"
OUTPUT_DIR = Path(__file__).parent.parent / "GuitarSet_Data"


def normalize_root(root):
    """Convert enharmonic equivalents (flats) to sharps."""
    return ENHARMONIC_MAP.get(root, root)


def simplify_chord(chord):
    """Map chord to major/minor only. Returns None for non-maj/min chords."""
    if chord == "N" or not chord:
        return None
    try:
        root, suffix = chord.split(":")
        root = normalize_root(root)
        
        if root not in ROOTS:
            return None
            
        if suffix in ["maj", "7", "maj7", "maj6", "maj9"]:
            return f"{root}"  # Major chord folder name (e.g., "A", "C#")
        elif suffix in ["min", "min7", "min6", "min9", "hdim7", "dim", "dim7"]:
            return f"{root}m"  # Minor chord folder name (e.g., "Am", "C#m")
        else:
            return None  # Skip other chord types
            
    except ValueError:
        return None


def get_chord_segments(annot_file_path):
    """Parse JAMS file and return chord segments with timing info."""
    with open(annot_file_path, "r") as f:
        jams_data = json.load(f)

    segments = []
    
    for annotation in jams_data["annotations"]:
        if annotation["namespace"] == "chord":
            for data in annotation["data"]:
                chord_label = simplify_chord(data['value'])
                if chord_label is not None:  # Only keep major/minor
                    segments.append({
                        'start': data['time'],
                        'duration': data['duration'],
                        'chord': chord_label
                    })
            break  # Only process first chord annotation
    
    return segments


def extract_and_save_samples(audio_path, segments, output_base_dir, file_prefix):
    """Extract whole chord segments and save them with chord label in filename."""
    # Load audio file
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    sample_idx = 0
    for seg in segments:
        start_time = seg['start']
        duration = seg['duration']
        chord = seg['chord']
        
        # Extract the whole chord segment
        start_sample = int(start_time * SAMPLE_RATE)
        end_sample = int((start_time + duration) * SAMPLE_RATE)
        FILE_HEALTH_CHECK= end_sample <= len(y) and start_sample < end_sample
        FILE_LENGTH_CHECK= duration>=2.0
        FIXED_LEGTH= int((start_time + 2) * SAMPLE_RATE)  # Currently taking 2second samples only 
        if FILE_HEALTH_CHECK and FILE_LENGTH_CHECK:
            segment_audio = y[start_sample:FIXED_LEGTH] 
            
            # Create output directory for this chord class
            chord_dir = output_base_dir / chord
            chord_dir.mkdir(parents=True, exist_ok=True)
            
            # Save with chord label in filename
            # Format: {chord}_{sample_idx}_{original_file_prefix}.wav
            output_filename = f"{chord}_{sample_idx:04d}_{file_prefix}.wav"
            output_path = chord_dir / output_filename
            
            sf.write(output_path, segment_audio, SAMPLE_RATE)
            sample_idx += 1
    


def process_split(split_name):
    """Process a single data split (Train, Test, or Validation)."""
    print(f"\n{'='*50}")
    print(f"Processing {split_name} split...")
    print(f"{'='*50}")
    
    annot_dir = ANNOTATION_DIR / split_name
    audio_dir = AUDIO_DIR / split_name
    output_dir = OUTPUT_DIR / split_name


    
    # Process each annotation file
    annot_files = sorted(annot_dir.glob("*.jams"))
    print(f"Found {len(annot_files)} annotation files")
    
    for annot_file in annot_files:
        # Construct corresponding audio filename
        # Annotation: 00_BN1-129-Eb_comp.jams -> Audio: 00_BN1-129-Eb_comp_mic.wav
        audio_filename = annot_file.stem + "_mic.wav"
        audio_path = audio_dir / audio_filename

        
        # Get chord segments from annotation
        segments = get_chord_segments(annot_file)
        
        # Extract and save audio samples
        file_prefix = annot_file.stem
        extract_and_save_samples(
            audio_path, segments, output_dir, file_prefix, 
        )

        print(f"  Processed {annot_file.name}")


def count_samples_per_class():
    """Count the number of samples in each chord class folder after processing."""
    print("\n" + "=" * 50)
    print("Sample Counts Per Class")
    print("=" * 50)
    
    for split in ["Train", "Test", "Validation"]:
        split_dir = OUTPUT_DIR / split
        
        if not split_dir.exists():
            print(f"\n{split}: Directory not found")
            continue
        
        print(f"\n{split}:")
        print("-" * 40)
        
        # Get all chord folders
        chord_folders = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        total = 0
        for chord_dir in chord_folders:
            count = len(list(chord_dir.glob("*.wav")))
            total += count
            print(f"  {chord_dir.name:8s}: {count:>6,} samples")
        
        print("-" * 40)
        print(f"  {'TOTAL':8s}: {total:>6,} samples")
        print(f"  Classes: {len(chord_folders)}")


# -------- Main Program Execution ---------------

if __name__ == "__main__":
    print("GuitarSet Chord Audio Extraction (Whole Segments)")
    print("=" * 50)
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # Process each split
    for split in ["Train", "Test", "Validation"]:
        process_split(split)
    
    # Count samples per class after processing
    count_samples_per_class()