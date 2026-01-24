import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from nnAudio.features import CQT

# Constants (matching preprocessing module)
SAMPLE_RATE = 22050
ROOTS = ["A", "Bb", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
MAJOR_CHORDS = [f"{r}" for r in ROOTS]
MINOR_CHORDS = [f"{r}m" for r in ROOTS]
ALL_CHORDS = MAJOR_CHORDS + MINOR_CHORDS
CHORD_TO_IDX = {chord: i for i, chord in enumerate(ALL_CHORDS)}
IDX_TO_CHORD = {i: chord for i, chord in enumerate(ALL_CHORDS)}
NUM_CLASSES = len(ALL_CHORDS)  # 24 classes


class ChordCQTDataset(Dataset):
    """
    PyTorch Dataset for chord classification using Constant Q Transform (CQT).
    
    Loads data from multiple sources in the Data directory:
    - Data/Kaggle_data/{split}/
    - Data/GuitarSet_Data/{split}/
    
    Args:
        data_dir: Path to the base Data directory (e.g., './Data')
        split: Data split to use ('Training' or 'Test')
        n_bins: Number of frequency bins for CQT (default: 72 for 6 octaves)
        hop_length: Hop length for CQT computation
        bins_per_octave: Number of bins per octave (default: 12 for semitone resolution)
        fmin: Minimum frequency (default: C2 ~65.4 Hz)
        transform: Optional transform to apply to the CQT output
        use_hpss: Whether to apply Harmonic-Percussive Source Separation before CQT
        use_kaggle: Whether to load data from Kaggle_data directory
        use_guitarset: Whether to load data from GuitarSet_Data directory
    """
    
    def __init__(
        self,
        data_dir: str | Path,
        split: str ,
        n_bins: int = 72,
        hop_length: int = 512,
        bins_per_octave: int = 12,
        fmin: Optional[float] = None,
        transform: Optional[callable] = None,
        use_hpss: bool = False,
        use_kaggle: bool = True,
        use_guitarset: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.n_bins = n_bins
        self.hop_length = hop_length
        self.bins_per_octave = bins_per_octave
        self.fmin = fmin if fmin is not None else librosa.note_to_hz('C2')
        self.transform = transform
        self.use_hpss = use_hpss
        self.use_kaggle = use_kaggle
        self.use_guitarset = use_guitarset
        
        # Define source directories
        self.kaggle_dir = self.data_dir / "Kaggle_data" / split
        self.guitarset_dir = self.data_dir / "GuitarSet_Data" / split
        
        # Collect all audio files and their labels
        self.samples: List[Tuple[Path, int]] = []
        
        # Initialize nnAudio CQT layer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cqt_layer = CQT(
            sr=SAMPLE_RATE,
            hop_length=self.hop_length,
            fmin=self.fmin,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            output_format='Magnitude',
        ).to(self.device)
        
        self._load_samples()

    def _compute_cqt(self, audio_path: Path) -> torch.Tensor:
        """Compute the Constant Q Transform of an audio file using torchaudio and nnAudio."""
        # Load audio using torchaudio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Apply HPSS to extract harmonic content (using librosa, then convert back to tensor) # TO be changed with custom pytorch function 
        if self.use_hpss:
            waveform_np = waveform.squeeze(0).numpy()
            y_harmonic, _ = librosa.effects.hpss(waveform_np)
            waveform = torch.tensor(y_harmonic, dtype=torch.float32).unsqueeze(0)
        
        # Move to device and compute CQT using nnAudio
        waveform = waveform.to(self.device)
        
        # nnAudio CQT expects shape (batch, samples), output is (batch, n_bins, time_frames)
        with torch.no_grad():
            cqt_mag = self.cqt_layer(waveform)  # Already magnitude from output_format='Magnitude'
        
        # Convert to dB scale
        cqt_db = torchaudio.transforms.AmplitudeToDB(stype='amplitude', top_db=80)(cqt_mag)
        
        # Move back to CPU for storage and return with shape (1, n_bins, time_frames)
        cqt_tensor = cqt_db.cpu()
        
        return cqt_tensor

    def _load_from_directory(self, data_dir: Path, source_name: str):
        """Load samples from a specific directory."""
        count = 0
        for chord_folder in data_dir.iterdir():
            if not chord_folder.is_dir():
                continue
                
            chord_name = chord_folder.name
            if chord_name not in CHORD_TO_IDX:
                print(f"Warning: Unknown chord folder '{chord_name}', skipping...")
                continue
            label = CHORD_TO_IDX[chord_name]
            print(f"Computing CQT for: {chord_folder} ({source_name})")
            for audio_file in chord_folder.iterdir():
                
                if audio_file.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']:
                    self.samples.append((self._compute_cqt(audio_file), label))
                    count += 1

        print(f"Loaded {count} samples from {data_dir} ({source_name})")



    def _load_samples(self):
        """Load all audio file paths and their corresponding labels."""
        # Load from Kaggle data directory if enabled
        if self.use_kaggle:
            if self.kaggle_dir.exists():
                self._load_from_directory(self.kaggle_dir, "Kaggle_data")
            else:
                print(f"Warning: Kaggle data directory not found: {self.kaggle_dir}")
        
        # Load from GuitarSet directory if enabled
        if self.use_guitarset:
            if self.guitarset_dir.exists():
                self._load_from_directory(self.guitarset_dir, "GuitarSet_Data")
            else:
                print(f"Warning: GuitarSet directory not found: {self.guitarset_dir}")
        
        print(f"Total loaded: {len(self.samples)} samples")
    

    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        cqt_tensor, label = self.samples[idx]  
        # Apply optional transform
        if self.transform:
            cqt_tensor = self.transform(cqt_tensor)
        
        return cqt_tensor, label
    
    def get_cqt_shape(self) -> Tuple[int, int, int]:
        """Return the shape of CQT output (channels, n_bins, time_frames)."""
        if len(self.samples) > 0:
            cqt, _ = self[0]
            return tuple(cqt.shape)
        return (1, self.n_bins, 0)


def create_dataloaders(
    data_dir: str | Path = "./Data",
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    n_bins: int = 84,
    hop_length: int = 512,
    bins_per_octave: int = 12,
    use_kaggle: bool = True,
    use_guitarset: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and test DataLoaders for chord classification.
    
    Args:
        data_dir: Path to base Data directory containing Kaggle_data and GuitarSet_Data
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        n_bins: Number of CQT frequency bins
        hop_length: Hop length for CQT
        bins_per_octave: Number of bins per octave for CQT
        use_kaggle: Whether to include Kaggle data
        use_guitarset: Whether to include GuitarSet data
        
    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    # Create datasets
    train_dataset = ChordCQTDataset(
        data_dir=data_dir,
        split="Training",
        n_bins=n_bins,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        use_kaggle=use_kaggle,
        use_guitarset=use_guitarset
    )
    
    test_dataset = ChordCQTDataset(
        data_dir=data_dir,
        split="Test",
        n_bins=n_bins,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        use_kaggle=use_kaggle,
        use_guitarset=use_guitarset
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for consistent batch sizes
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Train loader: {len(train_loader)} batches of size {batch_size}")
    print(f"Test loader: {len(test_loader)} batches of size {batch_size}")
    print(f"CQT shape: {train_dataset.get_cqt_shape()}")
    
    return train_loader, test_loader

