import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt

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
    
    Args:
        data_dir: Path to the data directory (e.g., './Data/Training' or './Data/Test')
        n_bins: Number of frequency bins for CQT (default: 72 for 6 octaves)
        hop_length: Hop length for CQT computation
        bins_per_octave: Number of bins per octave (default: 12 for semitone resolution)
        fmin: Minimum frequency (default: C2 ~65.4 Hz)
        transform: Optional transform to apply to the CQT output
        cache_cqt: Whether to cache CQT computations in memory (faster but uses more RAM)
        use_hpss: Whether to apply Harmonic-Percussive Source Separation before CQT
    """
    
    def __init__(
        self,
        data_dir: str | Path,
        n_bins: int = 72,
        hop_length: int = 512,
        bins_per_octave: int = 12,
        fmin: Optional[float] = None,
        transform: Optional[callable] = None,
        cache_cqt: bool = False,
        use_hpss: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.n_bins = n_bins
        self.hop_length = hop_length
        self.bins_per_octave = bins_per_octave
        self.fmin = fmin if fmin is not None else librosa.note_to_hz('C2')
        self.transform = transform
        self.cache_cqt = cache_cqt
        self.use_hpss = use_hpss
        self.cache = {}
        
        # Collect all audio files and their labels
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()


    def _compute_cqt(self, audio_path: Path) -> np.ndarray:
        """Compute the Constant Q Transform of an audio file."""
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        # Apply HPSS to extract harmonic content (removes percussive transients)
        if self.use_hpss:
            y_harmonic, _ = librosa.effects.hpss(y)
        else:
            y_harmonic = y
        
        # Compute CQT on harmonic component
        cqt = librosa.cqt(
            y_harmonic,
            sr=sr,
            hop_length=self.hop_length,
            fmin=self.fmin,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave
        )
        
        # Convert to magnitude (absolute value) and then to dB scale
        cqt_mag = np.abs(cqt)
        cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)
        cqt_tensor = torch.tensor(cqt_db, dtype=torch.float32).unsqueeze(0)  # Shape: (1, n_bins, time_frames)
        
        return cqt_tensor


    def _load_samples(self):
        """Load all audio file paths and their corresponding labels."""
        for chord_folder in self.data_dir.iterdir():

            chord_name = chord_folder.name
            if chord_name not in CHORD_TO_IDX:
                print(f"Warning: Unknown chord folder '{chord_name}', skipping...")
                continue
            label = CHORD_TO_IDX[chord_name]
            print("Computing CQT for :",chord_folder)
            for audio_file in chord_folder.iterdir():
                
                if audio_file.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']:

                    self.samples.append((self._compute_cqt(audio_file), label))

        
        print(f"Loaded {len(self.samples)} samples from {self.data_dir}")
    
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
    train_dir: str | Path = "./Data/Training",
    test_dir: str | Path = "./Data/Test",
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    n_bins: int = 84,
    hop_length: int = 512,
    bins_per_octave: int = 12,
    cache_cqt: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and test DataLoaders for chord classification.
    
    Args:
        train_dir: Path to training data directory
        test_dir: Path to test data directory
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        n_bins: Number of CQT frequency bins
        hop_length: Hop length for CQT
        bins_per_octave: Number of bins per octave for CQT
        cache_cqt: Whether to cache CQT computations
        
    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    # Create datasets
    train_dataset = ChordCQTDataset(
        data_dir=train_dir,
        n_bins=n_bins,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        cache_cqt=cache_cqt
    )
    
    test_dataset = ChordCQTDataset(
        data_dir=test_dir,
        n_bins=n_bins,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        cache_cqt=cache_cqt
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

class ChordCNNWithAttention(nn.Module):
    """
    CNN with frequency-attention mechanism for chord classification.
    
    This model adds a channel attention mechanism to emphasize
    important frequency bins for chord recognition.
    
    Args:
        n_bins: Number of CQT frequency bins (default: 72)
        num_classes: Number of chord classes (default: 24)
        dropout: Dropout rate for regularization
    """
    
    def __init__(
        self,
        n_bins: int = 72,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5
    ):
        super(ChordCNNWithAttention, self).__init__()
        
        self.n_bins = n_bins
        self.num_classes = num_classes
        
        # Convolutional blocks
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Channel attention (SE block)
        self.se_fc1 = nn.Linear(64, 64 // 4)
        self.se_fc2 = nn.Linear(64 // 4, 64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
    def _channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Apply squeeze-and-excitation channel attention."""
        b, c, h, w = x.size()
        # Squeeze: global average pooling
        y = x.view(b, c, -1).mean(dim=2)  # (b, c)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = F.relu(self.se_fc1(y))
        y = torch.sigmoid(self.se_fc2(y))
        # Scale
        y = y.view(b, c, 1, 1)
        return x * y
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv blocks 1-2
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Apply channel attention
        x = self._channel_attention(x)
        
        # Conv blocks 3-4
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Global pooling and classifier
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
