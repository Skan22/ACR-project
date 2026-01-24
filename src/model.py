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
