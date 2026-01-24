"""
Evaluation script for the chord classification model.
Computes accuracy, confidence scores, and confusion matrix on the test set.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from DataClasses import (
    ChordCQTDataset,
    ALL_CHORDS,
    IDX_TO_CHORD,
    NUM_CLASSES
)
from model import ChordCNNWithAttention
from torch.utils.data import DataLoader

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = Path('./checkpoints/best_model.pt')
DATA_DIR = Path('./Data')
BATCH_SIZE = 32
N_BINS = 72


def load_model(checkpoint_path: Path) -> ChordCNNWithAttention:
    """Load the trained model from checkpoint."""
    model = ChordCNNWithAttention(n_bins=N_BINS, num_classes=NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Checkpoint val_acc: {checkpoint.get('val_acc', 'N/A'):.4f}" if 'val_acc' in checkpoint else "")
    return model


def create_test_loader(data_dir: Path, batch_size: int, use_kaggle: bool = True, use_guitarset: bool = True) -> DataLoader:
    """Create test DataLoader."""
    test_dataset = ChordCQTDataset(
        data_dir=data_dir,
        split="Test",
        n_bins=N_BINS,
        hop_length=512,
        bins_per_octave=12,
        use_kaggle=use_kaggle,
        use_guitarset=use_guitarset
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid issues with nnAudio CQT layer in multiprocessing
        pin_memory=True
    )
    
    print(f"Test set: {len(test_dataset)} samples, {len(test_loader)} batches")
    return test_loader


@torch.no_grad()
def evaluate(model: ChordCNNWithAttention, test_loader: DataLoader):
    """
    Evaluate the model on the test set.
    
    Returns:
        all_preds: List of predicted class indices
        all_targets: List of true class indices
        all_confidences: List of confidence scores (max softmax probability)
        all_probs: Full probability distributions for each sample
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_confidences = []
    all_probs = []
    
    for inputs, targets in tqdm(test_loader, desc='Evaluating'):
        inputs = inputs.to(DEVICE)
        
        # Forward pass
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        
        # Get predictions and confidence
        confidences, preds = torch.max(probs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())
        all_confidences.extend(confidences.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_targets), np.array(all_confidences), np.array(all_probs)


def compute_metrics(preds: np.ndarray, targets: np.ndarray, confidences: np.ndarray):
    """Compute and print evaluation metrics."""
    # Overall accuracy
    accuracy = (preds == targets).mean() * 100
    
    # Confidence statistics
    mean_confidence = confidences.mean() * 100
    correct_confidence = confidences[preds == targets].mean() * 100 if (preds == targets).any() else 0
    wrong_confidence = confidences[preds != targets].mean() * 100 if (preds != targets).any() else 0
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\n📊 Overall Accuracy: {accuracy:.2f}%")
    print(f"   Correct: {(preds == targets).sum()} / {len(targets)}")
    print(f"\n📈 Confidence Statistics:")
    print(f"   Mean confidence: {mean_confidence:.2f}%")
    print(f"   Confidence on correct predictions: {correct_confidence:.2f}%")
    print(f"   Confidence on wrong predictions: {wrong_confidence:.2f}%")
    
    # Per-class accuracy
    print(f"\n📋 Per-Class Performance:")
    print("-"*40)
    
    class_accuracies = []
    for i, chord in enumerate(ALL_CHORDS):
        mask = targets == i
        if mask.sum() > 0:
            class_acc = (preds[mask] == targets[mask]).mean() * 100
            class_conf = confidences[mask].mean() * 100
            class_accuracies.append((chord, class_acc, mask.sum(), class_conf))
    
    # Sort by accuracy
    class_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    for chord, acc, count, conf in class_accuracies:
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"   {chord:4s}: {bar} {acc:6.2f}% ({count:3d} samples, conf: {conf:.1f}%)")
    
    return accuracy


def plot_confusion_matrix(preds: np.ndarray, targets: np.ndarray, save_path: str = "confusion_matrix.png"):
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(targets, preds, labels=range(NUM_CLASSES))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw counts
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=ALL_CHORDS,
        yticklabels=ALL_CHORDS,
        ax=axes[0]
    )
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)
    
    # Normalized (percentages)
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=ALL_CHORDS,
        yticklabels=ALL_CHORDS,
        ax=axes[1],
        vmin=0,
        vmax=1
    )
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Confusion matrix saved to: {save_path}")
    plt.show()


def plot_confidence_distribution(confidences: np.ndarray, preds: np.ndarray, targets: np.ndarray, 
                                  save_path: str = "confidence_distribution.png"):
    """Plot confidence distribution for correct and incorrect predictions."""
    correct_mask = preds == targets
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(confidences[correct_mask] * 100, bins=50, alpha=0.7, label=f'Correct ({correct_mask.sum()})', color='green')
    ax.hist(confidences[~correct_mask] * 100, bins=50, alpha=0.7, label=f'Incorrect ({(~correct_mask).sum()})', color='red')
    
    ax.set_xlabel('Confidence (%)')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Confidence distribution saved to: {save_path}")
    plt.show()


def main():
    print(f"Using device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Data directory: {DATA_DIR}")
    
    # Load model
    model = load_model(CHECKPOINT_PATH)
    
    # Create test loader
    test_loader = create_test_loader(DATA_DIR, BATCH_SIZE)
    
    # Evaluate
    preds, targets, confidences, probs = evaluate(model, test_loader)
    
    # Compute and print metrics
    accuracy = compute_metrics(preds, targets, confidences)
    
    # Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(targets, preds, target_names=ALL_CHORDS, zero_division=0))
    
    # Plot confusion matrix
    plot_confusion_matrix(preds, targets)
    
    # Plot confidence distribution
    plot_confidence_distribution(confidences, preds, targets)
    
    return accuracy, preds, targets, confidences


if __name__ == "__main__":
    main()
