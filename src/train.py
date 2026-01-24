
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from DataClasses import (
    create_dataloaders,
    IDX_TO_CHORD,
    NUM_CLASSES
)
from model import ChordCNNWithAttention

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
CHECKPOINT_DIR = Path('./checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)


def train_one_epoch(model, loader, criterion, optimizer, scheduler):
    """Train for one epoch."""
    model.to(DEVICE)
    model.train()

    total_loss, correct, total = 0, 0, 0
    
    for inputs, targets in tqdm(loader, desc='Training'):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)
    
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    """Validate the model."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    for inputs, targets in tqdm(loader, desc='Validating', leave=False):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)
    
    return total_loss / total, correct / total


def train():
    """Main training function."""
    print(f"Using device: {DEVICE}")
    
    # Data
    train_loader, val_loader = create_dataloaders(
        data_dir="./Data",
        batch_size=BATCH_SIZE,
        num_workers=4,
        n_bins=72,
        use_kaggle=True,
        use_guitarset=True
    )
    
    # Model
    model = ChordCNNWithAttention(n_bins=72, num_classes=NUM_CLASSES).to(DEVICE)
    print(f"Parameters: {model.count_parameters}")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=len(train_loader) * EPOCHS
    )
    
    # Training loop
    best_acc = 0
    patience, patience_counter = 10, 0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler
        )
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc*100:.2f}%")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / 'best_model.pt')
            print(f"  ✨ New best: {val_acc*100:.2f}%")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\nDone! Best accuracy: {best_acc*100:.2f}%")
    return model


if __name__ == "__main__":
    train()
