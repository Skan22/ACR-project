#!/usr/bin/env python3
"""
Export the trained ChordCNNWithAttention model to ONNX format for C++ inference.

Usage:
    python export_to_onnx.py [--verify] [--checkpoint PATH]

The exported model will be saved to ../checkpoints/chord_model.onnx
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model import ChordCNNWithAttention, NUM_CLASSES, IDX_TO_CHORD

# CQT parameters matching DataClasses.py
N_BINS = 72
SAMPLE_RATE = 22050
HOP_LENGTH = 512


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    n_bins: int = N_BINS,
    num_classes: int = NUM_CLASSES,
) -> None:
    """
    Export the PyTorch model to ONNX format.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        output_path: Path for the output .onnx file
        n_bins: Number of CQT frequency bins
        num_classes: Number of chord classes
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Initialize model
    model = ChordCNNWithAttention(n_bins=n_bins, num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"Model loaded successfully. Parameters: {model.count_parameters():,}")
    
    # Create dummy input matching expected CQT shape
    # Shape: (batch_size, channels, n_bins, time_frames)
    # Using 43 time frames as typical for ~1 second of audio at 22050Hz with hop_length=512
    dummy_time_frames = 43  # ~1 second: 22050 / 512 ≈ 43
    dummy_input = torch.randn(1, 1, n_bins, dummy_time_frames)
    
    print(f"Exporting to ONNX with input shape: {dummy_input.shape}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['cqt_input'],
        output_names=['chord_logits'],
        dynamic_axes={
            'cqt_input': {0: 'batch_size', 3: 'time_frames'},
            'chord_logits': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


def verify_onnx(
    checkpoint_path: Path,
    onnx_path: Path,
    n_bins: int = N_BINS,
    num_classes: int = NUM_CLASSES,
) -> bool:
    """
    Verify that ONNX model produces same outputs as PyTorch model.
    
    Returns:
        True if verification passes, False otherwise
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("Error: Please install onnx and onnxruntime:")
        print("  pip install onnx onnxruntime")
        return False
    
    print("\n--- Verifying ONNX export ---")
    
    # Load PyTorch model
    model = ChordCNNWithAttention(n_bins=n_bins, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Verify ONNX model is valid
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model validation passed")
    
    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(str(onnx_path))
    
    # Test with multiple input sizes
    test_shapes = [(1, 1, n_bins, 43), (1, 1, n_bins, 86), (2, 1, n_bins, 43)]
    
    all_passed = True
    for shape in test_shapes:
        test_input = torch.randn(*shape)
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = model(test_input).numpy()
        
        # ONNX inference
        onnx_output = ort_session.run(
            None,
            {'cqt_input': test_input.numpy()}
        )[0]
        
        # Compare outputs
        max_diff = np.abs(pytorch_output - onnx_output).max()
        passed = max_diff < 1e-5
        
        status = "✓" if passed else "✗"
        print(f"{status} Shape {shape}: max difference = {max_diff:.2e}")
        
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All verification tests passed!")
        
        # Demo: show chord prediction for random input
        demo_input = torch.randn(1, 1, n_bins, 43)
        with torch.no_grad():
            logits = model(demo_input)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        print(f"\nDemo prediction (random input):")
        print(f"  Predicted chord: {IDX_TO_CHORD[pred_idx]}")
        print(f"  Confidence: {confidence:.2%}")
    else:
        print("\n✗ Some verification tests failed!")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Export chord model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).parent.parent / "checkpoints" / "best_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for ONNX model (default: checkpoints/chord_model.onnx)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify ONNX output matches PyTorch"
    )
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        args.output = args.checkpoint.parent / "chord_model.onnx"
    
    # Ensure checkpoint exists
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
    
    # Export model
    export_to_onnx(args.checkpoint, args.output)
    
    # Optionally verify
    if args.verify:
        success = verify_onnx(args.checkpoint, args.output)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
