"""
Training Script for Chess Neural Network
Trains the model using real game data from PGN files or generated data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import List, Tuple, Optional
import os
import argparse
import time

from chess_board import ChessBoard, evaluate_position_simple
from chess_model import ChessCNN
import chess


def _bitboards_to_tensor(bbs: np.ndarray) -> torch.Tensor:
    """Decode (12,) uint64 bitboards → (12, 8, 8) float32 tensor.

    Each uint64 is a chess bitboard: bit i → square i (rank = i//8, file = i%8).
    Output layout: tensor[ch, 7 - rank, file] = 1.0 (matching chess_board.py).
    """
    out = torch.zeros(12, 8, 8, dtype=torch.float32)
    for ch in range(12):
        mask = int(bbs[ch])
        if mask == 0:
            continue
        while mask:
            sq = (mask & -mask).bit_length() - 1
            out[ch, 7 - (sq >> 3), sq & 7] = 1.0
            mask &= mask - 1
    return out


# Vectorized version for batch decoding (used for in-memory datasets)
_BITS = np.arange(64, dtype=np.uint64)

def _bitboards_batch_to_array(bbs: np.ndarray) -> np.ndarray:
    """Decode (N, 12) uint64 bitboards → (N, 12, 8, 8) float32 array (vectorized)."""
    n = bbs.shape[0]
    # (N, 12, 1) >> (64,) → (N, 12, 64) then & 1
    expanded = ((bbs[:, :, np.newaxis] >> _BITS) & np.uint64(1)).astype(np.float32)
    # Reshape 64 → 8x8 then flip rank: bit i → rank i//8, file i%8 → [7-rank, file]
    expanded = expanded.reshape(n, 12, 8, 8)
    expanded = expanded[:, :, ::-1, :]  # flip rank axis (7-rank)
    return np.ascontiguousarray(expanded)


class ChessDataset(Dataset):
    """Dataset for chess positions and evaluations.

    Supports three storage formats:
      - Bitboard (N, 12) uint64 — compact, decoded on-the-fly (preferred)
      - Float (N, 8, 8, 12) or (N, 12, 8, 8) — legacy, loaded as tensors
    Handles both in-memory and memory-mapped numpy arrays.
    """

    def __init__(self, positions: np.ndarray, evaluations: np.ndarray):
        self._is_mmap = isinstance(positions, np.memmap)
        self._is_bitboard = (positions.dtype == np.uint64 and
                             (positions.ndim == 2 or positions.ndim == 1))

        if self._is_mmap:
            # Keep as mmap — decode lazily in __getitem__
            self.positions = positions
            self.evaluations = evaluations
        elif self._is_bitboard:
            # Bitboard in RAM — decode full batch upfront to float tensors
            print(f"  Decoding {len(positions):,} bitboard positions to tensors...")
            arr = _bitboards_batch_to_array(positions)  # (N, 12, 8, 8)
            self.positions = torch.from_numpy(arr)
            self.evaluations = torch.FloatTensor(np.array(evaluations, dtype=np.float32))
        else:
            # Legacy float format
            if positions.ndim == 4 and positions.shape[-1] == 12:
                positions = positions.transpose(0, 3, 1, 2)
            self.positions = torch.FloatTensor(positions)
            self.evaluations = torch.FloatTensor(evaluations)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        if self._is_mmap:
            # Return raw numpy for mmap — decoded in collate_fn for speed
            return np.array(self.positions[idx]), float(self.evaluations[idx])
        return self.positions[idx], self.evaluations[idx]


def _collate_mmap_bitboard(batch):
    """Custom collate that batch-decodes bitboard positions (vectorized)."""
    bbs = np.stack([b[0] for b in batch])            # (B, 12) uint64
    evals = np.array([b[1] for b in batch], dtype=np.float32)  # (B,)
    pos_tensor = torch.from_numpy(_bitboards_batch_to_array(bbs))  # (B, 12, 8, 8)
    return pos_tensor, torch.from_numpy(evals)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    val_loader: Optional[DataLoader] = None,
):
    """Train the chess evaluation model.

    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        val_loader: Optional validation DataLoader for early stopping
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # OneCycleLR: much faster convergence than StepLR
    total_steps = num_epochs * len(train_loader) if num_epochs > 0 else 1000
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate * 10,
            total_steps=total_steps, pct_start=0.3,
            anneal_strategy='cos',
        )

    # Mixed precision (AMP) — works on both CPU & CUDA
    use_amp = (device != 'cpu')
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    model.train()

    # If num_epochs <= 0, train indefinitely until manually stopped
    epoch = 0
    while True:
        if num_epochs > 0 and epoch >= num_epochs:
            break

        epoch += 1
        total_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for batch_idx, (positions, evaluations) in enumerate(train_loader):
            positions = positions.to(device, non_blocking=True)
            evaluations = evaluations.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

            # Forward pass with optional AMP
            with torch.amp.autocast(device_type=device, enabled=use_amp):
                predictions = model(positions)
                loss = criterion(predictions, evaluations)

            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        elapsed = time.time() - t0
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        total_epochs = num_epochs if num_epochs > 0 else float("inf")
        lr_now = optimizer.param_groups[0]['lr']
        samples_per_sec = len(train_loader.dataset) / elapsed if elapsed > 0 else 0
        print(f"Epoch {epoch}/{total_epochs}, Loss: {avg_loss:.4f}, "
              f"LR: {lr_now:.6f}, {samples_per_sec:.0f} samples/s, {elapsed:.1f}s")

        # Validation & early stopping
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for vp, ve in val_loader:
                    vp = vp.to(device, non_blocking=True)
                    ve = ve.to(device, non_blocking=True).unsqueeze(1)
                    val_loss += criterion(model(vp), ve).item()
                    val_batches += 1
            avg_val = val_loss / val_batches if val_batches > 0 else 0
            print(f"  Val Loss: {avg_val:.4f}")
            model.train()

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break


def load_training_data_from_file(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load training data from numpy file(s).

    Supports two formats:
      1. Legacy .npz file (single file with 'positions' and 'evaluations' keys)
      2. Memory-mapped .npy files (positions.npy + evaluations.npy in same dir)
         — preferred for large datasets since they never need to fit in RAM.

    Args:
        data_path: Path to .npz file or directory containing .npy files

    Returns:
        Tuple of (positions, evaluations) as numpy arrays (may be mmap'd)
    """
    data_dir = os.path.dirname(data_path) or 'data'
    pos_npy = os.path.join(data_dir, 'positions.npy')
    eval_npy = os.path.join(data_dir, 'evaluations.npy')

    # Prefer .npy memory-mapped files (created by parse_pgn_data merge)
    if os.path.exists(pos_npy) and os.path.exists(eval_npy):
        print(f"Loading memory-mapped training data from {data_dir}/...")
        try:
            positions = np.load(pos_npy, mmap_mode='r')      # (N, 12) uint64 bitboards
            evaluations = np.load(eval_npy, mmap_mode='r')    # (N,)
            fmt = "bitboard" if positions.dtype == np.uint64 else "float32"
            size_mb = os.path.getsize(pos_npy) / (1024 * 1024)
            print(f"Loaded {len(positions):,} positions (memory-mapped, {fmt}, {size_mb:.0f} MB)")
            return positions, evaluations
        except Exception as e:
            print(f"Error loading .npy files: {e}, falling back to .npz")

    # Fallback: legacy .npz file
    print(f"Loading training data from {data_path}...")
    try:
        data = np.load(data_path, mmap_mode='r')
        positions = np.array(data['positions'])  # (N, 8, 8, 12)
        evaluations = np.array(data['evaluations'], dtype=np.float32)  # (N,)
        print(f"Loaded {len(positions)} positions from file")
        return positions, evaluations
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        return np.array([]), np.array([])


def mirror_positions(positions: np.ndarray, evaluations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Horizontally mirror positions for data augmentation.

    Flips along the file axis (left-right) to double the dataset.

    Args:
        positions: numpy array of shape (N, 8, 8, 12)
        evaluations: numpy array of shape (N,)

    Returns:
        Tuple of (augmented_positions, augmented_evaluations) with 2N samples
    """
    mirrored = np.flip(positions, axis=2).copy()  # flip along file axis
    aug_positions = np.concatenate([positions, mirrored], axis=0)
    aug_evaluations = np.concatenate([evaluations, evaluations], axis=0)
    print(f"Data augmentation: {len(positions)} -> {len(aug_positions)} positions")
    return aug_positions, aug_evaluations


def run_training(
    data_path: str = 'data/training_data.npz',
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 0.001,
    use_cpu: bool = False,
    augment: bool = False,
):
    """Run the full supervised training pipeline.
    
    This is the programmatic entry point used by both this script and `main.py`.
    Training data must be pre-generated from PGN files into a .npz file
    (e.g. via `parse_pgn_data.py`) and pointed to by `data_path`.
    """
    print("Chess Neural Network Training")
    print("=" * 50)
    
    # Force CPU usage
    if use_cpu:
        device = 'cpu'
        print("Using device: CPU (forced)")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
    
    # Load training data (must be created from PGN with parse_pgn_data.py)
    data_dir = os.path.dirname(data_path) or 'data'
    has_npy = (os.path.exists(os.path.join(data_dir, 'positions.npy'))
               and os.path.exists(os.path.join(data_dir, 'evaluations.npy')))
    if not has_npy and not os.path.exists(data_path):
        print(f"\nError: Training data not found at {data_path} or {data_dir}/positions.npy")
        print("Please run:")
        print("  1. python download_pgn_data.py  (to download PGN files)")
        print("  2. python parse_pgn_data.py    (to parse PGN files and create training data)")
        print("\nThen run this training script again.")
        return

    positions, evaluations = load_training_data_from_file(data_path)

    if len(positions) == 0:
        print(f"\nError: Loaded 0 positions from {data_path}.")
        print("Please check that parse_pgn_data.py completed successfully and produced non-empty data.")
        return

    # Data augmentation (horizontal mirror) — skip for mmap (too large for in-memory concat)
    if augment and not isinstance(positions, np.memmap):
        positions, evaluations = mirror_positions(positions, evaluations)
    elif augment:
        print("  Skipping augmentation (memory-mapped data too large for in-memory concat)")

    # Split into train and validation
    split_idx = int(len(positions) * 0.8)
    train_positions = positions[:split_idx]
    train_evaluations = evaluations[:split_idx]
    val_positions = positions[split_idx:]
    val_evaluations = evaluations[split_idx:]

    # Create datasets
    train_dataset = ChessDataset(train_positions, train_evaluations)
    val_dataset = ChessDataset(val_positions, val_evaluations)

    # Optimized DataLoader: multi-worker prefetch, pin_memory for CUDA
    # Disable workers for mmap datasets — Windows spawn copies the whole mmap per worker
    is_mmap = train_dataset._is_mmap
    pin = (device != 'cpu')
    num_workers = 0 if is_mmap else (min(4, os.cpu_count() or 1) if len(train_dataset) > 1000 else 0)
    collate = _collate_mmap_bitboard if is_mmap else None
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0), collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0), collate_fn=collate,
    )

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"DataLoader workers: {num_workers}, batch_size: {batch_size}")

    # Create model
    print("\nInitializing model...")
    model = ChessCNN(hidden_size=256)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Try torch.compile for faster execution (PyTorch 2.x, needs C++ compiler)
    _compiled = False
    if hasattr(torch, 'compile'):
        try:
            compiled_model = torch.compile(model)
            with torch.no_grad():
                compiled_model(torch.zeros(1, 12, 8, 8, device=device))
            model = compiled_model
            _compiled = True
            print("Model compiled with torch.compile()")
        except Exception:
            print("torch.compile() skipped (no C++ compiler), using eager mode")

    # Train model
    print("\nStarting training...")
    try:
        train_model(
            model=model,
            train_loader=train_loader,
            num_epochs=epochs,
            learning_rate=lr,
            device=device,
            val_loader=val_loader,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (KeyboardInterrupt).")
        print("Saving current model weights...")

    # Final validation
    print("\nFinal validation...")
    model.eval()
    total_val_loss = 0.0
    num_batches = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for positions, evaluations in val_loader:
            positions = positions.to(device, non_blocking=True)
            evaluations = evaluations.to(device, non_blocking=True).unsqueeze(1)
            total_val_loss += criterion(model(positions), evaluations).item()
            num_batches += 1

    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
    print(f"Final Validation Loss: {avg_val_loss:.4f}")

    # Save model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "chess_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


def main():
    """Main training function (CLI entry point)."""
    parser = argparse.ArgumentParser(description='Train chess neural network')
    parser.add_argument('--data', type=str, default='data/training_data.npz',
                       help='Path to training data file (default: data/training_data.npz)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20, use <= 0 for infinite)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--use-cpu', action='store_true',
                       help='Force CPU usage (default: auto-detect GPU)')
    parser.add_argument('--augment', action='store_true',
                       help='Enable data augmentation (horizontal mirror, doubles dataset)')

    args = parser.parse_args()

    run_training(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_cpu=args.use_cpu,
        augment=args.augment,
    )


if __name__ == "__main__":
    main()
