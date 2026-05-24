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
import threading
import queue as _queue

from chess_board import ChessBoard, evaluate_position_simple
from chess_model import ChessCNN
import chess


def _bitboards_to_tensor(bbs: np.ndarray) -> torch.Tensor:
    """Decode (12,) uint64 bitboards → (12, 8, 8) float32 tensor."""
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


def _bitboards_batch_to_array(bbs: np.ndarray) -> np.ndarray:
    """Decode (N, 12) uint64 bitboards → (N, 12, 8, 8) float32 array.

    Uses np.unpackbits on uint8 view — 2-3x faster than uint64 broadcasting.
    Little-endian x86: byte[b] of uint64 = bits for squares rank=b, files 0-7.
    After unpack: index sq = bit sq of uint64 = whether square sq has a piece.
    Reshape (N,12,64)→(N,12,8,8) then flip rank to match board_to_array layout.
    """
    n = bbs.shape[0]
    # View (N,12) uint64 as (N,12,8) uint8 — each uint64 = 8 bytes (little-endian)
    bbs_u8 = bbs.view(np.uint8).reshape(n, 12, 8)
    # Unpack bits LSB-first: bit i of each byte → output index i
    # Result: (N, 12, 64) uint8, where index sq = whether square sq occupied
    bits = np.unpackbits(bbs_u8, axis=-1, bitorder='little')   # (N,12,64) uint8
    # Reshape 64 → 8x8: (N,12, rank0-7, file0-7)
    bits = bits.reshape(n, 12, 8, 8)
    # Flip rank: row 0 becomes rank-8 (top of board) matching board_to_array
    bits = bits[:, :, ::-1, :]
    return np.ascontiguousarray(bits, dtype=np.float32)


class ChessDataset(Dataset):
    """Dataset for chess positions and evaluations."""

    def __init__(self, positions: np.ndarray, evaluations: np.ndarray):
        self._is_mmap = isinstance(positions, np.memmap)
        self._is_bitboard = (positions.dtype == np.uint64 and
                             (positions.ndim == 2 or positions.ndim == 1))

        if self._is_mmap:
            self.positions = positions
            self.evaluations = evaluations
        elif self._is_bitboard:
            print(f"  Decoding {len(positions):,} bitboard positions to tensors...")
            arr = _bitboards_batch_to_array(positions)
            self.positions = torch.from_numpy(arr)
            self.evaluations = torch.FloatTensor(np.array(evaluations, dtype=np.float32))
        else:
            if positions.ndim == 4 and positions.shape[-1] == 12:
                positions = positions.transpose(0, 3, 1, 2)
            self.positions = torch.FloatTensor(positions)
            self.evaluations = torch.FloatTensor(evaluations)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        if self._is_mmap:
            return np.array(self.positions[idx]), float(self.evaluations[idx])
        return self.positions[idx], self.evaluations[idx]


def _collate_mmap_bitboard(batch):
    """Custom collate that batch-decodes bitboard positions (vectorized)."""
    bbs = np.stack([b[0] for b in batch])
    evals = np.array([b[1] for b in batch], dtype=np.float32)
    pos_tensor = torch.from_numpy(_bitboards_batch_to_array(bbs))
    return pos_tensor, torch.from_numpy(evals)


# ──────────────────────────────────────────────────────────────────────────────
# DataPrefetcher: overlaps CPU bitboard-decode with GPU forward/backward pass
# Uses a background thread + queue to keep `num_prefetch` batches ready on GPU.
# ──────────────────────────────────────────────────────────────────────────────

class DataPrefetcher:
    """Prefetch batches onto the GPU in a background thread.

    Eliminates CPU/GPU idle time caused by sequential bitboard decoding and
    device transfer.  The background thread decodes the next batch while the
    GPU processes the current one.
    """

    def __init__(self, loader: DataLoader, device: str, num_prefetch: int = 4):
        self.device = device
        self._len = len(loader)
        self._q: _queue.Queue = _queue.Queue(maxsize=num_prefetch)
        self._use_cuda = device != 'cpu'
        if self._use_cuda:
            self._stream = torch.cuda.Stream()
        t = threading.Thread(target=self._worker, args=(loader,), daemon=True)
        t.start()

    def _worker(self, loader):
        try:
            for pos, ev in loader:
                if self._use_cuda:
                    with torch.cuda.stream(self._stream):
                        pos = pos.to(self.device, non_blocking=True)
                        ev = ev.to(self.device, non_blocking=True).unsqueeze(1)
                else:
                    ev = ev.unsqueeze(1)
                self._q.put((pos, ev))
        finally:
            self._q.put(None)  # sentinel

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            if self._use_cuda:
                # Ensure GPU ops wait until transfer from prefetch stream is done
                torch.cuda.current_stream().wait_stream(self._stream)
            yield item

    def __len__(self):
        return self._len


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    val_loader: Optional[DataLoader] = None,
    val_freq: int = 5,
    val_batches: int = 2000,
    checkpoint_dir: str = 'models',
):
    """Train the chess evaluation model.

    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs (<=0 = infinite)
        learning_rate: Learning rate for optimizer
        device: Training device ('cpu' or 'cuda')
        val_loader: Optional validation DataLoader
        val_freq: Run validation every this many epochs (0 = never)
        val_batches: Max validation batches per eval pass (0 = all)
        checkpoint_dir: Directory to save per-epoch checkpoints
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    total_steps = num_epochs * len(train_loader) if num_epochs > 0 else 1000
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate * 10,
            total_steps=total_steps, pct_start=0.3,
            anneal_strategy='cos',
        )

    use_amp = (device != 'cpu')
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    os.makedirs(checkpoint_dir, exist_ok=True)

    model.train()

    epoch = 0
    epoch_times = []

    while True:
        if num_epochs > 0 and epoch >= num_epochs:
            break

        epoch += 1
        total_loss = 0.0
        num_batches = 0
        t0 = time.time()

        prefetcher = DataPrefetcher(train_loader, device, num_prefetch=4)
        total_train_batches = len(train_loader)

        for positions, evaluations in prefetcher:
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device, enabled=use_amp):
                predictions = model(positions)
                loss = criterion(predictions, evaluations)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Print progress every 500 batches
            if num_batches % 500 == 0:
                elapsed = time.time() - t0
                batches_left = total_train_batches - num_batches
                eta_s = (elapsed / num_batches) * batches_left
                print(f"  [{num_batches}/{total_train_batches}] "
                      f"loss={total_loss/num_batches:.4f} "
                      f"ETA {eta_s/60:.1f}min", flush=True)

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        total_epochs = num_epochs if num_epochs > 0 else '∞'
        lr_now = optimizer.param_groups[0]['lr']
        samples_per_sec = len(train_loader.dataset) / elapsed if elapsed > 0 else 0

        # ETA for remaining epochs
        if num_epochs > 0 and len(epoch_times) > 0:
            avg_epoch_t = sum(epoch_times) / len(epoch_times)
            remaining_epochs = num_epochs - epoch
            eta_total = avg_epoch_t * remaining_epochs
            eta_str = f", ETA {eta_total/3600:.1f}h"
        else:
            eta_str = ""

        print(f"Epoch {epoch}/{total_epochs} | Loss: {avg_loss:.4f} | "
              f"LR: {lr_now:.6f} | {samples_per_sec:.0f} samp/s | "
              f"{elapsed/60:.1f}min{eta_str}")

        # Save checkpoint after every epoch
        ckpt_path = os.path.join(checkpoint_dir, f"chess_model_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)

        # Validation (every val_freq epochs)
        if val_loader is not None and val_freq > 0 and epoch % val_freq == 0:
            model.eval()
            val_loss = 0.0
            val_batches_done = 0
            t_val = time.time()
            with torch.no_grad():
                for vp, ve in DataPrefetcher(val_loader, device, num_prefetch=4):
                    with torch.amp.autocast(device_type=device, enabled=use_amp):
                        val_loss += criterion(model(vp), ve).item()
                    val_batches_done += 1
                    if val_batches > 0 and val_batches_done >= val_batches:
                        break
            avg_val = val_loss / val_batches_done if val_batches_done > 0 else 0
            val_elapsed = time.time() - t_val
            print(f"  Val Loss: {avg_val:.4f} ({val_batches_done} batches, {val_elapsed:.1f}s)")
            model.train()

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                # Save best model
                best_path = os.path.join(checkpoint_dir, "chess_model_best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"  New best model saved → {best_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break


def load_training_data_from_file(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load training data from numpy file(s)."""
    data_dir = os.path.dirname(data_path) or 'data'
    pos_npy = os.path.join(data_dir, 'positions.npy')
    eval_npy = os.path.join(data_dir, 'evaluations.npy')

    if os.path.exists(pos_npy) and os.path.exists(eval_npy):
        print(f"Loading memory-mapped training data from {data_dir}/...")
        try:
            positions = np.load(pos_npy, mmap_mode='r')
            evaluations = np.load(eval_npy, mmap_mode='r')
            fmt = "bitboard" if positions.dtype == np.uint64 else "float32"
            size_mb = os.path.getsize(pos_npy) / (1024 * 1024)
            print(f"Loaded {len(positions):,} positions (memory-mapped, {fmt}, {size_mb:.0f} MB)")
            return positions, evaluations
        except Exception as e:
            print(f"Error loading .npy files: {e}, falling back to .npz")

    print(f"Loading training data from {data_path}...")
    try:
        data = np.load(data_path, mmap_mode='r')
        positions = np.array(data['positions'])
        evaluations = np.array(data['evaluations'], dtype=np.float32)
        print(f"Loaded {len(positions)} positions from file")
        return positions, evaluations
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        return np.array([]), np.array([])


def mirror_positions(positions: np.ndarray, evaluations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Horizontally mirror positions for data augmentation."""
    mirrored = np.flip(positions, axis=2).copy()
    aug_positions = np.concatenate([positions, mirrored], axis=0)
    aug_evaluations = np.concatenate([evaluations, evaluations], axis=0)
    print(f"Data augmentation: {len(positions)} -> {len(aug_positions)} positions")
    return aug_positions, aug_evaluations


def run_training(
    data_path: str = 'data/training_data.npz',
    epochs: int = 20,
    batch_size: int = 2048,
    lr: float = 0.001,
    use_cpu: bool = False,
    augment: bool = False,
    val_freq: int = 5,
    val_batches: int = 2000,
    samples_per_epoch: int = 20_000_000,
):
    """Run the full supervised training pipeline."""
    print("Chess Neural Network Training")
    print("=" * 50)

    if use_cpu:
        device = 'cpu'
        print("Using device: CPU (forced)")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

    # CUDA performance flags
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True       # auto-tune kernels for fixed input sizes
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        torch.cuda.empty_cache()

    data_dir = os.path.dirname(data_path) or 'data'
    has_npy = (os.path.exists(os.path.join(data_dir, 'positions.npy'))
               and os.path.exists(os.path.join(data_dir, 'evaluations.npy')))
    if not has_npy and not os.path.exists(data_path):
        print(f"\nError: Training data not found at {data_path} or {data_dir}/positions.npy")
        print("Please run:")
        print("  1. python scripts/download_pgn_data.py")
        print("  2. python scripts/parse_pgn_data.py")
        return

    positions, evaluations = load_training_data_from_file(data_path)

    if len(positions) == 0:
        print(f"\nError: Loaded 0 positions from {data_path}.")
        return

    if augment and not isinstance(positions, np.memmap):
        positions, evaluations = mirror_positions(positions, evaluations)
    elif augment:
        print("  Skipping augmentation (memory-mapped data too large for in-memory concat)")

    split_idx = int(len(positions) * 0.8)
    train_positions = positions[:split_idx]
    train_evaluations = evaluations[:split_idx]
    val_positions = positions[split_idx:]
    val_evaluations = evaluations[split_idx:]

    train_dataset = ChessDataset(train_positions, train_evaluations)
    val_dataset = ChessDataset(val_positions, val_evaluations)

    is_mmap = train_dataset._is_mmap
    pin = (device != 'cpu')
    # workers=0 on Windows with mmap (spawn would copy entire mmap per worker)
    num_workers = 0 if is_mmap else (min(4, os.cpu_count() or 1) if len(train_dataset) > 1000 else 0)
    collate = _collate_mmap_bitboard if is_mmap else None

    # RandomSampler with num_samples avoids generating a full 165M permutation
    # (which needs 1.3 GB RAM + several seconds), while still sampling randomly.
    # replacement=True is fine: with 20M draws from 165M, overlap is ~12%.
    use_subset = 0 < samples_per_epoch < len(train_dataset)
    if use_subset:
        train_sampler = torch.utils.data.RandomSampler(
            train_dataset, replacement=True, num_samples=samples_per_epoch
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin, drop_last=True,
            persistent_workers=(num_workers > 0), collate_fn=collate,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin, drop_last=True,
            persistent_workers=(num_workers > 0), collate_fn=collate,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 4, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0), collate_fn=collate,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    batches_per_epoch = len(train_loader)
    eff_samples = samples_per_epoch if use_subset else len(train_dataset)
    print(f"\nTraining samples:   {len(train_dataset):,}  (using {eff_samples:,}/epoch)")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Batch size:         {batch_size}")
    print(f"Batches/epoch:      {batches_per_epoch:,}")
    print(f"DataLoader workers: {num_workers}")
    print(f"Val every {val_freq} epochs, max {val_batches} val batches")

    print("\nInitializing model...")
    model = ChessCNN(hidden_size=256)
    model = model.to(device)   # must be on GPU before torch.compile()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    import logging
    logging.getLogger("torch._inductor").setLevel(logging.ERROR)

    _compiled = False
    if hasattr(torch, 'compile') and device == 'cuda':
        # Try backends in order of expected speedup.
        # 'reduce-overhead': inductor + CUDA-graph capture (~1.5-1.8x), needs MSVC
        # 'cudagraphs': records & replays CUDA ops, no compiler needed (~1.1-1.3x)
        # Note: warm-up uses inference only — BatchNorm running-stats updates
        #       cause graph-breaks in backward, which torch.compile handles gracefully.
        _compile_attempts = [
            ('inductor/reduce-overhead', dict(mode='reduce-overhead', fullgraph=False)),
            ('cudagraphs',               dict(backend='cudagraphs',   fullgraph=False)),
        ]
        dummy = torch.zeros(2, 12, 8, 8, device=device)
        _last_err = ''
        for label, kwargs in _compile_attempts:
            try:
                cmodel = torch.compile(model, **kwargs)
                # Forward-only warm-up (backward is fine in eager; BatchNorm breaks
                # static CUDA graphs so we don't force-compile the backward path)
                cmodel.eval()
                with torch.no_grad():
                    cmodel(dummy)
                cmodel.train()
                model = cmodel
                _compiled = True
                print(f"torch.compile() OK — backend: {label}")
                break
            except Exception as e:
                _last_err = f"{type(e).__name__}: {e}"
                continue
        if not _compiled:
            print(f"torch.compile() skipped — last error: {_last_err}")
            print("  GTX 1050 Ti (sm_61) is below Triton's minimum (sm_70).")
            print("  torch.compile() with GPU speedup requires RTX 2000+ or newer.")

    print("\nStarting training...")
    try:
        train_model(
            model=model,
            train_loader=train_loader,
            num_epochs=epochs,
            learning_rate=lr,
            device=device,
            val_loader=val_loader,
            val_freq=val_freq,
            val_batches=val_batches,
            checkpoint_dir='models',
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")

    # Final validation (full)
    print("\nFinal validation...")
    model.eval()
    total_val_loss = 0.0
    num_batches = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for vp, ve in DataPrefetcher(val_loader, device, num_prefetch=4):
            with torch.amp.autocast(device_type=device, enabled=(device != 'cpu')):
                total_val_loss += criterion(model(vp), ve).item()
            num_batches += 1
            if num_batches >= val_batches > 0:
                break

    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
    print(f"Final Validation Loss: {avg_val_loss:.4f}")

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "chess_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


def main():
    """Main training function (CLI entry point)."""
    parser = argparse.ArgumentParser(description='Train chess neural network')
    parser.add_argument('--data', type=str, default='data/training_data.npz',
                       help='Path to training data file')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (<=0 for infinite)')
    parser.add_argument('--batch-size', type=int, default=2048,
                       help='Batch size (default: 2048)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--use-cpu', action='store_true',
                       help='Force CPU usage')
    parser.add_argument('--augment', action='store_true',
                       help='Enable data augmentation (horizontal mirror)')
    parser.add_argument('--val-freq', type=int, default=5,
                       help='Validate every N epochs (default: 5, 0=never)')
    parser.add_argument('--val-batches', type=int, default=2000,
                       help='Max validation batches per eval (default: 2000, 0=all)')
    parser.add_argument('--samples-per-epoch', type=int, default=20_000_000,
                       help='Samples to draw per epoch (default: 20M, 0=full dataset). '
                            'Reduces epoch time 8x vs full 165M dataset.')

    args = parser.parse_args()

    run_training(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_cpu=args.use_cpu,
        augment=args.augment,
        val_freq=args.val_freq,
        val_batches=args.val_batches,
        samples_per_epoch=args.samples_per_epoch,
    )


if __name__ == "__main__":
    main()
