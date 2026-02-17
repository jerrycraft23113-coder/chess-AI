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

from chess_board import ChessBoard, evaluate_position_simple
from chess_model import ChessCNN
import chess


class ChessDataset(Dataset):
    """Dataset for chess positions and evaluations."""
    
    def __init__(self, positions: List[np.ndarray], evaluations: List[float]):
        """Initialize dataset.
        
        Args:
            positions: List of board arrays (8x8x12)
            evaluations: List of evaluation scores
        """
        self.positions = torch.FloatTensor(np.array(positions))
        self.evaluations = torch.FloatTensor(evaluations)
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        # Convert from (8, 8, 12) to (12, 8, 8) for CNN
        position = self.positions[idx].permute(2, 0, 1)
        return position, self.evaluations[idx]


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = 'cpu'
):
    """Train the chess evaluation model.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    model.train()
    
    # If num_epochs <= 0, train indefinitely until manually stopped
    epoch = 0
    while True:
        if num_epochs > 0 and epoch >= num_epochs:
            break
        
        epoch += 1
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (positions, evaluations) in enumerate(train_loader):
            positions = positions.to(device)
            evaluations = evaluations.to(device).unsqueeze(1)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(positions)
            loss = criterion(predictions, evaluations)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        scheduler.step()
        total_epochs = num_epochs if num_epochs > 0 else float("inf")
        print(f"Epoch {epoch}/{total_epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")


def load_training_data_from_file(data_path: str) -> Tuple[List[np.ndarray], List[float]]:
    """Load training data from numpy file (memory-efficient version).
    
    Args:
        data_path: Path to .npz file containing training data
        
    Returns:
        Tuple of (positions, evaluations)
    """
    print(f"Loading training data from {data_path}...")
    
    try:
        # Use memory mapping for large files to reduce RAM usage
        data = np.load(data_path, mmap_mode='r')
        num_positions = len(data['positions'])
        
        print(f"Found {num_positions} positions in file")
        print("Loading data into memory (this may take a moment for large files)...")
        
        # Load positions in chunks to avoid memory spikes
        chunk_size = 10000
        positions = []
        evaluations = []
        
        for i in range(0, num_positions, chunk_size):
            end_idx = min(i + chunk_size, num_positions)
            chunk_positions = data['positions'][i:end_idx]
            chunk_evaluations = data['evaluations'][i:end_idx]
            
            # Convert to list of arrays
            for pos in chunk_positions:
                positions.append(pos)
            evaluations.extend(chunk_evaluations.tolist())
            
            if (i + chunk_size) % 50000 == 0 or end_idx == num_positions:
                print(f"  Loaded {end_idx}/{num_positions} positions...")
        
        print(f"Loaded {len(positions)} positions from file")
        return positions, evaluations
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        return [], []


def run_training(
    data_path: str = 'data/training_data.npz',
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.001,
    use_cpu: bool = True,
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
    if not os.path.exists(data_path):
        print(f"\nError: Training data file not found: {data_path}")
        print("Please run:")
        print("  1. python download_pgn_data.py  (to download PGN files)")
        print("  2. python parse_pgn_data.py    (to parse PGN files and create training_data.npz)")
        print("\nThen run this training script again pointing --data to the generated .npz file.")
        return
    
    positions, evaluations = load_training_data_from_file(data_path)
    
    if len(positions) == 0:
        print(f"\nError: Loaded 0 positions from {data_path}.")
        print("Please check that parse_pgn_data.py completed successfully and produced non-empty data.")
        return
    
    # Split into train and validation
    split_idx = int(len(positions) * 0.8)
    train_positions = positions[:split_idx]
    train_evaluations = evaluations[:split_idx]
    val_positions = positions[split_idx:]
    val_evaluations = evaluations[split_idx:]
    
    # Create datasets
    train_dataset = ChessDataset(train_positions, train_evaluations)
    val_dataset = ChessDataset(val_positions, val_evaluations)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = ChessCNN(hidden_size=256)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nStarting training...")
    try:
        train_model(
            model=model,
            train_loader=train_loader,
            num_epochs=epochs,
            learning_rate=lr,
            device=device
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (KeyboardInterrupt).")
        print("Proceeding to validation and saving the current model weights...")
    
    # Validate
    print("\nValidating model...")
    model.eval()
    total_val_loss = 0.0
    num_batches = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for positions, evaluations in val_loader:
            positions = positions.to(device)
            evaluations = evaluations.to(device).unsqueeze(1)
            predictions = model(positions)
            loss = criterion(predictions, evaluations)
            total_val_loss += loss.item()
            num_batches += 1
    
    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
    print(f"Validation Loss: {avg_val_loss:.4f}")
    
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
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--use-cpu', action='store_true',
                       help='Force CPU usage (default: auto-detect GPU)')
    
    args = parser.parse_args()
    
    run_training(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_cpu=args.use_cpu,
    )


if __name__ == "__main__":
    main()
