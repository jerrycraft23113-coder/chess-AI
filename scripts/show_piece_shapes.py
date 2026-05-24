"""
Script để hiển thị shape và cách biểu diễn các quân cờ trong model
Shows the shape and representation of chess pieces in the neural network
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from chess_board import ChessBoard
from chess_model import ChessCNN


def show_board_shape_info():
    """Hiển thị thông tin về shape của board representation."""
    print("=" * 60)
    print("CHESS PIECE REPRESENTATION SHAPES")
    print("=" * 60)
    
    # Tạo một board mẫu
    board = ChessBoard()
    
    # Lấy board array
    board_array = board.board_to_array()
    
    print("\n1. BOARD ARRAY SHAPE:")
    print(f"   Shape: {board_array.shape}")
    print(f"   Type: {type(board_array)}")
    print(f"   Dtype: {board_array.dtype}")
    print(f"   Total elements: {board_array.size:,}")
    print(f"   Memory size: {board_array.nbytes / 1024:.2f} KB")
    
    print("\n2. CHANNEL MAPPING (12 channels):")
    print("   Channel 0:  White Pawn")
    print("   Channel 1:  White Rook")
    print("   Channel 2:  White Knight")
    print("   Channel 3:  White Bishop")
    print("   Channel 4:  White Queen")
    print("   Channel 5:  White King")
    print("   Channel 6:  Black Pawn")
    print("   Channel 7:  Black Rook")
    print("   Channel 8:  Black Knight")
    print("   Channel 9:  Black Bishop")
    print("   Channel 10: Black Queen")
    print("   Channel 11: Black King")
    
    print("\n3. BOARD DIMENSIONS:")
    print(f"   Rows (rank): 8 (0-7, from bottom to top)")
    print(f"   Columns (file): 8 (0-7, from left to right)")
    print(f"   Channels: 12 (piece types and colors)")
    
    print("\n4. SAMPLE BOARD (Starting Position):")
    print_board_channels(board_array)
    
    print("\n5. FOR NEURAL NETWORK INPUT:")
    # Convert to tensor format
    board_tensor = torch.FloatTensor(board_array)
    board_tensor_cnn = board_tensor.permute(2, 0, 1)  # (12, 8, 8) for CNN
    
    print(f"   Original shape: {board_array.shape} (8, 8, 12)")
    print(f"   CNN input shape: {board_tensor_cnn.shape} (12, 8, 8)")
    print(f"   Batch shape (single): (1, 12, 8, 8)")
    print(f"   Batch shape (batch of 32): (32, 12, 8, 8)")
    
    print("\n6. MODEL INPUT/OUTPUT SHAPES:")
    model = ChessCNN(hidden_size=256)
    
    # Create a sample batch
    sample_batch = torch.zeros(1, 12, 8, 8)
    with torch.no_grad():
        output = model(sample_batch)
    
    print(f"   Model input:  (batch_size, 12, 8, 8)")
    print(f"   Model output: {output.shape} (batch_size, 1)")
    print(f"   Output represents: Position evaluation score")
    
    print("\n7. PIECE COUNT IN STARTING POSITION:")
    count_pieces(board_array)
    
    print("\n" + "=" * 60)


def print_board_channels(board_array: np.ndarray):
    """In ra board với các channel được đánh dấu."""
    piece_names = [
        "WP", "WR", "WN", "WB", "WQ", "WK",  # White pieces
        "BP", "BR", "BN", "BB", "BQ", "BK"   # Black pieces
    ]
    
    print("\n   Channel visualization (1 = piece present, 0 = empty):")
    print("   " + "-" * 50)
    
    for channel in range(12):
        print(f"\n   Channel {channel:2d} ({piece_names[channel]}):")
        channel_data = board_array[:, :, channel]
        
        # Print board with pieces (column headers aligned with data)
        print("     " + " ".join(["a", "b", "c", "d", "e", "f", "g", "h"]))
        for rank in range(7, -1, -1):  # From rank 8 to rank 1
            row_str = f"   {rank + 1} "
            for file in range(8):
                if channel_data[rank, file] > 0.5:
                    row_str += "X "
                else:
                    row_str += ". "
            print(row_str)


def count_pieces(board_array: np.ndarray):
    """Đếm số lượng quân cờ trong board."""
    piece_names = [
        "White Pawn", "White Rook", "White Knight", "White Bishop", "White Queen", "White King",
        "Black Pawn", "Black Rook", "Black Knight", "Black Bishop", "Black Queen", "Black King"
    ]
    
    print("\n   Piece counts:")
    for channel in range(12):
        count = int(np.sum(board_array[:, :, channel]))
        if count > 0:
            print(f"   {piece_names[channel]:20s}: {count}")


def show_different_positions():
    """Hiển thị shape của các vị trí khác nhau."""
    print("\n" + "=" * 60)
    print("EXAMPLES OF DIFFERENT POSITIONS")
    print("=" * 60)
    
    # Create positions
    board0 = ChessBoard()

    board1 = ChessBoard()
    board1.make_move_from_uci("e2e4")

    board2 = ChessBoard()
    board2.make_move_from_uci("e2e4")
    board2.make_move_from_uci("e7e5")

    positions = [
        ("Starting Position", board0),
        ("After e2e4", board1),
        ("After e2e4 e7e5", board2),
    ]
    
    for name, board in positions:
        if board is None:
            continue
        board_array = board.board_to_array()
        print(f"\n{name}:")
        print(f"  Shape: {board_array.shape}")
        print(f"  Total pieces: {int(np.sum(board_array))}")
        count_pieces(board_array)


def show_model_architecture():
    """Hiển thị kiến trúc model và các layer shapes."""
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE AND LAYER SHAPES")
    print("=" * 60)
    
    model = ChessCNN(hidden_size=256)
    
    print("\nModel Architecture:")
    print(model)
    
    print("\nLayer Shapes (forward pass):")
    print("  Input:  (batch, 12, 8, 8)")
    print("  Conv1:  (batch, 64, 8, 8)   - 3x3 conv, padding=1")
    print("  Conv2:  (batch, 128, 8, 8)  - 3x3 conv, padding=1")
    print("  Conv3:  (batch, 128, 8, 8)  - 3x3 conv, padding=1")
    print("  Flatten: (batch, 128 * 8 * 8) = (batch, 8192)")
    print("  FC1:    (batch, 256)")
    print("  FC2:    (batch, 256)")
    print("  Output: (batch, 1)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size (float32): {total_params * 4 / 1024 / 1024:.2f} MB")


def main():
    """Main function."""
    show_board_shape_info()
    show_different_positions()
    show_model_architecture()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key Points:
1. Each chess position is represented as an 8×8×12 numpy array
2. The 12 channels represent 6 piece types × 2 colors
3. For CNN input, it's reshaped to (12, 8, 8) or (batch, 12, 8, 8)
4. The model outputs a single evaluation score per position
5. Each position uses 3.84 KB of memory (8×8×12×4 bytes for float32)
    """)


if __name__ == "__main__":
    main()
