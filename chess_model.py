"""
Neural Network Model for Chess Evaluation
Uses a convolutional neural network to evaluate chess positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import chess


class ResidualBlock(nn.Module):
    """Standard residual block with two conv layers."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessResNet(nn.Module):
    """Residual network for chess evaluation (stronger architecture).

    This is a new architecture that does NOT replace ChessCNN,
    so existing .pth model files remain compatible with ChessCNN.
    """
    def __init__(self, filters: int = 128, num_blocks: int = 6, hidden_size: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(12, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(*[ResidualBlock(filters) for _ in range(num_blocks)])
        flat = filters * 8 * 8
        self.head = nn.Sequential(
            nn.Linear(flat, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = x.reshape(x.size(0), -1)
        return self.head(x)


class ChessCNN(nn.Module):
    """Convolutional Neural Network for chess position evaluation."""

    def __init__(self, hidden_size: int = 256):
        """Initialize the chess CNN model.

        Args:
            hidden_size: Size of hidden layers
        """
        super(ChessCNN, self).__init__()

        # Input: 8x8x12 (board representation)
        # Convolutional layers
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Output: single evaluation score

        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 12, 8, 8)

        Returns:
            Evaluation scores of shape (batch_size, 1)
        """
        # Convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class ChessPolicyNetwork(nn.Module):
    """Policy network that outputs move probabilities."""

    def __init__(self, hidden_size: int = 256):
        """Initialize the policy network.

        Args:
            hidden_size: Size of hidden layers
        """
        super(ChessPolicyNetwork, self).__init__()

        # Shared convolutional layers
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)

        # Policy head (outputs move probabilities)
        # 4096 normal moves + 3*64 underpromotions = 4288
        self.policy_fc = nn.Linear(128 * 8 * 8, 4288)

        # Value head (outputs position evaluation)
        self.value_fc1 = nn.Linear(128 * 8 * 8, hidden_size)
        self.value_fc2 = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 12, 8, 8)

        Returns:
            Tuple of (policy_logits, value)
            - policy_logits: (batch_size, 4288) move probabilities
            - value: (batch_size, 1) position evaluation
        """
        # Shared convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten
        x_flat = x.reshape(x.size(0), -1)

        # Policy head
        policy = self.policy_fc(x_flat)

        # Value head
        value = F.relu(self.value_fc1(x_flat))
        value = self.dropout(value)
        value = self.value_fc2(value)

        return policy, value


# Promotion encoding: Q=0 (default), R=1, B=2, N=3
_PROMO_MAP = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}
_PROMO_UNMAP = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]


def move_to_index(move: chess.Move) -> int:
    """Convert a chess move to a linear index.

    Normal moves: [0, 4095] (from_square * 64 + to_square)
    Underpromotions (R/B/N): [4096, 4287] (4096 + promo_offset * 64 + to_square)
    Queen promotions use the normal index (most common case).

    Args:
        move: Chess move

    Returns:
        Index in range [0, 4287]
    """
    base = move.from_square * 64 + move.to_square
    if move.promotion and move.promotion != chess.QUEEN:
        promo_offset = _PROMO_MAP[move.promotion] - 1  # R=0, B=1, N=2
        return 4096 + promo_offset * 64 + move.to_square
    return base


def index_to_move(index: int) -> chess.Move:
    """Convert a linear index to a chess move.

    Args:
        index: Index in range [0, 4287]

    Returns:
        Chess move
    """
    if index >= 4096:
        idx = index - 4096
        promo_idx = idx // 64  # 0=R, 1=B, 2=N
        to_square = idx % 64
        # Infer from_square: pawn must be on rank 7 (white) or rank 2 (black)
        # For white: from_square = to_square - 8 (if straight)
        from_square = to_square - 8 if to_square >= 56 else to_square + 8
        return chess.Move(from_square, to_square, promotion=_PROMO_UNMAP[promo_idx + 1])
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)
