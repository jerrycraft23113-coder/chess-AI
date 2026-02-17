"""
Neural Network Model for Chess Evaluation
Uses a convolutional neural network to evaluate chess positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import chess


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
        # 8*8*8*8 = 4096 possible moves (from square * to square)
        self.policy_fc = nn.Linear(128 * 8 * 8, 4096)
        
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
            - policy_logits: (batch_size, 4096) move probabilities
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


def move_to_index(move: chess.Move) -> int:
    """Convert a chess move to a linear index (0-4095).
    
    Args:
        move: Chess move
        
    Returns:
        Index in range [0, 4095]
    """
    from_square = move.from_square
    to_square = move.to_square
    return from_square * 64 + to_square


def index_to_move(index: int) -> chess.Move:
    """Convert a linear index to a chess move.
    
    Args:
        index: Index in range [0, 4095]
        
    Returns:
        Chess move
    """
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)
