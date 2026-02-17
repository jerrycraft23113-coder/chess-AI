"""
Reinforcement Learning Training for Chess AI
Uses self-play and policy gradient methods to train the chess model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Tuple, Optional, Dict
import os
import argparse
from collections import deque
import chess

from chess_board import ChessBoard
from chess_model import ChessCNN, ChessPolicyNetwork


class SelfPlayAgent:
    """Agent for self-play that uses the neural network."""
    
    def __init__(self, model: nn.Module, temperature: float = 1.0):
        """Initialize self-play agent.
        
        Args:
            model: Neural network model (ChessCNN or ChessPolicyNetwork)
            temperature: Temperature for move selection (higher = more random)
        """
        self.model = model
        self.model.eval()
        self.temperature = temperature
    
    def select_move(self, board: ChessBoard, legal_moves: List[chess.Move]) -> chess.Move:
        """Select a move using the policy network.
        
        Args:
            board: Current board state
            legal_moves: List of legal moves
            
        Returns:
            Selected move
        """
        if not legal_moves:
            return None
        
        # Get board representation
        board_array = board.board_to_array()
        board_tensor = torch.FloatTensor(board_array).unsqueeze(0)
        board_tensor = board_tensor.permute(0, 3, 1, 2)  # (1, 12, 8, 8)
        
        with torch.no_grad():
            if isinstance(self.model, ChessPolicyNetwork):
                # Use policy network
                policy_logits, _ = self.model(board_tensor)
                policy_logits = policy_logits.squeeze(0)  # (4096,)
                
                # Convert moves to indices and get probabilities
                move_probs = torch.zeros(len(legal_moves))
                for i, move in enumerate(legal_moves):
                    move_idx = self._move_to_index(move)
                    if move_idx < len(policy_logits):
                        move_probs[i] = policy_logits[move_idx]
                
                # Apply temperature
                move_probs = move_probs / self.temperature
                move_probs = F.softmax(move_probs, dim=0)
                
                # Sample move
                move_idx = torch.multinomial(move_probs, 1).item()
                return legal_moves[move_idx]
            else:
                # Use value network with minimax-like selection
                # For simplicity, use greedy selection
                best_move = None
                best_value = float('-inf') if board.get_turn() else float('inf')

                for move in legal_moves:
                    board_copy = board.copy()
                    board_copy.make_move(move)

                    board_array = board_copy.board_to_array()
                    board_tensor = torch.FloatTensor(board_array).unsqueeze(0)
                    board_tensor = board_tensor.permute(0, 3, 1, 2)

                    # Model outputs evaluation from white's perspective
                    value = self.model(board_tensor).item()

                    # We want the best value from the CURRENT player's perspective
                    # If current player is black, negate the white-perspective value
                    if not board.get_turn():
                        value = -value

                    if board.get_turn():  # White is maximizing
                        if value > best_value:
                            best_value = value
                            best_move = move
                    else:  # Black is minimizing (from white's perspective)
                        if value < best_value:
                            best_value = value
                            best_move = move
                
                return best_move if best_move else random.choice(legal_moves)
    
    def _move_to_index(self, move: chess.Move) -> int:
        """Convert move to index."""
        from_square = move.from_square
        to_square = move.to_square
        return from_square * 64 + to_square


class GameTrajectory:
    """Stores a complete game trajectory for RL training."""
    
    def __init__(self):
        self.positions = []
        self.moves = []
        self.rewards = []
        self.values = []
        self.policy_logits = []
        self.is_white_turns = []  # Track whose turn it was
    
    def add_step(self, position: np.ndarray, move: chess.Move, 
                 value: float, is_white_turn: bool,
                 policy_logits: Optional[torch.Tensor] = None):
        """Add a step to the trajectory."""
        self.positions.append(position)
        self.moves.append(move)
        self.values.append(value)
        self.is_white_turns.append(is_white_turn)
        if policy_logits is not None:
            self.policy_logits.append(policy_logits)
    
    def set_rewards(self, final_result: str, is_white_turns: List[bool]):
        """Set rewards based on game result.
        
        Args:
            final_result: '1-0' (white wins), '0-1' (black wins), '1/2-1/2' (draw)
            is_white_turns: List of booleans indicating if each position was white's turn
        """
        if final_result == '1-0':
            white_reward = 1.0
            black_reward = -1.0
        elif final_result == '0-1':
            white_reward = -1.0
            black_reward = 1.0
        else:  # draw
            white_reward = 0.0
            black_reward = 0.0
        
        # Assign rewards based on turn
        self.rewards = []
        for is_white_turn in is_white_turns:
            reward = white_reward if is_white_turn else black_reward
            self.rewards.append(reward)


def play_self_play_game(model: nn.Module, max_moves: int = 200) -> GameTrajectory:
    """Play a self-play game and return trajectory.
    
    Args:
        model: Neural network model
        max_moves: Maximum moves before declaring draw
        
    Returns:
        GameTrajectory object
    """
    board = ChessBoard()
    agent = SelfPlayAgent(model, temperature=1.0)
    trajectory = GameTrajectory()
    
    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            break
        
        # Get current position
        position = board.board_to_array()
        
        # Get model prediction
        board_tensor = torch.FloatTensor(position).unsqueeze(0)
        board_tensor = board_tensor.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            if isinstance(model, ChessPolicyNetwork):
                policy_logits, value = model(board_tensor)
                policy_logits = policy_logits.squeeze(0)
            else:
                value = model(board_tensor)
                policy_logits = None
        
        # Select move
        move = agent.select_move(board, legal_moves)
        if move is None:
            break
        
        # Store trajectory step
        is_white_turn = board.get_turn()
        trajectory.add_step(position, move, value.item(), is_white_turn, policy_logits)
        
        # Make move
        board.make_move(move)
        move_count += 1
    
    # Set rewards based on final result
    result = board.get_result()
    if result:
        trajectory.set_rewards(result, trajectory.is_white_turns)
    else:
        # Draw by move limit or no result
        trajectory.set_rewards('1/2-1/2', trajectory.is_white_turns)
    
    # Debug: verify rewards were set
    if len(trajectory.rewards) == 0:
        print(f"    Warning: No rewards set for game! Result: {result}, Moves: {move_count}")
        # Set default rewards (draw)
        trajectory.set_rewards('1/2-1/2', trajectory.is_white_turns)
    
    return trajectory


def compute_policy_loss(model: nn.Module, trajectories: List[GameTrajectory], 
                       device: str = 'cpu') -> torch.Tensor:
    """Compute policy gradient loss from trajectories.
    
    Args:
        model: Policy network
        trajectories: List of game trajectories
        device: Device to compute on
        
    Returns:
        Loss tensor
    """
    if not isinstance(model, ChessPolicyNetwork):
        raise ValueError("Policy loss requires ChessPolicyNetwork")
    
    all_positions = []
    all_moves = []
    all_rewards = []
    
    # Collect all data
    for trajectory in trajectories:
        if len(trajectory.positions) == 0:
            continue
        all_positions.extend(trajectory.positions)
        all_moves.extend(trajectory.moves)
        all_rewards.extend(trajectory.rewards)
    
    if len(all_positions) == 0:
        return torch.tensor(0.0, device=device)
    
    # Convert to tensors
    positions_tensor = torch.FloatTensor(np.array(all_positions)).to(device)
    positions_tensor = positions_tensor.permute(0, 3, 1, 2)  # (batch, 12, 8, 8)
    rewards_tensor = torch.FloatTensor(all_rewards).to(device).unsqueeze(1)
    
    # Get policy and value predictions
    policy_logits, values = model(positions_tensor)
    
    # Compute advantages (rewards - values, detach values for baseline)
    advantages = rewards_tensor - values.detach()
    
    # Compute policy loss: REINFORCE
    policy_loss = 0.0
    valid_moves = 0
    for i, move in enumerate(all_moves):
        move_idx = move.from_square * 64 + move.to_square
        if move_idx < policy_logits.shape[1]:
            log_probs = F.log_softmax(policy_logits[i], dim=0)
            move_log_prob = log_probs[move_idx]
            advantage = advantages[i]
            # REINFORCE: -log_prob * advantage
            policy_loss -= move_log_prob * advantage
            valid_moves += 1
    
    if valid_moves > 0:
        policy_loss = policy_loss / valid_moves
    else:
        policy_loss = torch.tensor(0.0, device=device)
    
    # Value loss (MSE)
    value_loss = F.mse_loss(values, rewards_tensor)
    
    # Debug: check if rewards are all zero
    if rewards_tensor.abs().max().item() < 1e-6:
        print(f"    Warning: All rewards are near zero (likely all draws)")
        print(f"    Value predictions range: [{values.min().item():.4f}, {values.max().item():.4f}]")
    
    # Total loss
    total_loss = policy_loss + value_loss
    
    return total_loss


def train_rl(model: nn.Module, num_games: int = 100, batch_size: int = 10,
             learning_rate: float = 0.001, device: str = 'cpu',
             save_interval: int = 10):
    """Train model using reinforcement learning with self-play.
    
    Args:
        model: Neural network model to train
        num_games: Number of self-play games to generate
        batch_size: Number of games to collect before updating
        learning_rate: Learning rate for optimizer
        device: Device to train on
        save_interval: Save model every N games
    """
    model = model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting RL Training with Self-Play")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Total games: {'infinite' if num_games <= 0 else num_games}")
    print(f"Batch size: {batch_size}")
    print("=" * 50)
    
    trajectory_buffer = deque(maxlen=batch_size)
    
    try:
        # If num_games <= 0, run self-play games indefinitely until manually stopped
        game_num = 0
        while True:
            if num_games > 0 and game_num >= num_games:
                break
            
            game_num += 1
            # Play self-play game
            if game_num % 10 == 0:
                total_games = num_games if num_games > 0 else float("inf")
                print(f"\nGame {game_num}/{total_games}...")
            
            trajectory = play_self_play_game(model)
            trajectory_buffer.append(trajectory)
            
            # Update model when buffer is full
            if len(trajectory_buffer) >= batch_size:
                print(f"  Training on batch of {len(trajectory_buffer)} games...")
                
                loss = None
                if isinstance(model, ChessPolicyNetwork):
                    # Policy gradient training
                    loss = compute_policy_loss(model, list(trajectory_buffer), device)
                else:
                    # Value-based training (simplified)
                    # Collect positions and rewards
                    all_positions = []
                    all_rewards = []
                    
                    for traj in trajectory_buffer:
                        if len(traj.positions) > 0 and len(traj.rewards) > 0:
                            all_positions.extend(traj.positions)
                            all_rewards.extend(traj.rewards)
                    
                    if len(all_positions) > 0:
                        positions_tensor = torch.FloatTensor(np.array(all_positions)).to(device)
                        positions_tensor = positions_tensor.permute(0, 3, 1, 2)
                        rewards_tensor = torch.FloatTensor(all_rewards).to(device).unsqueeze(1)

                        # Debug: print reward stats
                        reward_mean = rewards_tensor.mean().item()
                        reward_std = rewards_tensor.std().item()
                        print(f"    Positions: {len(all_positions)}, Rewards: mean={reward_mean:.4f}, std={reward_std:.4f}")

                        predictions = model(positions_tensor)
                        loss = F.mse_loss(predictions, rewards_tensor)
                    else:
                        print("    Warning: No valid positions in trajectories, skipping batch")
                        trajectory_buffer.clear()
                        continue
                
                # Check if loss is valid
                if loss is None:
                    print("    Warning: Loss is None, skipping batch")
                    trajectory_buffer.clear()
                    continue
                
                # Check if loss is zero or very small
                if loss.item() == 0.0 or torch.isnan(loss):
                    print(f"    Warning: Loss is {loss.item()}, checking data...")
                    # Debug: check if rewards are all zero
                    if isinstance(model, ChessPolicyNetwork):
                        # For policy network, check trajectories
                        total_rewards = sum(sum(traj.rewards) for traj in trajectory_buffer)
                        print(f"    Total rewards in batch: {total_rewards}")
                    else:
                        # For value network, we already printed reward stats above
                        pass
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                print(f"    Loss: {loss.item():.6f}")
                
                # Clear buffer
                trajectory_buffer.clear()
            
            # Save model periodically
            if game_num % save_interval == 0:
                model_dir = "models"
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"chess_model_rl_{game_num}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"  Model saved to {model_path}")
    except KeyboardInterrupt:
        print("\nRL training interrupted by user (KeyboardInterrupt).")
        print("Saving current model weights before exiting...")
    finally:
        # Final save (also executed on KeyboardInterrupt)
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        final_path = os.path.join(model_dir, "chess_model_rl_final.pth")
        torch.save(model.state_dict(), final_path)
        print(f"\nFinal model saved to {final_path}")


def main():
    """Main function for RL training."""
    parser = argparse.ArgumentParser(description='Train chess model using RL')
    parser.add_argument('--model-type', choices=['value', 'policy'], default='value',
                       help='Type of model: value (ChessCNN) or policy (ChessPolicyNetwork)')
    parser.add_argument('--games', type=int, default=100,
                       help='Number of self-play games (default: 100)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for training (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to load existing model')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save model every N games (default: 10)')
    parser.add_argument('--use-cpu', action='store_true',
                       help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Device selection
    if args.use_cpu:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Chess RL Training")
    print("=" * 50)
    
    # Create model
    if args.model_type == 'policy':
        model = ChessPolicyNetwork(hidden_size=256)
        print("Using Policy Network (ChessPolicyNetwork)")
    else:
        model = ChessCNN(hidden_size=256)
        print("Using Value Network (ChessCNN)")
    
    # Load existing model if provided
    if args.load_model and os.path.exists(args.load_model):
        try:
            model.load_state_dict(torch.load(args.load_model, map_location=device))
            print(f"Loaded model from {args.load_model}")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Using device: {device}")
    print()
    
    # Train
    train_rl(
        model=model,
        num_games=args.games,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        save_interval=args.save_interval
    )
    
    print("\n" + "=" * 50)
    print("RL Training complete!")


if __name__ == "__main__":
    main()
