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
import time
from collections import deque
import chess

from chess_board import ChessBoard
from chess_model import ChessCNN, ChessPolicyNetwork


class SelfPlayAgent:
    """Agent for self-play that uses the neural network. Vectorized batch evaluation."""

    def __init__(self, model: nn.Module, temperature: float = 1.0, epsilon: float = 0.1):
        """Initialize self-play agent.

        Args:
            model: Neural network model (ChessCNN or ChessPolicyNetwork)
            temperature: Temperature for move selection (higher = more random)
            epsilon: Probability of choosing a random move (exploration)
        """
        self.model = model
        self.model.eval()
        self.temperature = temperature
        self.epsilon = epsilon

    def select_move(self, board: ChessBoard, legal_moves: List[chess.Move]) -> chess.Move:
        """Select a move using vectorized batch evaluation.

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

                # Vectorized: convert all move indices at once using numpy
                move_indices = np.array(
                    [m.from_square * 64 + m.to_square for m in legal_moves],
                    dtype=np.int64
                )
                valid = move_indices < len(policy_logits)
                move_probs = torch.zeros(len(legal_moves))
                move_probs[valid] = policy_logits[move_indices[valid]]

                # Apply temperature
                move_probs = move_probs / self.temperature
                move_probs = F.softmax(move_probs, dim=0)

                # Sample move
                move_idx = torch.multinomial(move_probs, 1).item()
                return legal_moves[move_idx]
            else:
                # Epsilon-greedy: random move with probability epsilon
                if random.random() < self.epsilon:
                    return random.choice(legal_moves)

                # Vectorized batch evaluation: evaluate ALL positions at once
                n_moves = len(legal_moves)
                batch_arrays = np.empty((n_moves, 8, 8, 12), dtype=np.float32)

                bb = board.board
                for i, move in enumerate(legal_moves):
                    bb.push(move)
                    wrapper = ChessBoard.__new__(ChessBoard)
                    wrapper.board = bb
                    batch_arrays[i] = wrapper.board_to_array()
                    bb.pop()

                # Single batch forward pass (vectorized)
                batch_tensor = torch.FloatTensor(batch_arrays).permute(0, 3, 1, 2)
                values = self.model(batch_tensor).squeeze(1)  # (n_moves,)

                # Select best move
                if board.get_turn():  # White maximizing
                    best_idx = values.argmax().item()
                else:  # Black minimizing
                    best_idx = values.argmin().item()

                return legal_moves[best_idx]

    def _move_to_index(self, move: chess.Move) -> int:
        """Convert move to index."""
        return move.from_square * 64 + move.to_square


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
        """Set rewards based on game result. Vectorized with numpy.

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

        # Vectorized reward assignment using numpy
        turns = np.array(is_white_turns, dtype=bool)
        rewards = np.where(turns, white_reward, black_reward)
        self.rewards = rewards.tolist()


def play_self_play_game(model: nn.Module, max_moves: int = 200,
                        temperature: float = 1.0) -> GameTrajectory:
    """Play a self-play game and return trajectory.

    Args:
        model: Neural network model
        max_moves: Maximum moves before declaring draw
        temperature: Temperature for move selection

    Returns:
        GameTrajectory object
    """
    board = ChessBoard()
    agent = SelfPlayAgent(model, temperature=temperature)
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
    
    # Vectorized policy loss: REINFORCE using batch indexing
    move_indices = np.array(
        [m.from_square * 64 + m.to_square for m in all_moves], dtype=np.int64
    )
    valid_mask = move_indices < policy_logits.shape[1]

    if valid_mask.any():
        valid_indices = torch.LongTensor(move_indices[valid_mask]).to(device)
        valid_positions = torch.where(torch.BoolTensor(valid_mask).to(device))[0]

        # Batch log_softmax over all valid positions
        log_probs = F.log_softmax(policy_logits[valid_positions], dim=1)
        # Gather the log probabilities for the selected moves
        selected_log_probs = log_probs.gather(1, valid_indices.unsqueeze(1))
        valid_advantages = advantages[valid_positions]

        # REINFORCE: -log_prob * advantage, averaged
        policy_loss = -(selected_log_probs * valid_advantages).mean()
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

    # Try torch.compile for faster execution (PyTorch 2.x, needs C++ compiler)
    if hasattr(torch, 'compile'):
        try:
            compiled_model = torch.compile(model)
            with torch.no_grad():
                compiled_model(torch.zeros(1, 12, 8, 8, device=device))
            model = compiled_model
            print("Model compiled with torch.compile()")
        except Exception:
            print("torch.compile() skipped (no C++ compiler), using eager mode")

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    print("Starting RL Training with Self-Play")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Total games: {'infinite' if num_games <= 0 else num_games}")
    print(f"Batch size: {batch_size}")
    print("=" * 50)

    trajectory_buffer = deque(maxlen=batch_size)

    # Win rate tracking
    wins_white = 0
    wins_black = 0
    draws = 0

    # Temperature annealing: start at 1.0, decay to 0.3
    temp_start = 1.0
    temp_end = 0.3
    total_for_anneal = max(num_games, 100)

    t_start = time.time()
    try:
        # If num_games <= 0, run self-play games indefinitely until manually stopped
        game_num = 0
        while True:
            if num_games > 0 and game_num >= num_games:
                break

            game_num += 1

            # Temperature annealing
            frac = min(game_num / total_for_anneal, 1.0)
            current_temp = temp_start + (temp_end - temp_start) * frac

            # Play self-play game
            if game_num % 10 == 0:
                total_games = num_games if num_games > 0 else float("inf")
                total_played = wins_white + wins_black + draws
                wr = (wins_white / total_played * 100) if total_played > 0 else 0
                br = (wins_black / total_played * 100) if total_played > 0 else 0
                dr = (draws / total_played * 100) if total_played > 0 else 0
                elapsed_total = time.time() - t_start
                gps = game_num / elapsed_total if elapsed_total > 0 else 0
                print(f"\nGame {game_num}/{total_games} (temp={current_temp:.2f}) "
                      f"W:{wr:.0f}% B:{br:.0f}% D:{dr:.0f}% "
                      f"[{gps:.1f} games/s]")

            t_game = time.time()
            trajectory = play_self_play_game(model, temperature=current_temp)
            trajectory_buffer.append(trajectory)

            # Track win rates
            if len(trajectory.rewards) > 0:
                final_reward = trajectory.rewards[-1] if trajectory.is_white_turns[-1] else -trajectory.rewards[-1]
                if final_reward > 0.5:
                    wins_white += 1
                elif final_reward < -0.5:
                    wins_black += 1
                else:
                    draws += 1
            
            # Update model when buffer is full
            if len(trajectory_buffer) >= batch_size:
                t_train = time.time()

                loss = None
                if isinstance(model, ChessPolicyNetwork):
                    # Policy gradient training
                    loss = compute_policy_loss(model, list(trajectory_buffer), device)
                else:
                    # Value-based training — efficient numpy stacking
                    pos_arrays = []
                    rew_arrays = []
                    for traj in trajectory_buffer:
                        if len(traj.positions) > 0 and len(traj.rewards) > 0:
                            pos_arrays.append(np.array(traj.positions, dtype=np.float32))
                            rew_arrays.append(np.array(traj.rewards, dtype=np.float32))

                    if pos_arrays:
                        all_pos = np.concatenate(pos_arrays, axis=0)     # (N,8,8,12)
                        all_rew = np.concatenate(rew_arrays, axis=0)     # (N,)
                        n_pos = len(all_pos)

                        # Transpose once and convert to tensor
                        positions_tensor = torch.from_numpy(
                            all_pos.transpose(0, 3, 1, 2).copy()
                        ).to(device)
                        rewards_tensor = torch.from_numpy(all_rew).to(device).unsqueeze(1)

                        reward_mean = all_rew.mean()
                        reward_std = all_rew.std()
                        print(f"  Training: {n_pos} positions, "
                              f"rewards mean={reward_mean:.4f} std={reward_std:.4f}")

                        # Mini-batch gradient accumulation for large batches
                        mini_bs = 512
                        optimizer.zero_grad(set_to_none=True)
                        total_loss = 0.0
                        n_mini = 0
                        for i in range(0, n_pos, mini_bs):
                            p_batch = positions_tensor[i:i+mini_bs]
                            r_batch = rewards_tensor[i:i+mini_bs]
                            pred = model(p_batch)
                            mb_loss = F.mse_loss(pred, r_batch)
                            mb_loss.backward()
                            total_loss += mb_loss.item()
                            n_mini += 1

                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        loss_val = total_loss / n_mini
                        print(f"  Loss: {loss_val:.6f}, train time: {time.time()-t_train:.1f}s")
                    else:
                        print("  Warning: No valid positions, skipping batch")
                        trajectory_buffer.clear()
                        continue

                # Policy network backward
                if isinstance(model, ChessPolicyNetwork) and loss is not None:
                    if torch.isnan(loss) or loss.item() == 0.0:
                        total_rewards = sum(sum(traj.rewards) for traj in trajectory_buffer)
                        print(f"  Warning: Loss={loss.item()}, total rewards={total_rewards}")

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    print(f"  Loss: {loss.item():.6f}, train time: {time.time()-t_train:.1f}s")

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
