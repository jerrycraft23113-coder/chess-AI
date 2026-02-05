"""
Chess Game Playing Interface
Allows playing chess against the trained neural network model.
"""

import torch
import numpy as np
import chess
from typing import Optional, List

from chess_board import ChessBoard, evaluate_position_advanced
from chess_model import ChessCNN


class ChessAI:
    """Chess AI using neural network for evaluation."""
    
    def __init__(self, model_path: Optional[str] = None, depth: int = 2,
                 classical_weight: float = 0.3):
        """Initialize Chess AI.
        
        Args:
            model_path: Path to trained model weights
            depth: Search depth for minimax algorithm
        """
        self.model = ChessCNN(hidden_size=256)
        self.depth = depth
        # How much weight to give to handcrafted evaluation versus neural net
        self.classical_weight = max(0.0, min(1.0, classical_weight))
        # Simple transposition table: maps (fen, depth, maximizing) -> eval
        self.transposition_table = {}
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                print(f"Loaded model from {model_path}")
            except:
                print(f"Warning: Could not load model from {model_path}, using untrained model")
        
        self.model.eval()
    
    def evaluate_position(self, board: ChessBoard) -> float:
        """Evaluate a position using neural network + classical features.
        
        Args:
            board: ChessBoard instance
            
        Returns:
            Evaluation score from current player's perspective
        """
        # Convert board to tensor (for neural evaluation)
        board_array = board.board_to_array()
        board_tensor = torch.FloatTensor(board_array).unsqueeze(0)  # Add batch dimension
        board_tensor = board_tensor.permute(0, 3, 1, 2)  # Convert to (1, 12, 8, 8)

        # Neural-net evaluation (from side-to-move perspective)
        with torch.no_grad():
            nn_eval = self.model(board_tensor).item()

        if not board.get_turn():
            nn_eval = -nn_eval

        # Classical evaluation from white perspective, convert to side-to-move
        classical_eval = evaluate_position_advanced(board)
        if not board.get_turn():
            classical_eval = -classical_eval

        # Blend the two
        alpha = self.classical_weight
        evaluation = (1.0 - alpha) * nn_eval + alpha * classical_eval

        return evaluation

    def _order_moves(self, board: ChessBoard, moves: List[chess.Move], maximizing: bool) -> List[chess.Move]:
        """Order moves to improve alpha-beta pruning efficiency."""
        piece_vals = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 325,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 10000,
        }
        scored_moves = []
        for move in moves:
            score = 0
            # Captures first (MVV-LVA style)
            if board.board.is_capture(move):
                captured = board.board.piece_at(move.to_square)
                attacker = board.board.piece_at(move.from_square)
                if captured:
                    score += 10 * piece_vals.get(captured.piece_type, 0)
                if attacker:
                    score -= piece_vals.get(attacker.piece_type, 0)
            # Promotions are usually strong
            if move.promotion is not None:
                score += 800
            # Checks are generally strong
            try:
                if board.board.gives_check(move):
                    score += 300
            except Exception:
                pass
            scored_moves.append((score, move))

        # For maximizing side, higher score first; for minimizing, reverse
        scored_moves.sort(key=lambda x: x[0], reverse=maximizing)
        return [m for _, m in scored_moves]

    def quiescence(self, board: ChessBoard, alpha: float, beta: float, maximizing: bool) -> float:
        """Quiescence search: extend search on noisy (capture) positions.
        
        Helps avoid horizon effect by not stopping on obviously tactical positions.
        """
        stand_pat = self.evaluate_position(board)

        if maximizing:
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
        else:
            if stand_pat <= alpha:
                return alpha
            if stand_pat < beta:
                beta = stand_pat

        # Only explore capture moves in quiescence
        captures = [m for m in board.get_legal_moves() if board.board.is_capture(m)]
        if not captures:
            return stand_pat

        ordered_captures = self._order_moves(board, captures, maximizing)

        for move in ordered_captures:
            board_copy = board.copy()
            board_copy.make_move(move)
            score = self.quiescence(board_copy, alpha, beta, not maximizing)

            if maximizing:
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    break
            else:
                if score < beta:
                    beta = score
                if beta <= alpha:
                    break

        return alpha if maximizing else beta
    
    def minimax(self, board: ChessBoard, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """Minimax algorithm with alpha-beta pruning.
        
        Args:
            board: ChessBoard instance
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing: True if maximizing player, False if minimizing
            
        Returns:
            Best evaluation score
        """
        # Transposition table lookup
        fen_key = (board.board.fen(), depth, maximizing)
        cached = self.transposition_table.get(fen_key)
        if cached is not None:
            return cached

        if depth == 0 or board.is_game_over():
            # At leaf nodes, use quiescence search instead of plain evaluation
            return self.quiescence(board, alpha, beta, maximizing)
        
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return self.evaluate_position(board)

        ordered_moves = self._order_moves(board, legal_moves, maximizing)
        
        if maximizing:
            max_eval = float('-inf')
            for move in ordered_moves:
                board_copy = board.copy()
                board_copy.make_move(move)
                eval_score = self.minimax(board_copy, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            # Store in transposition table with simple size cap
            if len(self.transposition_table) > 100_000:
                self.transposition_table.clear()
            self.transposition_table[fen_key] = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                board_copy = board.copy()
                board_copy.make_move(move)
                eval_score = self.minimax(board_copy, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            # Store in transposition table with simple size cap
            if len(self.transposition_table) > 100_000:
                self.transposition_table.clear()
            self.transposition_table[fen_key] = min_eval
            return min_eval
    
    def get_best_move(self, board: ChessBoard) -> Optional[chess.Move]:
        """Get the best move using minimax algorithm.
        
        Args:
            board: ChessBoard instance
            
        Returns:
            Best move or None if no legal moves
        """
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return None

        maximizing = board.get_turn() == chess.WHITE
        ordered_moves = self._order_moves(board, legal_moves, maximizing)

        best_move = None
        best_eval = float('-inf') if maximizing else float('inf')

        for move in ordered_moves:
            board_copy = board.copy()
            board_copy.make_move(move)
            eval_score = self.minimax(
                board_copy,
                self.depth - 1,
                float('-inf'),
                float('inf'),
                not maximizing
            )
            
            if maximizing:
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = move
            else:
                if eval_score < best_eval:
                    best_eval = eval_score
                    best_move = move
        
        return best_move


def print_board(board: ChessBoard):
    """Print the chess board in a readable format."""
    print("\n" + "=" * 50)
    print(board)
    print("=" * 50)


def play_game(ai_color: str = 'black', ai_depth: int = 2,
              model_path: Optional[str] = None, classical_weight: float = 0.3):
    """Play a game of chess against the AI.
    
    Args:
        ai_color: 'white' or 'black' - which color the AI plays
        ai_depth: Search depth for AI
        model_path: Path to trained model
    """
    print("Chess Game - Play against AI")
    print("=" * 50)
    print(f"AI plays: {ai_color}")
    print("Enter moves in UCI notation (e.g., 'e2e4')")
    print("Type 'quit' to exit, 'undo' to undo last move")
    print("=" * 50)
    
    board = ChessBoard()
    ai = ChessAI(model_path=model_path, depth=ai_depth,
                 classical_weight=classical_weight)
    
    while not board.is_game_over():
        print_board(board)
        
        current_player = "White" if board.get_turn() == chess.WHITE else "Black"
        print(f"\n{current_player} to move")
        
        # Check if it's AI's turn
        is_ai_turn = (ai_color == 'white' and board.get_turn() == chess.WHITE) or \
                     (ai_color == 'black' and board.get_turn() == chess.BLACK)
        
        if is_ai_turn:
            print("AI is thinking...")
            move = ai.get_best_move(board)
            if move:
                board.make_move(move)
                print(f"AI plays: {move.uci()}")
            else:
                print("AI has no legal moves!")
                break
        else:
            # Human player's turn
            while True:
                move_input = input("Your move: ").strip().lower()
                
                if move_input == 'quit':
                    print("Game ended by user.")
                    return
                
                if move_input == 'undo':
                    if len(board.board.move_stack) > 0:
                        board.board.pop()
                        print("Move undone.")
                        print_board(board)
                        continue
                    else:
                        print("No moves to undo.")
                        continue
                
                if board.make_move_from_uci(move_input):
                    break
                else:
                    print("Invalid move! Please try again.")
    
    # Game over
    print_board(board)
    result = board.get_result()
    if result == '1-0':
        print("\nWhite wins!")
    elif result == '0-1':
        print("\nBlack wins!")
    elif result == '1/2-1/2':
        print("\nDraw!")
    else:
        print("\nGame over!")


def main():
    """Main function for playing chess."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Play chess against AI')
    parser.add_argument('--ai-color', choices=['white', 'black'], default='black',
                       help='Color the AI plays (default: black)')
    parser.add_argument('--depth', type=int, default=2,
                       help='AI search depth (default: 2)')
    parser.add_argument('--model', type=str, default='models/chess_model.pth',
                       help='Path to trained model (default: models/chess_model.pth)')
    parser.add_argument('--classical-weight', type=float, default=0.3,
                       help='Weight for classical evaluation vs neural net (0-1, default: 0.3)')
    
    args = parser.parse_args()
    
    play_game(
        ai_color=args.ai_color,
        ai_depth=args.depth,
        model_path=args.model if args.model else None,
        classical_weight=args.classical_weight
    )


if __name__ == "__main__":
    main()
