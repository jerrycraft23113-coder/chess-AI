"""
Chess Game Playing Interface - Strong AI Engine
Uses iterative deepening, alpha-beta with enhancements:
  - Killer moves & history heuristic for move ordering
  - Null-move pruning (NMP)
  - Late move reductions (LMR)
  - Principal variation search (PVS)
  - Quiescence search with delta pruning
  - Transposition table
  - Time management
"""

import time
import torch
import numpy as np
import chess
from typing import Optional, List, Tuple

from chess_board import ChessBoard, evaluate_position_advanced, PIECE_VALUES
from chess_model import ChessCNN

# ── Constants ──────────────────────────────────────────────────
MATE_SCORE = 30000
MAX_DEPTH = 64
INFINITY = 999999

# Null-move pruning
NMP_REDUCTION = 3
NMP_MIN_DEPTH = 3

# Late-move reductions
LMR_FULL_DEPTH_MOVES = 4
LMR_REDUCTION_LIMIT = 3

# Delta pruning margin in quiescence (pawns)
DELTA_MARGIN = 2.0


class TranspositionEntry:
    """Entry in the transposition table."""
    __slots__ = ['key', 'depth', 'score', 'flag', 'best_move']
    EXACT = 0
    ALPHA = 1  # upper bound
    BETA = 2   # lower bound

    def __init__(self, key: str, depth: int, score: float, flag: int,
                 best_move: Optional[chess.Move]):
        self.key = key
        self.depth = depth
        self.score = score
        self.flag = flag
        self.best_move = best_move


class ChessAI:
    """Strong Chess AI using iterative deepening alpha-beta with enhancements."""

    def __init__(self, model_path: Optional[str] = None, depth: int = 5,
                 classical_weight: float = 0.7, time_limit: float = 0.0):
        """Initialize Chess AI.

        Args:
            model_path: Path to trained model weights
            depth: Maximum search depth (default 5)
            classical_weight: Weight for classical eval vs neural net (0-1)
            time_limit: Time limit per move in seconds (0 = use depth only)
        """
        self.model = ChessCNN(hidden_size=256)
        self.depth = depth
        self.classical_weight = max(0.0, min(1.0, classical_weight))
        self.time_limit = time_limit

        # Transposition table
        self.transposition_table: dict = {}
        self.tt_max_size = 2_000_000

        # Killer moves: 2 slots per depth
        self.killer_moves = [[None, None] for _ in range(MAX_DEPTH)]

        # History heuristic table: [color][from_sq][to_sq]
        self.history = [[[0] * 64 for _ in range(64)] for _ in range(2)]

        # Search stats
        self.nodes_searched = 0
        self.start_time = 0.0
        self.time_up = False
        self.best_move_root = None

        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                print(f"Loaded model from {model_path}")
            except (FileNotFoundError, RuntimeError, ValueError,
                    torch.serialization.pickle.UnpicklingError) as e:
                print(f"Warning: Could not load model from {model_path}: {e}, using untrained model")

        self.model.eval()

    # ── Evaluation ─────────────────────────────────────────────

    def evaluate_position(self, board: ChessBoard) -> float:
        """Evaluate position. Returns score from white's perspective."""
        classical_eval = evaluate_position_advanced(board)

        if self.classical_weight >= 0.99:
            return classical_eval

        board_array = board.board_to_array()
        board_tensor = torch.FloatTensor(board_array).unsqueeze(0)
        board_tensor = board_tensor.permute(0, 3, 1, 2)
        with torch.no_grad():
            nn_eval = self.model(board_tensor).item()
        if not board.get_turn():
            nn_eval = -nn_eval

        alpha = self.classical_weight
        return alpha * classical_eval + (1.0 - alpha) * nn_eval

    # ── Time Management ────────────────────────────────────────

    def _check_time(self) -> bool:
        """Check if we've run out of time."""
        if self.time_limit <= 0:
            return False
        if time.time() - self.start_time >= self.time_limit:
            self.time_up = True
            return True
        return False

    # ── Move Ordering ──────────────────────────────────────────

    def _score_move(self, board: ChessBoard, move: chess.Move, depth: int,
                    tt_move: Optional[chess.Move]) -> int:
        """Score a move for ordering. Higher = searched first."""
        # TT move gets highest priority
        if tt_move and move == tt_move:
            return 10_000_000

        bb = board.board

        # Captures: MVV-LVA
        if bb.is_capture(move):
            captured = bb.piece_at(move.to_square)
            attacker = bb.piece_at(move.from_square)
            victim_val = PIECE_VALUES.get(captured.piece_type, 0) if captured else 0
            attacker_val = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
            return 1_000_000 + 10 * victim_val - attacker_val

        # Promotions
        if move.promotion:
            if move.promotion == chess.QUEEN:
                return 900_000
            return 800_000

        # Killer moves
        if depth < MAX_DEPTH:
            if self.killer_moves[depth][0] == move:
                return 700_000
            if self.killer_moves[depth][1] == move:
                return 600_000

        # History heuristic
        color = 1 if bb.turn else 0
        score = self.history[color][move.from_square][move.to_square]

        # Check bonus
        try:
            if bb.gives_check(move):
                score += 50_000
        except Exception:
            pass

        return score

    def _order_moves(self, board: ChessBoard, moves: List[chess.Move],
                     depth: int, tt_move: Optional[chess.Move] = None) -> List[chess.Move]:
        """Order moves for better alpha-beta pruning."""
        scored = [(self._score_move(board, m, depth, tt_move), m) for m in moves]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def _store_killer(self, move: chess.Move, depth: int):
        """Store a killer move (quiet move that caused cutoff)."""
        if depth >= MAX_DEPTH:
            return
        if self.killer_moves[depth][0] != move:
            self.killer_moves[depth][1] = self.killer_moves[depth][0]
            self.killer_moves[depth][0] = move

    def _update_history(self, board: ChessBoard, move: chess.Move, depth: int):
        """Update history heuristic for a quiet move that caused cutoff."""
        color = 1 if board.board.turn else 0
        self.history[color][move.from_square][move.to_square] += depth * depth

    # ── Transposition Table ────────────────────────────────────

    def _tt_probe(self, key: str, depth: int, alpha: float, beta: float
                  ) -> Tuple[Optional[float], Optional[chess.Move]]:
        """Probe transposition table. Returns (score_or_None, best_move_or_None)."""
        entry = self.transposition_table.get(key)
        if entry is None:
            return None, None

        best_move = entry.best_move

        if entry.depth >= depth:
            if entry.flag == TranspositionEntry.EXACT:
                return entry.score, best_move
            elif entry.flag == TranspositionEntry.ALPHA and entry.score <= alpha:
                return alpha, best_move
            elif entry.flag == TranspositionEntry.BETA and entry.score >= beta:
                return beta, best_move

        return None, best_move

    def _tt_store(self, key: str, depth: int, score: float, flag: int,
                  best_move: Optional[chess.Move]):
        """Store entry in transposition table."""
        if len(self.transposition_table) > self.tt_max_size:
            keys = list(self.transposition_table.keys())
            for k in keys[:len(keys) // 2]:
                del self.transposition_table[k]

        self.transposition_table[key] = TranspositionEntry(
            key, depth, score, flag, best_move)

    # ── Quiescence Search ──────────────────────────────────────

    def quiescence(self, board: ChessBoard, alpha: float, beta: float,
                   depth: int = 0) -> float:
        """Quiescence search to resolve tactical positions."""
        self.nodes_searched += 1

        if self.nodes_searched % 4096 == 0 and self._check_time():
            return 0.0

        stand_pat = self.evaluate_position(board)

        is_white = board.get_turn() == chess.WHITE

        if is_white:
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
        else:
            if stand_pat <= alpha:
                return alpha
            if stand_pat < beta:
                beta = stand_pat

        if depth >= 8:
            return stand_pat

        bb = board.board
        captures = [m for m in bb.legal_moves if bb.is_capture(m)]
        if not captures:
            return stand_pat

        # Order captures by MVV-LVA
        captures = self._order_moves(board, captures, 0)

        for move in captures:
            # Delta pruning
            captured = bb.piece_at(move.to_square)
            if captured:
                gain = PIECE_VALUES.get(captured.piece_type, 0) / 100.0
                if is_white and stand_pat + gain + DELTA_MARGIN < alpha:
                    continue
                if not is_white and stand_pat - gain - DELTA_MARGIN > beta:
                    continue

            board_copy = board.copy()
            board_copy.make_move(move)
            score = self.quiescence(board_copy, alpha, beta, depth + 1)

            if self.time_up:
                return 0.0

            if is_white:
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    break
            else:
                if score < beta:
                    beta = score
                if beta <= alpha:
                    break

        return alpha if is_white else beta

    # ── Alpha-Beta with Enhancements ───────────────────────────

    def _alpha_beta(self, board: ChessBoard, depth: int, alpha: float,
                    beta: float, do_null: bool = True) -> float:
        """Alpha-beta search with NMP, LMR, and PVS."""
        self.nodes_searched += 1

        if self.nodes_searched % 4096 == 0 and self._check_time():
            return 0.0

        bb = board.board
        is_white = bb.turn == chess.WHITE
        in_check = bb.is_check()

        # Check extension
        if in_check:
            depth += 1

        # Base cases
        if depth <= 0:
            return self.quiescence(board, alpha, beta)

        if bb.is_game_over():
            if bb.is_checkmate():
                return -MATE_SCORE if is_white else MATE_SCORE
            return 0.0

        fen_key = bb.fen()

        # TT probe
        tt_score, tt_move = self._tt_probe(fen_key, depth, alpha, beta)
        if tt_score is not None:
            return tt_score

        # ── Null-move pruning ──
        if (do_null and depth >= NMP_MIN_DEPTH and not in_check
                and not self._is_endgame(board)):
            board_copy = board.copy()
            board_copy.board.push(chess.Move.null())
            null_score = self._alpha_beta(board_copy, depth - 1 - NMP_REDUCTION,
                                          alpha, beta, do_null=False)
            if self.time_up:
                return 0.0

            if is_white and null_score >= beta:
                return beta
            if not is_white and null_score <= alpha:
                return alpha

        # Generate and order moves
        legal_moves = list(bb.legal_moves)
        if not legal_moves:
            return self.evaluate_position(board)

        ordered_moves = self._order_moves(board, legal_moves, depth, tt_move)

        best_score = -INFINITY if is_white else INFINITY
        best_move = ordered_moves[0]
        move_count = 0

        for move in ordered_moves:
            move_count += 1
            is_capture = bb.is_capture(move)
            is_promotion = move.promotion is not None
            gives_check = False
            try:
                gives_check = bb.gives_check(move)
            except Exception:
                pass

            board_copy = board.copy()
            board_copy.make_move(move)

            # ── Late-Move Reductions (LMR) ──
            reduction = 0
            if (depth >= LMR_REDUCTION_LIMIT
                    and move_count > LMR_FULL_DEPTH_MOVES
                    and not in_check and not is_capture
                    and not is_promotion and not gives_check):
                reduction = 1
                if move_count > 10:
                    reduction = 2

            # ── PVS / scout search ──
            if move_count == 1:
                score = self._alpha_beta(board_copy, depth - 1, alpha, beta)
            else:
                if is_white:
                    score = self._alpha_beta(board_copy, depth - 1 - reduction,
                                             alpha, alpha + 0.01)
                    if score > alpha and (reduction > 0 or score < beta):
                        score = self._alpha_beta(board_copy, depth - 1, alpha, beta)
                else:
                    score = self._alpha_beta(board_copy, depth - 1 - reduction,
                                             beta - 0.01, beta)
                    if score < beta and (reduction > 0 or score > alpha):
                        score = self._alpha_beta(board_copy, depth - 1, alpha, beta)

            if self.time_up:
                return 0.0

            # Update best
            if is_white:
                if score > best_score:
                    best_score = score
                    best_move = move
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    if not is_capture:
                        self._store_killer(move, depth)
                        self._update_history(board, move, depth)
                    break
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                if score < beta:
                    beta = score
                if beta <= alpha:
                    if not is_capture:
                        self._store_killer(move, depth)
                        self._update_history(board, move, depth)
                    break

        # Store in TT
        if is_white:
            if best_score <= alpha:
                flag = TranspositionEntry.ALPHA
            elif best_score >= beta:
                flag = TranspositionEntry.BETA
            else:
                flag = TranspositionEntry.EXACT
        else:
            if best_score >= beta:
                flag = TranspositionEntry.ALPHA
            elif best_score <= alpha:
                flag = TranspositionEntry.BETA
            else:
                flag = TranspositionEntry.EXACT

        self._tt_store(fen_key, depth, best_score, flag, best_move)

        return best_score

    def _is_endgame(self, board: ChessBoard) -> bool:
        """Check if position is an endgame."""
        bb = board.board
        queens = (len(bb.pieces(chess.QUEEN, chess.WHITE))
                  + len(bb.pieces(chess.QUEEN, chess.BLACK)))
        if queens == 0:
            return True
        total = 0
        for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            total += (len(bb.pieces(pt, chess.WHITE))
                      + len(bb.pieces(pt, chess.BLACK)))
        return total <= 4

    # ── Iterative Deepening ────────────────────────────────────

    def get_best_move(self, board: ChessBoard) -> Optional[chess.Move]:
        """Get best move using iterative deepening.

        Args:
            board: ChessBoard instance

        Returns:
            Best move or None if no legal moves
        """
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Reset search state
        self.nodes_searched = 0
        self.time_up = False
        self.start_time = time.time()
        self.best_move_root = legal_moves[0]

        # Reset killer moves
        self.killer_moves = [[None, None] for _ in range(MAX_DEPTH)]

        # Age history values
        for c in range(2):
            for f in range(64):
                for t in range(64):
                    self.history[c][f][t] //= 2

        is_white = board.get_turn() == chess.WHITE
        best_move = legal_moves[0]
        best_score = -INFINITY if is_white else INFINITY

        move_time = self.time_limit if self.time_limit > 0 else 0
        max_depth = self.depth if move_time <= 0 else MAX_DEPTH

        # Iterative deepening
        for d in range(1, max_depth + 1):
            if self.time_up:
                break

            current_best_move = None
            current_best_score = -INFINITY if is_white else INFINITY

            ordered_moves = self._order_moves(board, legal_moves, d,
                                              self.best_move_root)

            alpha = -INFINITY
            beta = INFINITY

            for move in ordered_moves:
                board_copy = board.copy()
                board_copy.make_move(move)

                score = self._alpha_beta(board_copy, d - 1, alpha, beta)

                if self.time_up:
                    break

                if is_white:
                    if score > current_best_score:
                        current_best_score = score
                        current_best_move = move
                    if score > alpha:
                        alpha = score
                else:
                    if score < current_best_score:
                        current_best_score = score
                        current_best_move = move
                    if score < beta:
                        beta = score

            if not self.time_up and current_best_move:
                best_move = current_best_move
                best_score = current_best_score
                self.best_move_root = best_move

                elapsed = time.time() - self.start_time
                nps = self.nodes_searched / elapsed if elapsed > 0 else 0
                print(f"  depth {d:2d}: {best_move.uci()} "
                      f"score={best_score:+.2f} "
                      f"nodes={self.nodes_searched:,} "
                      f"time={elapsed:.1f}s "
                      f"nps={nps:,.0f}")

                if abs(best_score) >= MATE_SCORE - 100:
                    break

                if move_time > 0 and elapsed > move_time * 0.5:
                    break

        return best_move


def print_board(board: ChessBoard):
    """Print the chess board in a readable format."""
    print("\n" + "=" * 50)
    print(board)
    print("=" * 50)


def play_game(ai_color: str = 'black', ai_depth: int = 5,
              model_path: Optional[str] = None, classical_weight: float = 0.7):
    """Play a game of chess against the AI.

    Args:
        ai_color: 'white' or 'black' - which color the AI plays
        ai_depth: Search depth for AI
        model_path: Path to trained model
    """
    print("Chess Game - Play against AI")
    print("=" * 50)
    print(f"AI plays: {ai_color}")
    print(f"Search depth: {ai_depth}")
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
    parser.add_argument('--depth', type=int, default=5,
                       help='AI search depth (default: 5)')
    parser.add_argument('--model', type=str, default='models/chess_model.pth',
                       help='Path to trained model (default: models/chess_model.pth)')
    parser.add_argument('--classical-weight', type=float, default=0.7,
                       help='Weight for classical evaluation vs neural net (0-1, default: 0.7)')
    parser.add_argument('--time-limit', type=float, default=0.0,
                       help='Time limit per move in seconds (0 = use depth, default: 0)')

    args = parser.parse_args()

    play_game(
        ai_color=args.ai_color,
        ai_depth=args.depth,
        model_path=args.model if args.model else None,
        classical_weight=args.classical_weight
    )


if __name__ == "__main__":
    main()
