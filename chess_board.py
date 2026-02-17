"""
Chess Board and Game Logic Module
Handles chess board representation, move generation, and game state management.
"""

import chess
import numpy as np
from typing import List, Tuple, Optional


class ChessBoard:
    """Wrapper around python-chess for easier integration with ML models."""
    
    def __init__(self, fen: Optional[str] = None):
        """Initialize chess board.
        
        Args:
            fen: FEN string to initialize board position (default: starting position)
        """
        self.board = chess.Board(fen) if fen else chess.Board()
    
    def get_legal_moves(self) -> List[chess.Move]:
        """Get all legal moves for current position."""
        return list(self.board.legal_moves)
    
    def make_move(self, move: chess.Move) -> bool:
        """Make a move on the board.
        
        Args:
            move: Chess move to make
            
        Returns:
            True if move was legal and made, False otherwise
        """
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False
    
    def make_move_from_uci(self, uci: str) -> bool:
        """Make a move from UCI notation.
        
        Args:
            uci: Move in UCI notation (e.g., "e2e4")
            
        Returns:
            True if move was legal and made, False otherwise
        """
        try:
            move = chess.Move.from_uci(uci)
            return self.make_move(move)
        except (ValueError, chess.InvalidMoveError):
            return False
    
    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.board.is_game_over()
    
    def get_result(self) -> Optional[str]:
        """Get game result: '1-0', '0-1', '1/2-1/2', or None if game not over."""
        if not self.is_game_over():
            return None
        return self.board.result()
    
    def is_check(self) -> bool:
        """Check if current player is in check."""
        return self.board.is_check()
    
    def is_checkmate(self) -> bool:
        """Check if current player is in checkmate."""
        return self.board.is_checkmate()
    
    def is_stalemate(self) -> bool:
        """Check if current position is stalemate."""
        return self.board.is_stalemate()
    
    def get_turn(self) -> bool:
        """Get current player: True for white, False for black."""
        return self.board.turn
    
    def board_to_array(self) -> np.ndarray:
        """Convert board to 8x8x12 array representation for neural network.
        
        Returns:
            8x8x12 numpy array where each channel represents a piece type and color
        """
        # 12 channels: 6 piece types * 2 colors
        # Order: White Pawn, White Rook, White Knight, White Bishop, White Queen, White King,
        #        Black Pawn, Black Rook, Black Knight, Black Bishop, Black Queen, Black King
        board_array = np.zeros((8, 8, 12), dtype=np.float32)
        
        piece_to_channel = {
            chess.PAWN: 0,
            chess.ROOK: 1,
            chess.KNIGHT: 2,
            chess.BISHOP: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = 7 - (square // 8)  # Flip vertically for standard board orientation
                col = square % 8
                color_offset = 0 if piece.color == chess.WHITE else 6
                channel = piece_to_channel[piece.piece_type] + color_offset
                board_array[row, col, channel] = 1.0
        
        return board_array
    
    def get_fen(self) -> str:
        """Get FEN representation of current position."""
        return self.board.fen()
    
    def copy(self) -> 'ChessBoard':
        """Create a copy of the board."""
        return ChessBoard(self.get_fen())
    
    def __str__(self) -> str:
        """String representation of the board."""
        return str(self.board)


# ─── Piece-Square Tables (from White's perspective, rank 1 = index 0) ──────
# Values in centipawns. For Black, we mirror vertically.

PAWN_PST = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]

KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_PST = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
]

QUEEN_PST = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]

KING_MIDDLEGAME_PST = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
]

KING_ENDGAME_PST = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]

PST_TABLES = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
}

# Material values in centipawns
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Phase weights for tapered eval (total = 24 at start)
PHASE_WEIGHTS = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}


def _pst_score(piece_type: int, square: int, is_white: bool, endgame_weight: float) -> float:
    """Get PST bonus for a piece (centipawns). square is 0-63, a1=0."""
    if is_white:
        idx = (7 - chess.square_rank(square)) * 8 + chess.square_file(square)
    else:
        idx = chess.square_rank(square) * 8 + chess.square_file(square)

    if piece_type == chess.KING:
        mg = KING_MIDDLEGAME_PST[idx]
        eg = KING_ENDGAME_PST[idx]
        return mg * (1.0 - endgame_weight) + eg * endgame_weight
    elif piece_type in PST_TABLES:
        return float(PST_TABLES[piece_type][idx])
    return 0.0


def evaluate_position_simple(board: ChessBoard) -> float:
    """Simple evaluation function based on piece values.

    Args:
        board: ChessBoard instance

    Returns:
        Evaluation score from white's perspective (positive = white better)
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }

    score = 0.0
    for square in chess.SQUARES:
        piece = board.board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value

    if board.is_checkmate():
        if board.get_turn() == chess.WHITE:
            score = -1000
        else:
            score = 1000

    return score


def evaluate_position_advanced(board: ChessBoard) -> float:
    """Strong classical evaluation with PST, pawn structure, king safety, etc.

    Returns score from white's perspective in centipawns (1 pawn = 100).
    """
    bb = board.board

    # ── Checkmate / Stalemate ──
    if bb.is_checkmate():
        return -30000.0 if bb.turn == chess.WHITE else 30000.0
    if bb.is_stalemate() or bb.is_insufficient_material():
        return 0.0

    # ── Game-phase for tapered eval ──
    phase = 0
    for sq in chess.SQUARES:
        p = bb.piece_at(sq)
        if p and p.piece_type in PHASE_WEIGHTS:
            phase += PHASE_WEIGHTS[p.piece_type]
    phase = min(phase, 24)
    endgame_weight = 1.0 - (phase / 24.0)  # 0 = full middlegame, 1 = pure endgame

    score = 0.0

    # ── Material + PST ──
    white_bishops = 0
    black_bishops = 0
    white_pawns_by_file = [0] * 8
    black_pawns_by_file = [0] * 8
    white_pawn_squares: List[int] = []
    black_pawn_squares: List[int] = []

    for sq in chess.SQUARES:
        p = bb.piece_at(sq)
        if not p:
            continue
        mat = PIECE_VALUES.get(p.piece_type, 0)
        pst = _pst_score(p.piece_type, sq, p.color == chess.WHITE, endgame_weight)

        if p.color == chess.WHITE:
            score += mat + pst
            if p.piece_type == chess.BISHOP:
                white_bishops += 1
            if p.piece_type == chess.PAWN:
                white_pawns_by_file[chess.square_file(sq)] += 1
                white_pawn_squares.append(sq)
        else:
            score -= mat + pst
            if p.piece_type == chess.BISHOP:
                black_bishops += 1
            if p.piece_type == chess.PAWN:
                black_pawns_by_file[chess.square_file(sq)] += 1
                black_pawn_squares.append(sq)

    # ── Bishop pair bonus ──
    if white_bishops >= 2:
        score += 50
    if black_bishops >= 2:
        score -= 50

    # ── Doubled pawns penalty ──
    for f in range(8):
        if white_pawns_by_file[f] > 1:
            score -= 20 * (white_pawns_by_file[f] - 1)
        if black_pawns_by_file[f] > 1:
            score += 20 * (black_pawns_by_file[f] - 1)

    # ── Isolated pawns penalty ──
    for f in range(8):
        has_left = (f > 0 and white_pawns_by_file[f - 1] > 0)
        has_right = (f < 7 and white_pawns_by_file[f + 1] > 0)
        if white_pawns_by_file[f] > 0 and not has_left and not has_right:
            score -= 15 * white_pawns_by_file[f]

        has_left_b = (f > 0 and black_pawns_by_file[f - 1] > 0)
        has_right_b = (f < 7 and black_pawns_by_file[f + 1] > 0)
        if black_pawns_by_file[f] > 0 and not has_left_b and not has_right_b:
            score += 15 * black_pawns_by_file[f]

    # ── Passed pawns bonus ──
    for sq in white_pawn_squares:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        is_passed = True
        for br in range(r + 1, 8):
            for df in (f - 1, f, f + 1):
                if 0 <= df <= 7:
                    bsq = chess.square(df, br)
                    bp = bb.piece_at(bsq)
                    if bp and bp.piece_type == chess.PAWN and bp.color == chess.BLACK:
                        is_passed = False
                        break
            if not is_passed:
                break
        if is_passed:
            score += 20 + 10 * r  # More advanced = bigger bonus

    for sq in black_pawn_squares:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        is_passed = True
        for br in range(r - 1, -1, -1):
            for df in (f - 1, f, f + 1):
                if 0 <= df <= 7:
                    bsq = chess.square(df, br)
                    wp = bb.piece_at(bsq)
                    if wp and wp.piece_type == chess.PAWN and wp.color == chess.WHITE:
                        is_passed = False
                        break
            if not is_passed:
                break
        if is_passed:
            score -= 20 + 10 * (7 - r)

    # ── Rook on open / semi-open files ──
    for sq in chess.SQUARES:
        p = bb.piece_at(sq)
        if not p or p.piece_type != chess.ROOK:
            continue
        f = chess.square_file(sq)
        w_pawns_on_file = white_pawns_by_file[f]
        b_pawns_on_file = black_pawns_by_file[f]
        if p.color == chess.WHITE:
            if w_pawns_on_file == 0 and b_pawns_on_file == 0:
                score += 25  # open file
            elif w_pawns_on_file == 0:
                score += 15  # semi-open
        else:
            if w_pawns_on_file == 0 and b_pawns_on_file == 0:
                score -= 25
            elif b_pawns_on_file == 0:
                score -= 15

    # ── Mobility ──
    original_turn = bb.turn
    try:
        bb.turn = chess.WHITE
        white_mobility = bb.legal_moves.count()
        bb.turn = chess.BLACK
        black_mobility = bb.legal_moves.count()
    finally:
        bb.turn = original_turn

    score += 4 * (white_mobility - black_mobility)

    # ── King safety (middlegame only) ──
    if endgame_weight < 0.6:
        safety_factor = 1.0 - endgame_weight
        wk = bb.king(chess.WHITE)
        bk = bb.king(chess.BLACK)

        def _pawn_shield(king_sq: int, color: bool) -> float:
            """Count friendly pawns near king."""
            if king_sq is None:
                return 0.0
            shield = 0.0
            kf = chess.square_file(king_sq)
            kr = chess.square_rank(king_sq)
            pawn_rank_dir = 1 if color == chess.WHITE else -1
            for df in (-1, 0, 1):
                ff = kf + df
                if ff < 0 or ff > 7:
                    continue
                for dr in (1, 2):
                    rr = kr + pawn_rank_dir * dr
                    if 0 <= rr <= 7:
                        sq2 = chess.square(ff, rr)
                        p2 = bb.piece_at(sq2)
                        if p2 and p2.piece_type == chess.PAWN and p2.color == color:
                            shield += 15.0 if dr == 1 else 8.0
            return shield

        score += _pawn_shield(wk, chess.WHITE) * safety_factor
        score -= _pawn_shield(bk, chess.BLACK) * safety_factor

        # Penalize open files near king
        if wk is not None:
            wkf = chess.square_file(wk)
            for df in (-1, 0, 1):
                ff = wkf + df
                if 0 <= ff <= 7 and white_pawns_by_file[ff] == 0:
                    score -= 15 * safety_factor
        if bk is not None:
            bkf = chess.square_file(bk)
            for df in (-1, 0, 1):
                ff = bkf + df
                if 0 <= ff <= 7 and black_pawns_by_file[ff] == 0:
                    score += 15 * safety_factor

    # ── Check bonus ──
    if bb.is_check():
        if bb.turn == chess.WHITE:
            score -= 20  # Black is giving check (bad for white)
        else:
            score += 20

    # Convert centipawns to pawns for compatibility
    return score / 100.0
