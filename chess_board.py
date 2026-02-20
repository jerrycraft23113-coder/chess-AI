"""
Chess Board and Game Logic Module
Handles chess board representation, move generation, and game state management.
Vectorized with numpy for maximum performance.
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
        """Convert board to 8x8x12 array using vectorized bitboard extraction.

        Returns:
            8x8x12 numpy array where each channel represents a piece type and color
        """
        board_array = np.zeros((8, 8, 12), dtype=np.float32)
        bb = self.board

        # Vectorized: extract each piece-type bitboard directly
        for channel, (piece_type, color) in enumerate(_PIECE_CHANNELS):
            mask = int(bb.pieces(piece_type, color))
            if mask == 0:
                continue
            # Extract set bit positions
            squares = _bb_to_squares(mask)
            rows = 7 - (squares >> 3)   # 7 - rank (flip for display)
            cols = squares & 7          # file
            board_array[rows, cols, channel] = 1.0

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


# ─── Precomputed constants for vectorized board_to_array ─────────────────────

# Channel mapping: (piece_type, color) for each of 12 channels
_PIECE_CHANNELS = [
    (chess.PAWN, chess.WHITE), (chess.ROOK, chess.WHITE),
    (chess.KNIGHT, chess.WHITE), (chess.BISHOP, chess.WHITE),
    (chess.QUEEN, chess.WHITE), (chess.KING, chess.WHITE),
    (chess.PAWN, chess.BLACK), (chess.ROOK, chess.BLACK),
    (chess.KNIGHT, chess.BLACK), (chess.BISHOP, chess.BLACK),
    (chess.QUEEN, chess.BLACK), (chess.KING, chess.BLACK),
]


def _bb_to_squares(bb_int: int) -> np.ndarray:
    """Convert a bitboard integer to array of square indices. Vectorized."""
    if bb_int == 0:
        return np.array([], dtype=np.int32)
    squares = []
    while bb_int:
        sq = (bb_int & -bb_int).bit_length() - 1
        squares.append(sq)
        bb_int &= bb_int - 1
    return np.array(squares, dtype=np.int32)


# ─── Piece-Square Tables as numpy arrays (vectorized lookup) ────────────────
# From White's perspective, index = (7-rank)*8 + file for white
# For black, index = rank*8 + file

_PAWN_PST_RAW = np.array([
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
], dtype=np.float64)

_KNIGHT_PST_RAW = np.array([
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
], dtype=np.float64)

_BISHOP_PST_RAW = np.array([
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
], dtype=np.float64)

_ROOK_PST_RAW = np.array([
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
], dtype=np.float64)

_QUEEN_PST_RAW = np.array([
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
], dtype=np.float64)

_KING_MG_PST_RAW = np.array([
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
], dtype=np.float64)

_KING_EG_PST_RAW = np.array([
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
], dtype=np.float64)

# ─── Precomputed PST lookup arrays (64 entries each, white indexing) ─────────
# For white: PST index = (7 - rank) * 8 + file = (7 - sq//8) * 8 + sq%8 = 56 - (sq&~7) + (sq&7) = 56 - sq + 2*(sq&7)
# Simpler: for white, the PST index = sq ^ 56 (XOR with 56 flips rank)
# For black: PST index = sq (no flip needed since PST is from white perspective)

# Build WHITE and BLACK indexed PST arrays for all piece types
# _W_PST[piece_type][square] = PST value when a WHITE piece is on square
# _B_PST[piece_type][square] = PST value when a BLACK piece is on square

def _build_indexed_pst(raw_pst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build square-indexed PST arrays for white and black."""
    w_pst = np.zeros(64, dtype=np.float64)
    b_pst = np.zeros(64, dtype=np.float64)
    for sq in range(64):
        rank = sq >> 3
        file = sq & 7
        w_idx = (7 - rank) * 8 + file  # White: flip rank
        b_idx = rank * 8 + file          # Black: no flip
        w_pst[sq] = raw_pst[w_idx]
        b_pst[sq] = raw_pst[b_idx]
    return w_pst, b_pst

# Precompute all PST arrays indexed by square [0..63]
_W_PAWN_PST, _B_PAWN_PST = _build_indexed_pst(_PAWN_PST_RAW)
_W_KNIGHT_PST, _B_KNIGHT_PST = _build_indexed_pst(_KNIGHT_PST_RAW)
_W_BISHOP_PST, _B_BISHOP_PST = _build_indexed_pst(_BISHOP_PST_RAW)
_W_ROOK_PST, _B_ROOK_PST = _build_indexed_pst(_ROOK_PST_RAW)
_W_QUEEN_PST, _B_QUEEN_PST = _build_indexed_pst(_QUEEN_PST_RAW)
_W_KING_MG_PST, _B_KING_MG_PST = _build_indexed_pst(_KING_MG_PST_RAW)
_W_KING_EG_PST, _B_KING_EG_PST = _build_indexed_pst(_KING_EG_PST_RAW)

# Combined PST tables indexed by [piece_type] for fast lookup
# piece_type: 1=PAWN, 2=KNIGHT, 3=BISHOP, 4=ROOK, 5=QUEEN, 6=KING
# Shape: (7, 64) - index 0 unused, 1-6 = piece types
_W_PST_ALL = np.zeros((7, 64), dtype=np.float64)
_B_PST_ALL = np.zeros((7, 64), dtype=np.float64)
_W_PST_ALL[chess.PAWN] = _W_PAWN_PST
_W_PST_ALL[chess.KNIGHT] = _W_KNIGHT_PST
_W_PST_ALL[chess.BISHOP] = _W_BISHOP_PST
_W_PST_ALL[chess.ROOK] = _W_ROOK_PST
_W_PST_ALL[chess.QUEEN] = _W_QUEEN_PST
_B_PST_ALL[chess.PAWN] = _B_PAWN_PST
_B_PST_ALL[chess.KNIGHT] = _B_KNIGHT_PST
_B_PST_ALL[chess.BISHOP] = _B_BISHOP_PST
_B_PST_ALL[chess.ROOK] = _B_ROOK_PST
_B_PST_ALL[chess.QUEEN] = _B_QUEEN_PST

# King PST: middlegame and endgame separate
_W_KING_MG = _W_KING_MG_PST
_W_KING_EG = _W_KING_EG_PST
_B_KING_MG = _B_KING_MG_PST
_B_KING_EG = _B_KING_EG_PST

# Material values as numpy array indexed by piece_type (0=unused, 1-6)
_MATERIAL_VALUES = np.array([0, 100, 320, 330, 500, 900, 0], dtype=np.float64)

# Phase weights as numpy array indexed by piece_type
_PHASE_WEIGHTS = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int32)

# Keep dict versions for play_chess.py compatibility
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

PHASE_WEIGHTS = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}

# ─── Precomputed file/adjacent masks as numpy arrays ────────────────────────

_FILE_MASKS = np.zeros(8, dtype=np.uint64)
for _f in range(8):
    _mask = np.uint64(0)
    for _r in range(8):
        _mask |= np.uint64(1 << (_r * 8 + _f))
    _FILE_MASKS[_f] = _mask

_ADJ_FILE_MASKS = np.zeros(8, dtype=np.uint64)
for _f in range(8):
    _mask = np.uint64(0)
    if _f > 0:
        _mask |= _FILE_MASKS[_f - 1]
    if _f < 7:
        _mask |= _FILE_MASKS[_f + 1]
    _ADJ_FILE_MASKS[_f] = _mask

# ─── Center control mask (d4, e4, d5, e5) ──────────────────────────────────
_CENTER_MASK = (1 << chess.D4) | (1 << chess.E4) | (1 << chess.D5) | (1 << chess.E5)

# ─── Precomputed passed pawn masks ──────────────────────────────────────────
# For each square, the mask of squares where enemy pawns would block a passed pawn

_WHITE_PASSED_MASKS = np.zeros(64, dtype=np.uint64)
_BLACK_PASSED_MASKS = np.zeros(64, dtype=np.uint64)

for _sq in range(64):
    _f = _sq & 7
    _r = _sq >> 3
    # White passed: check ranks above on file and adjacent files
    _wmask = np.uint64(0)
    for _ahead_r in range(_r + 1, 8):
        for _df in (_f - 1, _f, _f + 1):
            if 0 <= _df <= 7:
                _wmask |= np.uint64(1 << (_ahead_r * 8 + _df))
    _WHITE_PASSED_MASKS[_sq] = _wmask
    # Black passed: check ranks below
    _bmask = np.uint64(0)
    for _ahead_r in range(_r - 1, -1, -1):
        for _df in (_f - 1, _f, _f + 1):
            if 0 <= _df <= 7:
                _bmask |= np.uint64(1 << (_ahead_r * 8 + _df))
    _BLACK_PASSED_MASKS[_sq] = _bmask

# ─── Precomputed king pawn shield squares ───────────────────────────────────
# For each king square, list of (shield_sq, score_factor) for each color

_WHITE_SHIELD_1 = {}  # king_sq -> list of shield squares 1 rank ahead
_WHITE_SHIELD_2 = {}  # king_sq -> list of shield squares 2 ranks ahead
_BLACK_SHIELD_1 = {}
_BLACK_SHIELD_2 = {}

for _sq in range(64):
    _kf = _sq & 7
    _kr = _sq >> 3
    w1, w2, b1, b2 = [], [], [], []
    for _df in (-1, 0, 1):
        _ff = _kf + _df
        if _ff < 0 or _ff > 7:
            continue
        r1w = _kr + 1
        if 0 <= r1w <= 7:
            w1.append(chess.square(_ff, r1w))
        r2w = _kr + 2
        if 0 <= r2w <= 7:
            w2.append(chess.square(_ff, r2w))
        r1b = _kr - 1
        if 0 <= r1b <= 7:
            b1.append(chess.square(_ff, r1b))
        r2b = _kr - 2
        if 0 <= r2b <= 7:
            b2.append(chess.square(_ff, r2b))
    _WHITE_SHIELD_1[_sq] = w1
    _WHITE_SHIELD_2[_sq] = w2
    _BLACK_SHIELD_1[_sq] = b1
    _BLACK_SHIELD_2[_sq] = b2


# ─── Popcount - use fast int.bit_count() (Python 3.10+) or fallback ────────

try:
    (0).bit_count()  # Test if available
    def _popcount(bb_mask: int) -> int:
        return bb_mask.bit_count()
except AttributeError:
    def _popcount(bb_mask: int) -> int:
        return bin(bb_mask).count('1')


# ─── Legacy simple evaluation (kept for backward compat) ────────────────────

def evaluate_position_simple(board: ChessBoard) -> float:
    """Simple evaluation function based on piece values."""
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }
    score = 0.0
    for sq, p in board.board.piece_map().items():
        v = piece_values[p.piece_type]
        score += v if p.color == chess.WHITE else -v
    if board.is_checkmate():
        score = -1000 if board.get_turn() == chess.WHITE else 1000
    return score


# ─── Fast Python lists for hot-path eval (precomputed from numpy arrays) ────
# List lookup is ~3x faster than numpy indexing for single elements

_W_PST_LIST = [[0.0]*64 for _ in range(7)]  # [piece_type][square]
_B_PST_LIST = [[0.0]*64 for _ in range(7)]
for _pt in range(1, 6):  # PAWN..QUEEN
    for _sq in range(64):
        _W_PST_LIST[_pt][_sq] = float(_W_PST_ALL[_pt][_sq])
        _B_PST_LIST[_pt][_sq] = float(_B_PST_ALL[_pt][_sq])

_W_KING_MG_LIST = [float(_W_KING_MG[i]) for i in range(64)]
_W_KING_EG_LIST = [float(_W_KING_EG[i]) for i in range(64)]
_B_KING_MG_LIST = [float(_B_KING_MG[i]) for i in range(64)]
_B_KING_EG_LIST = [float(_B_KING_EG[i]) for i in range(64)]

_MAT_LIST = [0.0, 100.0, 320.0, 330.0, 500.0, 900.0, 0.0]  # indexed by piece_type
_PHASE_LIST = [0, 0, 1, 1, 2, 4, 0]  # indexed by piece_type

# Passed pawn masks as plain Python ints (avoid numpy uint64 conversion overhead)
_W_PASSED_INT = [int(_WHITE_PASSED_MASKS[i]) for i in range(64)]
_B_PASSED_INT = [int(_BLACK_PASSED_MASKS[i]) for i in range(64)]
_ADJ_FILE_INT = [int(_ADJ_FILE_MASKS[i]) for i in range(8)]


# ─── Lightweight eval: material + PST only (for quiescence search) ────────

def evaluate_material_pst(bb) -> float:
    """Fast material + PST eval for quiescence. Skips pawn structure/king safety.
    Takes a chess.Board directly (not ChessBoard wrapper). Returns score in pawns from white's perspective.
    """
    w_occ = int(bb.occupied_co[True])
    b_occ = int(bb.occupied_co[False])
    score = 0.0
    w_pst = _W_PST_LIST
    b_pst = _B_PST_LIST

    # Pawns
    tmp = int(bb.pawns) & w_occ
    wp1 = w_pst[1]
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score += 100.0 + wp1[sq]
    tmp = int(bb.pawns) & b_occ
    bp1 = b_pst[1]
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score -= 100.0 + bp1[sq]

    # Knights
    tmp = int(bb.knights) & w_occ
    wp2 = w_pst[2]
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score += 320.0 + wp2[sq]
    tmp = int(bb.knights) & b_occ
    bp2 = b_pst[2]
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score -= 320.0 + bp2[sq]

    # Bishops
    tmp = int(bb.bishops) & w_occ
    wp3 = w_pst[3]
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score += 330.0 + wp3[sq]
    tmp = int(bb.bishops) & b_occ
    bp3 = b_pst[3]
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score -= 330.0 + bp3[sq]

    # Rooks
    tmp = int(bb.rooks) & w_occ
    wp4 = w_pst[4]
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score += 500.0 + wp4[sq]
    tmp = int(bb.rooks) & b_occ
    bp4 = b_pst[4]
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score -= 500.0 + bp4[sq]

    # Queens
    tmp = int(bb.queens) & w_occ
    wp5 = w_pst[5]
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score += 900.0 + wp5[sq]
    tmp = int(bb.queens) & b_occ
    bp5 = b_pst[5]
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score -= 900.0 + bp5[sq]

    # Tempo
    score += 10.0 if bb.turn else -10.0
    return score * 0.01


# ─── Main evaluation: maximum speed, bitboard-only ────────────────────────

def evaluate_position_advanced(board: ChessBoard) -> float:
    """Ultra-fast classical evaluation. No is_checkmate/is_game_over calls.
    The search engine handles game-over detection separately.
    Returns score from white's perspective (pawns).
    """
    bb = board.board

    # ── Per-piece-type bitboard iteration (avoids piece_type_at calls) ──
    w_occ = int(bb.occupied_co[True])   # chess.WHITE
    b_occ = int(bb.occupied_co[False])  # chess.BLACK

    phase = 0
    score = 0.0
    w_bishops = 0
    b_bishops = 0
    w_minors = 0
    b_minors = 0

    # Local refs
    w_pst = _W_PST_LIST
    b_pst = _B_PST_LIST

    # ── Kings (always exactly one per side) ──
    wk_bb = int(bb.kings) & w_occ
    wk_sq = (wk_bb & -wk_bb).bit_length() - 1
    bk_bb = int(bb.kings) & b_occ
    bk_sq = (bk_bb & -bk_bb).bit_length() - 1

    # ── White pawns ──
    w_pawns_int = int(bb.pawns) & w_occ
    w_pawns_by_file = [0, 0, 0, 0, 0, 0, 0, 0]
    wp1 = w_pst[1]
    tmp = w_pawns_int
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score += 100.0 + wp1[sq]
        w_pawns_by_file[sq & 7] += 1

    # ── Black pawns ──
    b_pawns_int = int(bb.pawns) & b_occ
    b_pawns_by_file = [0, 0, 0, 0, 0, 0, 0, 0]
    bp1 = b_pst[1]
    tmp = b_pawns_int
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score -= 100.0 + bp1[sq]
        b_pawns_by_file[sq & 7] += 1

    # ── White knights (phase +1 each, minor) ──
    w_knights_bb = int(bb.knights) & w_occ
    wp2 = w_pst[2]
    tmp = w_knights_bb
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score += 320.0 + wp2[sq]
        phase += 1
        w_minors += 1

    # ── Black knights ──
    b_knights_bb = int(bb.knights) & b_occ
    bp2 = b_pst[2]
    tmp = b_knights_bb
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score -= 320.0 + bp2[sq]
        phase += 1
        b_minors += 1

    # ── White bishops (phase +1 each, minor) ──
    w_bishops_bb = int(bb.bishops) & w_occ
    wp3 = w_pst[3]
    tmp = w_bishops_bb
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score += 330.0 + wp3[sq]
        phase += 1
        w_bishops += 1
        w_minors += 1

    # ── Black bishops ──
    b_bishops_bb = int(bb.bishops) & b_occ
    bp3 = b_pst[3]
    tmp = b_bishops_bb
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score -= 330.0 + bp3[sq]
        phase += 1
        b_bishops += 1
        b_minors += 1

    # ── White rooks (phase +2 each) ──
    w_rooks_bb = int(bb.rooks) & w_occ
    wp4 = w_pst[4]
    w_rook_files = []
    tmp = w_rooks_bb
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score += 500.0 + wp4[sq]
        phase += 2
        w_rook_files.append(sq & 7)

    # ── Black rooks ──
    b_rooks_bb = int(bb.rooks) & b_occ
    bp4 = b_pst[4]
    b_rook_files = []
    tmp = b_rooks_bb
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score -= 500.0 + bp4[sq]
        phase += 2
        b_rook_files.append(sq & 7)

    # ── White queens (phase +4 each) ──
    w_queens_bb = int(bb.queens) & w_occ
    wp5 = w_pst[5]
    tmp = w_queens_bb
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score += 900.0 + wp5[sq]
        phase += 4

    # ── Black queens ──
    b_queens_bb = int(bb.queens) & b_occ
    bp5 = b_pst[5]
    tmp = b_queens_bb
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        score -= 900.0 + bp5[sq]
        phase += 4

    # ── Game phase and king PST ──
    if phase > 24:
        phase = 24
    eg_w = 1.0 - phase * 0.041666666666666664  # 1/24
    mg_w = 1.0 - eg_w

    score += _W_KING_MG_LIST[wk_sq] * mg_w + _W_KING_EG_LIST[wk_sq] * eg_w
    score -= _B_KING_MG_LIST[bk_sq] * mg_w + _B_KING_EG_LIST[bk_sq] * eg_w

    # ── Bishop pair bonus ──
    if w_bishops >= 2:
        score += 50.0
    if b_bishops >= 2:
        score -= 50.0

    # ── Pawn structure ──
    adj = _ADJ_FILE_INT
    w_pbf = w_pawns_by_file
    b_pbf = b_pawns_by_file
    for f in range(8):
        wp = w_pbf[f]; bp = b_pbf[f]
        a = adj[f]
        if wp > 1: score -= 20.0 * (wp - 1)
        if bp > 1: score += 20.0 * (bp - 1)
        if wp and not (w_pawns_int & a): score -= 15.0 * wp
        if bp and not (b_pawns_int & a): score += 15.0 * bp

    # ── Passed pawns ──
    w_passed = _W_PASSED_INT
    b_passed = _B_PASSED_INT

    if w_pawns_int:
        tmp = w_pawns_int
        while tmp:
            sq = (tmp & -tmp).bit_length() - 1
            if not (b_pawns_int & w_passed[sq]):
                score += 20.0 + 10.0 * (sq >> 3)
            tmp &= tmp - 1

    if b_pawns_int:
        tmp = b_pawns_int
        while tmp:
            sq = (tmp & -tmp).bit_length() - 1
            if not (w_pawns_int & b_passed[sq]):
                score -= 20.0 + 10.0 * (7 - (sq >> 3))
            tmp &= tmp - 1

    # ── Rook on open / semi-open files ──
    for f in w_rook_files:
        if w_pbf[f] == 0:
            score += 25.0 if b_pbf[f] == 0 else 15.0
    for f in b_rook_files:
        if b_pbf[f] == 0:
            score -= 25.0 if w_pbf[f] == 0 else 15.0

    # ── Center control ──
    w_center = _popcount(w_occ & _CENTER_MASK)
    b_center = _popcount(b_occ & _CENTER_MASK)
    score += (w_center - b_center) * 15.0

    # ── Mobility approximation (minor piece development proxy) ──
    score += (w_minors - b_minors) * 5.0

    # ── King safety (pawn shield + semi-open file penalty) ──
    if eg_w < 0.6:
        sf = 1.0 - eg_w
        # White king pawn shield + open file penalties
        kf = wk_sq & 7
        for df in range(-1, 2):
            ff = kf + df
            if 0 > ff or ff > 7:
                continue
            if w_pbf[ff] == 0:
                score -= 15.0 * sf
                if b_pbf[ff] == 0:
                    score -= 5.0 * sf
                else:
                    score -= 10.0 * sf
        # Black king pawn shield + open file penalties
        kf = bk_sq & 7
        for df in range(-1, 2):
            ff = kf + df
            if 0 > ff or ff > 7:
                continue
            if b_pbf[ff] == 0:
                score += 15.0 * sf
                if w_pbf[ff] == 0:
                    score += 5.0 * sf
                else:
                    score += 10.0 * sf

    # ── Tempo bonus (small bonus for side to move) ──
    score += 10.0 if bb.turn else -10.0

    # Convert centipawns to pawns
    return score * 0.01


def _iter_bits(bb_int: int):
    """Iterate over set bits in a bitboard integer. Yields square indices."""
    while bb_int:
        sq = (bb_int & -bb_int).bit_length() - 1
        yield sq
        bb_int &= bb_int - 1
