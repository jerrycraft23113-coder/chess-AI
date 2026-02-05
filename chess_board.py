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
        except:
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
        chess.KING: 0  # King value not used in material count
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
    
    # Add bonus for checkmate
    if board.is_checkmate():
        if board.get_turn() == chess.WHITE:
            score = -1000  # Black wins
        else:
            score = 1000  # White wins
    
    return score


def evaluate_position_advanced(board: ChessBoard) -> float:
    """Advanced evaluation with simple positional terms.

    Still returns score from white's perspective.
    """
    score = evaluate_position_simple(board)

    # --- Center control (bonus for occupying central squares) ---
    central_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    for sq in central_squares:
        piece = board.board.piece_at(sq)
        if not piece:
            continue
        bonus = 0.2
        if piece.color == chess.WHITE:
            score += bonus
        else:
            score -= bonus

    # --- Mobility: number of legal moves for each side ---
    original_turn = board.board.turn

    board.board.turn = chess.WHITE
    white_mobility = board.board.legal_moves.count()

    board.board.turn = chess.BLACK
    black_mobility = board.board.legal_moves.count()

    board.board.turn = original_turn

    mobility_factor = 0.05
    score += mobility_factor * (white_mobility - black_mobility)

    # --- King safety (very simple heuristic) ---
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    total_material = 0.0
    for sq in chess.SQUARES:
        p = board.board.piece_at(sq)
        if p and p.piece_type in piece_values:
            total_material += piece_values[p.piece_type]

    # Only care about king safety when there is enough material on board
    if total_material >= 14:
        white_king_sq = board.board.king(chess.WHITE)
        black_king_sq = board.board.king(chess.BLACK)

        def king_safety_penalty(king_sq: Optional[int], is_white: bool) -> float:
            if king_sq is None:
                return 0.0
            file_index = chess.square_file(king_sq)  # 0..7 for a..h
            rank_index = chess.square_rank(king_sq)  # 0..7 for 1..8

            penalty = 0.0
            # Penalize king in the center files (d/e) in middlegame
            if 2 <= file_index <= 5 and 2 <= rank_index <= 5:
                penalty += 0.5

            # Small bonus if king appears "castled" on g- or c-file
            if (is_white and rank_index == 0) or (not is_white and rank_index == 7):
                if file_index in (1, 2, 5, 6):  # b,c or f,g files
                    penalty -= 0.2

            return penalty

        score -= king_safety_penalty(white_king_sq, True)
        score += king_safety_penalty(black_king_sq, False)

    return score
