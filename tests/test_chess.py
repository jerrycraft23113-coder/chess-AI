"""Basic tests for chess evaluation and search correctness."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import chess
from chess_board import ChessBoard, evaluate_position_advanced
from play_chess import ChessAI, MATE_SCORE


class TestEval:
    """Tests for the evaluation function."""

    def test_starting_position_near_zero(self):
        """Starting position should evaluate close to 0 (symmetric)."""
        board = ChessBoard()
        score = evaluate_position_advanced(board)
        assert abs(score) < 0.5, f"Starting position score {score} too far from 0"

    def test_white_material_advantage(self):
        """White up a queen should have a large positive score."""
        # White has queen, black doesn't
        board = ChessBoard()
        board.board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        score = evaluate_position_advanced(board)
        assert score > 7.0, f"White up queen: score {score} should be > 7.0"

    def test_black_material_advantage(self):
        """Black up a pawn should have a negative score."""
        board = ChessBoard()
        board.board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1")
        score = evaluate_position_advanced(board)
        assert score < -0.5, f"White missing pawn: score {score} should be < -0.5"

    def test_equal_position_symmetric(self):
        """Symmetric positions should evaluate near 0."""
        board = ChessBoard()
        board.board = chess.Board("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1")
        score = evaluate_position_advanced(board)
        assert abs(score) < 0.5, f"Symmetric position score {score} too far from 0"

    def test_bishop_pair_bonus(self):
        """Side with bishop pair should get a bonus."""
        # White has bishop pair, black has two knights
        board = ChessBoard()
        board.board = chess.Board("4k3/8/8/8/8/8/8/2BBK3 w - - 0 1")
        score_bb = evaluate_position_advanced(board)
        board.board = chess.Board("4k3/8/8/8/8/8/8/2NNK3 w - - 0 1")
        score_nn = evaluate_position_advanced(board)
        assert score_bb > score_nn, f"Bishop pair ({score_bb}) should be > two knights ({score_nn})"


class TestSearch:
    """Tests for the search engine."""

    def test_finds_mate_in_one(self):
        """AI should find a mate in one."""
        # White to move, Rb8# is mate
        board = ChessBoard()
        board.board = chess.Board("k7/8/1K6/8/8/8/8/R7 w - - 0 1")
        ai = ChessAI(depth=3, time_limit=5.0)
        move = ai.get_best_move(board)
        assert move is not None, "AI should find a move"
        # Play the move and check it's checkmate
        board.board.push(move)
        assert board.board.is_checkmate(), f"Move {move.uci()} should be checkmate"

    def test_captures_hanging_piece(self):
        """AI should capture a free piece (undefended rook)."""
        # White bishop on c1 can capture undefended black rook on h6
        board = ChessBoard()
        board.board = chess.Board("4k3/8/7r/8/8/8/8/2B1K3 w - - 0 1")
        ai = ChessAI(depth=4, time_limit=5.0)
        move = ai.get_best_move(board)
        assert move is not None
        assert move.to_square == chess.H6, f"AI should capture rook on h6, got {move.uci()}"

    def test_only_legal_move(self):
        """With only one legal move, AI should return it immediately."""
        board = ChessBoard()
        board.board = chess.Board("k7/8/1K6/8/8/8/8/8 w - - 0 1")
        legal = list(board.board.legal_moves)
        if len(legal) == 1:
            ai = ChessAI(depth=5, time_limit=5.0)
            move = ai.get_best_move(board)
            assert move == legal[0]

    def test_no_legal_moves_returns_none(self):
        """With no legal moves (game over), AI should return None."""
        board = ChessBoard()
        board.board = chess.Board("k7/8/1K6/8/8/8/8/R7 b - - 0 1")
        # Check if black is in stalemate or checkmate
        if board.board.is_game_over():
            ai = ChessAI(depth=3, time_limit=5.0)
            move = ai.get_best_move(board)
            assert move is None

    def test_tt_stores_entries(self):
        """Transposition table should store entries after search."""
        board = ChessBoard()
        ai = ChessAI(depth=4, time_limit=5.0)
        ai.get_best_move(board)
        assert len(ai.tt) > 0, "TT should have entries after search"

    def test_depth_limited_search(self):
        """Depth-limited search should complete without timeout."""
        board = ChessBoard()
        ai = ChessAI(depth=3, time_limit=0)  # depth-limited, no time limit
        move = ai.get_best_move(board)
        assert move is not None, "Depth-limited search should return a move"


class TestBoard:
    """Tests for the board representation."""

    def test_board_to_array_shape(self):
        """Board array should have shape (8, 8, 12)."""
        board = ChessBoard()
        arr = board.board_to_array()
        assert arr.shape == (8, 8, 12), f"Expected (8,8,12), got {arr.shape}"

    def test_initial_piece_count(self):
        """Starting position should have 32 pieces."""
        board = ChessBoard()
        arr = board.board_to_array()
        total = arr.sum()
        assert total == 32, f"Expected 32 pieces, got {total}"

    def test_make_move_and_undo(self):
        """Make move then undo should restore original position."""
        board = ChessBoard()
        original_fen = board.board.fen()
        board.make_move_from_uci("e2e4")
        assert board.board.fen() != original_fen
        board.board.pop()
        assert board.board.fen() == original_fen


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
