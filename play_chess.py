"""
Chess AI Engine - Maximum speed optimized.
Negamax with Zobrist hashing, futility pruning, aggressive LMR,
aspiration windows, and minimal Python overhead.
"""

import time
import chess
import numpy as np
from typing import Optional, List, Tuple
import random as _random
import math

from chess_board import ChessBoard, evaluate_position_advanced, PIECE_VALUES

# Try torch for neural net, but make it optional
try:
    import torch
    from chess_model import ChessCNN
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────
MATE_SCORE = 30000
MAX_DEPTH = 64
INFINITY = 999999

# ── Precomputed MVV-LVA as plain Python list ───────────────────
_PV = [0, 100, 320, 330, 500, 900, 20000]
_MVV_LVA = [[0]*7 for _ in range(7)]
for _v in range(1, 7):
    for _a in range(1, 7):
        _MVV_LVA[_v][_a] = 10 * _PV[_v] - _PV[_a]

# ── Zobrist hashing ───────────────────────────────────────────
_rng = _random.Random(42)
_ZOBRIST_PIECES = [[[_rng.getrandbits(64) for _ in range(64)] for _ in range(7)] for _ in range(2)]
_ZOBRIST_TURN = _rng.getrandbits(64)
_ZOBRIST_CASTLE = [_rng.getrandbits(64) for _ in range(4)]
_ZOBRIST_EP = [_rng.getrandbits(64) for _ in range(8)]

# Flatten to tuples for faster access
_ZP_W = [None] + [tuple(_ZOBRIST_PIECES[1][pt]) for pt in range(1, 7)]
_ZP_B = [None] + [tuple(_ZOBRIST_PIECES[0][pt]) for pt in range(1, 7)]
_ZC = tuple(_ZOBRIST_CASTLE)
_ZE = tuple(_ZOBRIST_EP)
_ZT = _ZOBRIST_TURN

_BB_H1 = chess.BB_H1
_BB_A1 = chess.BB_A1
_BB_H8 = chess.BB_H8
_BB_A8 = chess.BB_A8


def _zobrist_hash(bb: chess.Board) -> int:
    """Compute Zobrist hash using bitboard iteration."""
    h = 0
    pta = bb.piece_type_at
    zpw = _ZP_W
    zpb = _ZP_B
    # White pieces
    tmp = int(bb.occupied_co[True])
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        h ^= zpw[pta(sq)][sq]
    # Black pieces
    tmp = int(bb.occupied_co[False])
    while tmp:
        sq = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        h ^= zpb[pta(sq)][sq]
    # Turn
    if bb.turn:
        h ^= _ZT
    # Castling
    cr = bb.castling_rights
    if cr & _BB_H1: h ^= _ZC[0]
    if cr & _BB_A1: h ^= _ZC[1]
    if cr & _BB_H8: h ^= _ZC[2]
    if cr & _BB_A8: h ^= _ZC[3]
    # En passant
    ep = bb.ep_square
    if ep is not None:
        h ^= _ZE[ep & 7]
    return h


# ── Futility/LMP margins ──────────────────────────────────────
_FUTILITY_MARGIN = [0, 200, 350, 500]  # depth 0,1,2,3
_LMP_COUNTS = [0, 5, 8, 13, 20]  # late move pruning thresholds per depth

# Precompute LMR reduction table (log formula)
_LMR_TABLE = [[0]*64 for _ in range(MAX_DEPTH)]
for _d in range(1, MAX_DEPTH):
    for _m in range(1, 64):
        _LMR_TABLE[_d][_m] = max(0, int(0.75 + math.log(_d) * math.log(_m) * 0.4))

# ── Thin wrapper for eval ─────────────────────────────────────
_EVAL_WRAPPER = ChessBoard.__new__(ChessBoard)

def _eval_fast(bb: chess.Board) -> float:
    """Evaluate from WHITE perspective."""
    _EVAL_WRAPPER.board = bb
    return evaluate_position_advanced(_EVAL_WRAPPER)


class ChessAI:
    """Chess AI - Negamax with maximum pruning for depth 10 in 15s."""

    def __init__(self, model_path: Optional[str] = None, depth: int = 5,
                 classical_weight: float = 0.7, time_limit: float = 0.0):
        self.depth = depth
        self.classical_weight = max(0.0, min(1.0, classical_weight))
        self.time_limit = time_limit

        # Transposition table
        self.tt: dict = {}
        self.tt_max_size = 4_000_000

        # Killer moves
        self.killers = [[None, None] for _ in range(MAX_DEPTH)]

        # History heuristic
        self.hist_w = np.zeros((64, 64), dtype=np.int32)
        self.hist_b = np.zeros((64, 64), dtype=np.int32)

        # Counter-move heuristic
        self.counter_moves: dict = {}

        # Search state
        self.nodes = 0
        self.start_time = 0.0
        self.time_up = False
        self.best_root = None

        # Neural net (optional)
        self.model = None
        if TORCH_AVAILABLE:
            self.model = ChessCNN(hidden_size=256)
            if model_path:
                try:
                    self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    print(f"Loaded model from {model_path}")
                except (FileNotFoundError, RuntimeError, ValueError) as e:
                    print(f"Warning: Could not load model: {e}")
            self.model.eval()

    # GUI compatibility properties
    @property
    def nodes_searched(self):
        return self.nodes

    @property
    def transposition_table(self):
        return self.tt

    # ── Quiescence ──────────────────────────────────────────────

    def _quiesce(self, bb, alpha, beta, qdepth=0):
        self.nodes += 1
        if (self.nodes & 4095) == 0:
            if self.time_limit > 0 and time.time() - self.start_time >= self.time_limit:
                self.time_up = True
                return 0.0
            time.sleep(0)

        # Stand-pat
        score = _eval_fast(bb)
        if not bb.turn:
            score = -score

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

        if qdepth >= 3:
            return alpha

        # Captures only, sorted by MVV-LVA inline
        mvvlva = _MVV_LVA
        pta = bb.piece_type_at
        caps = []
        idx = 0
        for m in bb.legal_moves:
            if bb.is_capture(m):
                c_pt = pta(m.to_square)
                a_pt = pta(m.from_square)
                if c_pt and a_pt:
                    caps.append((mvvlva[c_pt][a_pt], idx, m))
                else:
                    caps.append((0, idx, m))
                idx += 1
        if not caps:
            return alpha

        caps.sort(reverse=True)

        pv = _PV
        for _, _, move in caps:
            # Delta pruning
            c_pt = pta(move.to_square)
            if c_pt:
                if score + pv[c_pt] * 0.01 + 2.0 < alpha:
                    continue

            bb.push(move)
            s = -self._quiesce(bb, -beta, -alpha, qdepth + 1)
            bb.pop()

            if self.time_up:
                return 0.0
            if s >= beta:
                return beta
            if s > alpha:
                alpha = s

        return alpha

    # ── Negamax ────────────────────────────────────────────────

    def _negamax(self, bb, depth, alpha, beta, do_null=True, prev_move=None):
        self.nodes += 1

        # Time check every 4096 nodes + yield GIL so GUI stays responsive
        if (self.nodes & 4095) == 0:
            if self.time_limit > 0 and time.time() - self.start_time >= self.time_limit:
                self.time_up = True
                return 0.0
            time.sleep(0)

        # Check extension
        in_check = bb.is_check()
        if in_check:
            depth += 1

        if depth <= 0:
            return self._quiesce(bb, alpha, beta)

        is_pv = beta - alpha > 0.02

        # TT probe
        tt = self.tt
        tt_key = _zobrist_hash(bb)
        tt_move = None
        entry = tt.get(tt_key)
        if entry is not None:
            e_depth, e_score, e_flag, e_move = entry
            tt_move = e_move
            if e_depth >= depth and not is_pv:
                if e_flag == 0:  # EXACT
                    return e_score
                elif e_flag == 1 and e_score <= alpha:  # UPPER
                    return e_score
                elif e_flag == 2 and e_score >= beta:  # LOWER
                    return e_score

        # Static eval (only compute once)
        static_eval = _eval_fast(bb)
        if not bb.turn:
            static_eval = -static_eval

        # Razoring at depth 1-2
        if not in_check and not is_pv and depth <= 2:
            razor_margin = 3.0 if depth == 1 else 5.0
            if static_eval + razor_margin <= alpha:
                s = self._quiesce(bb, alpha, beta)
                if s <= alpha:
                    return s

        # Reverse futility pruning
        if not in_check and not is_pv and depth <= 3 and abs(beta) < MATE_SCORE - 100:
            margin = _FUTILITY_MARGIN[depth] * 0.01
            if static_eval - margin >= beta:
                return static_eval - margin

        # Null-move pruning
        if (do_null and depth >= 3 and not in_check
                and static_eval >= beta):
            # Quick endgame check via queens bitboard
            if bb.queens:
                R = 3 if depth >= 6 else 2
                bb.push(chess.Move.null())
                ns = -self._negamax(bb, depth - 1 - R, -beta, -beta + 0.01,
                                    do_null=False)
                bb.pop()
                if self.time_up:
                    return 0.0
                if ns >= beta:
                    return beta

        # Generate legal moves
        legal_moves = list(bb.legal_moves)
        if not legal_moves:
            if in_check:
                return -MATE_SCORE + (MAX_DEPTH - depth)
            return 0.0

        # IID: if no TT move at PV node, do a shallow search
        if is_pv and tt_move is None and depth >= 4:
            self._negamax(bb, depth - 2, alpha, beta, do_null=False, prev_move=prev_move)
            if self.time_up:
                return 0.0
            e2 = tt.get(tt_key)
            if e2 is not None:
                tt_move = e2[3]

        # ── Move ordering (inline for speed) ──
        killers = self.killers
        counter = self.counter_moves
        hist = self.hist_w if bb.turn else self.hist_b
        pta = bb.piece_type_at
        is_cap = bb.is_capture
        mvvlva = _MVV_LVA

        scored_moves = []
        idx = 0
        for m in legal_moves:
            if tt_move and m == tt_move:
                scored_moves.append((10_000_000, idx, m))
            elif is_cap(m):
                c_pt = pta(m.to_square)
                a_pt = pta(m.from_square)
                s = 1_000_000
                if c_pt and a_pt:
                    s += mvvlva[c_pt][a_pt]
                scored_moves.append((s, idx, m))
            elif m.promotion:
                scored_moves.append((900_000, idx, m))
            elif depth < MAX_DEPTH and killers[depth][0] == m:
                scored_moves.append((700_000, idx, m))
            elif depth < MAX_DEPTH and killers[depth][1] == m:
                scored_moves.append((600_000, idx, m))
            elif prev_move and counter.get(prev_move) == m:
                scored_moves.append((550_000, idx, m))
            else:
                scored_moves.append((int(hist[m.from_square, m.to_square]), idx, m))
            idx += 1

        scored_moves.sort(reverse=True)

        best_score = -INFINITY
        best_move = scored_moves[0][2]
        move_count = 0
        raised_alpha = False

        # Futility pruning flag
        can_futility = not in_check and not is_pv and depth <= 3 and abs(alpha) < MATE_SCORE - 100
        fmargin = _FUTILITY_MARGIN[min(depth, 3)] * 0.01
        futility_base = static_eval + fmargin if can_futility else 0

        # LMP threshold
        lmp_threshold = _LMP_COUNTS[min(depth, 4)] if not in_check and not is_pv else 999

        # LMR table
        lmr_tab = _LMR_TABLE[min(depth, MAX_DEPTH - 1)]

        for _, _, move in scored_moves:
            move_count += 1
            is_capture = is_cap(move)
            is_promo = move.promotion is not None

            # Late move pruning
            if move_count > lmp_threshold and not is_capture and not is_promo:
                continue

            # Futility pruning
            if can_futility and move_count > 1 and not is_capture and not is_promo:
                if futility_base <= alpha:
                    continue

            # LMR
            reduction = 0
            if depth >= 3 and move_count > 3 and not in_check and not is_capture and not is_promo:
                reduction = lmr_tab[min(move_count, 63)]
                # Reduce less for killers
                if depth < MAX_DEPTH and (killers[depth][0] == move or killers[depth][1] == move):
                    reduction = max(0, reduction - 1)
                # Don't reduce below 1
                reduction = min(reduction, depth - 2)

            bb.push(move)

            # PVS
            if move_count == 1:
                score = -self._negamax(bb, depth - 1, -beta, -alpha,
                                       prev_move=move)
            else:
                # Scout search with reduction
                score = -self._negamax(bb, depth - 1 - reduction,
                                       -alpha - 0.01, -alpha,
                                       prev_move=move)
                # Re-search if failed high
                if score > alpha and (reduction > 0 or score < beta):
                    score = -self._negamax(bb, depth - 1, -beta, -alpha,
                                           prev_move=move)

            bb.pop()

            if self.time_up:
                return 0.0

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score
                raised_alpha = True

            if alpha >= beta:
                if not is_capture:
                    # Killer
                    if depth < MAX_DEPTH and killers[depth][0] != move:
                        killers[depth][1] = killers[depth][0]
                        killers[depth][0] = move
                    # History bonus
                    bonus = depth * depth
                    val = hist[move.from_square, move.to_square]
                    hist[move.from_square, move.to_square] = min(val + bonus, 1_000_000)
                    # Counter-move
                    if prev_move:
                        counter[prev_move] = move
                break

        # TT store
        if best_score <= alpha and not raised_alpha:
            flag = 1  # UPPER
        elif best_score >= beta:
            flag = 2  # LOWER
        else:
            flag = 0  # EXACT

        old = tt.get(tt_key)
        if old is None or old[0] <= depth:
            if len(tt) > self.tt_max_size:
                tt.clear()
            tt[tt_key] = (depth, best_score, flag, best_move)

        return best_score

    # ── Iterative Deepening with Aspiration Windows ───────────

    def get_best_move(self, board: ChessBoard) -> Optional[chess.Move]:
        """Get best move using iterative deepening with aspiration windows."""
        legal_moves = list(board.board.legal_moves)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        self.nodes = 0
        self.time_up = False
        self.start_time = time.time()
        self.best_root = legal_moves[0]
        self.killers = [[None, None] for _ in range(MAX_DEPTH)]

        # Age history
        self.hist_w >>= 1
        self.hist_b >>= 1

        bb = board.board
        best_move = legal_moves[0]
        prev_score = 0.0

        move_time = self.time_limit if self.time_limit > 0 else 0
        max_depth = self.depth if move_time <= 0 else MAX_DEPTH

        for d in range(1, max_depth + 1):
            if self.time_up:
                break

            # Aspiration window
            if d <= 3:
                alpha = -INFINITY
                beta = INFINITY
            else:
                delta = 0.5
                alpha = prev_score - delta
                beta = prev_score + delta

            while True:
                score = self._search_root(bb, legal_moves, d, alpha, beta)

                if self.time_up:
                    break

                if score <= alpha:
                    alpha = max(alpha - delta * 4, -INFINITY)
                    delta *= 4
                elif score >= beta:
                    beta = min(beta + delta * 4, INFINITY)
                    delta *= 4
                else:
                    break

            if not self.time_up and self.best_root:
                best_move = self.best_root
                prev_score = score

                elapsed = time.time() - self.start_time
                nps = self.nodes / elapsed if elapsed > 0 else 0
                print(f"  depth {d:2d}: {best_move.uci()} "
                      f"score={prev_score:+.2f} "
                      f"nodes={self.nodes:,} "
                      f"time={elapsed:.1f}s "
                      f"nps={nps:,.0f}")

                if abs(prev_score) >= MATE_SCORE - 100:
                    break
                # Use 60% time limit to decide if we should start the next depth
                if move_time > 0 and elapsed > move_time * 0.6:
                    break

        return best_move

    def _search_root(self, bb, legal_moves, depth, alpha, beta):
        """Root search with negamax."""
        tt = self.tt
        tt_key = _zobrist_hash(bb)
        tt_move = None
        entry = tt.get(tt_key)
        if entry is not None:
            tt_move = entry[3]

        # Move ordering
        hist = self.hist_w if bb.turn else self.hist_b
        is_cap = bb.is_capture
        pta = bb.piece_type_at
        mvvlva = _MVV_LVA
        br = self.best_root

        scored = []
        idx = 0
        for m in legal_moves:
            if tt_move and m == tt_move:
                scored.append((10_000_000, idx, m))
            elif br and m == br:
                scored.append((9_000_000, idx, m))
            elif is_cap(m):
                c_pt = pta(m.to_square)
                a_pt = pta(m.from_square)
                s = 1_000_000
                if c_pt and a_pt:
                    s += mvvlva[c_pt][a_pt]
                scored.append((s, idx, m))
            elif m.promotion:
                scored.append((900_000, idx, m))
            else:
                scored.append((int(hist[m.from_square, m.to_square]), idx, m))
            idx += 1

        scored.sort(reverse=True)

        best_score = -INFINITY
        move_count = 0

        for _, _, move in scored:
            move_count += 1

            bb.push(move)

            if move_count == 1:
                score = -self._negamax(bb, depth - 1, -beta, -alpha,
                                       prev_move=move)
            else:
                score = -self._negamax(bb, depth - 1, -alpha - 0.01, -alpha,
                                       prev_move=move)
                if score > alpha and score < beta:
                    score = -self._negamax(bb, depth - 1, -beta, -alpha,
                                           prev_move=move)

            bb.pop()

            if self.time_up:
                break

            if score > best_score:
                best_score = score
                self.best_root = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                break

        return best_score


# ── Subprocess entry point (no Tk imports → safe for multiprocessing) ──

def ai_search_process(fen, depth, classical_weight, time_limit, result_queue):
    """Run AI search in a separate process so GUI timer stays smooth.
    
    This function is the target for multiprocessing.Process.
    It has its own GIL and cannot block the GUI main thread.
    """
    try:
        board = ChessBoard()
        board.board = chess.Board(fen)
        ai = ChessAI(depth=depth, classical_weight=classical_weight,
                      time_limit=time_limit)
        move = ai.get_best_move(board)
        result_queue.put(move.uci() if move else None)
    except Exception as e:
        print(f"AI subprocess error: {e}")
        result_queue.put(None)


# ── CLI Interface ──────────────────────────────────────────────

def print_board(board: ChessBoard):
    print("\n" + "=" * 50)
    print(board)
    print("=" * 50)


def play_game(ai_color: str = 'black', ai_depth: int = 5,
              model_path: Optional[str] = None, classical_weight: float = 0.7):
    print("Chess Game - Play against AI")
    print("=" * 50)
    print(f"AI plays: {ai_color}, depth: {ai_depth}")
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
    import argparse
    parser = argparse.ArgumentParser(description='Play chess against AI')
    parser.add_argument('--ai-color', choices=['white', 'black'], default='black')
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--model', type=str, default='models/chess_model.pth')
    parser.add_argument('--classical-weight', type=float, default=0.7)
    parser.add_argument('--time-limit', type=float, default=0.0)
    args = parser.parse_args()

    play_game(
        ai_color=args.ai_color,
        ai_depth=args.depth,
        model_path=args.model if args.model else None,
        classical_weight=args.classical_weight
    )


if __name__ == "__main__":
    main()
