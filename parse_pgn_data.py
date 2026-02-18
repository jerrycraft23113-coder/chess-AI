"""
Parse PGN files into training data (positions + evaluations).
Optimized for speed: inline board encoding, multiprocessing, numpy batching.
Chunk files are saved to data/chunks/ for resume support.
"""

import chess
import chess.pgn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import os
import argparse
import time
from multiprocessing import Pool, cpu_count

from chess_board import evaluate_position_simple, ChessBoard

# ── Directories ──────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
CHUNK_DIR = DATA_DIR / "chunks"

# ── Inline board encoding (must match chess_board._PIECE_CHANNELS exactly) ──
_PIECE_CHANNELS = [
    (chess.PAWN,   chess.WHITE),  # 0
    (chess.ROOK,   chess.WHITE),  # 1
    (chess.KNIGHT, chess.WHITE),  # 2
    (chess.BISHOP, chess.WHITE),  # 3
    (chess.QUEEN,  chess.WHITE),  # 4
    (chess.KING,   chess.WHITE),  # 5
    (chess.PAWN,   chess.BLACK),  # 6
    (chess.ROOK,   chess.BLACK),  # 7
    (chess.KNIGHT, chess.BLACK),  # 8
    (chess.BISHOP, chess.BLACK),  # 9
    (chess.QUEEN,  chess.BLACK),  # 10
    (chess.KING,   chess.BLACK),  # 11
]

# Piece values for simple eval
_PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                 chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def _board_to_array_fast(bb: chess.Board) -> np.ndarray:
    """Convert chess.Board to 8x8x12 array — no wrapper, inline for speed.
    Must match ChessBoard.board_to_array() exactly (row = 7 - rank)."""
    arr = np.zeros((8, 8, 12), dtype=np.float32)
    for ch, (pt, color) in enumerate(_PIECE_CHANNELS):
        mask = int(bb.pieces(pt, color))
        while mask:
            sq = mask & -mask          # isolate lowest set bit
            idx = sq.bit_length() - 1  # square index 0-63
            arr[7 - (idx >> 3), idx & 7, ch] = 1.0  # 7-rank to match chess_board.py
            mask ^= sq
    return arr


def _eval_simple_fast(bb: chess.Board, result: str) -> float:
    """Fast simple eval — no wrapper, inline piece_map scan + result hint."""
    score = 0.0
    for sq, p in bb.piece_map().items():
        v = _PIECE_VALUES[p.piece_type]
        if p.color == chess.WHITE:
            score += v
        else:
            score -= v
    # Result hint
    if result == "1-0":
        score += 0.1
    elif result == "0-1":
        score -= 0.1
    # Flip for side to move
    if not bb.turn:
        score = -score
    return score


def parse_single_pgn(args_tuple) -> Tuple[str, int, int]:
    """Parse one PGN file → save chunk to disk. Runs in worker process.

    Args:
        args_tuple: (pgn_path, max_games, chunk_dir)

    Returns:
        (filename, game_count, position_count)
    """
    pgn_path, max_games, chunk_dir = args_tuple
    filename = Path(pgn_path).stem
    chunk_path = Path(chunk_dir) / f"{filename}.npz"

    # Skip if chunk already exists
    if chunk_path.exists() and chunk_path.stat().st_size > 0:
        try:
            existing = np.load(chunk_path)
            n = len(existing['positions'])
            return (filename, -1, n)  # -1 = skipped
        except Exception:
            pass  # re-parse if chunk is corrupted

    positions = []
    evaluations = []
    game_count = 0

    try:
        with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                if max_games and game_count >= max_games:
                    break

                try:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                except Exception:
                    continue

                game_count += 1
                result = game.headers.get("Result", "*")
                bb = game.board()  # starting position (chess.Board)
                move_num = 0

                for move in game.mainline_moves():
                    bb.push(move)
                    move_num += 1

                    if bb.is_game_over():
                        break

                    # Sample positions: every 2 moves + all opening moves
                    if move_num % 2 == 0 or move_num <= 10:
                        positions.append(_board_to_array_fast(bb))
                        evaluations.append(_eval_simple_fast(bb, result))
    except Exception:
        pass

    if len(positions) == 0:
        return (filename, game_count, 0)

    # Save chunk
    pos_arr = np.array(positions, dtype=np.float32)
    eval_arr = np.array(evaluations, dtype=np.float32)
    np.savez_compressed(chunk_path, positions=pos_arr, evaluations=eval_arr)

    return (filename, game_count, len(positions))


def parse_all_pgn(pgn_dir: str = "data/pgn", max_games: Optional[int] = None,
                  workers: int = 0) -> None:
    """Parse all PGN files in directory using multiprocessing.

    Args:
        pgn_dir: Directory containing .pgn files.
        max_games: Max games per file (None = all).
        workers: Number of worker processes (0 = auto).
    """
    pgn_dir_path = Path(pgn_dir)
    pgn_files = sorted(pgn_dir_path.glob("*.pgn"))

    if not pgn_files:
        print(f"No PGN files found in {pgn_dir}")
        return

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    if workers <= 0:
        workers = max(1, cpu_count() - 1)

    print(f"Found {len(pgn_files)} PGN files")
    print(f"Workers: {workers}")
    print(f"Chunks saved to: {CHUNK_DIR}")
    print("=" * 60)

    args_list = [(str(f), max_games, str(CHUNK_DIR)) for f in pgn_files]

    t0 = time.time()
    total_games = 0
    total_positions = 0
    skipped = 0
    done = 0

    with Pool(processes=workers) as pool:
        for filename, games, positions in pool.imap_unordered(parse_single_pgn, args_list):
            done += 1
            if games == -1:
                skipped += 1
                total_positions += positions
                print(f"[{done}/{len(pgn_files)}] SKIP {filename} ({positions} positions cached)")
            elif positions > 0:
                total_games += games
                total_positions += positions
                print(f"[{done}/{len(pgn_files)}] OK   {filename}: {games} games, {positions} positions")
            else:
                print(f"[{done}/{len(pgn_files)}] EMPTY {filename}")

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"Done in {elapsed:.1f}s")
    print(f"  Parsed:    {total_games} games")
    print(f"  Positions: {total_positions:,}")
    print(f"  Skipped:   {skipped} (already cached)")
    print(f"  Chunks:    {CHUNK_DIR}")


def merge_chunks(output_path: str = "data/training_data.npz",
                 shuffle: bool = True) -> None:
    """Merge all chunk files into one training data file.

    Args:
        output_path: Output .npz file path.
        shuffle: Whether to shuffle positions.
    """
    chunk_files = sorted(CHUNK_DIR.glob("*.npz"))
    if not chunk_files:
        print(f"No chunk files found in {CHUNK_DIR}")
        return

    print(f"Merging {len(chunk_files)} chunk files...")
    t0 = time.time()

    # First pass: count total to pre-allocate
    total = 0
    valid_chunks = []
    for cf in chunk_files:
        try:
            data = np.load(cf)
            n = len(data['positions'])
            if n > 0:
                total += n
                valid_chunks.append(cf)
        except Exception as e:
            print(f"  Warning: skipping {cf.name} ({e})")

    if total == 0:
        print("No valid positions found!")
        return

    print(f"  Total positions: {total:,} from {len(valid_chunks)} chunks")

    # Pre-allocate arrays
    all_pos = np.empty((total, 8, 8, 12), dtype=np.float32)
    all_eval = np.empty(total, dtype=np.float32)

    # Second pass: fill arrays
    offset = 0
    for i, cf in enumerate(valid_chunks):
        data = np.load(cf)
        n = len(data['positions'])
        all_pos[offset:offset + n] = data['positions']
        all_eval[offset:offset + n] = data['evaluations']
        offset += n
        if (i + 1) % 50 == 0 or i == len(valid_chunks) - 1:
            print(f"  Loaded {offset:,}/{total:,} ({offset * 100 // total}%)")

    # Shuffle
    if shuffle:
        print("  Shuffling...")
        idx = np.random.permutation(total)
        all_pos = all_pos[idx]
        all_eval = all_eval[idx]

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    print(f"  Saving to {output_path}...")
    np.savez_compressed(output_path, positions=all_pos, evaluations=all_eval)

    file_mb = Path(output_path).stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  {total:,} positions → {output_path} ({file_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Parse PGN files into training data")
    sub = parser.add_subparsers(dest="command", help="Command")

    # parse command
    p_parse = sub.add_parser("parse", help="Parse PGN files into chunks")
    p_parse.add_argument("--pgn-dir", default="data/pgn", help="PGN directory (default: data/pgn)")
    p_parse.add_argument("--max-games", type=int, default=None, help="Max games per file")
    p_parse.add_argument("--workers", type=int, default=0, help="Worker processes (0=auto)")

    # merge command
    p_merge = sub.add_parser("merge", help="Merge chunks into training_data.npz")
    p_merge.add_argument("--output", default="data/training_data.npz", help="Output file")
    p_merge.add_argument("--no-shuffle", action="store_true", help="Don't shuffle")

    # all command (parse + merge)
    p_all = sub.add_parser("all", help="Parse PGN files then merge into training data")
    p_all.add_argument("--pgn-dir", default="data/pgn", help="PGN directory")
    p_all.add_argument("--output", default="data/training_data.npz", help="Output file")
    p_all.add_argument("--max-games", type=int, default=None, help="Max games per file")
    p_all.add_argument("--workers", type=int, default=0, help="Worker processes (0=auto)")
    p_all.add_argument("--no-shuffle", action="store_true", help="Don't shuffle")

    # Legacy: no subcommand = "all" behavior
    parser.add_argument("--pgn-dir", default="data/pgn", help="PGN directory")
    parser.add_argument("--output", default="data/training_data.npz", help="Output file")
    parser.add_argument("--max-games", type=int, default=None, help="Max games per file")
    parser.add_argument("--parallel", type=int, default=0, help="Worker processes (0=auto)")

    args = parser.parse_args()

    if args.command == "parse":
        parse_all_pgn(args.pgn_dir, args.max_games, args.workers)
    elif args.command == "merge":
        merge_chunks(args.output, shuffle=not args.no_shuffle)
    elif args.command == "all":
        parse_all_pgn(args.pgn_dir, args.max_games, args.workers)
        print()
        merge_chunks(args.output, shuffle=not args.no_shuffle)
    else:
        # Legacy: no subcommand → parse + merge
        print("PGN Data Parser")
        print("=" * 60)
        workers = args.parallel if args.parallel > 0 else 0
        parse_all_pgn(args.pgn_dir, args.max_games, workers)
        print()
        merge_chunks(args.output)


if __name__ == "__main__":
    main()
