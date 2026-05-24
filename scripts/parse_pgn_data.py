"""
Parse PGN files into training data (positions + evaluations).
Optimized for speed: inline board encoding, multiprocessing, numpy batching.
Chunk files are saved to data/chunks/ for resume support.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import chess
import chess.pgn
import numpy as np
from typing import Optional, Tuple, List
import os
import argparse
import time
import logging
from multiprocessing import Pool, cpu_count

logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)

from chess_board import evaluate_position_simple, ChessBoard

# ── Directories (under repository root) ────────────────────
DATA_DIR = _REPO_ROOT / "data"
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


def _board_to_bitboards(bb: chess.Board) -> np.ndarray:
    """Convert chess.Board to 12 bitboard uint64s — compact storage.

    Each uint64 is the raw bitboard for one piece channel.
    Decode to 8x8x12 float32 at training time via bitboards_to_array().
    """
    bbs = np.empty(12, dtype=np.uint64)
    for ch, (pt, color) in enumerate(_PIECE_CHANNELS):
        bbs[ch] = np.uint64(int(bb.pieces(pt, color)))
    return bbs


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


FLUSH_EVERY = 50_000  # flush positions to disk every N to avoid OOM


def _flush_to_chunk(positions, evaluations, chunk_path, part_idx):
    """Save a batch of positions (bitboards) to a numbered sub-chunk file."""
    if part_idx == 0:
        out = chunk_path
    else:
        out = chunk_path.with_name(f"{chunk_path.stem}_part{part_idx}.npz")
    pos_arr = np.array(positions, dtype=np.uint64)   # (N, 12) bitboards
    eval_arr = np.array(evaluations, dtype=np.float32)
    # Use uncompressed savez — bitboard uint64 is already compact (96 B/pos)
    # and compression doubles RAM usage for minimal size savings
    np.savez(out, positions=pos_arr, evaluations=eval_arr)
    return out


def _merge_parts(chunk_path, part_paths):
    """Merge multiple sub-chunk files into one final chunk, then delete parts.
    Uses pre-allocated arrays to avoid OOM from loading all parts at once."""
    # First pass: count total positions (use min of pos/eval to handle mismatches)
    counts = []
    for p in part_paths:
        with np.load(p) as data:
            n = min(len(data['positions']), len(data['evaluations']))
            counts.append(n)
    total = sum(counts)

    # Pre-allocate final arrays
    all_pos = np.empty((total, 12), dtype=np.uint64)
    all_eval = np.empty(total, dtype=np.float32)

    # Second pass: copy into pre-allocated arrays one part at a time
    offset = 0
    for p, n in zip(part_paths, counts):
        with np.load(p) as data:
            all_pos[offset:offset + n] = data['positions'][:n]
            all_eval[offset:offset + n] = data['evaluations'][:n]
            offset += n

    tmp_path = chunk_path.with_suffix('.tmp.npz')
    np.savez(tmp_path, positions=all_pos, evaluations=all_eval)
    del all_pos, all_eval
    for p in part_paths:
        try:
            p.unlink(missing_ok=True)
        except PermissionError:
            pass
    tmp_path.replace(chunk_path)


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

    # Skip if chunk already exists in the current bitboard format
    if chunk_path.exists() and chunk_path.stat().st_size > 0:
        try:
            with np.load(chunk_path) as existing:
                pos = existing['positions']
                # Only skip if it's in the new uint64 bitboard format
                if pos.dtype == np.uint64 and pos.ndim == 2:
                    return (filename, -1, len(pos))
                # Old float32 format — delete and re-parse
                chunk_path.unlink(missing_ok=True)
        except Exception:
            pass  # re-parse if chunk is corrupted

    # Clean up any stale _part*.npz files from previous runs
    chunk_dir_path = Path(chunk_dir)
    for old_part in chunk_dir_path.glob(f"{filename}_part*.npz"):
        old_part.unlink(missing_ok=True)

    positions = []
    evaluations = []
    game_count = 0
    total_positions = 0
    part_idx = 0
    part_paths = []

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
                bb = game.board()
                move_num = 0

                try:
                    for move in game.mainline_moves():
                        bb.push(move)
                        move_num += 1

                        if bb.is_game_over():
                            break

                        if move_num % 2 == 0 or move_num <= 10:
                            positions.append(_board_to_bitboards(bb))
                            evaluations.append(_eval_simple_fast(bb, result))
                except Exception:
                    continue

                # Flush to disk periodically to avoid OOM
                if len(positions) >= FLUSH_EVERY:
                    p = _flush_to_chunk(positions, evaluations, chunk_path, part_idx)
                    part_paths.append(p)
                    total_positions += len(positions)
                    positions.clear()
                    evaluations.clear()
                    part_idx += 1
    except Exception:
        pass

    # Flush remaining
    if positions:
        p = _flush_to_chunk(positions, evaluations, chunk_path, part_idx)
        part_paths.append(p)
        total_positions += len(positions)
        positions.clear()
        evaluations.clear()

    if total_positions == 0:
        return (filename, game_count, 0)

    # If multiple parts were written, merge them into one chunk
    if len(part_paths) > 1:
        _merge_parts(chunk_path, part_paths)

    return (filename, game_count, total_positions)


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
        workers = min(4, max(1, cpu_count() - 1))

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
    """Merge all chunk files into memory-mapped .npy files on disk.

    Writes two files (positions.npy, evaluations.npy) next to output_path.
    Uses memory-mapped I/O so the full dataset never needs to fit in RAM.

    Args:
        output_path: Base output path (directory derived from this).
        shuffle: Whether to shuffle positions (uses block-shuffle for large data).
    """
    chunk_files = sorted(CHUNK_DIR.glob("*.npz"))
    # Filter out .tmp.npz files from interrupted merges
    chunk_files = [f for f in chunk_files if '.tmp.' not in f.name]
    if not chunk_files:
        print(f"No chunk files found in {CHUNK_DIR}")
        return

    print(f"Merging {len(chunk_files)} chunk files...")
    t0 = time.time()

    # First pass: count total positions (only bitboard uint64 format)
    total = 0
    valid_chunks = []
    skipped_old = 0
    for cf in chunk_files:
        try:
            with np.load(cf) as data:
                pos = data['positions']
                if pos.dtype != np.uint64 or pos.ndim != 2:
                    skipped_old += 1
                    continue
                n = min(len(pos), len(data['evaluations']))
            if n > 0:
                total += n
                valid_chunks.append((cf, n))
        except Exception as e:
            print(f"  Warning: skipping {cf.name} ({e})")
    if skipped_old:
        print(f"  Skipped {skipped_old} old-format chunks (re-run parse to convert)")

    if total == 0:
        print("No valid positions found!")
        return

    print(f"  Total positions: {total:,} from {len(valid_chunks)} chunks")

    out_dir = Path(os.path.dirname(output_path) or 'data')
    out_dir.mkdir(parents=True, exist_ok=True)
    pos_path = out_dir / "positions.npy"
    eval_path = out_dir / "evaluations.npy"

    # Create memory-mapped files on disk (written sequentially, never all in RAM)
    # Bitboard format: (N, 12) uint64 — 32× smaller than float32 8x8x12
    print(f"  Creating {pos_path} and {eval_path} ...")
    pos_mmap = np.lib.format.open_memmap(
        str(pos_path), mode='w+', dtype=np.uint64, shape=(total, 12))
    eval_mmap = np.lib.format.open_memmap(
        str(eval_path), mode='w+', dtype=np.float32, shape=(total,))

    # Second pass: copy chunks into mmap files one at a time
    offset = 0
    for i, (cf, n) in enumerate(valid_chunks):
        with np.load(cf) as data:
            pos_mmap[offset:offset + n] = data['positions'][:n]
            eval_mmap[offset:offset + n] = data['evaluations'][:n]
            offset += n
        if (i + 1) % 50 == 0 or i == len(valid_chunks) - 1:
            print(f"  Loaded {offset:,}/{total:,} ({offset * 100 // total}%)")

    # Shuffle: use block-shuffle to avoid loading everything into RAM
    if shuffle:
        print("  Shuffling (block shuffle)...")
        BLOCK = 1_000_000  # shuffle in blocks of 1M
        rng = np.random.default_rng()
        # Shuffle indices of blocks, then shuffle within each block
        block_starts = list(range(0, total, BLOCK))
        rng.shuffle(block_starts)
        # Shuffle within each block in-place
        for bs in block_starts:
            be = min(bs + BLOCK, total)
            idx = rng.permutation(be - bs)
            pos_mmap[bs:be] = pos_mmap[bs:be][idx]
            eval_mmap[bs:be] = eval_mmap[bs:be][idx]

    # Flush to disk
    del pos_mmap, eval_mmap

    pos_mb = pos_path.stat().st_size / (1024 * 1024)
    eval_mb = eval_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  {total:,} positions")
    print(f"  {pos_path} ({pos_mb:.1f} MB)")
    print(f"  {eval_path} ({eval_mb:.1f} MB)")


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
