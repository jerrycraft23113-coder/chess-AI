"""
Script để parse PGN files và tạo training data
Converts PGN chess games into training positions
"""

import chess
import chess.pgn
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import random
import os
import tempfile
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("Warning: h5py not available. Install with: pip install h5py")

import torch

from chess_board import ChessBoard, evaluate_position_simple
from chess_model import ChessCNN, ChessPolicyNetwork

# Use a neural-network evaluator if available, otherwise fall back to a
# handcrafted evaluation. No external engines (e.g. Stockfish) are used.
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("models", "chess_model.pth"))
MODEL_DEVICE = "cpu"
_EVAL_MODEL: Optional[torch.nn.Module] = None


def get_eval_model() -> Optional[torch.nn.Module]:
    """Load and cache evaluation model (ChessCNN / ChessPolicyNetwork) if available."""
    global _EVAL_MODEL
    if _EVAL_MODEL is not None:
        return _EVAL_MODEL
    
    if not os.path.exists(MODEL_PATH):
        return None
    
    try:
        # For now assume a value network (ChessCNN). If you later switch to policy,
        # adjust this logic to detect the correct class.
        model = ChessCNN(hidden_size=256)
        state = torch.load(MODEL_PATH, map_location=MODEL_DEVICE)
        model.load_state_dict(state)
        model.to(MODEL_DEVICE)
        model.eval()
        _EVAL_MODEL = model
        print(f"Loaded evaluation model from {MODEL_PATH} for PGN parsing")
        return _EVAL_MODEL
    except Exception as e:
        print(f"Warning: Could not load eval model from {MODEL_PATH}: {e}")
        _EVAL_MODEL = None
        return None


def find_existing_chunks(temp_base: Optional[Path] = None) -> Optional[Path]:
    """Find existing temp directory with chunk files from previous runs.
    
    Args:
        temp_base: Base directory to search (default: system temp)
        
    Returns:
        Path to temp directory with chunks, or None if not found
    """
    if temp_base is None:
        temp_base = Path(tempfile.gettempdir())
    
    # Look for directories matching chess_parse_* pattern
    chess_parse_dirs = sorted(temp_base.glob("chess_parse_*"), reverse=True)
    
    for temp_dir in chess_parse_dirs:
        if not temp_dir.is_dir():
            continue
        
        # Check if it contains chunk files
        chunk_files = sorted(temp_dir.glob("chunk_*.npz"))
        if len(chunk_files) > 0:
            return temp_dir
    
    return None


def parse_pgn_file(pgn_path: str, max_games: Optional[int] = None, verbose: bool = True) -> Tuple[List[np.ndarray], List[float], int]:
    """Parse a PGN file and extract positions with evaluations.
    
    Args:
        pgn_path: Path to PGN file
        max_games: Maximum number of games to parse (None for all)
        verbose: Whether to print progress messages
        
    Returns:
        Tuple of (positions_array, evaluations, game_count)
    """
    positions = []
    evaluations = []
    
    if verbose:
        print(f"Parsing {Path(pgn_path).name}...")
    
    # Try to get neural evaluator first (preferred)
    eval_model = get_eval_model()
    
    try:
        with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
            game_count = 0
            
            while True:
                if max_games and game_count >= max_games:
                    break
                
                try:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    game_count += 1
                    if verbose and game_count % 100 == 0:
                        print(f"  [{Path(pgn_path).name}] Parsed {game_count} games, extracted {len(positions)} positions...")
                    
                    # Replay the game and extract positions
                    board = ChessBoard()
                    move_count = 0
                    
                    for move in game.mainline_moves():
                        # Make the move
                        board.make_move(move)
                        move_count += 1
                        
                        # Skip if game is over
                        if board.is_game_over():
                            break
                        
                        # Extract position every few moves (to get diverse positions)
                        # Also extract positions from different stages of the game
                        if move_count % 2 == 0 or move_count <= 10:  # More positions from opening
                            position = board.board_to_array()
                            
                            # Preferred: use existing neural evaluation model if available
                            eval_score = None
                            if eval_model is not None:
                                try:
                                    # Use the same position array we already built
                                    board_tensor = torch.from_numpy(position).unsqueeze(0)  # (1, 8, 8, 12)
                                    board_tensor = board_tensor.permute(0, 3, 1, 2).to(MODEL_DEVICE)  # (1, 12, 8, 8)
                                    with torch.no_grad():
                                        if isinstance(eval_model, ChessPolicyNetwork):
                                            _, value = eval_model(board_tensor)
                                        else:  # ChessCNN or any value-only net
                                            value = eval_model(board_tensor)
                                    eval_score = float(value.item())
                                except Exception:
                                    eval_score = None
                            
                            # Fallback: simple material-based evaluation + game result hint
                            if eval_score is None:
                                result = game.headers.get("Result", "*")
                                eval_score = evaluate_position_simple(board)
                                
                                if result == "1-0":  # White won
                                    eval_score += 0.1
                                elif result == "0-1":  # Black won
                                    eval_score -= 0.1
                            
                            # If it's black's turn, flip the evaluation so label is always "side to move"
                            if not board.get_turn():
                                eval_score = -eval_score
                            
                            positions.append(position)
                            evaluations.append(eval_score)
                    
                except Exception:
                    # Skip games that can't be parsed
                    continue
        
        if verbose:
            print(f"  [{Path(pgn_path).name}] Finished: {game_count} games, {len(positions)} positions")
        
        return positions, evaluations, game_count
        
    except Exception as e:
        if verbose:
            print(f"  Error parsing {pgn_path}: {e}")
        return [], [], 0


def parse_multiple_pgn_files(pgn_dir: str, max_games_per_file: Optional[int] = None, max_workers: int = 5, 
                             chunk_size: int = 10000, output_path: Optional[str] = None) -> Tuple[List[np.ndarray], List[float]]:
    """Parse multiple PGN files in parallel and save incrementally to reduce memory usage.
    
    Args:
        pgn_dir: Directory containing PGN files
        max_games_per_file: Maximum games to parse per file
        max_workers: Number of parallel workers (default: 5)
        chunk_size: Number of positions to accumulate before saving (default: 10000)
        output_path: Optional path to save data incrementally (if None, returns all data)
        
    Returns:
        Tuple of (positions, evaluations) - empty if output_path is provided
    """
    pgn_dir_path = Path(pgn_dir)
    pgn_files = list(pgn_dir_path.glob("*.pgn"))
    
    if not pgn_files:
        print(f"No PGN files found in {pgn_dir}")
        return [], []
    
    print(f"Found {len(pgn_files)} PGN files")
    print(f"Using {max_workers} parallel workers for parsing...")
    if output_path:
        print(f"Memory-efficient mode: saving in chunks of {chunk_size} positions")
    print("=" * 50)
    
    # If output_path is provided, use incremental saving
    if output_path:
        return parse_and_save_incremental(pgn_files, max_games_per_file, max_workers, chunk_size, output_path)
    
    # Otherwise, use old behavior (but with chunked accumulation)
    all_positions = []
    all_evaluations = []
    total_games = 0
    lock = Lock()  # For thread-safe printing
    
    def parse_file_with_status(pgn_file, index, total):
        """Parse a single PGN file and return results."""
        file_name = Path(pgn_file).name
        with lock:
            print(f"[{index}/{total}] Starting: {file_name}")
        
        positions, evaluations, game_count = parse_pgn_file(
            str(pgn_file), 
            max_games=max_games_per_file,
            verbose=False  # Disable verbose to avoid too much output
        )
        
        with lock:
            print(f"[{index}/{total}] ✓ Completed: {file_name} - {game_count} games, {len(positions)} positions")
        
        return positions, evaluations, game_count
    
    # Use ThreadPoolExecutor for parallel parsing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all parsing tasks
        future_to_file = {
            executor.submit(parse_file_with_status, pgn_file, i+1, len(pgn_files)): pgn_file
            for i, pgn_file in enumerate(pgn_files)
        }
        
        # Process completed parsing tasks
        for future in as_completed(future_to_file):
            try:
                positions, evaluations, game_count = future.result()
                all_positions.extend(positions)
                all_evaluations.extend(evaluations)
                total_games += game_count
                
                # Clear memory periodically if list gets too large
                if len(all_positions) > chunk_size * 2:
                    with lock:
                        print(f"  Memory check: {len(all_positions)} positions in memory...")
            except Exception as e:
                pgn_file = future_to_file[future]
                with lock:
                    print(f"✗ Error processing {Path(pgn_file).name}: {e}")
    
    print("\n" + "=" * 50)
    print(f"Total: {total_games} games parsed, {len(all_positions)} positions extracted")
    return all_positions, all_evaluations


def parse_and_save_incremental(pgn_files: List[Path], max_games_per_file: Optional[int], 
                                max_workers: int, chunk_size: int, output_path: str) -> Tuple[List[np.ndarray], List[float]]:
    """Parse files and save data incrementally - append to file after each PGN file is parsed.
    
    This approach saves data immediately after each PGN file is processed, preventing RAM overflow.
    """
    # Check for existing chunks from previous runs
    existing_temp_dir = find_existing_chunks()
    if existing_temp_dir:
        existing_chunks = sorted(existing_temp_dir.glob("chunk_*.npz"))
        if len(existing_chunks) > 0:
            print(f"\n{'='*50}")
            print(f"Found existing chunk files from previous run!")
            print(f"Location: {existing_temp_dir}")
            print(f"Chunks found: {len(existing_chunks)}")
            print(f"{'='*50}")
            print("Reusing existing chunks. If you want to start fresh, delete the temp directory first.")
            print(f"Temp directory: {existing_temp_dir}")
            print(f"{'='*50}\n")
            
            # Use existing temp directory and chunks
            temp_dir = existing_temp_dir
            chunk_files = existing_chunks
            # Find the highest chunk number to continue from
            if chunk_files:
                last_chunk_name = chunk_files[-1].name
                # Extract number from chunk_XXXXXX.npz
                try:
                    chunk_counter = int(last_chunk_name.replace("chunk_", "").replace(".npz", ""))
                except ValueError:
                    chunk_counter = len(chunk_files)
            else:
                chunk_counter = 0
            
            # Calculate totals from existing chunks
            print("Calculating totals from existing chunks...")
            total_games = 0  # Games count is not stored per chunk, so we'll count during merge
            total_positions = 0
            for chunk_file in chunk_files:
                try:
                    chunk_data = np.load(chunk_file)
                    total_positions += len(chunk_data['positions'])
                except Exception as e:
                    print(f"Warning: Could not read {chunk_file.name}: {e}")
            
            print(f"Found {total_positions:,} positions in {len(chunk_files)} existing chunks")
            
            # Check if all PGN files have been parsed (chunks >= PGN files)
            if len(chunk_files) >= len(pgn_files):
                print(f"\n{'='*50}")
                print(f"All PGN files have been parsed ({len(chunk_files)} chunks >= {len(pgn_files)} files)")
                print("Skipping parsing, proceeding directly to merge...")
                print(f"{'='*50}\n")
                
                # Skip parsing, go directly to merge
                # Ensure chunk_files is sorted
                chunk_files = sorted(chunk_files)
                print(f"Merging {len(chunk_files)} chunk files...")
                
                # Merge all chunks into final file
                try:
                    merge_chunks_to_file(chunk_files, output_path)
                    print(f"✓ Merge completed successfully!")
                except Exception as e:
                    print(f"\n✗ Error during merge: {e}")
                    print(f"Chunk files are still available in: {temp_dir}")
                    print("You can merge them later using:")
                    print(f"  python parse_pgn_data.py --merge-only {temp_dir} --output {output_path}")
                    raise
                
                # Clean up temporary files only after successful merge
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Temporary files cleaned up")
                except Exception as e:
                    print(f"Warning: Could not clean up temp directory {temp_dir}: {e}")
                    print("You may need to clean it up manually")
                
                print(f"Final file saved to: {output_path}")
                return [], []  # Return empty since data is saved to file
        else:
            # Directory exists but no chunks, create new
            temp_dir = Path(tempfile.mkdtemp(prefix="chess_parse_"))
            chunk_files = []
            chunk_counter = 0
            total_games = 0
            total_positions = 0
    else:
        # No existing chunks, create new temp directory
        temp_dir = Path(tempfile.mkdtemp(prefix="chess_parse_"))
        chunk_files = []
        chunk_counter = 0
        total_games = 0
        total_positions = 0
    
    file_lock = Lock()  # Lock for file operations
    
    def parse_and_save_file(pgn_file, index, total):
        """Parse a single PGN file and immediately save to disk."""
        nonlocal chunk_counter, total_games, total_positions, chunk_files
        
        file_name = Path(pgn_file).name
        with file_lock:
            print(f"[{index}/{total}] Starting: {file_name}")
        
        # Parse the PGN file
        positions, evaluations, game_count = parse_pgn_file(
            str(pgn_file), 
            max_games=max_games_per_file,
            verbose=False
        )
        
        if len(positions) == 0:
            with file_lock:
                print(f"[{index}/{total}] ⚠ No positions extracted from: {file_name}")
            return 0
        
        # Immediately save this file's data to disk (no RAM accumulation)
        with file_lock:
            chunk_counter += 1
            chunk_file = temp_dir / f"chunk_{chunk_counter:06d}.npz"
            
            # Convert to numpy arrays and save immediately
            positions_array = np.array(positions, dtype=np.float32)
            evaluations_array = np.array(evaluations, dtype=np.float32)
            np.savez_compressed(chunk_file, positions=positions_array, evaluations=evaluations_array)
            
            chunk_files.append(chunk_file)
            total_games += game_count
            total_positions += len(positions)
            
            print(f"[{index}/{total}] ✓ Saved: {file_name} - {game_count} games, {len(positions)} positions → chunk_{chunk_counter:06d}.npz (total: {total_positions} positions)")
        
        # Clear memory immediately after saving
        del positions, evaluations, positions_array, evaluations_array
        
        return game_count
    
    # Use ThreadPoolExecutor for parallel parsing
    print("Parsing PGN files and saving immediately to disk (no RAM accumulation)...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(parse_and_save_file, pgn_file, i+1, len(pgn_files)): pgn_file
            for i, pgn_file in enumerate(pgn_files)
        }
        
        for future in as_completed(future_to_file):
            try:
                future.result()  # Wait for completion
            except Exception as e:
                pgn_file = future_to_file[future]
                with file_lock:
                    print(f"✗ Error processing {Path(pgn_file).name}: {e}")
    
    print("\n" + "=" * 50)
    print(f"Total: {total_games} games parsed, {total_positions} positions extracted")
    print(f"Saved {len(chunk_files)} chunk files. Merging into final file...")
    
    # Ensure chunk_files is sorted
    chunk_files = sorted(chunk_files)
    print(f"Merging {len(chunk_files)} chunk files...")
    
    # Merge all chunks into final file
    try:
        merge_chunks_to_file(chunk_files, output_path)
        print(f"✓ Merge completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during merge: {e}")
        print(f"Chunk files are still available in: {temp_dir}")
        print("You can merge them later using:")
        print(f"  python parse_pgn_data.py --merge-only {temp_dir} --output {output_path}")
        raise
    
    # Clean up temporary files only after successful merge
    try:
        shutil.rmtree(temp_dir)
        print(f"Temporary files cleaned up")
    except Exception as e:
        print(f"Warning: Could not clean up temp directory {temp_dir}: {e}")
        print("You may need to clean it up manually")
    
    print(f"Final file saved to: {output_path}")
    
    return [], []  # Return empty since data is saved to file


def merge_chunks_to_file(chunk_files: List[Path], output_path: str):
    """Merge multiple chunk files into a single file (ultra memory-efficient - process one chunk at a time).
    
    Supports resume: if process is interrupted, can continue from where it left off.
    """
    if len(chunk_files) == 0:
        print("No chunk files to merge!")
        return
    
    print(f"Merging {len(chunk_files)} chunk files...")
    
    # Progress tracking file
    progress_file = Path(str(output_path) + '.progress')
    processed_chunks = set()
    start_batch = 0
    
    # Check if we can resume
    if progress_file.exists() and Path(output_path).exists():
        try:
            import json
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                processed_chunks = set(progress_data.get('processed_chunks', []))
                start_batch = progress_data.get('last_batch', 0)
                print(f"Resuming merge: {len(processed_chunks)}/{len(chunk_files)} chunks already processed")
                print(f"Continuing from batch {start_batch}")
        except Exception as e:
            print(f"Could not load progress file: {e}")
            print("Starting fresh merge...")
            processed_chunks = set()
            start_batch = 0
    
    # First pass: collect all data sizes and calculate total
    # Filter out corrupted/invalid chunk files
    total_positions = 0
    chunk_sizes = []
    remaining_chunks = []
    valid_chunk_files = []
    original_count = len(chunk_files)
    
    print("Validating chunk files...")
    for i, chunk_file in enumerate(chunk_files):
        # Check if file exists and has valid size
        if not chunk_file.exists():
            print(f"Warning: Chunk file not found: {chunk_file.name}, skipping...")
            continue
        
        file_size = chunk_file.stat().st_size
        if file_size == 0:
            print(f"Warning: Empty chunk file: {chunk_file.name}, skipping...")
            continue
        
        # Try to load and validate the chunk file
        try:
            data = np.load(chunk_file, mmap_mode='r')
            if 'positions' not in data or 'evaluations' not in data:
                print(f"Warning: Invalid chunk file (missing keys): {chunk_file.name}, skipping...")
                continue
            chunk_size = len(data['positions'])
            if chunk_size == 0:
                print(f"Warning: Empty chunk data: {chunk_file.name}, skipping...")
                continue
            chunk_sizes.append(chunk_size)
            total_positions += chunk_size
            valid_chunk_files.append(chunk_file)
            # Map original index to new index for processed_chunks tracking
            new_idx = len(valid_chunk_files) - 1
            if i not in processed_chunks:
                remaining_chunks.append((new_idx, chunk_file))
        except (zipfile.BadZipFile, ValueError, OSError, KeyError) as e:
            print(f"Warning: Corrupted chunk file {chunk_file.name}: {e}, skipping...")
            continue
        except Exception as e:
            print(f"Warning: Error reading chunk file {chunk_file.name}: {e}, skipping...")
            continue
    
    # Update chunk_files to only include valid ones
    skipped_count = original_count - len(valid_chunk_files)
    chunk_files = valid_chunk_files
    if len(chunk_files) == 0:
        print("Error: No valid chunk files found!")
        return
    
    if skipped_count > 0:
        print(f"Found {len(chunk_files)} valid chunk files (skipped {skipped_count} invalid/corrupted files)")
        # If chunks were skipped, we need to rebuild processed_chunks based on file paths
        # For simplicity, if we skipped chunks, we'll reset processed_chunks
        # (user can resume manually if needed)
        if len(processed_chunks) > 0:
            print("Note: Some chunks were skipped. Resetting progress tracking.")
            processed_chunks = set()
            remaining_chunks = [(i, chunk_file) for i, chunk_file in enumerate(chunk_files)]
    else:
        print(f"All {len(chunk_files)} chunk files are valid")
    
    if len(processed_chunks) > 0:
        print(f"Already processed: {len(processed_chunks)} chunks")
        print(f"Remaining: {len(remaining_chunks)} chunks")
    
    print(f"Total positions to merge: {total_positions:,}")
    
    if not H5PY_AVAILABLE:
        raise ImportError("h5py is required for memory-efficient merging. Install with: pip install h5py")
    
    # Use HDF5 for direct disk-based appending (no RAM accumulation)
    temp_h5_path = str(output_path).replace('.npz', '_temp.h5')
    
    # Check if resuming from existing output
    existing_count = 0
    if Path(output_path).exists() and len(processed_chunks) > 0:
        print("Resuming from existing output file...")
        try:
            with h5py.File(output_path, 'r') as hf:
                if 'positions' in hf and 'evaluations' in hf:
                    existing_count = len(hf['positions'])
                    print(f"Found {existing_count:,} existing positions")
        except Exception as e:
            print(f"Warning: Could not read existing output: {e}, starting fresh...")
            existing_count = 0
    
    try:
        # Open HDF5 file for writing (append mode)
        with h5py.File(temp_h5_path, 'a') as hf:
            # Create datasets if they don't exist
            if 'positions' not in hf:
                hf.create_dataset('positions', 
                                 shape=(0, 8, 8, 12), 
                                 maxshape=(None, 8, 8, 12), 
                                 dtype=np.float32,
                                 chunks=True,
                                 compression='gzip')
                hf.create_dataset('evaluations',
                                 shape=(0,),
                                 maxshape=(None,),
                                 dtype=np.float32,
                                 chunks=True,
                                 compression='gzip')
            
            processed = existing_count
            
            # Process in very small batches to avoid memory issues
            batch_size = 2  # Process only 2 chunks at a time
            
            # Process only remaining chunks in batches
            for batch_idx in range(0, len(remaining_chunks), batch_size):
                batch_end = min(batch_idx + batch_size, len(remaining_chunks))
                batch_items = remaining_chunks[batch_idx:batch_end]
                
                batch_positions = []
                batch_evaluations = []
                batch_chunk_indices = []
                
                for chunk_idx, chunk_file in batch_items:
                    try:
                        data = np.load(chunk_file, mmap_mode='r')
                        # Load into memory (small batch)
                        batch_positions.append(np.array(data['positions']))
                        batch_evaluations.append(np.array(data['evaluations']))
                        batch_chunk_indices.append(chunk_idx)
                    except (zipfile.BadZipFile, ValueError, OSError, KeyError) as e:
                        print(f"  Warning: Error loading chunk {chunk_file.name} during merge: {e}, skipping...")
                        continue
                    except Exception as e:
                        print(f"  Warning: Unexpected error loading chunk {chunk_file.name}: {e}, skipping...")
                        continue
                
                # Concatenate this small batch and append directly to HDF5
                if batch_positions:
                    batch_pos = np.concatenate(batch_positions, axis=0)
                    batch_eval = np.concatenate(batch_evaluations, axis=0)
                    
                    # Append directly to HDF5 (no RAM accumulation)
                    current_size = hf['positions'].shape[0]
                    new_size = current_size + len(batch_pos)
                    
                    hf['positions'].resize((new_size, 8, 8, 12))
                    hf['evaluations'].resize((new_size,))
                    
                    hf['positions'][current_size:new_size] = batch_pos
                    hf['evaluations'][current_size:new_size] = batch_eval
                    
                    processed += len(batch_pos)
                    
                    # Mark chunks as processed
                    for chunk_idx in batch_chunk_indices:
                        processed_chunks.add(chunk_idx)
                    
                    print(f"  Processed {processed:,}/{total_positions:,} positions ({processed*100//max(total_positions,1)}%) - {len(processed_chunks)}/{len(chunk_files)} chunks")
                    
                    # Save progress after each batch
                    try:
                        import json
                        progress_data = {
                            'processed_chunks': list(processed_chunks),
                            'last_batch': batch_idx // batch_size + 1,
                            'total_chunks': len(chunk_files)
                        }
                        with open(progress_file, 'w') as f:
                            json.dump(progress_data, f)
                    except Exception as e:
                        print(f"Warning: Could not save progress: {e}")
                    
                    # Clear batch immediately
                    del batch_positions, batch_evaluations, batch_pos, batch_eval, batch_chunk_indices
        
        # Now shuffle and convert HDF5 to NPZ format (write directly, no appending)
        print("Shuffling and converting to NPZ format...")
        with h5py.File(temp_h5_path, 'r') as hf:
            total_positions = len(hf['positions'])
        
        # Shuffle and save in chunks to avoid memory issues
        print(f"Shuffling {total_positions:,} positions (this may take a while)...")
        print("Note: Using chunked shuffle and direct write to avoid memory issues...")
        
        # Load and shuffle in manageable chunks, write directly to NPZ
        chunk_size = 100000  # 100k positions at a time
        all_shuffled_pos = []
        all_shuffled_eval = []
        
        # Create random order for chunks
        num_chunks = (total_positions + chunk_size - 1) // chunk_size
        chunk_order = np.random.permutation(num_chunks)
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Use a temporary HDF5 file for shuffled output (then convert to NPZ at end)
        shuffled_h5_path = str(output_path).replace('.npz', '_shuffled_temp.h5')
        
        with h5py.File(shuffled_h5_path, 'w') as out_hf:
            out_hf.create_dataset('positions', 
                                 shape=(0, 8, 8, 12), 
                                 maxshape=(None, 8, 8, 12), 
                                 dtype=np.float32,
                                 chunks=True,
                                 compression='gzip')
            out_hf.create_dataset('evaluations',
                                 shape=(0,),
                                 maxshape=(None,),
                                 dtype=np.float32,
                                 chunks=True,
                                 compression='gzip')
            
            current_size = 0
            
            for chunk_idx in chunk_order:
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_positions)
                
                # Load chunk from HDF5
                with h5py.File(temp_h5_path, 'r') as hf:
                    chunk_pos = np.array(hf['positions'][start_idx:end_idx])
                    chunk_eval = np.array(hf['evaluations'][start_idx:end_idx])
                
                # Shuffle within chunk
                chunk_indices = np.random.permutation(len(chunk_pos))
                chunk_pos = chunk_pos[chunk_indices]
                chunk_eval = chunk_eval[chunk_indices]
                
                all_shuffled_pos.append(chunk_pos)
                all_shuffled_eval.append(chunk_eval)
                
                # Append directly to output HDF5 when we have enough
                if len(all_shuffled_pos) >= 10:
                    accumulated_pos = np.concatenate(all_shuffled_pos, axis=0)
                    accumulated_eval = np.concatenate(all_shuffled_eval, axis=0)
                    
                    # Append to HDF5 (no RAM accumulation)
                    new_size = current_size + len(accumulated_pos)
                    out_hf['positions'].resize((new_size, 8, 8, 12))
                    out_hf['evaluations'].resize((new_size,))
                    out_hf['positions'][current_size:new_size] = accumulated_pos
                    out_hf['evaluations'][current_size:new_size] = accumulated_eval
                    current_size = new_size
                    
                    all_shuffled_pos = []
                    all_shuffled_eval = []
                    del accumulated_pos, accumulated_eval
                
                processed = min((chunk_idx + 1) * chunk_size, total_positions)
                if processed % 500000 == 0 or processed == total_positions:
                    print(f"  Processed {processed:,}/{total_positions:,} positions ({processed*100//total_positions}%)")
            
            # Save remaining
            if all_shuffled_pos:
                print("Saving final shuffled data...")
                final_positions = np.concatenate(all_shuffled_pos, axis=0)
                final_evaluations = np.concatenate(all_shuffled_eval, axis=0)
                
                new_size = current_size + len(final_positions)
                out_hf['positions'].resize((new_size, 8, 8, 12))
                out_hf['evaluations'].resize((new_size,))
                out_hf['positions'][current_size:new_size] = final_positions
                out_hf['evaluations'][current_size:new_size] = final_evaluations
                
                del all_shuffled_pos, all_shuffled_eval, final_positions, final_evaluations
        
        # Now convert shuffled HDF5 to NPZ
        # Since NPZ doesn't support append, we need to load all data
        # For very large datasets, we'll try to do it in manageable chunks
        print("Converting shuffled HDF5 to NPZ format...")
        with h5py.File(shuffled_h5_path, 'r') as hf:
            total_positions = len(hf['positions'])
        
        # Estimate memory needed
        estimated_memory_gb = (total_positions * 8 * 8 * 12 * 4) / (1024**3)  # float32 = 4 bytes
        print(f"Estimated memory needed: {estimated_memory_gb:.2f} GB")
        
        if estimated_memory_gb > 15:  # If more than 15GB, use chunked approach
            print("Dataset too large for single NPZ conversion. Using chunked approach...")
            # Convert in chunks and save to multiple files, then merge
            # For now, we'll just copy the HDF5 file and rename it
            # Or we can save as HDF5 instead of NPZ
            h5_output_path = str(output_path).replace('.npz', '.h5')
            print(f"Copying shuffled data to {h5_output_path}...")
            import shutil
            shutil.copy2(shuffled_h5_path, h5_output_path)
            print(f"✓ Saved as HDF5 format: {h5_output_path}")
            print("Note: NPZ conversion skipped due to memory constraints. Use HDF5 file instead.")
            # Also try to create a smaller NPZ sample if possible
            print("Creating a sample NPZ file with first 1M positions...")
            sample_size = min(1000000, total_positions)
            with h5py.File(shuffled_h5_path, 'r') as hf:
                sample_pos = np.array(hf['positions'][:sample_size])
                sample_eval = np.array(hf['evaluations'][:sample_size])
            np.savez_compressed(output_path, positions=sample_pos, evaluations=sample_eval)
            print(f"✓ Created sample NPZ with {sample_size:,} positions")
        else:
            # Convert to NPZ in chunks - accumulate all, then write once
            convert_chunk_size = 500000  # 500k at a time
            all_npz_pos = []
            all_npz_eval = []
            
            print("Loading all data from HDF5 for final NPZ conversion...")
            for i in range(0, total_positions, convert_chunk_size):
                end_idx = min(i + convert_chunk_size, total_positions)
                
                with h5py.File(shuffled_h5_path, 'r') as hf:
                    chunk_pos = np.array(hf['positions'][i:end_idx])
                    chunk_eval = np.array(hf['evaluations'][i:end_idx])
                
                all_npz_pos.append(chunk_pos)
                all_npz_eval.append(chunk_eval)
                
                if (i + convert_chunk_size) % 2000000 == 0 or end_idx == total_positions:
                    print(f"  Loaded {end_idx:,}/{total_positions:,} positions ({end_idx*100//total_positions}%)")
            
            # Concatenate all and write once
            print("Concatenating and writing to NPZ (this may take a moment)...")
            final_positions = np.concatenate(all_npz_pos, axis=0)
            final_evaluations = np.concatenate(all_npz_eval, axis=0)
            
            # Write once - this is the final step
            np.savez_compressed(output_path, positions=final_positions, evaluations=final_evaluations)
            del all_npz_pos, all_npz_eval, final_positions, final_evaluations
        
        # Clean up shuffled temp file
        if os.path.exists(shuffled_h5_path):
            try:
                import time
                time.sleep(0.1)
                os.remove(shuffled_h5_path)
            except Exception as e:
                print(f"Warning: Could not delete shuffled temp file: {e}")
        
        file_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"✓ Merged {len(chunk_files)} chunks ({total_positions:,} positions) into {output_path} ({file_size:.2f} MB)")
        
        # Clean up progress file on success
        if progress_file.exists():
            progress_file.unlink()
            print("Progress file cleaned up")
        
    except KeyboardInterrupt:
        print("\n\nMerge interrupted by user!")
        print(f"Progress saved. You can resume by running the same command again.")
        print(f"Processed {len(processed_chunks)}/{len(chunk_files)} chunks so far")
        raise
    except Exception as e:
        print(f"\n\nError during merge: {e}")
        print(f"Progress saved. You can resume by running the same command again.")
        print(f"Processed {len(processed_chunks)}/{len(chunk_files)} chunks so far")
        raise
    finally:
        # Clean up temp HDF5 file
        if os.path.exists(temp_h5_path):
            try:
                # Ensure file is closed before deleting
                import time
                time.sleep(0.1)  # Brief delay to ensure file handles are released
                os.remove(temp_h5_path)
            except PermissionError:
                print(f"Warning: Could not delete temp file {temp_h5_path} (may be in use)")
            except Exception as e:
                print(f"Warning: Could not delete temp file: {e}")


def save_training_data(positions: List[np.ndarray], evaluations: List[float], output_path: str):
    """Save training data to numpy file.
    
    Args:
        positions: List of position arrays
        evaluations: List of evaluation scores
        output_path: Path to save the data
    """
    print(f"\nSaving training data to {output_path}...")
    
    positions_array = np.array(positions, dtype=np.float32)
    evaluations_array = np.array(evaluations, dtype=np.float32)
    
    np.savez_compressed(
        output_path,
        positions=positions_array,
        evaluations=evaluations_array
    )
    
    file_size = Path(output_path).stat().st_size / (1024 * 1024)  # Size in MB
    print(f"Saved {len(positions)} positions ({file_size:.2f} MB)")


def load_training_data(input_path: str) -> Tuple[List[np.ndarray], List[float]]:
    """Load training data from numpy file (memory-efficient version).
    
    Args:
        input_path: Path to the data file
        
    Returns:
        Tuple of (positions, evaluations)
    """
    print(f"Loading training data from {input_path}...")
    
    # Use memory mapping for large files
    data = np.load(input_path, mmap_mode='r')
    num_positions = len(data['positions'])
    
    print(f"Found {num_positions} positions. Loading in chunks...")
    
    # Load in chunks to reduce memory spikes
    chunk_size = 10000
    positions = []
    evaluations = []
    
    for i in range(0, num_positions, chunk_size):
        end_idx = min(i + chunk_size, num_positions)
        chunk_positions = data['positions'][i:end_idx]
        chunk_evaluations = data['evaluations'][i:end_idx]
        
        # Convert to list of arrays
        for pos in chunk_positions:
            positions.append(pos)
        evaluations.extend(chunk_evaluations.tolist())
        
        if (i + chunk_size) % 50000 == 0 or end_idx == num_positions:
            print(f"  Loaded {end_idx}/{num_positions} positions...")
    
    print(f"Loaded {len(positions)} positions")
    return positions, evaluations


def main():
    """Main function to parse PGN files into training data."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse PGN files into training data')
    parser.add_argument('--pgn-dir', type=str, default='data/pgn',
                       help='Directory containing PGN files (default: data/pgn)')
    parser.add_argument('--output', type=str, default='data/training_data.npz',
                       help='Output file for training data (default: data/training_data.npz)')
    parser.add_argument('--max-games', type=int, default=None,
                       help='Maximum games to parse per PGN file (default: all)')
    parser.add_argument('--parallel', type=int, default=5,
                       help='Number of parallel workers for parsing (default: 5)')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='[Deprecated] Each PGN file is saved immediately to prevent RAM overflow')
    # Memory-efficient mode is enabled by default
    # Use --no-memory-efficient to disable it
    parser.add_argument('--no-memory-efficient', action='store_true',
                       help='Disable memory-efficient incremental saving (not recommended for large datasets)')
    parser.add_argument('--merge-only', type=str, default=None,
                       help='Merge existing chunk files from directory (skip parsing)')
    
    args = parser.parse_args()
    
    # Memory-efficient is default (True), unless --no-memory-efficient flag is used
    args.memory_efficient = not args.no_memory_efficient
    
    print("PGN Data Parser")
    print("=" * 50)
    
    # If merge-only mode, just merge existing chunks
    if args.merge_only:
        chunk_dir = Path(args.merge_only)
        if not chunk_dir.exists():
            print(f"Error: Directory not found: {chunk_dir}")
            return
        
        chunk_files = sorted(chunk_dir.glob("chunk_*.npz"))
        if len(chunk_files) == 0:
            print(f"No chunk files found in {chunk_dir}")
            return
        
        print(f"Found {len(chunk_files)} chunk files to merge")
        print(f"Output: {args.output}")
        print("=" * 50)
        
        merge_chunks_to_file(chunk_files, args.output)
        print("\n" + "=" * 50)
        print("Merge complete!")
        return
    
    # Parse PGN files with incremental saving to reduce memory usage
    if args.memory_efficient:
        positions, evaluations = parse_multiple_pgn_files(
            args.pgn_dir,
            max_games_per_file=args.max_games,
            max_workers=args.parallel,
            chunk_size=args.chunk_size,
            output_path=args.output
        )
        
        # Data is already saved, just print completion message
        print("\n" + "=" * 50)
        print("Parsing complete!")
        print(f"Training data saved to: {args.output}")
        print(f"Use this file with train.py to train the model")
    else:
        # Old behavior (loads everything into memory)
        positions, evaluations = parse_multiple_pgn_files(
            args.pgn_dir,
            max_games_per_file=args.max_games,
            max_workers=args.parallel,
            chunk_size=args.chunk_size,
            output_path=None
        )
        
        if len(positions) == 0:
            print("\nNo positions extracted! Please check:")
            print("  1. PGN files exist in the specified directory")
            print("  2. Run download_pgn_data.py first to download games")
            return
        
        # Shuffle the data
        print("\nShuffling data...")
        combined = list(zip(positions, evaluations))
        random.shuffle(combined)
        positions, evaluations = zip(*combined)
        positions = list(positions)
        evaluations = list(evaluations)
        
        # Save training data
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        save_training_data(positions, evaluations, args.output)
        
        print("\n" + "=" * 50)
        print("Parsing complete!")
        print(f"Training data saved to: {args.output}")
        print(f"Use this file with train.py to train the model")


if __name__ == "__main__":
    main()
