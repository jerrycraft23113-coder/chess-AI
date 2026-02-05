# Chess Machine Learning Project

A machine learning project for learning and playing chess using neural networks. This project implements a convolutional neural network (CNN) to evaluate chess positions and uses minimax with alpha-beta pruning to make moves.

## Features

- **Neural Network Evaluation**: CNN-based model that evaluates chess positions
- **Minimax AI**: AI player using minimax algorithm with alpha-beta pruning
- **Interactive Gameplay**: Play chess against the trained AI
- **Training Pipeline**: Generate training data and train the model
- **Board Representation**: Efficient 8x8x12 tensor representation of chess positions

## Project Structure

```
.
├── main.py                  # Main entry point
├── chess_board.py          # Chess board and game logic
├── chess_model.py          # Neural network models
├── train.py                # Training script
├── play_chess.py           # Game playing interface
├── download_pgn_data.py    # Download PGN files from pgnmentor.com
├── parse_pgn_data.py       # Parse PGN files into training data
├── requirements.txt        # Python dependencies
├── README.md              # This file (English)
└── README_VI.md           # Documentation (Vietnamese)
```

## Installation

1. **Clone or download this project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Download Chess Game Data

Download PGN files from pgnmentor.com:

```bash
python download_pgn_data.py
```

This script will download games from top players (Carlsen, Caruana, Anand, Kasparov, etc.) and save them to `data/pgn/`.

### Step 2: Parse Data into Training Format

Convert PGN files into training data:

```bash
python parse_pgn_data.py
```

Or with options:

```bash
# Parse with limit on games per file
python parse_pgn_data.py --max-games 1000

# Specify PGN directory and output file
python parse_pgn_data.py --pgn-dir data/pgn --output data/training_data.npz
```

The data will be saved to `data/training_data.npz`.

### Step 3: Training the Model

Train the neural network on real game data:

```bash
python main.py train
```

Or directly:
```bash
python train.py
```

With options:
```bash
# Specify data file
python train.py --data data/training_data.npz

# Adjust epochs and batch size
python train.py --epochs 30 --batch-size 64 --lr 0.0005

# Model automatically trains on CPU (default)
```

The training script will:
- Load training data from `data/training_data.npz` (or generate random data if not found)
- Train the CNN model for specified epochs (default: 20)
- Save the trained model to `models/chess_model.pth`
- **Note**: Model trains on CPU by default

### Step 4: Playing Chess

Play against the AI:

```bash
python main.py play
```

Or directly:
```bash
python play_chess.py
```

**Command-line options:**
- `--ai-color`: Choose which color the AI plays (`white` or `black`, default: `black`)
- `--depth`: Set AI search depth (default: 2, higher = stronger but slower)
- `--model`: Path to trained model file (default: `models/chess_model.pth`)

**Examples:**
```bash
# AI plays white
python main.py play --ai-color white

# AI with higher search depth
python main.py play --depth 3

# Use custom model
python main.py play --model models/my_model.pth --depth 2
```

### Game Controls

- **Move input**: Enter moves in UCI notation (e.g., `e2e4`, `g1f3`)
- **Undo move**: Type `undo` to undo the last move
- **Quit game**: Type `quit` to exit

## How It Works

### Board Representation

The chess board is represented as an 8×8×12 tensor:
- 8×8 for the board squares
- 12 channels for different piece types and colors:
  - White: Pawn, Rook, Knight, Bishop, Queen, King
  - Black: Pawn, Rook, Knight, Bishop, Queen, King

### Neural Network Architecture

The model uses a CNN with:
- 3 convolutional layers with batch normalization
- Fully connected layers for evaluation
- Output: Single value representing position evaluation (from current player's perspective)

### AI Decision Making

The AI uses the **minimax algorithm** with **alpha-beta pruning**:
1. Evaluates positions using the neural network
2. Searches ahead to a specified depth
3. Chooses the move that maximizes its position evaluation

## Complete Workflow

```bash
# 1. Download data
python download_pgn_data.py

# 2. Parse data
python parse_pgn_data.py

# 3. Train model
python train.py

# 4. Play chess
python main.py play
```

## Model Training

The training process:
1. Downloads PGN files from pgnmentor.com (games from top players)
2. Parses games to extract positions
3. Evaluates positions using a simple heuristic (piece values)
4. Trains the CNN to predict these evaluations
5. Validates on a held-out test set

**Note**: For better performance, you can:
- Train on more positions (download more PGN files)
- Use stronger evaluation functions
- Implement reinforcement learning (self-play)
- Use pre-trained models or game databases

## Customization

### Adjusting Training Parameters

Edit `train.py` to modify:
- Number of training positions (`num_positions`)
- Number of epochs (`num_epochs`)
- Learning rate (`learning_rate`)
- Batch size
- Model architecture

### Improving the AI

- Increase search depth (slower but stronger)
- Train on more data
- Use a stronger evaluation function
- Implement opening book or endgame tablebase
- Add move ordering heuristics

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy 1.24+
- python-chess 3.1+

## Future Improvements

- [ ] Reinforcement learning with self-play
- [ ] Policy network for move selection
- [ ] Opening book integration
- [ ] Endgame tablebase support
- [ ] UCI engine interface
- [ ] Web interface for playing
- [ ] Training on real game databases

## License

This project is open source and available for educational purposes.

## Contributing

Feel free to fork, modify, and improve this project!
