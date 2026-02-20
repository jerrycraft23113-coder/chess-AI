"""
Chess Machine Learning Project - Main Entry Point
"""

import argparse
import os


def main():
    """Main entry point for the chess ML project."""
    parser = argparse.ArgumentParser(
        description='Chess Machine Learning Project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model (supervised learning)
  python main.py train

  # Train using RL (self-play)
  python main.py train-rl --games 100

  # Train supervised + RL in one go
  python main.py train-all --epochs 20 --rl-games 100

  # Play against AI in console (AI plays black)
  python main.py play

  # Play against AI with GUI
  python main.py gui

  # Play with GUI (AI plays white)
  python main.py gui --ai-color white

  # Play with custom model and depth
  python main.py gui --model models/chess_model.pth --depth 3
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command (supervised)
    train_parser = subparsers.add_parser('train', help='Train the chess neural network (supervised learning)')
    train_parser.add_argument('--data', type=str, default='data/training_data.npz',
                              help='Path to training data file (default: data/training_data.npz)')
    train_parser.add_argument('--epochs', type=int, default=20,
                              help='Number of training epochs (default: 20, use <= 0 for infinite)')
    train_parser.add_argument('--batch-size', type=int, default=128,
                              help='Batch size (default: 128)')
    train_parser.add_argument('--lr', type=float, default=0.001,
                              help='Learning rate (default: 0.001)')
    train_parser.add_argument('--use-cpu', action='store_true',
                              help='Force CPU usage (default: auto-detect GPU)')
    train_parser.add_argument('--use-gpu', action='store_true',
                              help='Force GPU usage if available')
    
    # Train RL command
    train_rl_parser = subparsers.add_parser('train-rl', help='Train the chess neural network using RL (self-play)')
    train_rl_parser.add_argument('--games', type=int, default=100,
                                 help='Number of self-play games (default: 100, use <= 0 for infinite)')
    train_rl_parser.add_argument('--batch-size', type=int, default=10,
                                 help='Batch size for training (default: 10)')
    train_rl_parser.add_argument('--lr', type=float, default=0.001,
                                help='Learning rate (default: 0.001)')
    train_rl_parser.add_argument('--model-type', choices=['value', 'policy'], default='value',
                                help='Model type: value or policy (default: value)')
    train_rl_parser.add_argument('--load-model', type=str, default=None,
                                help='Path to load existing model')
    train_rl_parser.add_argument('--save-interval', type=int, default=10,
                                help='Save model every N games (default: 10)')
    
    # Train-all command (supervised → RL)
    train_all_parser = subparsers.add_parser('train-all',
        help='Train supervised then automatically continue with RL self-play')
    train_all_parser.add_argument('--data', type=str, default='data/training_data.npz',
                                  help='Path to training data file (default: data/training_data.npz)')
    train_all_parser.add_argument('--epochs', type=int, default=20,
                                  help='Supervised training epochs (default: 20)')
    train_all_parser.add_argument('--batch-size', type=int, default=128,
                                  help='Batch size for supervised training (default: 128)')
    train_all_parser.add_argument('--lr', type=float, default=0.001,
                                  help='Learning rate (default: 0.001)')
    train_all_parser.add_argument('--use-cpu', action='store_true',
                                  help='Force CPU usage')
    train_all_parser.add_argument('--augment', action='store_true',
                                  help='Enable data augmentation for supervised training')
    train_all_parser.add_argument('--rl-games', type=int, default=100,
                                  help='Number of RL self-play games (default: 100, use <= 0 for infinite)')
    train_all_parser.add_argument('--rl-batch-size', type=int, default=10,
                                  help='RL batch size (default: 10)')
    train_all_parser.add_argument('--rl-lr', type=float, default=0.0005,
                                  help='RL learning rate (default: 0.0005, lower than supervised)')
    train_all_parser.add_argument('--save-interval', type=int, default=10,
                                  help='Save RL model every N games (default: 10)')

    # Play command (console)
    play_parser = subparsers.add_parser('play', help='Play chess against the AI (console mode)')
    play_parser.add_argument('--ai-color', choices=['white', 'black'], default='black',
                            help='Color the AI plays (default: black)')
    play_parser.add_argument('--depth', type=int, default=5,
                            help='AI search depth (default: 5)')
    play_parser.add_argument('--model', type=str, default='models/chess_model.pth',
                            help='Path to trained model (default: models/chess_model.pth)')
    play_parser.add_argument('--classical-weight', type=float, default=0.7,
                            help='Weight for classical evaluation vs neural net (0-1, default: 0.7)')

    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Play chess against the AI (GUI mode)')
    gui_parser.add_argument('--ai-color', choices=['white', 'black'], default='black',
                           help='Color the AI plays (default: black)')
    gui_parser.add_argument('--depth', type=int, default=5,
                           help='AI search depth (default: 5)')
    gui_parser.add_argument('--model', type=str, default='models/chess_model.pth',
                           help='Path to trained model (default: models/chess_model.pth)')
    gui_parser.add_argument('--classical-weight', type=float, default=0.7,
                           help='Weight for classical evaluation vs neural net (0-1, default: 0.7)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print("Starting training (supervised learning)...")
        from train import run_training
        run_training(
            data_path=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            use_cpu=args.use_cpu,
        )
    elif args.command == 'train-rl':
        print("Starting RL training (self-play)...")
        from train_rl import train_rl, ChessCNN, ChessPolicyNetwork
        import torch
        
        # Device selection
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create model
        if args.model_type == 'policy':
            model = ChessPolicyNetwork(hidden_size=256)
            print("Using Policy Network")
        else:
            model = ChessCNN(hidden_size=256)
            print("Using Value Network")
        
        # Load existing model if provided
        if args.load_model and os.path.exists(args.load_model):
            try:
                state_dict = torch.load(args.load_model, map_location=device)
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                print(f"Loaded model from {args.load_model}")
            except Exception as e:
                print(f"Warning: Could not load model: {e}")
        
        train_rl(
            model=model,
            num_games=args.games,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            save_interval=args.save_interval
        )
    elif args.command == 'train-all':
        import torch

        # ── Phase 1: Supervised Training ──
        print("=" * 60)
        print("  Phase 1/2: Supervised Training")
        print("=" * 60)
        from train import run_training
        run_training(
            data_path=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            use_cpu=args.use_cpu,
            augment=args.augment,
        )

        # ── Phase 2: RL Self-Play ──
        print()
        print("=" * 60)
        print("  Phase 2/2: RL Self-Play Training")
        print("=" * 60)
        from train_rl import train_rl
        from chess_model import ChessCNN

        device = 'cpu' if args.use_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
        model = ChessCNN(hidden_size=256)

        # Load the model that was just trained in Phase 1
        model_path = os.path.join('models', 'chess_model.pth')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            print(f"Loaded supervised model from {model_path}")
        else:
            print("Warning: No supervised model found, starting RL from scratch")

        train_rl(
            model=model,
            num_games=args.rl_games,
            batch_size=args.rl_batch_size,
            learning_rate=args.rl_lr,
            device=device,
            save_interval=args.save_interval,
        )

        print()
        print("=" * 60)
        print("  Training complete! (Supervised + RL)")
        print("=" * 60)

    elif args.command == 'play':
        print("Starting chess game (console mode)...")
        from play_chess import play_game
        play_game(
            ai_color=args.ai_color,
            ai_depth=args.depth,
            model_path=args.model,
            classical_weight=args.classical_weight
        )
    elif args.command == 'gui':
        print("Starting chess game (GUI mode)...")
        from chess_gui import ChessGUI
        app = ChessGUI(
            ai_color=args.ai_color,
            ai_depth=args.depth,
            model_path=args.model if args.model else None,
            classical_weight=args.classical_weight
        )
        app.run()
    else:
        parser.print_help()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
