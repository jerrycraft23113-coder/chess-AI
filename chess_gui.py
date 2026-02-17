"""
Chess GUI - Play chess against AI with a graphical interface
Modern UI using CustomTkinter
"""

try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
except ImportError:
    import tkinter as tk
    import tkinter.ttk as ttk
    CTK_AVAILABLE = False
    print("CustomTkinter not available, using standard tkinter")

import tkinter as tk
from tkinter import ttk, messagebox
import chess
from typing import Optional, Tuple, Dict
import threading
import time
from pathlib import Path
from PIL import Image, ImageTk

from chess_board import ChessBoard
from play_chess import ChessAI


class ChessGUI:
    """Graphical user interface for playing chess against AI."""
    
    # Piece image file mapping
    PIECE_IMAGES = {
        'K': 'wk.png', 'Q': 'wq.png', 'R': 'wr.png', 'B': 'wb.png', 'N': 'wn.png', 'P': 'wp.png',  # White
        'k': 'bk.png', 'q': 'bq.png', 'r': 'br.png', 'b': 'bb.png', 'n': 'bn.png', 'p': 'bp.png'   # Black
    }
    
    # Colors - modern and improved
    LIGHT_SQUARE = '#f0d9b5'  # Light beige
    DARK_SQUARE = '#769656'   # Dark greenish-brown
    LIGHT_SQUARE_HOVER = '#e8d4a5'
    DARK_SQUARE_HOVER = '#6b8e5a'
    HIGHLIGHT = '#ffd700'     # Gold for legal moves (more elegant)
    SELECTED = '#4a90e2'      # Blue for selected square
    CHECK_COLOR = '#e74c3c'   # Modern red for check
    HEADER_BG = '#2c3e50'     # Modern dark blue-gray header
    SIDEBAR_BG = '#ecf0f1'    # Light gray sidebar
    WHITE_PIECE_COLOR = '#1a1a1a'  # Very dark/black for white pieces (better contrast)
    BLACK_PIECE_COLOR = '#000000'  # Pure black for black pieces (maximum contrast)
    LABEL_COLOR = '#34495e'   # Modern dark gray for labels
    TEXT_COLOR = '#2c3e50'    # Text color
    BUTTON_BG = '#3498db'    # Modern blue button
    BUTTON_HOVER = '#2980b9' # Darker blue on hover
    
    def __init__(self, ai_color: str = 'black', ai_depth: int = 5,
                 model_path: Optional[str] = None, classical_weight: float = 0.7):
        """Initialize the chess GUI.
        
        Args:
            ai_color: 'white' or 'black' - which color the AI plays
            ai_depth: Search depth for AI
            model_path: Path to trained model
        """
        self.board = ChessBoard()
        self.ai = ChessAI(model_path=model_path, depth=ai_depth,
                          classical_weight=classical_weight,
                          time_limit=15.0)
        self.ai_color = ai_color
        self.ai_is_white = (ai_color == 'white')
        self.classical_weight = classical_weight
        
        # Game state
        self.selected_square = None
        self.legal_moves = []
        self.is_ai_thinking = False
        self.game_timed_out = False

        # Chess clock - 10 minutes per player
        self.time_limit = 600.0
        self.white_time = self.time_limit
        self.black_time = self.time_limit
        self.timer_running = False
        self.last_tick_time = None
        self.timer_after_id = None
        
        # Initialize image variables (will be loaded after root window is created)
        self.images_dir = Path(__file__).parent / 'images'
        self.piece_images: Dict[str, ImageTk.PhotoImage] = {}
        self.board_image = None
        self.square_size = 170  # Size of each square in pixels (225% scale = 150% x 150%)
        # Store image references to prevent garbage collection
        self._image_refs = []
        
        # Create main window FIRST (needed for ImageTk.PhotoImage)
        if CTK_AVAILABLE:
            ctk.set_appearance_mode("light")
            ctk.set_default_color_theme("blue")
            self.root = ctk.CTk()
            self.root.title("Chess - Play against AI")
            self.root.resizable(False, False)
            self.use_ctk = True
        else:
            self.root = tk.Tk()
            self.root.title("Chess - Play against AI")
            self.root.resizable(False, False)
            self.root.configure(bg='white')
            self.use_ctk = False
        
        # Load images AFTER root window is created
        self.load_images()
        
        # Create UI
        self.create_widgets()
        self.update_display()
        
        # Start AI move if AI plays white
        if self.ai_is_white:
            self.root.after(500, self.make_ai_move)
    
    def load_images(self):
        """Load all piece images and board image."""
        try:
            print(f"Loading images from: {self.images_dir}")
            
            # Load board image
            board_path = self.images_dir / 'board.png'
            if board_path.exists():
                board_img = Image.open(board_path)
                # Resize board to fit 8x8 squares
                board_size = self.square_size * 8
                board_img = board_img.resize((board_size, board_size), Image.Resampling.LANCZOS)
                self.board_image = ImageTk.PhotoImage(board_img)
                self._image_refs.append(self.board_image)  # Keep reference
                print(f"Loaded board image: {board_size}x{board_size}")
            else:
                print(f"Warning: Board image not found: {board_path}")
            
            # Load piece images
            for piece_symbol, filename in self.PIECE_IMAGES.items():
                piece_path = self.images_dir / filename
                if piece_path.exists():
                    piece_img = Image.open(piece_path)
                    # Resize piece to fit in square (slightly smaller than square)
                    piece_size = int(self.square_size * 0.85)
                    piece_img = piece_img.resize((piece_size, piece_size), Image.Resampling.LANCZOS)
                    photo_img = ImageTk.PhotoImage(piece_img)
                    self.piece_images[piece_symbol] = photo_img
                    self._image_refs.append(photo_img)  # Keep reference to prevent garbage collection
                    print(f"Loaded piece image: {filename} -> {piece_symbol}")
                else:
                    print(f"Warning: Piece image not found: {piece_path}")
            
            print(f"Loaded {len(self.piece_images)} piece images")
        except Exception as e:
            import traceback
            print(f"Error loading images: {e}")
            traceback.print_exc()
            self.piece_images = {}
            self.board_image = None
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main frame
        if self.use_ctk:
            main_frame = ctk.CTkFrame(self.root, fg_color='white')
            main_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=0, pady=0)
            
            # Header with "Opponent" - modern design
            header_frame = ctk.CTkFrame(self.root, fg_color=self.HEADER_BG, height=105, corner_radius=0)
            header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=0, pady=0)
            header_frame.grid_propagate(False)

            # User icon
            icon_label = ctk.CTkLabel(
                header_frame,
                text="👤",
                font=ctk.CTkFont(size=34),
                fg_color=self.HEADER_BG,
                text_color='white'
            )
            icon_label.pack(side=tk.LEFT, padx=24, pady=20)

            # "Opponent" text
            opponent_label = ctk.CTkLabel(
                header_frame,
                text="Opponent",
                font=ctk.CTkFont(size=30, weight='bold'),
                fg_color=self.HEADER_BG,
                text_color='white'
            )
            opponent_label.pack(side=tk.LEFT, padx=8, pady=20)
        else:
            main_frame = tk.Frame(self.root, bg='white', padx=0, pady=0)
            main_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Header with "Opponent"
            header_frame = tk.Frame(self.root, bg=self.HEADER_BG, height=105)
            header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=0, pady=0)
            header_frame.pack_propagate(False)

            # User icon
            icon_label = tk.Label(
                header_frame,
                text="👤",
                font=('Arial', 34),
                bg=self.HEADER_BG,
                fg='white'
            )
            icon_label.pack(side=tk.LEFT, padx=24, pady=20)

            # "Opponent" text
            opponent_label = tk.Label(
                header_frame,
                text="Opponent",
                font=('Arial', 30, 'bold'),
                bg=self.HEADER_BG,
                fg='white'
            )
            opponent_label.pack(side=tk.LEFT, padx=8, pady=20)
        
        # Left panel - Board with labels
        if self.use_ctk:
            board_container = ctk.CTkFrame(main_frame, fg_color='white', corner_radius=10)
        else:
            board_container = tk.Frame(main_frame, bg='white')
        board_container.grid(row=0, column=0, padx=20, pady=20)
        
        # Row labels (1-8) - left of the board, pixel-aligned to canvas squares
        row_labels_frame = tk.Canvas(
            board_container, bg='white',
            width=45, height=self.square_size * 8,
            highlightthickness=0, borderwidth=0
        )
        row_labels_frame.grid(row=0, column=0, padx=(0, 6), sticky=tk.N)
        for row in range(8):
            cy = row * self.square_size + self.square_size // 2
            row_labels_frame.create_text(
                22, cy, text=str(8 - row),
                font=('Arial', 22, 'bold'),
                fill=self.LABEL_COLOR,
                anchor=tk.CENTER
            )
        
        # Create chess board using Canvas
        board_size = self.square_size * 8
        self.chess_canvas = tk.Canvas(
            board_container,
            width=board_size,
            height=board_size,
            highlightthickness=0,
            borderwidth=0
        )
        self.chess_canvas.grid(row=0, column=1, padx=0, pady=0)
        self.chess_canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Store piece positions on canvas for click detection
        self.square_rects = {}  # (row, col) -> canvas rectangle ID
        
        # Column labels (a-h) - below the board, pixel-aligned to canvas squares
        col_labels_frame = tk.Canvas(
            board_container, bg='white',
            width=self.square_size * 8, height=45,
            highlightthickness=0, borderwidth=0
        )
        col_labels_frame.grid(row=1, column=1, padx=0, pady=(6, 0))
        for col in range(8):
            cx = col * self.square_size + self.square_size // 2
            col_labels_frame.create_text(
                cx, 22, text=chr(97 + col),
                font=('Arial', 22, 'bold'),
                fill=self.LABEL_COLOR,
                anchor=tk.CENTER
            )
        
        # Right panel - Info and controls
        if self.use_ctk:
            sidebar_height = self.square_size * 8 + 45 + 6  # board + col labels + padding
            info_frame = ctk.CTkFrame(main_frame, fg_color=self.SIDEBAR_BG, corner_radius=10,
                                      width=510, height=sidebar_height)
            info_frame.grid(row=0, column=1, padx=18, pady=20, sticky=(tk.N))
            info_frame.grid_propagate(False)

            # Game info
            title_label = ctk.CTkLabel(
                info_frame,
                text="Chess Game",
                font=ctk.CTkFont(size=32, weight='bold'),
                text_color=self.TEXT_COLOR
            )
            title_label.pack(pady=(24, 8))

            self.status_label = ctk.CTkLabel(
                info_frame,
                text="White to move",
                font=ctk.CTkFont(size=24),
                text_color=self.TEXT_COLOR
            )
            self.status_label.pack(pady=8)

            self.info_label = ctk.CTkLabel(
                info_frame,
                text="",
                font=ctk.CTkFont(size=20),
                text_color=self.LABEL_COLOR,
                wraplength=460
            )
            self.info_label.pack(pady=8, padx=18)

            # Timer display
            timer_frame = ctk.CTkFrame(info_frame, fg_color='transparent')
            timer_frame.pack(pady=8, padx=18, fill=tk.X)

            self.white_timer_label = ctk.CTkLabel(
                timer_frame,
                text="♔ White   10:00",
                font=ctk.CTkFont(size=22, weight='bold'),
                text_color=self.TEXT_COLOR
            )
            self.white_timer_label.pack(pady=4, anchor=tk.W)

            self.black_timer_label = ctk.CTkLabel(
                timer_frame,
                text="♚ Black   10:00",
                font=ctk.CTkFont(size=22),
                text_color=self.LABEL_COLOR
            )
            self.black_timer_label.pack(pady=4, anchor=tk.W)

            # Controls
            control_frame = ctk.CTkFrame(info_frame, fg_color='transparent')
            control_frame.pack(pady=14, padx=18, fill=tk.X)

            new_game_btn = ctk.CTkButton(
                control_frame,
                text="New Game",
                command=self.new_game,
                fg_color=self.BUTTON_BG,
                hover_color=self.BUTTON_HOVER,
                corner_radius=10,
                height=56,
                font=ctk.CTkFont(size=22)
            )
            new_game_btn.pack(pady=6, fill=tk.X)

            undo_btn = ctk.CTkButton(
                control_frame,
                text="Undo Move",
                command=self.undo_move,
                fg_color=self.BUTTON_BG,
                hover_color=self.BUTTON_HOVER,
                corner_radius=10,
                height=56,
                font=ctk.CTkFont(size=22)
            )
            undo_btn.pack(pady=6, fill=tk.X)

            settings_btn = ctk.CTkButton(
                control_frame,
                text="AI Settings",
                command=self.show_ai_settings,
                fg_color=self.BUTTON_BG,
                hover_color=self.BUTTON_HOVER,
                corner_radius=10,
                height=56,
                font=ctk.CTkFont(size=22)
            )
            settings_btn.pack(pady=6, fill=tk.X)

            # Move history
            history_label = ctk.CTkLabel(
                info_frame,
                text="Move History",
                font=ctk.CTkFont(size=24, weight='bold'),
                text_color=self.TEXT_COLOR
            )
            history_label.pack(pady=(14, 8))

            history_frame = ctk.CTkFrame(info_frame, fg_color='white', corner_radius=5)
            history_frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 20))

            self.history_text = tk.Text(
                history_frame,
                height=14,
                width=28,
                wrap=tk.WORD,
                font=('Arial', 19),
                bg='white',
                fg=self.TEXT_COLOR,
                relief=tk.FLAT,
                borderwidth=0,
                padx=8,
                pady=8
            )
            scrollbar = ctk.CTkScrollbar(history_frame, command=self.history_text.yview)
            self.history_text.configure(yscrollcommand=scrollbar.set)
            self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        else:
            sidebar_height = self.square_size * 8 + 45 + 6
            info_frame = tk.Frame(main_frame, bg=self.SIDEBAR_BG, width=510, height=sidebar_height)
            info_frame.grid(row=0, column=1, padx=18, pady=20, sticky=(tk.N))
            info_frame.grid_propagate(False)

            # Game info
            ttk.Label(info_frame, text="Chess Game", font=('Arial', 28, 'bold')).pack(pady=8)

            self.status_label = ttk.Label(info_frame, text="White to move", font=('Arial', 22))
            self.status_label.pack(pady=8)

            self.info_label = ttk.Label(info_frame, text="", font=('Arial', 18), wraplength=400)
            self.info_label.pack(pady=8)

            # Timer display
            timer_frame = ttk.Frame(info_frame)
            timer_frame.pack(pady=8, fill=tk.X)

            self.white_timer_label = ttk.Label(
                timer_frame,
                text="♔ White   10:00",
                font=('Arial', 20, 'bold')
            )
            self.white_timer_label.pack(pady=4, anchor=tk.W)

            self.black_timer_label = ttk.Label(
                timer_frame,
                text="♚ Black   10:00",
                font=('Arial', 20)
            )
            self.black_timer_label.pack(pady=4, anchor=tk.W)

            # Separator
            ttk.Separator(info_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

            # Controls
            control_frame = ttk.Frame(info_frame)
            control_frame.pack(pady=8)

            ttk.Button(control_frame, text="New Game", command=self.new_game).pack(pady=6, fill=tk.X)
            ttk.Button(control_frame, text="Undo Move", command=self.undo_move).pack(pady=6, fill=tk.X)
            ttk.Button(control_frame, text="AI Settings", command=self.show_ai_settings).pack(pady=6, fill=tk.X)

            # Separator
            ttk.Separator(info_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

            # Move history
            ttk.Label(info_frame, text="Move History", font=('Arial', 20, 'bold')).pack()

            history_frame = ttk.Frame(info_frame)
            history_frame.pack(fill=tk.BOTH, expand=True)

            scrollbar = ttk.Scrollbar(history_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            self.history_text = tk.Text(history_frame, height=14, width=28, yscrollcommand=scrollbar.set, wrap=tk.WORD, font=('Arial', 17))
            self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=self.history_text.yview)
        
        # Disable text editing
        self.history_text.config(state=tk.DISABLED)
    
    
    def get_square_color(self, row: int, col: int) -> str:
        """Get background color for a square."""
        is_light = (row + col) % 2 == 0
        return self.LIGHT_SQUARE if is_light else self.DARK_SQUARE
    
    def update_display(self):
        """Update the chess board display using images."""
        # Clear canvas
        self.chess_canvas.delete("all")
        self.square_rects = {}
        
        # Draw board background
        if self.board_image:
            self.chess_canvas.create_image(0, 0, anchor=tk.NW, image=self.board_image)
        else:
            # Fallback: draw colored squares
            for row in range(8):
                for col in range(8):
                    x1 = col * self.square_size
                    y1 = row * self.square_size
                    x2 = x1 + self.square_size
                    y2 = y1 + self.square_size
                    bg_color = self.get_square_color(row, col)
                    self.chess_canvas.create_rectangle(x1, y1, x2, y2, fill=bg_color, outline='')
        
        # Draw pieces and highlights
        for row in range(8):
            for col in range(8):
                chess_square = chess.square(col, 7 - row)  # chess uses (file, rank)
                piece = self.board.board.piece_at(chess_square)
                
                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                center_x = x1 + self.square_size // 2
                center_y = y1 + self.square_size // 2
                
                # Draw highlights first (behind pieces)
                # Highlight selected square
                if self.selected_square == (row, col):
                    self.chess_canvas.create_rectangle(x1, y1, x2, y2, fill=self.SELECTED, outline='', tags='highlight')
                
                # Highlight legal moves
                elif (row, col) in self.legal_moves:
                    self.chess_canvas.create_rectangle(x1, y1, x2, y2, fill=self.HIGHLIGHT, outline='', tags='highlight')
                
                # Highlight check
                elif self.board.is_check() and piece and piece.piece_type == chess.KING and piece.color == self.board.get_turn():
                    self.chess_canvas.create_rectangle(x1, y1, x2, y2, fill=self.CHECK_COLOR, outline='', tags='highlight')
                
                # Draw piece image (on top of highlights)
                if piece:
                    piece_symbol = piece.symbol()
                    if piece_symbol in self.piece_images:
                        piece_img = self.piece_images[piece_symbol]
                        # Draw piece image
                        img_id = self.chess_canvas.create_image(center_x, center_y, image=piece_img, tags='piece', anchor=tk.CENTER)
                    else:
                        print(f"Warning: No image for piece symbol: {piece_symbol}")
                
                # Store rectangle for click detection (invisible, on top)
                rect_id = self.chess_canvas.create_rectangle(x1, y1, x2, y2, fill='', outline='', tags='square')
                self.square_rects[(row, col)] = rect_id
        
        # Update status
        self.update_status()
        self.update_move_history()
    
    def on_canvas_click(self, event):
        """Handle canvas click event."""
        if self.is_ai_thinking or self.board.is_game_over() or self.game_timed_out:
            return
        
        # Calculate which square was clicked
        col = int(event.x // self.square_size)
        row = int(event.y // self.square_size)
        
        if 0 <= row < 8 and 0 <= col < 8:
            self.on_square_click(row, col)
    
    def update_status(self):
        """Update status information."""
        if self.game_timed_out:
            loser = "White" if self.white_time <= 0 else "Black"
            winner = "Black" if loser == "White" else "White"
            status_msg = f"{winner} wins on time!"
            if self.use_ctk:
                self.status_label.configure(text=status_msg)
                self.info_label.configure(text=f"{loser} ran out of time")
            else:
                self.status_label.config(text=status_msg)
                self.info_label.config(text=f"{loser} ran out of time")
            return

        if self.is_ai_thinking:
            if self.use_ctk:
                self.status_label.configure(text="AI is thinking...")
            else:
                self.status_label.config(text="AI is thinking...")
            return

        if self.board.is_game_over():
            result = self.board.get_result()
            if result == '1-0':
                status_msg = "White wins!"
            elif result == '0-1':
                status_msg = "Black wins!"
            elif result == '1/2-1/2':
                status_msg = "Draw!"
            else:
                status_msg = "Game over!"
            
            if self.use_ctk:
                self.status_label.configure(text=status_msg)
                self.info_label.configure(text="")
            else:
                self.status_label.config(text=status_msg)
                self.info_label.config(text="")
            return
        
        current_player = "White" if self.board.get_turn() == chess.WHITE else "Black"
        status_text = f"{current_player} to move"
        
        if self.board.is_check():
            status_text += " (Check!)"
        
        if self.use_ctk:
            self.status_label.configure(text=status_text)
        else:
            self.status_label.config(text=status_text)
        
        # Show whose turn it is
        is_ai_turn = (self.ai_is_white and self.board.get_turn() == chess.WHITE) or \
                     (not self.ai_is_white and self.board.get_turn() == chess.BLACK)
        
        if is_ai_turn:
            info_msg = "Waiting for AI to move..."
        else:
            info_msg = "Your turn - Click to move"
        
        if self.use_ctk:
            self.info_label.configure(text=info_msg)
        else:
            self.info_label.config(text=info_msg)
    
    def update_move_history(self):
        """Update move history display."""
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete(1.0, tk.END)
        
        move_stack = self.board.board.move_stack
        for i in range(0, len(move_stack), 2):
            move_num = (i // 2) + 1
            white_move = move_stack[i].uci() if i < len(move_stack) else ""
            black_move = move_stack[i + 1].uci() if i + 1 < len(move_stack) else ""
            
            line = f"{move_num}. {white_move}"
            if black_move:
                line += f" {black_move}"
            line += "\n"
            
            self.history_text.insert(tk.END, line)
        
        self.history_text.see(tk.END)
        self.history_text.config(state=tk.DISABLED)
    
    def on_square_click(self, row: int, col: int):
        """Handle square click event."""
        if self.is_ai_thinking or self.board.is_game_over() or self.game_timed_out:
            return
        
        chess_square = chess.square(col, 7 - row)
        
        # Check if it's AI's turn
        is_ai_turn = (self.ai_is_white and self.board.get_turn() == chess.WHITE) or \
                     (not self.ai_is_white and self.board.get_turn() == chess.BLACK)
        
        if is_ai_turn:
            return
        
        # If a square is already selected, try to make a move
        if self.selected_square:
            from_row, from_col = self.selected_square
            from_square = chess.square(from_col, 7 - from_row)
            to_square = chess_square
            
            move = chess.Move(from_square, to_square)
            
            # Check if this is a legal move
            legal_moves = self.board.get_legal_moves()
            
            # Find all moves from from_square to to_square
            matching_moves = [m for m in legal_moves if m.from_square == from_square and m.to_square == to_square]
            
            if matching_moves:
                # Check if any of them require promotion
                promotion_moves = [m for m in matching_moves if m.promotion]
                
                if promotion_moves:
                    # Show promotion dialog
                    promotion_piece = self.show_promotion_dialog()
                    if promotion_piece:
                        # Find the move with the selected promotion
                        move = next((m for m in promotion_moves if m.promotion == promotion_piece), None)
                        if not move:
                            move = chess.Move(from_square, to_square, promotion=promotion_piece)
                    else:
                        # User cancelled, deselect
                        self.selected_square = None
                        self.legal_moves = []
                        self.update_display()
                        return
                else:
                    # Regular move (no promotion needed)
                    move = matching_moves[0]
                
                if self.make_move(move):
                    self.selected_square = None
                    self.legal_moves = []
                    self.update_display()

                    # Check if game is over after human move
                    if self.board.is_game_over():
                        self._stop_timer()
                        result = self.board.get_result()
                        if result == '1-0':
                            messagebox.showinfo("Game Over", "White wins!")
                        elif result == '0-1':
                            messagebox.showinfo("Game Over", "Black wins!")
                        elif result == '1/2-1/2':
                            messagebox.showinfo("Game Over", "Draw!")
                        return

                    # AI's turn
                    self.root.after(500, self.make_ai_move)
            else:
                # Invalid move, deselect
                self.selected_square = None
                self.legal_moves = []
                self.update_display()
        else:
            # Select a piece
            piece = self.board.board.piece_at(chess_square)
            if piece and piece.color == self.board.get_turn():
                self.selected_square = (row, col)
                # Get legal moves for this piece
                self.legal_moves = []
                for move in self.board.get_legal_moves():
                    if move.from_square == chess_square:
                        to_row = 7 - (move.to_square // 8)
                        to_col = move.to_square % 8
                        self.legal_moves.append((to_row, to_col))
                self.update_display()
            else:
                self.selected_square = None
                self.legal_moves = []
                self.update_display()
    
    def make_move(self, move: chess.Move):
        """Make a move on the board."""
        if self.board.make_move(move):
            # Start timer on first move, keep ticking for subsequent moves
            if not self.timer_running and not self.game_timed_out:
                self._start_timer()
            return True
        return False
    
    def make_ai_move(self):
        """Make AI move in a separate thread."""
        if self.is_ai_thinking or self.board.is_game_over() or self.game_timed_out:
            return
        
        is_ai_turn = (self.ai_is_white and self.board.get_turn() == chess.WHITE) or \
                     (not self.ai_is_white and self.board.get_turn() == chess.BLACK)
        
        if not is_ai_turn:
            return
        
        self.is_ai_thinking = True
        self.update_display()
        
        # Run AI thinking in a separate thread to avoid freezing GUI
        def ai_thread():
            move = self.ai.get_best_move(self.board)
            if move:
                self.root.after(0, lambda: self.complete_ai_move(move))
            else:
                self.root.after(0, lambda: self.complete_ai_move(None))
        
        thread = threading.Thread(target=ai_thread, daemon=True)
        thread.start()
    
    def complete_ai_move(self, move: Optional[chess.Move]):
        """Complete AI move on main thread."""
        self.is_ai_thinking = False

        if self.game_timed_out:
            self.update_display()
            return

        if move and self.make_move(move):
            self.update_display()

            # Check if game is over
            if self.board.is_game_over():
                self._stop_timer()
                result = self.board.get_result()
                if result == '1-0':
                    messagebox.showinfo("Game Over", "White wins!")
                elif result == '0-1':
                    messagebox.showinfo("Game Over", "Black wins!")
                elif result == '1/2-1/2':
                    messagebox.showinfo("Game Over", "Draw!")
        else:
            self.update_display()
    
    def new_game(self):
        """Start a new game."""
        if self.is_ai_thinking:
            messagebox.showinfo("New Game", "Please wait for AI to finish thinking.")
            return

        if messagebox.askyesno("New Game", "Start a new game?"):
            # Stop and reset timer
            self._stop_timer()
            self.white_time = self.time_limit
            self.black_time = self.time_limit
            self.game_timed_out = False

            self.board = ChessBoard()
            self.selected_square = None
            self.legal_moves = []
            self.is_ai_thinking = False
            # Clear AI transposition table for the new game
            self.ai.transposition_table.clear()
            self._update_timer_display()
            self.update_display()

            # Start AI move if AI plays white
            if self.ai_is_white:
                self.root.after(500, self.make_ai_move)

    def undo_move(self):
        """Undo the last move."""
        if self.is_ai_thinking:
            messagebox.showinfo("Undo", "Please wait for AI to finish thinking.")
            return

        if len(self.board.board.move_stack) == 0:
            messagebox.showinfo("Undo", "No moves to undo")
            return

        # Undo two moves (player + AI) if possible
        if len(self.board.board.move_stack) >= 2:
            self.board.board.pop()  # Undo AI move
            self.board.board.pop()  # Undo player move
        else:
            self.board.board.pop()  # Undo single move

        self.selected_square = None
        self.legal_moves = []
        self.update_display()
    
    def show_promotion_dialog(self) -> Optional[chess.PieceType]:
        """Show promotion piece selection dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Promote Pawn")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        result = [None]

        tk.Label(dialog, text="Choose promotion piece:", font=('Arial', 22),
                 bg='white', fg=self.TEXT_COLOR).pack(pady=16)

        button_frame = tk.Frame(dialog, bg='white')
        button_frame.pack(pady=16)

        pieces = [
            (chess.QUEEN, '♕', 'Queen'),
            (chess.ROOK, '♖', 'Rook'),
            (chess.BISHOP, '♗', 'Bishop'),
            (chess.KNIGHT, '♘', 'Knight')
        ]

        for piece_type, symbol, name in pieces:
            btn = tk.Button(
                button_frame,
                text=f"{symbol}\n{name}",
                width=8,
                height=3,
                font=('Arial', 24),
                command=lambda pt=piece_type: self._set_promotion(result, pt, dialog)
            )
            btn.pack(side=tk.LEFT, padx=8)
        
        # Wait for user selection
        dialog.wait_window()
        return result[0]
    
    def _set_promotion(self, result: list, piece_type: chess.PieceType, dialog: tk.Toplevel):
        """Set promotion piece and close dialog."""
        result[0] = piece_type
        dialog.destroy()
    
    def show_ai_settings(self):
        """Show AI settings dialog (200% scaled)."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("AI Settings")
        settings_window.resizable(False, False)

        main_frame = tk.Frame(settings_window, bg='white', padx=40, pady=30)
        main_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(main_frame, text="AI Settings", font=('Arial', 28, 'bold'),
                 bg='white', fg=self.TEXT_COLOR).grid(row=0, column=0, columnspan=2, pady=(0, 24))

        tk.Label(main_frame, text=f"AI Color: {self.ai_color.capitalize()}",
                 font=('Arial', 20), bg='white', fg=self.TEXT_COLOR).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=8)

        # Depth setting
        tk.Label(main_frame, text="Search depth:", font=('Arial', 20),
                 bg='white', fg=self.TEXT_COLOR).grid(row=2, column=0, sticky=tk.W, pady=8)
        depth_var = tk.IntVar(value=self.ai.depth)
        depth_spin = tk.Spinbox(main_frame, from_=1, to=10, textvariable=depth_var,
                                width=6, font=('Arial', 20))
        depth_spin.grid(row=2, column=1, sticky=tk.W, pady=8, padx=(16, 0))

        # Classical weight setting
        tk.Label(main_frame, text="Classical eval weight:", font=('Arial', 20),
                 bg='white', fg=self.TEXT_COLOR).grid(row=3, column=0, sticky=tk.W, pady=(16, 8))
        weight_var = tk.DoubleVar(value=self.classical_weight)
        weight_scale = tk.Scale(main_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                variable=weight_var, resolution=0.01, length=300,
                                font=('Arial', 16), bg='white', highlightthickness=0,
                                troughcolor='#ddd', sliderrelief=tk.FLAT)
        weight_scale.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=(16, 8), padx=(16, 0))

        weight_label = tk.Label(main_frame, text=f"{self.classical_weight:.2f}",
                                font=('Arial', 18), bg='white', fg=self.LABEL_COLOR)
        weight_label.grid(row=4, column=1, sticky=tk.W, pady=(0, 10), padx=(16, 0))

        def on_weight_change(*_):
            weight_label.config(text=f"{weight_var.get():.2f}")

        weight_var.trace_add("write", on_weight_change)

        # Buttons
        button_frame = tk.Frame(main_frame, bg='white')
        button_frame.grid(row=5, column=0, columnspan=2, pady=(24, 0))

        def apply_settings():
            try:
                new_depth = int(depth_var.get())
                new_depth = max(1, min(10, new_depth))
                self.ai.depth = new_depth
                new_weight = max(0.0, min(1.0, float(weight_var.get())))
                self.classical_weight = new_weight
                self.ai.classical_weight = new_weight
            except Exception:
                pass
            settings_window.destroy()

        ok_btn = tk.Button(button_frame, text="OK", command=apply_settings,
                           font=('Arial', 18), width=10, height=1,
                           bg=self.BUTTON_BG, fg='white', relief=tk.FLAT,
                           activebackground=self.BUTTON_HOVER, activeforeground='white')
        ok_btn.pack(side=tk.LEFT, padx=10)

        cancel_btn = tk.Button(button_frame, text="Cancel", command=settings_window.destroy,
                               font=('Arial', 18), width=10, height=1,
                               bg='#95a5a6', fg='white', relief=tk.FLAT,
                               activebackground='#7f8c8d', activeforeground='white')
        cancel_btn.pack(side=tk.LEFT, padx=10)

        # Center the dialog on screen
        settings_window.update_idletasks()
        sw = settings_window.winfo_width()
        sh = settings_window.winfo_height()
        x = (settings_window.winfo_screenwidth() // 2) - (sw // 2)
        y = (settings_window.winfo_screenheight() // 2) - (sh // 2)
        settings_window.geometry(f"+{x}+{y}")
    
    # ─── Chess Clock Methods ────────────────────────────────────

    def _start_timer(self):
        """Start the chess clock ticking."""
        if self.timer_running:
            return
        self.timer_running = True
        self.last_tick_time = time.time()
        self._tick_timer()

    def _stop_timer(self):
        """Stop the chess clock."""
        self.timer_running = False
        if self.timer_after_id is not None:
            self.root.after_cancel(self.timer_after_id)
            self.timer_after_id = None

    def _tick_timer(self):
        """Tick the chess clock (called every 100ms)."""
        if not self.timer_running or self.board.is_game_over() or self.game_timed_out:
            return

        now = time.time()
        elapsed = now - self.last_tick_time
        self.last_tick_time = now

        # Deduct from current player's clock
        if self.board.get_turn() == chess.WHITE:
            self.white_time -= elapsed
            if self.white_time <= 0:
                self.white_time = 0
                self._update_timer_display()
                self._handle_timeout(chess.WHITE)
                return
        else:
            self.black_time -= elapsed
            if self.black_time <= 0:
                self.black_time = 0
                self._update_timer_display()
                self._handle_timeout(chess.BLACK)
                return

        self._update_timer_display()
        self.timer_after_id = self.root.after(100, self._tick_timer)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        seconds = max(0, seconds)
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{mins:02d}:{secs:02d}"

    def _update_timer_display(self):
        """Update timer labels with current times."""
        white_str = f"\u2654 White   {self._format_time(self.white_time)}"
        black_str = f"\u265a Black   {self._format_time(self.black_time)}"

        is_white_turn = self.board.get_turn() == chess.WHITE
        white_low = self.white_time < 30
        black_low = self.black_time < 30

        if self.use_ctk:
            # Active player bold, inactive normal
            self.white_timer_label.configure(
                text=white_str,
                font=ctk.CTkFont(size=22, weight='bold' if is_white_turn else 'normal'),
                text_color='#e74c3c' if white_low else (self.TEXT_COLOR if is_white_turn else self.LABEL_COLOR)
            )
            self.black_timer_label.configure(
                text=black_str,
                font=ctk.CTkFont(size=22, weight='bold' if not is_white_turn else 'normal'),
                text_color='#e74c3c' if black_low else (self.TEXT_COLOR if not is_white_turn else self.LABEL_COLOR)
            )
        else:
            self.white_timer_label.config(
                text=white_str,
                font=('Arial', 20, 'bold' if is_white_turn else 'normal'),
                foreground='#e74c3c' if white_low else ''
            )
            self.black_timer_label.config(
                text=black_str,
                font=('Arial', 20, 'bold' if not is_white_turn else 'normal'),
                foreground='#e74c3c' if black_low else ''
            )

    def _handle_timeout(self, color):
        """Handle when a player runs out of time."""
        self.timer_running = False
        self.game_timed_out = True
        if self.timer_after_id is not None:
            self.root.after_cancel(self.timer_after_id)
            self.timer_after_id = None

        loser = "White" if color == chess.WHITE else "Black"
        winner = "Black" if color == chess.WHITE else "White"
        messagebox.showinfo("Time's Up!", f"{loser} ran out of time!\n{winner} wins!")
        self.update_display()

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


def main():
    """Main function to start the chess GUI."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Chess GUI - Play against AI')
    parser.add_argument('--ai-color', choices=['white', 'black'], default='black',
                       help='Color the AI plays (default: black)')
    parser.add_argument('--depth', type=int, default=5,
                       help='AI search depth (default: 5)')
    parser.add_argument('--model', type=str, default='models/chess_model.pth',
                       help='Path to trained model (default: models/chess_model.pth)')
    parser.add_argument('--classical-weight', type=float, default=0.7,
                       help='Weight for classical evaluation vs neural net (0-1, default: 0.7)')
    
    args = parser.parse_args()
    
    # Create and run GUI
    app = ChessGUI(
        ai_color=args.ai_color,
        ai_depth=args.depth,
        model_path=args.model if args.model else None,
        classical_weight=args.classical_weight
    )
    app.run()


if __name__ == "__main__":
    main()
