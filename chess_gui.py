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
import copy
import logging
import sys
import time
from pathlib import Path
from PIL import Image, ImageTk

from chess_board import ChessBoard
from play_chess import ChessAI, PLAY_STYLES, STYLE_RANDOM

logger = logging.getLogger(__name__)
_DIRTY = object()  # sentinel: forces canvas redraw on next update_display

# ── Sound effects (Windows only, non-blocking) ──────────────────
if sys.platform == 'win32':
    try:
        import winsound
        def _beep(freq, duration):
            threading.Thread(target=winsound.Beep, args=(freq, duration), daemon=True).start()
        def _sound_move():    _beep(800, 80)
        def _sound_capture(): _beep(400, 100)
        def _sound_check():   _beep(1200, 150)
        def _sound_game_end(): _beep(600, 300)
    except Exception:
        def _sound_move(): pass
        def _sound_capture(): pass
        def _sound_check(): pass
        def _sound_game_end(): pass
else:
    def _sound_move(): pass
    def _sound_capture(): pass
    def _sound_check(): pass
    def _sound_game_end(): pass


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
    LAST_MOVE_COLOR = '#cdd16a'  # Yellowish-green for last move (like Lichess)
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
        """Initialize the chess GUI."""
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
        self.last_move = None  # Track last move for highlighting

        # Board flip: flip when player plays black (AI is white)
        self.flip_board = (ai_color == 'white')

        # Chess clock - 15 minutes per player
        self.time_limit = 900.0
        self.white_time = self.time_limit
        self.black_time = self.time_limit
        self.timer_running = False
        self.last_tick_time = None
        self.timer_after_id = None
        self._clock_color = chess.WHITE

        # AI threading state
        self._ai_thread = None
        self._ai_result = [None]
        self._game_gen = 0  # incremented on new game to invalidate stale callbacks

        # Pondering state
        self._ponder_thread = None
        self._ponder_move = None  # Predicted opponent move

        # Initialize image variables
        self.images_dir = Path(__file__).parent / 'images'
        self.piece_images: Dict[str, ImageTk.PhotoImage] = {}
        self.board_image = None
        self.square_size = 170
        self._image_refs = []

        # Create main window
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

        # Pre-create cached fonts for timer
        self._timer_font_bold = ('Consolas', 20, 'bold')
        self._timer_font_normal = ('Consolas', 20)
        self._last_white_text = None
        self._last_black_text = None
        self._last_white_bold = None
        self._last_black_bold = None
        self._last_white_color = None
        self._last_black_color = None

        # Load images
        self.load_images()

        # Create UI
        self.create_widgets()
        self._setup_canvas()
        self.update_display()

        # Start AI move if AI plays white
        if self.ai_is_white:
            self.root.after(500, self.make_ai_move)

    def load_images(self):
        """Load all piece images and board image."""
        try:
            logger.info(f"Loading images from: {self.images_dir}")

            # Load board image
            board_path = self.images_dir / 'board.png'
            if board_path.exists():
                board_img = Image.open(board_path)
                board_size = self.square_size * 8
                board_img = board_img.resize((board_size, board_size), Image.Resampling.LANCZOS)
                self.board_image = ImageTk.PhotoImage(board_img)
                self._image_refs.append(self.board_image)

            # Load piece images
            for piece_symbol, filename in self.PIECE_IMAGES.items():
                piece_path = self.images_dir / filename
                if piece_path.exists():
                    piece_img = Image.open(piece_path)
                    piece_size = int(self.square_size * 0.85)
                    piece_img = piece_img.resize((piece_size, piece_size), Image.Resampling.LANCZOS)
                    photo_img = ImageTk.PhotoImage(piece_img)
                    self.piece_images[piece_symbol] = photo_img
                    self._image_refs.append(photo_img)

            logger.info(f"Loaded {len(self.piece_images)} piece images")
        except Exception as e:
            import traceback
            logger.error(f"Error loading images: {e}")
            traceback.print_exc()
            self.piece_images = {}
            self.board_image = None

    def create_widgets(self):
        """Create all GUI widgets."""
        # Main frame
        if self.use_ctk:
            main_frame = ctk.CTkFrame(self.root, fg_color='white')
            main_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=0, pady=0)

            # Header with "Opponent"
            header_frame = ctk.CTkFrame(self.root, fg_color=self.HEADER_BG, height=105, corner_radius=0)
            header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=0, pady=0)
            header_frame.grid_propagate(False)

            icon_label = ctk.CTkLabel(
                header_frame, text="\U0001f464",
                font=ctk.CTkFont(size=34),
                fg_color=self.HEADER_BG, text_color='white'
            )
            icon_label.pack(side=tk.LEFT, padx=24, pady=20)

            opponent_label = ctk.CTkLabel(
                header_frame, text="Opponent",
                font=ctk.CTkFont(size=30, weight='bold'),
                fg_color=self.HEADER_BG, text_color='white'
            )
            opponent_label.pack(side=tk.LEFT, padx=8, pady=20)
        else:
            main_frame = tk.Frame(self.root, bg='white', padx=0, pady=0)
            main_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

            header_frame = tk.Frame(self.root, bg=self.HEADER_BG, height=105)
            header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=0, pady=0)
            header_frame.pack_propagate(False)

            icon_label = tk.Label(
                header_frame, text="\U0001f464",
                font=('Arial', 34), bg=self.HEADER_BG, fg='white'
            )
            icon_label.pack(side=tk.LEFT, padx=24, pady=20)

            opponent_label = tk.Label(
                header_frame, text="Opponent",
                font=('Arial', 30, 'bold'), bg=self.HEADER_BG, fg='white'
            )
            opponent_label.pack(side=tk.LEFT, padx=8, pady=20)

        # Left panel - Board with labels
        if self.use_ctk:
            board_container = ctk.CTkFrame(main_frame, fg_color='white', corner_radius=10)
        else:
            board_container = tk.Frame(main_frame, bg='white')
        board_container.grid(row=0, column=0, padx=20, pady=20)

        # Row labels (1-8) - left of the board
        self.row_labels_canvas = tk.Canvas(
            board_container, bg='white',
            width=45, height=self.square_size * 8,
            highlightthickness=0, borderwidth=0
        )
        self.row_labels_canvas.grid(row=0, column=0, padx=(0, 6), sticky=tk.N)
        self._draw_row_labels()

        # Chess board canvas
        board_size = self.square_size * 8
        self.chess_canvas = tk.Canvas(
            board_container, width=board_size, height=board_size,
            highlightthickness=0, borderwidth=0
        )
        self.chess_canvas.grid(row=0, column=1, padx=0, pady=0)
        self.chess_canvas.bind("<Button-1>", self.on_canvas_click)

        self.square_rects = {}

        # Eval bar (30px wide, between board and sidebar)
        self.eval_bar_canvas = tk.Canvas(
            board_container, width=30, height=board_size,
            bg='white', highlightthickness=1, highlightbackground='#ccc'
        )
        self.eval_bar_canvas.grid(row=0, column=2, padx=(6, 0), sticky=tk.N)
        self.update_eval_bar(0.0)  # Start at even

        # Column labels (a-h)
        self.col_labels_canvas = tk.Canvas(
            board_container, bg='white',
            width=self.square_size * 8, height=45,
            highlightthickness=0, borderwidth=0
        )
        self.col_labels_canvas.grid(row=1, column=1, padx=0, pady=(6, 0))
        self._draw_col_labels()

        # Right panel - Info and controls
        if self.use_ctk:
            sidebar_height = self.square_size * 8 + 45 + 6
            info_frame = ctk.CTkFrame(main_frame, fg_color=self.SIDEBAR_BG, corner_radius=10,
                                      width=510, height=sidebar_height)
            info_frame.grid(row=0, column=1, padx=18, pady=20, sticky=(tk.N))
            info_frame.grid_propagate(False)

            title_label = ctk.CTkLabel(
                info_frame, text="Chess Game",
                font=ctk.CTkFont(size=32, weight='bold'),
                text_color=self.TEXT_COLOR
            )
            title_label.pack(pady=(24, 8))

            self.status_label = ctk.CTkLabel(
                info_frame, text="White to move",
                font=ctk.CTkFont(size=24),
                text_color=self.TEXT_COLOR,
                wraplength=460
            )
            self.status_label.pack(pady=8)

            self.info_label = ctk.CTkLabel(
                info_frame, text="",
                font=ctk.CTkFont(size=20),
                text_color=self.LABEL_COLOR,
                wraplength=460
            )
            self.info_label.pack(pady=8, padx=18)

            # Timer display
            timer_frame = tk.Frame(info_frame, bg='white', bd=2, relief=tk.GROOVE,
                                   padx=12, pady=8)
            timer_frame.pack(pady=8, padx=18, fill=tk.X)

            self.white_timer_label = tk.Label(
                timer_frame, text="\u2654 White   15:00",
                font=('Consolas', 20, 'bold'),
                bg='white', fg=self.TEXT_COLOR, anchor='w'
            )
            self.white_timer_label.pack(pady=(4, 2), padx=8, anchor=tk.W, fill=tk.X)

            self.black_timer_label = tk.Label(
                timer_frame, text="\u265a Black   15:00",
                font=('Consolas', 20),
                bg='white', fg=self.LABEL_COLOR, anchor='w'
            )
            self.black_timer_label.pack(pady=(2, 4), padx=8, anchor=tk.W, fill=tk.X)

            # Controls
            control_frame = ctk.CTkFrame(info_frame, fg_color='transparent')
            control_frame.pack(pady=14, padx=18, fill=tk.X)

            new_game_btn = ctk.CTkButton(
                control_frame, text="New Game", command=self.new_game,
                fg_color=self.BUTTON_BG, hover_color=self.BUTTON_HOVER,
                corner_radius=10, height=56, font=ctk.CTkFont(size=22)
            )
            new_game_btn.pack(pady=6, fill=tk.X)

            undo_btn = ctk.CTkButton(
                control_frame, text="Undo Move", command=self.undo_move,
                fg_color=self.BUTTON_BG, hover_color=self.BUTTON_HOVER,
                corner_radius=10, height=56, font=ctk.CTkFont(size=22)
            )
            undo_btn.pack(pady=6, fill=tk.X)

            settings_btn = ctk.CTkButton(
                control_frame, text="AI Settings", command=self.show_ai_settings,
                fg_color=self.BUTTON_BG, hover_color=self.BUTTON_HOVER,
                corner_radius=10, height=56, font=ctk.CTkFont(size=22)
            )
            settings_btn.pack(pady=6, fill=tk.X)

            # Move history
            history_label = ctk.CTkLabel(
                info_frame, text="Move History",
                font=ctk.CTkFont(size=24, weight='bold'),
                text_color=self.TEXT_COLOR
            )
            history_label.pack(pady=(14, 8))

            history_frame = ctk.CTkFrame(info_frame, fg_color='white', corner_radius=5)
            history_frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 20))

            self.history_text = tk.Text(
                history_frame, height=14, width=28,
                wrap=tk.WORD, font=('Arial', 19),
                bg='white', fg=self.TEXT_COLOR,
                relief=tk.FLAT, borderwidth=0, padx=8, pady=8
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

            ttk.Label(info_frame, text="Chess Game", font=('Arial', 28, 'bold')).pack(pady=8)

            self.status_label = ttk.Label(info_frame, text="White to move",
                                          font=('Arial', 22), wraplength=400)
            self.status_label.pack(pady=8)

            self.info_label = ttk.Label(info_frame, text="",
                                        font=('Arial', 18), wraplength=400)
            self.info_label.pack(pady=8)

            # Timer display
            timer_frame = tk.Frame(info_frame, bg='white', bd=2, relief=tk.GROOVE,
                                   padx=12, pady=8)
            timer_frame.pack(pady=8, padx=18, fill=tk.X)

            self.white_timer_label = tk.Label(
                timer_frame, text="\u2654 White   15:00",
                font=self._timer_font_bold,
                bg='white', fg=self.TEXT_COLOR, anchor='w'
            )
            self.white_timer_label.pack(pady=4, anchor=tk.W, fill=tk.X)

            self.black_timer_label = tk.Label(
                timer_frame, text="\u265a Black   15:00",
                font=self._timer_font_normal,
                bg='white', fg=self.LABEL_COLOR, anchor='w'
            )
            self.black_timer_label.pack(pady=4, anchor=tk.W, fill=tk.X)

            ttk.Separator(info_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

            control_frame = ttk.Frame(info_frame)
            control_frame.pack(pady=8)

            ttk.Button(control_frame, text="New Game", command=self.new_game).pack(pady=6, fill=tk.X)
            ttk.Button(control_frame, text="Undo Move", command=self.undo_move).pack(pady=6, fill=tk.X)
            ttk.Button(control_frame, text="AI Settings", command=self.show_ai_settings).pack(pady=6, fill=tk.X)

            ttk.Separator(info_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

            ttk.Label(info_frame, text="Move History", font=('Arial', 20, 'bold')).pack()

            history_frame = ttk.Frame(info_frame)
            history_frame.pack(fill=tk.BOTH, expand=True)

            scrollbar = ttk.Scrollbar(history_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            self.history_text = tk.Text(history_frame, height=14, width=28,
                                        yscrollcommand=scrollbar.set,
                                        wrap=tk.WORD, font=('Arial', 17))
            self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=self.history_text.yview)

        # Disable text editing
        self.history_text.config(state=tk.DISABLED)

    def _draw_row_labels(self):
        """Draw row labels (rank numbers) on the side canvas."""
        self.row_labels_canvas.delete("all")
        for row in range(8):
            cy = row * self.square_size + self.square_size // 2
            if self.flip_board:
                label = str(row + 1)  # 1 at top, 8 at bottom when flipped
            else:
                label = str(8 - row)  # 8 at top, 1 at bottom normally
            self.row_labels_canvas.create_text(
                22, cy, text=label,
                font=('Arial', 22, 'bold'),
                fill=self.LABEL_COLOR, anchor=tk.CENTER
            )

    def _draw_col_labels(self):
        """Draw column labels (file letters) below the board."""
        self.col_labels_canvas.delete("all")
        for col in range(8):
            cx = col * self.square_size + self.square_size // 2
            if self.flip_board:
                label = chr(104 - col)  # h-a when flipped
            else:
                label = chr(97 + col)   # a-h normally
            self.col_labels_canvas.create_text(
                cx, 22, text=label,
                font=('Arial', 22, 'bold'),
                fill=self.LABEL_COLOR, anchor=tk.CENTER
            )

    def _setup_canvas(self):
        """Create persistent canvas items once. Called after create_widgets()."""
        c = self.chess_canvas
        sz = self.square_size

        # Layer 1: Board background
        if self.board_image:
            c.create_image(0, 0, anchor=tk.NW, image=self.board_image, tags='bg')
        else:
            for row in range(8):
                for col in range(8):
                    x1, y1 = col * sz, row * sz
                    bg = self.LIGHT_SQUARE if (row + col) % 2 == 0 else self.DARK_SQUARE
                    c.create_rectangle(x1, y1, x1 + sz, y1 + sz, fill=bg, outline='', tags='bg')

        # Layer 2: Highlight rectangles (initially hidden)
        self._hl_ids = [[None] * 8 for _ in range(8)]
        for row in range(8):
            for col in range(8):
                x1, y1 = col * sz, row * sz
                rid = c.create_rectangle(x1, y1, x1 + sz, y1 + sz, fill='', outline='', state='hidden')
                self._hl_ids[row][col] = rid

        # Layer 3: Piece images (initially blank)
        self._pc_ids = [[None] * 8 for _ in range(8)]
        # Blank 1x1 transparent image as placeholder
        self._blank_img = ImageTk.PhotoImage(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))
        self._image_refs.append(self._blank_img)
        for row in range(8):
            for col in range(8):
                cx = col * sz + sz // 2
                cy = row * sz + sz // 2
                pid = c.create_image(cx, cy, image=self._blank_img, anchor=tk.CENTER)
                self._pc_ids[row][col] = pid

        # Cache previous state for dirty checking
        self._prev_highlights = [[None] * 8 for _ in range(8)]
        self._prev_pieces = [[None] * 8 for _ in range(8)]

    def _gui_square_to_chess(self, row: int, col: int) -> int:
        """Convert GUI (row, col) to chess square index, accounting for board flip."""
        if self.flip_board:
            return chess.square(7 - col, row)
        else:
            return chess.square(col, 7 - row)

    def _chess_square_to_gui(self, sq: int) -> Tuple[int, int]:
        """Convert chess square index to GUI (row, col), accounting for board flip."""
        file = sq % 8
        rank = sq // 8
        if self.flip_board:
            return (rank, 7 - file)
        else:
            return (7 - rank, file)

    def get_square_color(self, row: int, col: int) -> str:
        """Get background color for a square."""
        is_light = (row + col) % 2 == 0
        return self.LIGHT_SQUARE if is_light else self.DARK_SQUARE

    def update_display(self):
        """Update the chess board display — incremental, only updates changed items."""
        c = self.chess_canvas
        hl_ids = self._hl_ids
        pc_ids = self._pc_ids
        prev_hl = self._prev_highlights
        prev_pc = self._prev_pieces
        board = self.board.board
        is_check = board.is_check()
        turn = board.turn

        # Pre-compute legal move set for fast lookup
        legal_set = set(self.legal_moves) if self.legal_moves else set()

        for row in range(8):
            for col in range(8):
                chess_square = self._gui_square_to_chess(row, col)
                piece = board.piece_at(chess_square)

                # Determine highlight color for this square
                hl_color = None
                if self.last_move and (chess_square == self.last_move.from_square or chess_square == self.last_move.to_square):
                    hl_color = self.LAST_MOVE_COLOR
                if self.selected_square == (row, col):
                    hl_color = self.SELECTED
                elif (row, col) in legal_set:
                    hl_color = self.HIGHLIGHT
                elif is_check and piece and piece.piece_type == chess.KING and piece.color == turn:
                    hl_color = self.CHECK_COLOR

                # Update highlight only if changed
                if hl_color != prev_hl[row][col]:
                    if hl_color:
                        c.itemconfig(hl_ids[row][col], fill=hl_color, state='normal')
                    else:
                        c.itemconfig(hl_ids[row][col], state='hidden')
                    prev_hl[row][col] = hl_color

                # Determine piece symbol
                pc_sym = piece.symbol() if piece else None

                # Update piece only if changed
                if pc_sym != prev_pc[row][col]:
                    if pc_sym and pc_sym in self.piece_images:
                        c.itemconfig(pc_ids[row][col], image=self.piece_images[pc_sym])
                    else:
                        c.itemconfig(pc_ids[row][col], image=self._blank_img)
                    prev_pc[row][col] = pc_sym

        self.update_status()
        self.update_move_history()
        # Force immediate screen render (don't let ponder thread delay it)
        self.root.update_idletasks()

    def on_canvas_click(self, event):
        """Handle canvas click event."""
        if self.is_ai_thinking or self.board.is_game_over() or self.game_timed_out:
            return

        # Pause pondering to free CPU/GIL for responsive GUI
        if self._ponder_thread and self._ponder_thread.is_alive():
            self.ai.time_up = True

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
        """Update move history display — only append new moves, don't rebuild all."""
        move_stack = self.board.board.move_stack
        n = len(move_stack)

        # Skip if nothing changed
        if n == getattr(self, '_hist_len', -1):
            return

        # If moves were undone, do a full rebuild
        if n < getattr(self, '_hist_len', 0):
            self._hist_len = 0
            self._hist_san = []
            self._hist_board = chess.Board()
            self.history_text.config(state=tk.NORMAL)
            self.history_text.delete(1.0, tk.END)

        # Init on first call
        if not hasattr(self, '_hist_san'):
            self._hist_len = 0
            self._hist_san = []
            self._hist_board = chess.Board()

        # Compute SAN for new moves only
        self.history_text.config(state=tk.NORMAL)
        for i in range(self._hist_len, n):
            move = move_stack[i]
            try:
                san = self._hist_board.san(move)
                self._hist_san.append(san)
                self._hist_board.push(move)
            except Exception:
                self._hist_san.append(move.uci())
                try:
                    self._hist_board.push(move)
                except Exception:
                    break

            idx = len(self._hist_san) - 1
            if idx % 2 == 0:
                # White move — start new line
                move_num = (idx // 2) + 1
                num_str = f"{move_num}.".rjust(4)
                self.history_text.insert(tk.END, f"{num_str} {san:<7s} ")
            else:
                # Black move — append to current line
                self.history_text.insert(tk.END, f"{san}\n")

        self._hist_len = n
        self.history_text.see(tk.END)
        self.history_text.config(state=tk.DISABLED)

    def on_square_click(self, row: int, col: int):
        """Handle square click event."""
        if self.is_ai_thinking or self.board.is_game_over() or self.game_timed_out:
            return

        chess_square = self._gui_square_to_chess(row, col)

        is_ai_turn = (self.ai_is_white and self.board.get_turn() == chess.WHITE) or \
                     (not self.ai_is_white and self.board.get_turn() == chess.BLACK)

        if is_ai_turn:
            return

        if self.selected_square:
            from_row, from_col = self.selected_square
            from_square = self._gui_square_to_chess(from_row, from_col)
            to_square = chess_square

            move = chess.Move(from_square, to_square)
            legal_moves = self.board.get_legal_moves()
            matching_moves = [m for m in legal_moves if m.from_square == from_square and m.to_square == to_square]

            if matching_moves:
                promotion_moves = [m for m in matching_moves if m.promotion]

                if promotion_moves:
                    promotion_piece = self.show_promotion_dialog()
                    if promotion_piece:
                        move = next((m for m in promotion_moves if m.promotion == promotion_piece), None)
                        if not move:
                            move = chess.Move(from_square, to_square, promotion=promotion_piece)
                    else:
                        self.selected_square = None
                        self.legal_moves = []
                        self.update_display()
                        return
                else:
                    move = matching_moves[0]

                if self.make_move(move):
                    self.selected_square = None
                    self.legal_moves = []
                    self.update_display()

                    if self.board.is_game_over():
                        self._stop_timer()
                        _sound_game_end()
                        result = self.board.get_result()
                        if result == '1-0':
                            messagebox.showinfo("Game Over", "White wins!")
                        elif result == '0-1':
                            messagebox.showinfo("Game Over", "Black wins!")
                        elif result == '1/2-1/2':
                            messagebox.showinfo("Game Over", "Draw!")
                        return

                    self.root.after(500, self.make_ai_move)
            else:
                self.selected_square = None
                self.legal_moves = []
                self.update_display()
        else:
            piece = self.board.board.piece_at(chess_square)
            if piece and piece.color == self.board.get_turn():
                self.selected_square = (row, col)
                self.legal_moves = []
                for move in self.board.get_legal_moves():
                    if move.from_square == chess_square:
                        to_row, to_col = self._chess_square_to_gui(move.to_square)
                        self.legal_moves.append((to_row, to_col))
                self.update_display()
            else:
                self.selected_square = None
                self.legal_moves = []
                self.update_display()

    def make_move(self, move: chess.Move):
        """Make a move on the board with sound effects."""
        is_capture = self.board.board.is_capture(move)
        gives_check = self.board.board.gives_check(move)

        if self.board.make_move(move):
            self.last_move = move
            self._clock_color = self.board.board.turn
            # Sound effects
            if gives_check:
                _sound_check()
            elif is_capture:
                _sound_capture()
            else:
                _sound_move()
            if not self.timer_running and not self.game_timed_out:
                self._start_timer()
            return True
        return False

    def make_ai_move(self):
        """Start AI search in a background thread (preserves TT across moves)."""
        if self.is_ai_thinking or self.board.is_game_over() or self.game_timed_out:
            return

        is_ai_turn = (self.ai_is_white and self.board.get_turn() == chess.WHITE) or \
                     (not self.ai_is_white and self.board.get_turn() == chess.BLACK)

        if not is_ai_turn:
            return

        # Stop pondering before starting actual search
        self._stop_pondering(actual_move=self.last_move)

        self.is_ai_thinking = True
        self._ai_result = [None]
        self.update_status()

        # Smart time management: allocate time based on remaining clock
        remaining = self.white_time if self.board.get_turn() == chess.WHITE else self.black_time
        move_number = len(self.board.board.move_stack) // 2 + 1
        # More aggressive: assume ~35 moves per game, keep buffer
        moves_left = max(12, 40 - move_number)
        move_time = remaining / moves_left
        # Cap: fast enough to not bore player, strong enough to play well
        move_time = max(1.0, min(5.0, move_time))
        # Endgame: search is faster, don't waste time
        piece_count = bin(self.board.board.occupied).count('1')
        if piece_count <= 10:
            move_time = min(move_time, 3.0)
        self.ai.time_limit = move_time
        self.ai._tl = move_time

        def _run():
            try:
                move = self.ai.get_best_move(self.board)
                self._ai_result[0] = move
            except Exception as e:
                logger.error(f"AI thread error: {e}")

        self._ai_thread = threading.Thread(target=_run, daemon=True)
        self._ai_thread.start()
        self._poll_ai_result(self._game_gen)

    def _poll_ai_result(self, gen=None):
        """Poll for AI result every 50ms on the main thread."""
        # Stale callback from a previous game — discard
        if gen is not None and gen != self._game_gen:
            return

        if self.game_timed_out:
            self.is_ai_thinking = False
            self.update_status()
            return

        if self._ai_thread is not None and self._ai_thread.is_alive():
            self.root.after(50, lambda: self._poll_ai_result(gen))
            return

        move = self._ai_result[0]
        self._ai_thread = None
        self.is_ai_thinking = False
        self.complete_ai_move(move)

    def complete_ai_move(self, move: Optional[chess.Move]):
        """Complete AI move on main thread."""
        self.is_ai_thinking = False

        if self.game_timed_out:
            self.update_display()
            return

        if move and self.make_move(move):
            # Update eval bar with AI's last evaluation score
            try:
                score = self.ai.last_score
                self.update_eval_bar(score)
            except AttributeError:
                pass

            self.update_display()

            if self.board.is_game_over():
                self._stop_timer()
                _sound_game_end()
                result = self.board.get_result()
                if result == '1-0':
                    messagebox.showinfo("Game Over", "White wins!")
                elif result == '0-1':
                    messagebox.showinfo("Game Over", "Black wins!")
                elif result == '1/2-1/2':
                    messagebox.showinfo("Game Over", "Draw!")
            else:
                # Game still going — delay ponder so canvas renders first
                self.root.after(200, self._start_pondering)
        else:
            self.update_display()

    # ── Pondering ─────────────────────────────────────────────

    def _start_pondering(self):
        """Start pondering: predict opponent's move and pre-search our response."""
        if self.board.is_game_over() or self.game_timed_out:
            return

        bb = self.board.board
        tt_key = bb._transposition_key()
        entry = self.ai.tt.get(tt_key)
        if not entry or not entry[3]:
            return

        ponder_move = entry[3]
        # Validate it's legal in current position
        if ponder_move not in bb.legal_moves:
            return

        self._ponder_move = ponder_move
        logger.info("Pondering: predicting opponent plays %s", ponder_move.uci())

        # Deep-copy board and apply predicted opponent move
        ponder_board = copy.deepcopy(bb)
        ponder_board.push(ponder_move)

        # Wrapper matching ChessBoard interface for get_best_move
        wrapper = type('_PonderBoard', (), {'board': ponder_board})()

        def _ponder_run():
            # Set generous time limit (will be interrupted when opponent moves)
            old_tl = self.ai.time_limit
            old_tl_cached = self.ai._tl
            self.ai.time_limit = 300.0
            self.ai._tl = 300.0
            try:
                self.ai.get_best_move(wrapper)
            except Exception as e:
                logger.debug("Ponder search ended: %s", e)
            finally:
                self.ai.time_limit = old_tl
                self.ai._tl = old_tl_cached

        self._ponder_thread = threading.Thread(target=_ponder_run, daemon=True)
        self._ponder_thread.start()

    def _stop_pondering(self, actual_move=None):
        """Stop pondering and report hit/miss."""
        if self._ponder_thread is None:
            return

        was_pondering = self._ponder_thread.is_alive()
        if was_pondering:
            self.ai.time_up = True
            self._ponder_thread.join(timeout=1.0)

        if was_pondering and actual_move and self._ponder_move:
            if actual_move == self._ponder_move:
                logger.info("Ponder HIT! (%s) — TT is warm", actual_move.uci())
            else:
                logger.info("Ponder miss (%s != %s)", actual_move.uci(), self._ponder_move.uci())

        self._ponder_thread = None
        self._ponder_move = None

    def update_eval_bar(self, score_pawns: float):
        """Draw eval bar. score_pawns is from white's perspective."""
        self.eval_bar_canvas.delete("all")
        h = self.square_size * 8
        w = 30
        # Clamp score to [-6, +6] pawns
        clamped = max(-6.0, min(6.0, score_pawns))
        # White's portion (bottom): 50% at 0, more at positive score
        white_frac = 0.5 + clamped / 12.0
        white_h = int(h * white_frac)
        black_h = h - white_h
        # Draw black portion (top)
        self.eval_bar_canvas.create_rectangle(0, 0, w, black_h, fill='#1a1a1a', outline='')
        # Draw white portion (bottom)
        self.eval_bar_canvas.create_rectangle(0, black_h, w, h, fill='#f0f0f0', outline='')
        # Score text
        if abs(score_pawns) < 100:
            txt = f"{abs(score_pawns):.1f}"
        else:
            txt = "M"
        # Put text on the winning side
        if clamped >= 0:
            ty = black_h + 12
            tc = '#555'
        else:
            ty = black_h - 12
            tc = '#ccc'
        self.eval_bar_canvas.create_text(w // 2, ty, text=txt, font=('Arial', 8), fill=tc)

    def new_game(self):
        """Start a new game."""
        if messagebox.askyesno("New Game", "Start a new game?"):
            # Stop pondering first
            self._stop_pondering()
            if self.is_ai_thinking:
                # Signal AI to stop
                self.ai.time_up = True
                if self._ai_thread is not None:
                    self._ai_thread.join(timeout=3)
                self._ai_thread = None
                self.is_ai_thinking = False
            # Stop and reset timer
            self._stop_timer()
            self.white_time = self.time_limit
            self.black_time = self.time_limit
            self.game_timed_out = False
            self._clock_color = chess.WHITE

            self._game_gen += 1  # invalidate stale poll callbacks
            self._ai_result = [None]
            self.board = ChessBoard()
            self.selected_square = None
            self.legal_moves = []
            self.last_move = None
            self.is_ai_thinking = False
            # Clear AI transposition table for the new game
            self.ai.transposition_table.clear()
            # Reset canvas dirty-check state (use sentinel to force full redraw)
            self._prev_highlights = [[_DIRTY] * 8 for _ in range(8)]
            self._prev_pieces = [[_DIRTY] * 8 for _ in range(8)]
            # Reset move history cache and clear the text widget
            self._hist_len = 0
            self._hist_san = []
            self._hist_board = chess.Board()
            self.history_text.config(state=tk.NORMAL)
            self.history_text.delete(1.0, tk.END)
            self.history_text.config(state=tk.DISABLED)
            # Reset cached timer state
            self._last_white_text = None
            self._last_black_text = None
            self._last_white_bold = None
            self._last_black_bold = None
            self._last_white_color = None
            self._last_black_color = None
            self._update_timer_display()
            self.update_eval_bar(0.0)
            self.update_display()

            if self.ai_is_white:
                self.root.after(500, self.make_ai_move)

    def undo_move(self):
        """Undo the last move."""
        self._stop_pondering()
        if self.is_ai_thinking:
            messagebox.showinfo("Undo", "Please wait for AI to finish thinking.")
            return

        if len(self.board.board.move_stack) == 0:
            messagebox.showinfo("Undo", "No moves to undo")
            return

        if len(self.board.board.move_stack) >= 2:
            self.board.board.pop()
            self.board.board.pop()
        else:
            self.board.board.pop()

        self.selected_square = None
        self.legal_moves = []
        # Update last_move to the new last move (or None)
        if self.board.board.move_stack:
            self.last_move = self.board.board.move_stack[-1]
        else:
            self.last_move = None
        self.update_display()

    def show_promotion_dialog(self) -> Optional[chess.PieceType]:
        """Show promotion piece selection dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Promote Pawn")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

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
            (chess.QUEEN, '\u2655', 'Queen'),
            (chess.ROOK, '\u2656', 'Rook'),
            (chess.BISHOP, '\u2657', 'Bishop'),
            (chess.KNIGHT, '\u2658', 'Knight')
        ]

        for piece_type, symbol, name in pieces:
            btn = tk.Button(
                button_frame, text=f"{symbol}\n{name}",
                width=8, height=3, font=('Arial', 24),
                command=lambda pt=piece_type: self._set_promotion(result, pt, dialog)
            )
            btn.pack(side=tk.LEFT, padx=8)

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

        tk.Label(main_frame, text="Search depth:", font=('Arial', 20),
                 bg='white', fg=self.TEXT_COLOR).grid(row=2, column=0, sticky=tk.W, pady=8)
        depth_var = tk.IntVar(value=self.ai.depth)
        depth_spin = tk.Spinbox(main_frame, from_=1, to=10, textvariable=depth_var,
                                width=6, font=('Arial', 20))
        depth_spin.grid(row=2, column=1, sticky=tk.W, pady=8, padx=(16, 0))

        tk.Label(main_frame, text="Play style:", font=('Arial', 20),
                 bg='white', fg=self.TEXT_COLOR).grid(row=3, column=0, sticky=tk.W, pady=8)
        style_labels = {
            'normal': 'Normal',
            'aggressive': 'Aggressive',
            'defensive': 'Defensive',
            'random': 'Random',
        }
        style_var = tk.StringVar(value=getattr(self.ai, 'play_style', STYLE_RANDOM))
        style_menu = tk.OptionMenu(main_frame, style_var,
                                   *[s for s in PLAY_STYLES])
        style_menu.config(font=('Arial', 18), width=12)
        style_menu.grid(row=3, column=1, sticky=tk.W, pady=8, padx=(16, 0))

        tk.Label(main_frame, text="Classical eval weight:", font=('Arial', 20),
                 bg='white', fg=self.TEXT_COLOR).grid(row=4, column=0, sticky=tk.W, pady=(16, 8))
        weight_var = tk.DoubleVar(value=self.classical_weight)
        weight_scale = tk.Scale(main_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                variable=weight_var, resolution=0.01, length=300,
                                font=('Arial', 16), bg='white', highlightthickness=0,
                                troughcolor='#ddd', sliderrelief=tk.FLAT)
        weight_scale.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=(16, 8), padx=(16, 0))

        weight_label = tk.Label(main_frame, text=f"{self.classical_weight:.2f}",
                                font=('Arial', 18), bg='white', fg=self.LABEL_COLOR)
        weight_label.grid(row=5, column=1, sticky=tk.W, pady=(0, 10), padx=(16, 0))

        def on_weight_change(*_):
            weight_label.config(text=f"{weight_var.get():.2f}")

        weight_var.trace_add("write", on_weight_change)

        button_frame = tk.Frame(main_frame, bg='white')
        button_frame.grid(row=6, column=0, columnspan=2, pady=(24, 0))

        def apply_settings():
            try:
                new_depth = int(depth_var.get())
                new_depth = max(1, min(10, new_depth))
                self.ai.depth = new_depth
                new_weight = max(0.0, min(1.0, float(weight_var.get())))
                self.classical_weight = new_weight
                self.ai.classical_weight = new_weight
                new_style = style_var.get()
                if new_style in PLAY_STYLES:
                    self.ai.play_style = new_style
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
        if not self.timer_running or self.game_timed_out:
            return

        now = time.time()
        elapsed = now - self.last_tick_time
        self.last_tick_time = now

        if self._clock_color == chess.WHITE:
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

        is_white_turn = self._clock_color == chess.WHITE
        white_low = self.white_time < 30
        black_low = self.black_time < 30

        white_bold = is_white_turn
        black_bold = not is_white_turn
        white_color = '#e74c3c' if white_low else (self.TEXT_COLOR if is_white_turn else self.LABEL_COLOR)
        black_color = '#e74c3c' if black_low else (self.TEXT_COLOR if not is_white_turn else self.LABEL_COLOR)

        if white_str != self._last_white_text:
            self.white_timer_label.config(text=white_str)
            self._last_white_text = white_str
        if black_str != self._last_black_text:
            self.black_timer_label.config(text=black_str)
            self._last_black_text = black_str

        if white_bold != self._last_white_bold:
            self.white_timer_label.config(
                font=self._timer_font_bold if white_bold else self._timer_font_normal
            )
            self._last_white_bold = white_bold
        if black_bold != self._last_black_bold:
            self.black_timer_label.config(
                font=self._timer_font_bold if black_bold else self._timer_font_normal
            )
            self._last_black_bold = black_bold
        if white_color != self._last_white_color:
            self.white_timer_label.config(fg=white_color)
            self._last_white_color = white_color
        if black_color != self._last_black_color:
            self.black_timer_label.config(fg=black_color)
            self._last_black_color = black_color

    def _handle_timeout(self, color):
        """Handle when a player runs out of time."""
        self.timer_running = False
        self.game_timed_out = True
        if self.timer_after_id is not None:
            self.root.after_cancel(self.timer_after_id)
            self.timer_after_id = None

        _sound_game_end()
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

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )

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

    app = ChessGUI(
        ai_color=args.ai_color,
        ai_depth=args.depth,
        model_path=args.model if args.model else None,
        classical_weight=args.classical_weight
    )
    app.run()


if __name__ == "__main__":
    main()
