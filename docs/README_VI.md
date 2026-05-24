# Dự Án Machine Learning Cho Cờ Vua

Dự án machine learning để học và chơi cờ vua sử dụng neural network. Dự án này sử dụng mạng neural network dạng CNN để đánh giá vị trí cờ và sử dụng thuật toán minimax với alpha-beta pruning để chọn nước đi.

## Tính Năng

- **Đánh Giá Bằng Neural Network**: Model CNN đánh giá vị trí cờ
- **AI Minimax**: AI sử dụng thuật toán minimax với alpha-beta pruning
- **Chơi Game Tương Tác**: Chơi cờ với AI đã được train
- **Pipeline Training**: Tải và parse dữ liệu từ PGN files, train model
- **Biểu Diễn Bàn Cờ**: Tensor 8x8x12 hiệu quả cho vị trí cờ

## Cài Đặt

1. **Cài đặt dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Hướng Dẫn Sử Dụng

### Bước 1: Tải Dữ Liệu Cờ Vua

Tải các file PGN từ pgnmentor.com:

```bash
python scripts/download_pgn_data.py
```

Script này sẽ tải các ván cờ của các kỳ thủ hàng đầu (Carlsen, Caruana, Anand, Kasparov, v.v.) và lưu vào thư mục `data/pgn/`.

### Bước 2: Parse Dữ Liệu Thành Training Data

Chuyển đổi các file PGN thành dữ liệu training:

```bash
python parse_pgn_data.py
```

Hoặc với các tùy chọn:

```bash
# Parse với giới hạn số game mỗi file
python parse_pgn_data.py --max-games 1000

# Chỉ định thư mục PGN và file output
python parse_pgn_data.py --pgn-dir data/pgn --output data/training_data.npz
```

Dữ liệu sẽ được lưu vào `data/training_data.npz`.

### Bước 3: Train Model

Train neural network với dữ liệu đã tải:

```bash
python train.py
```

Hoặc với các tùy chọn:

```bash
# Chỉ định file dữ liệu
python train.py --data data/training_data.npz

# Điều chỉnh số epochs và batch size
python train.py --epochs 30 --batch-size 64 --lr 0.0005

# Model sẽ tự động train trên CPU
```

**Lưu ý**: Model mặc định sẽ train trên CPU. Nếu bạn muốn dùng GPU, bỏ flag `--use-cpu` (mặc định là True).

### Bước 4: Chơi Cờ Với AI

Chơi cờ với AI đã được train:

```bash
python main.py play
```

Hoặc:

```bash
python play_chess.py
```

**Tùy chọn**:
- `--ai-color`: Màu AI chơi (`white` hoặc `black`, mặc định: `black`)
- `--depth`: Độ sâu tìm kiếm của AI (mặc định: 2, cao hơn = mạnh hơn nhưng chậm hơn)
- `--model`: Đường dẫn đến model đã train (mặc định: `models/chess_model.pth`)

**Ví dụ**:
```bash
# AI chơi trắng
python main.py play --ai-color white

# AI với độ sâu cao hơn
python main.py play --depth 3

# Dùng model tùy chỉnh
python main.py play --model models/my_model.pth --depth 2
```

### Điều Khiển Game

- **Nhập nước đi**: Nhập nước đi theo UCI notation (ví dụ: `e2e4`, `g1f3`)
- **Undo nước đi**: Gõ `undo` để hoàn tác nước đi cuối
- **Thoát game**: Gõ `quit` để thoát

## Quy Trình Hoàn Chỉnh

```bash
# 1. Tải dữ liệu
python download_pgn_data.py

# 2. Parse dữ liệu
python parse_pgn_data.py

# 3. Train model
python train.py

# 4. Chơi cờ
python main.py play
```

## Cấu Trúc Dự Án

```
.
├── main.py                  # Entry point chính
├── chess_board.py          # Logic bàn cờ và game
├── chess_model.py          # Neural network models
├── train.py                # Script training
├── play_chess.py           # Giao diện chơi game
├── download_pgn_data.py    # Tải PGN files từ pgnmentor.com
├── parse_pgn_data.py       # Parse PGN files thành training data
├── requirements.txt        # Dependencies
├── README.md              # Tài liệu tiếng Anh
└── README_VI.md           # Tài liệu tiếng Việt (file này)
```

## Cách Hoạt Động

### Biểu Diễn Bàn Cờ

Bàn cờ được biểu diễn dưới dạng tensor 8×8×12:
- 8×8 cho các ô trên bàn cờ
- 12 channels cho các loại quân và màu:
  - Trắng: Tốt, Xe, Mã, Tượng, Hậu, Vua
  - Đen: Tốt, Xe, Mã, Tượng, Hậu, Vua

### Kiến Trúc Neural Network

Model sử dụng CNN với:
- 3 lớp convolutional với batch normalization
- Các lớp fully connected để đánh giá
- Output: Một giá trị đại diện cho đánh giá vị trí (từ góc nhìn người chơi hiện tại)

### Quyết Định Của AI

AI sử dụng thuật toán **minimax** với **alpha-beta pruning**:
1. Đánh giá vị trí bằng neural network
2. Tìm kiếm trước đến độ sâu chỉ định
3. Chọn nước đi tối ưu dựa trên đánh giá

## Training Model

Quá trình training:
1. Tải các file PGN từ pgnmentor.com (các ván cờ của kỳ thủ hàng đầu)
2. Parse các ván cờ để trích xuất vị trí
3. Đánh giá vị trí bằng heuristic đơn giản (giá trị quân)
4. Train CNN để dự đoán các đánh giá này
5. Validate trên tập test

**Lưu ý**: Để có hiệu suất tốt hơn, bạn có thể:
- Train trên nhiều vị trí hơn
- Sử dụng hàm đánh giá mạnh hơn
- Implement reinforcement learning (self-play)
- Sử dụng pre-trained models hoặc game databases

## Tùy Chỉnh

### Điều Chỉnh Tham Số Training

Chỉnh sửa `train.py` hoặc dùng command-line arguments:
- Số lượng vị trí training (`--num-positions`)
- Số epochs (`--epochs`)
- Learning rate (`--lr`)
- Batch size (`--batch-size`)
- Kiến trúc model

### Cải Thiện AI

- Tăng độ sâu tìm kiếm (chậm hơn nhưng mạnh hơn)
- Train trên nhiều dữ liệu hơn
- Sử dụng hàm đánh giá mạnh hơn
- Implement opening book hoặc endgame tablebase
- Thêm move ordering heuristics

## Yêu Cầu

- Python 3.8+
- PyTorch 2.0+
- NumPy 1.24+
- python-chess 3.1+

## Cải Tiến Trong Tương Lai

- [ ] Reinforcement learning với self-play
- [ ] Policy network cho việc chọn nước đi
- [ ] Tích hợp opening book
- [ ] Hỗ trợ endgame tablebase
- [ ] UCI engine interface
- [ ] Giao diện web để chơi
- [ ] Training trên real game databases

## License

Dự án này là mã nguồn mở và có sẵn cho mục đích giáo dục.

## Đóng Góp

Hãy tự do fork, chỉnh sửa và cải thiện dự án này!
