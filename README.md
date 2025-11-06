#  Snake AI - Deep Q-Learning

Dự án này huấn luyện một agent AI tự chơi trò **Rắn săn mồi (Snake Game)** bằng thuật toán **Deep Q-Learning (DQN)** sử dụng thư viện **PyTorch**.

---

##  1. Cài đặt

Cài đặt Python và các thư viện cần thiết:

```bash
pip install torch numpy matplotlib pygame
```

### (Tùy chọn) Kiểm tra GPU

```python
import torch
print(torch.cuda.is_available())
```

Nếu in ra `True`, chương trình sẽ tự động sử dụng GPU để tăng tốc huấn luyện.

---

##  2. Cách chạy

### Huấn luyện mô hình (không giao diện – nhanh hơn)

Mở file `agent.py`, chỉnh:

```python
from game_no_ui import Game
# from game import Game
```

Sau đó bật huấn luyện:

```python
if __name__ == "__main__":
    train()   # bật train mode
    # test()  # tắt test
```

Chạy lệnh:

```bash
python agent.py
```

Trong quá trình huấn luyện:
- Mô hình tự động lưu sau mỗi vòng (`model/model.pth`).
- Có thể dừng bằng `Ctrl + C`, chương trình sẽ tự lưu model hiện tại.
- Biểu đồ động hiển thị `Curr Score` và `Avg Score` theo thời gian.

---

### Kiểm tra mô hình (hiển thị giao diện pygame)

Sau khi đã có `model/model.pth`, bật giao diện:

```python
from game import Game
# from game_no_ui import Game
```

Và chỉnh:

```python
if __name__ == "__main__":
    # train()
    test(200)  # chạy test với giao diện
```

Chạy lệnh:

```bash
python agent.py
```

Cửa sổ trò chơi Snake sẽ hiển thị.  
Rắn tự động di chuyển theo chiến lược đã học, đồng thời biểu đồ động hiển thị điểm và trung bình 50 tập gần nhất.

---
