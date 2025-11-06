1 Cài đặt
Cài đặt Python và thư viện cần thiết
pip install torch numpy matplotlib pygame

(Tùy chọn) Kiểm tra GPU
import torch
print(torch.cuda.is_available())


Nếu in ra True, chương trình sẽ tự động dùng GPU để tăng tốc huấn luyện.

2 Cách chạy
Huấn luyện mô hình (không giao diện — nhanh hơn)

Trong file agent.py, chọn dòng:

from game_no_ui import Game
# from game import Game


Sau đó bật huấn luyện:

if __name__ == "__main__":
    train()   # bật train mode
    # test()  # tắt test


Chạy lệnh:

python agent.py


Trong quá trình huấn luyện:

Mô hình sẽ tự động lưu sau mỗi vòng (model/model.pth).

Bạn có thể dừng bằng Ctrl + C, chương trình sẽ tự lưu lại model hiện tại.

Biểu đồ động hiển thị Curr Score và Avg Score theo thời gian.

Kiểm tra mô hình (hiển thị UI pygame)

Sau khi đã có model (model.pth), bật UI:

from game import Game
# from game_no_ui import Game


Và chọn:

if __name__ == "__main__":
    # train()
    test(200)  # chạy test với giao diện


Chạy:

python agent.py


Cửa sổ trò chơi Snake sẽ hiển thị, rắn tự động di chuyển theo chiến lược mà model đã học.
Đồng thời, bạn sẽ thấy biểu đồ động hiển thị điểm và trung bình 50 tập gần nhất.
