# Kiểm tra bản tích hợp follower + A*

Trước khi đóng gói, đã chạy A* đầy đủ ở chế độ headless với preset `one_column.json` (khoảng 54.61 s mô phỏng).

| Trường hợp | Đạt đích XY | XY RMS (mm) | Z RMS trong frame cố định (mm) | Khoảng hở nhỏ nhất của envelope (mm) |
|---|---|---|---|---|
| Không nhiễu, tau/delay = 0 | Có | 0.13515 | 0.02282 | 2.12026 |
| Noise all, seed 42, tau/delay = 10 ms | Có | 0.14602 | 1.02542 | 2.13483 |

Cả hai không vi phạm margin 2 mm. Số liệu đầy đủ ở `docs/validation_results.json`.

Đã kiểm tra từ chối đích trong vật cản, bản đồ bị chặn, giao đoạn thẳng với vùng cấm, giới hạn tốc độ/gia tốc. Follower cũ sau tái cấu trúc cho CSV giống từng phần tử trong các ca đối chiếu.

Viewer tương tác Windows chưa được kiểm tra trên máy lab; các ca trên chạy headless. Tiêu chí success hiện xét XY và margin; chất lượng Z báo cáo riêng. Plant, bể, nhiễu và trễ chưa được hiệu chỉnh bằng số đo lab; 10 ms là giả định thử độ nhạy.

Khi đóng gói: đã kiểm tra cú pháp toàn bộ Python và tính toàn vẹn ZIP. Không chạy lại dynamics ở môi trường đóng gói vì Python tại đây thiếu MuJoCo; bảng trên là kết quả kiểm tra tích hợp đã lưu trước đó.

## Tường box: kiểm tra mới

MuJoCo 3.11.0: chạy đủ 361.25 giây cả hai ca không nhiễu và noise all seed42 + tau/delay 10 ms. Kết quả tại `docs/flat_wall_validation.json`, đồ thị `docs/flat_wall_clean.png` và `docs/flat_wall_noise.png`. Bộ 4 test hình học đã qua, map chỉ cột vẫn cho đường và thời gian giống bản cũ. Viewer tương tác Windows chưa kiểm tra.
