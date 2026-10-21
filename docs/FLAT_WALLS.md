# Tường phẳng và map 2D

Chạy tại gốc project:

```powershell
python scripts/run_simulation.py --scenario configs/scenarios/map.json
```

Viewer bật mặc định; `--headless` tắt viewer. Preset bật nhiễu all và giả định tau/delay 10 ms. PID+FF giữ Z bằng dòng, robot dẫn XY, capsule không có PID XY/góc.

## Sửa map

- `obstacles_x_y_radius_mm`: `[x, y, radius]`, cột tròn vật lý.
- `walls_cx_cy_width_height_mm`: `[center_x, center_y, size_x, size_y]`, tường hộp thật; các kích thước là toàn chiều rộng/dài theo XY, KHÔNG phải chiều cao Z.
- `x0_mm`, `y0_mm`: điểm đầu của capsule trong frame map; robot giải IK để đặt DEMA phía trên điểm này trước khi chạy physics.
- `goal_xy_mm`: đích.
- Frame map giữ cố định tại pose DEMA baseline, không cập nhật theo robot sau IK.
- Chiều cao tường được dựng từ đáy dầu tới dưới mặt dầu 5 mm, như cột. Tường chỉ song song trục map X/Y trong bản này.

Đã bỏ hoàn toàn cách sinh tường từ chuỗi trụ. Không dùng key cũ `walls_xyxy_radius_mm`; đổi sang key mới.

Planner tính khoảng cách chính xác giữa đoạn đường và chữ nhật, so với bán kính bao capsule + margin. Vùng cấm có góc bo tròn do mở rộng hình học; tường vật lý vẫn phẳng. Giám sát capsule và đồ thị dùng cùng các hình chữ nhật.

## Các file code đã sửa

Dưới `src/endoscopy_capsule_control/`:

- `planning/collision.py`: khoảng cách đoạn/chữ nhật, signed distance, đường bao để vẽ.
- `planning/astar.py`: hỗ trợ đồng thời cột/tường; clearance và metadata.
- `planning/environment.py`: đọc tường từ JSON.
- `simulation/scene_builder.py`: dựng tường bằng MuJoCo box.
- `simulation/path_simulation.py`: khởi tạo robot theo điểm đầu và kiểm tra khoảng hở ban đầu.
- `evaluation/results.py`: vẽ tường và vùng cấm.

`tests/test_planning_walls.py` kiểm tra đi thẳng, đi vòng, giao cắt, tiếp xúc và điểm đầu không hợp lệ.

IK khởi tạo là đặt điều kiện ban đầu trong mô phỏng, không phải quỹ đạo robot thật đi từ home tới start. Plant/bể và trễ vẫn chưa hiệu chỉnh theo phần cứng lab.
