# DEMA Capsule Control

Mô phỏng capsule từ tính trong MuJoCo với hệ DEMA gắn trên robot AUBO i10.

Project gồm ba bài toán:
- **XY motion + Z hovering:** robot dẫn hướng XY, PID+FF điều chỉnh dòng để giữ Z.
- **Điều khiển góc:** thay đổi dòng hai cuộn hoặc kết hợp robot và dòng.
- **Path planning:** A* tìm đường XY tránh vật cản, sau đó tạo trajectory để robot dẫn capsule đi theo.

Capsule chuyển động và quay tự do trong mô phỏng. Plant hiện chưa được điều chỉnh đầy đủ giống phần cứng lab 100%.

## Cấu trúc chính

```text
endoscopy_capsule_control/
├── configs/scenarios/
│   ├── follower_circle.json       # Chuyển động tròn
│   ├── one_column.json            # Map một cột
│   └── map.json                   # Map gồm cột và tường
├── scripts/
│   ├── run_simulation.py          # Hover, follower và path planning
│   ├── test_current_angle.py      # Góc bằng dòng, robot cố định
│   ├── test_coordinated_angle.py  # Góc phối hợp robot + dòng
│   ├── test_dema_tilt.py          # Thử quay DEMA
│   ├── test_passive_follower.py   # Lệnh tương thích với follower
│   └── run_follower_sweep.py      # Chạy nhiều seed/tham số
├── src/endoscopy_capsule_control/
│   ├── config.py                 # Thông số plant, controller và nhiễu
│   ├── control/                  # Controller và phân bổ dòng
│   ├── magnetic/                 # Từ trường, lực và mô-men từ
│   ├── plant/                    # Nguồn dòng, định vị, chất lỏng
│   ├── planning/                 # A*, collision và trajectory
│   ├── simulation/               # MuJoCo, robot, viewer và safety
│   ├── evaluation/               # Metrics và đồ thị
│   └── models/                   # XML và mesh AUBO i10
├── tests/
├── docs/
├── pyproject.toml
└── README.md
```

## Cài đặt

Yêu cầu **Python ≥ 3.10** và **MuJoCo 3.11.0**.

Chạy tại thư mục chứa `pyproject.toml`:

```bash
python -m pip install mujoco==3.11.0 numpy scipy matplotlib
python -m pip install -e .
```

Kiểm tra MuJoCo đã cài:

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

Kết quả mong đợi: `3.11.0`.

Chạy thử mô phỏng và mở MuJoCo viewer:

```bash
python scripts/run_simulation.py --mode hold --time 10 --noise none --camera tank
```

Viewer bật mặc định. Dùng `--headless` nếu chỉ muốn tính toán mà không mở cửa sổ.

## 1. XY motion + Z hovering

Giữ robot tại chỗ, kiểm tra capsule giữ độ cao:

```bash
python scripts/run_simulation.py --mode hold --x0-mm 1 --time 10 --noise all --seed 42 --plot --csv results/hover.csv
```

Robot di chuyển thẳng theo X:

```bash
python scripts/run_simulation.py --mode follow --axis x --distance-mm 5 --move-time 5 --time 10 --x0-mm 0 --noise all --seed 42 --plot --csv results/follow_line.csv
```

Chuyển động tròn:

```bash
python scripts/run_simulation.py --scenario configs/scenarios/follower_circle.json --noise all --seed 42
```

Trong luồng này, PID+FF giữ Z bằng dòng chung `I1 = I2`. Capsule theo robot trong XY nhờ lực từ; chưa có PID riêng phản hồi sai số XY của capsule.

## 2. Điều khiển góc

Thay đổi dòng hai cuộn, robot giữ pose:

```bash
python scripts/test_current_angle.py --angle-deg 3 --ramp-s 20 --time 35 --noise all --seed 42 --plot --csv results/current_angle.csv
```

Phối hợp robot và dòng:

```bash
python scripts/test_coordinated_angle.py --angle-deg 3 --ramp-s 20 --time 35 --noise all --seed 42 --plot --csv results/coordinated_angle.csv
```

Góc được đo trong mặt XZ của DEMA ban đầu: 0° dọc +Z, góc dương nghiêng về +X.

Điều khiển chỉ bằng dòng có thể đạt góc và giữ Z nhưng làm capsule lệch XY. Cần đánh giá cả góc, độ cao và vị trí tương đối với DEMA. Luồng phối hợp mặc định điều chỉnh robot XY; đây vẫn là bộ điều khiển thử nghiệm.

## 3. Path planning

Map đơn giản:

```bash
python scripts/run_simulation.py --scenario configs/scenarios/one_column.json
```

Map có cột và tường:

```bash
python scripts/run_simulation.py --scenario configs/scenarios/map.json
```

Đổi đích và tốc độ:

```bash
python scripts/run_simulation.py --scenario configs/scenarios/one_column.json --target-mm 50 20 --speed-mm-s 2 --csv results/path_custom.csv
```

Luồng chạy:

**Map → A* → rút gọn đường → trajectory → robot dẫn XY + dòng giữ Z → capsule.**

Map được cung cấp trước. Planner làm việc trong XY, không tìm đường qua trên/dưới vật cản. Thời gian chạy path được tính tự động.

### Sửa map

Mở `configs/scenarios/map.json`:

| Tham số | Ý nghĩa |
|---|---|
| `x0_mm`, `y0_mm` | Điểm đầu |
| `path.goal_xy_mm` | Điểm đích `[x, y]` |
| `path.obstacles_x_y_radius_mm` | Cột `[x, y, radius]` |
| `path.walls_cx_cy_width_height_mm` | Tường `[x, y, size_x, size_y]` |
| `path.grid_mm` | Bước lưới A* |
| `path.margin_mm` | Khoảng dự phòng va chạm |
| `path.speed_mm_s` | Tốc độ trajectory |
| `path.acceleration_mm_s2` | Gia tốc trajectory |

Đơn vị hình học trong map là mm. Capsule dài 30 mm, đường kính 11.4 mm. Planner coi tâm capsule là điểm và mở rộng vật cản thêm bán kính bao 15 mm cộng margin; kiểm tra cả đoạn nối giữa các node.

## Tham số thường dùng

| Flag | Chức năng |
|---|---|
| `--noise none/current/localization/robot/all` | Chọn nguồn nhiễu |
| `--seed 42` | Seed tái lập thí nghiệm |
| `--current-tau-ms 10` | Time constant nguồn dòng |
| `--localization-delay-ms 10` | Trễ định vị, bội số 10 ms |
| `--camera tank` | Camera nhìn bể |
| `--headless` | Không mở viewer |
| `--plot` | Lưu đồ thị PNG |
| `--csv results/example.csv` | Đường dẫn kết quả |

Viewer bật mặc định. Sau khi mô phỏng kết thúc và lưu kết quả, đóng viewer để thoát.

`--noise none` không tự tắt trễ. Để chạy baseline không nhiễu và không trễ, thêm:

```bash
--noise none --current-tau-ms 0 --localization-delay-ms 0
```

Tau/trễ là tham số mô phỏng, chưa phải số đo nguồn dòng và định vị thực tế.

## Kết quả

Các luồng chính lưu:

```text
results/
├── example.csv     # Time series
├── example.json    # Trạng thái và metrics
└── example.png     # Đồ thị khi bật --plot
```

- Hover/follower: sai số Z, sai số bám đường, alignment và dòng điện.
- Angle: sai số góc, độ cao, drift XY và khả thi phân bổ.
- Path: sai số tới đích, clearance và `path_metrics.success`.

Chạy hết thời gian không tự đồng nghĩa đạt yêu cầu. Với path, chất lượng Z được báo cáo riêng với điều kiện thành công XY.

## Kiểm tra

```bash
python -m pip install pytest
python -m pytest tests -q
```

Xem thêm `docs/FILE_FLOW.md`, `docs/PATH_PLANNING.md` và `docs/FLAT_WALLS.md`.
