# DEMA capsule simulation

Project đầy đủ: follower và A* XY tránh cột trụ trong MuJoCo. Điểm chạy chính là `scripts/run_simulation.py`.

## Chạy nhanh

Từ thư mục gốc có `src`, `scripts`, `pyproject.toml`:

```powershell
python -m pip install -e .
python scripts/run_simulation.py --scenario configs/scenarios/follower_circle.json
```

Viewer mở mặc định. Đóng viewer sau khi kết quả đã lưu. Preset bật PNG; kết quả mặc định ở `results/follower_circle.csv`, `.json`, `.png`.

```powershell
python scripts/run_simulation.py --scenario configs/scenarios/follower_circle.json --noise all --seed 42 --current-tau-ms 10 --localization-delay-ms 10 --csv results/follower_noise.csv
```

Tau/trễ 10 ms là giả định sensitivity test, không phải số đo nguồn thật. `--noise none` không tắt tau/trễ. Dùng `--headless` để tắt viewer. Lệnh cũ `python scripts/test_passive_follower.py ...` vẫn được hỗ trợ bằng wrapper, không chứa bản controller thứ hai.

## Đọc code theo thứ tự

1. `scripts/run_simulation.py`: điểm vào.
2. `simulation/follower_options.py`: đọc tham số, preset, kiểm tra đầu vào.
3. `simulation/path_simulation.py`: khởi tạo và vòng lặp chính.
4. `planning/trajectory.py`: tham chiếu đường thẳng/vòng tròn.
5. `control/common_current_z.py`: điều khiển dòng giữ Z.
6. `plant/z_hover_plant.py`: thực thi dynamics và mô hình đo.
7. `evaluation/results.py`: ghi CSV/JSON, vẽ đồ thị và in chỉ số.

Từ mục 2 trở đi, đường dẫn nằm dưới `src/endoscopy_capsule_control/`.

Xem `docs/FILE_FLOW.md` để biết luồng và cần sửa file nào.

Đã ghép A* XY và cột trụ vào cùng luồng chạy; đã hỗ trợ tường chữ nhật. Xem `docs/PATH_PLANNING.md`. Plant, gains, axes, luật dòng, cubic trajectory line/circle và thứ tự lấy mẫu giữ nguyên. Robot dẫn hướng XY; capsule không có PID XY/góc. PID+FF dùng dòng giữ Z trong LCS DEMA. Chuyển động thật vẫn 3D. Simulink không nằm trong thay đổi này.

## Path planning XY

```powershell
python scripts/run_simulation.py --scenario configs/scenarios/one_column.json --noise none
```

Bản đồ và vật cản chỉ khai báo một lần trong scenario. Viewer và PNG bật trong preset. Các lệnh circle/line cũ vẫn chạy như trước.

## Nội dung gói ZIP

Gói chứa toàn bộ source, model/mesh AUBO i10, cấu hình, tests và tài liệu. Đã bỏ `scripts/legacy/`, cache và kết quả chạy cũ. Giải nén vào thư mục riêng, không chép đè lên project cũ.

`scripts/test_passive_follower.py` là wrapper tương thích dùng chung code chính, cũng được sweep sử dụng. Các module controller/plant cũ còn được package/tests tham chiếu vẫn được giữ để tránh lỗi import. Không cần chạy installer hay apply-update.

Đọc `docs/FILE_FLOW.md` để chọn đúng file cần sửa; `docs/PATH_PLANNING.md` hướng dẫn đổi đích, vật cản và tốc độ. `VALIDATION.md` ghi các kiểm tra đã thực hiện.

## Tường phẳng

Dùng `configs/scenarios/map.json`. Xem `docs/FLAT_WALLS.md` để sửa cột, tường, điểm đầu và đích. Tường dùng box thật; ảnh xem trước ở `docs/flat_wall_map.png`.
