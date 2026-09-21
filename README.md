Project mô phỏng điều khiển viên nang nội soi từ tính bằng hệ DEMA (Dual-Electromagnet Actuation) gắn trên robot AUBO i10, sử dụng MuJoCo.

Hiện tại project tập trung vào plant và điều khiển Z-hovering. Các phần XY/XYZ sẽ được phát triển lại sau trên nền plant hiện tại.

1. Mục tiêu hiện tại
- Mô phỏng capsule trong môi trường chất lỏng
- Tính lực và mô-men từ do hai electromagnet tạo ra.
- Điều khiển capsule giữ độ cao theo trục Z.
- Mô phỏng các sai số thực tế:
+ nhiễu dòng điện của bộ nguồn,
+ nhiễu localization,
+ sai số vị trí của AUBO i10.

Cho phép bật/tắt từng loại nhiễu để test controller.
Luồng chính:
Z target
   ↓
Z controller
   ↓
Current command
   ↓
Power supply model + current noise
   ↓
Magnetic force
   ↓
Capsule dynamics + fluid + gravity
   ↓
True state
   ↓
Localization + noise
   ↓
Measured state → controller

2. Cài đặt
Yêu cầu khuyến nghị: Python 3.13, MuJoCo, NumPy
Tại thư mục project:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .

Kiểm tra cài đặt:

python -m unittest discover -s tests -v

Nếu hiện OK thì có thể chạy simulation.

3. Các lệnh test quan trọng
- Test các model nhiễu
python scripts/test_plant.py
Script kiểm tra:
+ current noise,
+ RF localization noise,
+ AUBO i10 pose uncertainty.

- Test lực từ theo dòng điện
python scripts/test_magnetic_z.py

Dùng để kiểm tra dấu và độ đối xứng của lực từ.

- Test Z plant open-loop
Một vài case tham khảo:

python scripts/test_z_plant_open_loop.py --noise none --time 0.05 --i1 -10 --i2 10
python scripts/test_z_plant_open_loop.py --noise none --time 0.05 --i1 -15 --i2 15
python scripts/test_z_plant_open_loop.py --noise none --time 0.05 --i1 -20 --i2 20

Tại operating point hiện tại:

-10, +10 A: lực nâng chưa đủ,

-15, +15 A: gần điểm cân bằng,

-20, +20 A: lực nâng lớn hơn trọng lực biểu kiến.

4. Chạy Z-hovering
Không có nhiễu: python scripts/run_z_hover.py --noise none --time 10 --settling-time 5
Có MuJoCo viewer: python scripts/run_z_hover.py --noise none --time 10 --settling-time 5 --viewer --camera overview

Camera khác:

--camera capsule

5. Bật / tắt nhiễu
Thay option --noise.
- Chỉ current noise:  python scripts/run_z_hover.py --noise current --seed 42 --viewer --camera overview
- Chỉ localization noise: python scripts/run_z_hover.py --noise localization --seed 42 --viewer --camera overview
- Chỉ sai số AUBO i10: python scripts/run_z_hover.py --noise robot --seed 42 --viewer --camera overview
- Bật tất cả: python scripts/run_z_hover.py --noise all --seed 42 --viewer --camera overview

Dùng cùng một --seed để có thể lặp lại cùng một experiment.

6. Các sai số đang mô phỏng
Current noise, Model đang dùng bộ nguồn tương đương Kepco BOP 20-20.
Baseline hiện tại:
RMS current noise ≈ 6 mA / channel
Hai kênh electromagnet có nhiễu độc lập.
Localization noise
Baseline được xây dựng từ kết quả dynamic localization của bài báo DEMA.
sigma ≈ 0.93 mm / axis
3D RMSE ≈ 1.61 mm
AUBO i10 uncertainty
Hiện mô phỏng dưới dạng pose bias cố định trong mỗi episode.
repeatability bound ≈ 0.05 mm

7. Thay đổi chất lỏng
Các thông số fluid nằm trong:
src/endoscopy_capsule_control/plant/fluid.py
Khi đổi fluid cần chú ý hai thông số chính: density rho, viscosity.
Density ảnh hưởng buoyancy, còn viscosity ảnh hưởng drag.
Sau khi đổi fluid nên chạy lại: python scripts/test_z_plant_open_loop.py --noise none --time 0.05 --i1 -15 --i2 15
Nếu capsule không còn cân bằng tại -15, +15 A thì phải cập nhật lại feedforward/equilibrium current trong config/controller.

8. Thay đổi mức nhiễu
Các tham số chính nằm trong:
src/endoscopy_capsule_control/config.py
Có thể chỉnh:current noise,localization sigma,AUBO repeatability, capsule mass/geometry, magnetic moment, controller gains, control timestep.
Sau khi thay config nên chạy lại theo thứ tự:

python -m unittest discover -s tests -v
python scripts/test_plant.py
python scripts/test_magnetic_z.py
python scripts/run_z_hover.py --noise none --time 5 --settling-time 2

Sau đó mới bật từng loại noise.

9. Cấu trúc project

scripts/
    run_z_hover.py
    test_magnetic_z.py
    test_plant.py
    test_z_plant_open_loop.py

src/endoscopy_capsule_control/
    control/       # đây là bdk PID+FF
    magnetic/      # động lực học từ
    plant/         # toàn bộ plant từ sai số localization, nguồn, robpt
    simulation/    # MuJoCo, AUBO IK, viewer
    models/        # AUBO i10 XML + meshes
    config.py

tests/
    test_noise_models.py
