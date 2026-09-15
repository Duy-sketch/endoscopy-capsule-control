# endoscopy-capsule-control
Endoscopy Capsule Control

Mô phỏng và điều khiển capsule nội soi từ bằng hệ DEMA trong MuJoCo.

Chức năng chính

Mô hình capsule 6-DOF

Mô hình lực và mô-men từ

PID trục Z + ổn định cục bộ

Động học dòng điện

Nhiễu và trễ đo lường

Thống kê RMS sau quá độ

Viewer theo capsule hoặc toàn hệ DEMA

Cài đặt

pip install -e .

Chạy mô phỏng

python scripts/run_hover.py --time 10 --settling-time 5 --x0-mm 1 --z0-mm 2 --theta0-deg 1

Viewer toàn hệ DEMA:

python scripts/run_hover.py --time 10 --settling-time 5 --x0-mm 1 --z0-mm 2 --theta0-deg 1 --viewer --camera overview

Cấu trúc

scripts/
src/endoscopy_capsule_control/
    control/
    dynamics/
    magnetic/
    sensing/
    simulation/
    models/
Video mô phỏng z hovering

https://github.com/user-attachments/assets/ec13038b-2922-4344-a22e-b9134a6b9c62


