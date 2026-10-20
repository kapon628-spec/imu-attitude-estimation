#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IMU 3D 姿态可视化 —— 直接用四元数驱动立方体（不经过 Euler 角）

协议（18 字段）：
    D,dt,ax,ay,az,gx10,gy10,gz10,accR10,accP10,gyrR10,gyrP10,mahR10,mahP10,qw,qx,qy,qz
    最后 4 个是四元数 × 10000

依赖：pip install pyserial matplotlib numpy
运行前请关闭 imu_plot.py / SSCOM（串口独占）
"""

import sys
import threading

import numpy as np
import serial
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ==================== 配置 ====================
PORT = "COM5"        # ← 改成你的 CH340 端口号
BAUD = 115200

# ==================== 共享状态 ====================
q = [1.0, 0.0, 0.0, 0.0]      # 四元数 (w, x, y, z)，初始为单位四元数
total = 0
lock = threading.Lock()
running = True

# ==================== 打开串口 ====================
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
except Exception as e:
    print(f"打开串口失败: {e}")
    print("请检查：")
    print("  1) PORT 是否与设备管理器的 CH340 一致")
    print("  2) imu_plot.py / 串口助手 是否还开着（会独占端口）")
    sys.exit(1)

print(f"已连接 {PORT} @ {BAUD}")


def reader():
    """后台线程：解析四元数"""
    global q, total, running
    while running:
        try:
            line = ser.readline().decode("ascii", errors="ignore").strip()
        except Exception:
            break
        if not line.startswith("D,"):
            continue
        parts = line.split(",")
        if len(parts) != 18:
            continue
        try:
            v = [int(x) for x in parts[1:]]
        except ValueError:
            continue

        # v: [0]=dt [1..3]=acc [4..6]=gyro [7..8]acc角 [9..10]陀螺角
        #    [11..12]Mahony角 [13..16]=四元数 w,x,y,z
        qq = [v[13] / 10000.0, v[14] / 10000.0,
              v[15] / 10000.0, v[16] / 10000.0]
        n = float(np.linalg.norm(qq))
        if n > 0.5:                       # 剔除无效帧
            with lock:
                q = [c / n for c in qq]
            total += 1


threading.Thread(target=reader, daemon=True).start()

# ==================== 模型：一块板子 ====================
# 实测：模块长边沿 Y 轴，短边沿 X 轴
LX, LY, LZ = 1.4, 2.2, 0.3         # X 短、Y 长、Z 薄

# 显示校正：整个模型「从上往下看」绕世界 Z 轴旋转的角度（度）。
# 用来把模型的朝向和真实空间的摆放对上。
# 如果发现方向反了或不对，改成 0 / -90 / 180 再试。
WORLD_YAW_DEG = 90.0

_yaw = np.radians(WORLD_YAW_DEG)
RZ = np.array([
    [np.cos(_yaw), -np.sin(_yaw), 0.0],
    [np.sin(_yaw),  np.cos(_yaw), 0.0],
    [0.0,          0.0,           1.0],
])

V = np.array([
    [-LX/2, -LY/2, -LZ/2], [ LX/2, -LY/2, -LZ/2],
    [ LX/2,  LY/2, -LZ/2], [-LX/2,  LY/2, -LZ/2],
    [-LX/2, -LY/2,  LZ/2], [ LX/2, -LY/2,  LZ/2],
    [ LX/2,  LY/2,  LZ/2], [-LX/2,  LY/2,  LZ/2],
])

FACES = [
    [0, 1, 2, 3],      # -Z 面（板子背面）
    [4, 5, 6, 7],      # +Z 面（板子正面，用橙色标记）
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
]
FACE_COLORS = ["#5b7c99", "#e8871e", "#4a90d9", "#3d7ab5", "#2c5f8a", "#6aabe0"]


def rotmat(w, x, y, z):
    """四元数 -> 旋转矩阵（体坐标 -> 世界坐标）"""
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def quat_to_euler(w, x, y, z):
    """转成 Euler 仅用于文字显示（内部姿态用四元数）"""
    r2d = 57.2957795
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)) * r2d
    s = np.clip(2*(w*y - z*x), -1.0, 1.0)
    pitch = np.arcsin(s) * r2d
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)) * r2d
    return roll, pitch, yaw


# ==================== 画布 ====================
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection="3d")
fig.suptitle("IMU 3D 姿态（四元数驱动）", fontsize=13)

ax.set_box_aspect((1, 1, 1))
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_zlim(-2.2, 2.2)
ax.set_xlabel("世界 X")
ax.set_ylabel("世界 Y")
ax.set_zlabel("世界 Z（上）")

# 世界参考：Z 轴（竖直）
ax.plot([0, 0], [0, 0], [-2.0, 2.0], color="gray", ls="--", lw=1.0, alpha=0.7)

# 地面网格（z = -1.8）
for t in np.linspace(-1.8, 1.8, 5):
    ax.plot([t, t], [-1.8, 1.8], [-1.8, -1.8], color="#cccccc", lw=0.6)
    ax.plot([-1.8, 1.8], [t, t], [-1.8, -1.8], color="#cccccc", lw=0.6)

# 立方体
poly = Poly3DCollection([[V[i] for i in FACES[0]]],
                        facecolors=FACE_COLORS, edgecolors="#1a1a1a", linewidths=0.8)
ax.add_collection3d(poly)

# 体坐标三轴
AX_VEC = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
AX_COL = ["#d62728", "#2ca02c", "#1f77b4"]
ax_lines = [ax.plot([0, 1], [0, 0], [0, 0], color=AX_COL[i], lw=2.6)[0]
            for i in range(3)]
ax_texts = [ax.text(1.15, 0, 0, "X", color=AX_COL[0], fontsize=11, fontweight="bold"),
            ax.text(0, 1.15, 0, "Y", color=AX_COL[1], fontsize=11, fontweight="bold"),
            ax.text(0, 0, 1.15, "Z", color=AX_COL[2], fontsize=11, fontweight="bold")]

info = fig.text(0.02, 0.95, "", fontsize=10, va="top")


def update(_frame):
    with lock:
        w, x, y, z = q
        n = total

    R = rotmat(w, x, y, z)
    RT = RZ @ R                       # 叠加显示校正（绕世界 Z 轴）

    # 板子顶点转到世界坐标
    W = V @ RT.T
    poly.set_verts([[W[i] for i in f] for f in FACES])

    # 三轴
    for i in range(3):
        d = RT @ AX_VEC[i] * 1.5
        ax_lines[i].set_data_3d([0, d[0]], [0, d[1]], [0, d[2]])
        ax_texts[i].set_position_3d(d * 1.12)

    roll, pitch, yaw = quat_to_euler(w, x, y, z)
    info.set_text(
        f"帧数 {n}   "
        f"四元数 ({w:+.3f}, {x:+.3f}, {y:+.3f}, {z:+.3f})\n"
        f"roll {roll:7.1f}°    pitch {pitch:7.1f}°    yaw {yaw:7.1f}°\n"
        f"橙色面 = 模块 +Z（背面）")

    return [poly] + ax_lines + ax_texts


ani = FuncAnimation(fig, update, interval=40, blit=False, cache_frame_data=False)
plt.tight_layout()
plt.show()

running = False
ser.close()
