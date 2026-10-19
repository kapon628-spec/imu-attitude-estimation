#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IMU 实时波形上位机 —— 依赖: pip install pyserial matplotlib

数据协议（14 字段）：
    D,dt_us,ax,ay,az,gx10,gy10,gz10,ar10,ap10,gr10,gp10,fr10,fp10
    - ax/ay/az        : 加速度 LSB
    - gx10/gy10/gz10  : 陀螺仪 0.1°/s（已减零偏）
    - ar10/ap10       : 加速度计解算的 roll/pitch，0.1°
    - gr10/gp10       : 纯陀螺积分的 roll/pitch，0.1°
    - fr10/fp10       : 互补滤波融合后的 roll/pitch，0.1°
"""

import sys
import threading
from collections import deque

import serial
import matplotlib.pyplot as plt
import numpy as np
# ===== 中文显示修复 =====
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False   # 修复负号变方块

from matplotlib.animation import FuncAnimation

# ==================== 配置 ====================
PORT = "COM5"        # ← 改成你的 CH340 端口号
BAUD = 115200
WINDOW = 300         # 显示最近 300 点（50Hz 下约 6 秒）

N_CH = 12            # 3 加速度 + 3 陀螺 + 2 加速度角 + 2 陀螺角 + 2 融合角

# 通道索引
I_AX, I_AY, I_AZ = 0, 1, 2
I_GX, I_GY, I_GZ = 3, 4, 5
I_AR, I_AP = 6, 7
I_GR, I_GP = 8, 9
I_FR, I_FP = 10, 11

# ==================== 数据缓冲 ====================
ch = [deque(maxlen=WINDOW) for _ in range(N_CH)]
dt_hist = deque(maxlen=WINDOW)
running = True
total_count = 0      # 累计收到的帧数（用来确认数据仍在流入）

# ==================== 打开串口 ====================
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
except Exception as e:
    print(f"打开串口失败: {e}")
    print("请检查：")
    print("  1) PORT 是否与设备管理器里的 CH340 端口一致")
    print("  2) 串口助手是否还开着（会独占端口，必须先关闭）")
    sys.exit(1)

print(f"已连接 {PORT} @ {BAUD}")


def reader():
    """后台线程：持续读串口并解析"""
    global running, total_count
    while running:
        try:
            line = ser.readline().decode("ascii", errors="ignore").strip()
        except Exception:
            break
        if not line.startswith("D,"):
            continue
        parts = line.split(",")
        if len(parts) != 14:          # D + dt + 3 + 3 + 2 + 2 + 2
            continue
        try:
            vals = [int(x) for x in parts[1:]]
        except ValueError:
            continue

        # vals: [0]=dt [1..3]=acc [4..6]=gyro [7..8]=acc角 [9..10]=陀螺角 [11..12]=融合角
        dt_hist.append(vals[0])
        for i in range(3):
            ch[I_AX + i].append(vals[1 + i])            # 加速度 LSB
            ch[I_GX + i].append(vals[4 + i] / 10.0)     # 陀螺 °/s

        ch[I_AR].append(vals[7] / 10.0)
        ch[I_AP].append(vals[8] / 10.0)
        ch[I_GR].append(vals[9] / 10.0)
        ch[I_GP].append(vals[10] / 10.0)
        ch[I_FR].append(vals[11] / 10.0)
        ch[I_FP].append(vals[12] / 10.0)
        total_count += 1


threading.Thread(target=reader, daemon=True).start()

# ==================== 绘图 ====================
fig, (ax_acc, ax_gyr, ax_roll, ax_pitch) = plt.subplots(
    4, 1, figsize=(11, 12), sharex=True)
fig.suptitle("IMU 实时波形 · 传感器融合", fontsize=13)

C = ["#d62728", "#2ca02c", "#1f77b4"]

ln_acc = [ax_acc.plot([], [], color=C[i], lw=1, label=n)[0]
          for i, n in enumerate(["ax", "ay", "az"])]
ln_gyr = [ax_gyr.plot([], [], color=C[i], lw=1, label=n)[0]
          for i, n in enumerate(["gx", "gy", "gz"])]

# 每条对比曲线：加速度计 / 纯陀螺 / 融合
LN_ACC_A, LN_GYR_ANG, LN_FUS = "#9ecae1", "#fd8d3c", "#d62728"
ln_roll = [
    ax_roll.plot([], [], color=LN_ACC_A, lw=1.0, label="加速度计")[0],
    ax_roll.plot([], [], color=LN_GYR_ANG, lw=1.0, ls="--", label="纯陀螺积分")[0],
    ax_roll.plot([], [], color=LN_FUS, lw=1.8, label="融合")[0],
]
ln_pitch = [
    ax_pitch.plot([], [], color=LN_ACC_A, lw=1.0, label="加速度计")[0],
    ax_pitch.plot([], [], color=LN_GYR_ANG, lw=1.0, ls="--", label="纯陀螺积分")[0],
    ax_pitch.plot([], [], color=LN_FUS, lw=1.8, label="融合")[0],
]

ax_acc.set_ylabel("加速度 (LSB)")
ax_acc.set_ylim(-35000, 35000)
ax_acc.legend(loc="upper right", ncol=3, fontsize=9)
ax_acc.grid(True, alpha=0.3)

ax_gyr.set_ylabel("陀螺仪 (°/s，已标定)")
ax_gyr.set_ylim(-260, 260)
ax_gyr.legend(loc="upper right", ncol=3, fontsize=9)
ax_gyr.grid(True, alpha=0.3)

for ax, title in ((ax_roll, "roll (°)"), (ax_pitch, "pitch (°)")):
    ax.set_ylabel(title)
    ax.set_ylim(-270, 270)
    ax.axhline(0, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax.legend(loc="upper right", ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)

ax_pitch.set_xlabel("样本序号")

info = ax_acc.text(0.01, 0.97, "", transform=ax_acc.transAxes,
                   fontsize=9, va="top", family="monospace")

ALL_LN = ln_acc + ln_gyr + ln_roll + ln_pitch


def update(_frame):
    n = len(ch[0])
    if n == 0:
        return ALL_LN + [info]

    x = list(range(n))

    for i in range(3):
        ln_acc[i].set_data(x, list(ch[I_AX + i]))
        ln_gyr[i].set_data(x, list(ch[I_GX + i]))

    # roll / pitch：加速度计、纯陀螺、融合 三条对比
    ln_roll[0].set_data(x, list(ch[I_AR]))
    ln_roll[1].set_data(x, list(ch[I_GR]))
    ln_roll[2].set_data(x, list(ch[I_FR]))

    ln_pitch[0].set_data(x, list(ch[I_AP]))
    ln_pitch[1].set_data(x, list(ch[I_GP]))
    ln_pitch[2].set_data(x, list(ch[I_FP]))

    for ax in (ax_acc, ax_gyr, ax_roll, ax_pitch):
        ax.set_xlim(0, WINDOW)

    info.set_text(
        f"buffer={n}/{WINDOW}  total={total_count}  dt={dt_hist[-1]}us\n"
        f"roll:  融合={ch[I_FR][-1]:7.2f}  陀螺={ch[I_GR][-1]:7.2f}  acc={ch[I_AR][-1]:7.2f}\n"
        f"pitch: 融合={ch[I_FP][-1]:7.2f}  陀螺={ch[I_GP][-1]:7.2f}  acc={ch[I_AP][-1]:7.2f}")

    return ALL_LN + [info]


ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
plt.tight_layout()
plt.show()

running = False
ser.close()
