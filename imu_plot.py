#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IMU 实时波形上位机 —— 依赖: pip install pyserial matplotlib"""

import sys
import threading
from collections import deque

import serial
import matplotlib.pyplot as plt

# ===== 中文显示修复 =====
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False   # 修复负号变方块

from matplotlib.animation import FuncAnimation

# ==================== 配置 ====================
PORT = "COM5"        # ← 改成你的 CH340 端口号
BAUD = 115200
WINDOW = 300         # 显示最近 300 点（50Hz 下约 6 秒）

# ==================== 数据缓冲 ====================
ch = [deque(maxlen=WINDOW) for _ in range(6)]   # ax,ay,az,gx,gy,gz
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
        if len(parts) != 8:
            continue
        try:
            vals = [int(x) for x in parts[1:]]
        except ValueError:
            continue
        dt_hist.append(vals[0])
        for i in range(6):
            ch[i].append(vals[1 + i])
        total_count += 1


threading.Thread(target=reader, daemon=True).start()

# ==================== 绘图 ====================
fig, (ax_acc, ax_gyr) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
fig.suptitle("IMU 实时波形", fontsize=13)

COLORS = ["#d62728", "#2ca02c", "#1f77b4"]

ln_acc = [ax_acc.plot([], [], color=COLORS[i], lw=1, label=n)[0]
          for i, n in enumerate(["ax", "ay", "az"])]
ln_gyr = [ax_gyr.plot([], [], color=COLORS[i], lw=1, label=n)[0]
          for i, n in enumerate(["gx", "gy", "gz"])]

ax_acc.set_ylabel("加速度 (LSB)")
ax_acc.set_ylim(-35000, 35000)
ax_acc.legend(loc="upper right", ncol=3, fontsize=9)
ax_acc.grid(True, alpha=0.3)

ax_gyr.set_ylabel("陀螺仪 (LSB)")
ax_gyr.set_ylim(-35000, 35000)
ax_gyr.set_xlabel("样本序号")
ax_gyr.legend(loc="upper right", ncol=3, fontsize=9)
ax_gyr.grid(True, alpha=0.3)

info = ax_acc.text(0.01, 0.97, "", transform=ax_acc.transAxes,
                   fontsize=9, va="top", family="monospace")


def update(_frame):
    n = len(ch[0])
    if n == 0:
        return ln_acc + ln_gyr + [info]

    x = list(range(n))
    for i in range(3):
        ln_acc[i].set_data(x, list(ch[i]))
        ln_gyr[i].set_data(x, list(ch[3 + i]))

    ax_acc.set_xlim(0, WINDOW)
    ax_gyr.set_xlim(0, WINDOW)
    info.set_text(f"buffer={n}/{WINDOW}  total={total_count}  dt={dt_hist[-1]}us")
    return ln_acc + ln_gyr + [info]


ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
plt.tight_layout()
plt.show()

running = False
ser.close()