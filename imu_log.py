#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""采集 N 秒静止数据，统计各通道噪声并保存 CSV"""

import time
import serial
import numpy as np

# ==================== 配置 ====================
PORT = "COM5"
BAUD = 115200
DURATION = 60          # 采集秒数
OUT = "imu_log.csv"

# ==================== 打开串口 ====================
ser = serial.Serial(PORT, BAUD, timeout=1)
rows = []
t0 = time.time()
print(f"开始采集 {DURATION} 秒 —— 板子放到桌上，手离开，别碰！")

while time.time() - t0 < DURATION:
    line = ser.readline().decode("ascii", errors="ignore").strip()
    if not line.startswith("D,"):
        continue
    parts = line.split(",")
    if len(parts) != 8:
        continue
    try:
        rows.append([int(x) for x in parts[1:]])
    except ValueError:
        continue

ser.close()

a = np.array(rows)
np.savetxt(OUT, a, delimiter=",", fmt="%d",
           header="dt,ax,ay,az,gx,gy,gz", comments="")

# ==================== 统计 ====================
names = ["dt", "ax", "ay", "az", "gx", "gy", "gz"]

print(f"\n共采集 {len(a)} 帧（约 {len(a)/50:.1f} 秒）\n")
print(f"{'通道':<5}{'均值':>10}{'标准差':>10}{'最小':>8}{'最大':>8}{'峰峰值':>8}")
print("-" * 55)
for i, n in enumerate(names):
    c = a[:, i]
    print(f"{n:<5}{c.mean():>10.1f}{c.std():>10.1f}{c.min():>8.0f}{c.max():>8.0f}{c.max()-c.min():>8.0f}")

print("\n===== 物理量换算 =====")
print(f"加速度均值  (mg) :  ax={a[:,1].mean()/16.384:8.1f}  ay={a[:,2].mean()/16.384:8.1f}  az={a[:,3].mean()/16.384:8.1f}")
print(f"陀螺零偏    (°/s):  gx={a[:,4].mean()/131:8.3f}  gy={a[:,5].mean()/131:8.3f}  gz={a[:,6].mean()/131:8.3f}")
print(f"陀螺噪声std (°/s):  gx={a[:,4].std()/131:8.3f}  gy={a[:,5].std()/131:8.3f}  gz={a[:,6].std()/131:8.3f}")
print(f"\n数据已保存到 {OUT}")