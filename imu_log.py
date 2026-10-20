#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""采集 N 秒静止数据，统计噪声并对比「纯陀螺积分 / 加速度计 / 互补融合」

用法：
    python imu_log.py                    # 60 秒，存 imu_log.csv
    python imu_log.py 60 after_fusion    # 60 秒，存 after_fusion.csv
"""
import sys
import time
import serial
import numpy as np

# ==================== 配置 ====================
PORT = "COM5"
BAUD = 115200
DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 60
LABEL = sys.argv[2] if len(sys.argv) > 2 else "imu_log"
OUT = f"{LABEL}.csv" if LABEL != "imu_log" else "imu_log.csv"

# ==================== 采集 ====================
ser = serial.Serial(PORT, BAUD, timeout=1)
rows = []
t0 = time.time()
print(f"开始采集 {DURATION} 秒 —— 标签 [{LABEL}]，板子放稳，手离开！")

while time.time() - t0 < DURATION:
    line = ser.readline().decode("ascii", errors="ignore").strip()
    if not line.startswith("D,"):
        continue
    parts = line.split(",")
    if len(parts) != 18:          # D + dt + 3加速度 + 3陀螺 + 2+2+2 角度 + 4 四元数
        continue
    try:
        rows.append([int(x) for x in parts[1:]])
    except ValueError:
        continue

ser.close()

a = np.array(rows)
HEADER = ("dt,ax,ay,az,gx10,gy10,gz10,"
          "acc_roll10,acc_pitch10,gyr_roll10,gyr_pitch10,"
          "fus_roll10,fus_pitch10")
np.savetxt(OUT, a, delimiter=",", fmt="%d", header=HEADER, comments="")

# ==================== 列索引 ====================
# 0=dt 1..3=acc 4..6=gyro 7..8=acc角 9..10=陀螺角 11..12=融合角
I_AX, I_AY, I_AZ = 1, 2, 3
I_GX, I_GY, I_GZ = 4, 5, 6
I_AR, I_AP = 7, 8
I_GR, I_GP = 9, 10
I_FR, I_FP = 11, 12

n = len(a)
print(f"\n共采集 {n} 帧（约 {n/50:.1f} 秒）")

# ==================== 基础统计 ====================
names = ["dt", "ax", "ay", "az", "gx10", "gy10", "gz10",
         "acc_R", "acc_P", "gyr_R", "gyr_P", "fus_R", "fus_P"]
print(f"\n{'通道':<8}{'均值':>10}{'标准差':>10}{'最小':>10}{'最大':>10}{'峰峰值':>9}")
print("-" * 60)
for i, nm in enumerate(names):
    c = a[:, i]
    print(f"{nm:<8}{c.mean():>10.1f}{c.std():>10.1f}{c.min():>10.0f}{c.max():>10.0f}{c.max()-c.min():>9.0f}")

# ==================== 加速度 / 重力 ====================
ax_mg = a[:, I_AX].mean() / 16.384
ay_mg = a[:, I_AY].mean() / 16.384
az_mg = a[:, I_AZ].mean() / 16.384
mag = np.sqrt(ax_mg**2 + ay_mg**2 + az_mg**2)
print("\n===== 加速度 =====")
print(f"  均值 (mg) :  ax={ax_mg:8.1f}  ay={ay_mg:8.1f}  az={az_mg:8.1f}")
print(f"  重力模长  :  |a| = {mag:7.1f} mg = {mag/1000:.4f} g")

# ==================== 陀螺仪 ====================
gx, gy, gz = a[:, I_GX] / 10.0, a[:, I_GY] / 10.0, a[:, I_GZ] / 10.0
print("\n===== 陀螺仪（已减零偏，单位 °/s）=====")
print(f"  残余零偏 :  gx={gx.mean():8.4f}  gy={gy.mean():8.4f}  gz={gz.mean():8.4f}")
print(f"  噪声 std :  gx={gx.std():8.4f}  gy={gy.std():8.4f}  gz={gz.std():8.4f}")

# ==================== 三种方法的姿态角对比 ====================
acc_r, acc_p = a[:, I_AR] / 10.0, a[:, I_AP] / 10.0
gyr_r, gyr_p = a[:, I_GR] / 10.0, a[:, I_GP] / 10.0
fus_r, fus_p = a[:, I_FR] / 10.0, a[:, I_FP] / 10.0

print("\n" + "=" * 68)
print("姿态角对比（静止采集，看「漂移」和「噪声」两个维度）")
print("=" * 68)
print(f"{'':<10}{'起始':>12}{'结束':>12}{'总漂移':>12}{'噪声 std':>12}")
print("-" * 68)

for nm, arr in (("加速度计", acc_r), ("纯陀螺", gyr_r), ("融合", fus_r)):
    print(f"roll  {nm:<6}{arr[0]:>12.2f}{arr[-1]:>12.2f}"
          f"{arr[-1]-arr[0]:>12.2f}{arr.std():>12.3f}")

print()
for nm, arr in (("加速度计", acc_p), ("纯陀螺", gyr_p), ("融合", fus_p)):
    print(f"pitch {nm:<6}{arr[0]:>12.2f}{arr[-1]:>12.2f}"
          f"{arr[-1]-arr[0]:>12.2f}{arr.std():>12.3f}")

print("\n判读：")
print("  加速度计 → 不长漂，但噪声大（运动时更差）")
print("  纯陀螺   → 平滑，但会持续漂移")
print("  融合     → 漂移应显著小于纯陀螺，噪声应显著小于加速度计")

print(f"\n数据已保存到 {OUT}")
