#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 13 性能评估：采样时序、丢帧、长时间漂移、延迟构成

用法：
    python analyze_perf.py                      # 分析 imu_log.csv
    python analyze_perf.py long_run.csv         # 分析指定文件
    python analyze_perf.py --live 10            # 实时测量串口到达时序（10 秒）

输出：终端报告 + docs/figs/*.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "docs", "figs")
os.makedirs(OUTDIR, exist_ok=True)

PORT = "COM5"
BAUD = 115200
NOMINAL_DT_US = 5000.0          # 200 Hz 采样
NOMINAL_RATE_HZ = 200.0
DECIM = 4                       # 固件每 4 个采样发送 1 帧（上传 50 Hz）
                                # 注意：协议里的 dt 是「采样周期」而不是「发送间隔」，
                                #       所以重建时间轴必须乘以 DECIM，否则时长少算 4 倍


# ==================== 数据加载 ====================
def load_csv(path):
    """返回 dict；兼容 13 列（14 字段协议）和 17 列（18 字段协议）"""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    ncol = data.shape[1]

    base = ["dt", "ax", "ay", "az", "gx10", "gy10", "gz10",
            "accR", "accP", "gyrR", "gyrP", "fusR", "fusP"]
    if ncol == 13:
        names = base
    elif ncol == 17:
        names = base + ["qw", "qx", "qy", "qz"]
    else:
        print(f"警告：未知列数 {ncol}，按前 13 列解析")
        names = base
    return {n: data[:, i] for i, n in enumerate(names[:ncol]) if i < len(names)}, ncol


# ==================== 时序分析 ====================
def analyze_timing(d, fig_prefix="dt"):
    dt = d["dt"]
    dt_ms = dt / 1000.0

    print("\n" + "=" * 68)
    print("一、采样时序质量")
    print("=" * 68)
    print(f"  总帧数            : {len(dt)}")
    print(f"  标称周期          : {NOMINAL_DT_US:.0f} us ({NOMINAL_RATE_HZ:.0f} Hz)")
    print(f"  实测平均周期      : {dt.mean():.1f} us")
    print(f"  标准差            : {dt.std():.2f} us")
    print(f"  最小 / 最大       : {dt.min():.0f} / {dt.max():.0f} us")
    print(f"  峰峰值            : {dt.max() - dt.min():.0f} us")

    # 丢帧 / 多帧判定
    miss = np.sum(dt > NOMINAL_DT_US * 1.5)
    short = np.sum(dt < NOMINAL_DT_US * 0.5)
    exact = np.sum(np.isclose(dt, NOMINAL_DT_US))
    rate = 1e6 / dt.mean()

    print(f"  正好等于标称周期  : {exact} 帧 ({100*exact/len(dt):.1f}%)")
    print(f"  丢帧（dt>1.5x）   : {miss} 帧")
    print(f"  异常短帧（dt<0.5x）: {short} 帧")
    print(f"  有效采样率        : {rate:.2f} Hz")
    print(f"  实际上传率        : {rate/DECIM:.2f} Hz（每 {DECIM} 帧发 1 帧）")
    print(f"  数据总时长        : {len(dt)*NOMINAL_DT_US*DECIM/1e6:.1f} s")
    lost_time = (dt.sum() * DECIM - len(dt) * NOMINAL_DT_US * DECIM) / 1e6
    print(f"  相对标称的累计时间偏差: {lost_time:+.3f} s")

    # 到达间隔的直方图
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(dt_ms, bins=60, color="#4a90d9", edgecolor="white")
    ax.set_xlabel("采样周期 dt (ms)")
    ax.set_ylabel("帧数")
    ax.set_title(f"采样周期分布（中位 {np.median(dt_ms):.3f} ms，标准差 {dt.std():.2f} us）")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    p = os.path.join(OUTDIR, f"{fig_prefix}_histogram.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  图：{p}")

    return dt


# ==================== 漂移分析 ====================
def wrap180(a):
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def analyze_drift(d):
    # 每个发送帧覆盖 DECIM 个采样周期，时间轴要乘回去
    t = np.cumsum(d["dt"] * DECIM) / 1e6          # 秒
    t = t - t[0]
    dur_min = t[-1] / 60.0

    print("\n" + "=" * 68)
    print(f"二、长时间漂移分析（时长 {t[-1]:.1f} s = {dur_min:.2f} min）")
    print("=" * 68)

    series = {
        "roll": [("加速度计", d["accR"] / 10.0), ("纯陀螺", d["gyrR"] / 10.0),
                 ("Mahony", d["fusR"] / 10.0)],
        "pitch": [("加速度计", d["accP"] / 10.0), ("纯陀螺", d["gyrP"] / 10.0),
                  ("Mahony", d["fusP"] / 10.0)],
    }

    results = {}
    print(f"{'通道':<8}{'方法':<10}{'漂移率(°/min)':>16}{'去趋势残差σ(°)':>18}{'总变化(°)':>12}")
    print("-" * 68)
    for axis_name, items in series.items():
        results[axis_name] = []
        for label, raw in items:
            ang = np.asarray(raw, dtype=float)
            if axis_name == "roll":
                ang = np.degrees(np.unwrap(np.radians(ang)))     # 解缠后再拟合
            k, b = np.polyfit(t, ang, 1)
            resid = ang - (k * t + b)
            drift_per_min = k * 60.0
            results[axis_name].append((label, drift_per_min, resid.std(), ang[-1] - ang[0]))
            print(f"{axis_name:<8}{label:<10}{drift_per_min:>16.3f}"
                  f"{resid.std():>18.3f}{ang[-1]-ang[0]:>12.3f}")
        print("-" * 68)

    # 漂移曲线图
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, axis_name in zip(axes, ("roll", "pitch")):
        for (label, raw), color, lw in zip(series[axis_name],
                                           ("#7fb8e8", "#f0a030", "#d62728"),
                                           (1.0, 1.2, 2.0)):
            ang = np.asarray(raw, dtype=float)
            if axis_name == "roll":
                ang = np.degrees(np.unwrap(np.radians(ang)))
            k, b = np.polyfit(t, ang, 1)
            ax.plot(t, ang, color=color, lw=lw, label=label, alpha=0.9)
            ax.plot(t, k * t + b, color=color, lw=0.8, ls=":",
                    label=f"{label} 拟合 {k*60:+.2f}°/min")
        ax.set_ylabel(f"{axis_name} (°)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", ncol=2)
    axes[-1].set_xlabel("时间 (s)")
    axes[0].set_title("长时间漂移：三路姿态角与线性拟合")
    fig.tight_layout()
    p = os.path.join(OUTDIR, "longrun_drift.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  图：{p}")

    return results, t


# ==================== 噪声分析 ====================
def analyze_noise(d, t):
    print("\n" + "=" * 68)
    print("三、噪声与精度（去趋势后的残差标准差）")
    print("=" * 68)
    rows = [
        ("陀螺仪 gx", d["gx10"] / 10.0, "°/s"),
        ("陀螺仪 gy", d["gy10"] / 10.0, "°/s"),
        ("陀螺仪 gz", d["gz10"] / 10.0, "°/s"),
        ("加速度计 ax", d["ax"] / 16.384, "mg"),
        ("加速度计 ay", d["ay"] / 16.384, "mg"),
        ("加速度计 az", d["az"] / 16.384, "mg"),
    ]
    print(f"{'通道':<16}{'均值':>12}{'残差σ':>12}{'峰峰值':>12}{'单位':>8}")
    print("-" * 68)
    for name, arr, unit in rows:
        arr = np.asarray(arr, dtype=float)
        k, b = np.polyfit(t, arr, 1)
        resid = arr - (k * t + b)
        print(f"{name:<16}{arr.mean():>12.3f}{resid.std():>12.3f}"
              f"{arr.max()-arr.min():>12.3f}{unit:>8}")

    # 角度噪声
    print()
    print(f"{'角度通道':<20}{'噪声σ(°)':>14}")
    print("-" * 68)
    for name, arr in (("acc_roll", d["accR"] / 10.0), ("acc_pitch", d["accP"] / 10.0),
                      ("mahony_roll", d["fusR"] / 10.0), ("mahony_pitch", d["fusP"] / 10.0)):
        arr = np.asarray(arr, dtype=float)
        k, b = np.polyfit(t, arr, 1)
        print(f"{name:<20}{(arr-(k*t+b)).std():>14.4f}")


# ==================== 实时延迟测量 ====================
def live_timing(seconds=10.0):
    import time
    import serial

    print("\n" + "=" * 68)
    print(f"四、串口到达时序实时测量（{seconds:.0f} 秒）")
    print("=" * 68)
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
    except Exception as e:
        print(f"  打开串口失败：{e}")
        return

    stamps = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        line = ser.readline()
        if line.startswith(b"D,"):
            stamps.append(time.perf_counter())
    ser.close()

    if len(stamps) < 10:
        print("  数据太少，跳过")
        return

    iv = np.diff(stamps) * 1000.0          # ms
    print(f"  收到帧数          : {len(stamps)}")
    print(f"  平均到达间隔      : {iv.mean():.2f} ms  ({1000/iv.mean():.1f} Hz)")
    print(f"  抖动 σ            : {iv.std():.2f} ms")
    print(f"  最小 / 最大间隔   : {iv.min():.2f} / {iv.max():.2f} ms")

    n = len(stamps)
    expected = np.arange(n) * (iv.mean() / 1000.0)
    actual = np.array(stamps) - stamps[0]
    late = (actual - expected) * 1000.0
    print(f"  累计偏斜范围      : {late.min():+.1f} ~ {late.max():+.1f} ms")

    print("\n  延迟构成（数据路径）")
    print("  " + "-" * 50)
    print(f"  ① 采样周期一半    : {NOMINAL_DT_US/2000:.1f} ms（平均等待下一个采样点）")
    print(f"  ② 发送分组        : 0 ~ {4*NOMINAL_DT_US/1000:.0f} ms（每 4 帧发一次，平均 "
          f"{2*NOMINAL_DT_US/1000:.1f} ms）")
    print(f"  ③ 串口传输        : 约 7.6 ms（88 字节 @115200，估算）")
    print(f"  => 数据路径合计   : 约 {NOMINAL_DT_US/2000 + 2*NOMINAL_DT_US/1000 + 7.6:.1f} ms")
    print("  " + "-" * 50)
    print("  ④ 显示刷新        : 0 ~ 1/刷新率（面板 10 Hz 时最多 100 ms，30 Hz 时 33 ms）")
    print("  => 端到端总计     : 数据路径 + 显示刷新")


# ==================== main ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=os.path.join(HERE, "imu_log.csv"))
    ap.add_argument("--live", type=float, default=0.0,
                    help="实时测量串口到达时序的秒数")
    args = ap.parse_args()

    if os.path.exists(args.csv):
        d, ncol = load_csv(args.csv)
        print("=" * 68)
        print(f"数据文件：{args.csv}   列数 {ncol}")
        print("=" * 68)
        dt = analyze_timing(d)
        results, t = analyze_drift(d)
        analyze_noise(d, t)
    else:
        print(f"找不到文件：{args.csv}")

    if args.live > 0:
        live_timing(args.live)

    print("\n分析完成。")


if __name__ == "__main__":
    main()
