#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Euler 角 vs 四元数：量化对比实验

实验 1  奇点敏感性
        pitch 接近 ±90° 时，同样的物理小转动会造成 roll/yaw 的巨大变化
        —— 这就是万向节死锁的数值体现

实验 2  积分精度
        同样的陀螺仪数据，分别用三种方式积分出姿态：
          ① 朴素 Euler 积分   roll += p*dt （Day 10 代码里的做法）
          ② 严格 Euler 运动学 含 1/cos(pitch)，在 ±90° 处发散
          ③ 四元数积分       q += 0.5*q⊗ω*dt
        与解析真值比较姿态误差

输出：终端表格 + docs/figs/*.png
"""

import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "figs")
os.makedirs(OUTDIR, exist_ok=True)


# ==================== 基础工具 ====================
def quat_from_euler(roll, pitch, yaw):
    """ZYX 约定，输入为度"""
    r, p, y = np.radians([roll, pitch, yaw])
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def euler_from_quat(q):
    w, x, y, z = q
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return roll, pitch, yaw


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0, 0, 0])


def attitude_error_deg(q1, q2):
    """两个姿态之间的夹角（度）"""
    d = abs(float(np.dot(quat_normalize(q1), quat_normalize(q2))))
    return np.degrees(2.0 * np.arccos(min(1.0, d)))


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


# ==================== 实验 1：奇点敏感性 ====================
def experiment_1():
    print("=" * 66)
    print("实验 1：Euler 表示的奇点敏感性")
    print("  物理动作：在给定姿态下，绕体轴 Y 转 1.0°（非常小的动作）")
    print("  观察指标：Euler 角的 roll / yaw 因此改变了多少度")
    print("=" * 66)
    print(f"{'pitch(°)':>10}{'Δroll(°)':>12}{'Δyaw(°)':>12}{'1/cos(pitch)':>15}")
    print("-" * 66)

    eps = 1.0
    dq_body = quat_from_euler(0.0, eps, 0.0)     # 绕体 Y 转 1°
    pitches = [0, 30, 60, 80, 85, 88, 89, 89.5, 89.9]
    rows = []
    for pitch in pitches:
        q0 = quat_from_euler(20.0, pitch, 30.0)
        q1 = quat_normalize(quat_mul(q0, dq_body))   # 体坐标下叠加小转动
        r0, p0, y0 = euler_from_quat(q0)
        r1, p1, y1 = euler_from_quat(q1)
        dr = abs(float(wrap180(r1 - r0)))
        dy = abs(float(wrap180(y1 - y0)))
        amp = 1.0 / np.cos(np.radians(pitch))
        rows.append((pitch, dr, dy, amp))
        print(f"{pitch:>10.1f}{dr:>12.2f}{dy:>12.2f}{amp:>15.1f}")

    print("-" * 66)
    print("  结论：pitch→±90° 时，1° 的物理转动会让 yaw 变化上百甚至上千度；")
    print("        同样动作在一般姿态下只改变约 1°。这就是万向节死锁。")

    # 画图
    ps = np.linspace(-89.95, 89.95, 400)
    amp = 1.0 / np.cos(np.radians(ps))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(ps, amp, color="#d62728", lw=2)
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="一般姿态（放大约 1 倍）")
    ax.set_xlabel("pitch (°)")
    ax.set_ylabel("Euler 角变化放大倍数  1/cos(pitch)")
    ax.set_title("Euler 表示的奇点：pitch → ±90° 时放大倍数发散")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.annotate("pitch = ±90°\n奇点", xy=(88, 1 / np.cos(np.radians(88))),
                xytext=(40, 300), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="black"))
    fig.tight_layout()
    p1 = os.path.join(OUTDIR, "gimbal_sensitivity.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    print(f"\n  图已保存：{p1}")

    return rows


# ==================== 实验 2：积分精度 ====================
def experiment_2(duration=6.0, fs=200.0):
    dt = 1.0 / fs
    n = int(duration * fs)
    t = np.arange(n) * dt

    # 体坐标恒定角速度（绕固定体轴转 → 真值有解析解）
    # 选这组是为了让 Euler pitch 逼近 ±90°（实测最低 −89.9°），踩到奇点边缘
    omega = np.array([15.0, 80.0, 15.0])      # °/s
    w_mag = float(np.linalg.norm(omega))
    axis = np.radians(omega) / np.radians(w_mag)
    w_rad = np.radians(w_mag)

    # 解析真值：绕固定体轴匀速旋转
    def truth(k):
        half = 0.5 * w_rad * t[k]
        return np.array([np.cos(half), *(np.sin(half) * axis)])

    # ① 朴素 Euler 积分
    roll_n = pitch_n = yaw_n = 0.0
    # ② 严格 Euler 运动学积分
    roll_k = pitch_k = yaw_k = 0.0
    # ③ 四元数积分
    q = np.array([1.0, 0.0, 0.0, 0.0])

    err_n, err_k, err_q = [], [], []
    pitch_case_k = []
    diverged_at = None
    max_dyaw = 0.0          # ② 单步 yaw 变化的最大值
    yaw_k_prev = 0.0
    max_abs_pitch_true = 0.0

    p_dps, q_dps, r_dps = omega

    for k in range(1, n):
        # ---- ① 朴素 ----
        roll_n += p_dps * dt
        pitch_n += q_dps * dt
        yaw_n += r_dps * dt
        qn = quat_from_euler(roll_n, pitch_n, yaw_n)
        err_n.append(attitude_error_deg(truth(k), qn))

        # ---- ② 严格 Euler 运动学 ----
        phi = np.radians(roll_k)
        th = np.radians(pitch_k)
        cth = np.cos(th)
        if abs(cth) < 1e-8:                     # 奇点
            if diverged_at is None:
                diverged_at = t[k]
            err_k.append(np.nan)
        else:
            phid = p_dps + q_dps * np.sin(phi) * np.tan(th) + r_dps * np.cos(phi) * np.tan(th)
            thd = q_dps * np.cos(phi) - r_dps * np.sin(phi)
            psid = (q_dps * np.sin(phi) + r_dps * np.cos(phi)) / cth
            roll_k += phid * dt
            pitch_k += thd * dt
            yaw_k += psid * dt
            max_dyaw = max(max_dyaw, abs(yaw_k - yaw_k_prev))
            yaw_k_prev = yaw_k
            qk = quat_from_euler(roll_k, pitch_k, yaw_k)
            err_k.append(attitude_error_deg(truth(k), qk))
        pitch_case_k.append(pitch_k)
        max_abs_pitch_true = max(max_abs_pitch_true, abs(euler_from_quat(truth(k))[1]))

        # ---- ③ 四元数 ----
        wx, wy, wz = np.radians(omega)
        hdt = 0.5 * dt
        dq = np.array([
            -q[1] * wx - q[2] * wy - q[3] * wz,
            q[0] * wx + q[2] * wz - q[3] * wy,
            q[0] * wy - q[1] * wz + q[3] * wx,
            q[0] * wz + q[1] * wy - q[2] * wx,
        ]) * hdt
        q = quat_normalize(q + dq)
        err_q.append(attitude_error_deg(truth(k), q))

    err_n = np.array(err_n)
    err_k = np.array(err_k)
    err_q = np.array(err_q)
    tt = t[1:]

    print("\n" + "=" * 66)
    print("实验 2：大角度三维旋转下的积分精度")
    print(f"  体坐标角速度 ω = ({omega[0]:.0f}, {omega[1]:.0f}, {omega[2]:.0f}) °/s，"
          f"|ω| = {w_mag:.1f} °/s，持续 {duration:.0f} s")
    print(f"  总转角 = {w_mag * duration:.0f}°，采样 {fs:.0f} Hz")
    print("=" * 66)
    print(f"{'方法':<26}{'最大误差(°)':>14}{'末端误差(°)':>14}")
    print("-" * 66)

    def stat(a):
        a2 = a[np.isfinite(a)]
        return (np.max(a2) if len(a2) else float("nan"),
                a[-1] if np.isfinite(a[-1]) else float("nan"))

    for name, arr in (("① 朴素 Euler 积分", err_n),
                      ("② 严格 Euler 运动学", err_k),
                      ("③ 四元数积分", err_q)):
        mx, last = stat(arr)
        mxs = "发散" if not np.isfinite(mx) or mx > 1e4 else f"{mx:.2f}"
        ls = "发散" if not np.isfinite(last) else f"{last:.2f}"
        print(f"{name:<26}{mxs:>14}{ls:>14}")
    print("-" * 66)
    if diverged_at is not None:
        print(f"  ② 严格 Euler 运动学在 t = {diverged_at:.2f} s 处 pitch 抵达 ±90°，数学上发散")

    print(f"  ① 朴素 Euler 最大误差 {np.nanmax(err_n):.1f}°，"
          f"相当于四元数的 {np.nanmax(err_n) / max(np.nanmax(err_q), 1e-9):.0f} 倍")
    print(f"  ③ 四元数积分最大误差 {np.nanmax(err_q):.4f}°（一阶积分本身的截断误差）")
    print(f"  真实 pitch 最大 |pitch| = {max_abs_pitch_true:.2f}°（逼近 ±90° 奇点）")
    print(f"  ② 的 Euler 坐标稳定性：单步 yaw 变化最大 {max_dyaw:.1f}°")
    print("     （正常应约 0.4°；接近奇点时坐标会暴涨到几十度——这才是它不能做融合状态量的原因）")
    print("  结论：同样的陀螺数据，用 Euler 角直接累加会累积严重误差；")
    print("        四元数积分在整个大角度旋转过程中保持亚度级精度。")

    # 画图
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1]})
    a1.plot(tt, np.minimum(err_n, 1e3), color="#f0a030", lw=1.6,
            label="① 朴素 Euler 积分")
    a1.plot(tt, np.where(np.isfinite(err_k), err_k, 1e3), color="#8e44ad",
            lw=1.6, ls="-.", label="② 严格 Euler 运动学")
    a1.plot(tt, np.maximum(err_q, 1e-6), color="#d62728", lw=2.2,
            label="③ 四元数积分")
    a1.set_yscale("log")
    a1.set_ylabel("姿态误差 (°)，对数坐标")
    a1.set_title("大角度三维旋转下，三种姿态积分方式的误差")
    a1.grid(True, which="both", alpha=0.3)
    a1.legend(loc="center right")

    true_pitch = [euler_from_quat(truth(k))[1] for k in range(1, n)]
    a2.plot(tt, true_pitch, color="#1f77b4", lw=1.4, label="真实 pitch")
    a2.axhline(90, color="gray", ls="--", lw=1)
    a2.axhline(-90, color="gray", ls="--", lw=1)
    a2.set_xlabel("时间 (s)")
    a2.set_ylabel("pitch (°)")
    a2.set_title(f"真实 pitch（范围 {min(true_pitch):.1f}° ~ {max(true_pitch):.1f}°，"
                 f"逼近 ±90° 奇点区）", fontsize=10)
    a2.grid(True, alpha=0.3)
    a2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    p2 = os.path.join(OUTDIR, "euler_vs_quat_error.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    print(f"\n  图已保存：{p2}")

    return err_n, err_k, err_q


if __name__ == "__main__":
    experiment_1()
    experiment_2()
    print("\n全部实验完成。")
