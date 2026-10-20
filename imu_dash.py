#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IMU 综合监控面板 —— 3D 姿态 + 角度曲线 + 数字面板（性能优化版）

性能优化（实测数据）：
    · 整画布重绘          180 ms  ->  5.5 Hz
    · 中文文本每帧重绘      69 ms  （其中 40 ms 纯排版开销）
    · => 改为「背景缓存 + 局部重绘(blit)」，中文标签只画一次，
          每帧只更新 3D 几何、曲线和 ASCII 数字。
    期望 25~50 Hz。

协议（18 字段）：
    D,dt,ax,ay,az,gx10,gy10,gz10,accR10,accP10,gyrR10,gyrP10,mahR10,mahP10,qw,qx,qy,qz

依赖：pip install pyserial matplotlib numpy
运行前请关闭 imu_plot.py / imu_3d.py / 串口助手（串口独占）
"""

import sys
import time
import threading
from collections import deque

import numpy as np
import serial
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

from matplotlib.patches import Polygon

# ==================== 配置 ====================
PORT = "COM5"          # ← 改成你的 CH340 端口号
BAUD = 115200
WINDOW = 300           # 曲线显示最近 300 点（50Hz 下约 6 秒）

LX, LY, LZ = 1.4, 2.2, 0.3      # 模型尺寸：长边沿 Y
WORLD_YAW_DEG = 90.0            # 显示校正：整体绕世界 Z 轴旋转
TICK_MS = 33                    # 渲染间隔（约 30 Hz 上限）

# ==================== 共享数据 ====================
lock = threading.Lock()
latest = {"q": (1.0, 0.0, 0.0, 0.0), "g": (0.0, 0.0, 0.0),
          "dt": 0, "total": 0}
buf = {k: deque(maxlen=WINDOW) for k in ("ar", "ap", "gr", "gp", "mr", "mp")}

# ==================== 串口 ====================
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
except Exception as e:
    print(f"打开串口失败: {e}")
    print("检查：1) PORT 是否与 CH340 一致   2) 是否有别的程序占用串口")
    sys.exit(1)
print(f"已连接 {PORT} @ {BAUD}")


def reader():
    while True:
        try:
            line = ser.readline().decode("ascii", errors="ignore").strip()
        except Exception:
            break
        if not line.startswith("D,"):
            continue
        p = line.split(",")
        if len(p) != 18:
            continue
        try:
            v = [int(x) for x in p[1:]]
        except ValueError:
            continue
        qq = (v[13] / 10000.0, v[14] / 10000.0, v[15] / 10000.0, v[16] / 10000.0)
        n = float(np.linalg.norm(qq))
        if n < 0.5:
            continue
        qq = tuple(c / n for c in qq)
        with lock:
            latest["q"] = qq
            latest["g"] = (v[4] / 10.0, v[5] / 10.0, v[6] / 10.0)
            latest["dt"] = v[0]
            latest["total"] += 1
            buf["ar"].append(v[7] / 10.0)
            buf["ap"].append(v[8] / 10.0)
            buf["gr"].append(v[9] / 10.0)
            buf["gp"].append(v[10] / 10.0)
            buf["mr"].append(v[11] / 10.0)
            buf["mp"].append(v[12] / 10.0)


threading.Thread(target=reader, daemon=True).start()

# ==================== 数学工具 ====================
_yaw = np.radians(WORLD_YAW_DEG)
RZ = np.array([
    [np.cos(_yaw), -np.sin(_yaw), 0.0],
    [np.sin(_yaw),  np.cos(_yaw), 0.0],
    [0.0,          0.0,           1.0],
])


def rotmat(w, x, y, z):
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def quat_to_euler(w, x, y, z):
    r2d = 57.2957795
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)) * r2d
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0)) * r2d
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)) * r2d
    return roll, pitch, yaw


def wrap180(a):
    """把角度包到 (-180, 180]，避免 ±180 处的竖直跳线"""
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


# 曲线以「每路自己的首个样本」为零点，显示相对变化量
_ref = {"r": None, "p": None}


# ==================== 画布 ====================
fig = plt.figure(figsize=(16, 9), dpi=100)
fig.suptitle("IMU 姿态估计 · 综合监控面板（Mahony 四元数）", fontsize=15)

gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1.0],
                      height_ratios=[1.0, 1.0, 0.78],
                      left=0.035, right=0.975, top=0.925, bottom=0.055,
                      wspace=0.13, hspace=0.30)

# ---------- 左：3D（自写正交投影，2D 轴绘制；比 mplot3d 快 ~4 倍） ----------
ELEV_DEG, AZIM_DEG = 22.0, -60.0
_th, _ph = np.radians(AZIM_DEG), np.radians(ELEV_DEG)
E_CAM = np.array([np.cos(_ph)*np.cos(_th), np.cos(_ph)*np.sin(_th), np.sin(_ph)])
R_CAM = np.array([-np.sin(_th), np.cos(_th), 0.0])
U_CAM = np.cross(E_CAM, R_CAM)


def project(P):
    """世界坐标 -> 屏幕坐标 (u, v) 与深度；P 形状 (...,3)"""
    P = np.asarray(P, dtype=float)
    return P @ R_CAM, P @ U_CAM, P @ E_CAM


ax3 = fig.add_subplot(gs[:, 0])
ax3.set_aspect("equal")
ax3.set_xlim(-2.0, 2.0)
ax3.set_ylim(-2.05, 1.75)
ax3.axis("off")

# 世界参考（静态）：竖直虚线 + 地面网格
_u, _v, _ = project(np.array([[0, 0, -1.25], [0, 0, 1.55]]))
ax3.plot(_u, _v, color="gray", ls="--", lw=1.0, alpha=0.7)

for t in np.linspace(-1.2, 1.2, 5):
    for seg in ([[t, -1.2, -1.2], [t, 1.2, -1.2]],
                [[-1.2, t, -1.2], [1.2, t, -1.2]]):
        _u, _v, _ = project(np.array(seg))
        ax3.plot(_u, _v, color="#dcdcdc", lw=0.6)

ax3.text(0.01, 0.97, "板子姿态（四元数直接驱动）", transform=ax3.transAxes,
         fontsize=11, va="top")

V = np.array([
    [-LX/2, -LY/2, -LZ/2], [ LX/2, -LY/2, -LZ/2],
    [ LX/2,  LY/2, -LZ/2], [-LX/2,  LY/2, -LZ/2],
    [-LX/2, -LY/2,  LZ/2], [ LX/2, -LY/2,  LZ/2],
    [ LX/2,  LY/2,  LZ/2], [-LX/2,  LY/2,  LZ/2],
])
FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
         [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
FACE_COLORS = ["#5b7c99", "#e8871e", "#4a90d9", "#3d7ab5", "#2c5f8a", "#6aabe0"]

# 6 个面用 Polygon 补丁，按深度排序（画家算法）
face_polys = []
for i in range(6):
    fp = Polygon(np.zeros((4, 2)), closed=True, facecolor=FACE_COLORS[i],
                 edgecolor="#1a1a1a", linewidth=0.8)
    ax3.add_patch(fp)
    face_polys.append(fp)

AX_VEC = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
AX_COL = ["#d62728", "#2ca02c", "#1f77b4"]
ax_lines = [ax3.plot([0, 1], [0, 1], color=AX_COL[i], lw=2.6, zorder=20)[0]
            for i in range(3)]
ax_texts = [ax3.text(0, 0, "X", color=AX_COL[0], fontsize=11, fontweight="bold", zorder=25),
            ax3.text(0, 0, "Y", color=AX_COL[1], fontsize=11, fontweight="bold", zorder=25),
            ax3.text(0, 0, "Z", color=AX_COL[2], fontsize=11, fontweight="bold", zorder=25)]

_face_order = list(range(6))          # 每帧按深度重排

# ---------- 右：两条曲线 ----------
ax_r = fig.add_subplot(gs[0, 1])
ax_p = fig.add_subplot(gs[1, 1])

C_ACC, C_GYR, C_MAH = "#7fb8e8", "#f0a030", "#d62728"

lines = {}
for axx, ylab, key in ((ax_r, "roll 相对变化 (°)", "roll"),
                       (ax_p, "pitch 相对变化 (°)", "pitch")):
    l1 = axx.plot([], [], color=C_ACC, lw=1.0, label="加速度计")[0]
    l2 = axx.plot([], [], color=C_GYR, lw=1.0, ls="--", label="纯陀螺积分")[0]
    l3 = axx.plot([], [], color=C_MAH, lw=2.0, label="Mahony")[0]
    lines[key] = (l1, l2, l3)
    axx.set_ylabel(ylab, fontsize=10)
    axx.set_ylim(-60, 60)
    axx.set_xlim(0, WINDOW)
    axx.grid(True, alpha=0.3)
    axx.tick_params(labelsize=9)
    axx.legend(loc="upper right", fontsize=9, ncol=3, framealpha=0.85)
    axx.axhline(0, color="gray", lw=0.6, alpha=0.5)

ax_p.set_xlabel("采样点（50 Hz，窗口约 6 秒）", fontsize=10)

dyn2d = list(lines["roll"]) + list(lines["pitch"])

# ---------- 右下：数字面板 ----------
# 中文标签 = 静态（只画一次，进背景缓存）；数字 = 动态（ASCII + 等宽字体，快 4 倍）
ax_n = fig.add_subplot(gs[2, 1])
ax_n.axis("off")

PANEL_ROWS = [0.95, 0.76, 0.57, 0.38, 0.15]
PANEL_LABELS = ["采样总数", "刷新率 / 周期", "roll / pitch / yaw",
                "陀螺 gx / gy / gz", "四元数 w / x / y / z"]

for lab, yy in zip(PANEL_LABELS, PANEL_ROWS):
    ax_n.text(0.00, yy, lab, transform=ax_n.transAxes, va="top",
              fontsize=11, color="#666")

# 每行一个动态数字文本：ASCII + 等宽字体，对齐好且渲染快
val_arts = []
for yy in PANEL_ROWS:
    val_arts.append(
        ax_n.text(0.34, yy, "", transform=ax_n.transAxes, va="top", ha="left",
                  fontsize=11, family="DejaVu Sans Mono", color="#111")
    )
v_total, v_rate, v_rpy, v_gyro, v_quat = val_arts

# 所有"每帧会变"的元素：拍背景时必须先隐藏，否则背景里会印上旧内容 → 重影
dyn3d = [*face_polys, *ax_lines, *ax_texts]
dyn_all = [*dyn3d, *dyn2d, *val_arts]

# ==================== 渲染循环（手动 blit） ====================
renderer = fig.canvas.get_renderer()
_bg = {"bbox": None}
_t_prev = time.time()
_fps = 0.0
_frame = 0


def invalidate_bg():
    _bg["bbox"] = None


def _capture_bg():
    """拍背景：先把会变的元素藏起来，否则重影"""
    for a in dyn_all:
        a.set_visible(False)
    fig.canvas.draw()
    _bg["bbox"] = fig.canvas.copy_from_bbox(fig.bbox)
    for a in dyn_all:
        a.set_visible(True)


def auto_ylim(axx, data):
    """自适应纵轴（带迟滞）；变更时需重拍背景"""
    if len(data) < 30:
        return False
    lo, hi = float(np.min(data)), float(np.max(data))
    pad = max(8.0, (hi - lo) * 0.30)
    new_lo = np.floor((lo - pad) / 20.0) * 20.0
    new_hi = np.ceil((hi + pad) / 20.0) * 20.0
    y0, y1 = axx.get_ylim()
    span = y1 - y0
    if (new_lo < y0) or (new_hi > y1) or \
       ((new_lo > y0 + 0.30 * span) and (new_hi < y1 - 0.30 * span)):
        axx.set_ylim(new_lo, new_hi)
        return True
    return False


def tick():
    global _t_prev, _fps, _frame
    _frame += 1

    with lock:
        w, x, y, z = latest["q"]
        gx, gy, gz = latest["g"]
        dt = latest["dt"]
        total = latest["total"]
        s_ar = list(buf["ar"]); s_ap = list(buf["ap"])
        s_gr = list(buf["gr"]); s_gp = list(buf["gp"])
        s_mr = list(buf["mr"]); s_mp = list(buf["mp"])

    # ---- 帧率 ----
    now = time.time()
    d = now - _t_prev
    _t_prev = now
    if d > 1e-6:
        _fps = 1.0 / d if _fps == 0 else 0.9 * _fps + 0.1 * (1.0 / d)

    # ---- 3D：自写正交投影 + 画家算法排序 ----
    RT = RZ @ rotmat(w, x, y, z)
    Wv = V @ RT.T                         # 8 个顶点转世界坐标
    pu, pv, pd = project(Wv)

    for fi, f in enumerate(FACES):
        face_polys[fi].set_xy(np.column_stack([pu[f], pv[f]]))
    # 远的先画（深度小的先画），近的盖上去 —— 直接用绘制顺序实现画家算法
    _face_order[:] = np.argsort([pd[f].mean() for f in FACES]).tolist()

    for i in range(3):
        tip = RT @ (AX_VEC[i] * 1.55)
        tu, tv, _ = project(tip)
        ax_lines[i].set_data([0.0, tu], [0.0, tv])
        tu2, tv2, _ = project(tip / 1.55 * 1.78)
        ax_texts[i].set_position((tu2, tv2))

    # ---- 曲线：显示相对起始值的变化量 ----
    n = len(s_ar)
    xx = np.arange(n)
    changed = False
    if n >= 2:
        if _ref["r"] is None:
            _ref["r"] = (s_ar[0], s_gr[0], s_mr[0])
            _ref["p"] = (s_ap[0], s_gp[0], s_mp[0])
        ra, rg, rm = _ref["r"]
        pa, pg, pm = _ref["p"]

        au = wrap180(np.asarray(s_ar) - ra)
        gu = wrap180(np.asarray(s_gr) - rg)
        mu = wrap180(np.asarray(s_mr) - rm)
        lines["roll"][0].set_data(xx, au)
        lines["roll"][1].set_data(xx, gu)
        lines["roll"][2].set_data(xx, mu)

        av = wrap180(np.asarray(s_ap) - pa)
        gv = wrap180(np.asarray(s_gp) - pg)
        mv = wrap180(np.asarray(s_mp) - pm)
        lines["pitch"][0].set_data(xx, av)
        lines["pitch"][1].set_data(xx, gv)
        lines["pitch"][2].set_data(xx, mv)

        # 自适应纵轴：每 60 帧才评估一次（每次变更要重拍背景，很贵）
        if _frame % 60 == 0:
            changed |= auto_ylim(ax_r, np.concatenate([au, gu, mu]))
            changed |= auto_ylim(ax_p, np.concatenate([av, gv, mv]))

    # ---- 数字（ASCII + 等宽，渲染快）----
    roll, pitch, yaw = quat_to_euler(w, x, y, z)
    v_total.set_text(f"{total:d}")
    v_rate.set_text(f"{_fps:5.1f} Hz   {dt:5d} us")
    v_rpy.set_text(f"{roll:>7.2f} {pitch:>7.2f} {yaw:>7.2f}")
    v_gyro.set_text(f"{gx:>+6.2f} {gy:>+6.2f} {gz:>+6.2f}")
    v_quat.set_text(f"{w:>+.3f} {x:>+.3f} {y:>+.3f} {z:>+.3f}")

    # ---- 重绘 ----
    if _bg["bbox"] is None or changed:
        _capture_bg()
    fig.canvas.restore_region(_bg["bbox"])
    # 注意：不用 ax3.draw()（TkAgg 下不生效），改用 draw_artist 逐个画
    for fi in _face_order:
        ax3.draw_artist(face_polys[fi])
    for a in ax_lines:
        ax3.draw_artist(a)
    for a in ax_texts:
        ax3.draw_artist(a)
    for ln in dyn2d:
        (ax_r if ln in lines["roll"] else ax_p).draw_artist(ln)
    for art in val_arts:
        ax_n.draw_artist(art)
    fig.canvas.blit(fig.bbox)


fig.canvas.mpl_connect("resize_event", lambda _e: invalidate_bg())

# 先画一次，让背景缓存就绪
tick()

timer = fig.canvas.new_timer(interval=TICK_MS)
timer.add_callback(tick)
timer.start()

plt.show()

timer.stop()
ser.close()
