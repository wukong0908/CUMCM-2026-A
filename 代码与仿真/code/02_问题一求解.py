"""
2026 A题 问题一
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from openpyxl import Workbook


# 0. 路径与开关

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, 'newpage1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 你的 PCHIP 插值文件路径，按实际改
PCHIP_FILE = os.path.join(ROOT, '..', '..', '结果', '插值结果_PCHIP.xlsx')
COMSOL_FILE = os.path.join(ROOT, '..', 'COMSOL仿真', '温度.csv')


# 1. 物理参数（附录2）

R = 0.02                 # 药材半径 m
L = 0.25                 # 药材长度 m（一维模型中约去，仅用于面积/体积计算）
rho = 820.0              # 密度 kg/m^3
cp = 2600.0              # 比热容 J/(kg·K)
k_cond = 0.36            # 导热系数 W/(m·K)
h_T = 25.0               # 对流换热系数 W/(m^2·K)
h_m = 8.0e-7             # 对流传质系数 m/s

T0 = 28.0                # 初始温度 °C
C0 = 2.55                # 初始水分浓度 kg/kg

alpha = k_cond / (rho * cp)


def D_func(C):
    """水分扩散系数 D(C) = 7e-9 * exp(-0.89/C)"""
    C = np.asarray(C, dtype=float)
    return 7.0e-9 * np.exp(-0.89 / np.maximum(C, 1e-8))



# 2. 读取环境数据（PCHIP 插值后的每秒数据）

def load_environment(path=PCHIP_FILE):
    """
    返回 (t, Ta, Ca)。
    支持列名：'时间(s)'/'温度(°C)'/'水分浓度(kg/kg)'
    或 '时间'/'温度'/'水分浓度'
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'找不到文件：{path}')

    df = pd.read_excel(path)
    cols = df.columns.tolist()
    if '时间(s)' in cols:
        t = df['时间(s)'].values.astype(float)
        Ta = df['温度(°C)'].values.astype(float)
        Ca = df['水分浓度(kg/kg)'].values.astype(float)
    else:
        t = df.iloc[:, 0].values.astype(float)
        Ta = df.iloc[:, 1].values.astype(float)
        Ca = df.iloc[:, 2].values.astype(float)

    # 保证按时间排序
    idx = np.argsort(t)
    t, Ta, Ca = t[idx], Ta[idx], Ca[idx]
    return t, Ta, Ca



# 3. 网格与几何量（FVM）

def make_grid(R, N):
    """
    节点：r_j = j * dr, j = 0..N
    控制体：轴心 [0, r_{1/2}]，内部 [r_{j-1/2}, r_{j+1/2}]，表面 [r_{N-1/2}, R]
    返回 r, dr, V, A_face, A_surf
    """
    dr = R / N
    r = np.linspace(0.0, R, N + 1)

    # 控制体体积（乘以单位长度 L，一维方程两侧会约掉，这里保留 L 以保持一致）
    V = np.empty(N + 1)
    V[0] = math.pi * (0.5 * dr) ** 2 * L
    V[1:N] = 2.0 * math.pi * r[1:N] * dr * L
    V[N] = math.pi * (R ** 2 - (R - 0.5 * dr) ** 2) * L

    # 界面面积
    r_face = 0.5 * (r[:-1] + r[1:])
    A_face = 2.0 * math.pi * r_face * L     # 长度 N
    A_surf = 2.0 * math.pi * R * L          # 外表面面积

    return r, dr, V, A_face, A_surf



# 4. 显式 FVM 单步推进

def step_explicit(T, C, Ta, Ca, dt, r, dr, V, A_face, A_surf):
    """
    一步显式 FVM。
    温度场：常物性 k
    水分场：D(C) 用上一时刻 C 计算，界面用调和平均
    表面：Robin 边界
    """
    N = len(T) - 1

    # ---------- 温度场 ----------
    T_new = T.copy()

    # 内部界面导热系数（常物性，直接 k）
    # 界面通量（+r 方向为正）
    def flux_T(j):
        # 界面 j+1/2 的通量
        return -k_cond * A_face[j] * (T[j + 1] - T[j]) / dr

    # 中心节点 j=0：只有右侧界面
    T_new[0] = T[0] + dt / (rho * cp * V[0]) * (-flux_T(0))

    # 内部节点 j=1..N-1
    for j in range(1, N):
        T_new[j] = T[j] + dt / (rho * cp * V[j]) * (flux_T(j - 1) - flux_T(j))

    # 表面节点 j=N：内侧导热 + 对流换热
    q_in = flux_T(N - 1)                            # 从 N-1 流入 N 的通量
    q_conv = h_T * A_surf * (Ta - T[N])             # 对流换热（Ta 为烘房温度）
    T_new[N] = T[N] + dt / (rho * cp * V[N]) * (q_in + q_conv)

    # ---------- 水分场 ----------
    C_new = C.copy()
    D_node = D_func(C)

    # 界面扩散系数：调和平均
    D_face = np.zeros(N)
    for j in range(N):
        Dj, Dj1 = D_node[j], D_node[j + 1]
        D_face[j] = 2.0 * Dj * Dj1 / (Dj + Dj1 + 1e-30)

    def flux_C(j):
        return -D_face[j] * A_face[j] * (C[j + 1] - C[j]) / dr

    # 中心节点
    C_new[0] = C[0] + dt / V[0] * (-flux_C(0))

    # 内部节点
    for j in range(1, N):
        C_new[j] = C[j] + dt / V[j] * (flux_C(j - 1) - flux_C(j))

    # 表面节点
    q_in_C = flux_C(N - 1)
    q_conv_C = h_m * A_surf * (Ca - C[N])
    C_new[N] = C[N] + dt / V[N] * (q_in_C + q_conv_C)

    return T_new, C_new



# 5. 主求解

def solve_problem1(N=20, dt=1.0, t_end=1800.0):
    """
    N=20 时 dr=1mm，节点恰好落在 0,0.1,...,2.0 cm 输出位置。
    """
    t_env, Ta_env, Ca_env = load_environment()
    env_T = lambda t: float(np.interp(t, t_env, Ta_env))
    env_C = lambda t: float(np.interp(t, t_env, Ca_env))

    r, dr, V, A_face, A_surf = make_grid(R, N)
    n_steps = int(round(t_end / dt))
    n_rows = n_steps + 1

    T = np.full(N + 1, T0)
    C = np.full(N + 1, C0)
    T_hist = np.zeros((n_rows, N + 1))
    C_hist = np.zeros((n_rows, N + 1))
    T_hist[0] = T
    C_hist[0] = C

    print(f'N={N}, dr={dr*1000:.3f} mm, dt={dt}s, t_end={t_end}s')

    for n in range(1, n_rows):
        t_now = n * dt
        Ta = env_T(t_now)
        Ca = env_C(t_now)
        T, C = step_explicit(T, C, Ta, Ca, dt, r, dr, V, A_face, A_surf)
        T_hist[n] = T
        C_hist[n] = C

        if n % 300 == 0:
            print(f'  t={t_now:6.0f}s  T_surf={T[-1]:.4f}°C  C_surf={C[-1]:.4f}')

    t_arr = np.arange(n_rows) * dt
    return t_arr, r, T_hist, C_hist



# 6. 输出 result1.xlsx

def write_result1(t_arr, r, T_hist, C_hist):
    wb = Workbook()
    ws_T = wb.active
    ws_T.title = '温度'
    ws_C = wb.create_sheet('水分浓度')

    dist_cm = np.round(r * 100, 1)
    header = ['时间\\到药材中心的距离'] + [float(d) for d in dist_cm]

    for ws, data in [(ws_T, T_hist), (ws_C, C_hist)]:
        ws.append(header)
        for n in range(len(t_arr)):
            row = [int(round(t_arr[n]))] + [round(float(v), 4) for v in data[n]]
            ws.append(row)

    out = os.path.join(ROOT, '..', '..', '结果', 'result1.xlsx')
    wb.save(out)
    print(f'已保存 {out}')



# 7. 打印论文表 1、表 2

def print_tables(t_arr, r, T_hist, C_hist):
    times_req = [100, 300, 600, 900, 1200, 1500, 1800]
    dists_req = [0.0, 0.5, 1.0, 1.5, 2.0]
    dr_cm = r[1] * 100 - r[0] * 100
    idx = [int(round(d / dr_cm)) for d in dists_req]

    print('\n表1 预热平衡阶段药材温度 T(r,t) / °C')
    print(f'{"t/s":>6} | ' + ' | '.join([f'r={d}cm' for d in dists_req]))
    for t in times_req:
        n = int(round(t / (t_arr[1] - t_arr[0])))
        vals = [T_hist[n, i] for i in idx]
        print(f'{t:>6} | ' + ' | '.join([f'{v:10.4f}' for v in vals]))

    print('\n表2 预热平衡阶段水分浓度 C(r,t) / (kg/kg)')
    print(f'{"t/s":>6} | ' + ' | '.join([f'r={d}cm' for d in dists_req]))
    for t in times_req:
        n = int(round(t / (t_arr[1] - t_arr[0])))
        vals = [C_hist[n, i] for i in idx]
        print(f'{t:>6} | ' + ' | '.join([f'{v:10.4f}' for v in vals]))



# 8. 图 1、图 2：温度/水分随时间曲线（r=0, 0.5, 1, 1.5, 2 cm）

def plot_curves(t_arr, r, T_hist, C_hist):
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Songti SC', 'Heiti SC',
                                       'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    dists = [0.0, 0.5, 1.0, 1.5, 2.0]
    dr_cm = r[1] * 100 - r[0] * 100
    idx = [int(round(d / dr_cm)) for d in dists]

    # 温度曲线
    plt.figure(figsize=(7, 4.5))
    for d, i in zip(dists, idx):
        plt.plot(t_arr / 60.0, T_hist[:, i], label=f'r={d} cm')
    plt.xlabel('时间 t / min')
    plt.ylabel('温度 T / °C')
    plt.legend()
    plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_T_time.png'), dpi=300)
    plt.close()

    # 水分曲线
    plt.figure(figsize=(7, 4.5))
    for d, i in zip(dists, idx):
        plt.plot(t_arr / 60.0, C_hist[:, i], label=f'r={d} cm')
    plt.xlabel('时间 t / min')
    plt.ylabel('水分浓度 C / (kg/kg)')
    plt.legend()
    plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_C_time.png'), dpi=300)
    plt.close()

    print(' 已保存 fig_T_time.png, fig_C_time.png')



# 9. 图 3、图 4：二维圆平面温度/水分云图（取 1800 s）

def plot_disk(t_arr, r, T_hist, C_hist, t_snap=1800):
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Songti SC', 'Heiti SC',
                                       'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    n = int(round(t_snap / (t_arr[1] - t_arr[0])))
    T_snap = T_hist[n]
    C_snap = C_hist[n]

    # 构造二维网格
    Ngrid = 400
    x = np.linspace(-R, R, Ngrid)
    y = np.linspace(-R, R, Ngrid)
    X, Y = np.meshgrid(x, y)
    Rr = np.sqrt(X ** 2 + Y ** 2)

    # 用节点值插值到二维网格
    T2 = np.interp(Rr.ravel(), r, T_snap).reshape(Rr.shape)
    C2 = np.interp(Rr.ravel(), r, C_snap).reshape(Rr.shape)

    # 圆外置为 NaN
    mask = Rr > R
    T2[mask] = np.nan
    C2[mask] = np.nan

    # 温度云图
    plt.figure(figsize=(5.5, 5))
    cf = plt.contourf(X * 100, Y * 100, T2, levels=30, cmap='hot')
    plt.colorbar(cf, label='温度 T / °C')
    circle = Circle((0, 0), R * 100, fill=False, color='k', lw=1.0)
    plt.gca().add_patch(circle)
    plt.axis('equal')
    plt.xlabel('x / cm')
    plt.ylabel('y / cm')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_T_disk.png'), dpi=300)
    plt.close()

    # 水分云图
    plt.figure(figsize=(5.5, 5))
    cf = plt.contourf(X * 100, Y * 100, C2, levels=30, cmap='Blues')
    plt.colorbar(cf, label='水分浓度 C / (kg/kg)')
    circle = Circle((0, 0), R * 100, fill=False, color='k', lw=1.0)
    plt.gca().add_patch(circle)
    plt.axis('equal')
    plt.xlabel('x / cm')
    plt.ylabel('y / cm')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_C_disk.png'), dpi=300)
    plt.close()

    print(' 已保存 fig_T_disk.png, fig_C_disk.png')
# -*- coding: utf-8 -*-
"""
从 COMSOL 导出的 txt 中读取温度-时间数据并画线
文件格式：每行两列，注释行以 % 开头；不同位置的数据块以“时间重新回到 0”分隔
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 1. 文件路径

TXT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'COMSOL仿真', '温度.csv')

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outs')
os.makedirs(OUT_DIR, exist_ok=True)



# 2. 读取并解析

def read_data(path):
    """
    返回 list of (t, T)，每段是一个位置的温度-时间序列
    """
    t_all = []
    T_all = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            parts = line.replace(' ', '').split(',')   # COMSOL 导出 csv:时间,值
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
                T = float(parts[1])
            except ValueError:
                continue
            t_all.append(t)
            T_all.append(T)

    t_all = np.array(t_all)
    T_all = np.array(T_all)

    # 按“时间重新回到 0”切分
    blocks = []
    start = 0
    for i in range(1, len(t_all)):
        if t_all[i] == 0.0 and t_all[i - 1] > 0.0:
            blocks.append((t_all[start:i], T_all[start:i]))
            start = i
    blocks.append((t_all[start:], T_all[start:]))

    return blocks



# 3. 画线

def plot_blocks(blocks):
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Songti SC',
                                       'Heiti SC', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (t, T) in enumerate(blocks):
        # 只画前 1800 s（问题一/二关注范围），若想看全部，去掉这一行
        mask = t <= 1800
        ax.plot(t[mask], T[mask], label=f'位置 {i+1}（{len(t)} 点）', lw=1.5)

    ax.set_xlabel('时间 t / s')
    ax.set_ylabel('温度 T / ℃')
    ax.set_title('COMSOL 温度-时间曲线')
    ax.legend()
    ax.grid(True, ls='--', alpha=0.5)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'comsol_T_time.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'已保存 {out}')




# 10. 主程序

if __name__ == '__main__':
    t_arr, r, T_hist, C_hist = solve_problem1(N=20, dt=1.0, t_end=1800.0)

    write_result1(t_arr, r, T_hist, C_hist)
    print_tables(t_arr, r, T_hist, C_hist)

    plot_curves(t_arr, r, T_hist, C_hist)
    plot_disk(t_arr, r, T_hist, C_hist, t_snap=1800)
    blocks = read_data(TXT_FILE)
    print(f'共解析出 {len(blocks)} 段数据')
    for i, (t, T) in enumerate(blocks):
        print(f'  段 {i+1}：{len(t)} 点，时间范围 {t.min():.0f}~{t.max():.0f} s，'
              f'温度范围 {T.min():.4f}~{T.max():.4f} ℃')
    plot_blocks(blocks)
