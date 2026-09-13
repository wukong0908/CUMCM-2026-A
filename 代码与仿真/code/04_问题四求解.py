# -*- coding: utf-8 -*-
"""
2026 A题 问题四：考虑尺寸变化的药材烘干时间确定
物质坐标动边界显式 FVM + Kirchhoff 势界面通量

输出：
  - result4.xlsx
  - 表 6 打印
  - 可视化图
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openpyxl import Workbook

# ============================================================
# 0. 路径
# ============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, 'output_q4')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PCHIP_FILE = os.path.join(ROOT, '..', '..', '结果', '插值结果_PCHIP.xlsx')   # 附件1插值结果
R_FILE     = os.path.join(ROOT, '..', '附件', '附件2.xlsx')            # 附件2 半径数据

# ============================================================
# 1. 物理参数
# ============================================================
R0 = 0.02          # 初始半径 m
L  = 0.25          # 长度 m
T0 = 301.15        # 初始温度 K（对应 28 °C）
C0 = 2.55          # 初始水分浓度 kg/kg

h_T = 25.0         # 对流换热系数 W/(m^2·K)
h_m = 8.0e-7       # 对流传质系数 m/s

C_DRY = 0.15       # 烘干终点判据 kg/kg

DT = 1.0           # 时间步长 s
NXI = 20           # 物质坐标控制体数
DXI = 1.0 / NXI    # Δξ = 0.05

T_END = 259200.0   # 3 天

# ============================================================
# 2. 附录 4 物性
# ============================================================
def rho_fun(C):
    return 760.0 + 90.0 * C

def cp_fun(C):
    return 1850.0 + 2150.0 * C / (C + 1.0)

def k_fun(C):
    return 0.12 + 0.20 * C / (C + 1.0)

def D_fun(C, T):
    """T 单位 K"""
    C = np.maximum(C, 1e-8)
    return 4.2e-4 * np.exp(-0.30 / C) * np.exp(-3850.0 / T)

# ============================================================
# 3. Kirchhoff 势
# ============================================================
def H_fun(C):
    """
    H(C) = ∫_0^C e^{-0.30/s} ds
         = C e^{-0.30/C} - 0.30 E1(0.30/C)
    """
    C = np.asarray(C, dtype=float)
    C = np.maximum(C, 1e-12)
    try:
        from scipy.special import exp1
        return C * np.exp(-0.30 / C) - 0.30 * exp1(0.30 / C)
    except ImportError:
        return C * np.exp(-0.30 / C)

# ============================================================
# 4. 环境边界
# ============================================================
def load_environment(path=PCHIP_FILE):
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
    idx = np.argsort(t)
    return t[idx], Ta[idx], Ca[idx]

def make_env_funcs(t_env, T_env, C_env, t_switch=4 * 3600.0,
                   T_hold=50.165, C_hold=0.04986):
    def T_inf(t):
        if t <= t_switch:
            return float(np.interp(t, t_env, T_env)) + 273.15
        else:
            return T_hold + 273.15

    def C_inf(t):
        if t <= t_switch:
            return float(np.interp(t, t_env, C_env))
        else:
            return C_hold

    return T_inf, C_inf

# ============================================================
# 5. 半径数据 R(t)
# ============================================================
def load_radius(path=R_FILE):
    """
    读取附件2：时间(s)，半径(cm)
    返回 t_R (s), R_R (m)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'找不到文件：{path}')
    df = pd.read_excel(path)
    cols = df.columns.tolist()
    if '时间' in cols and '半径' in cols:
        t = df['时间'].values.astype(float)
        R = df['半径'].values.astype(float)
    else:
        t = df.iloc[:, 0].values.astype(float)
        R = df.iloc[:, 1].values.astype(float)
    idx = np.argsort(t)
    t = t[idx]
    R = R[idx] / 100.0   # cm -> m
    return t, R

def make_radius_funcs(t_R, R_R):
    """
    分段线性插值 R(t)，分段差分给出 Rdot(t)
    """
    def R_fun(t):
        if t <= t_R[0]:
            return R_R[0]
        if t >= t_R[-1]:
            return R_R[-1]
        return float(np.interp(t, t_R, R_R))

    def Rdot_fun(t):
        if t <= t_R[0]:
            return 0.0
        if t >= t_R[-1]:
            return 0.0
        i = np.searchsorted(t_R, t) - 1
        i = max(0, min(i, len(t_R) - 2))
        dt = t_R[i + 1] - t_R[i]
        if dt <= 0:
            return 0.0
        return (R_R[i + 1] - R_R[i]) / dt

    return R_fun, Rdot_fun

# ============================================================
# 6. 网格与几何（物质坐标）
# ============================================================
def make_grid(NXI, L):
    """
    物质坐标 ξ ∈ [0,1]，节点 ξ_j = j Δξ
    返回：
      xi      : 节点坐标 (N+1)
      V_bar   : 单位 R^2 下的控制体体积
      Axi_face: 单位 R 下的界面面积
      Axi_surf: 单位 R 下的外表面面积
    """
    dxi = 1.0 / NXI
    xi = np.linspace(0.0, 1.0, NXI + 1)

    V_bar = np.empty(NXI + 1)
    V_bar[0] = math.pi * (0.5 * dxi) ** 2 * L
    V_bar[1:NXI] = 2.0 * math.pi * xi[1:NXI] * dxi * L
    V_bar[NXI] = math.pi * (1.0 - (1.0 - 0.5 * dxi) ** 2) * L

    xi_face = 0.5 * (xi[:-1] + xi[1:])
    Axi_face = 2.0 * math.pi * xi_face * L
    Axi_surf = 2.0 * math.pi * 1.0 * L

    return xi, dxi, V_bar, Axi_face, Axi_surf

# ============================================================
# 7. 单步显式推进（物质坐标，动边界）
# ============================================================
def step_explicit(T, C, Ta, Ca, R, Rdot, dt,
                  xi, dxi, V_bar, Axi_face, Axi_surf):
    """
    物质坐标显式 FVM 单步
    温度与水分均含：
      - 扩散项：乘 R^{-2}
      - 收缩对流项：ξ Rdot / R
    """
    N = len(T) - 1

    # ---------- 几何量 ----------
    V = V_bar * R ** 2
    A_face = Axi_face * R
    A_surf = Axi_surf * R

    # ---------- 温度场 ----------
    T_new = T.copy()

    # 节点热导
    k_node = k_fun(C)
    k_face = np.zeros(N)
    for j in range(N):
        kj, kj1 = k_node[j], k_node[j + 1]
        k_face[j] = 2.0 * kj * kj1 / (kj + kj1 + 1e-30)

    rho_cp = rho_fun(C) * cp_fun(C)

    def flux_T(j):
        # 扩散通量，物质坐标下分母 R Δξ
        return -k_face[j] * A_face[j] * (T[j + 1] - T[j]) / (R * dxi)

    # 扩散推进
    T_diff = T.copy()
    T_diff[0] = T[0] + dt / (rho_cp[0] * V[0]) * (-flux_T(0))
    for j in range(1, N):
        T_diff[j] = T[j] + dt / (rho_cp[j] * V[j]) * (flux_T(j - 1) - flux_T(j))
    q_in = flux_T(N - 1)
    q_conv = h_T * A_surf * (Ta - T[N])
    T_diff[N] = T[N] + dt / (rho_cp[N] * V[N]) * (q_in + q_conv)

    # 收缩对流项：ξ Rdot / R * ∂T/∂ξ
    if abs(Rdot) > 1e-30:
        conv_T = np.zeros(N + 1)
        for j in range(1, N):
            dTdxi = (T_diff[j + 1] - T_diff[j - 1]) / (2.0 * dxi)
            conv_T[j] = xi[j] * Rdot / R * dTdxi
        # 表面取后向差分
        dTdxi_N = (T_diff[N] - T_diff[N - 1]) / dxi
        conv_T[N] = xi[N] * Rdot / R * dTdxi_N
        # 轴心 ξ=0，对流项为 0
        conv_T[0] = 0.0
        T_new = T_diff + dt * conv_T
    else:
        T_new = T_diff

    # ---------- 水分场 ----------
    C_new = C.copy()

    H_node = H_fun(C)
    T_K = T_new
    B_node = 4.2e-4 * np.exp(-3850.0 / T_K)
    B_face = 0.5 * (B_node[:-1] + B_node[1:])

    def flux_C(j):
        return -B_face[j] * A_face[j] * (H_node[j + 1] - H_node[j]) / (R * dxi)

    C_diff = C.copy()
    C_diff[0] = C[0] + dt / V[0] * (-flux_C(0))
    for j in range(1, N):
        C_diff[j] = C[j] + dt / V[j] * (flux_C(j - 1) - flux_C(j))
    q_in_C = flux_C(N - 1)
    q_conv_C = h_m * A_surf * (Ca - C[N])
    C_diff[N] = C[N] + dt / V[N] * (q_in_C + q_conv_C)

    # 收缩对流项
    if abs(Rdot) > 1e-30:
        conv_C = np.zeros(N + 1)
        for j in range(1, N):
            dCdxi = (C_diff[j + 1] - C_diff[j - 1]) / (2.0 * dxi)
            conv_C[j] = xi[j] * Rdot / R * dCdxi
        dCdxi_N = (C_diff[N] - C_diff[N - 1]) / dxi
        conv_C[N] = xi[N] * Rdot / R * dCdxi_N
        conv_C[0] = 0.0
        C_new = C_diff + dt * conv_C
    else:
        C_new = C_diff

    return T_new, C_new

# ============================================================
# 8. 全程求解
# ============================================================
def solve_full(dt=DT, t_end=T_END, store_every=60):
    t_env, T_env, C_env = load_environment()
    T_inf, C_inf = make_env_funcs(t_env, T_env, C_env)

    t_R, R_R = load_radius()
    R_fun, Rdot_fun = make_radius_funcs(t_R, R_R)

    xi, dxi, V_bar, Axi_face, Axi_surf = make_grid(NXI, L)

    n_steps = int(round(t_end / dt))
    n_store = n_steps // store_every + 1

    T = np.full(NXI + 1, T0)
    C = np.full(NXI + 1, C0)

    t_hist = np.zeros(n_store)
    T_hist = np.zeros((n_store, NXI + 1))
    C_hist = np.zeros((n_store, NXI + 1))
    R_hist = np.zeros(n_store)
    C_max_hist = np.zeros(n_store)

    t_hist[0] = 0.0
    T_hist[0] = T
    C_hist[0] = C
    R_hist[0] = R_fun(0.0)
    C_max_hist[0] = C.max()

    t_dry = None
    k_store = 1

    print(f'[INFO] 问题四求解：NXI={NXI}, Δξ={dxi:.3f}, Δt={dt}s, t_end={t_end}s')

    for n in range(1, n_steps + 1):
        t_now = n * dt
        Ta = T_inf(t_now)
        Ca = C_inf(t_now)
        R = R_fun(t_now)
        Rdot = Rdot_fun(t_now)

        T, C = step_explicit(T, C, Ta, Ca, R, Rdot, dt,
                             xi, dxi, V_bar, Axi_face, Axi_surf)

        C_max = C.max()

        if n % store_every == 0:
            t_hist[k_store] = t_now
            T_hist[k_store] = T
            C_hist[k_store] = C
            R_hist[k_store] = R
            C_max_hist[k_store] = C_max
            k_store += 1

        if t_dry is None and C_max <= C_DRY:
            t_dry = t_now
            print(f'[INFO] 烘干终点命中 t_dry = {t_dry:.1f} s = {t_dry/3600:.2f} h')

        if n % 3600 == 0:
            print(f'  t={t_now/3600:6.2f} h  R={R*100:.3f} cm  '
                  f'C_max={C_max:.4f}  C_center={C[0]:.4f}  C_surf={C[-1]:.4f}')

    t_hist = t_hist[:k_store]
    T_hist = T_hist[:k_store]
    C_hist = C_hist[:k_store]
    R_hist = R_hist[:k_store]
    C_max_hist = C_max_hist[:k_store]

    return dict(t_hist=t_hist, xi=xi, T_hist=T_hist, C_hist=C_hist,
                R_hist=R_hist, C_max_hist=C_max_hist, t_dry=t_dry,
                N=NXI, dxi=dxi)

# ============================================================
# 9. 物质坐标 -> 物理半径插值
# ============================================================
def interp_at_r(C_row, xi, R, r_target):
    """
    给定物质坐标下的 C(ξ)，物理半径 r_target，返回 C(r_target)
    物质坐标与物理半径关系：r = ξ R
    """
    xi_target = r_target / R
    if xi_target < 0 or xi_target > 1:
        return np.nan
    return float(np.interp(xi_target, xi, C_row))

# ============================================================
# 10. 打印表 6
# ============================================================
def print_table6(res):
    t_hist = res['t_hist']
    xi = res['xi']
    C_hist = res['C_hist']
    R_hist = res['R_hist']
    t_dry = res['t_dry']

    times_h = [6, 12, 18, 24, 30, 36, 42, 48]
    r_list_cm = [0.0, 0.5, 1.0]

    print('\n' + '=' * 78)
    print('表 6  药材烘干过程的水分浓度（kg/kg）')
    print('=' * 78)

    header = f'{"时间/h":>8} | ' + ' | '.join([f'{"r=" + str(d) + "cm":>8}'
                                                for d in r_list_cm]) + ' | 药材表面'
    print(header)
    print('-' * len(header))

    for th in times_h:
        t_target = th * 3600.0
        n = int(np.argmin(np.abs(t_hist - t_target)))
        R = R_hist[n]
        vals = []
        for r_cm in r_list_cm:
            r = r_cm / 100.0
            vals.append(interp_at_r(C_hist[n], xi, R, r))
        c_surf = C_hist[n, -1]
        line = f'{th:>8} | ' + ' | '.join([f'{v:8.4f}' if np.isfinite(v) else '    ---'
                                            for v in vals]) + f' | {c_surf:8.4f}'
        print(line)

    if t_dry is not None:
        n = int(np.argmin(np.abs(t_hist - t_dry)))
        R = R_hist[n]
        vals = []
        for r_cm in r_list_cm:
            r = r_cm / 100.0
            vals.append(interp_at_r(C_hist[n], xi, R, r))
        c_surf = C_hist[n, -1]
        line = f'{"结束":>8} | ' + ' | '.join([f'{v:8.4f}' if np.isfinite(v) else '    ---'
                                                for v in vals]) + f' | {c_surf:8.4f}'
        print(line)
        print('-' * len(header))
        print(f'烘干所需时间：{t_dry/3600:.2f} h = {t_dry/3600/24:.2f} 天')

    print('=' * 78 + '\n')

# ============================================================
# 11. 保存 result4.xlsx
# ============================================================
def write_result4(res, out_path='result4.xlsx'):
    t_hist = res['t_hist']
    xi = res['xi']
    C_hist = res['C_hist']
    R_hist = res['R_hist']
    t_dry = res['t_dry']

    wb = Workbook()
    ws = wb.active
    ws.title = '水分浓度'

    # 输出距离：0 ~ 1.2 cm，步长 0.1 cm
    dists_cm = [round(0.1 * k, 1) for k in range(0, 13)]

    header = ['时间\\到药材中心的距离'] + [float(d) for d in dists_cm] + ['药材表面']
    ws.append(header)

    R_min = R_hist.min()
    print(f'[INFO] 最小半径 R_min = {R_min*100:.3f} cm')

    for n in range(len(t_hist)):
        R = R_hist[n]
        row = [int(round(t_hist[n]))]
        for d_cm in dists_cm:
            r = d_cm / 100.0
            v = interp_at_r(C_hist[n], xi, R, r)
            row.append(round(float(v), 4) if np.isfinite(v) else None)
        row.append(round(float(C_hist[n, -1]), 4))
        ws.append(row)

    if t_dry is not None:
        n = int(np.argmin(np.abs(t_hist - t_dry)))
        R = R_hist[n]
        row = [int(round(t_dry))]
        for d_cm in dists_cm:
            r = d_cm / 100.0
            v = interp_at_r(C_hist[n], xi, R, r)
            row.append(round(float(v), 4) if np.isfinite(v) else None)
        row.append(round(float(C_hist[n, -1]), 4))
        ws.append(row)

    wb.save(out_path)
    print(f'[INFO] 已保存 {out_path}')

# ============================================================
# 12. 可视化
# ============================================================
def plot_endpoint(res):
    t_hist = res['t_hist']
    C_max = res['C_max_hist']
    t_dry = res['t_dry']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(t_hist / 3600.0, C_max, 'b-', lw=1.8, label='max C')
    ax.axhline(C_DRY, color='r', ls='--', lw=1.2, label='判据 0.15')
    if t_dry is not None:
        ax.axvline(t_dry / 3600.0, color='k', ls=':', lw=1.2,
                   label=f't_dry={t_dry/3600:.2f} h')
    ax.set_xlabel('时间 t / h')
    ax.set_ylabel('全场最大水分浓度 / (kg/kg)')
    ax.set_title('全场最大水分浓度全程演化')
    ax.legend()
    ax.grid(True, ls='--', alpha=0.5)

    ax = axes[1]
    if t_dry is not None:
        mask = (t_hist >= t_dry - 3600 * 0.6) & (t_hist <= t_dry + 60)
        ax.plot(t_hist[mask] / 3600.0, C_max[mask], 'b-', lw=1.8)
        ax.axhline(C_DRY, color='r', ls='--', lw=1.2)
        ax.axvline(t_dry / 3600.0, color='k', ls=':', lw=1.2)
        ax.set_xlabel('时间 t / h')
        ax.set_ylabel('max C')
        ax.set_title('首达时刻放大窗')
        ax.grid(True, ls='--', alpha=0.5)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'q4_endpoint.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'[INFO] 已保存 {out}')


def plot_profiles(res):
    t_hist = res['t_hist']
    xi = res['xi']
    C_hist = res['C_hist']
    R_hist = res['R_hist']

    times_h = [6, 12, 18, 24, 30, 36, 42, 48]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for th in times_h:
        t_target = th * 3600.0
        n = int(np.argmin(np.abs(t_hist - t_target)))
        R = R_hist[n]
        r_phys = xi * R * 100.0  # cm
        ax.plot(r_phys, C_hist[n], lw=1.5, label=f'{th} h')

    ax.axhline(C_DRY, color='r', ls='--', lw=1.2, label='判据 0.15')
    ax.set_xlabel('到药材中心的距离 r / cm')
    ax.set_ylabel('水分浓度 C / (kg/kg)')
    ax.set_title('不同时刻水分浓度剖面（物理半径）')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, ls='--', alpha=0.5)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'q4_profiles.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'[INFO] 已保存 {out}')


def plot_shrink(res):
    t_hist = res['t_hist']
    R_hist = res['R_hist']
    C_hist = res['C_hist']
    C_max = res['C_max_hist']

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t_hist / 3600.0, R_hist * 100, 'k-', lw=1.8, label='R(t) / cm')
    ax.set_xlabel('时间 t / h')
    ax.set_ylabel('半径 R / cm')
    ax.set_title('半径收缩与中心水分浓度')
    ax.grid(True, ls='--', alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(t_hist / 3600.0, C_hist[:, 0], 'b-', lw=1.5, label='C_center')
    ax2.plot(t_hist / 3600.0, C_max, 'r--', lw=1.5, label='C_max')
    ax2.set_ylabel('水分浓度 / (kg/kg)')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'q4_shrink_effect.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'[INFO] 已保存 {out}')


# ============================================================
# 13. 主程序
# ============================================================
if __name__ == '__main__':
    res = solve_full(dt=DT, t_end=T_END, store_every=60)

    print_table6(res)
    write_result4(res, out_path=os.path.join(OUTPUT_DIR, 'result4.xlsx'))

    plot_endpoint(res)
    plot_profiles(res)
    plot_shrink(res)

    print('\n[INFO] 问题四全部完成。')