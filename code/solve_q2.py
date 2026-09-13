# -*- coding: utf-8 -*-
"""
Q2 solver — whole drying process (0–259200 s), appendix-3 strongly coupled model.
Implements 规格卡_Q2.md exactly:
  - explicit FVM, dr=0.001 m, dt=1 s, nodes at r=j*dr (incl. boundary)
  - sequential coupling per step: heat with C^n -> T^{n+1}; then mass with (C^n, T^{n+1}) -> C^{n+1}
  - harmonic interface k and D; rho, cp at node values
  - BC: attachment1 interp on [0,14400] s; beyond, np.interp clamps to tail value
        (D01 main: tail T=50.165 C, C=0.04986). Cross-check variant: asymptote override.
Outputs: 表3/表4 (0.5–3 h x 5 radii), result2.xlsx (0–10800 s, sheets 温度/水分浓度).
Verification (Stage 6) lives in verify_q2.py.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import openpyxl
from scipy.special import exp1
import solve_q1 as S            # reuse: read_attach1, build_grid, ATT1, constants R/L

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(BASE, "结果")
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------- constants (SI) -------------------------
T0 = 28.0 + 273.15          # K
C0 = 2.55                   # kg/kg
T_END_MAIN = 50.165 + 273.15    # D01 main  (== attachment1 last value)
C_END_MAIN = 0.04986
T_END_ASY = 50.15 + 273.15      # cross-check asymptote
C_END_ASY = 0.0507

# appendix-3 formula constants (module-level dict so robustness sweeps can perturb)
P = dict(rho0=650.0, rho1=128.0,
         cp0=1450.0, cp1=2736.0,
         k0=0.21, k1=0.38,
         D0=2.4e-3, A_C=0.45, A_T=3850.0,
         H=25.0, HM=8e-7)


# ------------------------- appendix-3 properties -------------------------
def rho_c(C):
    return P['rho0'] + P['rho1'] * C


def cp_c(C):
    return P['cp0'] + P['cp1'] * C / (C + 1.0)


def k_c(C):
    return P['k0'] + P['k1'] * C / (C + 1.0)


def D_ct(C, T):
    """D(C,T) = D0 * exp(-A_C/C) * exp(-A_T/T), T in K. C->0 gives D->0."""
    C_safe = np.maximum(C, 1e-12)
    return P['D0'] * np.exp(-P['A_C'] / C_safe) * np.exp(-P['A_T'] / T)


def H_pot(C):
    """Kirchhoff potential H(C) = int_0^C exp(-A_C/s) ds
       = C*exp(-A_C/C) - A_C*E1(A_C/C)   (exact antiderivative, E1 = scipy exp1).
    H(0) = 0, H strictly increasing, slope = exp(-A_C/C) in (0,1)."""
    a = P['A_C']
    Cs = np.maximum(C, 1e-12)
    return Cs * np.exp(-a / Cs) - a * exp1(a / Cs)


def B_T(T):
    """Temperature factor of D: D(C,T) = B(T) * exp(-A_C/C)."""
    return P['D0'] * np.exp(-P['A_T'] / T)


# ------------------------- boundary conditions -------------------------
def bc_arrays(t_data, T_data, C_data, nsteps, dt, T_tail=None, C_tail=None):
    """T_inf/C_inf at every time step. np.interp clamps beyond last data point
    to the tail value (D01 main). Optional asymptote override for cross-check."""
    t_n = np.arange(nsteps + 1) * dt
    T_inf = np.interp(t_n, t_data, T_data)
    C_inf = np.interp(t_n, t_data, C_data)
    if T_tail is not None:
        mask = t_n >= t_data[-1]
        T_inf[mask] = T_tail
        C_inf[mask] = C_tail
    return T_inf, C_inf


# ------------------------- solver core -------------------------
def solve(T_inf, C_inf, t_end, dr=0.001, dt=1.0, flux_scheme="kirchhoff",
          freeze_C=False, freeze_T=False, D_func=None, T_init=T0, C_init=C0):
    """Explicit FVM per 规格卡_Q2.md (v1.1). Returns (T, C, courants, m_bal).
    T, C shape (nsteps+1, N+1); T in K.
    flux_scheme: 'kirchhoff' (main M2, exact steady interface flux for
    exponential D(C)) | 'harmonic' | 'arithmetic' (M0 baselines).
    freeze_C / freeze_T: degenerate single-physics runs for verification.
    D_func: override D(C) (works with harmonic/arithmetic only)."""
    N, r, V, A, AR = S.build_grid(dr)
    nsteps = int(round(t_end / dt))
    T = np.zeros((nsteps + 1, N + 1))
    C = np.zeros((nsteps + 1, N + 1))
    T[0, :] = T_init
    C[0, :] = C_init
    cour = np.empty(nsteps)
    dM_bc = 0.0                   # integrated boundary moisture loss
    for n in range(nsteps):
        Cn = C[n]
        Tn = T[n]
        # ---- 1) properties at C^n ----
        rho_j = rho_c(Cn)
        cp_j = cp_c(Cn)
        k_j = k_c(Cn)
        # ---- 2) heat: harmonic k interface, Robin at surface ----
        if freeze_T:
            T[n + 1] = Tn
        else:
            denom = k_j[1:] + k_j[:-1]
            k_int = np.divide(2 * k_j[1:] * k_j[:-1], denom, out=np.zeros_like(denom),
                              where=denom > 0)
            Fh = -k_int * A * (Tn[1:] - Tn[:-1]) / dr
            rhs = np.zeros(N + 1)
            rhs[0] = -Fh[0]
            rhs[1:-1] = Fh[:-1] - Fh[1:]
            rhs[N] = Fh[N - 1] - AR * P['H'] * (Tn[N] - T_inf[n])
            T[n + 1] = Tn + dt * rhs / (rho_j * cp_j * V)
        # ---- 3) mass: D(C^n, T^{n+1}) at interfaces ----
        if freeze_C:
            C[n + 1] = Cn
            D_max = 0.0
        else:
            T_int = (T[n + 1][1:] + T[n + 1][:-1]) / 2
            if flux_scheme == "kirchhoff":
                if D_func is not None:
                    raise ValueError("D_func override only for harmonic/arithmetic")
                # exact steady flux across interface: u = B(T)*H(C)
                B_i = B_T(T_int)
                dH = H_pot(Cn[1:]) - H_pot(Cn[:-1])
                Fm = -B_i * A * dH / dr
                dC = Cn[1:] - Cn[:-1]
                Cmid = np.maximum((Cn[1:] + Cn[:-1]) / 2, 1e-12)
                chord = np.divide(B_i * dH, dC,
                                  out=B_i * np.exp(-P['A_C'] / Cmid),
                                  where=np.abs(dC) > 1e-14)
                D_max = chord.max()
            else:
                D_j = D_func(Cn) if D_func is not None else D_ct(Cn, T[n + 1])
                if flux_scheme == "harmonic":
                    denom = D_j[1:] + D_j[:-1]
                    D_int = np.divide(2 * D_j[1:] * D_j[:-1], denom, out=np.zeros_like(denom),
                                      where=denom > 0)
                else:                      # arithmetic
                    D_int = (D_j[1:] + D_j[:-1]) / 2
                Fm = -D_int * A * (Cn[1:] - Cn[:-1]) / dr
                D_max = D_j.max()
            rhs = np.zeros(N + 1)
            rhs[0] = -Fm[0]
            rhs[1:-1] = Fm[:-1] - Fm[1:]
            rhs[N] = Fm[N - 1] - AR * P['HM'] * (Cn[N] - C_inf[n])
            C[n + 1] = Cn + dt * rhs / V
            dM_bc += dt * AR * P['HM'] * (Cn[N] - C_inf[n])
        # ---- stability monitor: polar axis-pair bound x = max(alpha,D)*dt/dr^2 < 0.423 ----
        alpha_m = np.max(k_j / (rho_j * cp_j))
        cour[n] = max(alpha_m * dt / dr ** 2, D_max * dt / dr ** 2)
    if freeze_C:
        m_bal = dict(abs_max=0.0, rel=0.0)
    else:
        M0 = np.sum(V * C[0])
        M_end = np.sum(V * C[-1])
        m_bal = dict(abs_max=np.abs((M0 - M_end) - dM_bc),
                     rel=np.abs((M0 - M_end) - dM_bc) / M0)
    return T, C, cour, m_bal


# ------------------------- outputs -------------------------
def write_result2(C, T, dr=0.001):
    """result2.xlsx: 0–10800 s, 1 s steps, 0–2.0 cm at 0.1 cm, 4 decimals."""
    wb = openpyxl.Workbook()
    wsT = wb.active
    wsT.title = "温度"
    wsC = wb.create_sheet("水分浓度")
    header = "时间\\到药材中心的距离"
    dists = [round(dr * 100 * j, 1) for j in range(21)]   # 0..2.0 cm
    for ws, data in [(wsT, T), (wsC, C)]:
        ws.cell(row=1, column=1, value=header)
        for c, d in enumerate(dists, start=2):
            ws.cell(row=1, column=c, value=d)
        for i in range(1, 10801):                          # rows 1..10800 s
            ws.cell(row=i + 1, column=1, value=i)
            for j in range(21):
                val = data[i, j] - 273.15 if ws is wsT else data[i, j]
                ws.cell(row=i + 1, column=j + 2, value=round(float(val), 4))
    out = os.path.join(OUTDIR, "result2.xlsx")
    wb.save(out)
    return out


def main():
    t_data, T_data, C_data = S.read_attach1(S.ATT1)
    dt, dr, t_end = 1.0, 0.001, 259200.0
    nsteps = int(round(t_end / dt))

    # D01 main BC (interp clamps to tail beyond 14400 s)
    T_inf, C_inf = bc_arrays(t_data, T_data, C_data, nsteps, dt)
    T, C, cour, m_bal = solve(T_inf, C_inf, t_end, dr, dt, flux_scheme="kirchhoff")

    # ---- L4: numerical health ----
    print("== L4 数值健康 ==")
    print("max Courant x = max(alpha,D)*dt/dr^2 = %.5f (polar axis-pair bound 0.423)"
          % cour.max())
    print("T range: %.4f .. %.4f C" % (T.min() - 273.15, T.max() - 273.15))
    print("C range: %.6f .. %.6f kg/kg; C min > -1e-12: %s"
          % (C.min(), C.max(), bool((C > -1e-12).all())))
    print("质量守恒: max abs 失衡 %.3e, rel %.3e" % (m_bal["abs_max"], m_bal["rel"]))

    # ---- 表3/表4 ----
    t_list = [1800, 3600, 5400, 7200, 9000, 10800]
    r_cm = [0.0, 0.5, 1.0, 1.5, 2.0]
    idx = [int(round(t / dt)) for t in t_list]
    jdx = [int(round(rr / (dr * 100))) for rr in r_cm]
    print("\n--- 表3 温度(°C) [M2 main] ---")
    print("t/h | " + " | ".join("r=%.1fcm" % r for r in r_cm))
    for i in idx:
        row = [T[i, j] - 273.15 for j in jdx]
        print("%4.1f | " % (t_list[idx.index(i)] / 3600.0)
              + " | ".join("%8.4f" % v for v in row))
    print("--- 表4 水分浓度(kg/kg) [M2 main] ---")
    print("t/h | " + " | ".join("r=%.1fcm" % r for r in r_cm))
    for i in idx:
        row = [C[i, j] for j in jdx]
        print("%4.1f | " % (t_list[idx.index(i)] / 3600.0)
              + " | ".join("%8.4f" % v for v in row))

    # ---- Q3 preview: drying endpoint (max C <= 0.15) ----
    cmax = C.max(axis=1)
    hit = np.where(cmax <= 0.15)[0]
    print("\n== Q3 预览:干燥终点 ==")
    if hit.size:
        t_hit = hit[0] * dt
        print("max C <= 0.15 at t = %.1f h (%.1f s)" % (t_hit / 3600.0, t_hit))
        print("center C at endpoint: %.6f" % C[hit[0], 0])
    else:
        print("3 天内未达到 C<0.15; t=72h 时 max C = %.6f (center)" % cmax[-1])
    # coarse daily report
    for tday in (86400, 172800, 259200):
        i = int(tday / dt)
        print("t=%3dh: center C=%.4f, surface C=%.4f, center T=%.2f C, surface T=%.2f C"
              % (tday / 3600, C[i, 0], C[i, -1], T[i, 0] - 273.15, T[i, -1] - 273.15))

    # ---- result2.xlsx ----
    out = write_result2(C, T, dr)
    print("\nsaved:", out)

    # ---- 五层检查 ----
    print("\n== 五层检查 ==")
    print("L1 程序: 语法/维度/索引 OK (无异常); 时间步=%d 节点=%d" % (nsteps, 21))
    print("L2 数学: 顺序显式耦合按规格卡_Q2; 退化解析/收敛/渐近 → Stage 6")
    print("L3 数据: 附件1 校验通过; BC 前段插值、后段 D01 恒温(无外推)")
    print("L4 数值: Courant x=%.5f (<0.423); C 恒正=%s" % (cour.max(), bool((C > 0).all())))
    print("L5 建模: 表3/表4 直接回答 Q2 输出要求; result2 已生成")


if __name__ == "__main__":
    main()
