"""
灵敏度分析脚本(统一版):生成论文四张参数扰动表。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.interpolate import PchipInterpolator
import lib_q1 as S
import lib_q2 as Q
import lib_q4 as Q4


def q1_table():
    """表9.1:5 参数 ±10%,t=1800 s 最大变化。"""
    t, Td, Cd = S.read_attach1(S.ATT1)
    Tp = PchipInterpolator(t, Td, extrapolate=False)
    Cp = PchipInterpolator(t, Cd, extrapolate=False)
    P0 = dict(RHO=S.RHO, CP=S.CP, K=S.K, H=S.H, HM=S.HM)

    def run(fac, key):
        o = dict(P0)
        if key == "k":
            o["K"] *= fac
        elif key == "h":
            o["H"] *= fac
        elif key == "rcp":
            o["RHO"] *= fac          # ρcp 整体 ±10%,只动 ρ
        elif key == "hm":
            o["HM"] *= fac
        S.RHO, S.CP, S.K, S.H, S.HM = o["RHO"], o["CP"], o["K"], o["H"], o["HM"]
        Df = (lambda C: 7e-9 * fac * np.exp(-0.89 / C)) if key == "d0" else S.D_nonlin
        _, T, C, _, _ = S.solve(Tp, Cp, 1800.0, 0.001, 1.0, D_func=Df)
        S.RHO, S.CP, S.K, S.H, S.HM = P0["RHO"], P0["CP"], P0["K"], P0["H"], P0["HM"]
        return T, C

    _, T0, C0, _, _ = S.solve(Tp, Cp, 1800.0, 0.001, 1.0)
    print("\n===== 表9.1 问题一关键参数 ±10%(t=1800 s) =====")
    print("参数       温度最大变化/°C   水分浓度最大变化/(kg/kg)")
    for key, lab in (("k", "k"), ("h", "h"), ("rcp", "ρcp"), ("hm", "hm"), ("d0", "D0")):
        dT, dC = 0.0, 0.0
        for fac in (1.1, 0.9):
            T, C = run(fac, key)
            dT = max(dT, np.abs(T[1800] - T0[1800]).max())
            dC = max(dC, np.abs(C[1800] - C0[1800]).max())
        print("%-10s %.4f              %.4f" % (lab, dT, dC))


def q2_table():
    """表6.6:7 参数 ±10%,表3(0.5 h 表面温)与表4(3 h 中心/表面)影响。"""
    t, Td, Cd = S.read_attach1(S.ATT1)
    n2 = 10800
    T_inf, C_inf = Q.bc_arrays(t, Td, Cd, n2, 1.0)
    T0, C0, _, _ = Q.solve(T_inf, C_inf, 10800.0, 0.001, 1.0, flux_scheme="kirchhoff")
    P0 = dict(Q.P)
    print("\n===== 表6.6 问题二关键参数 ±10%(3 h 输出) =====")
    print("参数       表3影响(0.5h表面温/°C)      表4影响(3h中心/表面 C)")
    for key, lab in (("D0", "D0"), ("A_C", "AC"), ("A_T", "AT"),
                     ("HM", "hm"), ("H", "h"), ("cp", "cp"), ("k", "k")):
        rows = []
        for fac in (0.9, 1.1):
            Q.P.update(P0)
            if key in ("cp", "k"):
                Q.P[key + "0"] *= fac
                Q.P[key + "1"] *= fac
            else:
                Q.P[key] *= fac
            T, C, _, _ = Q.solve(T_inf, C_inf, 10800.0, 0.001, 1.0, flux_scheme="kirchhoff")
            Q.P.update(P0)
            dT = T[1800, -1] - T0[1800, -1]
            dCc = C[10800, 0] - C0[10800, 0]
            dCs = C[10800, -1] - C0[10800, -1]
            rows.append((dT, dCc, dCs))
        print("%-10s %+.4f / %+.4f °C      %+.3f/%+.3f ; %+.3f/%+.3f"
              % (lab, rows[0][0], rows[1][0], rows[0][1], rows[1][1], rows[0][2], rows[1][2]))


def q3_table():
    """表6.3.4:7 参数 ±10%,全程终点。"""
    t, Td, Cd = S.read_attach1(S.ATT1)
    n = 259200
    T_inf, C_inf = Q.bc_arrays(t, Td, Cd, n, 1.0)
    P0 = dict(Q.P)
    print("\n===== 表6.3.4 问题三关键参数 ±10%(烘干终点/h) =====")
    print("参数       −10%     +10%")
    t0 = time.time()
    for key, lab in (("D0", "D0"), ("A_C", "AC"), ("A_T", "AT"),
                     ("HM", "hm"), ("H", "h"), ("cp", "cp"), ("k", "k")):
        ends = []
        for fac in (0.9, 1.1):
            Q.P.update(P0)
            if key in ("cp", "k"):
                Q.P[key + "0"] *= fac
                Q.P[key + "1"] *= fac
            else:
                Q.P[key] *= fac
            T, C, _, _ = Q.solve(T_inf, C_inf, 259200.0, 0.001, 1.0, flux_scheme="kirchhoff")
            Q.P.update(P0)
            h = np.where(C.max(axis=1) <= 0.15)[0]
            ends.append(("%.2f" % (h[0] / 3600.0)) if h.size else ">72")
        print("%-10s %-8s %-8s   (%.1f min)" % (lab, ends[0], ends[1], (time.time() - t0) / 60.0))


def q4_table():
    """表6.4.2:7 参数 ±10%,动边界终点。"""
    t, Td, Cd = S.read_attach1(S.ATT1)
    n = int(round(Q4.T_END / 1.0))
    T_inf, C_inf = Q4.bc_arrays(t, Td, Cd, n, 1.0)
    P0 = dict(Q4.P)
    print("\n===== 表6.4.2 问题四关键参数 ±10%(烘干终点/h) =====")
    print("参数       −10%     +10%")
    t0 = time.time()
    for key, lab in (("D0", "D0"), ("A_C", "AC"), ("A_T", "AT"),
                     ("HM", "hm"), ("H", "h"), ("cp", "cp"), ("k", "k")):
        ends = []
        for fac in (0.9, 1.1):
            Q4.P.update(P0)
            if key in ("cp", "k"):
                Q4.P[key + "0"] *= fac
                Q4.P[key + "1"] *= fac
            else:
                Q4.P[key] *= fac
            T, C, Rarr, Rdot, xmax, led, cmax = Q4.solve_A(T_inf, C_inf, Q4.T_END, 1.0)
            Q4.P.update(P0)
            h = np.where(cmax <= 0.15)[0]
            ends.append(("%.2f" % (h[0] / 3600.0)) if h.size else ">72")
        print("%-10s %-8s %-8s   (%.1f min)" % (lab, ends[0], ends[1], (time.time() - t0) / 60.0))


if __name__ == "__main__":
    q1_table()
    q2_table()
    q3_table()
    q4_table()
    print("\nDONE")
