# -*- coding: utf-8 -*-
"""问题一 数值解 vs COMSOL 二维仿真对比图(fig_q1_comsol.png)。

数据:COMSOL仿真\温度.csv / 浓度.csv(5 块 × 1801 点,块序 = r 0→2 cm);
数值解:lib_q1 显式 FVM,PCHIP 边界(表 1 同口径)。
"""
from plot_common import *

import csv
from scipy.interpolate import PchipInterpolator

COMSOL_DIR = os.path.join(HERE, "..", "COMSOL仿真")


def read_comsol(path):
    """COMSOL 导出 csv:% 注释行跳过,逗号两列(时间,值),按时间回零切块。"""
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.replace(" ", "").split(",")
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    t = np.array([r[0] for r in rows])
    v = np.array([r[1] for r in rows])
    blocks = []
    start = 0
    for i in range(1, len(t)):
        if t[i] == 0.0 and t[i - 1] > 0.0:
            blocks.append((t[start:i], v[start:i]))
            start = i
    blocks.append((t[start:], v[start:]))
    return blocks


# ---- 数值解(PCHIP 边界,与表 1 同口径) ----
t_data, Td, Cd = S.read_attach1(S.ATT1)
Tf = PchipInterpolator(t_data, Td, extrapolate=False)
Cf = PchipInterpolator(t_data, Cd, extrapolate=False)
t_out, T1, C1, _, _ = S.solve(Tf, Cf, 1800.0, 0.001, 1.0)

TB = read_comsol(os.path.join(COMSOL_DIR, "温度.csv"))
CB = read_comsol(os.path.join(COMSOL_DIR, "浓度.csv"))
print("COMSOL 块数:温度 %d / 浓度 %d" % (len(TB), len(CB)))
r_labs = ["0.0", "0.5", "1.0", "1.5", "2.0"]
cmap = plt.cm.viridis(np.linspace(0, 0.85, 5))
j_sel = [0, 5, 10, 15, 20]

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for k, (jc, rr) in enumerate(zip(j_sel, r_labs)):
    tc, vc = TB[k]
    mask = tc <= 1800
    ax[0].plot(t_out, T1[:, jc] - 273.15, color=cmap[k], lw=1.6,
               label="数值 r = %s cm" % rr)
    ax[0].plot(tc[mask], vc[mask], color=cmap[k], lw=1.4, ls="--",
               label="COMSOL r = %s cm" % rr)
for k, (jc, rr) in enumerate(zip(j_sel, r_labs)):
    tc, vc = CB[k]
    mask = tc <= 1800
    ax[1].plot(t_out, C1[:, jc], color=cmap[k], lw=1.6,
               label="数值 r = %s cm" % rr)
    ax[1].plot(tc[mask], vc[mask], color=cmap[k], lw=1.4, ls="--",
               label="COMSOL r = %s cm" % rr)
ax[0].set(xlabel="时间 t / s", ylabel="温度 T / °C", title="(a) 温度场:数值解 vs COMSOL")
ax[1].set(xlabel="时间 t / s", ylabel="水分浓度 C / (kg/kg)", title="(b) 水分场:数值解 vs COMSOL")
for a in ax:
    a.grid(alpha=0.3)
    a.legend(fontsize=7, ncol=2)
save(fig, "fig_q1_comsol.png")

# ---- 附:表 X/表 Y 误差指标(MAE/RMSE/最大误差) ----
print("===== 误差指标(数值 vs COMSOL,1800 s 内全时刻) =====")
print("r/cm    T-MAE/°C   T-RMSE/°C   T-Max/°C | C-MAE      C-RMSE      C-Max")
for k, (jc, rr) in enumerate(zip(j_sel, r_labs)):
    tc, vc = TB[k]
    mask = (tc <= 1800) & (tc >= 0)
    tt = tc[mask]
    num_T = np.interp(tt, t_out, T1[:, jc] - 273.15)
    err_T = num_T - vc[mask]
    tc2, vc2 = CB[k]
    mask2 = (tc2 <= 1800) & (tc2 >= 0)
    num_C = np.interp(tc2[mask2], t_out, C1[:, jc])
    err_C = num_C - vc2[mask2]
    print("%-7s %10.4f %10.4f %10.4f | %10.2e %10.2e %10.2e"
          % (rr, np.abs(err_T).mean(), np.sqrt((err_T ** 2).mean()), np.abs(err_T).max(),
             np.abs(err_C).mean(), np.sqrt((err_C ** 2).mean()), np.abs(err_C).max()))
print("完成")
