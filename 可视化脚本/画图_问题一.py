# -*- coding: utf-8 -*-
"""问题一可视化:fig_q1_results(图3 四联)+ fig_q1_sens(图6.1 灵敏度 Δ 剖面)。
BC = PCHIP(最终口径);灵敏度含 ρcp 单因子扰动修正。
"""
from plot_common import *

from scipy.interpolate import PchipInterpolator

t_data, Td, Cd = S.read_attach1(S.ATT1)
Tf = PchipInterpolator(t_data, Td, extrapolate=False)
Cf = PchipInterpolator(t_data, Cd, extrapolate=False)
t_out, T1, C1, _, _ = S.solve(Tf, Cf, 1800.0, 0.001, 1.0)
r_cm = np.arange(21) * 0.1
j_sel = [0, 5, 10, 15, 20]
t_pts = [100, 300, 600, 900, 1200, 1500, 1800]
cmap = plt.cm.viridis(np.linspace(0, 0.85, 7))

# ---- 图3 复刻:4 联 ----
fig, ax = plt.subplots(2, 2, figsize=(11, 7.6))
for j in j_sel:
    ax[0, 0].plot(t_out, T1[:, j] - 273.15, lw=1.5, label="r = %.1f cm" % r_cm[j])
for j in j_sel:
    ax[0, 1].plot(t_out, C1[:, j], lw=1.5, label="r = %.1f cm" % r_cm[j])
for k, tp in enumerate(t_pts):
    ax[1, 0].plot(r_cm, T1[tp, :] - 273.15, color=cmap[k], lw=1.5, label="%d s" % tp)
for k, tp in enumerate(t_pts):
    ax[1, 1].plot(r_cm, C1[tp, :], color=cmap[k], lw=1.5, label="%d s" % tp)
ax[0, 0].set(xlabel="时间 t / s", ylabel="温度 T / °C", title="(a) 温度随时间变化")
ax[0, 1].set(xlabel="时间 t / s", ylabel="水分浓度 C / (kg/kg)", title="(b) 水分浓度随时间变化")
ax[1, 0].set(xlabel="到药材中心的距离 r / cm", ylabel="温度 T / °C", title="(c) 横截面温度分布")
ax[1, 1].set(xlabel="到药材中心的距离 r / cm", ylabel="水分浓度 C / (kg/kg)", title="(d) 横截面水分浓度分布")
for a in ax.flat:
    a.grid(alpha=0.3)
    a.legend(fontsize=7.5, ncol=2)
save(fig, "fig_q1_results.png")

# ---- 图6.1 复刻:参数 ±10% 灵敏度 Δ 剖面 ----
P0 = dict(RHO=S.RHO, CP=S.CP, K=S.K, H=S.H, HM=S.HM)


def q1_run(fac, key):
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
    t, T, C, _, _ = S.solve(Tf, Cf, 1800.0, 0.001, 1.0, D_func=Df)
    S.RHO, S.CP, S.K, S.H, S.HM = P0["RHO"], P0["CP"], P0["K"], P0["H"], P0["HM"]
    return T, C


fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for key, lab in (("k", "k ±10%"), ("h", "h ±10%"), ("rcp", "ρcp ±10%")):
    for fac, ls, c in ((1.1, "--", "C1"), (0.9, "-.", "C0")):
        T, _ = q1_run(fac, key)
        ax[0].plot(r_cm, T[1800, :] - T1[1800, :],
                   lw=1.3, ls=ls, color=c, label="%s %+d%%" % (lab.split()[0], int(round((fac - 1) * 100))))
ax[0].axhline(0, ls=":", color="gray", lw=1.0)
for key, lab in (("hm", "h_m ±10%"), ("d0", "D0 ±10%")):
    for fac, ls, c in ((1.1, "--", "C1"), (0.9, "-.", "C0")):
        _, C = q1_run(fac, key)
        ax[1].plot(r_cm, C[1800, :] - C1[1800, :], lw=1.3, ls=ls, color=c,
                   label="%s %+d%%" % (lab.split()[0], int(round((fac - 1) * 100))))
ax[1].axhline(0, ls=":", color="gray", lw=1.0)
ax[0].set(xlabel="到药材中心的距离 r / cm", ylabel="ΔT / °C", title="(a) 热学参数 ±10% 对温度剖面(t=1800 s)")
ax[1].set(xlabel="到药材中心的距离 r / cm", ylabel="ΔC / (kg/kg)", title="(b) 质学参数 ±10% 对水分剖面(t=1800 s)")
for a in ax:
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
save(fig, "fig_q1_sens.png")
print("问题一 2 图完成")
