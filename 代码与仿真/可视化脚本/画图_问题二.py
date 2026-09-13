# -*- coding: utf-8 -*-
"""问题二可视化:fig_q2_results(图6.2 四联)+ fig_q2_sens(图6.3 +10% 剖面)
+ fig_q2_process(全程三阶段)。数据 = PCHIP 全程解 q2_pchip_arrays.npz。"""
from plot_common import *

Z = load_or_make("q2_pchip_arrays.npz", make_q2)
T2, C2 = Z["T"], Z["C"]
r_cm = np.arange(21) * 0.1
j_sel = [0, 5, 10, 15, 20]
t2_pts = [1800, 3600, 5400, 7200, 9000, 10800]
labs2 = ["0.5 h", "1 h", "1.5 h", "2 h", "2.5 h", "3 h"]
cmap2 = plt.cm.viridis(np.linspace(0, 0.85, 6))
t3h = np.arange(10801) / 3600.0
nsteps = 259200
t_h = np.arange(nsteps + 1) / 3600.0

# ---- 图6.2 复刻:4 联(0–3 h) ----
fig, ax = plt.subplots(2, 2, figsize=(11, 7.6))
for j in j_sel:
    ax[0, 0].plot(t3h, T2[:10801, j] - 273.15, lw=1.5, label="r = %.1f cm" % r_cm[j])
for j in j_sel:
    ax[0, 1].plot(t3h, C2[:10801, j], lw=1.5, label="r = %.1f cm" % r_cm[j])
for k, tp in enumerate(t2_pts):
    ax[1, 0].plot(r_cm, T2[tp, :] - 273.15, color=cmap2[k], lw=1.5, label=labs2[k])
for k, tp in enumerate(t2_pts):
    ax[1, 1].plot(r_cm, C2[tp, :], color=cmap2[k], lw=1.5, label=labs2[k])
ax[0, 0].set(xlabel="时间 t / h", ylabel="温度 T / °C", title="(a) 温度随时间变化")
ax[0, 1].set(xlabel="时间 t / h", ylabel="水分浓度 C / (kg/kg)", title="(b) 水分浓度随时间变化")
ax[1, 0].set(xlabel="到药材中心的距离 r / cm", ylabel="温度 T / °C", title="(c) 3 h 横截面温度分布")
ax[1, 1].set(xlabel="到药材中心的距离 r / cm", ylabel="水分浓度 C / (kg/kg)", title="(d) 3 h 横截面水分浓度分布")
for a in ax.flat:
    a.grid(alpha=0.3)
    a.legend(fontsize=7.5, ncol=2)
save(fig, "fig_q2_results.png")

# ---- 图6.3 复刻:关键参数 +10% 对 3 h 剖面的影响 ----
Pq0 = dict(Q.P)
n2 = 10800
t_n, T_n, C_n = S.read_attach1(S.ATT1)
T_inf3, C_inf3 = Q.bc_arrays(t_n, T_n, C_n, n2, 1.0)
Tb, Cb, _, _ = Q.solve(T_inf3, C_inf3, 10800.0, 0.001, 1.0, flux_scheme="kirchhoff")


def q2_run(fac, key):
    Q.P.update(Pq0)
    if key in ("cp", "k"):
        Q.P[key + "0"] *= fac
        Q.P[key + "1"] *= fac
    else:
        Q.P[key] *= fac
    T, C, _, _ = Q.solve(T_inf3, C_inf3, 10800.0, 0.001, 1.0, flux_scheme="kirchhoff")
    Q.P.update(Pq0)
    return T, C


fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for key, lab in (("D0", "D0"), ("A_C", "A_C"), ("A_T", "A_T"), ("HM", "h_m")):
    _, C = q2_run(1.1, key)
    ax[0].plot(r_cm, C[10800, :] - Cb[10800, :], lw=1.4, label=lab + " +10%")
for key, lab in (("H", "h"), ("cp", "c_p"), ("k", "k")):
    T, _ = q2_run(1.1, key)
    ax[1].plot(r_cm, T[10800, :] - Tb[10800, :], lw=1.4, label=lab + " +10%")
ax[0].axhline(0, ls=":", color="gray", lw=1.0)
ax[1].axhline(0, ls=":", color="gray", lw=1.0)
ax[0].set(xlabel="到药材中心的距离 r / cm", ylabel="ΔC / (kg/kg)", title="(a) 扩散系数参数 +10% 对 3 h 水分剖面")
ax[1].set(xlabel="到药材中心的距离 r / cm", ylabel="ΔT / °C", title="(b) 换热/热物性参数 +10% 对 3 h 温度剖面")
for a in ax:
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
save(fig, "fig_q2_sens.png")

# ---- 全程三阶段 ----
T_inf, C_inf = Q.bc_arrays(t_n, T_n, C_n, nsteps, 1.0)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(t_h, C2[:, 0], lw=1.6, label="中心 C(0,t)")
ax[0].plot(t_h, C2[:, -1], lw=1.6, label="表面 C(R,t)")
ax[0].plot(t_h, C_inf, lw=1.2, ls="--", color="gray", label="烘房 C∞(t)")
ax[0].axhline(0.15, ls=":", color="red", lw=1.2)
ax[0].axvline(57.21, ls=":", color="red", lw=1.2)
ax[0].annotate("C = 0.15", xy=(66, 0.155), color="red", fontsize=9)
ax[0].annotate("t = 57.21 h", xy=(43, 2.25), color="red", fontsize=9)
ax[0].set(xlabel="时间 t / h", ylabel="水分浓度 C / (kg/kg)", title="(a) 水分浓度全程")
ax[1].plot(t_h, T2[:, 0] - 273.15, lw=1.6, label="中心 T(0,t)")
ax[1].plot(t_h, T2[:, -1] - 273.15, lw=1.6, label="表面 T(R,t)")
ax[1].plot(t_h, T_inf - 273.15, lw=1.2, ls="--", color="gray", label="烘房 T∞(t)")
ax[1].set(xlabel="时间 t / h", ylabel="温度 T / °C", title="(b) 温度全程")
for a in ax:
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
save(fig, "fig_q2_process.png")
print("问题二 3 图完成")
