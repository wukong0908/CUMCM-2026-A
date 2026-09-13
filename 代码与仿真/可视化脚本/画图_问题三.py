# -*- coding: utf-8 -*-
"""问题三可视化:q3_scheme_compare(假停滞三方案)+ q3_endpoint(判据首达)
+ q3_stab_probe(稳定性探针)+ q3_profiles(表5 剖面)+ q3_sensitivity(tornado)。
数据 = PCHIP 全程解。"""
from plot_common import *

Z = load_or_make("q2_pchip_arrays.npz", make_q2)
C2 = Z["C"]
nsteps = 259200
t_h = np.arange(nsteps + 1) / 3600.0
r_cm = np.arange(21) * 0.1
cmax = C2.max(axis=1)
hit = np.where(cmax <= 0.15)[0]
n_hit = hit[0]
Zc = load_or_make("q3_pchip_compare.npz", make_q3cmp)
Ch, Ca = Zc["Ch"], Zc["Ca"]

# ---- 假停滞三方案 ----
fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.plot(t_h, C2[:, 0], lw=1.8, label="Kirchhoff 势(本文)")
ax.plot(t_h, Ch[:, 0], lw=1.6, ls="--", label="调和平均")
ax.plot(t_h, Ca[:, 0], lw=1.6, ls="-.", label="算术平均")
ax.axhline(0.15, ls=":", color="red", lw=1.2)
ax.annotate("假停滞:3 天内无法完成", xy=(46, 0.275), fontsize=9)
ax.annotate("t = 57.21 h", xy=(48.5, 0.095), color="red", fontsize=9)
ax.annotate("t = 56.18 h(偏早)", xy=(49.5, 0.175), fontsize=9, color="C2")
ax.set(xlabel="时间 t / h", ylabel="中心水分浓度 C(0,t) / (kg/kg)",
       title="界面通量取法对干燥终点的影响", xlim=(0, 72), ylim=(0, 2.7))
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
save(fig, "q3_scheme_compare.png")

# ---- 判据首达 ----
fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.plot(t_h, cmax, lw=1.6, color="C0")
ax.axhline(0.15, ls=":", color="red", lw=1.2)
ax.axvline(n_hit / 3600.0, ls=":", color="red", lw=1.2)
ax.annotate("C = 0.15", xy=(64, 0.156), color="red", fontsize=9)
ax.annotate("首达 t = 205952 s = 57.21 h", xy=(n_hit / 3600.0 + 0.3, 0.9),
            color="red", fontsize=9)
ax.set(xlabel="时间 t / h", ylabel="全场最大水分浓度 max C / (kg/kg)",
       title="烘干终点判据的首达时刻", xlim=(0, 72), ylim=(0, 2.7))
ax.grid(alpha=0.3)
i0 = int(round(n_hit - 0.45 * 3600))
i1 = int(round(n_hit + 0.15 * 3600))
axin = inset_axes(ax, width="44%", height="44%", loc="lower right", borderpad=1.4)
axin.plot(t_h[i0:i1], cmax[i0:i1], lw=1.4, color="C0")
axin.axhline(0.15, ls=":", color="red", lw=1.0)
axin.axvline(n_hit / 3600.0, ls=":", color="red", lw=1.0)
axin.plot((n_hit - 60) / 3600.0, cmax[n_hit - 60], "o", ms=4, color="C3")
axin.annotate("前 60 s:%.4f" % cmax[n_hit - 60],
              xy=((n_hit - 60) / 3600.0 + 0.015, cmax[n_hit - 60] + 0.0005),
              fontsize=8, color="C3")
axin.tick_params(labelsize=7)
axin.grid(alpha=0.3)
mark_inset(ax, axin, loc1=2, loc2=4, fc="none", ec="0.6")
save(fig, "q3_endpoint.png")

# ---- 稳定性探针 ----
def probe(dt, n=1000):
    T_inf_c = np.full(n + 1, 323.315)
    C_inf_c = np.full(n + 1, 0.04986)
    T0 = 323.315 + 1e-6 * np.where(np.arange(21) % 2 == 0, 1.0, -1.0)
    T, C, _, _ = Q.solve(T_inf_c, C_inf_c, n * dt, 0.001, dt,
                         freeze_C=True, C_init=0.15, T_init=T0)
    return np.abs(T - 323.315).max(axis=1)


d1 = probe(1.0)
d2 = probe(2.0)
d1s = np.maximum(d1, 1e-30)
d2s = np.clip(np.nan_to_num(d2, nan=1e30, posinf=1e30), None, 1e30)
fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.semilogy(np.arange(len(d1s)), d1s, lw=1.6, label="Δt = 1 s(x = 0.215)")
ax.semilogy(np.arange(len(d2s)), d2s, lw=1.6, ls="--", label="Δt = 2 s(x = 0.429)")
ax.axhline(1e-6, ls=":", color="gray", lw=1.0)
ax.annotate("初始扰动 $10^{-6}$ K", xy=(20, 1.3e-6), fontsize=8, color="gray")
ax.annotate("立即衰减", xy=(30, 1e-10), fontsize=9, color="C0")
ax.annotate("指数发散", xy=(420, 1e10), fontsize=9, color="C1")
ax.set(xlabel="步数 n", ylabel="全场最大偏差 max |T − T∞| / K",
       title="极轴稳定性探针(冻结 C = 0.15,锯齿扰动)")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9)
save(fig, "q3_stab_probe.png")

# ---- 表5 剖面 ----
t_sel_h = [6, 12, 18, 24, 30, 36, 42, 48, 54]
cmap = plt.cm.viridis(np.linspace(0, 0.9, 10))
fig, ax = plt.subplots(figsize=(6.8, 4.4))
for k, th in enumerate(t_sel_h):
    ax.plot(r_cm, C2[int(th * 3600), :], color=cmap[k], lw=1.5, label="%d h" % th)
ax.plot(r_cm, C2[n_hit, :], color=cmap[9], lw=2.0, ls="--", label="57.21 h(终点)")
ax.axhline(0.15, ls=":", color="red", lw=1.2)
ax.annotate("C = 0.15", xy=(1.55, 0.153), color="red", fontsize=9)
ax.set(xlabel="到药材中心的距离 r / cm", ylabel="水分浓度 C / (kg/kg)",
       title="不同时刻的水分浓度剖面(表 5)")
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, ncol=2)
save(fig, "q3_profiles.png")

# ---- tornado(公共模块) ----
tornado("q3_sensitivity.png", "问题三:参数 ±10% 对烘干终点的影响", 57.21,
        ["$k$", "$c_p$", "$h$", "$h_m$", "$D_0$", "$A_C$", "$A_T$"],
        [57.21, 57.20, 57.22, 57.95, 62.84, 45.38, 23.13],
        [57.21, 57.22, 57.20, 56.63, 52.63, 72.0, 72.0],
        [False, False, False, False, True, True, True],
        [False, False, False, False, True, True, True], (20, 76))
print("问题三 5 图完成")
