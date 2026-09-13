# -*- coding: utf-8 -*-
"""可视化公共模块:路径、字体、npz 三级取数、save。

四个画图脚本(画图_问题一~四.py)共用本模块。
取数顺序:结果\ → math\结果\(只读回退) → 现场重算并存 结果\。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))      # 可视化脚本\
CODE = os.path.join(HERE, "..", "code")                 # 最终提交代码
RES = os.path.join(HERE, "..", "结果")                  # 结果与 npz 缓存
FIG = os.path.join(HERE, "..", "figs")                  # 输出图目录
RES2 = os.path.join(HERE, "..", "..", "结果")           # math\结果 只读缓存回退
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, CODE)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Patch

import lib_q1 as S
import lib_q2 as Q
import lib_q4 as Q4

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_or_make(name, make):
    """取数顺序:结果\ → math\结果\ → 现场重算并存 结果\。"""
    for d in (RES, RES2):
        p = os.path.join(d, name)
        if os.path.exists(p):
            print("load npz:", p)
            return np.load(p)
    print("MISSING %s -> 现场重算(约几十分钟)..." % name)
    z = make()
    np.savez_compressed(os.path.join(RES, name), **z)
    print("saved npz:", os.path.join(RES, name))
    return z


def make_q2():
    tn, Tn, Cn = S.read_attach1(S.ATT1)
    T_inf, C_inf = Q.bc_arrays(tn, Tn, Cn, 259200, 1.0)
    T, C, _, _ = Q.solve(T_inf, C_inf, 259200.0, 0.001, 1.0, flux_scheme="kirchhoff")
    return dict(T=T, C=C)


def make_q3cmp():
    tn, Tn, Cn = S.read_attach1(S.ATT1)
    T_inf, C_inf = Q.bc_arrays(tn, Tn, Cn, 259200, 1.0)
    Th, Ch, _, _ = Q.solve(T_inf, C_inf, 259200.0, 0.001, 1.0, flux_scheme="harmonic")
    Ta, Ca, _, _ = Q.solve(T_inf, C_inf, 259200.0, 0.001, 1.0, flux_scheme="arithmetic")
    return dict(Ch=Ch, Ca=Ca)


def make_q4():
    tn, Tn, Cn = S.read_attach1(S.ATT1)
    n = int(round(Q4.T_END / 1.0))
    T_inf, C_inf = Q4.bc_arrays(tn, Tn, Cn, n, 1.0)
    T, C, Rarr, Rdot, xmax, led, cmax = Q4.solve_A(T_inf, C_inf, Q4.T_END, 1.0)
    return dict(T=T, C=C, Rarr=Rarr, Rdot=Rdot, xmax=xmax, led=led)


def make_q4frozen():
    tn, Tn, Cn = S.read_attach1(S.ATT1)
    n = int(round(Q4.T_END / 1.0))
    T_inf, C_inf = Q4.bc_arrays(tn, Tn, Cn, n, 1.0)
    T, C, Rarr, Rdot, xmax, led, cmax = Q4.solve_A(T_inf, C_inf, Q4.T_END, 1.0,
                                                   freeze_R=True)
    return dict(C=C)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=200)
    plt.close(fig)
    print("saved", name)


def tornado(fname, title, nominal, params, lo, hi, lo_clip, hi_clip, xlim):
    """横向 tornado 图:左右延伸量 = 相对名义终点(小时)。"""
    y = np.arange(len(params))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for yi, p, l, h, lc, hc in zip(y, params, lo, hi, lo_clip, hi_clip):
        if l < nominal:
            ax.barh(yi, l - nominal, left=nominal, color="C0", alpha=0.75, height=0.55)
        if h > nominal:
            ax.barh(yi, h - nominal, left=nominal, color="C1", alpha=0.75, height=0.55)
        if lc and l < nominal:
            ax.annotate("%.1f" % l, xy=(l - 0.4, yi), ha="right", va="center",
                        fontsize=8, color="C0")
        if hc and h > nominal:
            lab = ">72" if (h >= 72) else "%.1f" % h
            ax.annotate(lab, xy=(h + 0.4, yi), ha="left", va="center",
                        fontsize=8, color="C1")
        if not hc and not lc:
            ax.annotate("%.2f" % h, xy=(h + 0.2, yi), ha="left", va="center",
                        fontsize=8, color="C1")
    ax.axvline(nominal, ls=":", color="red", lw=1.2)
    ax.annotate("名义 %.2f h" % nominal, xy=(nominal, len(params) - 0.25),
                color="red", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(params, fontsize=9)
    ax.set_xlabel("干燥终点 t_dry / h")
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.grid(alpha=0.3, axis="x")
    ax.legend(handles=[Patch(color="C0", alpha=0.75, label="−10%"),
                       Patch(color="C1", alpha=0.75, label="+10%")],
              fontsize=8, loc="lower right")
    save(fig, fname)
