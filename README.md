# 2026 高教社杯 A 题 药材烘干 — 支撑材料

**仓库地址**: https://github.com/wukong0908/CUMCM-2026-A(公开)

论文「中药材热风烘干的机理探究」的评审支撑材料:最终论文 PDF、全部代码、16 张数据图、四问交付件与 COMSOL 二维仿真模型。

**数据口径**:附件 1 采用 PCHIP 插值(4 h 后取末端值恒定);result1–4.xlsx 与论文表 1–表 6 同源。

## 目录结构

```
├─ 论文最终版.pdf            最终提交论文
├─ 代码与仿真\               代码 / 脚本 / 仿真 / 输入数据
│   ├─ code\                 求解 / 灵敏度 / 验证代码
│   │   ├─ 01_插值.py 01_插值评价.py      附件1 插值生成与 LOOCV 评价(表 6.4.1)
│   │   ├─ 02_问题一求解.py               问题一(显式 FVM,0–1800 s)
│   │   ├─ 03_问题二求解.py               问题二(附录 3 强耦合 + Kirchhoff 势,0–3 h)
│   │   ├─ 03_问题三求解.py               问题三(全程 3 天 + 终点判据)
│   │   ├─ 04_问题四求解.py               问题四(物质坐标动边界 + 附录 4)
│   │   ├─ 05_灵敏度.py                  四张扰动表(表 9.1/6.6/6.3.4/6.4.2)
│   │   ├─ 06_验证_问题一/问题二三/问题四.py   模型验证
│   │   └─ lib_q1.py lib_q2.py lib_q4.py lib_q3_output.py   方法库(供灵敏度/验证/绘图 import)
│   ├─ 可视化脚本\            分问绘图脚本
│   │   ├─ plot_common.py                公共模块(路径/npz 取数/save/tornado)
│   │   ├─ 画图_问题一.py                 2 张
│   │   ├─ 画图_问题二.py                 3 张
│   │   ├─ 画图_问题三.py                 5 张
│   │   ├─ 画图_问题四.py                 5 张
│   │   └─ 画图_COMSOL对比.py             数值解 vs COMSOL(表 X/表 Y 配图)
│   ├─ COMSOL仿真\            drying_2d.mph + 温度.csv + 浓度.csv(5 位置 × 1801 时刻)
│   └─ 附件\                  附件1.xlsx、附件2.xlsx(插值输入数据)
└─ 结果\                     全部输出:交付件 / 全程解缓存 / 数据图
    ├─ result1–4.xlsx                四问交付件
    ├─ 插值结果_PCHIP.xlsx            PCHIP 插值产物(四个求解脚本的输入)
    ├─ q2_pchip_arrays.npz           Q2/Q3 全程解
    ├─ q3_pchip_compare.npz          调和/算术界面取法全程解
    ├─ q4_pchip_arrays.npz           Q4 动边界全程解
    ├─ q4_pchip_frozen.npz           Q4 冻结几何消融解
    ├─ verify_q2/q4_results.json     验证输出记录
    └─ figs\                         16 张数据图(PCHIP 口径,200 dpi)
```

## 环境

Python 3.10+,包:numpy、scipy、pandas、openpyxl、matplotlib;绘图字体 Microsoft YaHei/SimHei。COMSOL 模型用 6.3 生成。

## 代码用法

**插值**

```bash
cd 代码与仿真/code
python 01_插值.py          # 生成 插值结果_PCHIP.xlsx(结果\,求解输入)+ 五种方法对比
python 01_插值评价.py       # 打印 CV-RMSE 与光滑性指标(表 6.4.1)
```

**求解**(输出写 `../../结果/result*.xlsx`)

```bash
python 02_问题一求解.py    # result1.xlsx
python 03_问题二求解.py    # result2.xlsx
python 03_问题三求解.py    # result3.xlsx
python 04_问题四求解.py    # result4.xlsx
```

**灵敏度 / 验证**

```bash
python 05_灵敏度.py        # 四张扰动表(问题三/四部分为全程重算)
python 06_验证_问题一.py   # 或 06_验证_问题二三.py / 06_验证_问题四.py
```

**可视化**(输出 `../../结果/figs/`,缺 npz 缓存时自动重算)

```bash
cd 代码与仿真/可视化脚本
python 画图_问题一.py
python 画图_问题二.py
python 画图_问题三.py
python 画图_问题四.py
python 画图_COMSOL对比.py
```

## 结果\figs\ 与论文对应

| 图文件 | 论文位置 |
|---|---|
| fig_q1_results / fig_q1_sens | 图 3 / 图 6.1 |
| fig_q2_results / fig_q2_sens | 图 6.2 / 图 6.3 |
| q3_sensitivity / q4_sensitivity | 图 6.3(Q3)/ 图 6.4 |
| fig_q1_comsol | §6.1.3(4) 表 X/表 Y 配图 |
| fig_q2_process、q3_endpoint、q3_profiles、q3_scheme_compare、q3_stab_probe、q4_R_shrink、q4_endpoint、q4_profiles、q4_shrink_effect | 对应 §6.3 / §6.4 各验证与结果小节 |
