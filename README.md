# 2026 高教社杯 A 题 药材烘干 — 提交材料

## 目录结构

```
├─ 论文最终版.pdf        最终提交论文(30 页)
├─ code\                最终提交代码(七件套,见下)
├─ 可视化脚本\           分问绘图脚本(公共模块 + 4 个画图脚本)
├─ figs\                全文 15 张数据图(PCHIP 口径,200 dpi)
├─ 结果\                result1–4.xlsx 交付件 + PCHIP 全程解存档 npz + 验证记录
├─ 附件\                附件1.xlsx、附件2.xlsx(题给输入)
└─ COMSOL仿真\          drying_2d.mph + 温度.csv/浓度.csv(二维对照数据)
```

## code\ 七件套与运行顺序

| 顺序 | 文件 | 作用 | 输出 |
|---|---|---|---|
| 1 | 01_插值.py / 01_插值评价.py | 附件 1 五种插值对比 + LOOCV 评价(论文表 6.4.1) | 插值结果_PCHIP.xlsx(后续求解输入,已生成) |
| 2 | 02_问题一求解.py | 问题一求解(显式 FVM,预热段 0–1800 s;含 COMSOL 对照读入) | result1.xlsx、newpage1\ 图 |
| 3 | 03_问题二求解.py / 03_问题三求解.py | 问题二(附录 3 强耦合,Kirchhoff 势,0–3 h)/ 问题三(全程 3 天 + 终点判据) | result2.xlsx、result3.xlsx |
| 4 | 04_问题四求解.py | 问题四求解(物质坐标动边界 + 附录 4) | result4.xlsx、output_q4\ 图 |
| 5 | 05_灵敏度.py | 四张参数扰动表(表 9.1/6.6/6.3.4/6.4.2) | 终端表格(Q3/Q4 部分需数小时) |
| 6 | 06_验证_问题一.py / 06_验证_问题二三.py / 06_验证_问题四.py | 模型验证(退化解析/守恒/收敛/扰动) | 结果\verify_*.json |
| — | lib_q1.py / lib_q2.py / lib_q4.py / lib_q3_output.py | 方法库(灵敏度/验证/绘图共用引擎) | — |

## 可视化脚本

```bash
cd 可视化脚本
python 画图_问题一.py   # fig_q1_results / fig_q1_sens
python 画图_问题二.py   # fig_q2_results / fig_q2_sens / fig_q2_process
python 画图_问题三.py   # q3 首达/剖面/三方案/探针/tornado
python 画图_问题四.py   # q4 半径/首达/剖面/消融/tornado
```

- 公共模块 `plot_common.py`:路径、字体、npz 三级取数(结果\ → math\结果\ 回退 → 现场重算)、save、tornado;
- 输出到 `..\figs\`,四个脚本各独立可跑;
- 当前 15 图已按 PCHIP 口径重绘(含 Q1 灵敏度 ρcp 扰动 bug 修复)。

## 运行要点

1. **先跑 01_插值.py**:生成 `插值结果_PCHIP.xlsx`(四个求解脚本的输入;缺失时问题二/问题四回退线性插值)。当前已生成;
2. 求解脚本输出 result1–4.xlsx 写入上级 `..\结果\`(评审交付件);
3. COMSOL 对照:02_问题一求解.py 读 `..\COMSOL仿真\温度.csv`(5 段 × 1801 点,逗号 CSV 已适配),浓度.csv 同格式备用;
4. Python 环境需 numpy/scipy/pandas/openpyxl/matplotlib。

## figs\ 与论文对应(15 张)

| 图 | 论文位置 | 状态 |
|---|---|---|
| fig_q1_results / fig_q1_sens | 图 3 / 图 6.1 | 替换论文现有 |
| fig_q2_results / fig_q2_sens | 图 6.2 / 图 6.3 | 替换论文现有 |
| q3_sensitivity / q4_sensitivity | 图 6.3(Q3)/ 图 6.4 | 替换论文现有 |
| fig_q2_process、q3_endpoint、q3_profiles、q3_scheme_compare、q3_stab_probe、q4_R_shrink、q4_endpoint、q4_profiles、q4_shrink_effect | 论文缺失 | 待插入(位置与图注见 `..\A题 国一工作归档\插图清单.md`) |

## 待办(提交前)

- 论文数据调整总表(`..\A题 国一工作归档\论文数据调整总表.md`)**尚未应用到论文最终版.pdf** — 表 1/3/8.1/6.4、表 4 两格、表 9.1 三格、±0.02 误差界、消融段、章号顺延(模型评价→七、参考文献→八、补结论)。
