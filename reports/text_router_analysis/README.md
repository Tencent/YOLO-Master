# 文本条件路由：三组独立训练对照与机制分析

本目录汇集 B1 文本条件路由研究的支撑材料。PR 正文介绍阶段成果、实现与主要发现。当前结果显示，文本条件能影响检测输出，但未形成多次训练中的稳定收益。

- [运行配置](experiment_settings.json)：模型、硬软件版本、训练和评测参数，并区分历史评测的配置批量 1 与实际批量 500。
- [统计结果](data/bootstrap_summary.csv)：18 组比较的点差及修正后区间；[新旧统计对照](data/bootstrap_before_after.csv)。
- [测试证据](tests/README.md)：69 项 CPU 测试及依赖、命令和精选输出。
- [实验产物索引](artifacts/README.md)：检查点与预测文件的记录信息。
- [同一检查点的门控与专家选择干预](controlled_interventions/README.md)：24 个实际 batch=1 条件、配对统计、协议与图表。

- 图表：[模块与尺度示意](figures/fig1_method.svg)、[训练路线比较](figures/fig2_training.svg)、[同一模型的文本替换实验](figures/fig3_text.svg)、[专家选择与评分差](figures/fig4_trace.svg)、[受控门控与选择干预](controlled_interventions/figures/fig5_controlled_interventions.svg)、[局部专家机会](controlled_interventions/figures/fig6_local_opportunity.svg)。
