# 当前代码复验与 benchmark

本目录使用训练提交 `58cb439` 产出的四个 `best.pt`，在合并上游
`d5afc4b` 后的代码提交 `9eb5076` 上复验。

- `post_merge_validation.csv`：VisDrone 548 张验证图的兼容性复验；
- `latency_rounds.csv`：3 轮原始延迟，每轮 200 次；
- 汇总值见 `../model_comparison_v2.csv`，采用各轮 P50/P95/P99 的中位数；
- 每个模型每轮至少预热 50 次且不少于 2 秒，轮间旋转模型顺序；
- 输入由局部随机生成器和 seed 0 产生，测速不会修改全局 RNG 或梯度模式。

首轮 EsMoE 的 P50 为 11.084 ms，后两轮为 13.418/13.284 ms，说明单轮数字会低估运行间
波动。因此报告保留每个模型的 run min/max，不把毫秒级差异解释为结构收益。
