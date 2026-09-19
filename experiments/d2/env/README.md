# P1 环境快照

正式 VOC 训练日志记录的环境为：

- Ultralytics `8.4.101`
- Python `3.12.14`
- PyTorch `2.11.0+cu128`
- CUDA device 0：NVIDIA GeForce RTX 3090，24124 MiB
- 单卡训练，`amp: false`，`deterministic: true`，`workers: 32`

教师模型和 revision、数据预算及其余解析参数分别记录在
[`training_environment.json`](training_environment.json)、
[`../p1_voc_matrix.csv`](../p1_voc_matrix.csv) 和每个运行的
`results/p1voc_<arm>-s<seed>/resolved_args.yaml` 中。

训练日志没有可靠记录对应的 Git SHA，因此不能从现有证据反推出训练时的精确代码提交；这是复现包的已知限制。当前报告和验证基于 `d2` 分支工作树，最终 PR 应在删除 P2 内容并提交后记录其 HEAD SHA。
