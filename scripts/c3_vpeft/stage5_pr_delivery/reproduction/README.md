# reproduction 复现包

由 `../assemble_reproduction.py` 生成(stage1-4 可复现小文件收集, 大产物 runs/权重不入库)。
- `data_and_env_from_stage1.md`: 环境重建与数据集准备流程(stage1 README)
- `datasets/`: 数据转换脚本 + `manifest.md`(来源/许可/统计/SHA 留档位置)
- `configs/`: env 模板 + 三策略训练命令样例 + 环境脚本
- `results/`: matrix + evidence + 四维对照表 + planner 审计
- `limitations.md`: 已知局限/许可风险/seed 稳定性(手写, 不随脚本覆盖)
