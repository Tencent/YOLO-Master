# reproduction 复现包

由 `../assemble_reproduction.py` 生成(stage1-4 可复现小文件收集, 大产物 runs/权重不入库)。
- `data_and_env_from_stage1.md`: 数据集来源/许可/SHA 记录 + 环境重建
- `datasets/`: 数据转换脚本
- `configs/`: env 模板 + 三策略训练命令样例 + 环境脚本
- `results/`: matrix + evidence + 四维对照表 + planner 审计
- 已知局限见 stage4/README(seed 稳定性/许可/planner cap<8)
