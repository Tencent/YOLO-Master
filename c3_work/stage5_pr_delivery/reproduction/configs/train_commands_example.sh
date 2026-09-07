#!/bin/bash
# 同预算三策略命令样例(从正式单元 resolved command 提取, seed/device 可改)

## neu
# frozen_backbone
/mnt/pfs/zitao_team/baiyouheng/conda_envs/yolo_master/bin/yolo detect train model=/mnt/pfs/zitao_team/baiyouheng/research/TX_yolo/YOLO-Master/YOLO-Master-EsMoE-N.pt data=/mnt/pfs/zitao_team/baiyouheng/datasets/NEU-DET-yolo/neu_det_yolo_v3/neu_det.yaml epochs=100 batch=8 imgsz=640 device=1 seed=2024 amp=false project=/mnt/pfs/zitao_team/baiyouheng/research/TX_yolo/YOLO-Master/c3_work/stage3_matrix/runs/neu_frozen_backbone_s2024/train name=frozen_backbone lora_save_adapters=True lora_r=0 freeze=11

# full_sft
/mnt/pfs/zitao_team/baiyouheng/conda_envs/yolo_master/bin/yolo detect train model=/mnt/pfs/zitao_team/baiyouheng/research/TX_yolo/YOLO-Master/YOLO-Master-EsMoE-N.pt data=/mnt/pfs/zitao_team/baiyouheng/datasets/NEU-DET-yolo/neu_det_yolo_v3/neu_det.yaml epochs=100 batch=8 imgsz=640 device=1 seed=2024 amp=false project=/mnt/pfs/zitao_team/baiyouheng/research/TX_yolo/YOLO-Master/c3_work/stage3_matrix/runs/neu_full_sft_s2024/train name=full_sft lora_save_adapters=True lora_r=0 freeze=0

# vpeft
/mnt/pfs/zitao_team/baiyouheng/conda_envs/yolo_master/bin/yolo detect train model=/mnt/pfs/zitao_team/baiyouheng/research/TX_yolo/YOLO-Master/YOLO-Master-EsMoE-N.pt data=/mnt/pfs/zitao_team/baiyouheng/datasets/NEU-DET-yolo/neu_det_yolo_v3/neu_det.yaml epochs=100 batch=8 imgsz=640 device=3 seed=2024 amp=false project=/mnt/pfs/zitao_team/baiyouheng/research/TX_yolo/YOLO-Master/c3_work/stage3_matrix/runs/neu_vpeft_s2024/train name=vpeft lora_save_adapters=True lora_planner_enabled=True lora_planner_backend=vpeft lora_adapter_budget=2100000 lora_vpeft_strict=True lora_r=8 lora_alpha=16 lora_exclude_modules=routing_network.2, dfl.conv, 0.conv

## pcb
# frozen_backbone
/mnt/pfs/zitao_team/baiyouheng/conda_envs/yolo_master/bin/yolo detect train model=/mnt/pfs/zitao_team/baiyouheng/research/TX_yolo/YOLO-Master/YOLO-Master-EsMoE-N.pt data=/mnt/pfs/zitao_team/baiyouheng/datasets/DeepPCB-yolo/deeppcb_v1/deeppcb.yaml epochs=100 batch=8 imgsz=640 device=5 seed=2024 amp=false project=/mnt/pfs/zitao_team/baiyouheng/research/TX_yolo/YOLO-Master/c3_work/stage3_matrix/runs/pcb_frozen_backbone_s2024/train name=frozen_backbone lora_save_adapters=True lora_r=0 freeze=11

# full_sft
/mnt/pfs/zitao_team/baiyouheng/conda_envs/yolo_master/bin/yolo detect train model=/mnt/pfs/zitao_team/baiyouheng/research/TX_yolo/YOLO-Master/YOLO-Master-EsMoE-N.pt data=/mnt/pfs/zitao_team/baiyouheng/datasets/DeepPCB-yolo/deeppcb_v1/deeppcb.yaml epochs=100 batch=8 imgsz=640 device=1 seed=2024 amp=false project=/mnt/pfs/zitao_team/baiyouheng/research/TX_yolo/YOLO-Master/c3_work/stage3_matrix/runs/pcb_full_sft_s2024/train name=full_sft lora_save_adapters=True lora_r=0 freeze=0

# vpeft
/mnt/pfs/zitao_team/baiyouheng/conda_envs/yolo_master/bin/yolo detect train model=/mnt/pfs/zitao_team/baiyouheng/research/TX_yolo/YOLO-Master/YOLO-Master-EsMoE-N.pt data=/mnt/pfs/zitao_team/baiyouheng/datasets/DeepPCB-yolo/deeppcb_v1/deeppcb.yaml epochs=100 batch=8 imgsz=640 device=3 seed=2024 amp=false project=/mnt/pfs/zitao_team/baiyouheng/research/TX_yolo/YOLO-Master/c3_work/stage3_matrix/runs/pcb_vpeft_s2024/train name=vpeft lora_save_adapters=True lora_planner_enabled=True lora_planner_backend=vpeft lora_adapter_budget=2100000 lora_vpeft_strict=True lora_r=8 lora_alpha=16 lora_exclude_modules=routing_network.2, dfl.conv, 0.conv

## stage3
# vpeft
/mnt/pfs/zitao_team/baiyouheng/conda_envs/yolo_master/bin/yolo detect train model=YOLO-Master-EsMoE-N.pt data=/mnt/pfs/zitao_team/baiyouheng/datasets/NEU-DET-yolo/neu_det_yolo_v3/neu_det.yaml epochs=3 batch=8 imgsz=640 device=1 seed=824 amp=false project=/mnt/pfs/zitao_team/baiyouheng/research/TX_yolo/YOLO-Master/c3_work/stage3_matrix/runs/stage3_pilot_vpeft_neu/train name=vpeft lora_save_adapters=True lora_planner_enabled=True lora_planner_backend=vpeft lora_adapter_budget=2100000 lora_vpeft_strict=True lora_r=8 lora_alpha=16 lora_exclude_modules=routing_network.2, dfl.conv, 0.conv

