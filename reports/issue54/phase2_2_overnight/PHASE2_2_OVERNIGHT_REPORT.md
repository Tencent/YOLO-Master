# Issue #54 Phase 2.2 Overnight Pilot

Evidence classification: `diagnostic_not_formal_evidence`.

This is a bounded long-run pilot, not a formal multi-seed conclusion.

## Environment and protocol

- Git: `cf233abb630ab6490bb2fb7a47e4b80cb9ab4822`
- GPU: `{'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'driver': '580.76.05'}`
- PyTorch/CUDA: `2.5.1+cu124` / `12.4`
- Dataset inventory: `4a7a03b08cf21d913ab85a86b40a75eee13579881fba9bc0d979b72b7f0a96fa`
- Wall time: `2223.1` seconds
- GPU process + routing time: `2220.4` seconds

## Precision selection

- Selected: `None`
- Reason: FP32 calibration/routing was not stable; protocol calibration is incomplete
- Evidence: `{'selected': None, 'reason': 'FP32 calibration/routing was not stable; protocol calibration is incomplete', 'threshold_amp_max_speed_ratio': 0.9, 'amp_stable': False, 'fp32_stable': False, 'amp_seconds_per_epoch': None, 'fp32_seconds_per_epoch': None, 'amp_over_fp32_speed_ratio': None, 'same_seed_independent_runs': False, 'interpretation': 'precision calibration only; AMP and FP32 are not independent seeds'}`

## Runs

| Run | Variant | Seed | Precision | Epochs | s/epoch | Peak GiB | mAP50 | mAP50-95 | Status |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| calibration_mot_amp | mot | 0 | amp | 0/3 |  | 7.90 | None | None | failed |
| calibration_mot_fp32 | mot | 0 | fp32 | 0/3 |  | 7.88 | None | None | failed |

## Routing and conclusion

- Routing exports: `[]`
- Two-seed MoT comparison: `{'available': False, 'reason': 'not evaluated'}`
- Ready for an explicitly approved formal MVP: `False`

Metrics are pilot-only. Two seeds, if available, are still insufficient for a formal aggregate claim. No formal 5-run protocol was launched.
