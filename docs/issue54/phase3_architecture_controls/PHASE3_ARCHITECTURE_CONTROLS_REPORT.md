# Phase 3 Architecture Controls Report

Generated at: `2026-08-02T13:03:43Z`

## 1. Scope

This report compares the final detection performance of:

- **EsMoE**: model key `v10`, seeds 0, 1, and 2;
- **MoA**: model key `v10_moa`, seed 0;
- **MoT**: model key `v10_mot`, seeds 0 through 4.

All runs used 30 epochs, batch size 8, image size 640, VisDrone2019-DET, and the formal Issue #54 protocol. MoT used FP32; EsMoE and MoA used AMP.

The highest-level experimental unit is an independently trained seed. Images or tokens are not treated as independent training repetitions.

## 2. Integrity checks

- All nine formal runs have `status=passed`.
- All nine runs completed 30 epochs.
- All checkpoint SHA256 values match their manifests.
- All nine formal checkpoints are mutually distinct.
- EsMoE and MoA contain no fabricated MoT routing artifacts.
- Existing MoT routing artifacts are preserved separately.

## 3. Per-run final metrics

| Architecture | Model | Seed | Precision | mAP50 | mAP50-95 | Checkpoint SHA256 |
|---|---:|---:|---:|---:|---:|---|
| EsMoE | `v10` | 0 | amp | 0.16273 | 0.08474 | `64d8c53b001b3db57fb38a1686d5896087f06be898a22e5a75cbd0ae297bd2e0` |
| EsMoE | `v10` | 1 | amp | 0.15606 | 0.08137 | `39561c585c98081c3c6f8297074cdafe7225a7e8878727fb2fac4bb61d6305d3` |
| EsMoE | `v10` | 2 | amp | 0.16124 | 0.08493 | `4b805479ceb38d0c2cde0662a258f381c0fa0f506b20a08b50b9c0d72c7c09a2` |
| MoA | `v10_moa` | 0 | amp | 0.15844 | 0.08164 | `9d939680ac80d802a5d46eab8d5262990af2d7991936edccb64a8316fda2978c` |
| MoT | `v10_mot` | 0 | fp32 | 0.16189 | 0.08469 | `ec18580cdafe91b49684007c1391b759a5612c2733bd6d5c0009d2e5b3117bda` |
| MoT | `v10_mot` | 1 | fp32 | 0.15701 | 0.08056 | `fd9a959f34fcc4c75db71f3ecb91dd7c43afa162a4664b5a136cd44453da3b97` |
| MoT | `v10_mot` | 2 | fp32 | 0.16176 | 0.08392 | `fa9ed0cd300ea2f2ec2eb2b506d668c7710f4d0f2d26217d6c0978ea1acab858` |
| MoT | `v10_mot` | 3 | fp32 | 0.16364 | 0.08457 | `b2e2dcd75497ae47430b2b17ac9c7514e3847930d7b49374723eefcb591e4dce` |
| MoT | `v10_mot` | 4 | fp32 | 0.15753 | 0.08182 | `43a7a6d20a84782d9078312e9030210086eac2a8d0cad9d1457b9c6beaa37ad7` |

## 4. Aggregate performance

Sample standard deviation is reported only when at least two independent seeds are available.

| Architecture | Independent seeds | mAP50 | mAP50-95 |
|---|---:|---:|---:|
| EsMoE | 3 | 0.16001 ± 0.00350 | 0.08368 ± 0.00200 |
| MoA | 1 | 0.15844 (single seed) | 0.08164 (single seed) |
| MoT | 5 | 0.16037 ± 0.00293 | 0.08311 ± 0.00183 |

## 5. Descriptive architecture differences

These differences are descriptive only. They are not formal significance tests and do not establish causal superiority.

| Comparison | ΔmAP50 | ΔmAP50-95 |
|---|---:|---:|
| MoT mean − EsMoE mean | +0.00036 | -0.00057 |
| MoA seed0 − EsMoE mean | -0.00157 | -0.00204 |
| MoA seed0 − MoT mean | -0.00193 | -0.00147 |

## 6. Evidence-bounded interpretation

1. **MoT and EsMoE have very similar mean detection performance under the current protocol.** The observed mean differences are small and mixed across the two metrics.
2. **MoA is represented by one independent training seed.** Its result is a single-run architecture control and cannot support claims about between-seed stability or variance.
3. **Performance stability and routing stability are different questions.** The existing MoT cross-seed report shows relatively stable detection performance alongside only moderate or low internal routing agreement, with strong layer-level differences.
4. The present evidence does not prove that routing instability reduces performance, that a specific expert has a fixed semantic role, or that high routing entropy implies high routing stability.

## 7. Statistical limitations

- Seed counts are unequal: MoT n=5, EsMoE n=3, MoA n=1.
- No formal hypothesis test is reported.
- MoA has no valid between-seed variance estimate.
- Conclusions should remain descriptive and protocol-specific.

## 8. Related MoT routing report

The detailed MoT routing analysis remains in:

`/root/autodl-tmp/MoT/results/phase3_formal_20260731T214515Z/reports/PHASE3_MOT_CROSS_SEED_REPORT.md`

This architecture report does not recreate or invent routing records for EsMoE or MoA.

## 9. Source roots

- Controls root: `/root/autodl-tmp/MoT/results/phase3_controls_20260801T181817Z`
- MoT formal root: `/root/autodl-tmp/MoT/results/phase3_formal_20260731T214515Z`
