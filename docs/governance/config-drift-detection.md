# Configuration Drift Detection

YOLO-Master uses a static configuration drift gate to keep its public CLI defaults, runtime resolvers, configuration dataclasses, and model YAML files aligned.

Run the gate from the repository root:

```bash
python tools/config_drift_detector.py
```

A successful run prints the number of Master model configurations checked and exits with status 0. Failures use stable issue codes, include the source path, and exit with status 1. Machine-readable output is available with `--json`.

## Checked contracts

The detector verifies:

1. `ultralytics/cfg/default.yaml` and every `ultralytics/cfg/models/master/**/*.yaml` file contain no duplicate mapping keys and parse as safe YAML.
2. Every local `LoRAConfig` and `MoLoRAConfig` dataclass field is exposed by its `from_args` mapping, and every mapped CLI key exists in `default.yaml`.
3. `MIXTURE_DEFAULTS` and `CLI_FIELDS` cover the same resolver fields, reference public CLI keys, and use the same default values as `default.yaml`.
4. `CFG_FLOAT_KEYS`, `CFG_FRACTION_KEYS`, `CFG_INT_KEYS`, and `CFG_BOOL_KEYS` reference existing public keys with compatible default types.
5. Master model YAML layer records have valid structure, reference modules actually imported or defined by `tasks.py`, and provide a positional argument count accepted after `parse_model` adds channels, repeats, or head inputs.

## CI boundary

This gate is intentionally static and fast enough for every pull request. It does not instantiate all 300+ historical model configurations or claim runtime numerical correctness. The existing configuration smoke tests and export roundtrip jobs remain responsible for construction, forward execution, and deployment parity.

When adding a new configuration field or model module, update all relevant public defaults and mappings in the same change. Do not suppress a drift issue unless the underlying configuration contract is intentionally changed and documented.
