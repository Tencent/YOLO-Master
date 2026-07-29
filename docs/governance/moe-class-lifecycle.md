# MoE Class Lifecycle Governance

The MoE package uses a single implementation source for every public class.
Historical module paths remain as import aliases so existing Python callers
and serialized checkpoints can still resolve their original symbols.

Run the structural gate with:

```bash
python scripts/check_moe_ssot.py
```

Run the usage audit with:

```bash
python scripts/audit_moe_usage.py
```

Record the YAML-visible mixture usage for a release with an explicit version:

```bash
python scripts/audit_moe_usage.py --record-version 8.4.101
```

The versioned ledger is `docs/governance/moe-variant-usage.json`. An
experimental YAML-visible class becomes eligible for `DEPRECATED_MOE_CLASSES`
only after it is absent from two consecutive recorded versions. The audit does
not edit the runtime API tier: release owners must first review external Python
imports and historical checkpoint pickle paths, then declare the class
deprecated for one release window. CI can detect ledger/tier drift with:

```bash
python scripts/audit_moe_usage.py --check-deprecated
```

The 2026-07-17 audit found 74 canonical public class definitions:

- 51 `retain`: referenced by YAML or directly covered by tests.
- 19 `freeze`: exported or otherwise referenced, but not directly used by YAML/tests.
- 4 `archive-candidate`: no detected repository references.

`archive-candidate` is not permission to delete. Removal requires a manual
review of external Python imports and historical checkpoint pickle paths,
followed by a deprecation window. This avoids treating YAML usage as the full
public compatibility surface.

Compatibility modules currently include `base.py`, `blocks_advanced.py`,
`experts_advanced.py`, `routers_advanced.py`, `hybrid.py`, and
`integration.py`. They must remain implementation-free re-export shims.
