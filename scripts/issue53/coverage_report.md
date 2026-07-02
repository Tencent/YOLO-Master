# Issue 53 MoA Coverage Report

Coverage command:

```bash
python -m pytest tests/test_moa.py --cov=ultralytics/nn/modules/moa --cov-report=term-missing -q
```

## Before

Baseline: repository `HEAD` before the Issue 53 boundary-test changes.

```text
7 passed

Name                                     Stmts   Miss  Cover
----------------------------------------------------------------------
ultralytics\nn\modules\moa\__init__.py       2      0   100%
ultralytics\nn\modules\moa\moa.py          295     35    88%
----------------------------------------------------------------------
TOTAL                                      297     35    88%
```

## After

After adding the Issue 53 MoA boundary tests and fixes.

```text
12 passed

Name                                     Stmts   Miss  Cover
----------------------------------------------------------------------
ultralytics\nn\modules\moa\__init__.py       2      0   100%
ultralytics\nn\modules\moa\moa.py          301     17    94%
----------------------------------------------------------------------
TOTAL                                      303     17    94%
```

## Summary

- `tests/test_moa.py` increased from 7 to 12 passing tests.
- MoA module coverage increased from 88% to 94%.
- Added boundary coverage for cross-scale fusion shape alignment, tiny router
  temperature stability, non-divisible attention head configurations, and
  nested `C2fMoA` aux-loss aggregation without double counting.
