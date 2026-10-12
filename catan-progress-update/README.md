# CATAN curator progress

Apply `catan-progress.patch` from the repository directory containing `catan/`:

```bash
git apply --check /path/to/catan-progress.patch
git apply /path/to/catan-progress.patch
```

The first command checks compatibility without modifying files. If it reports a
conflict, keep the current files and resolve that hunk; do not replace newer
modules wholesale with the test fixtures. The patch touches only progress
instrumentation, plus validation that requested-row block size is positive.

## Behavior

- One overall 0–100 range for a curator evaluation. Nested calculations report
  within their allocated range. The overall bar does not reset between statistics.
- Status text covers statistic calculation, similarity session pairs, overlap
  preparation/intersections, reductions, table construction/filtering, row
  projection, group combination, evidence construction, and counting.
- Similarity updates within session pairs, including candidate scans with no
  matches. Binary/weighted overlap reports for Cartesian and requested-row paths.
- Sparse reductions report within the reduction loop. Blockwise reductions and
  result assembly report through nested block ranges.
- Successful completion always sends 100, bypassing throttling. Exceptions and
  cancellation do not complete their range. Cancellation restores the previous
  task context.
- Percentage updates retain the existing half-second throttle; status text is
  deduplicated and throttled too. Initial and final percentages are immediate.
- No new diagnostic prints. No changes to ResourceMonitor or its existing guards
  that hide hover/selection background tasks. Foreground calls without a task
  context produce no progress signals.
- Statistics are identified from the query/definition; no CNN/SNR/name allowlist.

Percentages represent weighted work stages, not predicted time. Individual NumPy,
SciPy, sorting, and set operations report at their boundaries; their internals
cannot report intermediate progress. A lengthy individual operation can still
hold its percentage, and an expensive compatibility search may hold within one
partner group. This patch does not artificially advance to 99 while work remains.

## Files and compatibility

Ten modules are patched. The six newly supplied files are the baseline for
runtime, blockwise, footprint_pairs, overlap_statistics, requested_rows, and
curation_filter. Progress in table construction and sparse reductions also needs
engine.py, types.py, table.py, and sparse_values.py. Their baselines are the earlier
engine(5).py, types(5).py, and statistics(4).zip uploads. The engine changes are
localized so subsequent cache-key/foreground lookup additions can remain intact.

The patch does not change normalization, FFT/scoring formulas, reduction order,
candidate eligibility, or curator boolean/count semantics. There is no new GUI
toggle: overall progress is the default for curator evaluations. Individual
statistic evaluations still use their own 0–100 range.

## Verification

Eleven headless regression tests passed, covering multiple numerical and empty
input cases: blockwise reductions, sparse reductions/table conversion, requested
rows, binary and weighted overlap, nested AND/OR curator groups, cached and
uncached engine paths, nested progress, throttling, exceptions, and cancellation.
Similarity candidate/progress plumbing was checked with a deterministic scoring
stub; its existing FFT kernel was not available in the latest upload. The Qt
interface and a real full-dataset curator run were not available here.

For the reproducible checks (Python 3.11+, NumPy and SciPy):

```bash
python -P tests/verify.py
```

The `tests/` directory contains isolated fixture copies, not installation files.
Qt, data-event types, and the threshold validation method are stubbed for these
headless checks. Existing numerical routines are compared with the supplied
originals; this is not a full application integration test.
