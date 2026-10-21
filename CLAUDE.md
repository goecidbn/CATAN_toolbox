# CATAN — project instructions for Claude

CATAN (Curating And Tracking Across Neurons) is a Python ≥3.11 toolbox with a PySide6/VisPy GUI and headless matching for inspecting, curating and tracking neurons across longitudinal calcium-imaging sessions (mainly CaImAn output). Maintainer: Alexander Schmidt. Bernstein Conference 2026 poster presented; paper (Schmidt, Aamir, Wolf) at an early conceptual stage.

## How to work with Alexander

- **Never commit or push** unless explicitly asked. Alexander reviews, commits and pushes himself.
- Make **targeted changes**: name file, class/method and location, and explain briefly why. Preserve existing names, architecture and working features.
- **Read the current source before changing it.** Never rewrite a module from memory or an older copy; a past broad replacement silently dropped `StatisticDefinition.foreground_safe`.
- Full-file replacement is acceptable only for `gui/panels/ModelResults.py`.
- When presenting changes in chat (not as an applied diff), show the snippets with exact location rather than a patch file.
- "Go on" / "works" means the previous changes are applied and confirmed.
- Always state the status: implemented / tested in isolation (synthetic or offscreen) / confirmed by Alexander on real data.
- Keep explanations concise but complete.
- Preserve **panel hot-reload** compatibility: panels are reloaded via `importlib.reload` (display_section). Use stable `node_type` markers and helpers like `is_filter_condition()` / `is_filter_group()`, not `isinstance` against reloadable classes.
- For thresholds and parameters, prefer values derived from the data's own distributions (percentiles of same-neuron vs nearest-other-neuron populations) over guesses.

## Architecture

- `src/catan/core`: shared structures (`SessionData`, snapshots, io backends, alignment), persistence.
- `src/catan/tracking`: headless matching (`Tracking` in `neuron_tracking.py`, `Model` in `tracking/structures/model.py`, `Assignments`, analytics).
- `src/catan/gui`: PySide6/VisPy. `gui/structures/data.py` (`Data(Tracking)`: orchestration, `notify_change`), `gui/structures/state.py` (selection, revisions), `gui/background_tasks` (TaskManager, runtime progress/cancellation), `gui/panels` (`BasePlot.BaseDisplayController`, registered in `GUI_elements/display_section.py`).
- Dependencies point gui → core/tracking, never the reverse.
- Dependency chain: geometry/alignment → model counts → fitted model → assignments. Upstream changes mark downstream stale (path-keyed counts, `stale_counts`, `fit_stale`, stale assignment paths). Applying a model preview marks assignments stale but does not rerun tracking.
- Terminology: `counts["same"]` = within-session evidence (self-pairs excluded), NOT same-neuron evidence. NN / non-NN are evidence categories, not ground-truth labels.
- Fine-grained change notifications (`core/changes.py` `ChangeKind`): unrelated actions (e.g. review status) must not invalidate expensive statistics.
- Selection colours: hovered blue, selected green, focused strong green (max one), highlighted orange.

## Statistics (`gui/data/statistics/`)

- Query-driven: `StatisticDefinition` (dims, reductions, parameters, dependencies, `foreground_safe`), `StatisticQuery`, `StatisticEngine` (cache, revisions), `PickTable`, `SparseStatisticArray`.
- Imported lookup statistics are `session:<name>` / `match:<name>` from loaded data; names are arbitrary — never hardcode CNN/SNR/r-values in logic.
- Event rate uses `event_estimator.get_events()` (mutates input → pass a copy; check its real return order, not the docstring); results persist in `SessionData.derived_statistics`.
- Curator selections use **sparse fast paths** (applied when the value selection rejects 0 / has a finite upper bound and all axes are keep/single; otherwise the old blockwise path):
  - `overlap_statistics._sparse_overlap_selection`: only intersecting supports (binary: sparse product; weighted: pixel join).
  - `distance_statistics._sparse_distance_selection`: KD-tree candidates within the selection radius (modes cross/within/to_union/union).
  - `sparse_pairs.py`: `SparsePairCollector`, `selection_upper_bound`.
  - Footprint similarity: `core/image_correlation.footprint_similarity_peaks` (batched FFTs on a thread pool, bitwise-identical to `footprint_similarity_peak`), vectorised candidates in `footprint_pairs.py`.
- Equivalence of any change to these paths must be checked against the previous implementation (identical rows/values/metadata).

## Curator (`gui/data/curation_filter.py`, panel `gui/panels/CurationFilter.py`)

- Presets are trees of AND/OR groups; match levels neuron / footprint / relationship (+ reassignment, neuron_pair); `apply_to` source/target/between_sources/between_targets; `CountSpec` (eq/ge/le per anchor); `review_statuses` domain; `manipulation` (split / merge / neuron_merge / reassign, plus batch actions mark_reviewed / mark_uncertain / exclude_neuron / exclude_footprint).
- Shipped presets: `src/catan/resources/curation_filters/*.json` (auto-discovered, key `builtin:<stem>`).
- Only AND relationship groups evaluate progressively (`requested_pairs`); nested groups, OR groups and neuron/footprint groups get full domains — a known optimisation opportunity.
- Curator results currently include excluded neurons; batch actions skip them.
- Threshold semantics: "greater" means ≥, "less" means ≤.

## Status (2026-10-21)

Delivered to Alexander as snippets/patches; check `git status`/`git log` to see what is already applied:
1. Statistics speedup (sparse overlap/distances, batched similarity) — `catan_statistics_speedup.patch`. Verified identical results on synthetic data and the 9-session real dataset; high-confidence preset 17.0 s → 5.6 s there. Not yet confirmed on the 60-session dataset (previously ~1 h).
2. Noise preset `noise_removal_candidates.json` ("Noise components: removal candidates"): ≤3 occurrences AND poor quality/reproducibility (max CNN < 0.5, max SNR < 4, max r-values < 0.8, max size < 300, median same-neuron similarity < 0.85, max displacement > 5 px) AND no plausible partner (other neuron, other session: distance < 4 AND overlap > 0.6 AND similarity > 0.85). Uses `session:CNN/SNR/r-values` by name.
3. Curator batch actions — `catan_curator_batch_actions.patch` (`BATCH_ACTIONS`, `RetainedResults.target_components`, `Data.exclude_neurons` / `exclude_components`, `CuratorActions._apply_batch`). Tested in the GUI on real data.
4. "Suggest" thresholds (2026-10-21, applied in working tree, uncommitted): `StatisticEngine.sample_values` → `statistics/sampling.py` (strategies pairs / full / neurons; cache keyed by statistic revisions + budget), `StatisticDefinition.neuron_local`, `footprint_pairs.footprint_similarity_for_pairs`, Suggest row + percentile grid in `CurationFilterConditionPopup` (click = threshold, Shift+click = upper bound). Tested on synthetic data and offscreen GUI only (sampled values identical to full evaluation incl. float16 compaction); not yet confirmed on real data. Known limit: per-neuron reductions of similarity over session pairs (noise preset's median similarity) reach only ~15 neurons in 2 s at 30 sessions ("Sample more" = 10 s).

Calibration from the 9-session dataset: same-neuron similarity p5 0.88; same-neuron distance p95 4.9 px; same-neuron overlap (containment) p5 0.75; nearest other neuron: distance median ~7 px, overlap median ~0.6, similarity median ~0.90 (similarity alone barely separates neighbours).

## Next

1. **"Suggest" thresholds**: implemented (Status 4); awaiting confirmation on real data. Open: approximate mode for per-neuron session-pair reductions; reduced partner axes (e.g. min over neuron_j) fall back to "Calculate all".
2. Optional conditions for imported statistics (skip when unavailable).
3. Remaining agreed presets: edge artifacts, within-session duplicates, size outliers, fragmented tracks (neuron_merge), misassigned footprints / orphans (reassign), merge candidates (mirror of split), refined high-confidence (+ mark_reviewed), nearly clean, unstable tracks, swap risk, low model confidence.
4. Progressive evaluation for nested groups / neuron-level AND groups.
5. Model-results panel (`gui/panels/ModelResults.py`): latest changes (apply placement, spacing, 3D labels, tick spacing, count-axis rescale, session-difference switching, "Calculate missing evidence") not yet user-confirmed. Open: separate detection uncertainty from centroid movement (σ_D currently = `sigma_eff`); `use_cdf` probabilities vs PDF-on-bins plotting are distinct.

## Housekeeping (not yet acted on)

- `src/catan/gui/data/statistics.zip` is committed inside the package (likely accidental).
- `pyproject.toml` URLs point to woxxel/CATAN_toolbox; README clones goecidbn; README expands CATAN differently.
- ~800 ruff findings, mostly style; not a priority.

## Commands

- Install: `pip install -e ".[gui,dev]"`
- Tests: `pytest` (headless: `QT_QPA_PLATFORM=offscreen pytest`)
- GUI: `catan-gui` (`--fresh` for temporary clean settings); CLI: `catan`
