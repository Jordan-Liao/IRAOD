# Mohu

## 1) Not Implemented
- None for the SFOD-RS full-fix rerun scope (`2026-04-24` to `2026-04-27`).

## 2) Ambiguities
- A0001: Whether the next round should keep strict target-only adaptation or allow controlled source replay to stabilize self-training collapse.
- A0002: Whether CGA weight rule (`0.7*teacher + 0.3*clip_prob_orig`) should be tuned per corruption.

## Resolved (archive)
- [x] M0032: `test.py` SFOD-RS diagnostics missing in evaluation path.
  - Evidence: target-eval logs include `stage=target_eval`, `use_labeled_source_in_adaptation=False`, `target_domain=<corr>`, `cga_mode=sfodrs`, `keep_label=True`, score rule string.

- [x] M0033: Full script could not start from clean source training automatically.
  - Evidence: `scripts/run_rsar_sfodrs_full_3gpu.sh` accepts `auto`; run log shows `step=source_train` first.

- [x] M0034: Full rerun got blocked during cutover wait state.
  - Evidence: split/resume scripts added:
    - `scripts/cutover_split_main_to_shard.sh`
    - `scripts/cutover_resume_after_wait.sh`
    - `scripts/run_single_corr_ddp.sh`
  - Remaining domains (`point_target`, `noise_suppression`) completed in parallel and converged to `done`.

- [x] M0035: Missing final aggregate outputs under run root.
  - Evidence: generated
    - `work_dirs/rsar_sfodrs_full_fix_20260424_172627/rsar_sfodrs_results.csv`
    - `work_dirs/rsar_sfodrs_full_fix_20260424_172627/rsar_sfodrs_results.md`
