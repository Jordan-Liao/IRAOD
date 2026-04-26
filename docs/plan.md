# RSAR SFOD-RS Full Fix Plan (Completed)

## Scope
- Project: IRAOD RSAR SFOD-RS full rerun
- Remote repo: `/mnt/SSD1_8TB/zechuan/IRAOD`
- Run root: `work_dirs/rsar_sfodrs_full_fix_20260424_172627`
- Runtime window: `2026-04-24 17:26:27 CST` to `2026-04-26 21:13:36 CST`
- Total elapsed wall time: `51.786 h`

## Objective
1. Enforce SFOD-RS protocol with target-only adaptation on RSAR 7 corruptions.
2. Make full pipeline runnable from `auto` source checkpoint generation (no manual source ckpt handoff).
3. Add explicit SFOD-RS diagnostics in evaluation stage.
4. Produce one final aggregate result table (`csv` + `md`) for the completed run.

## Fixed Protocol
1. `source_train`: clean labeled train split only.
2. `source_clean_test`: clean test split.
3. For each corruption in:
   `chaff`, `gaussian_white_noise`, `point_target`, `noise_suppression`,
   `am_noise_horizontal`, `smart_suppression`, `am_noise_vertical`
   - `direct_test` on `corruptions/<corr>/test/images`
   - `self_training` on `corruptions/<corr>/val/images` then eval on `test/images`
   - `self_training_plus_cga` on `corruptions/<corr>/val/images` then eval on `test/images`
4. Do not use corruption train split.

## Code/Script Delta Included
- `test.py`: SFOD-RS diagnostics print in evaluation stage.
- `scripts/run_rsar_sfodrs_full_3gpu.sh`: support `source_arg=auto` to train source detector first and then continue full 7-corr pipeline.
- `scripts/run_rsar_sfodrs_7corr.sh`, `scripts/exp_rsar_sfodrs_adapt.sh`: protocol wiring updates.
- `tools/collect_rsar_sfodrs_results.py`: aggregate final metrics table.
- Split-run utilities added during rerun recovery:
  - `scripts/cutover_split_main_to_shard.sh`
  - `scripts/cutover_resume_after_wait.sh`
  - `scripts/run_single_corr_ddp.sh`

## Execution Matrix
- Total planned tasks: `37`
  - `1` source train
  - `1` clean source test
  - `7 x 5` target tasks (`direct_test`, `adapt_nocga`, `eval_nocga`, `adapt_cga`, `eval_cga`)
- Completion: `37/37`

## Parallel Cutover Record
- Main line covered source + 2 corruptions + partial `point_target`.
- Shard line covered:
  - `am_noise_horizontal`
  - `smart_suppression`
  - `am_noise_vertical`
- Final parallel completion for remaining two domains:
  - `point_target` finished `2026-04-26 21:12:25 CST`
  - `noise_suppression` finished `2026-04-26 21:13:36 CST`

## Final Outputs
- `work_dirs/rsar_sfodrs_full_fix_20260424_172627/rsar_sfodrs_results.csv`
- `work_dirs/rsar_sfodrs_full_fix_20260424_172627/rsar_sfodrs_results.md`
- Both generated at `2026-04-27 00:51 CST`.

## Acceptance Status
- Protocol completion check: `23/23` artifact checks passed
  - `source_train` ckpt present
  - `source_clean_test` eval json present
  - `7` corruptions x `3` eval json present (`direct_test`, `self_training`, `self_training_plus_cga`)
- Training/eval processes ended cleanly; no IRAOD run process remained active after completion.
