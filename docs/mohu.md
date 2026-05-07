# Mohu

## 1) Not Implemented
- None for the SFOD-RS full-fix rerun scope (`2026-04-24` to `2026-04-27`).

## 2) Ambiguities
- A0002: Whether CGA weight rule (`0.7*teacher + 0.3*clip_prob_orig`) should be tuned per corruption.

## Resolved (archive)

- [x] A0003: Whether TENT (BN affine entropy minimization) can outperform corr-aug direct_test on any RSAR corruption domain.
  - Evidence（E0129）: TENT mean mAP=0.4656（Δ=-0.86pp vs direct 0.4742）。未能超越 direct，但是所有适应方法中最接近 direct 的（vs E0128 -7.1pp）。point_target 仅 -0.10pp，am_noise_horizontal 从伪标签方案的 -14.9pp 收窄到 -1.06pp。
  - Conclusion: TENT 是最优适应方法，但无适应（corr-aug direct）仍是最高 mAP 策略。下一候选：TENT+direct ensemble 融合。
- [x] A0001: Whether the next round should keep strict target-only adaptation or allow controlled source replay.
  - Evidence: Phase 8 (loose + weight_l=0.5) eliminated strict collapse; Phase 9 (corr-aug source) raised direct_test mean from 0.337 to 0.474. Conclusion: loose mode resolves collapse; corr-aug source is the stronger lever. Strict target-only is deprecated for RSAR.

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

- [x] M0036: `pseudo_num(acc)=0.000` 全程被误判为"伪标签全部被过滤"。
  - Evidence（Phase 10 诊断）: `pseudo_num(acc)` 是 IoU TP 指标，目标域无 GT 注释 → 分母为 0 → 恒为 0。实际 pseudo/img=1.74，mean_score=0.916（thr=0.7），伪标签正常生成。真实问题是高置信度伪标签缺乏新适应信号，不是过滤 bug。
  - Fix: 新增 `[PseudoScoreDist]` 日志 + `PseudoStatsAndEarlyStopHook` 警告，暴露 `RSAR_THR_SCHEDULE` 到 config。
  - Outcome: E0127（thr=0.2→0.4）和 E0128（thr=0.4 fixed）均验证阈值调优无法修复，类别不平衡是根本障碍。
