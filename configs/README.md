# Config Map

`configs/current/` is the stable entry layer for new RSAR work.

| Config | Purpose | Backing config |
| --- | --- | --- |
| `current/rsar_sfodrs.py` | Main RSAR SFOD-RS source/direct/adapt/eval config. Selects behavior with `RSAR_STAGE`. | `unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_sfodrs_rsar.py` |
| `current/rsar_source.py` | Short name for clean RSAR source training. | Same as `current/rsar_sfodrs.py` with `RSAR_STAGE=source_train` |
| `current/rsar_eval_ema.py` | Short name for non-CGA EMA detector evaluation. | `baseline/ema_config/sfodrs_oriented_rcnn_ema_rsar.py` |

Historical configs are intentionally kept in place:

- `configs/baseline/`: baseline and EMA model definitions.
- `configs/experiments/`: stable DIOR/RSAR baseline and frontier experiment configs.
- `configs/unbiased_teacher/sfod/`: source-free adaptation and phase experiment configs.

Prefer `configs/current/*.py` in new scripts and docs. Use historical paths only
when reproducing a specific older experiment that recorded that exact config.
