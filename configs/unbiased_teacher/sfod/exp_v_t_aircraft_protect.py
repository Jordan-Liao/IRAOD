# Start from the best Exp T student/EMA weights so aircraft filtering can be
# tested without the cold-start confound in Exp U.
_base_ = './exp_m_wu_schedule.py'

score = [0.5, 0.95, 0.5, 0.85, 0.75, 0.6]

# Exp T checkpoints keep stale resume metadata, so use explicit student/EMA
# weights as initialization instead of resuming runner state.
load_from = 'work_dirs/exp_t_m_resume/iter_126146.pth'
resume_from = None

model = dict(
    ema_ckpt='work_dirs/exp_t_m_resume/iter_126146_ema.pth',
    cfg=dict(
        score_thr=score,
    ),
)
