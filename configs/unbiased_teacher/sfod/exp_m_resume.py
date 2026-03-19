# Resume config for Exp M (inherits from exp_m_wu_schedule)
_base_ = './exp_m_wu_schedule.py'

# Override non-existent paths with actual baseline checkpoint
load_from = None  # Not needed when resuming
resume_from = 'work_dirs/exp_m_wu_schedule/latest.pth'

# Semi-supervised resume also needs the matching EMA checkpoint.
model = dict(
    ema_ckpt='work_dirs/exp_m_wu_schedule/latest_ema.pth',
)
