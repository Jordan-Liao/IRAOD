# Resume config for Exp M recovery (inherits from exp_m_wu_schedule)
_base_ = './exp_m_wu_schedule.py'

load_from = None
resume_from = 'work_dirs/exp_m_wu_schedule/latest.pth'

model = dict(
    ema_ckpt='work_dirs/exp_m_wu_schedule/latest_ema.pth',
)
