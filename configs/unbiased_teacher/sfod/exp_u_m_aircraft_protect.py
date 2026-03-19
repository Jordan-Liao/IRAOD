# Follow-up to Exp T: keep the best M schedule but heavily filter aircraft pseudo labels.
_base_ = './exp_p_m_aircraft_protect.py'

load_from = 'work_dirs/exp_rsar_baseline/epoch_12.pth'

model = dict(
    ema_ckpt='work_dirs/exp_rsar_baseline/epoch_12.pth',
)
