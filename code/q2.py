import pickle as pkf
import warnings

import optuna

from vspace import *

warnings.filterwarnings('ignore')


def fitness(trial: optuna.Trial = None, **kwargs):
    _getv = lambda k, l, h: trial.suggest_float(k, l, h, step=2e-2) if trial else kwargs[k]

    y = _getv('y', -R, R)  # 吸收塔位置
    z_hs = _getv('z_hs', 2, 6)  # 定日镜安装高度
    Lw = _getv('Lw', 2, 8)
    Lh = _getv('Lh', 2, min(Lw, 8, z_hs * 2))  # 定日镜尺寸
    itv = _getv('itv', 5 + Lw, 20)  # 定日镜周向间隔
    r_itva = [_getv(f'r_itva', 5 + Lw, 20)]  # 定日镜径向间隔函数

    try:
        hf = HeliostatField(y, r_itva, itv)
        assert len(hf.pos_hs) < 5e3
        sv = SolverBase(hf.pos_hs, hf.pos_clt, z_hs=z_hs, Lwh=Lw + Lh * 1j, dpi=[4, 4], nm=[3, 4])
        sv.refresh()
        # 计算一半的定日镜
        return sv.fitness(3e4, verbose=bool(kwargs))

    except Exception as error:
        raise optuna.TrialPruned


best = {'y': -58.1, 'z_hs': 5.8, 'Lw': 7.56, 'Lh': 7.38, 'itv': 12.72, 'r_itva': 12.78}

if __name__ == '__main__':
    opt = optuna.create_study(direction='maximize')
    file = Path('trials.bin')
    if file.is_file(): opt.add_trials(pkf.loads(file.read_bytes()))
    opt.enqueue_trial(best)

    # fitness(**best)
    while True:
        opt.optimize(fitness, 10)
        file.write_bytes(pkf.dumps([x for x in opt.trials if x.state == 1]))
        print('Automatically saved.')
