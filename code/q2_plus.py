import pickle as pkf
import warnings

import optuna

from vspace import *

warnings.filterwarnings('ignore')


def fitness(trial: optuna.Trial = None, **kwargs):
    _getv = lambda k, l, h: trial.suggest_float(k, l, h, step=1e-2) if trial else kwargs[k]

    y = _getv('y', -R, 0)  # 吸收塔位置
    z_hs = _getv('z_hs', 2, 6)  # 定日镜安装高度
    Lw = _getv('Lw', 2, 8)
    Lh = _getv('Lh', 2, min(8, Lw, z_hs * 2))  # 定日镜尺寸
    itv = _getv('itv', 5 + Lw, 20)  # 定日镜周向间隔

    try:
        hf = HeliostatFieldPlus(y, itv)
        sv = SolverBase(hf.pos_hs, hf.pos_clt, z_hs=z_hs, Lwh=Lw + Lh * 1j, dpi=[4, 4], nm=[2, 3])
        sv.refresh()

        d = np.stack([sv.pos_hs.real, sv.pos_hs.imag, (sv.eta * wt).mean(axis=(-1, -2))], axis=-1)
        pd.DataFrame(d).to_csv('tmp.csv')

        sv.res2excel('2-plus')
        sv.hs2csv(2)
        print(f'定日镜总面数: {len(sv.pos_hs)}')
        return sv.fitness(6e4, verbose=bool(kwargs))

    except Exception as error:
        raise optuna.TrialPruned


# 3277
best = {'y': -104.18, 'z_hs': 2.9, 'Lw': 5.88, 'Lh': 5.64, 'itv': 11.14}

if __name__ == '__main__':
    fitness(**best)
    opt = optuna.create_study(direction='maximize')
    file = Path('trials-plus.bin')
    if file.is_file(): opt.add_trials(pkf.loads(file.read_bytes()))
    opt.enqueue_trial(best)

    while True:
        opt.optimize(fitness, 10)
        file.write_bytes(pkf.dumps([x for x in opt.trials if x.state == 1]))
        print('Automatically saved.')
