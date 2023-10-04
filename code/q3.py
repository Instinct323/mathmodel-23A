from tqdm import tqdm

from q2_plus import best
from vspace import *

E_tar = 6e4
res = Result(Path(), title=('E_mean', 'E_field', 'fitness'))


class Optimizer(SolverBase):
    file = Path('result3.csv')

    def __init__(self, y, itv, z_hs, Lw, Lh):
        assert len(wt) == 5, '修改 vspace 中的 omega, wt, 以进行全面的优化'
        hf = HeliostatFieldPlus(y, itv)
        super().__init__(hf.pos_hs, hf.pos_clt, z_hs=z_hs, Lwh=Lw + Lh * 1j, dpi=[4, 4], nm=[3, 4])
        self.max_lw = min(8, itv - 5)
        # 如果数据文件存在, 加载参数

        if self.file.is_file():
            data = pd.read_csv(self.file).to_numpy()
            self.Lwh = complex(data)
            self.z_hs = data[:, -1]
        self.refresh()
        fit = self.fitness(E_tar)
        self.best_fit = fit['fitness']
        print(fit)

    def fitness(self, E_tar, verbose=True):
        fit = super().fitness(E_tar, verbose=verbose)
        fit['fitness'] = fit['E_mean'] - np.maximum(E_tar - fit['E_field'], 0) / 1e4
        return fit

    def save(self):
        ret = self.fitness(E_tar, verbose=True)
        fit = ret['fitness']
        if fit > self.best_fit:
            self.best_fit = fit
            # 保存参数数据
            self.hs2csv(3)
            self.res2excel(3)
        return ret

    def fit(self, epochs, lr=.2, randfloat=.1, randbias=.05):
        for i in tqdm(range(len(res), epochs)):
            self.refresh()
            eta_sb = (self.eta_sb * wt).mean(axis=(-1, -2))
            eta = (self.eta * wt).mean(axis=(-1, -2))

            # 每个定日镜与周围镜子比较, 并提供梯度
            eta_sb = (eta_sb[:, None] - eta_sb[self.near]).mean(axis=-1)
            eta = (eta[:, None] - eta[self.near]).mean(axis=-1)

            # 随机变动值
            rfloat = lr * np.random.uniform(1 - randfloat, 1 + randfloat, [3, len(eta_sb)])
            bfloat = lr * np.random.normal(0, randbias, [3, len(eta_sb)])

            self.z_hs -= eta_sb * rfloat[0] + bfloat[0]
            self.Lwh += eta * (rfloat[1] + rfloat[2] * 1j) + (bfloat[1] + bfloat[2] * 1j)
            self.Lwh *= np.sqrt(E_tar / (self.E_field * wt).mean())

            # 输出限幅
            self.Lwh.real = np.clip(self.Lwh.real, a_min=2, a_max=self.max_lw)
            self.Lwh.imag = np.clip(self.Lwh.imag, a_min=2, a_max=np.minimum(
                self.Lwh.real, np.minimum(self.z_hs * 2, 8)))
            self.z_hs = np.clip(self.z_hs, a_min=2, a_max=6)

            res.record(self.save())
        return res


if __name__ == '__main__':
    # Optimizer(**best).fit(100)
    plt.plot(res.index, res['E_mean'], color='deepskyblue')
    plt.show()
