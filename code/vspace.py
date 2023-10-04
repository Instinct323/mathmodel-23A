import logging
from datetime import date

import matplotlib.patches as pch

from utils import *

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)
np.set_printoptions(precision=3, suppress=True)
# 维度填充
unsqueeze = lambda x: x[..., None, None]

# 公式 1
phi = np.deg2rad(39.4)  # 当地纬度
omega = np.pi / 12 * (DTYPE([9, 10.5, 12, 13.5, 15]) - 12)  # 太阳时角 [1, 5]
delta = np.arcsin(np.sin(2 * np.pi / 365 *
                         DTYPE([(date(2023, i, 21) - date(2023, 3, 21)).days for i in range(1, 13)])
                         ) * np.sin(2 * np.pi / 360 * 23.45))[:, None]  # 太阳赤纬角 [12, 1]
sin_alpha_s = np.cos(delta) * np.cos(phi) * np.cos(omega) + np.sin(delta) * np.sin(phi)
cos_alpha_s = np.sqrt(1 - np.square(sin_alpha_s))  # 太阳高度角 [12, 5]
cos_gamma_s = (np.sin(delta) - sin_alpha_s * np.sin(phi)) / (cos_alpha_s * np.cos(phi))
sin_gamma_s = np.sqrt(np.maximum(1 - np.square(cos_gamma_s), 0))  # 太阳方位角 [12, 5]
wt = DTYPE([1, 1, 1, 1, 1])  # * .6  # 按时间加权的权值

# 公式 2
H = 3  # 海拔 (km)
G0 = 1.366
_a = .4237 - .00821 * (6 - H) ** 2
_b = .5055 + .00595 * (6.5 - H) ** 2
_c = .2711 + .01858 * (2.5 - H) ** 2
dni = G0 * (_a + _b * np.exp(- _c / sin_alpha_s))  # 法向直接辐射辐照度 (kW/m^2)

# 公式 4
eta_ref = .92  # 镜面反射率
f_eta_at = lambda d_hr: .99321 - .0001176 * d_hr + 1.97e-8 * np.square(d_hr)  # 大气透射率 η_at(d_HR), d_HR ≤ 1000

# 自定义变量
B = 4.65e-3
E_field = 60  # 额定年平均输出热功率
R = 350  # 圆形定日镜场外径
r = 100  # 圆形定日镜场内径
zc = 80  # 集热器中心 (吸收塔高度)
hc = 8  # 集热器高度
rc = 3.5  # 集热器外径
b = unitize(np.stack([sin_gamma_s, cos_gamma_s, sin_alpha_s / cos_alpha_s]), axis=-3)  # 向量 [3, 12, 3]: 定日镜中心 -> 太阳


class SolverBase:

    def __init__(self, pos_hs, pos_clt, z_hs, Lwh, dpi=[5, 5], nm=[4, 6]):
        self.pos_hs = pos_hs  # 定日镜的位置 [n,]
        self.pos_clt = pos_clt  # 集热器的 x-y 位置 [1,]
        self.z_hs = z_hs if isinstance(z_hs, np.ndarray) else np.full_like(pos_hs.real, z_hs)  # 定日镜的安装高度 [n,]
        self.Lwh = Lwh if isinstance(Lwh, np.ndarray) else np.full_like(pos_hs, Lwh)  # 定日镜镜面长度、宽度

        self._dpi = dpi
        self._nm = nm

        # 为定日镜进行匹配
        rho = np.abs(pos_hs)
        adj = np.abs(pos_hs[:, None] - pos_hs)
        adj += (rho[:, None] < rho) * rho
        self.near = np.argsort(adj, axis=-1)[:, 1: 5]

    def refresh(self):
        pos_hs = np.stack([self.pos_hs.real, self.pos_hs.imag, self.z_hs], axis=-1)
        pos_clt = DTYPE([self.pos_clt.real, self.pos_clt.imag, zc])
        a = pos_clt - pos_hs
        d_hr = np.linalg.norm(a, axis=-1)  # 镜面中心到集热器中心的距离

        self.a = a / d_hr[..., None]  # 向量 [n, 3]: 定日镜中心 -> 集热器中心
        self.n = unitize(unsqueeze(self.a) + b, axis=-3)  # 定日镜单位法向量
        self.eta_at = unsqueeze(f_eta_at(d_hr))  # 大气透射率
        self.eta_cos = (b * self.n).sum(axis=-3)  # 余弦效率

        # 利用旋转矩阵转化为绝对坐标 (xyz 偏移量) [n, 12, 3, 3, 3]
        rota32 = self.get_rotate(self.n.transpose(0, 2, 3, 1)).transpose(2, 3, 4, 0, 1)[..., :2]
        # 采样若干个散点 (x-y 偏移量)
        bias = (np.concatenate(make_grid(*self._dpi, center=True) - .5)[:, None] *
                np.stack([self.Lwh.real, self.Lwh.imag], axis=-1))[..., None, None, None, :]
        bias = (bias * rota32).sum(axis=-1)

        # 入射光与吸收塔的交点
        a_ = (self.a * [1, 1, 0])[:, None, None]
        b_ = b.transpose(1, 2, 0)
        pi = pos_hs[:, None, None] + bias
        # 统计遮挡情况 [dpi, n, 12, 3]
        is_out = self.is_miss_tower(ipoint(a_, b_, pos_clt, pi), zlim=0)
        # print(f'吸收塔遮挡率: {(is_out * wt).mean()}')

        # 求解阴影遮挡效率
        n = self.n[self.near].transpose(0, 1, 3, 4, 2)
        pc = pos_hs[self.near][:, :, None, None]
        pi_ = pi[:, :, None]
        # 出射光与所匹配镜子的交点
        outp = complex(ipoint(n, self.a[:, None, None, None], pc, pi_))
        # 入射光与所匹配镜子的交点
        inp = complex(ipoint(n, b_, pc, pi_))

        # 每个镜子四个顶点的坐标 (偏移量)
        Lw, Lh = self.Lwh.real, self.Lwh.imag
        bias = complex((np.array([[-Lw, Lh], [Lw, Lh], [Lw, -Lh], [-Lw, -Lh]]
                                 ).transpose(0, 2, 1)[:, :, None, None, None] / 2 * rota32).sum(axis=-1))[:, :, None]
        w, h = map(np.abs, [bias[0] - bias[1], bias[1] - bias[2]])
        ps = complex(pc) + bias
        # 每个镜子的入射光、出射光被遮挡的情况
        for p in (inp, outp):
            is_out_ = np.full_like(p, False, dtype=np.bool_)
            # 计算点线欧氏距离
            for i, l in enumerate([h, w, h, w]):
                dist = edist(p, ps[i], ps[(i + 1) % 4])
                is_out_ |= dist > l
            is_out *= np.all(is_out_, axis=2)

        self.eta_sb = is_out.mean(axis=0)  # 阴影遮挡效率 (入射、出射被挡)

        rota33 = self.get_rotate(self.a).transpose(2, 0, 1)
        # 将出射光分解成光锥型光线
        sigma = np.linspace(0, B, self._nm[0])
        tau = np.linspace(0, 2 * np.pi, self._nm[1], endpoint=False)
        sigma, tau = map(np.ravel, np.meshgrid(sigma, tau))
        e = np.stack([np.sin(sigma) * np.cos(tau),
                      np.sin(sigma) * np.sin(tau),
                      np.cos(sigma)], axis=-1)[:, None, None]
        e = (rota33 * e).sum(axis=-1)[:, None, :, None, None]
        # 出射光无法进入吸收塔
        self.eta_trunc = (~ self.is_miss_tower(ipoint(a_, e, pos_clt, pi), zlim=zc - hc / 2)).mean(axis=(0, 1))
        # print(f'集热器截断效率: {(self.eta_trunc * wt).mean()}')

        self.eta_pure = self.eta_cos * self.eta_at * self.eta_trunc * eta_ref
        self.eta = self.eta_sb * self.eta_pure  # 光学效率
        area = Lw * Lh
        self.E_field = dni * (unsqueeze(area) * self.eta).sum(axis=0)
        self.E_mean = self.E_field / area.sum()

    def fitness(self, E_tar, verbose=False):
        E_mean = (self.E_mean * wt).mean()
        E_field = (self.E_field * wt).mean()
        fitness = E_mean - np.maximum(E_tar - E_field, 0)
        return {'E_mean': E_mean, 'E_field': E_field, 'fitness': fitness} if verbose else fitness

    @staticmethod
    def get_rotate(n):
        # 求解定日镜的方位角、俯仰角
        dire = np.angle(complex(n))
        elev = np.arccos(n[..., -1])
        sind, sine = map(np.sin, [dire, elev])
        cosd, cose = map(np.cos, [dire, elev])
        # 利用旋转矩阵转化为绝对坐标 (xyz 偏移量) [n, 12, 3, 3, 3]
        return np.array([[sind, cosd * cose, cosd * sine],
                         [-cosd, sind * cose, sind * sine],
                         [np.zeros_like(sine), -sine, cose]])

    def is_miss_tower(self, xsp, zlim):
        # 是否错过吸收塔
        xsp -= [self.pos_clt.real, self.pos_clt.imag, 0]
        rho = np.abs(complex(xsp))
        return (xsp[..., -1] < zlim) | (xsp[..., -1] > zc + hc / 2) | (rho > rc)

    def plot_sun(self, month, ihour):
        # 绘制太阳方位
        gamma_s = np.arccos(cos_gamma_s[month - 1, ihour] + 1e-8)
        xy = rect(R, gamma_s)
        plt.scatter(xy.imag, xy.real, color='gray', s=50, marker='p')

    def plot_tower(self, fig):
        fig.add_patch(pch.Circle([self.pos_clt.real, self.pos_clt.imag], radius=rc, color='pink'))

    def plot_eta(self, key, month, ihour, fig=None):
        plt.title({'': '光学效率', 'trunc': '集热器截断效率', 'sb': '阴影遮挡效率', 'cos': '余弦效率'}[key])
        self.plot_tower(fig if fig else plt.subplot())
        self.plot_sun(month, ihour)
        eta = getattr(self, f'eta_{key}'.strip('_'))[..., month - 1, ihour]
        # plt.xticks([], []), plt.yticks([], [])
        plt.scatter(self.pos_hs.real, self.pos_hs.imag, color='deepskyblue', alpha=eta, s=20)

    def plot_hs(self, month, ihour):
        n = self.n[..., month - 1, ihour]

        fig = plt.subplot(1, 2, 1)
        phi = np.rad2deg(np.arccos(n[:, -1]))
        plt.title(f'俯仰角 ∈ [{phi.min():.1f}, {phi.max():.1f}]')
        # phi = (phi - phi.min()) / (phi.max() - phi.min()) * (1 - alpha) + alpha
        theta = np.angle(complex(n)) % (2 * np.pi)
        for *args, c in zip(self.pos_hs.real, self.pos_hs.imag, *(n[:, :2].T * 15), hcolor(theta)):
            plt.arrow(*args, color=c, width=.5)
        self.plot_tower(fig)
        self.plot_sun(month, ihour)

        # 自制颜色棒
        plt.subplot(1, 2, 2)
        yticks = np.arange(0, 361, 45)
        theta = np.linspace(0, np.pi * 2, 361)
        plt.imshow(np.repeat(hcolor(theta)[:, None], 10, axis=-2))
        plt.xticks([], []), plt.yticks(yticks, yticks)

    def hs2csv(self, id):
        data = pd.DataFrame(np.stack([self.Lwh.real, self.Lwh.imag,
                                      self.pos_hs.real, self.pos_hs.imag, self.z_hs], axis=-1))
        data.to_csv(f'result{id}.csv')

    def res2excel(self, id):
        ''' 导出可放置于论文的表格'''
        writer = pd.ExcelWriter(f'question{id}.xlsx')

        data1 = np.stack([self.eta, self.eta_cos, self.eta_sb, self.eta_trunc], axis=-2).mean(axis=0)
        data1 = np.concatenate([data1, self.E_mean[:, None]], axis=1)
        data1 = (data1 * wt).mean(axis=-1)
        data1_ = pd.DataFrame(data1, columns=['平均\n光学效率', '平均\n余弦效率', '平均阴影\n遮挡效率',
                                              '平均\n截断效率', '单位面积镜面平均输出\n热功率 (kW/m²)'],
                              index=[f'{i}月21日' for i in range(1, 13)])
        data1_.index.name = '日期'
        data1_.to_excel(writer, '表1')

        data2 = np.array([*data1[:, :4].mean(axis=0), (self.E_field * wt).mean() / 1e3, (self.E_mean * wt).mean()])
        data2 = pd.DataFrame(data2[None],
                             columns=['年平均\n光学效率', '年平均\n余弦效率', '年平均阴影\n遮挡效率',
                                      '年平均\n截断效率',
                                      '年平均输出热\n功率 (MW)', '单位面积镜面年平均\n输出热功率 (kW/m²)'])
        data2.to_excel(writer, '表2')

        writer.close()


class HeliostatField:
    # (0, 0): 定日镜场中心
    pos_clt = property(lambda self: self.xy)

    def __init__(self, tower_y, r_itva, interval):
        self.xy = tower_y * 1j
        self.dist = np.abs(self.xy)
        self.rrange = np.maximum(self.dist - R, 100.), self.dist + R - 1e-8
        # 布局的相关参数
        self.r_itva = np.poly1d(r_itva)  # 半径间隔函数: int -> m
        self(interval)

    def __call__(self, interval):
        self.pos_hs = np.array([])
        r = [self.rrange[0]]

        for i in range(100):
            theta = self.get_theta(r[-1])
            n = round(np.diff(theta).item() * r[-1] / interval)
            phi = np.linspace(*theta, n)
            self.pos_hs = np.concatenate([self.pos_hs, rect(r[-1], phi) + self.xy])
            # 使用间隔函数推导出下一个半径
            # assert itv >= interval
            r.append(r[-1] + self.r_itva(i))
            if r[-1] >= self.rrange[1]: break

    def get_theta(self, r):
        theta = np.array([- np.pi, np.pi]) / 2
        if self.dist + r < R: return theta * 2
        # 吸收塔外径 r 处的圆不被包含, 即相切
        a = np.arccos((R ** 2 + self.dist ** 2 - r ** 2) / (2 * R * self.dist))
        if self.xy.imag > 0:
            theta += [a - np.pi, -a]
        else:
            theta += [a, np.pi - a]
        # 求出交点坐标与吸收塔的相位关系
        theta = np.angle(rect(R, theta) - self.xy)
        theta[0] -= (theta[0] > theta[1]) * np.pi * 2
        return theta

    def plot(self):
        plt.scatter(self.pos_hs.real, self.pos_hs.imag), plt.show()


class HeliostatFieldPlus(HeliostatField):
    ''' 等边三角形密铺'''

    def __init__(self, tower_y, interval):
        self.xy = tower_y * 1j
        max_r = (np.abs(self.xy) + R) * 1.1

        x = np.arange(0, max_r, interval)
        x = np.concatenate([-x[1:][::-1], x])
        y = np.arange(0, max_r, np.sqrt(3) / 2 * interval)
        y = np.concatenate([-y[1:][::-1], y])
        y = y[1:] if len(y) & 1 else y

        x = np.stack([x, x + .5 * interval])
        x = np.concatenate([x for i in range(len(y) // 2)], axis=0)
        y = np.repeat(y[:, None], len(x[0]), axis=-1)
        x, y = map(np.ravel, [x, y])

        xy = x + y * 1j
        xy = xy[np.abs(xy) >= 100] + self.xy

        self.pos_hs = xy[np.abs(xy) < R]


if __name__ == '__main__':
    kwd = dict(xticks=['9:00', '10:30', '12:00', '13:30', '15:00'],
               yticks=[f'{i}.21' for i in range(1, 13)],
               xrotate=0, yrotate=0, pos=[-0.25, 0], fformat='%4.1f', cmap='Blues')

    sin_alpha_s = np.concatenate([sin_alpha_s, np.flip(sin_alpha_s[..., :2], axis=-1)], axis=-1)
    hotmap(np.rad2deg(np.arcsin(sin_alpha_s)), plt.subplot(1, 3, 1), title='太阳高度角', **kwd)

    gamma_s = np.rad2deg(np.arccos(cos_gamma_s + 1e-8))
    gamma_s = np.concatenate([gamma_s, np.flip(360 - gamma_s[..., :2], axis=-1)], axis=-1)
    hotmap(gamma_s, plt.subplot(1, 3, 2), title='太阳方位角', **kwd)
    plt.show()
