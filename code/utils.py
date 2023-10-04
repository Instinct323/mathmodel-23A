import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.chdir('../tmp')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

DTYPE = np.float64
POS_HS = pd.read_excel('../info/附件.xlsx').to_numpy(dtype=DTYPE).T
POS_HS = POS_HS[0] + POS_HS[1] * 1j


def rect(r, phi):
    # 极坐标 -> 复数
    return r * (np.cos(phi) + 1j * np.sin(phi))


def complex(arr):
    return arr[..., 0] + arr[..., 1] * 1j


def unitize(x, axis=None):
    return x / np.linalg.norm(x, axis=axis, keepdims=True)


def make_grid(w, h, percent=True, center=False, axis=-1):
    coord = np.stack(np.meshgrid(*map(np.arange, (w, h))), axis=axis).astype(np.float32)
    if center: coord += .5
    return coord / np.array([w, h]) if percent else coord


def ipoint(n, e, pc, pi):
    ''' n: 平面的法向量 [..., 3]
        e: 线的方向向量 [..., 3]
        pc: 平面所过定点 [..., 3]
        pi: 线所过定点 [..., 3]
        return: 线与平面的交点 [..., 3]'''
    dot = (e * n).sum(axis=-1, keepdims=True) + 1e-8
    en = (e[..., None] * n[..., None, :])[..., None, :, :]

    pc, pi = np.broadcast_arrays(pc, pi)
    pci = pc - pi
    zero = np.zeros_like(pci[..., 0])
    w = np.array(
        [[[pc[..., 0], pci[..., 1], pci[..., 2]], [zero, pi[..., 0], zero], [zero, zero, pi[..., 0]]],
         [[pi[..., 1], zero, zero], [pci[..., 0], pc[..., 1], pci[..., 2]], [zero, zero, pi[..., 1]]],
         [[pi[..., 2], zero, zero], [zero, pi[..., 2], zero], [pci[..., 0], pci[..., 1], pc[..., 2]]]]
    )
    return (w.transpose(*range(3, w.ndim), *range(3)) * en).sum(axis=(-1, -2)) / dot


def edist(p, p1, p2):
    d01 = p1 - p2
    dv0 = p - p1
    return np.abs(d01.imag * dv0.real - d01.real * dv0.imag) / (np.abs(d01) + 1e-8)


def hcolor(theta):
    theta = theta / np.pi * 90
    s = v = np.full_like(theta, fill_value=255, dtype=np.uint8)
    hsv = np.stack([theta.astype(np.uint8), s, v], axis=-1)
    return cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB)[0] / 255


def hotmap(array, fig=None, pos=0, fformat='%f', cmap='Blues', size=10, title=None, colorbar=False,
           xticks=None, yticks=None, xlabel=None, ylabel=None, xrotate=0, yrotate=90):
    pos = np.array([-.1, .05]) + pos
    # 去除坐标轴
    fig = plt.subplot() if fig is None else fig
    plt.title(title)
    for key in 'right', 'top', 'left', 'bottom':
        fig.spines[key].set_color('None')
    fig.xaxis.set_ticks_position('top')
    fig.xaxis.set_label_position('top')
    # 显示热力图
    plt.imshow(array, cmap=cmap, vmax=(array.max() - array.min()) * 0.15 + array.max())
    if colorbar: plt.colorbar()
    # 标注数据信息
    for i, row in enumerate(array):
        for j, item in enumerate(row):
            if np.isfinite(item):
                plt.annotate(fformat % item, pos + [j, i], size=size)
    # 坐标轴标签
    plt.xticks(range(len(array[0])), xticks, rotation=xrotate)
    plt.yticks(range(len(array)), yticks, rotation=yrotate)
    plt.xlabel(xlabel), plt.ylabel(ylabel)


class Result(pd.DataFrame):
    __exist__ = []
    orient = 'index'
    project = Path()
    file = property(lambda self: self.project / 'result.json')

    def __init__(self, project: Path, title: tuple):
        self.project = project
        super().__init__(pd.read_json(self.file, orient=self.orient)) \
            if self.file.is_file() else super().__init__(columns=title)
        # 检查项目是否复用
        if project in self.__exist__:
            raise AssertionError(f'Multiple <{type(self).__name__}> are used in {project}')
        self.__exist__.append(project)

    def record(self, metrics, i: int = None):
        i = len(self) if i is None else i
        self.loc[i] = metrics
        super().__init__(self.convert_dtypes())
        self.to_json(self.file, orient=self.orient, indent=4)


if __name__ == '__main__':
    import sympy as sp

    x, y, z = sp.symbols('x, y, z')
    xc, yc, zc = sp.symbols('xc, yc, zc')
    xi, yi, zi = sp.symbols('xi, yi, zi')
    nx, ny, nz = sp.symbols('nx, ny, nz')
    ex, ey, ez = sp.symbols('ex, ey, ez')

    equ = [
        nx * (xc - x) + ny * (yc - y) + nz * (zc - z),
        (x - xi) / ex - (y - yi) / ey,
        (x - xi) / ex - (z - zi) / ez
    ]
    for k, v in sp.solve(equ, [x, y, z]).items(): print(k, '=', sp.factor(v))
