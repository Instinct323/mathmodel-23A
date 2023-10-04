import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import minmax_scale

x = pd.read_csv('result3.csv', index_col=0).to_numpy()
area = (x[:, 0] * x[:, 1]).sum()
print(area)
y = minmax_scale(x).clip(min=0, max=1)

kwd = dict(marker='h', s=10)

plt.subplot(1, 2, 1)
plt.scatter(x[:, 2], x[:, 3], alpha=y[..., 0], **kwd)
plt.subplot(1, 2, 2)
plt.scatter(x[:, 2], x[:, 3], alpha=y[..., 4], **kwd)
plt.show()
