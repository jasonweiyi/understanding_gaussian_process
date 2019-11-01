import numpy as np
import matplotlib.pyplot as plt
from utils import multiple_formatter



def generate_points(start, end, section=7, points_quota=[5, 10], noise=0.1):
    np.random.seed(1234)
    x = []

    section_length = (end - start) / float(section)
    for s in range(section):
        # section_start = start * float(s) / float(section) + start
        # section_end = end * float(s+1) / float(section)
        section_start = s * section_length + start
        section_end = (s + 1) * section_length + start

        if s % 2 == 0:
            points = points_quota[0]
        else:
            points = points_quota[1]
        x.extend(np.linspace(section_start, section_end, points).tolist())

    x = np.array(x)
    data_points = len(x)
    y = np.sin(x) + np.random.normal(scale=noise, size=len(x))


    for xx, yy in zip(x, y):
        print(str(xx) +',' + str(yy))

    return x, y


# x, y = generate_points(start=0, end=4*np.pi, section=7, points_quota=[5, 10])
#
#
# s = [60] * len(x)
# plt.scatter(x, y, marker='x', s=s)
#
# ax = plt.gca()
# ax.legend()
# plt.xlim(x[0], x[-1])
# ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
# ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
# ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
#
# plt.show()



