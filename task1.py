import math
import matplotlib.pyplot as plt
from math import log10
import pandas as pd

def solve_test(n):
    a = 0
    b = 4
    h = (b - a) / n

    def f1(t, y1, y2):
        return y2 * math.exp(2*t)

    def f2(t, y1, y2):
        return y1 * (-math.exp(- 2*t))

    y1 = [0 for i in range(n)]
    y2 = [0 for i in range(n)]
    _h = [a + i * h for i in range(n)]

    y1[0] = 1
    y2[0] = 1

    for i in range(0, n - 1):
        t = a + i * h

        k1 = f1(t, y1[i], y2[i])
        m1 = f2(t, y1[i], y2[i])

        k2 = f1(t + h / 2, y1[i] + h * k1 / 2, y2[i] + h * m1 / 2)
        m2 = f2(t + h / 2, y1[i] + h * k1 / 2, y2[i] + h * m1 / 2)

        k3 = f1(t + h / 2, y1[i] + h * k2 / 2, y2[i] + h * m2 / 2)
        m3 = f2(t + h / 2, y1[i] + h * k2 / 2, y2[i] + h * m2 / 2)

        k4 = f1(t + h, y1[i] + h * k3, y2[i] + h * m3)
        m4 = f2(t + h, y1[i] + h * k3, y2[i] + h * m3)

        y1[i + 1] = y1[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        y2[i + 1] = y2[i] + h * (m1 + 2 * m2 + 2 * m3 + m4) / 6

    u1= [math.exp(_) for _ in _h]
    u2= [math.exp(-_) for _ in _h]

    err1 = [0 for _ in range(n)]
    err2 = [0 for _ in range(n)]

    for i in range(n):
        err1[i] = abs(y1[i] - u1[i])
        err2[i] = abs(y2[i] - u2[i])

    # dt = pd.DataFrame({"y1": y1, "y2": y2, "u": u, "err": err1})

    dt = pd.DataFrame({"y1": y1, "y2": y2, "u1": u1, "u2": u2,"err1": err1,"err2": err2})
    # dt.to_excel('temp.xlsx')
    print(dt)

    max_err1 = max(err1)
    print(max_err1)
    max_err2 = max(err2)
    print(max_err2)
    print(f'При шаге {h}')
    print(f'Погрешность {max_err1}\n')
    print(max_err1 <= h ** 4)
    print(max_err1 / (h ** 4))

    return max_err1, h

def logs():
    ns = list()
    errors = list()
    hs = list()

    for n in range(40, 200):
        ns.append(n)
        error, h = solve_test(n)
        errors.append(error)
        hs.append(h)

    alphas = [0 for _ in range(len(errors))]
    for i in range(len(hs) - 1):
        alphas[i] = log10(errors[i + 1] / errors[i]) / log10(hs[i + 1] / hs[i])

    alpha = sum(alphas) / len(alphas)

    print(f'Alpha = {alpha}')

    errors = list()
    hs = list()
    for n in range(40, 200):
        ns.append(n)
        error, h = solve_test(n)
        errors.append(error / h ** alpha)
        hs.append(h)

    print(hs)
    print(errors)
    solve_test()


def simple():
    ns = list()
    errors = list()
    hs = list()
    for n in range(10, 40):
        ns.append(n)
        error, h = solve_test(n)
        errors.append(error / h)
        hs.append(h)

    plt.plot(hs, errors)
    plt.show()



if __name__ == '__main__':
     simple()
     # logs()