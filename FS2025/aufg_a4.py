# -*- coding: utf-8 -*-
"""
Created on Mon May 19 13:02:32 2025

@author: delo
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# %% a)
x, y = sp.symbols("x y")
"""
g = 10*(x-0.42)**2 + (y-x**2)**2 + 4
g = g.expand()
print(sp.latex(g))
"""
g = x**4 - 2 * x**2 * y + 10 * x**2 - 8.4 * x + y**2 + 5.764
g_ = sp.Matrix([g])  # macht g zur Matrix, da .jacobian nur auf Matrizen funktioniert

Dg = g_.jacobian((x, y))  # 1P

f1 = sp.lambdify((x, y), Dg[0], modules="numpy")  # 1P
f2 = sp.lambdify((x, y), Dg[1], modules="numpy")

# Graphische Lösung                                       # 2P
X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))

c1 = plt.contour(X, Y, f1(X, Y), levels=[0], colors="r")
h1, _ = c1.legend_elements()
c2 = plt.contour(X, Y, f2(X, Y), levels=[0], colors="b")
h2, _ = c2.legend_elements()
plt.legend([h1[0], h2[0]], ["f1", "f2"])

# Minimalstelle gemäss Plot
x0 = np.array([0.41, 0.18])

# Exakte / symbolische Lösung des Gleichungssystems Dg = 0
sol = sp.solve([Dg[0], Dg[1]], (x, y), dict=True)

print(sol)

# Erste Lösung als NumPy-Array verwenden
x0_exact = np.array([float(sol[0][x]), float(sol[0][y])])

print("Exakter berechneter Punkt:", x0_exact)


plt.plot(*x0, "+k")
plt.plot(*x0_exact, "+g")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Grafische Lösung von f1=0 und f2=0")
plt.grid(True)

plt.show()


# %% b) Numerische Lösung mit Newton-Verfaren


def newton(f, df, x0, tol, max_iter=20):
    num_iter = 0
    x = np.array(x0)
    err = tol + 1
    while err > tol:
        fx = f(x).reshape(-1)
        Jx = df(x)
        d = np.linalg.solve(Jx, fx)
        x -= d

        err = np.linalg.norm(f(x), np.inf)  # 1P

        num_iter += 1
        if num_iter > max_iter:
            raise Exception("Keine Konvergenz")
    return (x, num_iter)


Hg = Dg.jacobian((x, y))  # 1P

f = sp.lambdify([(x, y)], Dg, modules="numpy")  # 1P
Df = sp.lambdify([(x, y)], Hg, modules="numpy")

x0 = np.array([0.2, 0.8])
tol = 1e-8
xk, k = newton(f, Df, x0, tol)  # 1P

# %% c)
print("x_{} = {}".format(k, xk))  # 1P
g = sp.lambdify([(x, y)], g, modules="numpy")
print("g(x_{}) = {}".format(k, g(xk)))  # 1P

"""
x_4 = [0.42   0.1764]
g(x_4) = 4.0
"""
