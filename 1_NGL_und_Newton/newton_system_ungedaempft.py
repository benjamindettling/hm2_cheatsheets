# -*- coding: utf-8 -*-
"""
newton_system_ungedaempft.py

Aufgabenstellung
================
Gegeben ist das nichtlineare Gleichungssystem

    f1(x, y) = 20 - 18*x - 2*y**2 = 0
    f2(x, y) = -4*y*(x - y**2)    = 0

Gesucht ist die Lösung in der Nähe von x0 = (1.1, 0.9)^T mit dem
ungedämpften Newton-Verfahren für Systeme.

Rechenidee:
In jeder Iteration wird das lineare Gleichungssystem

    J(x_k) * delta_k = -F(x_k)

nach delta_k gelöst. Danach gilt x_{k+1} = x_k + delta_k.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% 1) Gegebene Werte und Funktionen eintragen

x, y = sp.symbols("x y")
variablen = (x, y)

F_sp = sp.Matrix([
    20 - 18*x - 2*y**2,
    -4*y*(x - y**2),
])

J_sp = F_sp.jacobian(variablen)             # Jacobi-Matrix symbolisch berechnen

F = sp.lambdify([(x, y)], F_sp, modules="numpy")
J = sp.lambdify([(x, y)], J_sp, modules="numpy")

x0 = np.array([1.1, 0.9], dtype=float)
tol = 1e-10
max_iter = 20


# %% 2) Newton-Verfahren für Systeme

punkte = [x0.copy()]
residuen = [np.linalg.norm(F(x0).reshape(-1), 2)]

for k in range(max_iter):
    xk = punkte[-1]
    Fx = F(xk).reshape(-1)
    Jx = J(xk)

    delta = np.linalg.solve(Jx, -Fx)        # J(x_k)*delta = -F(x_k)
    x_next = xk + delta

    punkte.append(x_next.copy())
    residuen.append(np.linalg.norm(F(x_next).reshape(-1), 2))

    print(
        "k={:2d}: x_k={}, F-Norm={:.3e}, delta={}".format(
            k, np.round(xk, 12), np.linalg.norm(Fx, 2), np.round(delta, 12)
        )
    )

    if np.linalg.norm(F(x_next).reshape(-1), 2) < tol or np.linalg.norm(delta, 2) < tol:
        break

punkte = np.array(punkte)
residuen = np.array(residuen)


# %% 3) Resultat ausgeben

print("\nF(x, y) =")
sp.pprint(F_sp)
print("\nJacobi-Matrix J(x, y) =")
sp.pprint(J_sp)

print("\nResultat:")
print("Lösung [x, y] ≈", np.round(punkte[-1], 12))
print("F(Lösung)   ≈", F(punkte[-1]).reshape(-1))
print("||F||_2     ≈ {:.3e}".format(residuen[-1]))
print("Iterationen =", len(punkte) - 1)


# %% 4) Plot: Nullstellenkurven und Newton-Pfad

xx = np.linspace(0.75, 1.25, 300)
yy = np.linspace(0.65, 1.15, 300)
X, Y = np.meshgrid(xx, yy)

F1 = sp.lambdify((x, y), F_sp[0], modules="numpy")
F2 = sp.lambdify((x, y), F_sp[1], modules="numpy")

plt.figure()
c1 = plt.contour(X, Y, F1(X, Y), levels=[0])
c2 = plt.contour(X, Y, F2(X, Y), levels=[0], linestyles="dashed")

# Dummy-Linien für saubere Legende
plt.plot([], [], label="f1(x,y)=0")
plt.plot([], [], linestyle="dashed", label="f2(x,y)=0")
plt.plot(punkte[:, 0], punkte[:, 1], "o-", label="Newton-Pfad")
plt.plot(punkte[-1, 0], punkte[-1, 1], "x", markersize=10, label="Lösung")

for i, p in enumerate(punkte):
    plt.annotate(str(i), (p[0], p[1]), textcoords="offset points", xytext=(5, 5))

plt.xlabel("x")
plt.ylabel("y")
plt.title("Ungedämpftes Newton-Verfahren für ein NGS")
plt.grid(True)
plt.legend()
plt.show()


# %% 5) Plot: Residuum pro Iteration

plt.figure()
plt.semilogy(np.arange(len(residuen)), residuen, "o-")
plt.xlabel("Iteration k")
plt.ylabel("||F(x_k)||_2")
plt.title("Konvergenz des Newton-Verfahrens")
plt.grid(True)
plt.show()
