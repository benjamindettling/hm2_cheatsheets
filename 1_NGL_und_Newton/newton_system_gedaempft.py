# -*- coding: utf-8 -*-
"""
newton_system_gedaempft.py

Aufgabenstellung
================
Gegeben ist das nichtlineare Gleichungssystem

    f1(x, y) = x**2 + y**2 - 4           = 0
    f2(x, y) = exp(x - 1) + y - 2        = 0

Gesucht ist die Lösung mit dem gedämpften Newton-Verfahren.
Die Dämpfung sucht den kleinsten Wert p aus {0, ..., pmax}, sodass

    ||F(x_k + delta / 2**p)||_2 < ||F(x_k)||_2.

Der Startwert ist absichtlich nicht extrem nahe an der Lösung gewählt,
damit die Dämpfung im Iterationsverlauf sichtbar wird.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% 1) Gegebene Werte und Funktionen eintragen

x, y = sp.symbols("x y")
variablen = (x, y)

F_sp = sp.Matrix([
    x**2 + y**2 - 4,
    sp.exp(x - 1) + y - 2,
])

J_sp = F_sp.jacobian(variablen)

F = sp.lambdify([(x, y)], F_sp, modules="numpy")
J = sp.lambdify([(x, y)], J_sp, modules="numpy")

x0 = np.array([1.5, 1.5], dtype=float)
tol = 1e-10
max_iter = 30
pmax = 10


def norm_F(v):
    """Euklidische Norm des Residuums F(v)."""
    return np.linalg.norm(F(v).reshape(-1), 2)


# %% 2) Gedämpftes Newton-Verfahren für Systeme

punkte = [x0.copy()]
residuen = [norm_F(x0)]
p_werte = []

for k in range(max_iter):
    xk = punkte[-1]
    Fx = F(xk).reshape(-1)
    Jx = J(xk)

    delta = np.linalg.solve(Jx, -Fx)        # Ungedämpfter Newton-Schritt

    # Dämpfung: Schritt halbieren, bis die Residuum-Norm sinkt
    p = 0
    while p <= pmax and norm_F(xk + delta / 2**p) >= norm_F(xk):
        p += 1

    if p > pmax:
        p = 0                               # Falls nichts akzeptiert wird: voller Newton-Schritt

    damp = 1 / 2**p
    x_next = xk + damp * delta

    punkte.append(x_next.copy())
    residuen.append(norm_F(x_next))
    p_werte.append(p)

    print(
        "k={:2d}: x_k={}, ||F||_2={:.3e}, delta={}, p={}, Faktor={:.5f}".format(
            k,
            np.round(xk, 12),
            np.linalg.norm(Fx, 2),
            np.round(delta, 12),
            p,
            damp,
        )
    )

    if norm_F(x_next) < tol or np.linalg.norm(damp * delta, 2) < tol:
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
print("Gewählte p-Werte:", p_werte)


# %% 4) Plot: Nullstellenkurven und gedämpfter Newton-Pfad

xx = np.linspace(-1.3, 2.4, 400)
yy = np.linspace(-0.9, 2.4, 400)
X, Y = np.meshgrid(xx, yy)

F1 = sp.lambdify((x, y), F_sp[0], modules="numpy")
F2 = sp.lambdify((x, y), F_sp[1], modules="numpy")

plt.figure()
plt.contour(X, Y, F1(X, Y), levels=[0])
plt.contour(X, Y, F2(X, Y), levels=[0], linestyles="dashed")
plt.plot([], [], label="f1(x,y)=0")
plt.plot([], [], linestyle="dashed", label="f2(x,y)=0")
plt.plot(punkte[:, 0], punkte[:, 1], "o-", label="gedämpfter Newton-Pfad")
plt.plot(punkte[-1, 0], punkte[-1, 1], "x", markersize=10, label="Lösung")

for i, p in enumerate(punkte):
    plt.annotate(str(i), (p[0], p[1]), textcoords="offset points", xytext=(5, 5))

plt.xlabel("x")
plt.ylabel("y")
plt.title("Gedämpftes Newton-Verfahren für ein NGS")
plt.grid(True)
plt.legend()
plt.show()


# %% 5) Plot: Residuum pro Iteration

plt.figure()
plt.semilogy(np.arange(len(residuen)), residuen, "o-")
plt.xlabel("Iteration k")
plt.ylabel("||F(x_k)||_2")
plt.title("Residuum des gedämpften Newton-Verfahrens")
plt.grid(True)
plt.show()
