# -*- coding: utf-8 -*-
"""
linearisierung_jacobi_tangentialebene.py

Aufgabenstellung
================
Gegeben ist die skalarwertige Funktion

    f(x, y) = sin(x*y) + 0.3*x**2 - 0.2*y**2 + 1.

Gesucht ist die Linearisierung von f im Punkt

    (x0, y0) = (1.0, 0.8).

Für eine skalarwertige Funktion f: R^2 -> R ist die Jacobi-Matrix ein
Zeilenvektor mit den partiellen Ableitungen. Die Linearisierung ist die
Tangentialebene

    g(x, y) = f(x0, y0) + f_x(x0, y0)*(x - x0) + f_y(x0, y0)*(y - y0).
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% 1) Gegebene Funktion und Entwicklungspunkt eintragen

x, y = sp.symbols("x y")

f_sp = sp.sin(x*y) + sp.Rational(3, 10)*x**2 - sp.Rational(1, 5)*y**2 + 1
punkt = np.array([1.0, 0.8], dtype=float)

# Als 1x1-Matrix formulieren, damit jacobian direkt verwendet werden kann
f_matrix = sp.Matrix([f_sp])
J_sp = f_matrix.jacobian((x, y))            # Jacobi-Matrix / Gradient als Zeile


# %% 2) Jacobi-Matrix und Linearisierung berechnen

f0 = float(f_sp.subs({x: punkt[0], y: punkt[1]}))
J0 = np.array(J_sp.subs({x: punkt[0], y: punkt[1]}), dtype=float).reshape(-1)

# Symbolische Linearisierung g(x, y)
g_sp = f0 + J0[0]*(x - punkt[0]) + J0[1]*(y - punkt[1])
g_sp = sp.expand(g_sp)

f = sp.lambdify((x, y), f_sp, modules="numpy")
g = sp.lambdify((x, y), g_sp, modules="numpy")


# %% 3) Resultate ausgeben

print("Funktion f(x, y):")
sp.pprint(f_sp)

print("\nJacobi-Matrix Df(x, y):")
sp.pprint(J_sp)

print("\nEntwicklungspunkt:")
print("x0 =", punkt)
print("f(x0) = {:.12f}".format(f0))
print("Df(x0) =", J0)

print("\nLinearisierung / Tangentialebene g(x, y):")
sp.pprint(g_sp)

# Kontrollauswertung an einem Punkt in der Nähe
kontrollpunkt = np.array([1.08, 0.75])
print("\nKontrolle nahe beim Entwicklungspunkt:")
print("Punkt              =", kontrollpunkt)
print("f(Punkt)           = {:.12f}".format(f(*kontrollpunkt)))
print("g(Punkt)           = {:.12f}".format(g(*kontrollpunkt)))
print("Absoluter Fehler   = {:.3e}".format(abs(f(*kontrollpunkt) - g(*kontrollpunkt))))


# %% 4) Plot: Fläche f und Tangentialebene g

xx = np.linspace(0.4, 1.6, 80)
yy = np.linspace(0.2, 1.4, 80)
X, Y = np.meshgrid(xx, yy)
Zf = f(X, Y)
Zg = g(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(X, Y, Zf, alpha=0.65)
ax.plot_surface(X, Y, Zg, alpha=0.35)
ax.scatter(punkt[0], punkt[1], f0, s=60, label="Entwicklungspunkt")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Funktion f(x,y) und Tangentialebene g(x,y)")
ax.legend()
plt.show()


# %% 5) Zusatzplot: Fehler |f-g| in der Umgebung

fehler = np.abs(Zf - Zg)

plt.figure()
plt.contourf(X, Y, fehler, levels=20)
plt.colorbar(label="|f(x,y) - g(x,y)|")
plt.plot(punkt[0], punkt[1], "o", label="Entwicklungspunkt")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linearisierungsfehler in der Umgebung")
plt.grid(True)
plt.legend()
plt.show()
