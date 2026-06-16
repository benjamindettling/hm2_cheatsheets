# -*- coding: utf-8 -*-
"""
Lineare Ausgleichsrechnung mit Normalgleichung
================================================

Aufgabenstellung
----------------
Gegeben sind Messwerte (x_i, y_i). Gesucht ist eine quadratische
Ausgleichsfunktion

    f(x) = lambda_0 + lambda_1*x + lambda_2*x**2

welche die Messwerte im Sinne der kleinsten Fehlerquadrate bestmöglich
approximiert.

Berechnet werden:
- Designmatrix A
- Normalgleichung A.T @ A @ lambda = A.T @ y
- Parameter lambda_0, lambda_1, lambda_2
- Residuen und Fehlerfunktional E = ||y - A lambda||_2^2
- Plot der Daten, der Ausgleichskurve und der Residuen
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)

# %% 1) Gegebene Werte

# Nicht exakt quadratische Messdaten: dadurch entsteht ein echtes
# Ausgleichsproblem und keine reine Interpolation.
xdat = np.array([0., 1., 2., 3., 4., 5., 6.])
ydat = np.array([2.1, 2.9, 5.0, 8.1, 12.4, 17.0, 23.1])

# Symbolische Variable und Basisfunktionen
x = sp.symbols("x")
basis_sp = [1, x, x**2]
basis_names = ["1", "x", "x**2"]

# Numerische Basisfunktionen für den Aufbau der Matrix A
basis_np = [sp.lambdify(x, phi, modules="numpy") for phi in basis_sp]


def basis_auswerten(phi_np, xwerte):
    """Wertet eine Basisfunktion aus; Konstanten werden auf alle x_i erweitert."""
    werte = phi_np(xwerte)
    if np.ndim(werte) == 0:
        werte = np.ones_like(xwerte, dtype=float) * float(werte)
    return np.asarray(werte, dtype=float)

# %% 2) Designmatrix A aufstellen

# A_ij = phi_j(x_i)
A = np.column_stack([basis_auswerten(phi, xdat) for phi in basis_np])
y = ydat.reshape(-1, 1)

print("Designmatrix A:")
print(A)
print("\nVektor y:")
print(y.reshape(-1))

# %% 3) Normalgleichung lösen

ATA = A.T @ A
ATy = A.T @ y

print("\nA.T @ A:")
print(ATA)
print("\nA.T @ y:")
print(ATy.reshape(-1))

# Normalgleichung: A.T A lambda = A.T y
lam = np.linalg.solve(ATA, ATy).reshape(-1)

print("\nLoesung lambda = [lambda_0, lambda_1, lambda_2]:")
print(lam)

# Symbolische Darstellung der Ausgleichsfunktion
f_sp = sum(lam[i] * basis_sp[i] for i in range(len(basis_sp)))
f_sp = sp.expand(f_sp)
print("\nAusgleichsfunktion:")
print("f(x) =", f_sp)

# %% 4) Residuen und Fehlerfunktional

yfit = A @ lam.reshape(-1, 1)
residuen = y.reshape(-1) - yfit.reshape(-1)
E = np.linalg.norm(residuen, 2)**2

print("\nResiduen r_i = y_i - f(x_i):")
for xi, yi, fi, ri in zip(xdat, ydat, yfit.reshape(-1), residuen):
    print(f"x={xi:4.1f}, y={yi:8.4f}, f(x)={fi:8.4f}, r={ri:9.5f}")

print("\nFehlerfunktional E = ||y - A lambda||_2^2:")
print(E)

# %% 5) Plot der Daten und der Ausgleichsfunktion

f_np = sp.lambdify(x, f_sp, modules="numpy")
xx = np.linspace(xdat.min() - 0.3, xdat.max() + 0.3, 400)
yy = f_np(xx)

plt.figure()
plt.plot(xdat, ydat, "o", label="Messdaten")
plt.plot(xx, yy, label="quadratische Ausgleichsfunktion")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Lineare Ausgleichsrechnung mit Normalgleichung")
plt.grid(True)
plt.legend()
plt.show()

# %% 6) Residuenplot

plt.figure()
plt.axhline(0, linewidth=1)
plt.plot(xdat, residuen, "o-")
plt.xlabel("x")
plt.ylabel("Residuum y_i - f(x_i)")
plt.title("Residuen der Normalgleichungs-Loesung")
plt.grid(True)
plt.show()
