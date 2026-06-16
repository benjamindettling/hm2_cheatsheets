# -*- coding: utf-8 -*-
"""
Lineare Ausgleichsrechnung mit frei wählbaren Basisfunktionen
=============================================================

Aufgabenstellung
----------------
Gegeben sind Messwerte (x_i, y_i). Die Ansatzfunktion ist nicht einfach ein
Polynom, aber sie ist linear in den unbekannten Parametern:

    f(x) = lambda_0*1 + lambda_1*x + lambda_2*exp(-0.6*x)

Die Basisfunktionen sind also

    phi_0(x) = 1,
    phi_1(x) = x,
    phi_2(x) = exp(-0.6*x).

Gesucht sind die Parameter lambda_0, lambda_1 und lambda_2 im Sinne der
kleinsten Fehlerquadrate.

Das Skript löst das Ausgleichsproblem standardmässig mit QR-Zerlegung und
zeigt zum Vergleich auch die Lösung über die Normalgleichung.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)

# %% 1) Gegebene Werte

xdat = np.array([0., 0.5, 1., 1.5, 2., 3., 4., 5., 6.])
ydat = np.array([4.35, 3.67, 3.10, 2.79, 2.55, 2.36, 2.50, 2.82, 3.14])

# Symbolische Variable
x = sp.symbols("x")

# Frei wählbare Basisfunktionen.
# Wichtig: Die Funktion darf in x kompliziert sein, aber die Parameter lambda_j
# müssen linear auftreten.
basis_sp = [1, x, sp.exp(-0.6*x)]
basis_labels = ["1", "x", "exp(-0.6*x)"]

# Numerische Versionen der Basisfunktionen
basis_np = [sp.lambdify(x, phi, modules="numpy") for phi in basis_sp]


def basis_auswerten(phi_np, xwerte):
    """Wertet eine Basisfunktion aus; Konstanten werden auf alle x_i erweitert."""
    werte = phi_np(xwerte)
    if np.ndim(werte) == 0:
        werte = np.ones_like(xwerte, dtype=float) * float(werte)
    return np.asarray(werte, dtype=float)

# %% 2) Designmatrix A aufstellen

A = np.column_stack([basis_auswerten(phi, xdat) for phi in basis_np])
y = ydat.reshape(-1, 1)

print("Basisfunktionen:")
for j, label in enumerate(basis_labels):
    print(f"phi_{j}(x) = {label}")

print("\nDesignmatrix A:")
print(A)

# %% 3) Lösung mit QR-Zerlegung

Q, R = np.linalg.qr(A, mode="reduced")
lam_qr = np.linalg.solve(R, Q.T @ y).reshape(-1)

print("\nLoesung mit QR:")
for j, wert in enumerate(lam_qr):
    print(f"lambda_{j} = {wert:.8f}")

# Vergleich: Lösung mit Normalgleichung
lam_normal = np.linalg.solve(A.T @ A, A.T @ y).reshape(-1)

print("\nVergleichsloesung mit Normalgleichung:")
for j, wert in enumerate(lam_normal):
    print(f"lambda_{j} = {wert:.8f}")

print("\nDifferenz QR - Normalgleichung:")
print(lam_qr - lam_normal)

print("\nKonditionszahlen:")
print("cond(A)       =", np.linalg.cond(A))
print("cond(A.T @ A) =", np.linalg.cond(A.T @ A))
print("cond(R)       =", np.linalg.cond(R))

# %% 4) Ausgleichsfunktion und Fehlerfunktional

f_sp = sp.expand(sum(lam_qr[j] * basis_sp[j] for j in range(len(basis_sp))))
f_np = sp.lambdify(x, f_sp, modules="numpy")

print("\nAusgleichsfunktion:")
print("f(x) =", f_sp)

yfit = A @ lam_qr.reshape(-1, 1)
residuen = y.reshape(-1) - yfit.reshape(-1)
E = np.linalg.norm(residuen, 2)**2

print("\nResiduen und Funktionswerte:")
for xi, yi, fi, ri in zip(xdat, ydat, yfit.reshape(-1), residuen):
    print(f"x={xi:4.1f}, y={yi:8.4f}, f(x)={fi:8.4f}, r={ri:9.5f}")

print("\nFehlerfunktional E = ||y - A lambda||_2^2:")
print(E)

# %% 5) Plot Daten und Ausgleichsfunktion

xx = np.linspace(xdat.min(), xdat.max(), 500)

plt.figure()
plt.plot(xdat, ydat, "o", label="Messdaten")
plt.plot(xx, f_np(xx), label="Fit mit frei gewählten Basisfunktionen")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Lineare Ausgleichsrechnung mit allgemeinen Basisfunktionen")
plt.grid(True)
plt.legend()
plt.show()

# %% 6) Residuenplot

plt.figure()
plt.axhline(0, linewidth=1)
plt.plot(xdat, residuen, "o-")
plt.xlabel("x")
plt.ylabel("Residuum y_i - f(x_i)")
plt.title("Residuen des allgemeinen linearen Ausgleichsproblems")
plt.grid(True)
plt.show()
