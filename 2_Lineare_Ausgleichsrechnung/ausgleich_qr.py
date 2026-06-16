# -*- coding: utf-8 -*-
"""
Lineare Ausgleichsrechnung mit QR-Zerlegung
===========================================

Aufgabenstellung
----------------
Es werden dieselben Messdaten wie in der Normalgleichungs-Aufgabe verwendet.
Gesucht ist wieder eine quadratische Ausgleichsfunktion

    f(x) = lambda_0 + lambda_1*x + lambda_2*x**2

Diesmal wird das lineare Ausgleichsproblem über die QR-Zerlegung gelöst:

    A = Q R
    R lambda = Q.T y

Zusätzlich werden die Konditionszahlen verglichen, um den Stabilitätsvorteil
gegenüber der expliziten Bildung von A.T @ A sichtbar zu machen.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)

# %% 1) Gegebene Werte

xdat = np.array([0., 1., 2., 3., 4., 5., 6.])
ydat = np.array([2.1, 2.9, 5.0, 8.1, 12.4, 17.0, 23.1])

x = sp.symbols("x")
basis_sp = [1, x, x**2]
basis_np = [sp.lambdify(x, phi, modules="numpy") for phi in basis_sp]


def basis_auswerten(phi_np, xwerte):
    """Wertet eine Basisfunktion aus; Konstanten werden auf alle x_i erweitert."""
    werte = phi_np(xwerte)
    if np.ndim(werte) == 0:
        werte = np.ones_like(xwerte, dtype=float) * float(werte)
    return np.asarray(werte, dtype=float)

# %% 2) Designmatrix A und y-Vektor

A = np.column_stack([basis_auswerten(phi, xdat) for phi in basis_np])
y = ydat.reshape(-1, 1)

print("Designmatrix A:")
print(A)

# %% 3) QR-Zerlegung und Lösung des Ausgleichsproblems

# Reduzierte QR-Zerlegung: Q hat nur so viele Spalten wie A.
Q, R = np.linalg.qr(A, mode="reduced")
rechte_seite = Q.T @ y

# R ist eine obere Dreiecksmatrix. Mathematisch entspricht das
# Rückwärtseinsetzen; numerisch erledigt np.linalg.solve diesen Schritt.
lam_qr = np.linalg.solve(R, rechte_seite).reshape(-1)

print("\nMatrix Q:")
print(Q)
print("\nMatrix R:")
print(R)
print("\nQ.T @ y:")
print(rechte_seite.reshape(-1))

print("\nLoesung mit QR lambda = [lambda_0, lambda_1, lambda_2]:")
print(lam_qr)

# Zum Vergleich: Lösung über Normalgleichung
lam_normal = np.linalg.solve(A.T @ A, A.T @ y).reshape(-1)
print("\nVergleichsloesung mit Normalgleichung:")
print(lam_normal)
print("\nDifferenz QR - Normalgleichung:")
print(lam_qr - lam_normal)

# Konditionszahlen
print("\nKonditionszahlen:")
print("cond(A)       =", np.linalg.cond(A))
print("cond(A.T @ A) =", np.linalg.cond(A.T @ A))
print("cond(R)       =", np.linalg.cond(R))

# %% 4) Ausgleichsfunktion, Residuen und Fehlerfunktional

f_sp = sp.expand(sum(lam_qr[i] * basis_sp[i] for i in range(len(basis_sp))))
f_np = sp.lambdify(x, f_sp, modules="numpy")

print("\nAusgleichsfunktion aus QR-Loesung:")
print("f(x) =", f_sp)

yfit = A @ lam_qr.reshape(-1, 1)
residuen = y.reshape(-1) - yfit.reshape(-1)
E = np.linalg.norm(residuen, 2)**2

print("\nResiduen:")
for xi, yi, fi, ri in zip(xdat, ydat, yfit.reshape(-1), residuen):
    print(f"x={xi:4.1f}, y={yi:8.4f}, f(x)={fi:8.4f}, r={ri:9.5f}")

print("\nFehlerfunktional E =", E)

# %% 5) Plot Daten und Ausgleichsfunktion

xx = np.linspace(xdat.min() - 0.3, xdat.max() + 0.3, 400)

plt.figure()
plt.plot(xdat, ydat, "o", label="Messdaten")
plt.plot(xx, f_np(xx), label="QR-Ausgleichsfunktion")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Lineare Ausgleichsrechnung mit QR-Zerlegung")
plt.grid(True)
plt.legend()
plt.show()

# %% 6) Residuenplot

plt.figure()
plt.axhline(0, linewidth=1)
plt.plot(xdat, residuen, "o-")
plt.xlabel("x")
plt.ylabel("Residuum y_i - f(x_i)")
plt.title("Residuen der QR-Loesung")
plt.grid(True)
plt.show()
