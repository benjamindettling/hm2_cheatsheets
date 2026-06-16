# -*- coding: utf-8 -*-
"""
gauss_newton_normalgleichung.py

Aufgabe:
    An gegebene Messdaten soll die nichtlineare Modellfunktion

        f(x; a,b,c) = a * exp(-b*x) + c

    angepasst werden. Gesucht sind die Parameter lambda = (a,b,c)^T,
    welche das Fehlerfunktional

        E(lambda) = sum_i (y_i - f(x_i; lambda))^2

    minimieren.

    Das Skript löst das nichtlineare Ausgleichsproblem mit dem
    Gauss-Newton-Verfahren über die Normalgleichung

        Dg(lambda_k)^T Dg(lambda_k) delta_k = -Dg(lambda_k)^T g(lambda_k)

    und plottet Datenpunkte, Fitfunktion und Residuen.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% Aufgabenstellung / gegebene Werte

# Messdaten: abklingender Prozess mit Grenzwert, z.B. Abkühlung oder Entladung.
xdat = np.array([0., 1., 2., 3., 4., 5.])
ydat = np.array([5.10, 3.75, 3.02, 2.62, 2.37, 2.21])

# Startwert für lambda = (a, b, c)^T.
# Die Werte sind bewusst nur grobe Schätzungen.
lam0 = np.array([[3.0], [0.6], [1.5]])

tol = 1e-8
max_iter = 20


# %% Symbolischer Teil: Modell, Residuum und Jacobi-Matrix

x, a, b, c = sp.symbols("x a b c")
lam_syms = (a, b, c)

# Nichtlineare Modellfunktion.
f_sp = a * sp.exp(-b * x) + c

# Residuum g(lambda) = y - f(x, lambda) für alle Datenpunkte.
g_sp = sp.Matrix([yi - f_sp.subs(x, xi) for xi, yi in zip(xdat, ydat)])

# Jacobi-Matrix von g bezüglich der Parameter a, b, c.
Dg_sp = g_sp.jacobian(lam_syms)

print("Symbolische Jacobi-Matrix Dg(lambda):")
sp.print_latex(Dg_sp)
print(Dg_sp)
print("")

# Numerisch auswertbare Funktionen aus den sympy-Ausdrücken erzeugen.
g_num = sp.lambdify((a, b, c), g_sp, modules="numpy")
Dg_num = sp.lambdify((a, b, c), Dg_sp, modules="numpy")
f_num = sp.lambdify((x, a, b, c), f_sp, modules="numpy")


def g(lam):
    """Residuenvektor als Spaltenvektor."""
    return np.array(g_num(*lam.reshape(-1)), dtype=float).reshape(-1, 1)


def Dg(lam):
    """Jacobi-Matrix des Residuenvektors."""
    return np.array(Dg_num(*lam.reshape(-1)), dtype=float)


def E(lam):
    """Fehlerfunktional E(lambda) = ||g(lambda)||_2^2."""
    return float(np.linalg.norm(g(lam))**2)


# %% Gauss-Newton-Verfahren mit Normalgleichung

lam = lam0.copy()
E_values = [E(lam)]

print("Gauss-Newton mit Normalgleichung")
print("k |        a        b        c | ||delta||_2 | E(lambda)")
print("-" * 68)
print(f"0 | {lam[0,0]:8.5f} {lam[1,0]:8.5f} {lam[2,0]:8.5f} | {'-':>11} | {E(lam):.6e}")

for k in range(1, max_iter + 1):
    # Lineares Ausgleichsproblem der aktuellen Linearisierung.
    J = Dg(lam)
    r = g(lam)

    # Normalgleichung: J.T @ J @ delta = -J.T @ r
    delta = np.linalg.solve(J.T @ J, -J.T @ r)

    # Newton-artiges Parameter-Update.
    lam = lam + delta
    E_values.append(E(lam))

    print(
        f"{k:1d} | {lam[0,0]:8.5f} {lam[1,0]:8.5f} {lam[2,0]:8.5f} "
        f"| {np.linalg.norm(delta):11.3e} | {E(lam):.6e}"
    )

    # Abbruch, sobald die Parameteränderung klein genug ist.
    if np.linalg.norm(delta) < tol:
        break


# %% Resultate ausgeben

print("\nResultat:")
print(f"Anzahl Iterationen: {k}")
print(f"a = {lam[0,0]:.8f}")
print(f"b = {lam[1,0]:.8f}")
print(f"c = {lam[2,0]:.8f}")
print(f"Fehlerfunktional E = {E(lam):.8e}")
print(f"Residuen = {g(lam).reshape(-1)}")


# %% Plot: Daten, Fit und Residuen

xf = np.linspace(xdat.min(), xdat.max(), 400)
yf = f_num(xf, *lam.reshape(-1))
yfit_dat = f_num(xdat, *lam.reshape(-1))
residuen = ydat - yfit_dat

plt.figure()
plt.plot(xdat, ydat, "o", label="Messdaten")
plt.plot(xf, yf, label="Gauss-Newton-Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Nichtlineare Ausgleichsrechnung: Exponentialmodell")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure()
plt.axhline(0, linewidth=1)
plt.stem(xdat, residuen, basefmt=" ")
plt.xlabel("x")
plt.ylabel("Residuum y_i - f(x_i)")
plt.title("Residuen des nichtlinearen Fits")
plt.grid(True)
plt.tight_layout()

plt.figure()
plt.semilogy(np.arange(len(E_values)), E_values, "o-")
plt.xlabel("Iteration k")
plt.ylabel("E(lambda_k)")
plt.title("Abnahme des Fehlerfunktionals")
plt.grid(True)
plt.tight_layout()

plt.show()
