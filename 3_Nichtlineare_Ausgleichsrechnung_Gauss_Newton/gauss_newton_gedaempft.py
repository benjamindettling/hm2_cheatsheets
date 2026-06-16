# -*- coding: utf-8 -*-
"""
gauss_newton_gedaempft.py

Aufgabe:
    Die rationale Modellfunktion

        T(x; A,B,p,q) = (A*x + B) / (x^2 + p*x + q)

    soll an Messdaten angepasst werden. Der Startwert ist absichtlich eher grob.
    Deshalb verwenden wir das gedämpfte Gauss-Newton-Verfahren.

    Pro Iteration wird zuerst der volle Gauss-Newton-Schritt delta berechnet.
    Danach wird der Schritt halbiert, bis das Fehlerfunktional kleiner wird:

        lambda_{k+1} = lambda_k + delta / 2^p

    mit kleinstem p aus {0, 1, ..., pmax}, sodass E(lambda_{k+1}) < E(lambda_k).
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% Aufgabenstellung / gegebene Werte

xdat = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
ydat = np.array([0.39, 0.89, 1.45, 1.23, 0.86, 0.83, 0.58, 0.50, 0.44, 0.34])

# Grober Startwert. Ohne Dämpfung ist der erste Schritt oft unnötig aggressiv.
lam0 = np.array([[8.0], [8.0], [0.0], [8.0]])

tol = 1e-8
max_iter = 30
pmax = 8


# %% Symbolischer Teil: Modell, Residuum und Jacobi-Matrix

x, A, B, p, q = sp.symbols("x A B p q")
lam_syms = (A, B, p, q)

T_sp = (A * x + B) / (x**2 + p * x + q)
g_sp = sp.Matrix([yi - T_sp.subs(x, xi) for xi, yi in zip(xdat, ydat)])
Dg_sp = g_sp.jacobian(lam_syms)

g_num = sp.lambdify((A, B, p, q), g_sp, modules="numpy")
Dg_num = sp.lambdify((A, B, p, q), Dg_sp, modules="numpy")
T_num = sp.lambdify((x, A, B, p, q), T_sp, modules="numpy")


def g(lam):
    return np.array(g_num(*lam.reshape(-1)), dtype=float).reshape(-1, 1)


def Dg(lam):
    return np.array(Dg_num(*lam.reshape(-1)), dtype=float)


def E(lam):
    return float(np.linalg.norm(g(lam))**2)


# %% Gedämpftes Gauss-Newton-Verfahren

lam = lam0.copy()
E_values = [E(lam)]
p_values = []

print("Gedämpftes Gauss-Newton-Verfahren")
print("k |        A        B        p        q | Dämpfung | ||Schritt|| | E(lambda)")
print("-" * 94)
print(
    f"0 | {lam[0,0]:8.5f} {lam[1,0]:8.5f} {lam[2,0]:8.5f} {lam[3,0]:8.5f} "
    f"| {'-':>8} | {'-':>10} | {E(lam):.6e}"
)

for k in range(1, max_iter + 1):
    J = Dg(lam)
    r = g(lam)

    # Der Gauss-Newton-Schritt wird hier mit Normalgleichungen berechnet.
    delta = np.linalg.solve(J.T @ J, -J.T @ r)

    # Dämpfung: Schrittweite delta / 2^p suchen.
    p_damp = 0
    step = delta
    while E(lam + step) >= E(lam) and p_damp < pmax:
        p_damp += 1
        step = delta / 2**p_damp

    # Falls bis pmax keine Verbesserung gefunden wurde, wird trotzdem der
    # kleinste getestete Schritt verwendet. Für die hier gewählte Aufgabe
    # tritt dieser Fall normalerweise nicht auf.
    lam = lam + step
    E_values.append(E(lam))
    p_values.append(p_damp)

    print(
        f"{k:1d} | {lam[0,0]:8.5f} {lam[1,0]:8.5f} {lam[2,0]:8.5f} {lam[3,0]:8.5f} "
        f"| {p_damp:8d} | {np.linalg.norm(step):10.3e} | {E(lam):.6e}"
    )

    if np.linalg.norm(step) < tol:
        break


# %% Resultate ausgeben

print("\nResultat:")
print(f"Anzahl Iterationen: {k}")
print(f"A = {lam[0,0]:.8f}")
print(f"B = {lam[1,0]:.8f}")
print(f"p = {lam[2,0]:.8f}")
print(f"q = {lam[3,0]:.8f}")
print(f"Fehlerfunktional E = {E(lam):.8e}")
print(f"Verwendete Dämpfungen p: {p_values}")


# %% Plot: Fit, Fehlerfunktional und Dämpfungswerte

xf = np.linspace(xdat.min(), xdat.max(), 600)
yf = T_num(xf, *lam.reshape(-1))

plt.figure()
plt.semilogy(xdat, ydat, "*", label="Messdaten")
plt.semilogy(xf, yf, label="gedämpfter Gauss-Newton-Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Gedämpftes Gauss-Newton-Verfahren")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure()
plt.semilogy(np.arange(len(E_values)), E_values, "o-")
plt.xlabel("Iteration k")
plt.ylabel("E(lambda_k)")
plt.title("Abnahme des Fehlerfunktionals durch Dämpfung")
plt.grid(True)
plt.tight_layout()

plt.figure()
plt.step(np.arange(1, len(p_values) + 1), p_values, where="mid")
plt.xlabel("Iteration k")
plt.ylabel("Dämpfungsexponent p")
plt.title("Gewählte Dämpfung pro Iteration")
plt.grid(True)
plt.tight_layout()

plt.show()
