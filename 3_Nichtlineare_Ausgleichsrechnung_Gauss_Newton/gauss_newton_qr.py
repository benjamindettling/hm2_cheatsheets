# -*- coding: utf-8 -*-
"""
gauss_newton_qr.py

Aufgabe:
    An Messdaten soll eine rationale Modellfunktion angepasst werden:

        T(x; A,B,p,q) = (A*x + B) / (x^2 + p*x + q)

    Die Parameter lambda = (A,B,p,q)^T treten nichtlinear im Nenner auf.
    Deshalb verwenden wir Gauss-Newton.

    In jeder Iteration wird das linearisierte Ausgleichsproblem

        min || g(lambda_k) + Dg(lambda_k) delta ||_2

    mit einer QR-Zerlegung gelöst. Dadurch wird die explizite Bildung von
    Dg^T Dg vermieden.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% Aufgabenstellung / gegebene Werte

# Messdaten einer rational abfallenden Kurve.
xdat = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
ydat = np.array([0.39, 0.89, 1.45, 1.23, 0.86, 0.83, 0.58, 0.50, 0.44, 0.34])

# Startwert. Er ist bewusst nicht exakt, aber nahe genug für Konvergenz.
lam0 = np.array([[3.0], [2.0], [-1.0], [4.0]])

tol = 1e-8
max_iter = 30


# %% Symbolischer Teil: Modell, Residuum und Jacobi-Matrix

x, A, B, p, q = sp.symbols("x A B p q")
lam_syms = (A, B, p, q)

T_sp = (A * x + B) / (x**2 + p * x + q)

g_sp = sp.Matrix([yi - T_sp.subs(x, xi) for xi, yi in zip(xdat, ydat)])
Dg_sp = g_sp.jacobian(lam_syms)

print("Symbolische Jacobi-Matrix Dg(lambda):")
sp.print_latex(Dg_sp)
print(Dg_sp)
print("")

g_num = sp.lambdify((A, B, p, q), g_sp, modules="numpy")
Dg_num = sp.lambdify((A, B, p, q), Dg_sp, modules="numpy")
T_num = sp.lambdify((x, A, B, p, q), T_sp, modules="numpy")


def g(lam):
    """Residuenvektor g(lambda) = y - T(x, lambda)."""
    return np.array(g_num(*lam.reshape(-1)), dtype=float).reshape(-1, 1)


def Dg(lam):
    """Jacobi-Matrix des Residuenvektors."""
    return np.array(Dg_num(*lam.reshape(-1)), dtype=float)


def E(lam):
    """Fehlerfunktional E(lambda) = ||g(lambda)||_2^2."""
    return float(np.linalg.norm(g(lam))**2)


# %% Gauss-Newton mit QR-Zerlegung

lam = lam0.copy()
E_values = [E(lam)]

print("Gauss-Newton mit QR-Zerlegung")
print("k |        A        B        p        q | ||delta||_2 | E(lambda)")
print("-" * 82)
print(
    f"0 | {lam[0,0]:8.5f} {lam[1,0]:8.5f} {lam[2,0]:8.5f} {lam[3,0]:8.5f} "
    f"| {'-':>11} | {E(lam):.6e}"
)

for k in range(1, max_iter + 1):
    J = Dg(lam)
    r = g(lam)

    # QR-Zerlegung des linearen Ausgleichsproblems J*delta ≈ -r.
    # mode="reduced" liefert Q mit so vielen Spalten wie Parameter.
    Q, R = np.linalg.qr(J, mode="reduced")
    delta = np.linalg.solve(R, Q.T @ (-r))

    lam = lam + delta
    E_values.append(E(lam))

    print(
        f"{k:1d} | {lam[0,0]:8.5f} {lam[1,0]:8.5f} {lam[2,0]:8.5f} {lam[3,0]:8.5f} "
        f"| {np.linalg.norm(delta):11.3e} | {E(lam):.6e}"
    )

    if np.linalg.norm(delta) < tol:
        break


# %% Resultate ausgeben

print("\nResultat:")
print(f"Anzahl Iterationen: {k}")
print(f"A = {lam[0,0]:.8f}")
print(f"B = {lam[1,0]:.8f}")
print(f"p = {lam[2,0]:.8f}")
print(f"q = {lam[3,0]:.8f}")
print(f"Fehlerfunktional E = {E(lam):.8e}")
print(f"Konditionszahl von Dg am Ende = {np.linalg.cond(Dg(lam)):.3e}")
print(f"Konditionszahl von R am Ende  = {np.linalg.cond(R):.3e}")


# %% Plot: Daten, Fit und Fehlerverlauf

xf = np.linspace(xdat.min(), xdat.max(), 600)
yf = T_num(xf, *lam.reshape(-1))

plt.figure()
plt.plot(xdat, ydat, "*", label="Messdaten")
plt.plot(xf, yf, label="Fit mit rationaler Funktion")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Gauss-Newton mit QR-Zerlegung")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure()
plt.semilogy(np.arange(len(E_values)), E_values, "o-")
plt.xlabel("Iteration k")
plt.ylabel("E(lambda_k)")
plt.title("Fehlerfunktional pro Iteration")
plt.grid(True)
plt.tight_layout()

plt.show()
