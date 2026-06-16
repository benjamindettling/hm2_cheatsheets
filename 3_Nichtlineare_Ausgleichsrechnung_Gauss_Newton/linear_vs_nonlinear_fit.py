# -*- coding: utf-8 -*-
"""
linear_vs_nonlinear_fit.py

Aufgabe:
    Dieselben Messdaten werden einmal mit einem linearen Modell und einmal
    mit einem nichtlinearen Modell approximiert.

    Lineares Modell:
        f_lin(x; m,d) = m*x + d
        -> linear in den Parametern, daher direkte lineare Ausgleichsrechnung.

    Nichtlineares Modell:
        f_exp(x; a,b,c) = a*exp(-b*x) + c
        -> b steht im Exponenten, daher Gauss-Newton.

    Ziel:
        Den Unterschied zwischen linearer und nichtlinearer Ausgleichsrechnung
        sichtbar machen und die Fehlerfunktionale vergleichen.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% Aufgabenstellung / gegebene Werte

# Messdaten mit klar nichtlinearer Krümmung.
xdat = np.array([0., 1., 2., 3., 4., 5.])
ydat = np.array([5.10, 3.75, 3.02, 2.62, 2.37, 2.21])


# %% Teil 1: Lineares Ausgleichsproblem f(x) = m*x + d

# Designmatrix A für m*x + d.
# Erste Spalte: x-Werte, zweite Spalte: konstante 1.
A_lin = np.column_stack((xdat, np.ones_like(xdat)))

# QR-Zerlegung als stabile Standardvariante für lineare Ausgleichsrechnung.
Q, R = np.linalg.qr(A_lin, mode="reduced")
lam_lin = np.linalg.solve(R, Q.T @ ydat.reshape(-1, 1))

m = lam_lin[0, 0]
d = lam_lin[1, 0]

y_lin_dat = A_lin @ lam_lin
res_lin = ydat.reshape(-1, 1) - y_lin_dat
E_lin = float(np.linalg.norm(res_lin)**2)

print("Lineare Ausgleichsrechnung")
print(f"m = {m:.8f}")
print(f"d = {d:.8f}")
print(f"E_lin = {E_lin:.8e}")
print("")


# %% Teil 2: Nichtlineares Ausgleichsproblem f(x) = a*exp(-b*x) + c

x, a, b, c = sp.symbols("x a b c")
lam_syms = (a, b, c)

f_exp_sp = a * sp.exp(-b * x) + c
g_exp_sp = sp.Matrix([yi - f_exp_sp.subs(x, xi) for xi, yi in zip(xdat, ydat)])
Dg_exp_sp = g_exp_sp.jacobian(lam_syms)

g_num = sp.lambdify((a, b, c), g_exp_sp, modules="numpy")
Dg_num = sp.lambdify((a, b, c), Dg_exp_sp, modules="numpy")
f_exp_num = sp.lambdify((x, a, b, c), f_exp_sp, modules="numpy")


def g_exp(lam):
    return np.array(g_num(*lam.reshape(-1)), dtype=float).reshape(-1, 1)


def Dg_exp(lam):
    return np.array(Dg_num(*lam.reshape(-1)), dtype=float)


def E_exp(lam):
    return float(np.linalg.norm(g_exp(lam))**2)


# Startwert aus grober Betrachtung der Daten:
# Anfangswert ca. 5, Grenzwert ca. 2, also a ca. 3 und c ca. 2.
lam_exp = np.array([[3.0], [0.6], [1.5]])
tol = 1e-8
max_iter = 20
E_exp_values = [E_exp(lam_exp)]

print("Nichtlineare Ausgleichsrechnung mit Gauss-Newton/QR")
print("k |        a        b        c | ||delta||_2 | E_exp")
print("-" * 66)
print(
    f"0 | {lam_exp[0,0]:8.5f} {lam_exp[1,0]:8.5f} {lam_exp[2,0]:8.5f} "
    f"| {'-':>11} | {E_exp(lam_exp):.6e}"
)

for k in range(1, max_iter + 1):
    J = Dg_exp(lam_exp)
    r = g_exp(lam_exp)

    # QR-Lösung des linearen Ausgleichsproblems J*delta ≈ -r.
    Q, R = np.linalg.qr(J, mode="reduced")
    delta = np.linalg.solve(R, Q.T @ (-r))

    lam_exp = lam_exp + delta
    E_exp_values.append(E_exp(lam_exp))

    print(
        f"{k:1d} | {lam_exp[0,0]:8.5f} {lam_exp[1,0]:8.5f} {lam_exp[2,0]:8.5f} "
        f"| {np.linalg.norm(delta):11.3e} | {E_exp(lam_exp):.6e}"
    )

    if np.linalg.norm(delta) < tol:
        break

E_nonlin = E_exp(lam_exp)

print("\nNichtlinearer Fit:")
print(f"a = {lam_exp[0,0]:.8f}")
print(f"b = {lam_exp[1,0]:.8f}")
print(f"c = {lam_exp[2,0]:.8f}")
print(f"E_exp = {E_nonlin:.8e}")
print("")

print("Vergleich der Fehlerfunktionale:")
print(f"Lineares Modell:      E = {E_lin:.8e}")
print(f"Nichtlineares Modell: E = {E_nonlin:.8e}")
print(f"Verbesserungsfaktor:  E_lin / E_exp = {E_lin / E_nonlin:.2f}")


# %% Plot: Daten, linearer Fit, nichtlinearer Fit und Residuen

xf = np.linspace(xdat.min(), xdat.max(), 400)
y_lin = m * xf + d
y_exp = f_exp_num(xf, *lam_exp.reshape(-1))

plt.figure()
plt.plot(xdat, ydat, "o", label="Messdaten")
plt.plot(xf, y_lin, label="linearer Fit")
plt.plot(xf, y_exp, label="nichtlinearer Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linearer vs. nichtlinearer Ausgleich")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Residuen direkt an den Datenpunkten vergleichen.
y_exp_dat = f_exp_num(xdat, *lam_exp.reshape(-1))
res_exp = ydat - y_exp_dat

plt.figure()
breite = 0.35
indices = np.arange(len(xdat))
plt.bar(indices - breite/2, res_lin.reshape(-1), width=breite, label="linear")
plt.bar(indices + breite/2, res_exp, width=breite, label="nichtlinear")
plt.axhline(0, linewidth=1)
plt.xticks(indices, [str(x) for x in xdat.astype(int)])
plt.xlabel("Datenpunkt x_i")
plt.ylabel("Residuum")
plt.title("Residuenvergleich")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure()
plt.semilogy(np.arange(len(E_exp_values)), E_exp_values, "o-")
plt.xlabel("Iteration k")
plt.ylabel("E_exp(lambda_k)")
plt.title("Konvergenz des nichtlinearen Fits")
plt.grid(True)
plt.tight_layout()

plt.show()
