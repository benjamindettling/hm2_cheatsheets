# -*- coding: utf-8 -*-
"""
Fehlerabschaetzung und Schrittweite bei Quadraturformeln
========================================================

Aufgabenstellung
----------------
Fuer

    I = integral_0^0.5 exp(-x^2) dx

soll eine Schrittweite h bestimmt werden, sodass der absolute Fehler hoechstens

    eps = 1e-5

betraegt. Betrachtet werden:

    1. summierte Trapezregel
    2. summierte Simpson-Regel

Dazu werden mit sympy die benoetigten Ableitungen gebildet und auf dem Intervall
numerisch abgeschaetzt.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% Angaben

x = sp.symbols("x")
a = 0.0
b = 0.5
eps = 1e-5

f_sym = sp.exp(-x**2)
f2_sym = sp.diff(f_sym, x, 2)      # fuer Trapezregel
f4_sym = sp.diff(f_sym, x, 4)      # fuer Simpson-Regel

f = sp.lambdify(x, f_sym, "numpy")
f2 = sp.lambdify(x, f2_sym, "numpy")
f4 = sp.lambdify(x, f4_sym, "numpy")

print("Fehlerabschaetzung und Schrittweite")
print("====================================")
print(f"f(x) = {f_sym}")
print(f"Intervall: [{a}, {b}]")
print(f"eps = {eps}")
print()
print("Ableitungen")
print("-----------")
print(f"f''(x)  = {sp.simplify(f2_sym)}")
print(f"f''''(x)= {sp.simplify(f4_sym)}")
print()


# %% Maximum der Ableitungen numerisch abschaetzen

x_grid = np.linspace(a, b, 20001)
M2 = np.max(np.abs(f2(x_grid)))
M4 = np.max(np.abs(f4(x_grid)))

print("Numerische Maxima auf dem Intervall")
print("-----------------------------------")
print(f"max |f''(x)|   ~= {M2:.10f}")
print(f"max |f''''(x)| ~= {M4:.10f}")
print()


# %% Schrittweitenformeln

# Fehler Trapezregel:
# |I - T(h)| <= h^2/12 * (b-a) * max |f''|
h_trapez = np.sqrt(12 * eps / ((b - a) * M2))
n_trapez = int(np.ceil((b - a) / h_trapez))
h_trapez_effektiv = (b - a) / n_trapez

# Fehler Simpson-Regel:
# |I - S(h)| <= h^4/2880 * (b-a) * max |f''''|
h_simpson = (2880 * eps / ((b - a) * M4))**0.25
n_simpson = int(np.ceil((b - a) / h_simpson))
h_simpson_effektiv = (b - a) / n_simpson

print("Schrittweiten aus der Fehlerabschaetzung")
print("----------------------------------------")
print("Trapezregel:")
print(f"  h <= {h_trapez:.10f}")
print(f"  n >= {(b-a)/h_trapez:.4f}  ->  n = {n_trapez}")
print(f"  verwendetes h = {h_trapez_effektiv:.10f}")
print()
print("Simpson-Regel:")
print(f"  h <= {h_simpson:.10f}")
print(f"  n >= {(b-a)/h_simpson:.4f}  ->  n = {n_simpson}")
print(f"  verwendetes h = {h_simpson_effektiv:.10f}")
print()


# %% Quadraturfunktionen zur Kontrolle


def trapezregel(f, a, b, n):
    h = (b - a) / n
    xs = np.linspace(a, b, n + 1)
    return h * ((f(a) + f(b)) / 2 + np.sum(f(xs[1:-1])))


def rechteckregel(f, a, b, n):
    h = (b - a) / n
    xs = np.linspace(a, b, n + 1)
    x_mid = xs[:-1] + h / 2
    return h * np.sum(f(x_mid))


def simpsonregel(f, a, b, n):
    # Simpson in der Skriptform: S = 1/3 * (T + 2R)
    return (trapezregel(f, a, b, n) + 2 * rechteckregel(f, a, b, n)) / 3


I_exakt = float(sp.integrate(f_sym, (x, a, b)).evalf(20))
I_T = trapezregel(f, a, b, n_trapez)
I_S = simpsonregel(f, a, b, n_simpson)

print("Kontrolle der tatsaechlichen Fehler")
print("-----------------------------------")
print(f"I exakt       = {I_exakt:.12f}")
print(f"T_n           = {I_T:.12f}, Fehler = {abs(I_T - I_exakt):.4e}")
print(f"S_n           = {I_S:.12f}, Fehler = {abs(I_S - I_exakt):.4e}")


# %% Plot 1: Ableitungsbetraege

plt.figure()
plt.plot(x_grid, np.abs(f2(x_grid)), label="|f''(x)|")
plt.plot(x_grid, np.abs(f4(x_grid)), label="|f''''(x)|")
plt.xlabel("x")
plt.ylabel("Betrag der Ableitung")
plt.title("Ableitungen fuer die Fehlerabschaetzung")
plt.grid(True)
plt.legend()
plt.show()


# %% Plot 2: Fehler gegen Anzahl Teilintervalle

n_values = np.array([2, 4, 8, 16, 32, 64, 128])
err_T = np.array([abs(trapezregel(f, a, b, int(n)) - I_exakt) for n in n_values])
err_S = np.array([abs(simpsonregel(f, a, b, int(n)) - I_exakt) for n in n_values])

plt.figure()
plt.loglog(n_values, err_T, "o-", label="Trapezregel")
plt.loglog(n_values, err_S, "s-", label="Simpson-Regel")
plt.axhline(eps, linestyle="--", label="Toleranz eps")
plt.xlabel("n")
plt.ylabel("absoluter Fehler")
plt.title("Fehlerverhalten bei kleinerer Schrittweite")
plt.grid(True, which="both")
plt.legend()
plt.show()
