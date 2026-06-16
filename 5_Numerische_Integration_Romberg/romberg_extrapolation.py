# -*- coding: utf-8 -*-
"""
Romberg-Extrapolation
=====================

Aufgabenstellung
----------------
Berechne naeherungsweise

    I = integral_0^pi cos(x^2) dx

mit Romberg-Extrapolation bis m = 4.

Vorgehen:

    1. Erste Spalte mit summierter Trapezregel berechnen.
    2. Schrittweite jeweils halbieren: h_j = (b-a)/2^j.
    3. Romberg-Rekursion anwenden:

       R[j,k] = (4^k * R[j,k-1] - R[j-1,k-1]) / (4^k - 1).

Die Werte auf der Diagonalen sind sukzessiv verbesserte Approximationen.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% Angaben

a = 0.0
b = np.pi
m = 4


def f(x):
    return np.cos(x**2)


x_sym = sp.symbols("x")
f_sym = sp.cos(x_sym**2)

print("Romberg-Extrapolation")
print("======================")
print("Integral: integral_0^pi cos(x^2) dx")
print(f"m = {m}")
print()


# %% Summierte Trapezregel


def trapezregel(f, a, b, n):
    h = (b - a) / n
    xs = np.linspace(a, b, n + 1)
    return h * ((f(a) + f(b)) / 2 + np.sum(f(xs[1:-1])))


# %% Erste Spalte und Romberg-Tabelle aufbauen

R = np.zeros((m + 1, m + 1))

for j in range(m + 1):
    n = 2**j
    R[j, 0] = trapezregel(f, a, b, n)

for k in range(1, m + 1):
    for j in range(k, m + 1):
        R[j, k] = (4**k * R[j, k - 1] - R[j - 1, k - 1]) / (4**k - 1)


# %% Referenzwert

# cos(x^2) hat keine elementare Stammfunktion. Als Referenz verwenden wir
# eine sehr feine Trapezregel, nur fuer den Fehlervergleich im Skript.
n_ref = 200000
I_ref = trapezregel(f, a, b, n_ref)


# %% Tabelle ausgeben

print("Romberg-Tabelle")
print("---------------")
for j in range(m + 1):
    zeile = []
    for k in range(j + 1):
        zeile.append(f"R[{j},{k}] = {R[j,k]: .12f}")
    print("   ".join(zeile))
print()

print("Resultat")
print("--------")
print(f"Beste Naeherung R[{m},{m}] = {R[m,m]:.12f}")
print(f"Referenzwert fein          = {I_ref:.12f}")
print(f"Abweichung                 = {abs(R[m,m] - I_ref):.4e}")
print()

print("Diagonalwerte")
print("-------------")
for k in range(m + 1):
    print(f"R[{k},{k}] = {R[k,k]:.12f}, Fehler ~= {abs(R[k,k] - I_ref):.4e}")


# %% Plot 1: Integrand

x_plot = np.linspace(a, b, 800)
plt.figure()
plt.plot(x_plot, f(x_plot), label="cos(x^2)")
plt.axhline(0, linewidth=0.8)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Integrand fuer Romberg-Extrapolation")
plt.grid(True)
plt.legend()
plt.show()


# %% Plot 2: Konvergenz der Diagonalwerte

diag = np.array([R[k, k] for k in range(m + 1)])
err_diag = np.abs(diag - I_ref)

plt.figure()
plt.semilogy(range(m + 1), err_diag, "o-")
plt.xlabel("Romberg-Stufe k")
plt.ylabel("Fehler gegen Referenzwert")
plt.title("Konvergenz der Romberg-Diagonalwerte")
plt.grid(True, which="both")
plt.show()
