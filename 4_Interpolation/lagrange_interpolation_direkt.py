# -*- coding: utf-8 -*-
"""
Lagrange-Interpolation direkt nach Formel

Aufgabenstellung
================
Gegeben sind Messwerte fuer den Luftdruck in Abhaengigkeit der Hoehe.
Gesucht ist das Interpolationspolynom P_3(x), das exakt durch alle
Stützpunkte geht. Anschliessend wird der Druck bei x = 2.5 km berechnet.

Stützstellen x_i: Hoehe in km
Stützwerte   y_i: Druck in hPa
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% Angaben

# Symbolische Variable fuer das Interpolationspolynom
x = sp.symbols("x")

# Gegebene Datenpunkte: nicht trivial, aber noch gut von Hand nachvollziehbar
x_data = np.array([0.0, 1.5, 3.0, 4.5], dtype=float)          # Hoehe in km
y_data = np.array([1013.0, 846.0, 706.0, 589.0], dtype=float) # Druck in hPa

# Gesuchter Auswertungspunkt
x_eval = 2.5


# %% Lagrange-Polynome direkt nach Definition aufbauen

def lagrange_basis_polynom(i, x_data, x_symbol):
    """Berechnet das i-te Lagrange-Basispolynom l_i(x)."""
    li = 1
    xi = x_data[i]

    for j in range(len(x_data)):
        if j != i:
            xj = x_data[j]
            li *= (x_symbol - xj) / (xi - xj)

    return sp.simplify(li)


# Alle Lagrange-Basispolynome berechnen
L = []
for i in range(len(x_data)):
    L.append(lagrange_basis_polynom(i, x_data, x))

# Interpolationspolynom P(x) = Summe y_i * l_i(x)
P = 0
for i in range(len(x_data)):
    P += y_data[i] * L[i]

P = sp.expand(P)


# %% Auswertung an der gesuchten Stelle

P_func = sp.lambdify(x, P, modules="numpy")
y_eval = P_func(x_eval)


# %% Resultatausgabe

print("Lagrange-Interpolation direkt nach Formel")
print("==========================================")
print("Gegebene Stützpunkte (x_i, y_i):")
for xi, yi in zip(x_data, y_data):
    print(f"  ({xi:5.2f} km, {yi:8.3f} hPa)")

print("\nLagrange-Basispolynome:")
for i, li in enumerate(L):
    print(f"l_{i}(x) = {sp.expand(li)}")

print("\nInterpolationspolynom:")
print(f"P(x) = {P}")

print("\nAuswertung:")
print(f"P({x_eval}) = {y_eval:.6f} hPa")

# Kontrolle: Das Polynom muss alle Stützwerte exakt rekonstruieren
print("\nKontrolle an den Stützstellen:")
for xi, yi in zip(x_data, y_data):
    print(f"  x = {xi:5.2f}: P(x) = {P_func(xi):10.6f}, y_i = {yi:10.6f}")


# %% Plot

# Dichteres Gitter fuer glatte Kurve
x_plot = np.linspace(x_data.min(), x_data.max(), 400)
y_plot = P_func(x_plot)

plt.figure()
plt.plot(x_plot, y_plot, label="Lagrange-Polynom P_3(x)")
plt.plot(x_data, y_data, "o", label="Stützpunkte")
plt.plot(x_eval, y_eval, "x", markersize=10, label=f"Interpolierter Wert bei x={x_eval}")
plt.xlabel("Hoehe x [km]")
plt.ylabel("Druck y [hPa]")
plt.title("Direkte Lagrange-Interpolation")
plt.grid(True)
plt.legend()
plt.show()
