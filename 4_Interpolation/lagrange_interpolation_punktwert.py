# -*- coding: utf-8 -*-
"""
Lagrange-Interpolation fuer einen einzelnen Punktwert

Aufgabenstellung
================
Gegeben sind vier Messpunkte einer Temperaturkurve. Gesucht ist nur der
interpolierte Wert bei x = 4.0. Das Skript berechnet die einzelnen
Lagrange-Gewichte l_i(4.0) numerisch und setzt sie dann in

    P(x) = y_0*l_0(x) + ... + y_n*l_n(x)

ein.

Dieses Skript ist bewusst kompakt gehalten, weil Prüfungsaufgaben oft nur
einen bestimmten Zwischenwert verlangen.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% Angaben

x = sp.symbols("x")

# Messdaten, z.B. Zeit x in Stunden und Temperatur y in Grad Celsius
x_data = np.array([0.0, 2.0, 5.0, 7.0], dtype=float)
y_data = np.array([3.2, 4.1, 6.8, 7.3], dtype=float)

# Gesuchter Punkt
x_eval = 4.0


# %% Numerische Lagrange-Gewichte am Punkt x_eval berechnen

def lagrange_gewicht_am_punkt(i, x_data, x_eval):
    """Berechnet l_i(x_eval) direkt als Zahl."""
    li = 1.0
    xi = x_data[i]

    for j in range(len(x_data)):
        if j != i:
            xj = x_data[j]
            li *= (x_eval - xj) / (xi - xj)

    return li


l_values = np.zeros(len(x_data))
for i in range(len(x_data)):
    l_values[i] = lagrange_gewicht_am_punkt(i, x_data, x_eval)

# Interpolierter Wert: Skalarprodukt der Stützwerte mit den Lagrange-Gewichten
y_eval = np.sum(y_data * l_values)


# %% Zusätzlich: symbolisches Polynom fuer Kontrollplot aufbauen

P = 0
for i in range(len(x_data)):
    li_symbolisch = 1
    for j in range(len(x_data)):
        if j != i:
            li_symbolisch *= (x - x_data[j]) / (x_data[i] - x_data[j])
    P += y_data[i] * li_symbolisch

P = sp.expand(P)
P_func = sp.lambdify(x, P, modules="numpy")


# %% Resultatausgabe

print("Lagrange-Interpolation: Punktwert")
print("=================================")
print("Gegebene Datenpunkte:")
for xi, yi in zip(x_data, y_data):
    print(f"  x_i = {xi:5.2f}, y_i = {yi:7.3f}")

print(f"\nGesucht: P({x_eval})")
print("\nLagrange-Gewichte am gesuchten Punkt:")
for i, li in enumerate(l_values):
    print(f"  l_{i}({x_eval}) = {li: .8f}")

print("\nEinsetzen in P(x_eval) = Summe y_i*l_i(x_eval):")
for i in range(len(x_data)):
    print(f"  Beitrag {i}: {y_data[i]:7.3f} * {l_values[i]: .8f} = {y_data[i]*l_values[i]: .8f}")

print(f"\nResultat: P({x_eval}) = {y_eval:.8f}")
print(f"Symbolisches Kontrollpolynom: P(x) = {P}")


# %% Plot

x_plot = np.linspace(x_data.min(), x_data.max(), 400)
y_plot = P_func(x_plot)

plt.figure()
plt.plot(x_plot, y_plot, label="Interpolationspolynom")
plt.plot(x_data, y_data, "o", label="Stützpunkte")
plt.plot(x_eval, y_eval, "x", markersize=10, label="gesuchter Punktwert")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Lagrange-Interpolation an einem Punkt")
plt.grid(True)
plt.legend()
plt.show()
