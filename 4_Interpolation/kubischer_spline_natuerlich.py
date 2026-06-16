# -*- coding: utf-8 -*-
"""
Natürliche kubische Splineinterpolation

Aufgabenstellung
================
Gegeben sind mehrere Stützpunkte. Gesucht ist die natürliche kubische
Splinefunktion S(x). Auf jedem Intervall [x_i, x_{i+1}] gilt

    S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3

mit den natürlichen Randbedingungen c_0 = 0 und c_n = 0.
Zusätzlich wird S(3.0) berechnet und die stückweise Splinekurve geplottet.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% Angaben

x_symbol = sp.symbols("x")

# Stützpunkte: ungleichmässige Abstände, damit das Beispiel prüfungsnah bleibt
x_data = np.array([0.0, 1.0, 2.5, 4.0, 5.5], dtype=float)
y_data = np.array([1.0, 2.1, 1.4, 3.2, 2.8], dtype=float)

# Gesuchter Auswertungspunkt
x_eval = 3.0


# %% Natürlichen kubischen Spline berechnen

n = len(x_data) - 1                 # Anzahl Intervalle
h = np.diff(x_data)                 # h_i = x_{i+1} - x_i

# a_i sind direkt die linken Stützwerte y_i
a = y_data[:-1].copy()

# Lineares Gleichungssystem fuer c_0, ..., c_n aufbauen
A = np.zeros((n + 1, n + 1))
rhs = np.zeros(n + 1)

# Natürliche Randbedingungen: c_0 = 0 und c_n = 0
A[0, 0] = 1.0
A[n, n] = 1.0

# Innere Gleichungen fuer i = 1, ..., n-1
for i in range(1, n):
    A[i, i - 1] = h[i - 1]
    A[i, i] = 2 * (h[i - 1] + h[i])
    A[i, i + 1] = h[i]
    rhs[i] = 3 * ((y_data[i + 1] - y_data[i]) / h[i]
                  - (y_data[i] - y_data[i - 1]) / h[i - 1])

# Loese nach c_0, ..., c_n
c = np.linalg.solve(A, rhs)

# Restliche Koeffizienten b_i und d_i berechnen
b = np.zeros(n)
d = np.zeros(n)

for i in range(n):
    b[i] = ((y_data[i + 1] - y_data[i]) / h[i]
            - h[i] / 3 * (c[i + 1] + 2 * c[i]))
    d[i] = (c[i + 1] - c[i]) / (3 * h[i])


# %% Spline auswerten

def spline_auswerten(x_wert, x_data, a, b, c, d):
    """Wertet den natürlichen kubischen Spline an einer Stelle aus."""
    # Passendes Intervall suchen
    if x_wert == x_data[-1]:
        i = len(a) - 1
    else:
        i = np.searchsorted(x_data, x_wert) - 1

    dx = x_wert - x_data[i]
    return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3


y_eval = spline_auswerten(x_eval, x_data, a, b, c, d)


# %% Symbolische Darstellung der Teilpolynome

S_symbolisch = []
for i in range(n):
    dx = x_symbol - x_data[i]
    Si = a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
    S_symbolisch.append(sp.expand(Si))


# %% Resultatausgabe

print("Natürliche kubische Splineinterpolation")
print("=======================================")
print("Gegebene Stützpunkte:")
for xi, yi in zip(x_data, y_data):
    print(f"  ({xi:5.2f}, {yi:8.4f})")

print("\nSchrittweiten h_i:")
for i, hi in enumerate(h):
    print(f"  h_{i} = {hi:.6f}")

print("\nKoeffizienten der Teilpolynome:")
for i in range(n):
    print(f"Intervall [{x_data[i]}, {x_data[i+1]}]:")
    print(f"  a_{i} = {a[i]: .8f}")
    print(f"  b_{i} = {b[i]: .8f}")
    print(f"  c_{i} = {c[i]: .8f}")
    print(f"  d_{i} = {d[i]: .8f}")
    print(f"  S_{i}(x) = {S_symbolisch[i]}")

print(f"\nAuswertung: S({x_eval}) = {y_eval:.8f}")


# %% Plot

plt.figure()

# Jedes Splineintervall separat zeichnen
for i in range(n):
    x_plot = np.linspace(x_data[i], x_data[i + 1], 120)
    dx = x_plot - x_data[i]
    y_plot = a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
    plt.plot(x_plot, y_plot, label=f"S_{i}(x)")

plt.plot(x_data, y_data, "o", label="Stützpunkte")
plt.plot(x_eval, y_eval, "x", markersize=10, label=f"S({x_eval})")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Natürliche kubische Splineinterpolation")
plt.grid(True)
plt.legend()
plt.show()
