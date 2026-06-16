# -*- coding: utf-8 -*-
"""
Vergleich: Lagrange-Interpolation gegen natürliche kubische Splines

Aufgabenstellung
================
Zu denselben Stützpunkten werden zwei Interpolationsfunktionen erstellt:

1. Ein globales Lagrange-Polynom durch alle Punkte.
2. Eine natürliche kubische Splinefunktion mit stückweisen Polynomen.

Der Plot zeigt, dass das globale Polynom stärker schwingen kann, während
der Spline lokal und meist ruhiger verläuft.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% Angaben

x = sp.symbols("x")

# Stützpunkte einer leicht oszillierenden Messreihe
x_data = np.array([-2.0, -1.2, -0.4, 0.5, 1.3, 2.0, 2.7], dtype=float)
y_data = np.array([0.25, 0.72, 1.15, 0.94, 0.55, 0.36, 0.30], dtype=float)

# Vergleichspunkt im Inneren des Intervalls
x_eval = 1.0


# %% Lagrange-Polynom berechnen

P = 0
for i in range(len(x_data)):
    li = 1
    for j in range(len(x_data)):
        if j != i:
            li *= (x - x_data[j]) / (x_data[i] - x_data[j])
    P += y_data[i] * li

P = sp.expand(P)
P_func = sp.lambdify(x, P, modules="numpy")


# %% Natürlichen kubischen Spline berechnen

def spline_koeffizienten_natuerlich(x_data, y_data):
    """Berechnet a_i, b_i, c_i, d_i des natürlichen kubischen Splines."""
    n = len(x_data) - 1
    h = np.diff(x_data)

    a = y_data[:-1].copy()
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)

    # Randbedingungen c_0 = 0, c_n = 0
    A[0, 0] = 1.0
    A[n, n] = 1.0

    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 3 * ((y_data[i + 1] - y_data[i]) / h[i]
                      - (y_data[i] - y_data[i - 1]) / h[i - 1])

    c = np.linalg.solve(A, rhs)

    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        b[i] = ((y_data[i + 1] - y_data[i]) / h[i]
                - h[i] / 3 * (c[i + 1] + 2 * c[i]))
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return a, b, c, d


def spline_auswerten(x_wert, x_data, a, b, c, d):
    """Wertet den Spline an einem Punkt oder einem NumPy-Array aus."""
    x_arr = np.asarray(x_wert)
    y_arr = np.zeros_like(x_arr, dtype=float)

    for idx, x_single in np.ndenumerate(x_arr):
        if x_single == x_data[0]:
            i = 0
        elif x_single == x_data[-1]:
            i = len(a) - 1
        else:
            i = np.searchsorted(x_data, x_single) - 1
        dx = x_single - x_data[i]
        y_arr[idx] = a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3

    if np.isscalar(x_wert):
        return float(y_arr)
    return y_arr


a_s, b_s, c_s, d_s = spline_koeffizienten_natuerlich(x_data, y_data)

# Werte am Vergleichspunkt
y_lagrange_eval = P_func(x_eval)
y_spline_eval = spline_auswerten(x_eval, x_data, a_s, b_s, c_s, d_s)


# %% Resultatausgabe

print("Vergleich: Lagrange-Interpolation gegen natürlichen kubischen Spline")
print("====================================================================")
print("Gegebene Stützpunkte:")
for xi, yi in zip(x_data, y_data):
    print(f"  ({xi:6.3f}, {yi:8.4f})")

print("\nGlobales Lagrange-Polynom:")
print(f"P(x) = {P}")

print(f"\nVergleich am Punkt x = {x_eval}:")
print(f"  Lagrange: P({x_eval}) = {y_lagrange_eval:.8f}")
print(f"  Spline:   S({x_eval}) = {y_spline_eval:.8f}")

print("\nSpline-Koeffizienten je Intervall:")
for i in range(len(a_s)):
    print(f"  Intervall [{x_data[i]:.2f}, {x_data[i+1]:.2f}]: "
          f"a={a_s[i]: .6f}, b={b_s[i]: .6f}, c={c_s[i]: .6f}, d={d_s[i]: .6f}")


# %% Plot

x_plot = np.linspace(x_data.min(), x_data.max(), 800)
y_lagrange = P_func(x_plot)
y_spline = spline_auswerten(x_plot, x_data, a_s, b_s, c_s, d_s)

plt.figure()
plt.plot(x_plot, y_lagrange, label="globales Lagrange-Polynom")
plt.plot(x_plot, y_spline, label="natürlicher kubischer Spline")
plt.plot(x_data, y_data, "o", label="Stützpunkte")
plt.plot(x_eval, y_lagrange_eval, "x", markersize=10, label="Lagrange bei x=1.0")
plt.plot(x_eval, y_spline_eval, "+", markersize=12, label="Spline bei x=1.0")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Interpolation: Lagrange vs. Spline")
plt.grid(True)
plt.legend()
plt.show()
