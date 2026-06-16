# -*- coding: utf-8 -*-
"""
Gauss-Quadratur mit G1, G2 und G3
=================================

Aufgabenstellung
----------------
Berechne

    I = integral_0^0.5 exp(-x^2) dx

mit den Gauss-Formeln mit 1, 2 und 3 Stuetzstellen.

Die Standardknoten auf [-1, 1] werden auf [a, b] transformiert:

    x = (b-a)/2 * t + (a+b)/2.

Zum Vergleich wird auch die einfache Simpson-Regel mit 3 Funktionsauswertungen
berechnet.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% Angaben

a = 0.0
b = 0.5
x = sp.symbols("x")
f_sym = sp.exp(-x**2)
f = sp.lambdify(x, f_sym, "numpy")

I_exakt = float(sp.integrate(f_sym, (x, a, b)).evalf(20))

print("Gauss-Quadratur G1, G2, G3")
print("===========================")
print(f"f(x) = {f_sym}")
print(f"Intervall: [{a}, {b}]")
print(f"Referenzwert I = {I_exakt:.12f}")
print()


# %% Hilfsfunktion: Transformation von [-1,1] auf [a,b]


def transformiere_knoten(t, a, b):
    return (b - a) / 2 * t + (a + b) / 2


def gauss_quadratur(f, a, b, t, w):
    x_nodes = transformiere_knoten(t, a, b)
    return (b - a) / 2 * np.sum(w * f(x_nodes)), x_nodes


# %% Gauss-Formeln definieren

# G1: ein Knoten t=0, Gewicht 2
t1 = np.array([0.0])
w1 = np.array([2.0])

# G2: Knoten +-1/sqrt(3), Gewichte 1 und 1
t2 = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
w2 = np.array([1.0, 1.0])

# G3: Knoten +-sqrt(0.6), 0, Gewichte 5/9, 8/9, 5/9
t3 = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
w3 = np.array([5 / 9, 8 / 9, 5 / 9])

I_G1, x_G1 = gauss_quadratur(f, a, b, t1, w1)
I_G2, x_G2 = gauss_quadratur(f, a, b, t2, w2)
I_G3, x_G3 = gauss_quadratur(f, a, b, t3, w3)


# %% Simpson-Regel mit 3 Funktionsauswertungen

x_simpson = np.array([a, (a + b) / 2, b])
I_simpson = (b - a) / 6 * (f(a) + 4 * f((a + b) / 2) + f(b))


# %% Resultate ausgeben

print("Gauss-Knoten nach Transformation")
print("--------------------------------")
print("G1:", x_G1)
print("G2:", x_G2)
print("G3:", x_G3)
print()

print("Resultate")
print("---------")
print(f"G1       = {I_G1:.12f}, Fehler = {abs(I_G1 - I_exakt):.4e}")
print(f"G2       = {I_G2:.12f}, Fehler = {abs(I_G2 - I_exakt):.4e}")
print(f"G3       = {I_G3:.12f}, Fehler = {abs(I_G3 - I_exakt):.4e}")
print(f"Simpson  = {I_simpson:.12f}, Fehler = {abs(I_simpson - I_exakt):.4e}")


# %% Plot 1: Funktion und Knoten

x_plot = np.linspace(a, b, 400)

plt.figure()
plt.plot(x_plot, f(x_plot), label="exp(-x^2)")
plt.plot(x_G1, f(x_G1), "o", label="G1-Knoten")
plt.plot(x_G2, f(x_G2), "s", label="G2-Knoten")
plt.plot(x_G3, f(x_G3), "x", label="G3-Knoten")
plt.plot(x_simpson, f(x_simpson), "+", label="Simpson-Knoten")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gauss-Knoten und Simpson-Knoten")
plt.grid(True)
plt.legend()
plt.show()


# %% Plot 2: Fehlervergleich

methoden = ["G1", "G2", "G3", "Simpson"]
fehler = [abs(I_G1 - I_exakt), abs(I_G2 - I_exakt), abs(I_G3 - I_exakt), abs(I_simpson - I_exakt)]

plt.figure()
plt.semilogy(methoden, fehler, "o-")
plt.ylabel("absoluter Fehler")
plt.title("Fehlervergleich Gauss-Quadratur")
plt.grid(True, which="both")
plt.show()
