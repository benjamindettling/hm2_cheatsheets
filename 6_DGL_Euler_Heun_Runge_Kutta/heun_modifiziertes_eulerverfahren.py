# -*- coding: utf-8 -*-
"""
Modifiziertes Eulerverfahren / Heun-Verfahren
=============================================

Aufgabenstellung
----------------
Gegeben ist das Anfangswertproblem

    y'(x) = 2*(1 - x)*y(x),      y(0) = 1

auf dem Intervall [0, 3].

Gesucht ist die numerische Loesung mit dem Heun-Verfahren. Zum Vergleich
werden das explizite Eulerverfahren und die exakte Loesung geplottet.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 1. Angaben der Aufgabe
# -----------------------------------------------------------------------------

x_sym, y_sym = sp.symbols("x y")
f_sym = 2 * (1 - x_sym) * y_sym
f = sp.lambdify((x_sym, y_sym), f_sym, modules="numpy")

a = 0.0
y0 = 1.0
b = 3.0
h = 0.25
n = int((b - a) / h)


def y_exakt(x):
    return np.exp(2 * x - x**2)


# -----------------------------------------------------------------------------
# 2. Numerische Verfahren
# -----------------------------------------------------------------------------

def euler(f, a, y0, b, n):
    """Explizites Eulerverfahren."""
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0] = a
    y[0] = y0
    h = (b - a) / n

    for i in range(n):
        y[i + 1] = y[i] + h * f(x[i], y[i])
        x[i + 1] = x[i] + h

    return x, y


def heun(f, a, y0, b, n):
    """Heun-Verfahren: Euler-Voraussage + gemittelte Steigung."""
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0] = a
    y[0] = y0
    h = (b - a) / n

    for i in range(n):
        k1 = f(x[i], y[i])                         # Steigung am linken Rand
        y_pred = y[i] + h * k1                     # Euler-Vorschlag
        k2 = f(x[i] + h, y_pred)                   # Steigung am rechten Rand
        y[i + 1] = y[i] + h / 2 * (k1 + k2)        # Mittelwert der Steigungen
        x[i + 1] = x[i] + h

    return x, y


x_euler, y_euler = euler(f, a, y0, b, n)
x_heun, y_heun = heun(f, a, y0, b, n)


# -----------------------------------------------------------------------------
# 3. Resultatausgabe
# -----------------------------------------------------------------------------

print("Heun-Verfahren / modifiziertes Eulerverfahren")
print("=============================================")
print("DGL: y' =", f_sym)
print("h = {:.4f}, n = {}".format(h, n))
print("")
print("Endwert Euler: y({:.1f}) = {:.8f}".format(b, y_euler[-1]))
print("Endwert Heun:  y({:.1f}) = {:.8f}".format(b, y_heun[-1]))
print("Endwert exakt: y({:.1f}) = {:.8f}".format(b, y_exakt(b)))
print("")
print("Fehler Euler: {:.8e}".format(abs(y_euler[-1] - y_exakt(b))))
print("Fehler Heun:  {:.8e}".format(abs(y_heun[-1] - y_exakt(b))))

print("\nIterationstabelle fuer Heun:")
print(" k      x_k          y_k          y_exakt       Fehler")
for k in range(len(x_heun)):
    fehler = abs(y_heun[k] - y_exakt(x_heun[k]))
    print("{:2d}   {:8.4f}   {:11.7f}   {:11.7f}   {:10.3e}".format(
        k, x_heun[k], y_heun[k], y_exakt(x_heun[k]), fehler
    ))


# -----------------------------------------------------------------------------
# 4. Plot
# -----------------------------------------------------------------------------

x_plot = np.linspace(a, b, 400)

plt.figure()
plt.plot(x_plot, y_exakt(x_plot), label="exakte Loesung")
plt.plot(x_euler, y_euler, "o--", label="Euler")
plt.plot(x_heun, y_heun, "s--", label="Heun")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Euler vs. Heun fuer y' = 2(1-x)y")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(x_euler, np.abs(y_euler - y_exakt(x_euler)), "o--", label="Euler")
plt.plot(x_heun, np.abs(y_heun - y_exakt(x_heun)), "s--", label="Heun")
plt.xlabel("x")
plt.ylabel("absoluter Fehler")
plt.title("Fehlervergleich Euler und Heun")
plt.grid(True)
plt.legend()
plt.show()
