# -*- coding: utf-8 -*-
"""
Mittelpunktverfahren fuer ein Anfangswertproblem
================================================

Aufgabenstellung
----------------
Gegeben ist das Anfangswertproblem

    y'(x) = 2*(1 - x)*y(x),      y(0) = 1

auf dem Intervall [0, 3].

Gesucht ist die numerische Loesung mit dem Mittelpunktverfahren.
Zum Vergleich werden Euler, Heun und die exakte Loesung berechnet.
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
# 2. Verfahren
# -----------------------------------------------------------------------------

def euler(f, a, y0, b, n):
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
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0] = a
    y[0] = y0
    h = (b - a) / n
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h, y[i] + h * k1)
        y[i + 1] = y[i] + h / 2 * (k1 + k2)
        x[i + 1] = x[i] + h
    return x, y


def mittelpunkt(f, a, y0, b, n):
    """Explizites Mittelpunktverfahren, ein Runge-Kutta-Verfahren 2. Ordnung."""
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0] = a
    y[0] = y0
    h = (b - a) / n

    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + h / 2 * k1)
        y[i + 1] = y[i] + h * k2
        x[i + 1] = x[i] + h

    return x, y


x_eu, y_eu = euler(f, a, y0, b, n)
x_he, y_he = heun(f, a, y0, b, n)
x_mp, y_mp = mittelpunkt(f, a, y0, b, n)


# -----------------------------------------------------------------------------
# 3. Resultatausgabe
# -----------------------------------------------------------------------------

print("Mittelpunktverfahren")
print("====================")
print("DGL: y' =", f_sym)
print("h = {:.4f}, n = {}".format(h, n))
print("")
print("Endwert exakt:       {:.10f}".format(y_exakt(b)))
print("Endwert Euler:       {:.10f}, Fehler = {:.3e}".format(y_eu[-1], abs(y_eu[-1] - y_exakt(b))))
print("Endwert Heun:        {:.10f}, Fehler = {:.3e}".format(y_he[-1], abs(y_he[-1] - y_exakt(b))))
print("Endwert Mittelpunkt: {:.10f}, Fehler = {:.3e}".format(y_mp[-1], abs(y_mp[-1] - y_exakt(b))))


# -----------------------------------------------------------------------------
# 4. Plot
# -----------------------------------------------------------------------------

x_plot = np.linspace(a, b, 400)

plt.figure()
plt.plot(x_plot, y_exakt(x_plot), label="exakte Loesung")
plt.plot(x_eu, y_eu, "o--", label="Euler")
plt.plot(x_he, y_he, "s--", label="Heun")
plt.plot(x_mp, y_mp, "d--", label="Mittelpunkt")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Vergleich Euler, Heun und Mittelpunktverfahren")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.semilogy(x_eu, np.abs(y_eu - y_exakt(x_eu)), "o--", label="Euler")
plt.semilogy(x_he, np.abs(y_he - y_exakt(x_he)), "s--", label="Heun")
plt.semilogy(x_mp, np.abs(y_mp - y_exakt(x_mp)), "d--", label="Mittelpunkt")
plt.xlabel("x")
plt.ylabel("absoluter Fehler")
plt.title("Fehlervergleich")
plt.grid(True)
plt.legend()
plt.show()
