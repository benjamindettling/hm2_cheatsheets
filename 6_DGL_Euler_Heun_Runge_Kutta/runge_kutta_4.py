# -*- coding: utf-8 -*-
"""
Klassisches Runge-Kutta-Verfahren 4. Ordnung
============================================

Aufgabenstellung
----------------
Gegeben ist das Anfangswertproblem

    y'(x) = 2*(1 - x)*y(x),      y(0) = 1

auf dem Intervall [0, 3].

Gesucht ist die numerische Loesung mit dem klassischen RK4-Verfahren.
Zum Vergleich werden Euler, Heun und die exakte Loesung mitgeplottet.
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
h = 0.3
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


def rk4(f, a, y0, b, n):
    """Klassisches Runge-Kutta-Verfahren 4. Ordnung."""
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0] = a
    y[0] = y0
    h = (b - a) / n

    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + h / 2 * k1)
        k3 = f(x[i] + h / 2, y[i] + h / 2 * k2)
        k4 = f(x[i] + h, y[i] + h * k3)

        y[i + 1] = y[i] + h / 6 * (k1 + 2*k2 + 2*k3 + k4)
        x[i + 1] = x[i] + h

    return x, y


x_eu, y_eu = euler(f, a, y0, b, n)
x_he, y_he = heun(f, a, y0, b, n)
x_rk, y_rk = rk4(f, a, y0, b, n)


# -----------------------------------------------------------------------------
# 3. Resultatausgabe
# -----------------------------------------------------------------------------

print("Klassisches Runge-Kutta-Verfahren RK4")
print("=====================================")
print("DGL: y' =", f_sym)
print("h = {:.4f}, n = {}".format(h, n))
print("")
print("Endwert exakt: y({:.1f}) = {:.10f}".format(b, y_exakt(b)))
print("Euler:          {:.10f}, Fehler = {:.3e}".format(y_eu[-1], abs(y_eu[-1] - y_exakt(b))))
print("Heun:           {:.10f}, Fehler = {:.3e}".format(y_he[-1], abs(y_he[-1] - y_exakt(b))))
print("RK4:            {:.10f}, Fehler = {:.3e}".format(y_rk[-1], abs(y_rk[-1] - y_exakt(b))))

# Erste RK4-Schrittgroessen als Rechenweg-Beispiel ausgeben
k1 = f(x_rk[0], y_rk[0])
k2 = f(x_rk[0] + h / 2, y_rk[0] + h / 2 * k1)
k3 = f(x_rk[0] + h / 2, y_rk[0] + h / 2 * k2)
k4 = f(x_rk[0] + h, y_rk[0] + h * k3)
print("\nErster RK4-Schritt:")
print("k1 = {:.8f}, k2 = {:.8f}, k3 = {:.8f}, k4 = {:.8f}".format(k1, k2, k3, k4))
print("y_1 = {:.8f}".format(y_rk[1]))


# -----------------------------------------------------------------------------
# 4. Plot
# -----------------------------------------------------------------------------

x_plot = np.linspace(a, b, 400)

plt.figure()
plt.plot(x_plot, y_exakt(x_plot), label="exakte Loesung")
plt.plot(x_eu, y_eu, "o--", label="Euler")
plt.plot(x_he, y_he, "s--", label="Heun")
plt.plot(x_rk, y_rk, "d--", label="RK4")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Euler, Heun und RK4 im Vergleich")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.semilogy(x_eu, np.abs(y_eu - y_exakt(x_eu)), "o--", label="Euler")
plt.semilogy(x_he, np.abs(y_he - y_exakt(x_he)), "s--", label="Heun")
plt.semilogy(x_rk, np.abs(y_rk - y_exakt(x_rk)), "d--", label="RK4")
plt.xlabel("x")
plt.ylabel("absoluter Fehler")
plt.title("Fehlervergleich")
plt.grid(True)
plt.legend()
plt.show()
