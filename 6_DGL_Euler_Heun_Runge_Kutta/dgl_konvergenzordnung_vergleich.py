# -*- coding: utf-8 -*-
"""
Experimentelle Bestimmung der Konvergenzordnung
================================================

Aufgabenstellung
----------------
Gegeben ist das Anfangswertproblem

    y'(x) = 2*(1 - x)*y(x),      y(0) = 1

auf dem Intervall [0, 3].

Gesucht ist der globale Fehler am Endpunkt fuer verschiedene Schrittweiten.
Aus den Fehlern wird die experimentelle Konvergenzordnung bestimmt:

    p ≈ log(error_h / error_halbe) / log(2).

Verglichen werden Euler, Heun und RK4.
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
h_werte = np.array([0.3, 0.15, 0.075, 0.0375])


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


methoden = {
    "Euler": euler,
    "Heun": heun,
    "RK4": rk4,
}


# -----------------------------------------------------------------------------
# 3. Fehler und Ordnung berechnen
# -----------------------------------------------------------------------------

fehler = {}
ordnungen = {}

for name, methode in methoden.items():
    fehler[name] = []

    for h in h_werte:
        n = int((b - a) / h)
        x_num, y_num = methode(f, a, y0, b, n)
        fehler_endpunkt = abs(y_num[-1] - y_exakt(b))
        fehler[name].append(fehler_endpunkt)

    fehler[name] = np.array(fehler[name])
    ordnungen[name] = np.log(fehler[name][:-1] / fehler[name][1:]) / np.log(2)


# -----------------------------------------------------------------------------
# 4. Resultatausgabe
# -----------------------------------------------------------------------------

print("Experimentelle Konvergenzordnung")
print("=================================")
print("DGL: y' =", f_sym)
print("Intervall: [{:.1f}, {:.1f}]".format(a, b))
print("Exakter Endwert y({:.1f}) = {:.10f}".format(b, y_exakt(b)))

for name in methoden:
    print("\n" + name)
    print("h          Fehler am Endpunkt")
    for h, err in zip(h_werte, fehler[name]):
        print("{:.5f}    {:.8e}".format(h, err))
    print("Geschaetzte Ordnungen:", np.round(ordnungen[name], 4))
    print("Mittlere Ordnung: {:.4f}".format(np.mean(ordnungen[name])))


# -----------------------------------------------------------------------------
# 5. Plot
# -----------------------------------------------------------------------------

plt.figure()
for name in methoden:
    plt.loglog(h_werte, fehler[name], "o--", label=name)
plt.xlabel("Schrittweite h")
plt.ylabel("Fehler am Endpunkt")
plt.title("Fehler gegen Schrittweite")
plt.grid(True, which="both")
plt.legend()
plt.show()
