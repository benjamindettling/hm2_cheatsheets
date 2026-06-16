# -*- coding: utf-8 -*-
"""
Stabilitaet des expliziten Eulerverfahrens
==========================================

Aufgabenstellung
----------------
Untersucht wird die lineare Testgleichung

    y'(t) = lambda*y(t),      y(0) = 1

mit lambda = -8.

Fuer das explizite Eulerverfahren gilt

    y_{n+1} = (1 + h*lambda)*y_n.

Stabilitaet liegt fuer reelles negatives lambda vor, wenn

    |1 + h*lambda| < 1.

Gesucht ist ein Vergleich verschiedener Schrittweiten.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 1. Angaben der Aufgabe
# -----------------------------------------------------------------------------

t_sym, y_sym, lam_sym, h_sym = sp.symbols("t y lambda h")
f_sym = lam_sym * y_sym
stabilitaetsfaktor = 1 + h_sym * lam_sym

lam = -8.0
y0 = 1.0
a = 0.0
b = 2.0
h_werte = [0.10, 0.24, 0.27]                      # stabil, knapp stabil, instabil


def f(t, y):
    return lam * y


def y_exakt(t):
    return np.exp(lam * t)


# -----------------------------------------------------------------------------
# 2. Eulerverfahren
# -----------------------------------------------------------------------------

def euler(f, a, y0, b, h):
    n = int((b - a) / h)
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)
    t[0] = a
    y[0] = y0

    for i in range(n):
        y[i + 1] = y[i] + h * f(t[i], y[i])
        t[i + 1] = t[i] + h

    return t, y


# -----------------------------------------------------------------------------
# 3. Resultatausgabe
# -----------------------------------------------------------------------------

print("Stabilitaet des expliziten Eulerverfahrens")
print("==========================================")
print("Testgleichung: y' = lambda*y mit lambda = {:.1f}".format(lam))
print("Symbolischer Stabilitaetsfaktor:", stabilitaetsfaktor)
print("Stabilitaetsbedingung: |1 + h*lambda| < 1")
print("Fuer lambda = -8 gilt: 0 < h < {:.4f}".format(2 / abs(lam)))
print("")
print("h          |1+h*lambda|      stabil?")
for h in h_werte:
    faktor = abs(1 + h * lam)
    stabil = faktor < 1
    print("{:.4f}     {:12.6f}      {}".format(h, faktor, stabil))


# -----------------------------------------------------------------------------
# 4. Plot
# -----------------------------------------------------------------------------

t_plot = np.linspace(a, b, 500)

plt.figure()
plt.plot(t_plot, y_exakt(t_plot), label="exakte Loesung")

for h in h_werte:
    t_num, y_num = euler(f, a, y0, b, h)
    faktor = abs(1 + h * lam)
    if faktor < 1:
        label = "Euler h={:.2f} stabil".format(h)
    else:
        label = "Euler h={:.2f} instabil".format(h)
    plt.plot(t_num, y_num, "o--", label=label)

plt.xlabel("t")
plt.ylabel("y")
plt.title("Stabilitaet des expliziten Eulerverfahrens")
plt.grid(True)
plt.legend()
plt.show()
