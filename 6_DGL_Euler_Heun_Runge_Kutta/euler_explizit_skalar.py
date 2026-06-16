# -*- coding: utf-8 -*-
"""
Explizites Eulerverfahren fuer ein skalares Anfangswertproblem
================================================================

Aufgabenstellung
----------------
Gegeben ist das Anfangswertproblem

    y'(x) = 2*(1 - x)*y(x),      y(0) = 1

auf dem Intervall [0, 3].

Gesucht ist die numerische Loesung mit dem expliziten Eulerverfahren.
Zum Vergleich wird die exakte Loesung

    y(x) = exp(2*x - x**2)

verwendet.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 1. Angaben der Aufgabe
# -----------------------------------------------------------------------------

x_sym, y_sym = sp.symbols("x y")
f_sym = 2 * (1 - x_sym) * y_sym                 # rechte Seite der DGL symbolisch

# Numerische Funktion fuer die Berechnung
f = sp.lambdify((x_sym, y_sym), f_sym, modules="numpy")

a = 0.0                                         # Start des Intervalls
y0 = 1.0                                        # Anfangswert y(a)
b = 3.0                                         # Ende des Intervalls
h = 0.25                                        # Schrittweite
n = int((b - a) / h)                             # Anzahl Schritte


def y_exakt(x):
    """Exakte Loesung zur Kontrolle."""
    return np.exp(2 * x - x**2)


# -----------------------------------------------------------------------------
# 2. Explizites Eulerverfahren
# -----------------------------------------------------------------------------

def euler_explizit(f, a, y0, b, n):
    """Berechnet y' = f(x,y) mit dem expliziten Eulerverfahren."""
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)

    x[0] = a
    y[0] = y0
    h = (b - a) / n

    for i in range(n):
        steigung = f(x[i], y[i])                 # f(x_i, y_i)
        y[i + 1] = y[i] + h * steigung           # Euler-Schritt
        x[i + 1] = x[i] + h

    return x, y


x_num, y_num = euler_explizit(f, a, y0, b, n)


# -----------------------------------------------------------------------------
# 3. Resultatausgabe
# -----------------------------------------------------------------------------

print("Explizites Eulerverfahren")
print("=========================")
print("DGL:        y' =", f_sym)
print("Intervall:  [{:.1f}, {:.1f}]".format(a, b))
print("h:          {:.4f}".format(h))
print("n:          {}".format(n))
print("")
print("Endwert numerisch y({:.1f}) = {:.8f}".format(b, y_num[-1]))
print("Endwert exakt     y({:.1f}) = {:.8f}".format(b, y_exakt(b)))
print("Absoluter Fehler          = {:.8e}".format(abs(y_num[-1] - y_exakt(b))))

print("\nIterationstabelle:")
print(" k      x_k          y_k          y_exakt       Fehler")
for k in range(len(x_num)):
    fehler = abs(y_num[k] - y_exakt(x_num[k]))
    print("{:2d}   {:8.4f}   {:11.7f}   {:11.7f}   {:10.3e}".format(
        k, x_num[k], y_num[k], y_exakt(x_num[k]), fehler
    ))


# -----------------------------------------------------------------------------
# 4. Plot
# -----------------------------------------------------------------------------

x_plot = np.linspace(a, b, 400)

plt.figure()
plt.plot(x_plot, y_exakt(x_plot), label="exakte Loesung")
plt.plot(x_num, y_num, "o--", label="Euler explizit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Explizites Eulerverfahren fuer y' = 2(1-x)y")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(x_num, np.abs(y_num - y_exakt(x_num)), "o--")
plt.xlabel("x")
plt.ylabel("absoluter Fehler")
plt.title("Fehler des expliziten Eulerverfahrens")
plt.grid(True)
plt.show()
