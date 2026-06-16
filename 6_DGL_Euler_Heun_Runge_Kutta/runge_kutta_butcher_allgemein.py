# -*- coding: utf-8 -*-
"""
Allgemeines explizites Runge-Kutta-Verfahren mit Butcher-Tabelle
=================================================================

Aufgabenstellung
----------------
Gegeben ist wieder das Anfangswertproblem

    y'(x) = 2*(1 - x)*y(x),      y(0) = 1

auf dem Intervall [0, 3].

Gesucht ist eine Implementierung, die ein explizites Runge-Kutta-Verfahren
ueber seine Butcher-Tabelle berechnet.

Verglichen werden:

    - Euler-Verfahren                 1-stufig
    - Mittelpunktverfahren            2-stufig
    - Heun-Verfahren                  2-stufig
    - Klassisches RK4                 4-stufig
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

a_intervall = 0.0
y0 = 1.0
b_intervall = 3.0
h = 0.25
n = int((b_intervall - a_intervall) / h)


def y_exakt(x):
    return np.exp(2 * x - x**2)


# -----------------------------------------------------------------------------
# 2. Butcher-Tabellen
# -----------------------------------------------------------------------------

verfahren = {
    "Euler": {
        "A": np.array([[0.0]]),
        "b": np.array([1.0]),
        "c": np.array([0.0]),
    },
    "Mittelpunkt": {
        "A": np.array([[0.0, 0.0],
                       [0.5, 0.0]]),
        "b": np.array([0.0, 1.0]),
        "c": np.array([0.0, 0.5]),
    },
    "Heun": {
        "A": np.array([[0.0, 0.0],
                       [1.0, 0.0]]),
        "b": np.array([0.5, 0.5]),
        "c": np.array([0.0, 1.0]),
    },
    "RK4": {
        "A": np.array([[0.0, 0.0, 0.0, 0.0],
                       [0.5, 0.0, 0.0, 0.0],
                       [0.0, 0.5, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0]]),
        "b": np.array([1/6, 1/3, 1/3, 1/6]),
        "c": np.array([0.0, 0.5, 0.5, 1.0]),
    },
}


# -----------------------------------------------------------------------------
# 3. Allgemeines explizites Runge-Kutta-Verfahren
# -----------------------------------------------------------------------------

def runge_kutta_butcher(f, a, y0, b, n, A, gewicht, c):
    """Berechnet ein explizites RK-Verfahren aus der Butcher-Tabelle."""
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0] = a
    y[0] = y0
    h = (b - a) / n
    s = len(gewicht)                                # Anzahl Stufen

    for i in range(n):
        k = np.zeros(s)

        for r in range(s):
            # Explizit: in Stufe r werden nur die bereits bekannten k_0,...,k_{r-1} verwendet
            y_stufe = y[i] + h * np.sum(A[r, :r] * k[:r])
            x_stufe = x[i] + c[r] * h
            k[r] = f(x_stufe, y_stufe)

        y[i + 1] = y[i] + h * np.dot(gewicht, k)
        x[i + 1] = x[i] + h

    return x, y


# -----------------------------------------------------------------------------
# 4. Berechnung fuer alle Verfahren
# -----------------------------------------------------------------------------

resultate = {}
for name, daten in verfahren.items():
    x_num, y_num = runge_kutta_butcher(
        f,
        a_intervall,
        y0,
        b_intervall,
        n,
        daten["A"],
        daten["b"],
        daten["c"],
    )
    resultate[name] = (x_num, y_num)


# -----------------------------------------------------------------------------
# 5. Resultatausgabe
# -----------------------------------------------------------------------------

print("Allgemeines Runge-Kutta-Verfahren mit Butcher-Tabelle")
print("=====================================================")
print("DGL: y' =", f_sym)
print("h = {:.4f}, n = {}".format(h, n))
print("")
print("Verfahren          y(b) numerisch      Fehler am Endpunkt")
for name, (x_num, y_num) in resultate.items():
    fehler = abs(y_num[-1] - y_exakt(b_intervall))
    print("{:<16s} {:16.10f}      {:12.3e}".format(name, y_num[-1], fehler))

print("\nButcher-Tabelle fuer RK4:")
print("A =")
print(verfahren["RK4"]["A"])
print("b =", verfahren["RK4"]["b"])
print("c =", verfahren["RK4"]["c"])


# -----------------------------------------------------------------------------
# 6. Plot
# -----------------------------------------------------------------------------

x_plot = np.linspace(a_intervall, b_intervall, 400)

plt.figure()
plt.plot(x_plot, y_exakt(x_plot), label="exakte Loesung")
for name, (x_num, y_num) in resultate.items():
    plt.plot(x_num, y_num, "o--", label=name)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Explizite Runge-Kutta-Verfahren via Butcher-Tabelle")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
for name, (x_num, y_num) in resultate.items():
    plt.semilogy(x_num, np.abs(y_num - y_exakt(x_num)), "o--", label=name)
plt.xlabel("x")
plt.ylabel("absoluter Fehler")
plt.title("Fehlervergleich der Butcher-Verfahren")
plt.grid(True)
plt.legend()
plt.show()
