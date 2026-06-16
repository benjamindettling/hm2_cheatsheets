# -*- coding: utf-8 -*-
"""
Vergleich mehrerer Integrationsverfahren
========================================

Aufgabenstellung
----------------
Fuer die Funktion

    f(x) = exp(-x^2) * (1 + 0.2*sin(6x))

soll das Integral auf [0, 1.5] numerisch berechnet werden.
Verglichen werden:

    1. summierte Rechteckregel
    2. summierte Trapezregel
    3. summierte Simpson-Regel
    4. Romberg-Extrapolation
    5. Gauss-Quadratur G1, G2, G3

Das Skript gibt eine kompakte Resultattabelle aus und plottet den Fehler
gegen die Anzahl verwendeter Funktionsauswertungen.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% Angaben

x = sp.symbols("x")
a = 0.0
b = 1.5
n = 8       # Teilintervalle fuer Newton-Cotes
m = 4       # Romberg-Stufe, also feinste Trapezregel mit 2^m Intervallen

f_sym = sp.exp(-x**2) * (1 + sp.Rational(1, 5) * sp.sin(6 * x))
f = sp.lambdify(x, f_sym, "numpy")
# Die Funktion ist absichtlich nicht so gewaehlt, dass eine einfache
# Stammfunktion sichtbar ist. Als Referenz verwenden wir eine sehr feine
# Trapezregel. Das ist fuer den Methodenvergleich in diesem Skript ausreichend.
x_ref = np.linspace(a, b, 500000)
I_exakt = np.trapezoid(f(x_ref), x_ref)

print("Vergleich mehrerer Integrationsverfahren")
print("========================================")
print(f"f(x) = {f_sym}")
print(f"Intervall: [{a}, {b}]")
print(f"Referenzwert I = {I_exakt:.12f}")
print()


# %% Newton-Cotes-Verfahren


def rechteckregel(f, a, b, n):
    h = (b - a) / n
    xs = np.linspace(a, b, n + 1)
    x_mid = xs[:-1] + h / 2
    return h * np.sum(f(x_mid))


def trapezregel(f, a, b, n):
    h = (b - a) / n
    xs = np.linspace(a, b, n + 1)
    return h * ((f(a) + f(b)) / 2 + np.sum(f(xs[1:-1])))


def simpsonregel(f, a, b, n):
    # Simpson in der Form S = 1/3 * (T + 2R)
    return (trapezregel(f, a, b, n) + 2 * rechteckregel(f, a, b, n)) / 3


I_R = rechteckregel(f, a, b, n)
I_T = trapezregel(f, a, b, n)
I_S = simpsonregel(f, a, b, n)


# %% Romberg-Extrapolation

R = np.zeros((m + 1, m + 1))
for j in range(m + 1):
    R[j, 0] = trapezregel(f, a, b, 2**j)

for k in range(1, m + 1):
    for j in range(k, m + 1):
        R[j, k] = (4**k * R[j, k - 1] - R[j - 1, k - 1]) / (4**k - 1)

I_Romberg = R[m, m]


# %% Gauss-Quadratur G1, G2, G3


def transformiere_knoten(t, a, b):
    return (b - a) / 2 * t + (a + b) / 2


def gauss_quadratur(f, a, b, t, w):
    x_nodes = transformiere_knoten(t, a, b)
    return (b - a) / 2 * np.sum(w * f(x_nodes))


I_G1 = gauss_quadratur(f, a, b, np.array([0.0]), np.array([2.0]))
I_G2 = gauss_quadratur(
    f,
    a,
    b,
    np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)]),
    np.array([1.0, 1.0]),
)
I_G3 = gauss_quadratur(
    f,
    a,
    b,
    np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]),
    np.array([5 / 9, 8 / 9, 5 / 9]),
)


# %% Resultattabelle

namen = np.array([
    "Rechteck",
    "Trapez",
    "Simpson",
    "Romberg",
    "Gauss G1",
    "Gauss G2",
    "Gauss G3",
])

werte = np.array([I_R, I_T, I_S, I_Romberg, I_G1, I_G2, I_G3])
fehler = np.abs(werte - I_exakt)

# ungefaehre Anzahl eindeutiger Funktionsauswertungen
# Rechteck: n Mittelpunkte
# Trapez: n+1 Rand-/Gitterpunkte
# Simpson in dieser Form: n Mittelpunkte + n+1 Gitterpunkte
# Romberg optimiert betrachtet: 2^m + 1 Gitterpunkte
# Gauss: Anzahl Gauss-Knoten
anzahl_f = np.array([n, n + 1, 2 * n + 1, 2**m + 1, 1, 2, 3])

print("Resultattabelle")
print("---------------")
print("Methode          Auswertungen        Wert              Fehler")
for name, evals, wert, err in zip(namen, anzahl_f, werte, fehler):
    print(f"{name:12s} {evals:10d}   {wert: .12f}   {err:.4e}")
print()

print("Romberg-Tabelle")
print("---------------")
for j in range(m + 1):
    zeile = []
    for k in range(j + 1):
        zeile.append(f"{R[j,k]: .10f}")
    print("   ".join(zeile))


# %% Plot 1: Funktion

x_plot = np.linspace(a, b, 600)
plt.figure()
plt.plot(x_plot, f(x_plot), label="f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Integrand fuer den Verfahrensvergleich")
plt.grid(True)
plt.legend()
plt.show()


# %% Plot 2: Fehler gegen Anzahl Funktionsauswertungen

plt.figure()
plt.loglog(anzahl_f, fehler, "o")
for name, evals, err in zip(namen, anzahl_f, fehler):
    plt.annotate(name, (evals, err), textcoords="offset points", xytext=(5, 5))
plt.xlabel("Anzahl Funktionsauswertungen")
plt.ylabel("absoluter Fehler")
plt.title("Effizienzvergleich der Integrationsverfahren")
plt.grid(True, which="both")
plt.show()
