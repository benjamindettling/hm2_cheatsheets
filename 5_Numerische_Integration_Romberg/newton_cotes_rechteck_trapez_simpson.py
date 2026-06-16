# -*- coding: utf-8 -*-
"""
Numerische Integration mit Newton-Cotes-Formeln
================================================

Aufgabenstellung
----------------
Ein Teilchen der Masse m = 10 kg wird in einer Fluessigkeit abgebremst.
Der Widerstand sei

    R(v) = -v * sqrt(v)

Fuer die Zeit beim Abbremsen von v = 20 m/s auf v = 5 m/s entsteht das Integral

    I = integral_5^20 10 / (-v * sqrt(v)) dv.

Dieses Integral soll mit n = 5 Teilintervallen berechnet werden:

    1. summierte Rechteckregel / Mittelpunktregel
    2. summierte Trapezregel
    3. summierte Simpson-Regel als gewichtetes Mittel

Zum Schluss werden die numerischen Werte mit dem exakten Wert verglichen
und die Integrationspunkte geplottet.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% Angaben

m = 10.0
v_start = 5.0      # untere Integralgrenze
v_ende = 20.0      # obere Integralgrenze
n = 5              # Anzahl Teilintervalle


def f(v):
    """Integrand f(v) = m / R(v)."""
    return -m / (v * np.sqrt(v))


# Symbolische Version fuer den exakten Vergleich
v = sp.symbols("v")
f_sym = -m / (v * sp.sqrt(v))


# %% Vorbereitung: Schrittweite und Stuetzstellen

a = v_start
b = v_ende
h = (b - a) / n

x = np.linspace(a, b, n + 1)           # Randpunkte der Teilintervalle
x_mid = x[:-1] + h / 2                 # Mittelpunkte der Teilintervalle

print("Numerische Integration mit Newton-Cotes-Formeln")
print("================================================")
print(f"Integrand: f(v) = -{m:g} / (v * sqrt(v))")
print(f"Intervall: [{a:g}, {b:g}]")
print(f"n = {n}, h = {h:g}")
print()
print("Stuetzstellen:", x)
print("Mittelpunkte:", x_mid)
print()


# %% Rechteckregel / Mittelpunktregel

# R(h) = h * Summe f(x_i + h/2)
I_rechteck = h * np.sum(f(x_mid))


# %% Trapezregel

# T(h) = h * ( (f(a)+f(b))/2 + Summe der inneren Funktionswerte )
I_trapez = h * ((f(a) + f(b)) / 2 + np.sum(f(x[1:-1])))


# %% Simpson-Regel

# In der im Skript verwendeten Schreibweise:
# S(h) = 1/3 * (T(h) + 2 * R(h))
I_simpson = (I_trapez + 2 * I_rechteck) / 3


# %% Exakter Wert mit sympy

I_exakt_sym = sp.integrate(f_sym, (v, a, b))
I_exakt = float(I_exakt_sym)


# %% Resultate ausgeben

print("Resultate")
print("---------")
print(f"Rechteckregel: I_R = {I_rechteck:.10f}")
print(f"Trapezregel:   I_T = {I_trapez:.10f}")
print(f"Simpson-Regel: I_S = {I_simpson:.10f}")
print(f"Exakter Wert:  I   = {I_exakt:.10f}")
print()
print("Absolute Fehler")
print("---------------")
print(f"|I_R - I| = {abs(I_rechteck - I_exakt):.10e}")
print(f"|I_T - I| = {abs(I_trapez - I_exakt):.10e}")
print(f"|I_S - I| = {abs(I_simpson - I_exakt):.10e}")


# %% Plot: Funktion, Stuetzstellen und Mittelpunkte

v_plot = np.linspace(a, b, 400)

plt.figure()
plt.plot(v_plot, f(v_plot), label="f(v) = -10 / (v sqrt(v))")
plt.plot(x, f(x), "o", label="Stuetzstellen Trapez")
plt.plot(x_mid, f(x_mid), "x", label="Mittelpunkte Rechteck")
plt.axhline(0, linewidth=0.8)
plt.xlabel("v")
plt.ylabel("f(v)")
plt.title("Newton-Cotes: Integrand und verwendete Punkte")
plt.grid(True)
plt.legend()
plt.show()


# %% Plot: Fehlervergleich

methoden = ["Rechteck", "Trapez", "Simpson"]
fehler = [abs(I_rechteck - I_exakt), abs(I_trapez - I_exakt), abs(I_simpson - I_exakt)]

plt.figure()
plt.bar(methoden, fehler)
plt.ylabel("absoluter Fehler")
plt.title("Fehlervergleich der Newton-Cotes-Formeln")
plt.grid(True, axis="y")
plt.show()
