# -*- coding: utf-8 -*-
"""
Newton-Interpolation mit dividierten Differenzen

Aufgabenstellung
================
Gegeben sind Stützpunkte einer Messreihe. Gesucht ist das Newtonsche
Interpolationspolynom. Dazu wird zuerst die Tabelle der dividierten
Differenzen berechnet. Danach wird das Polynom symbolisch aufgebaut,
an einem Punkt ausgewertet und geplottet.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% Angaben

x = sp.symbols("x")

# Gegebene Datenpunkte
x_data = np.array([0.0, 1.0, 2.0, 4.0], dtype=float)
y_data = np.array([1.0, 2.2, 1.8, 4.6], dtype=float)

# Gesuchter Wert
x_eval = 3.0


# %% Tabelle der dividierten Differenzen berechnen

def dividierte_differenzen(x_data, y_data):
    """Erstellt die obere Dreieckstabelle der dividierten Differenzen."""
    n = len(x_data)
    table = np.zeros((n, n))

    # Erste Spalte: Funktionswerte y_i
    table[:, 0] = y_data

    # Weitere Spalten: dividierte Differenzen
    for j in range(1, n):
        for i in range(n - j):
            zaehler = table[i + 1, j - 1] - table[i, j - 1]
            nenner = x_data[i + j] - x_data[i]
            table[i, j] = zaehler / nenner

    return table


dd_table = dividierte_differenzen(x_data, y_data)

# Die Newton-Koeffizienten stehen in der ersten Zeile der Tabelle
coeff = dd_table[0, :]


# %% Newton-Polynom symbolisch aufbauen

P = coeff[0]
produkt = 1

for k in range(1, len(x_data)):
    produkt *= (x - x_data[k - 1])
    P += coeff[k] * produkt

P = sp.expand(P)
P_func = sp.lambdify(x, P, modules="numpy")
y_eval = P_func(x_eval)


# %% Resultatausgabe

print("Newton-Interpolation mit dividierten Differenzen")
print("================================================")
print("Gegebene Stützpunkte:")
for xi, yi in zip(x_data, y_data):
    print(f"  ({xi:5.2f}, {yi:8.4f})")

print("\nTabelle der dividierten Differenzen:")
for i in range(len(x_data)):
    zeile = []
    for j in range(len(x_data)):
        if j <= len(x_data) - 1 - i:
            zeile.append(f"{dd_table[i, j]:12.6f}")
        else:
            zeile.append("            ")
    print(" ".join(zeile))

print("\nNewton-Koeffizienten:")
for k, ak in enumerate(coeff):
    print(f"  a_{k} = {ak:.8f}")

print("\nInterpolationspolynom:")
print(f"P(x) = {P}")
print(f"\nAuswertung: P({x_eval}) = {y_eval:.8f}")


# %% Plot

x_plot = np.linspace(x_data.min(), x_data.max(), 400)
y_plot = P_func(x_plot)

plt.figure()
plt.plot(x_plot, y_plot, label="Newton-Interpolationspolynom")
plt.plot(x_data, y_data, "o", label="Stützpunkte")
plt.plot(x_eval, y_eval, "x", markersize=10, label=f"P({x_eval})")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Newton-Interpolation mit dividierten Differenzen")
plt.grid(True)
plt.legend()
plt.show()
