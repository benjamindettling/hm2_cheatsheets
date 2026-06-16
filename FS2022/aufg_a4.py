"""SEP 22.06.2022, Aufgabe 4b.

Integral von 0 bis 4 ueber 6x^2-2x mit Simpson fuer die Schrittweiten h=4
und h=1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hm2_toolbox import bestimmtes_integral_symbolisch, simpsonregel, summierte_simpsonregel


def funktion(x):
    """Integrand f(x)=6x^2-2x."""
    return 6.0 * x**2 - 2.0 * x


untere_grenze = 0.0
obere_grenze = 4.0

# h=4 bedeutet ein Simpson-Panel ueber [0,4], also die einfache Simpsonregel.
wert_h_4 = simpsonregel(funktion, untere_grenze, obere_grenze)

# h=1 bedeutet vier Teilintervalle auf [0,4], also summierte Simpsonregel mit n=4.
wert_h_1 = summierte_simpsonregel(funktion, untere_grenze, obere_grenze, anzahl_intervalle=4).wert

x = sp.symbols("x")
exakt = bestimmtes_integral_symbolisch(6 * x**2 - 2 * x, x, 0, 4)

print("Simpson mit h=4:", wert_h_4)
print("Simpson mit h=1:", wert_h_1)
print("Exakter Wert:", exakt)
print()
print("Beobachtung: Simpson ist fuer Polynome bis Grad 3 exakt; deshalb sind beide Werte exakt 112.")
print("Das erklaert auch die Romberg-Beobachtung aus Teil a: Extrapolation trifft hier bereits sehr frueh den exakten Wert.")

