"""SEP 22.06.2022, Aufgabe 5b.

Numerische Loesung von y'=x^2-y, y(0)=1 mit dem zweistufigen
Runge-Kutta-Verfahren aus dem Butcher-Schema der Aufgabe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hm2_toolbox import explizites_runge_kutta_verfahren, richtungsfeld_daten


def funktion(x, y):
    """Rechte Seite der DGL y'=x^2-y."""
    return x**2 - y


x_start = 0.0
x_ende = 2.0
y_start = 1.0
schrittweite = 0.1
anzahl_schritte = int(round((x_ende - x_start) / schrittweite))

# Butcher-Schema:
# 0   |
# 2/3 | 2/3
# ----+---------
#     | 1/4 3/4
butcher_a = [[0.0, 0.0], [2.0 / 3.0, 0.0]]
butcher_b = [1.0 / 4.0, 3.0 / 4.0]
butcher_c = [0.0, 2.0 / 3.0]

ergebnis = explizites_runge_kutta_verfahren(
    funktion,
    x_start,
    y_start,
    x_ende,
    anzahl_schritte,
    butcher_a,
    butcher_b,
    butcher_c,
    methode_name="zweistufiges RK-Verfahren aus Aufgabe 5",
)

print(ergebnis.als_tabelle())
print()
print(f"y({x_ende}) ~= {ergebnis.y_werte[-1]:.12f}")

plot_ordner = Path(__file__).resolve().parent / "plots"
plot_ordner.mkdir(exist_ok=True)

feld = richtungsfeld_daten(funktion, (0.0, 2.0), (0.0, 2.0), anzahl_x=9, anzahl_y=9)

plt.figure(figsize=(7, 4))
plt.quiver(feld["X"], feld["Y"], feld["U"], feld["V"], color="0.65")
plt.plot(ergebnis.x_werte, ergebnis.y_werte, marker="o", color="tab:blue", label="RK2-Loesung")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0, 2)
plt.ylim(0, 2)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plot_ordner / "aufg_a5_rk2_richtungsfeld.png", dpi=160)

print("Plot gespeichert:", plot_ordner / "aufg_a5_rk2_richtungsfeld.png")
