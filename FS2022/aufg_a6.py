"""SEP 22.06.2022, Aufgabe 6b.

Numerische Integration der DGL 3. Ordnung als System 1. Ordnung mit dem
Mittelpunktverfahren fuer h=0.2 und h=0.02.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hm2_toolbox import globalen_fehler_berechnen, mittelpunkt_verfahren_system


def system(x, zustand):
    """System zu x^3 y''' - 3x^2 y'' + 6x y' - 6y = 6x^4.

    z1=y, z2=y', z3=y''
    z1'=z2
    z2'=z3
    z3'=6x + 3 z3/x - 6 z2/x^2 + 6 z1/x^3
    """
    z1, z2, z3 = zustand
    return np.array([
        z2,
        z3,
        6.0 * x + 3.0 * z3 / x - 6.0 * z2 / x**2 + 6.0 * z1 / x**3,
    ])


def exakte_loesung(x):
    """Exakte Loesung y(x)=2x+x^2-x^3+x^4."""
    return 2.0 * x + x**2 - x**3 + x**4


x_start = 1.0
x_ende = 4.0
startzustand = [3.0, 5.0, 8.0]

ergebnisse = {}
for h in [0.2, 0.02]:
    n = int(round((x_ende - x_start) / h))
    ergebnisse[h] = mittelpunkt_verfahren_system(system, x_start, startzustand, x_ende, n)

for h, ergebnis in ergebnisse.items():
    fehler = np.abs(globalen_fehler_berechnen(exakte_loesung, ergebnis.x_werte, ergebnis.y_werte[:, 0]))
    print(f"h={h}")
    print(f"  y_num({x_ende}) = {ergebnis.y_werte[-1, 0]:.12f}")
    print(f"  y_exakt({x_ende}) = {exakte_loesung(x_ende):.12f}")
    print(f"  Endfehler = {fehler[-1]:.12e}")
    print(f"  Maximaler Fehler = {np.max(fehler):.12e}")

fehler_grob = abs(ergebnisse[0.2].y_werte[-1, 0] - exakte_loesung(x_ende))
fehler_fein = abs(ergebnisse[0.02].y_werte[-1, 0] - exakte_loesung(x_ende))
print()
print("Fehlerquotient h=0.2 zu h=0.02:", fehler_grob / fehler_fein)
print("Kommentar: Bei Ordnung p=2 erwartet man bei Faktor 10 kleinerer Schrittweite etwa Faktor 10^2 bessere Genauigkeit.")

plot_ordner = Path(__file__).resolve().parent / "plots"
plot_ordner.mkdir(exist_ok=True)

x_plot = np.linspace(x_start, x_ende, 500)
y_plot = np.array([exakte_loesung(x) for x in x_plot])

plt.figure(figsize=(8, 4.5))
plt.plot(x_plot, y_plot, color="black", label="exakt")
for h, ergebnis in ergebnisse.items():
    plt.plot(ergebnis.x_werte, ergebnis.y_werte[:, 0], marker="o" if h == 0.2 else None, label=f"Mittelpunkt h={h}")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plot_ordner / "aufg_a6_loesungen.png", dpi=160)

plt.figure(figsize=(8, 4.5))
for h, ergebnis in ergebnisse.items():
    fehler = np.abs(globalen_fehler_berechnen(exakte_loesung, ergebnis.x_werte, ergebnis.y_werte[:, 0]))
    plt.semilogy(ergebnis.x_werte, fehler, marker="o" if h == 0.2 else None, label=f"|Fehler| h={h}")
plt.xlabel("x")
plt.ylabel("|y_exakt - y_num|")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig(plot_ordner / "aufg_a6_fehler_semilog.png", dpi=160)

print("Plots gespeichert:", plot_ordner / "aufg_a6_loesungen.png", "und", plot_ordner / "aufg_a6_fehler_semilog.png")

