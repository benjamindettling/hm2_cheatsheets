"""SEP 22.06.2022, Aufgabe 3a/b.

Plot des logistischen Wachstumsmodells und Parameterfit mit gedaempftem
Gauss-Newton-Verfahren.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hm2_toolbox import (
    gedaempftes_gauss_newton_verfahren,
    jacobi_residuen_numerisch,
    residuentabelle,
)


def hefemodell(t, parameter):
    """Logistisches Modell H(t)=G/(1+(G/A-1)*exp(-K*G*t))."""
    A, G, K = parameter
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        wert = G / (1.0 + (G / A - 1.0) * np.exp(-K * G * t))
    return float(wert)


def werte_vektorisiert(t_werte, parameter):
    """Wertet das Hefemodell fuer viele Zeiten aus."""
    return np.array([hefemodell(t, parameter) for t in t_werte], dtype=float)


plot_ordner = Path(__file__).resolve().parent / "plots"
plot_ordner.mkdir(exist_ok=True)

# Aufgabe 3a
parameter_a = np.array([25.0, 500.0, 0.0025])
t_plot = np.linspace(0.0, 10.0, 400)
plt.figure(figsize=(7, 4))
plt.plot(t_plot, werte_vektorisiert(t_plot, parameter_a), label="A=25, G=500, K=0.0025")
plt.xlabel("t")
plt.ylabel("H(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plot_ordner / "aufg_a3_modell.png", dpi=160)

# Aufgabe 3b
zeiten = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
messwerte = np.array([52.9, 184.0, 426.0, 529.0, 499.0, 510.0])
startparameter = np.array([20.0, 450.0, 0.001])

ergebnis = None
minimales_p_max = None
for p_max in range(0, 11):
    try:
        kandidat = gedaempftes_gauss_newton_verfahren(
            hefemodell,
            lambda p: jacobi_residuen_numerisch(hefemodell, zeiten, p),
            zeiten,
            messwerte,
            startparameter,
            toleranz=1e-7,
            maximale_iterationen=200,
            p_max=p_max,
            rueckgabe_verlauf=True,
        )
    except np.linalg.LinAlgError:
        continue
    if kandidat.konvergiert:
        minimales_p_max = p_max
        ergebnis = kandidat
        break

if ergebnis is None:
    raise RuntimeError("Gauss-Newton ist fuer p_max<=10 nicht konvergiert.")

parameter_fit = ergebnis.loesung
modellwerte = werte_vektorisiert(zeiten, parameter_fit)

print("Minimal erforderliches p_max:", minimales_p_max)
print("Gefittete Parameter [A, G, K]:", parameter_fit)
print("Iterationen:", ergebnis.iterationen)
print("Residualnorm:", ergebnis.residualnorm)
print()
print("Residuentabelle:")
print(residuentabelle(zeiten, messwerte, modellwerte))
print()
print("Letzte Iterationen:")
print(ergebnis.verlauf_als_tabelle().tail())

plt.figure(figsize=(7, 4))
plt.scatter(zeiten, messwerte, color="black", label="Messdaten")
plt.plot(t_plot, werte_vektorisiert(t_plot, parameter_fit), color="tab:red", label="Gauss-Newton-Fit")
plt.xlabel("t")
plt.ylabel("H(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plot_ordner / "aufg_a3_fit.png", dpi=160)

print()
print("Plots gespeichert:", plot_ordner / "aufg_a3_modell.png", "und", plot_ordner / "aufg_a3_fit.png")
