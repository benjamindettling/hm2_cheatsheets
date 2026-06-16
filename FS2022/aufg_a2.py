"""SEP 22.06.2022, Aufgabe 2a/b.

Natuerlicher kubischer Spline fuer die Werkstueck-Bewegung, Auswertung von
Position, Geschwindigkeit und Beschleunigung bei t=1 sowie Plot der
Geschwindigkeit.
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
    natuerlicher_kubischer_spline_ableitung_auswerten,
    natuerlicher_kubischer_spline_auswerten,
    natuerlicher_kubischer_spline_koeffizienten,
    spline_abschnitt_finden,
    spline_tabelle_erstellen,
)


zeiten = np.array([0.0, 0.5, 2.0, 3.0])
positionen = np.array([1.0, 2.0, 2.5, 0.0])
zeit_auswertung = 1.0

koeffizienten = natuerlicher_kubischer_spline_koeffizienten(zeiten, positionen)
tabelle = spline_tabelle_erstellen(koeffizienten)

position = natuerlicher_kubischer_spline_auswerten(koeffizienten, zeit_auswertung)
geschwindigkeit = natuerlicher_kubischer_spline_ableitung_auswerten(koeffizienten, zeit_auswertung, ordnung=1)
beschleunigung = natuerlicher_kubischer_spline_ableitung_auswerten(koeffizienten, zeit_auswertung, ordnung=2)

print("Spline-Koeffizienten:")
print(tabelle)
print()
print(f"x({zeit_auswertung})  = {position:.12f}")
print(f"x'({zeit_auswertung}) = {geschwindigkeit:.12f}")
print(f"x''({zeit_auswertung})= {beschleunigung:.12f}")

plot_ordner = Path(__file__).resolve().parent / "plots"
plot_ordner.mkdir(exist_ok=True)

t_plot = np.linspace(zeiten[0], zeiten[-1], 400)
v_plot = [
    natuerlicher_kubischer_spline_ableitung_auswerten(koeffizienten, t, ordnung=1)
    for t in t_plot
]

plt.figure(figsize=(7, 4))
plt.plot(t_plot, v_plot, label="Spline-Geschwindigkeit x'(t)")
plt.scatter(zeiten, [natuerlicher_kubischer_spline_ableitung_auswerten(koeffizienten, t, 1) for t in zeiten], color="black", s=20)
plt.axvline(zeit_auswertung, color="tab:red", linestyle="--", label="t=1")
plt.xlabel("t")
plt.ylabel("x'(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plot_ordner / "aufg_a2_geschwindigkeit.png", dpi=160)

print()
print("Plot gespeichert:", plot_ordner / "aufg_a2_geschwindigkeit.png")

