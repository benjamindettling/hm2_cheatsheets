"""SEP 22.06.2022, Aufgabe 1b.

Bestimmt fuer c=1 die Anzahl Newton-Iterationen, bis der maximale absolute
Fehler zur angegebenen exakten Loesung hoechstens 1e-4 ist.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hm2_toolbox import newton_verfahren


c = 1.0
startwert = np.array([0.0, 3.0])
toleranz_fehler = 1e-4


def funktion(punkt):
    """Gleichungssystem f(x,y)=0 aus der Aufgabe."""
    x, y = punkt
    return [c * x + y - 4.0, x**2 + y**2 - 9.0]


def jacobi(punkt):
    """Jacobi-Matrix des Gleichungssystems."""
    x, y = punkt
    return [[c, 1.0], [2.0 * x, 2.0 * y]]


exakte_loesung = np.array([2.0 - math.sqrt(2.0) / 2.0, 2.0 + math.sqrt(2.0) / 2.0])

ergebnis = newton_verfahren(
    funktion,
    jacobi,
    startwert,
    toleranz=1e-14,
    maximale_iterationen=20,
    rueckgabe_verlauf=True,
)

print("Exakte Loesung fuer c=1:", exakte_loesung)
print("Newton-Iteration ab Startwert:", startwert)
print()

benoetigte_iteration = None
for eintrag in ergebnis.verlauf:
    iteration = eintrag["iteration"] + 1
    x_neu = eintrag["x"] + eintrag["delta"]
    maximaler_fehler = np.max(np.abs(x_neu - exakte_loesung))
    print(f"k={iteration:2d}, x_k={x_neu}, max_abs_fehler={maximaler_fehler:.6e}")
    if benoetigte_iteration is None and maximaler_fehler <= toleranz_fehler:
        benoetigte_iteration = iteration

print()
print(f"Benötigte Iterationen fuer max. absoluten Fehler <= {toleranz_fehler:g}: {benoetigte_iteration}")

