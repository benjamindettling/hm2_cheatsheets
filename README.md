# HM2 All-in-one Toolbox

## Import-Hinweis

Diese drei Imports werden in allen Beispielen vorausgesetzt und als gegeben angenommen:

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
```

Alle weiteren benoetigten Imports, z. B. `import pandas as pd` oder `import math`, sind in den Tabellen pro Definition separat aufgefuehrt. Die kanonische Datei ist [`hm2_all_in_one.py`](hm2_all_in_one.py); die aufgeteilten Module in `hm2_toolbox/` sind Legacy.

## Inhaltsverzeichnis

- [Skripte](#skripte)
- [Definitionen](#definitionen)
  - [Datentypen](#datentypenpy)
  - [Validierung](#validierungpy)
  - [Lineare Algebra](#lineare-algebrapy)
  - [Nichtlineare Gleichungen](#nichtlineare-gleichungenpy)
  - [Ausgleichsrechnung](#ausgleichsrechnungpy)
  - [Interpolation](#interpolationpy)
  - [Integration](#integrationpy)
  - [Differentialgleichungen und Plotting](#differentialgleichungenpy)
  - [Fehleranalyse](#fehleranalysepy)
  - [Analysis-Hilfen](#analysis-hilfenpy)
  - [Daten und Tabellen](#daten-tabellenpy)
  - [Physik-Beispiele](#physik-beispielepy)

## Skripte

| Skript | Verwendung |
|---|---|
| [`a4_test.py`](a4_test.py) | Lokales Tests-/Experimentierskript. |
| [`examples/demo_hm2_pruefungsaufgaben.py`](examples/demo_hm2_pruefungsaufgaben.py) | Demo-Skript mit Beispielaufrufen. |
| [`FS2022/aufg_a1.py`](FS2022/aufg_a1.py) | Loesungsskript zu einer Code-Aufgabe aus FS2022. |
| [`FS2022/aufg_a2.py`](FS2022/aufg_a2.py) | Loesungsskript zu einer Code-Aufgabe aus FS2022. |
| [`FS2022/aufg_a3.py`](FS2022/aufg_a3.py) | Loesungsskript zu einer Code-Aufgabe aus FS2022. |
| [`FS2022/aufg_a4.py`](FS2022/aufg_a4.py) | Loesungsskript zu einer Code-Aufgabe aus FS2022. |
| [`FS2022/aufg_a5.py`](FS2022/aufg_a5.py) | Loesungsskript zu einer Code-Aufgabe aus FS2022. |
| [`FS2022/aufg_a6.py`](FS2022/aufg_a6.py) | Loesungsskript zu einer Code-Aufgabe aus FS2022. |
| [`FS2025/aufg_a4.py`](FS2025/aufg_a4.py) | Beispiel-/Referenzloesung aus FS2025. |
| [`FS2025/aufg_a5.py`](FS2025/aufg_a5.py) | Beispiel-/Referenzloesung aus FS2025. |
| [`FS2025/aufg_a6.py`](FS2025/aufg_a6.py) | Beispiel-/Referenzloesung aus FS2025. |
| [`hm2_all_in_one.py`](hm2_all_in_one.py) | Kanonische All-in-one-Datei; alle Links unten zeigen hierhin. |
| [`hm2_toolbox/__init__.py`](hm2_toolbox/__init__.py) | Legacy-Modul der urspruenglich aufgeteilten Toolbox. |
| [`hm2_toolbox/analysis_hilfen.py`](hm2_toolbox/analysis_hilfen.py) | Legacy-Modul der urspruenglich aufgeteilten Toolbox. |
| [`hm2_toolbox/ausgleichsrechnung.py`](hm2_toolbox/ausgleichsrechnung.py) | Legacy-Modul der urspruenglich aufgeteilten Toolbox. |
| [`hm2_toolbox/daten_tabellen.py`](hm2_toolbox/daten_tabellen.py) | Legacy-Modul der urspruenglich aufgeteilten Toolbox. |
| [`hm2_toolbox/datentypen.py`](hm2_toolbox/datentypen.py) | Legacy-Modul der urspruenglich aufgeteilten Toolbox. |
| [`hm2_toolbox/differentialgleichungen.py`](hm2_toolbox/differentialgleichungen.py) | Legacy-Modul der urspruenglich aufgeteilten Toolbox. |
| [`hm2_toolbox/fehleranalyse.py`](hm2_toolbox/fehleranalyse.py) | Legacy-Modul der urspruenglich aufgeteilten Toolbox. |
| [`hm2_toolbox/integration.py`](hm2_toolbox/integration.py) | Legacy-Modul der urspruenglich aufgeteilten Toolbox. |
| [`hm2_toolbox/interpolation.py`](hm2_toolbox/interpolation.py) | Legacy-Modul der urspruenglich aufgeteilten Toolbox. |
| [`hm2_toolbox/lineare_algebra.py`](hm2_toolbox/lineare_algebra.py) | Legacy-Modul der urspruenglich aufgeteilten Toolbox. |
| [`hm2_toolbox/nichtlineare_gleichungen.py`](hm2_toolbox/nichtlineare_gleichungen.py) | Legacy-Modul der urspruenglich aufgeteilten Toolbox. |
| [`hm2_toolbox/physik_beispiele.py`](hm2_toolbox/physik_beispiele.py) | Legacy-Modul der urspruenglich aufgeteilten Toolbox. |
| [`hm2_toolbox/validierung.py`](hm2_toolbox/validierung.py) | Legacy-Modul der urspruenglich aufgeteilten Toolbox. |
| [`tests/test_smoke.py`](tests/test_smoke.py) | Smoke-Test-Skript. |

## Definitionen

Alle 200 Top-Level-Definitionen sind unten nach dem urspruenglichen logischen Skript gruppiert. Private Hilfsfunktionen sind als `Intern` markiert, bleiben aber verlinkt, weil sie im All-in-one-File sichtbar sind.

### Datentypen
<a id="datentypenpy"></a>

Diese Tabelle enthaelt die zentralen Ergebnis-Klassen der Toolbox.

| Definition | Inputs | Output | Verwendung | Extra Imports |
|---|---|---|---|---|
| [`IterationsErgebnis`](hm2_all_in_one.py#L27) | `loesung, iterationen, konvergiert, residualnorm, schrittnorm, verlauf, nachricht` | Dataclass mit `als_dict()` und `verlauf_als_tabelle()`. | Einheitliche Rueckgabe fuer Newton-, Gauss-Newton- und aehnliche Verfahren. | import pandas as pd<br>from dataclasses import asdict<br>from dataclasses import dataclass<br>from typing import Any |
| [`AusgleichsErgebnis`](hm2_all_in_one.py#L102) | `koeffizienten, residuen, fehlerfunktional, rang, konditionszahl, methode, tabelle` | Dataclass mit Dictionary- und Tabellenzugriff. | Koeffizienten, Residuen und Qualitaetskennzahlen zusammenhalten. | import pandas as pd<br>from dataclasses import asdict<br>from dataclasses import dataclass<br>from typing import Any |
| [`SplineKoeffizienten`](hm2_all_in_one.py#L161) | `stuetzstellen, a_werte, b_werte, c_werte, d_werte` | Dataclass mit Tabellenzugriff. | Speichert die Abschnittskoeffizienten in S_i(x)=a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3. | import pandas as pd<br>from dataclasses import asdict<br>from dataclasses import dataclass<br>from typing import Any |
| [`QuadraturErgebnis`](hm2_all_in_one.py#L237) | `wert, methode, schrittweite, anzahl_intervalle, fehlergrenze, zusatzinfos` | Dataclass mit Dictionary- und Tabellenzugriff. | Wert, Methode, Schrittweite und Zusatzinfos gemeinsam zurueckgeben. | import pandas as pd<br>from dataclasses import asdict<br>from dataclasses import dataclass |
| [`DglErgebnis`](hm2_all_in_one.py#L298) | `x_werte, y_werte, methode, schrittweite, ordnung, tabelle` | Dataclass mit Dictionary- und Tabellenzugriff. | x-Werte, y-Werte, Methode und Tabelle gemeinsam zurueckgeben. | import pandas as pd<br>from dataclasses import asdict<br>from dataclasses import dataclass<br>from typing import Any |

### Validierung
<a id="validierungpy"></a>

| Definition | Inputs | Output | Verwendung | Extra Imports |
|---|---|---|---|---|
| [`als_vektor`](hm2_all_in_one.py#L371) | `eingabe, name='vektor'` | np.ndarray | Einheitliche Vektorform fuer numerische Verfahren. | - |
| [`als_matrix`](hm2_all_in_one.py#L389) | `eingabe, name='matrix'` | np.ndarray | Einheitliche Matrixform fuer LGS, QR und Ausgleichsrechnung. | - |
| [`als_sympy_matrix`](hm2_all_in_one.py#L409) | `eingabe, name='matrix'` | sp.Matrix | Symbolische Matrixoperationen mit deutscher Wrapper-API. | - |
| [`pruefe_endliche_werte`](hm2_all_in_one.py#L425) | `array, name='array'` | None | NaN und Inf frueh abfangen. | - |
| [`pruefe_gleiche_laenge`](hm2_all_in_one.py#L442) | `*arrays` | None | Messwerte, Stuetzstellen und Modellwerte passend halten. | - |
| [`pruefe_quadratische_matrix`](hm2_all_in_one.py#L459) | `matrix` | None | Determinante, Inverse und quadratische LGS absichern. | - |
| [`pruefe_dimensionen_matrix_vektor`](hm2_all_in_one.py#L476) | `matrix, vektor` | None | Lineare Gleichungssysteme vorab validieren. | - |
| [`pruefe_stuetzstellen_paarweise_verschieden`](hm2_all_in_one.py#L494) | `stuetzstellen` | None | Interpolation ohne Division durch null. | - |
| [`pruefe_stuetzstellen_streng_steigend`](hm2_all_in_one.py#L511) | `stuetzstellen` | None | Splines und nichtaequidistante Integration absichern. | - |
| [`pruefe_positive_schrittweite`](hm2_all_in_one.py#L528) | `schrittweite` | None | DGL- und Quadraturverfahren absichern. | - |
| [`pruefe_anzahl_intervalle`](hm2_all_in_one.py#L544) | `anzahl_intervalle` | None | Summierte Quadratur- und DGL-Verfahren absichern. | import numbers |
| [`pruefe_gerade_anzahl_intervalle`](hm2_all_in_one.py#L561) | `anzahl_intervalle` | None | Summierte Simpson-Regel absichern. | import numbers |
| [`ist_skalar`](hm2_all_in_one.py#L579) | `wert` | bool | Parameter wie Schrittweite oder Toleranz validieren. | - |

### Lineare Algebra
<a id="lineare-algebrapy"></a>

| Definition | Inputs | Output | Verwendung | Extra Imports |
|---|---|---|---|---|
| [`_kurzdoc`](hm2_all_in_one.py#L602) | `name: str, formel: str, beispiel: str` | str | Erstellt einen kurzen deutschen Standard-Docstring fuer interne Zwecke. | - |
| [`vektor_norm_1`](hm2_all_in_one.py#L616) | `vektor` | float | Berechnet die 1-Norm eines Vektors. | - |
| [`vektor_norm_2`](hm2_all_in_one.py#L629) | `vektor` | float | Berechnet die euklidische 2-Norm. | - |
| [`vektor_norm_inf`](hm2_all_in_one.py#L641) | `vektor` | float | Berechnet die Maximumsnorm eines Vektors. | - |
| [`matrix_norm_1`](hm2_all_in_one.py#L653) | `matrix` | float | Berechnet die Matrix-1-Norm. | - |
| [`matrix_norm_2`](hm2_all_in_one.py#L665) | `matrix` | float | Berechnet die Spektralnorm einer Matrix. | - |
| [`matrix_norm_inf`](hm2_all_in_one.py#L677) | `matrix` | float | Berechnet die Matrix-Unendlichnorm. | - |
| [`vektor_laenge`](hm2_all_in_one.py#L689) | `vektor` | float | Berechnet die Laenge eines Vektors. | - |
| [`einheitsvektor`](hm2_all_in_one.py#L701) | `vektor` | np.ndarray | Normiert einen Vektor auf Laenge 1. | - |
| [`skalarprodukt`](hm2_all_in_one.py#L717) | `vektor_a, vektor_b` | float | Berechnet das Skalarprodukt zweier Vektoren. | - |
| [`kreuzprodukt_3d`](hm2_all_in_one.py#L733) | `vektor_a, vektor_b` | np.ndarray | Berechnet das Kreuzprodukt in R^3. | - |
| [`winkel_zwischen_vektoren`](hm2_all_in_one.py#L749) | `vektor_a, vektor_b, in_grad=False` | float | Berechnet den Winkel zwischen zwei Vektoren. | - |
| [`sind_orthogonal`](hm2_all_in_one.py#L770) | `vektor_a, vektor_b, toleranz=1e-10` | bool | Prueft Orthogonalitaet zweier Vektoren. | - |
| [`matrix_addieren`](hm2_all_in_one.py#L782) | `matrix_a, matrix_b` | np.ndarray | Addiert zwei Matrizen. | - |
| [`matrix_subtrahieren`](hm2_all_in_one.py#L797) | `matrix_a, matrix_b` | np.ndarray | Subtrahiert zwei Matrizen. | - |
| [`matrix_skalieren`](hm2_all_in_one.py#L812) | `skalare_zahl, matrix` | np.ndarray | Multipliziert eine Matrix mit einem Skalar. | - |
| [`matrix_multiplizieren`](hm2_all_in_one.py#L824) | `matrix_a, matrix_b` | np.ndarray | Multipliziert zwei Matrizen. | - |
| [`matrix_transponieren`](hm2_all_in_one.py#L839) | `matrix` | Rueckgabewert gemaess Beschreibung | Transponiert eine Matrix. | - |
| [`einheitsmatrix`](hm2_all_in_one.py#L853) | `dimension, symbolisch=False` | Rueckgabewert gemaess Beschreibung | Erzeugt eine Einheitsmatrix. | - |
| [`diagonalmatrix`](hm2_all_in_one.py#L867) | `diagonalwerte, symbolisch=False` | Rueckgabewert gemaess Beschreibung | Erzeugt eine Diagonalmatrix. | - |
| [`ist_symmetrisch`](hm2_all_in_one.py#L879) | `matrix, toleranz=1e-10` | bool | Prueft, ob eine Matrix symmetrisch ist. | - |
| [`permutationsmatrix`](hm2_all_in_one.py#L893) | `dimension, zeile_i, zeile_j` | np.ndarray | Erzeugt eine Matrix zum Vertauschen zweier Zeilen. | - |
| [`determinante_1x1`](hm2_all_in_one.py#L907) | `matrix` | float | Berechnet die Determinante einer 1x1-Matrix. | - |
| [`determinante_2x2`](hm2_all_in_one.py#L922) | `matrix` | float | Berechnet die Determinante einer 2x2-Matrix. | - |
| [`determinante_3x3`](hm2_all_in_one.py#L937) | `matrix` | float | Berechnet die Determinante einer 3x3-Matrix nach Sarrus. | - |
| [`determinante`](hm2_all_in_one.py#L953) | `matrix, symbolisch=False` | Rueckgabewert gemaess Beschreibung | Berechnet die Determinante einer quadratischen Matrix. | - |
| [`inverse_2x2`](hm2_all_in_one.py#L968) | `matrix` | np.ndarray | Berechnet die inverse 2x2-Matrix. | - |
| [`matrix_inverse`](hm2_all_in_one.py#L985) | `matrix, symbolisch=False` | Rueckgabewert gemaess Beschreibung | Berechnet die inverse Matrix. | - |
| [`rang_matrix`](hm2_all_in_one.py#L1000) | `matrix, toleranz=1e-10` | int | Berechnet den numerischen Rang einer Matrix. | - |
| [`konditionszahl_matrix`](hm2_all_in_one.py#L1012) | `matrix, norm_ord=2` | float | Berechnet die Konditionszahl einer Matrix. | - |
| [`loese_lgs`](hm2_all_in_one.py#L1025) | `matrix_a, vektor_b, methode='numpy'` | np.ndarray | Loest ein lineares Gleichungssystem A x = b. | - |
| [`loese_lgs_2x2`](hm2_all_in_one.py#L1043) | `matrix_a, vektor_b` | np.ndarray | Loest ein 2x2-LGS. | - |
| [`loese_lgs_3x3`](hm2_all_in_one.py#L1059) | `matrix_a, vektor_b` | np.ndarray | Loest ein 3x3-LGS. | - |
| [`gauss_elimination`](hm2_all_in_one.py#L1075) | `matrix_a, vektor_b=None, pivotisierung=True` | dict | Fuehrt Gauss-Elimination aus. | - |
| [`zeilenstufenform`](hm2_all_in_one.py#L1109) | `matrix, symbolisch=False` | Rueckgabewert gemaess Beschreibung | Berechnet eine Zeilenstufenform. | - |
| [`reduzierte_zeilenstufenform`](hm2_all_in_one.py#L1123) | `matrix, symbolisch=True` | Rueckgabewert gemaess Beschreibung | Berechnet die reduzierte Zeilenstufenform. | - |
| [`lgs_loesbarkeit_pruefen`](hm2_all_in_one.py#L1138) | `matrix_a, vektor_b, toleranz=1e-10` | dict | Prueft Loesbarkeit eines LGS. | - |
| [`parameterdarstellung_lgs`](hm2_all_in_one.py#L1161) | `matrix_a, vektor_b` | dict | Berechnet eine symbolische Parameterdarstellung eines LGS. | - |
| [`homogenes_lgs_loesen`](hm2_all_in_one.py#L1176) | `matrix_a` | dict | Loest das homogene LGS A x = 0. | - |
| [`qr_zerlegung`](hm2_all_in_one.py#L1189) | `matrix_a, modus='complete'` | tuple[np.ndarray, np.ndarray] | Berechnet die QR-Zerlegung. | - |
| [`qr_zerlegung_reduziert`](hm2_all_in_one.py#L1204) | `matrix_a` | tuple[np.ndarray, np.ndarray] | Berechnet die reduzierte QR-Zerlegung. | - |
| [`householder_vektor`](hm2_all_in_one.py#L1216) | `vektor` | np.ndarray | Berechnet den Householder-Vektor. | - |
| [`householder_matrix`](hm2_all_in_one.py#L1236) | `vektor` | np.ndarray | Berechnet die Householder-Spiegelungsmatrix. | - |
| [`qr_zerlegung_householder`](hm2_all_in_one.py#L1249) | `matrix_a` | tuple[np.ndarray, np.ndarray] | Berechnet QR mit Householder-Spiegelungen. | - |
| [`loese_lgs_qr`](hm2_all_in_one.py#L1278) | `matrix_a, vektor_b` | np.ndarray | Loest A x=b ueber QR. | - |
| [`loese_ausgleich_qr`](hm2_all_in_one.py#L1294) | `matrix_a, vektor_y` | np.ndarray | Loest ein lineares Ausgleichsproblem per QR. | - |

### Nichtlineare Gleichungen
<a id="nichtlineare-gleichungenpy"></a>

| Definition | Inputs | Output | Verwendung | Extra Imports |
|---|---|---|---|---|
| [`funktionswert_vektor`](hm2_all_in_one.py#L1317) | `funktion, punkt` | np.ndarray | Einheitliche numerische Auswertung f(x). | - |
| [`jacobi_matrix_symbolisch`](hm2_all_in_one.py#L1337) | `funktionen, variablen` | sp.Matrix | Berechnet die symbolische Jacobi-Matrix. | - |
| [`jacobi_matrix_auswerten`](hm2_all_in_one.py#L1354) | `jacobi_matrix, variablen, punkt` | np.ndarray | Wertet eine symbolische Jacobi-Matrix numerisch aus. | - |
| [`jacobi_matrix_numerisch`](hm2_all_in_one.py#L1372) | `funktion, punkt, schrittweite=1e-06, methode='zentral'` | np.ndarray | Berechnet eine numerische Jacobi-Matrix. | - |
| [`linearisierung`](hm2_all_in_one.py#L1409) | `funktion, jacobi, entwicklungs_punkt` | callable fuer L(x). | Erzeugt die Linearisierung einer vektorwertigen Funktion. | - |
| [`tangentialebene_symbolisch`](hm2_all_in_one.py#L1437) | `funktion, variablen, punkt` | sp.Expr | Berechnet die symbolische Tangentialebene. | - |
| [`tangentialebene`](hm2_all_in_one.py#L1460) | `funktion, gradient, punkt` | callable fuer die Ebene. | Erzeugt eine Tangentialebene fuer z=f(x,y). | - |
| [`_newton_resultat`](hm2_all_in_one.py#L1488) | `loesung, iterationen, konvergiert, residualnorm, schrittnorm, verlauf, nachricht, rueckgabe_verlauf` | Rueckgabewert gemaess Beschreibung | Baut ein IterationsErgebnis und blendet den Verlauf optional aus. | - |
| [`newton_verfahren`](hm2_all_in_one.py#L1501) | `funktion, jacobi, startwert, toleranz=1e-10, maximale_iterationen=50, rueckgabe_verlauf=False` | IterationsErgebnis | Loest ein nichtlineares Gleichungssystem mit Newton. | - |
| [`vereinfachtes_newton_verfahren`](hm2_all_in_one.py#L1540) | `funktion, jacobi, startwert, toleranz=1e-10, maximale_iterationen=50, rueckgabe_verlauf=False` | IterationsErgebnis | Loest ein NGS mit vereinfachtem Newton-Verfahren. | - |
| [`gedaempftes_newton_verfahren`](hm2_all_in_one.py#L1575) | `funktion, jacobi, startwert, toleranz=1e-10, maximale_iterationen=50, p_max=20, rueckgabe_verlauf=False` | IterationsErgebnis | Loest ein NGS mit gedaempftem Newton-Verfahren. | - |
| [`newton_verfahren_2d`](hm2_all_in_one.py#L1619) | `funktion_1, funktion_2, jacobi_eintraege, startwert, **optionen` | IterationsErgebnis | Komfort-Wrapper fuer Newton in zwei Variablen. | - |
| [`newton_verfahren_skalar`](hm2_all_in_one.py#L1656) | `funktion, ableitung, startwert, toleranz=1e-10, maximale_iterationen=50` | IterationsErgebnis | Loest eine skalare Gleichung mit Newton. | - |
| [`nullstellenfehler_abschaetzen`](hm2_all_in_one.py#L1689) | `funktion, naeherung, epsilon` | dict | Schaetzt einen Nullstellenfehler ueber Vorzeichenwechsel ab. | - |
| [`vorzeichenwechsel_pruefen`](hm2_all_in_one.py#L1710) | `funktion, links, rechts` | bool | Prueft einen Vorzeichenwechsel auf einem Intervall. | - |

### Ausgleichsrechnung
<a id="ausgleichsrechnungpy"></a>

| Definition | Inputs | Output | Verwendung | Extra Imports |
|---|---|---|---|---|
| [`designmatrix_erstellen`](hm2_all_in_one.py#L1733) | `stuetzstellen, basisfunktionen` | np.ndarray | Erstellt die Designmatrix der linearen Ausgleichsrechnung. | - |
| [`residuen_berechnen`](hm2_all_in_one.py#L1752) | `matrix_a, koeffizienten, vektor_y` | np.ndarray | Berechnet Residuen y-A lambda. | - |
| [`fehlerfunktional_berechnen`](hm2_all_in_one.py#L1764) | `residuen` | float | Berechnet das Fehlerfunktional. | - |
| [`r_quadrat_berechnen`](hm2_all_in_one.py#L1777) | `stuetzwerte, prognosewerte` | float | Berechnet das Bestimmtheitsmass R^2. | - |
| [`residuentabelle_erstellen`](hm2_all_in_one.py#L1795) | `stuetzstellen, stuetzwerte, prognosewerte` | pd.DataFrame | Erstellt eine Residuentabelle. | import pandas as pd |
| [`_ausgleichs_ergebnis`](hm2_all_in_one.py#L1813) | `matrix_a, y, koeffizienten, methode` | Rueckgabewert gemaess Beschreibung | Erstellt ein AusgleichsErgebnis aus Designmatrix, Daten und Koeffizienten. | import pandas as pd |
| [`lineare_ausgleichsrechnung_normalgleichungen`](hm2_all_in_one.py#L1831) | `stuetzstellen, stuetzwerte, basisfunktionen` | AusgleichsErgebnis | Loest lineare Ausgleichsrechnung mit Normalgleichungen. | - |
| [`lineare_ausgleichsrechnung_qr`](hm2_all_in_one.py#L1855) | `stuetzstellen, stuetzwerte, basisfunktionen` | AusgleichsErgebnis | Loest lineare Ausgleichsrechnung per QR. | - |
| [`polynom_ausgleich`](hm2_all_in_one.py#L1877) | `stuetzstellen, stuetzwerte, grad, methode='qr'` | AusgleichsErgebnis | Passt ein Polynom im kleinsten-Quadrate-Sinn an. | - |
| [`ausgleichsgerade`](hm2_all_in_one.py#L1901) | `stuetzstellen, stuetzwerte, methode='qr'` | AusgleichsErgebnis | Berechnet eine Ausgleichsgerade. | - |
| [`ausgleichsparabel`](hm2_all_in_one.py#L1916) | `stuetzstellen, stuetzwerte, methode='qr'` | AusgleichsErgebnis | Berechnet eine Ausgleichsparabel. | - |
| [`ausgleichsfunktion_aus_koeffizienten`](hm2_all_in_one.py#L1931) | `koeffizienten, basisfunktionen` | Rueckgabewert gemaess Beschreibung | Erzeugt die Ausgleichsfunktion aus Koeffizienten. | - |
| [`residuenfunktion_erstellen`](hm2_all_in_one.py#L1957) | `modellfunktion, x_daten, y_daten` | Rueckgabewert gemaess Beschreibung | Erzeugt die Residuenfunktion fuer nichtlineare Ausgleichsrechnung. | - |
| [`jacobi_residuen_numerisch`](hm2_all_in_one.py#L1982) | `modellfunktion, x_daten, parameter, schrittweite=1e-06` | np.ndarray | Berechnet die numerische Jacobi-Matrix der Residuen. | - |
| [`gauss_newton_verfahren`](hm2_all_in_one.py#L2008) | `modellfunktion, jacobi_residuen, x_daten, y_daten, startparameter, toleranz=1e-10, maximale_iterationen=50, rueckgabe_verlauf=False` | IterationsErgebnis | Loest nichtlineare Ausgleichsrechnung mit Gauss-Newton. | - |
| [`gedaempftes_gauss_newton_verfahren`](hm2_all_in_one.py#L2043) | `modellfunktion, jacobi_residuen, x_daten, y_daten, startparameter, toleranz=1e-10, maximale_iterationen=50, p_max=20, rueckgabe_verlauf=False` | IterationsErgebnis | Loest nichtlineare Ausgleichsrechnung mit gedaempftem Gauss-Newton. | - |
| [`exponentialmodell`](hm2_all_in_one.py#L2088) | `x, parameter` | Rueckgabewert gemaess Beschreibung | Wertet das Exponentialmodell aus. | - |
| [`logistisches_modell`](hm2_all_in_one.py#L2101) | `x, parameter` | Rueckgabewert gemaess Beschreibung | Wertet das logistische Modell aus. | - |
| [`logistische_linearisierung`](hm2_all_in_one.py#L2114) | `stuetzstellen, stuetzwerte, kapazitaet_k` | AusgleichsErgebnis | Linearisiert ein logistisches Wachstum bei bekannter Kapazitaet. | - |

### Interpolation
<a id="interpolationpy"></a>

| Definition | Inputs | Output | Verwendung | Extra Imports |
|---|---|---|---|---|
| [`lagrange_polynom_symbolisch`](hm2_all_in_one.py#L2142) | `stuetzstellen, stuetzwerte, variable=None` | sp.Expr | Berechnet das Lagrange-Polynom symbolisch. | - |
| [`lagrange_basiswert`](hm2_all_in_one.py#L2171) | `stuetzstellen, index, x_wert` | float | Berechnet einen Lagrange-Basiswert. | - |
| [`lagrange_interpolation`](hm2_all_in_one.py#L2193) | `stuetzstellen, stuetzwerte, x_wert` | float | Wertet das Lagrange-Interpolationspolynom aus. | - |
| [`interpolationsfehler_abschaetzen`](hm2_all_in_one.py#L2212) | `stuetzstellen, x_wert, max_ableitung_n_plus_1` | float | Schaetzt den Interpolationsfehler ab. | import math |
| [`natuerlicher_kubischer_spline_koeffizienten`](hm2_all_in_one.py#L2231) | `stuetzstellen, stuetzwerte` | SplineKoeffizienten | Berechnet Koeffizienten eines natuerlichen kubischen Splines. | - |
| [`spline_abschnitt_finden`](hm2_all_in_one.py#L2274) | `stuetzstellen, x_wert` | int | Findet den Spline-Abschnitt fuer einen x-Wert. | - |
| [`natuerlicher_kubischer_spline_auswerten`](hm2_all_in_one.py#L2293) | `koeffizienten, x_wert` | float | Wertet einen natuerlichen kubischen Spline aus. | - |
| [`natuerlicher_kubischer_spline_ableitung_auswerten`](hm2_all_in_one.py#L2310) | `koeffizienten, x_wert, ordnung=1` | float | Wertet die erste oder zweite Ableitung eines natuerlichen kubischen Splines aus. | - |
| [`spline_abschnitte_als_text`](hm2_all_in_one.py#L2337) | `koeffizienten` | list[str] | Gibt Spline-Abschnitte als Textformeln aus. | - |
| [`spline_tabelle_erstellen`](hm2_all_in_one.py#L2356) | `koeffizienten` | pd.DataFrame | Erstellt eine Tabelle der Spline-Koeffizienten. | import pandas as pd |

### Integration
<a id="integrationpy"></a>

| Definition | Inputs | Output | Verwendung | Extra Imports |
|---|---|---|---|---|
| [`rechteckregel`](hm2_all_in_one.py#L2380) | `funktion, untere_grenze, obere_grenze` | float | Berechnet die einfache Mittelpunkt-Rechteckregel. | - |
| [`trapezregel`](hm2_all_in_one.py#L2398) | `funktion, untere_grenze, obere_grenze` | float | Berechnet die einfache Trapezregel. | - |
| [`simpsonregel`](hm2_all_in_one.py#L2416) | `funktion, untere_grenze, obere_grenze` | float | Berechnet die einfache Simpson-Regel. | - |
| [`summierte_rechteckregel`](hm2_all_in_one.py#L2435) | `funktion, untere_grenze, obere_grenze, anzahl_intervalle` | QuadraturErgebnis | Berechnet die summierte Mittelpunkt-Rechteckregel. | - |
| [`summierte_trapezregel`](hm2_all_in_one.py#L2459) | `funktion, untere_grenze, obere_grenze, anzahl_intervalle` | QuadraturErgebnis | Berechnet die summierte Trapezregel. | - |
| [`trapezregel_nicht_aequidistant`](hm2_all_in_one.py#L2484) | `stuetzstellen, stuetzwerte` | float | Integriert tabellierte nichtaequidistante Werte mit Trapezen. | - |
| [`summierte_simpsonregel`](hm2_all_in_one.py#L2503) | `funktion, untere_grenze, obere_grenze, anzahl_intervalle` | QuadraturErgebnis | Berechnet die summierte Simpson-Regel. | - |
| [`simpson_als_mittel`](hm2_all_in_one.py#L2529) | `trapezwert, rechteckwert` | float | Berechnet Simpson als gewichtetes Mittel aus Trapez und Rechteck. | - |
| [`integriere_tabelle_nicht_aequidistant`](hm2_all_in_one.py#L2541) | `stuetzstellen, werte` | QuadraturErgebnis | Integriert eine nichtaequidistante Tabelle. | - |
| [`funktion_aus_stuetzstellen_linear`](hm2_all_in_one.py#L2557) | `stuetzstellen, stuetzwerte` | Funktion f(x), die skalare x-Werte linear interpoliert auswertet. | Quadraturfunktionen wie `summierte_trapezregel` oder `romberg_extrapolation` erwarten eine Funktion. Mit diesem Wrapper koennen tabellierte Werte trotzdem als Funktionsinput genutzt werden. | - |
| [`fehlergrenze_rechteckregel`](hm2_all_in_one.py#L2588) | `schrittweite, untere_grenze, obere_grenze, max_zweite_ableitung` | float | Berechnet die Fehlergrenze der summierten Rechteckregel. | - |
| [`fehlergrenze_trapezregel`](hm2_all_in_one.py#L2600) | `schrittweite, untere_grenze, obere_grenze, max_zweite_ableitung` | float | Berechnet die Fehlergrenze der summierten Trapezregel. | - |
| [`fehlergrenze_simpsonregel`](hm2_all_in_one.py#L2612) | `schrittweite, untere_grenze, obere_grenze, max_vierte_ableitung` | float | Berechnet die Fehlergrenze der summierten Simpson-Regel. | - |
| [`schrittweite_fuer_trapezregel`](hm2_all_in_one.py#L2624) | `epsilon, untere_grenze, obere_grenze, max_zweite_ableitung` | float | Waehlt eine Schrittweite fuer die Trapezregel. | import math |
| [`schrittweite_fuer_simpsonregel`](hm2_all_in_one.py#L2637) | `epsilon, untere_grenze, obere_grenze, max_vierte_ableitung` | float | Waehlt eine Schrittweite fuer die Simpson-Regel. | - |
| [`anzahl_intervalle_aus_schrittweite`](hm2_all_in_one.py#L2649) | `untere_grenze, obere_grenze, schrittweite, gerade_erforderlich=False` | int | Berechnet eine Intervallanzahl aus einer maximalen Schrittweite. | import math |
| [`romberg_tabelle_erstellen`](hm2_all_in_one.py#L2665) | `trapezwerte` | pd.DataFrame | Erstellt eine Romberg-Tabelle aus Trapezwerten. | import pandas as pd |
| [`romberg_extrapolation`](hm2_all_in_one.py#L2686) | `funktion, untere_grenze, obere_grenze, ordnung_m` | tuple[float, pd.DataFrame] | Berechnet Romberg-Extrapolation. | import pandas as pd |
| [`romberg_extrapolation_aus_stuetzstellen`](hm2_all_in_one.py#L2710) | `stuetzstellen, stuetzwerte, ordnung_m` | tuple[float, pd.DataFrame] | Praktische Variante, wenn in der Pruefung nur Messwerte oder Tabellenwerte gegeben sind, aber die Romberg-Funktion eine auswertbare Funktion erwartet. | import pandas as pd |
| [`_gauss_transformiere`](hm2_all_in_one.py#L2739) | `knoten, gewichte, funktion, a, b` | Rueckgabewert gemaess Beschreibung | Transformiert Gauss-Legendre-Knoten von [-1,1] auf [a,b] und summiert. | - |
| [`gauss_legendre_1`](hm2_all_in_one.py#L2747) | `funktion, untere_grenze, obere_grenze` | QuadraturErgebnis | Berechnet Gauss-Legendre mit einem Stuetzpunkt. | - |
| [`gauss_legendre_2`](hm2_all_in_one.py#L2766) | `funktion, untere_grenze, obere_grenze` | QuadraturErgebnis | Berechnet Gauss-Legendre mit zwei Stuetzpunkten. | import math |
| [`gauss_legendre_3`](hm2_all_in_one.py#L2787) | `funktion, untere_grenze, obere_grenze` | QuadraturErgebnis | Berechnet Gauss-Legendre mit drei Stuetzpunkten. | import math |
| [`gauss_legendre_allgemein`](hm2_all_in_one.py#L2811) | `funktion, untere_grenze, obere_grenze, anzahl_stuetzstellen` | QuadraturErgebnis | Berechnet allgemeine Gauss-Legendre-Quadratur. | - |

### Differentialgleichungen und Plotting
<a id="differentialgleichungenpy"></a>

| Definition | Inputs | Output | Verwendung | Extra Imports |
|---|---|---|---|---|
| [`plot_basis`](hm2_all_in_one.py#L2843) | `x_label='x', y_label='y', titel='', grid=True` | Matplotlib-Achse `ax`. | Minimaler Startpunkt fuer Pruefungsplots, ohne komplexe Wrapper-Logik. | - |
| [`nullniveau_zwei_funktionen_plotten`](hm2_all_in_one.py#L2864) | `f1, f2, x_bereich=(0, 1), y_bereich=(0, 1), anzahl=100, labels=('f1', 'f2'), farben=('r', 'b'), titel='Grafische Loesung von f1=0 und f2=0'` | Matplotlib-Achse `ax`. | Direkte grafische Loesung eines nichtlinearen Gleichungssystems mit zwei Gleichungen, nahe am Standard-Pruefungscode mit `meshgrid` und `plt.contour`. | - |
| [`dgl_ergebnis_plotten`](hm2_all_in_one.py#L2897) | `ergebnis, titel=None, label=None, marker='o'` | Matplotlib-Achse `ax`. | Ergebnisobjekte von `euler_verfahren`, `heun_verfahren`, `runge_kutta_4` oder den Systemvarianten schnell sichtbar machen. | - |
| [`_dgl_ergebnis`](hm2_all_in_one.py#L2933) | `x_werte, y_werte, methode, schrittweite, ordnung` | Rueckgabewert gemaess Beschreibung | Erstellt ein DglErgebnis inklusive passender Pandas-Tabelle. | import pandas as pd |
| [`euler_verfahren`](hm2_all_in_one.py#L2948) | `funktion, x_start, y_start, x_ende, anzahl_schritte` | DglErgebnis | Loest ein AWP mit dem expliziten Euler-Verfahren. | - |
| [`mittelpunkt_verfahren`](hm2_all_in_one.py#L2977) | `funktion, x_start, y_start, x_ende, anzahl_schritte` | DglErgebnis | Loest ein AWP mit dem Mittelpunktverfahren. | - |
| [`heun_verfahren`](hm2_all_in_one.py#L3009) | `funktion, x_start, y_start, x_ende, anzahl_schritte` | DglErgebnis | Loest ein AWP mit dem Heun-Verfahren. | - |
| [`runge_kutta_4`](hm2_all_in_one.py#L3041) | `funktion, x_start, y_start, x_ende, anzahl_schritte` | DglErgebnis | Loest ein AWP mit klassischem Runge-Kutta 4. | import math |
| [`explizites_runge_kutta_verfahren`](hm2_all_in_one.py#L3077) | `funktion, x_start, y_start, x_ende, anzahl_schritte, butcher_a, butcher_b, butcher_c, methode_name='explizites Runge-Kutta'` | DglErgebnis | Loest ein AWP mit einem frei vorgegebenen expliziten Runge-Kutta-Schema. | - |
| [`euler_verfahren_system`](hm2_all_in_one.py#L3138) | `funktion, x_start, y_startvektor, x_ende, anzahl_schritte` | DglErgebnis | Loest ein DGL-System mit Euler. | - |
| [`mittelpunkt_verfahren_system`](hm2_all_in_one.py#L3160) | `funktion, x_start, y_startvektor, x_ende, anzahl_schritte` | DglErgebnis | Loest ein DGL-System mit Mittelpunktverfahren. | - |
| [`heun_verfahren_system`](hm2_all_in_one.py#L3183) | `funktion, x_start, y_startvektor, x_ende, anzahl_schritte` | DglErgebnis | Loest ein DGL-System mit Heun. | - |
| [`runge_kutta_4_system`](hm2_all_in_one.py#L3206) | `funktion, x_start, y_startvektor, x_ende, anzahl_schritte` | DglErgebnis | Loest ein DGL-System mit RK4. | - |
| [`_system_verfahren`](hm2_all_in_one.py#L3229) | `funktion, x_start, y_startvektor, x_ende, anzahl_schritte, methode, ordnung` | Rueckgabewert gemaess Beschreibung | Gemeinsamer Integrationskern fuer Euler, Mittelpunkt, Heun und RK4 bei Systemen. | - |
| [`dgl_hoeherer_ordnung_zu_system`](hm2_all_in_one.py#L3258) | `funktion_hoechste_ableitung, ordnung` | callable | Formt eine DGL hoeherer Ordnung in ein System erster Ordnung um. | - |
| [`stabilitaetsfunktion_euler`](hm2_all_in_one.py#L3281) | `z` | float | Berechnet die Euler-Stabilitaetsfunktion. | - |
| [`ist_euler_stabil`](hm2_all_in_one.py#L3293) | `alpha, schrittweite` | bool | Prueft Euler-Stabilitaet fuer y'=-alpha y. | - |
| [`stabilitaetsintervall_euler`](hm2_all_in_one.py#L3305) | `alpha` | tuple[float, float] | Gibt das stabile h-Intervall fuer Euler an. | - |
| [`globalen_fehler_berechnen`](hm2_all_in_one.py#L3317) | `exakte_loesung, x_werte, y_werte` | np.ndarray | Berechnet globale Fehler an Gitterpunkten. | - |
| [`konvergenzordnung_schaetzen`](hm2_all_in_one.py#L3337) | `fehler_grob, fehler_fein, faktor=2` | float | Schaetzt die Konvergenzordnung. | import math |
| [`boeing_brems_system`](hm2_all_in_one.py#L3350) | `masse=97000, konstante_bremskraft=570000, luftterm=5` | Rueckgabewert gemaess Beschreibung | Erzeugt ein Bremsmodell fuer eine Boeing. | - |
| [`raketen_beschleunigung`](hm2_all_in_one.py#L3368) | `t, v_rel=2600, startmasse=300000, endmasse=80000, brenndauer=190, g=9.81` | Rueckgabewert gemaess Beschreibung | Berechnet eine einfache Raketenbeschleunigung. | - |
| [`raketen_system`](hm2_all_in_one.py#L3385) | `v_rel=2600, startmasse=300000, endmasse=80000, brenndauer=190, g=9.81` | Rueckgabewert gemaess Beschreibung | Erzeugt ein Raketenmodell fuer Hoehe und Geschwindigkeit. | - |
| [`lotka_volterra_system`](hm2_all_in_one.py#L3401) | `t, zustand, a=1.0, b=0.5, c=0.75, d=0.25` | Rueckgabewert gemaess Beschreibung | Wertet das Lotka-Volterra-System aus. | - |
| [`logistisches_wachstum_mit_stoerung`](hm2_all_in_one.py#L3414) | `t, umsatz, stoer_amplitude=20` | Rueckgabewert gemaess Beschreibung | Berechnet logistisches Wachstum mit sinusfoermiger Stoerung. | import math |

### Fehleranalyse
<a id="fehleranalysepy"></a>

| Definition | Inputs | Output | Verwendung | Extra Imports |
|---|---|---|---|---|
| [`absoluter_fehler`](hm2_all_in_one.py#L3433) | `naeherung, exakt` | float | Berechnet den absoluten Fehler. | - |
| [`relativer_fehler`](hm2_all_in_one.py#L3445) | `naeherung, exakt` | float | Berechnet den relativen Fehler. | - |
| [`konditionszahl_funktion`](hm2_all_in_one.py#L3459) | `funktion, ableitung, x_wert` | float | Berechnet die Konditionszahl einer skalaren Funktion. | - |
| [`absoluter_fehler_fortpflanzung`](hm2_all_in_one.py#L3479) | `ableitung, x_wert, eingabefehler` | float | Schaetzt absolute Fehlerfortpflanzung. | - |
| [`relativer_fehler_fortpflanzung`](hm2_all_in_one.py#L3493) | `funktion, ableitung, x_wert, relativer_eingabefehler` | float | Schaetzt relative Fehlerfortpflanzung. | - |
| [`lgs_absoluter_fehler_abschaetzung`](hm2_all_in_one.py#L3511) | `matrix_a, stoerung_b, norm_ord=2` | float | Schaetzt absoluten LGS-Fehler bei Stoerung in b. | - |
| [`lgs_relativer_fehler_abschaetzung`](hm2_all_in_one.py#L3525) | `matrix_a, vektor_b, stoerung_b, norm_ord=2` | float | Schaetzt relativen LGS-Fehler. | - |

### Analysis-Hilfen
<a id="analysis-hilfenpy"></a>

| Definition | Inputs | Output | Verwendung | Extra Imports |
|---|---|---|---|---|
| [`ableitung_symbolisch`](hm2_all_in_one.py#L3546) | `funktion, variable, ordnung=1` | sp.Expr | Berechnet eine symbolische Ableitung. | - |
| [`partielle_ableitung_symbolisch`](hm2_all_in_one.py#L3560) | `funktion, variable, ordnung=1` | sp.Expr | Berechnet eine partielle Ableitung. | - |
| [`gradient_symbolisch`](hm2_all_in_one.py#L3574) | `funktion, variablen` | sp.Matrix | Berechnet den symbolischen Gradienten. | - |
| [`hessenmatrix_symbolisch`](hm2_all_in_one.py#L3588) | `funktion, variablen` | sp.Matrix | Berechnet die Hesse-Matrix. | - |
| [`stammfunktion_symbolisch`](hm2_all_in_one.py#L3602) | `funktion, variable` | sp.Expr | Berechnet eine Stammfunktion. | - |
| [`bestimmtes_integral_symbolisch`](hm2_all_in_one.py#L3616) | `funktion, variable, untere_grenze, obere_grenze` | sp.Expr | Berechnet ein bestimmtes Integral symbolisch. | - |
| [`uneigentliches_integral_symbolisch`](hm2_all_in_one.py#L3632) | `funktion, variable, untere_grenze, obere_grenze` | sp.Expr | Berechnet ein uneigentliches Integral symbolisch. | - |
| [`grenzwert_symbolisch`](hm2_all_in_one.py#L3647) | `funktion, variable, punkt, richtung='+-'` | sp.Expr | Berechnet einen symbolischen Grenzwert. | - |
| [`polynomdivision_symbolisch`](hm2_all_in_one.py#L3661) | `polynom, divisor, variable` | tuple[sp.Expr, sp.Expr] | Fuehrt symbolische Polynomdivision durch. | - |
| [`mitternachtsformel`](hm2_all_in_one.py#L3676) | `a, b, c` | tuple[complex, complex] | Loest eine quadratische Gleichung. | import cmath |
| [`funktion_verkettung`](hm2_all_in_one.py#L3691) | `funktion_f, funktion_g` | Rueckgabewert gemaess Beschreibung | Erzeugt die Verkettung g o f. | - |
| [`ist_gerade_funktion_symbolisch`](hm2_all_in_one.py#L3712) | `funktion, variable` | bool | Prueft symbolisch gerade Symmetrie. | - |
| [`ist_ungerade_funktion_symbolisch`](hm2_all_in_one.py#L3726) | `funktion, variable` | bool | Prueft symbolisch ungerade Symmetrie. | - |
| [`trigonometrische_nullstellen_sinus`](hm2_all_in_one.py#L3740) | `k_min, k_max` | list | Gibt Sinus-Nullstellen an. | - |
| [`trigonometrische_nullstellen_cosinus`](hm2_all_in_one.py#L3752) | `k_min, k_max` | list | Gibt Cosinus-Nullstellen an. | - |
| [`logarithmus_basis`](hm2_all_in_one.py#L3764) | `wert, basis` | float | Berechnet einen Logarithmus zu beliebiger Basis. | import math |
| [`_integriere_numerisch`](hm2_all_in_one.py#L3777) | `funktion, a, b, n=2000` | Rueckgabewert gemaess Beschreibung | Integriert eine skalare Funktion intern mit der summierten Trapezregel. | - |
| [`funktionsmittelwert`](hm2_all_in_one.py#L3784) | `funktion, untere_grenze, obere_grenze, numerisch=True` | float | Berechnet den Funktionsmittelwert. | - |
| [`flaeche_zwischen_kurven`](hm2_all_in_one.py#L3802) | `funktion_f, funktion_g, schnittpunkte` | float | Berechnet die Flaeche zwischen zwei Kurven. | - |
| [`rotationsvolumen_um_x_achse`](hm2_all_in_one.py#L3826) | `funktion, untere_grenze, obere_grenze` | float | Berechnet ein Rotationsvolumen um die x-Achse. | import math |
| [`bogenlaenge`](hm2_all_in_one.py#L3848) | `funktion, ableitung, untere_grenze, obere_grenze, numerisch=True` | float | Berechnet die Bogenlaenge eines Graphen. | import math |
| [`mantelflaeche_rotation`](hm2_all_in_one.py#L3872) | `funktion, ableitung, untere_grenze, obere_grenze, numerisch=True` | float | Berechnet die Mantelflaeche eines Rotationskoerpers. | import math |

### Daten und Tabellen
<a id="daten-tabellenpy"></a>

| Definition | Inputs | Output | Verwendung | Extra Imports |
|---|---|---|---|---|
| [`datenrahmen_aus_messwerten`](hm2_all_in_one.py#L3903) | `x_werte, y_werte, x_name='x', y_name='y'` | pd.DataFrame | Erstellt einen DataFrame aus Messwerten. | import pandas as pd |
| [`residuentabelle`](hm2_all_in_one.py#L3920) | `stuetzstellen, messwerte, modellwerte` | pd.DataFrame | Erstellt eine Residuentabelle. | import pandas as pd |
| [`iterationstabelle`](hm2_all_in_one.py#L3938) | `verlauf` | pd.DataFrame | Erstellt eine Tabelle aus Iterationsverlauf. | import pandas as pd |
| [`quadraturvergleich_tabelle`](hm2_all_in_one.py#L3951) | `ergebnisse` | pd.DataFrame | Vergleicht mehrere Quadraturergebnisse. | import pandas as pd |
| [`dgl_loesungstabelle`](hm2_all_in_one.py#L3972) | `x_werte, y_werte, spaltennamen=None` | pd.DataFrame | Erstellt eine Tabelle fuer DGL-Loesungen. | import pandas as pd |
| [`exportiere_tabelle_csv`](hm2_all_in_one.py#L3993) | `tabelle, dateiname` | None | Exportiert eine Tabelle als CSV. | import pandas as pd |

### Physik-Beispiele
<a id="physik-beispielepy"></a>

| Definition | Inputs | Output | Verwendung | Extra Imports |
|---|---|---|---|---|
| [`fall_mit_luftwiderstand_geschwindigkeit`](hm2_all_in_one.py#L4013) | `t, masse=80, g=9.81, k=0.25` | float | Berechnet Fallgeschwindigkeit mit quadratischem Luftwiderstand. | import math |
| [`strecke_fall_mit_luftwiderstand_simpson`](hm2_all_in_one.py#L4026) | `zeit_ende=10, intervalle=10` | QuadraturErgebnis | Berechnet Fallstrecke durch Simpson-Integration der Geschwindigkeit. | - |
| [`erdmasse_aus_dichtetabelle`](hm2_all_in_one.py#L4038) | `radius_km, dichte_kg_pro_m3` | QuadraturErgebnis | Schaetzt die Erdmasse aus radialer Dichtetabelle. | import math |
| [`boeing_landung_beispiel`](hm2_all_in_one.py#L4054) | `schrittweite=0.1, endzeit=20` | DglErgebnis | Berechnet ein Boeing-Bremsbeispiel. | - |
| [`rakete_beispiel`](hm2_all_in_one.py#L4067) | `endzeit=190, schritte=1900` | DglErgebnis | Berechnet ein Raketenbeispiel. | - |
| [`raeuber_beute_beispiel`](hm2_all_in_one.py#L4079) | `endzeit=15, schritte=150` | DglErgebnis | Berechnet ein Raeuber-Beute-Beispiel. | - |
| [`motorleistung_polynomfit_beispiel`](hm2_all_in_one.py#L4091) | `-` | AusgleichsErgebnis | Fuehrt einen Polynomfit fuer Motorleistungsdaten aus. | - |
| [`exponentialfit_gauss_newton_beispiel`](hm2_all_in_one.py#L4105) | `-` | IterationsErgebnis | Fuehrt einen Exponentialfit mit Gauss-Newton aus. | - |

