# SCRIPTME.md

README-Übersicht für vorbereitete Python-Skripte zur Prüfung in Höhere Mathematik 2.

Diese Übersicht geht davon aus, dass alle Skripte bereits in der unten angegebenen Ordnerstruktur existieren. Die Dateinamen sind als relative Links gesetzt, damit du direkt zum entsprechenden Skript springen kannst.

## Standard-Imports

Alle Skripte dürfen davon ausgehen, dass folgende Imports verfügbar sind:

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
```

## Inhaltsverzeichnis

1. [Nichtlineare Gleichungssysteme und Newton-Verfahren](#1-nichtlineare-gleichungssysteme-und-newton-verfahren)  
   Ordner: [`1_NGL_und_Newton/`](1_NGL_und_Newton/)
2. [Lineare Ausgleichsrechnung](#2-lineare-ausgleichsrechnung)  
   Ordner: [`2_Lineare_Ausgleichsrechnung/`](2_Lineare_Ausgleichsrechnung/)
3. [Nichtlineare Ausgleichsrechnung und Gauss-Newton](#3-nichtlineare-ausgleichsrechnung-und-gauss-newton)  
   Ordner: [`3_Nichtlineare_Ausgleichsrechnung_Gauss_Newton/`](3_Nichtlineare_Ausgleichsrechnung_Gauss_Newton/)
4. [Interpolation](#4-interpolation)  
   Ordner: [`4_Interpolation/`](4_Interpolation/)
5. [Numerische Integration und Romberg](#5-numerische-integration-und-romberg)  
   Ordner: [`5_Numerische_Integration_Romberg/`](5_Numerische_Integration_Romberg/)
6. [Anfangswertprobleme, Euler, Heun und Runge-Kutta](#6-anfangswertprobleme-euler-heun-und-runge-kutta)  
   Ordner: [`6_DGL_Euler_Heun_Runge_Kutta/`](6_DGL_Euler_Heun_Runge_Kutta/)
7. [Priorisierte Prüfungsliste](#7-priorisierte-pruefungsliste)

---

## 1. Nichtlineare Gleichungssysteme und Newton-Verfahren

**Ordner:** [`1_NGL_und_Newton/`](1_NGL_und_Newton/)

| Skript                                                                                                  | Beschreibung                                                                                                                                                                | Plot                                                                                 |
| ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| [`newton_1d_ungedaempft.py`](1_NGL_und_Newton/newton_1d_ungedaempft.py)                                 | Löst eine skalare nichtlineare Gleichung `f(x)=0` mit dem klassischen Newton-Verfahren. Ableitung wird mit `sympy` berechnet, Iterationstabelle wird ausgegeben.            | Plot von `f(x)`, x-Achse, gefundener Nullstelle und Newton-Iterierten.               |
| [`newton_1d_gedaempft.py`](1_NGL_und_Newton/newton_1d_gedaempft.py)                                     | Löst eine skalare nichtlineare Gleichung mit gedämpftem Newton-Verfahren. Schrittweite wird über `delta / 2**p` reduziert, bis der Residuenbetrag kleiner wird.             | Plot von `f(x)` mit Iterationspunkten; zusätzlich Vergleich ungedämpft vs. gedämpft. |
| [`newton_system_ungedaempft.py`](1_NGL_und_Newton/newton_system_ungedaempft.py)                         | Löst ein nichtlineares Gleichungssystem `F(x)=0` mit Jacobi-Matrix. `J(x)` wird mit `sympy.Matrix(...).jacobian(...)` erzeugt, danach wird `J delta = -F` numerisch gelöst. | Für 2D-Systeme: Konturlinien `f1(x,y)=0` und `f2(x,y)=0` plus Iterationspfad.        |
| [`newton_system_gedaempft.py`](1_NGL_und_Newton/newton_system_gedaempft.py)                             | Gedämpftes Newton-Verfahren für nichtlineare Gleichungssysteme. Pro Iteration wird das kleinste `p` gesucht, sodass `norm(F(x + delta/2**p)) < norm(F(x))`.                 | Konturlinien der Gleichungen und gedämpfter Iterationspfad; zusätzlich Plot `norm(F(x_k))` gegen Iteration. |
| [`newton_system_vereinfacht.py`](1_NGL_und_Newton/newton_system_vereinfacht.py)                         | Vereinfachtes Newton-Verfahren: Jacobi-Matrix wird nur einmal bei `x0` berechnet und dann in allen Iterationen wiederverwendet.                                             | Plot `norm(F(x_k))` gegen Iteration und Vergleich mit normalem Newton-Verfahren. |
| [`linearisierung_jacobi_tangentialebene.py`](1_NGL_und_Newton/linearisierung_jacobi_tangentialebene.py) | Berechnet Jacobi-Matrix, Funktionswert am Entwicklungspunkt und Linearisierung `g(x)=f(x0)+Df(x0)(x-x0)`. Für skalarwertige Funktionen auch Tangentialebene.                | 3D-Plot der Fläche `z=f(x,y)` und der Tangentialebene am Entwicklungspunkt.          |

---

## 2. Lineare Ausgleichsrechnung

**Ordner:** [`2_Lineare_Ausgleichsrechnung/`](2_Lineare_Ausgleichsrechnung/)

| Skript                                                                                                          | Beschreibung                                                                                                                                                  | Plot                                                                                                            |
| --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [`ausgleich_normalgleichung.py`](2_Lineare_Ausgleichsrechnung/ausgleich_normalgleichung.py)                     | Löst ein lineares Ausgleichsproblem mit Normalgleichung `A.T @ A @ lambda = A.T @ y`. Gibt Parameter, Residuen und Fehlerfunktional `E = norm(y-A lambda)^2` aus. | Datenpunkte und gefittete Ausgleichsfunktion; zusätzlich Residuenplot. |
| [`ausgleich_qr.py`](2_Lineare_Ausgleichsrechnung/ausgleich_qr.py)                                               | Löst dasselbe lineare Ausgleichsproblem numerisch stabiler über QR-Zerlegung `A=QR` und `R lambda = Q.T y`. Vergleicht mit Normalgleichung.                   | Datenpunkte und QR-Fit; zusätzlich Balken- oder Textvergleich der Konditionszahlen `cond(A.T@A)` und `cond(R)`. |
| [`polyfit_vandermonde_normal_qr.py`](2_Lineare_Ausgleichsrechnung/polyfit_vandermonde_normal_qr.py)             | Polynom-Ausgleich beliebigen Grades über Vandermonde-Matrix. Löst mit Normalgleichung und QR und vergleicht optional mit `np.polyfit`.                        | Datenpunkte und Fits mehrerer Polynomgrade; zusätzlich Fehler `E` gegen Polynomgrad.                            |
| [`ausgleich_basisfunktionen_allgemein.py`](2_Lineare_Ausgleichsrechnung/ausgleich_basisfunktionen_allgemein.py) | Allgemeines lineares Ausgleichsproblem mit frei wählbaren Basisfunktionen, z.B. `1`, `x`, `x²`, `sin(x)`, `exp(-x)`. Designmatrix wird automatisch aufgebaut. | Datenpunkte, Ausgleichsfunktion und Residuenplot.                                                               |

---

## 3. Nichtlineare Ausgleichsrechnung und Gauss-Newton

**Ordner:** [`3_Nichtlineare_Ausgleichsrechnung_Gauss_Newton/`](3_Nichtlineare_Ausgleichsrechnung_Gauss_Newton/)

| Skript                                                                                                              | Beschreibung                                                                                                                                        | Plot                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| [`gauss_newton_normalgleichung.py`](3_Nichtlineare_Ausgleichsrechnung_Gauss_Newton/gauss_newton_normalgleichung.py) | Nichtlinearer Fit, z.B. `f(x,a,b,c)=a*exp(-b*x)+c`, mit Gauss-Newton über Normalgleichung `Dg.T Dg delta = -Dg.T g`.                                | Datenpunkte und finaler Fit; zusätzlich Fehlerfunktional `norm(g(lambda_k))^2` gegen Iteration. |
| [`gauss_newton_qr.py`](3_Nichtlineare_Ausgleichsrechnung_Gauss_Newton/gauss_newton_qr.py)                           | Gauss-Newton, aber das linearisierte Least-Squares-Problem wird per QR-Zerlegung gelöst: `Dg delta ≈ -g`. Stabilere Variante als Normalgleichung.   | Datenpunkte und finaler Fit; zusätzlich Vergleich der Residuenentwicklung mit Normalgleichungsvariante. |
| [`gauss_newton_gedaempft.py`](3_Nichtlineare_Ausgleichsrechnung_Gauss_Newton/gauss_newton_gedaempft.py)             | Gedämpftes Gauss-Newton-Verfahren. Reduziert den Schritt `delta/2**p`, bis das Residuum kleiner wird. Robust bei schlechten Startwerten.            | Datenpunkte und Fit pro Iteration; zusätzlich Plot `norm(g(lambda_k))^2` gegen Iteration und verwendetes `p`. |
| [`linear_vs_nonlinear_fit.py`](3_Nichtlineare_Ausgleichsrechnung_Gauss_Newton/linear_vs_nonlinear_fit.py)           | Demonstriert die Entscheidung zwischen linearem Ausgleichsproblem und nichtlinearem Gauss-Newton-Fit. Vergleicht z.B. Gerade vs. Exponentialmodell. | Datenpunkte, linearer Fit und nichtlinearer Fit im selben Plot; zusätzlich Residuenvergleich.           |

---

## 4. Interpolation

**Ordner:** [`4_Interpolation/`](4_Interpolation/)

| Skript                                                                                                             | Beschreibung                                                                                                                                      | Plot                                                                                             |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| [`lagrange_interpolation_direkt.py`](4_Interpolation/lagrange_interpolation_direkt.py)                             | Baut das Lagrange-Interpolationspolynom direkt nach Formel auf. Gibt die Lagrange-Basisfunktionen und das ausmultiplizierte Polynom aus.          | Stützpunkte und Lagrange-Polynom über dem Intervall.                                             |
| [`lagrange_interpolation_punktwert.py`](4_Interpolation/lagrange_interpolation_punktwert.py)                       | Minimal-Skript für Prüfungsaufgaben vom Typ “Bestimme den interpolierten Wert bei `x_eval`”. Gibt alle `l_i(x_eval)` und `P(x_eval)` aus.         | Plot der Stützpunkte, des Interpolationspolynoms und des markierten Auswertungspunktes `x_eval`. |
| [`newton_interpolation_dividierte_differenzen.py`](4_Interpolation/newton_interpolation_dividierte_differenzen.py) | Erstellt die Tabelle der dividierten Differenzen und baut daraus das Newton-Interpolationspolynom auf. Vergleicht mit Lagrange.                   | Stützpunkte und Newton-Interpolationspolynom; zusätzlich Vergleich mit Lagrange-Kurve.           |
| [`kubischer_spline_natuerlich.py`](4_Interpolation/kubischer_spline_natuerlich.py)                                 | Berechnet natürliche kubische Splines. Löst das LGS für die `c_i` und berechnet `a_i`, `b_i`, `c_i`, `d_i` für jedes Intervall.                   | Stützpunkte und stückweise kubische Splinefunktion.                                              |
| [`interpolation_lagrange_vs_spline.py`](4_Interpolation/interpolation_lagrange_vs_spline.py)                       | Vergleicht globales Lagrange-Polynom mit natürlichem kubischem Spline auf denselben Daten. Nützlich zur Visualisierung von Polynom-Oszillationen. | Stützpunkte, Lagrange-Polynom und Spline im selben Plot.                                         |

---

## 5. Numerische Integration und Romberg

**Ordner:** [`5_Numerische_Integration_Romberg/`](5_Numerische_Integration_Romberg/)

| Skript                                                                                                                        | Beschreibung                                                                                                                                      | Plot                                                                                            |
| ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| [`newton_cotes_rechteck_trapez_simpson.py`](5_Numerische_Integration_Romberg/newton_cotes_rechteck_trapez_simpson.py)         | Implementiert summierte Rechteckregel, Trapezregel und Simpson-Regel auf `[a,b]`. Vergleicht mit exakter Lösung oder hochgenauer Referenz.        | Funktion `f(x)` mit approximierenden Flächen; zusätzlich Fehler gegen Anzahl Subintervalle `n`. |
| [`trapez_nicht_aequidistant.py`](5_Numerische_Integration_Romberg/trapez_nicht_aequidistant.py)                               | Trapezregel für nicht-äquidistante Stützstellen. Geeignet für tabellarische Messdaten, z.B. Dichte- oder Geschwindigkeitsprofile.                 | Messpunkte und eingezeichnete Trapezflächen.                                                    |
| [`quadratur_fehlerabschaetzung_h_bestimmen.py`](5_Numerische_Integration_Romberg/quadratur_fehlerabschaetzung_h_bestimmen.py) | Bestimmt Schrittweite `h` und Intervallzahl `n` für gewünschte Genauigkeit. Nutzt `sympy` für Ableitungen wie `f''` oder `f''''`.                 | Plot der theoretischen Fehlerschranke gegen `h` und Markierung der gewählten Schrittweite.      |
| [`romberg_extrapolation.py`](5_Numerische_Integration_Romberg/romberg_extrapolation.py)                                       | Erstellt vollständige Romberg-Tabelle aus Trapezwerten mit halbierten Schrittweiten. Gibt den genauesten Wert rechts unten aus.                   | Plot der Romberg-Diagonalwerte und ihres Fehlers gegen die Extrapolationsstufe.                 |
| [`gauss_quadratur_g1_g2_g3.py`](5_Numerische_Integration_Romberg/gauss_quadratur_g1_g2_g3.py)                                 | Implementiert Gauss-Quadratur mit 1, 2 und 3 Stützstellen auf `[a,b]`. Vergleicht mit Simpson bei gleicher Anzahl Funktionsauswertungen.          | Fehlervergleich als Balkendiagramm für G1, G2, G3 und Simpson.                                  |
| [`integration_vergleich_verfahren.py`](5_Numerische_Integration_Romberg/integration_vergleich_verfahren.py)                   | Vergleicht Rechteck, Trapez, Simpson, Romberg und Gauss-Quadratur auf derselben Funktion. Gibt Wert, Fehler und Anzahl Funktionsauswertungen aus. | Fehler gegen Anzahl Funktionsauswertungen, idealerweise mit logarithmischer y-Achse.            |

---

## 6. Anfangswertprobleme, Euler, Heun und Runge-Kutta

**Ordner:** [`6_DGL_Euler_Heun_Runge_Kutta/`](6_DGL_Euler_Heun_Runge_Kutta/)

| Skript                                                                                                        | Beschreibung                                                                                                                            | Plot                                                                                                 |
| ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| [`euler_explizit_skalar.py`](6_DGL_Euler_Heun_Runge_Kutta/euler_explizit_skalar.py)                           | Löst ein skalares Anfangswertproblem `y'=f(t,y)`, `y(t0)=y0` mit explizitem Eulerverfahren.                                             | Numerische Euler-Lösung gegen exakte Lösung; zusätzlich Fehlerkurve.                                 |
| [`euler_explizit_system.py`](6_DGL_Euler_Heun_Runge_Kutta/euler_explizit_system.py)                           | Eulerverfahren für DGL-Systeme `u'=F(t,u)`. Geeignet für Federschwingung oder Pendel als System erster Ordnung.                         | Zeitverläufe der Zustandsvariablen und Phasenportrait.                                               |
| [`heun_modifiziertes_eulerverfahren.py`](6_DGL_Euler_Heun_Runge_Kutta/heun_modifiziertes_eulerverfahren.py)   | Modifiziertes Eulerverfahren / Heun: Euler-Prädiktor und Trapez-Korrektor. Verfahren 2. Ordnung.                                        | Euler, Heun und exakte Lösung im Vergleich; zusätzlich Fehlerkurve.                                  |
| [`mittelpunktverfahren.py`](6_DGL_Euler_Heun_Runge_Kutta/mittelpunktverfahren.py)                             | Runge-Kutta-Verfahren 2. Ordnung mit Mittelpunktsteigung.                                                                               | Mittelpunktverfahren, Euler und Heun im Vergleich; zusätzlich Fehler am Endpunkt gegen Schrittweite. |
| [`runge_kutta_4.py`](6_DGL_Euler_Heun_Runge_Kutta/runge_kutta_4.py)                                           | Klassisches Runge-Kutta-Verfahren 4. Ordnung mit `k1`, `k2`, `k3`, `k4`.                                                                | Euler, Heun, Mittelpunkt und RK4 gegen exakte Lösung; zusätzlich Fehlervergleich.                    |
| [`runge_kutta_butcher_allgemein.py`](6_DGL_Euler_Heun_Runge_Kutta/runge_kutta_butcher_allgemein.py)           | Allgemeines explizites s-stufiges Runge-Kutta-Verfahren über Butcher-Tabelle `(A,b,c)`. Kann Euler, Mittelpunkt, Heun und RK4 abbilden. | Vergleich mehrerer Butcher-Verfahren auf demselben AWP.                                              |
| [`euler_stabilitaet_testgleichung.py`](6_DGL_Euler_Heun_Runge_Kutta/euler_stabilitaet_testgleichung.py)       | Testet Stabilität des Eulerverfahrens an `y'=lambda*y`. Zeigt die Bedingung `abs(1+h*lambda) < 1`.                                      | Stabile und instabile Euler-Lösungen für verschiedene Schrittweiten `h`.                             |
| [`dgl_konvergenzordnung_vergleich.py`](6_DGL_Euler_Heun_Runge_Kutta/dgl_konvergenzordnung_vergleich.py)       | Experimentelle Bestimmung der Konvergenzordnung über Fehler bei `h`, `h/2`, `h/4`, `h/8`. Vergleicht Euler, Heun, Mittelpunkt und RK4.  | Log-Log-Plot des Fehlers gegen Schrittweite `h` mit geschätzter Ordnung.                             |
| [`federschwingung_verfahrenvergleich.py`](6_DGL_Euler_Heun_Runge_Kutta/federschwingung_verfahrenvergleich.py) | Modelliert Federschwingung als System `x'=v`, `v'=-(k/m)x`, optional mit Dämpfung. Vergleicht Euler, Heun und RK4.                      | `x(t)`, `v(t)`, Phasenportrait `v(x)` und Energie `E(t)`.                                            |
| [`pendel_nonlinear_rk4.py`](6_DGL_Euler_Heun_Runge_Kutta/pendel_nonlinear_rk4.py)                             | Löst das nichtlineare Pendel `theta'=omega`, `omega'=-(g/l)sin(theta)` mit RK4 und vergleicht optional mit linearisiertem Pendel.       | `theta(t)`, `omega(t)`, Phasenportrait und Vergleich nichtlinear vs. linearisiert.                   |

---
