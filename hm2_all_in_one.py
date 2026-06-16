# HM2 All-in-one Toolbox
# Automatisch aus hm2_toolbox/*.py zusammengefuehrt.
# Diese Datei ist fuer direktes Kopieren in Pruefungsumgebungen gedacht.

from __future__ import annotations

import cmath
import math
import numbers
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt


# ===== datentypen.py =====
"""Zentrale Ergebnis-Datentypen der HM2-Toolbox."""





@dataclass
class IterationsErgebnis:
    """Speichert Resultate iterativer Verfahren.

    Zweck: Einheitliche Rueckgabe fuer Newton-, Gauss-Newton- und aehnliche Verfahren.
    Formel/Verfahren: Speichert Loesung, Iterationszahl, Konvergenzstatus und optional
    den Verlauf mit Residuen und Schritten.
    Input: Felder mit Loesung, iterationen, konvergiert, residualnorm, schrittnorm,
    verlauf und nachricht.
    Output: Dataclass mit `als_dict()` und `verlauf_als_tabelle()`.
    Voraussetzungen: Der Verlauf ist eine Liste von Dictionaries oder None.
    Numerische Hinweise: Residual- und Schrittnorm zeigen, ob das Verfahren stabil
    und plausibel konvergiert ist.
    Beispiel:
        >>> IterationsErgebnis(1.0, 3, True).als_dict()["konvergiert"]
        True
    """

    loesung: Any
    iterationen: int
    konvergiert: bool
    residualnorm: float | None = None
    schrittnorm: float | None = None
    verlauf: list[dict] | None = None
    nachricht: str = ""

    def als_dict(self) -> dict:
        """Gibt das Ergebnis als Dictionary zurueck.

        Zweck: Leichte Weiterverarbeitung in Skripten.
        Formel/Verfahren: `dataclasses.asdict`.
        Input: keine.
        Output: dict mit allen Feldern.
        Voraussetzungen: keine.
        Imports: from dataclasses import asdict.
        Numerische Hinweise: keine Veraenderung der Werte.
        Beispiel:
            >>> IterationsErgebnis(0, 0, False).als_dict()["iterationen"]
            0
        """
        return asdict(self)

    def verlauf_als_tabelle(self) -> pd.DataFrame:
        """Erzeugt eine Pandas-Tabelle aus dem Iterationsverlauf.

        Zweck: Pruefungsfreundliche Anzeige der Iterationen.
        Formel/Verfahren: Liste von Dictionaries wird zu `pd.DataFrame`.
        Input: keine.
        Output: Pandas-DataFrame.
        Voraussetzungen: Verlauf enthaelt tabellarische Eintraege.
        Imports: import pandas as pd.
        Numerische Hinweise: Tabellen erleichtern das Erkennen stagnierender Residuen.
        Beispiel:
            >>> IterationsErgebnis(1, 1, True, verlauf=[{"k": 0}]).verlauf_als_tabelle().shape
            (1, 1)
        """
        return pd.DataFrame(self.verlauf or [])

    def als_tabelle(self) -> pd.DataFrame:
        """Alias fuer `verlauf_als_tabelle`.

        Zweck: Einheitlicher Tabellenzugriff.
        Formel/Verfahren: delegiert an `verlauf_als_tabelle`.
        Input: keine.
        Output: Pandas-DataFrame.
        Voraussetzungen: keine.
        Imports: import pandas as pd.
        Numerische Hinweise: keine.
        Beispiel:
            >>> IterationsErgebnis(1, 0, True).als_tabelle().empty
            True
        """
        return self.verlauf_als_tabelle()


@dataclass
class AusgleichsErgebnis:
    """Speichert Ergebnisse einer Ausgleichsrechnung.

    Zweck: Koeffizienten, Residuen und Qualitaetskennzahlen zusammenhalten.
    Formel/Verfahren: Fehlerfunktional E = ||y - A lambda||_2^2.
    Input: Koeffizienten, Residuen, Fehlerfunktional, Rang, Konditionszahl, Methode,
    optionale Tabelle.
    Output: Dataclass mit Dictionary- und Tabellenzugriff.
    Voraussetzungen: Residuen passen zu den Messwerten.
    Numerische Hinweise: QR ist meist stabiler als Normalgleichungen.
    Beispiel:
        >>> AusgleichsErgebnis([1], [0], 0.0, 1, 1.0, "qr").als_dict()["methode"]
        'qr'
    """

    koeffizienten: Any
    residuen: Any
    fehlerfunktional: float
    rang: int
    konditionszahl: float
    methode: str
    tabelle: pd.DataFrame | None = None

    def als_dict(self) -> dict:
        """Gibt das Ergebnis als Dictionary zurueck.

        Zweck: Exportierbare Struktur fuer weitere Berechnungen.
        Formel/Verfahren: `dataclasses.asdict`.
        Input: keine.
        Output: dict.
        Voraussetzungen: keine.
        Imports: from dataclasses import asdict.
        Numerische Hinweise: NumPy-Arrays bleiben als Array-artige Werte enthalten.
        Beispiel:
            >>> AusgleichsErgebnis([1], [0], 0, 1, 1, "m").als_dict()["rang"]
            1
        """
        return asdict(self)

    def als_tabelle(self) -> pd.DataFrame:
        """Gibt die Residuentabelle zurueck.

        Zweck: Messwerte und Residuen direkt ansehen.
        Formel/Verfahren: Falls keine Tabelle existiert, werden Residuen tabelliert.
        Input: keine.
        Output: Pandas-DataFrame.
        Voraussetzungen: Residuen sind tabellierbar.
        Imports: import pandas as pd.
        Numerische Hinweise: Grosse Residuen markieren schlechte Modellanpassung.
        Beispiel:
            >>> AusgleichsErgebnis([1], [0], 0, 1, 1, "m").als_tabelle().shape[0]
            1
        """
        if self.tabelle is not None:
            return self.tabelle
        return pd.DataFrame({"residuum": np.asarray(self.residuen, dtype=float)})


@dataclass
class SplineKoeffizienten:
    """Koeffizienten eines natuerlichen kubischen Splines.

    Zweck: Speichert die Abschnittskoeffizienten in
    S_i(x)=a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3.
    Formel/Verfahren: Natuerliche Randbedingungen c_0=c_n=0.
    Input: Stuetzstellen und Koeffizienten a,b,c,d.
    Output: Dataclass mit Tabellenzugriff.
    Voraussetzungen: Stuetzstellen streng steigend.
    Numerische Hinweise: Enge oder fast gleiche Stuetzstellen koennen schlecht
    konditionierte Gleichungssysteme erzeugen.
    Beispiel:
        >>> SplineKoeffizienten([0, 1], [0], [1], [0], [0]).als_tabelle().shape[0]
        1
    """

    stuetzstellen: Any
    a_werte: Any
    b_werte: Any
    c_werte: Any
    d_werte: Any

    def als_dict(self) -> dict:
        """Gibt die Spline-Koeffizienten als Dictionary zurueck.

        Zweck: Einfacher Zugriff auf alle Koeffizienten.
        Formel/Verfahren: `dataclasses.asdict`.
        Input: keine.
        Output: dict.
        Voraussetzungen: keine.
        Imports: from dataclasses import asdict.
        Numerische Hinweise: keine.
        Beispiel:
            >>> stuetzstellen = np.array([0.0, 1.0])
            >>> a_werte = np.array([0.0])
            >>> b_werte = np.array([1.0])
            >>> c_werte = np.array([0.0])
            >>> d_werte = np.array([0.0])
            >>> koeffizienten = SplineKoeffizienten(stuetzstellen, a_werte, b_werte, c_werte, d_werte)
            >>> "a_werte" in koeffizienten.als_dict()
            True
        """
        return asdict(self)

    def als_tabelle(self) -> pd.DataFrame:
        """Erstellt eine Tabelle der Spline-Abschnitte.

        Zweck: Uebersichtliche Darstellung der Koeffizienten.
        Formel/Verfahren: Ein Abschnitt pro Zeile.
        Input: keine.
        Output: Pandas-DataFrame.
        Voraussetzungen: Koeffizientenlisten haben gleiche Laenge.
        Imports: import pandas as pd.
        Numerische Hinweise: Vorzeichen der d-Werte zeigt Kruemmungsaenderung.
        Beispiel:
            >>> stuetzstellen = np.array([0.0, 1.0])
            >>> a_werte = np.array([0.0])
            >>> b_werte = np.array([1.0])
            >>> c_werte = np.array([0.0])
            >>> d_werte = np.array([0.0])
            >>> koeffizienten = SplineKoeffizienten(stuetzstellen, a_werte, b_werte, c_werte, d_werte)
            >>> koeffizienten.als_tabelle().columns[0]
            'x_i'
        """
        x = np.asarray(self.stuetzstellen, dtype=float)
        return pd.DataFrame({
            "x_i": x[:-1],
            "x_i_plus_1": x[1:],
            "a": self.a_werte,
            "b": self.b_werte,
            "c": self.c_werte,
            "d": self.d_werte,
        })


@dataclass
class QuadraturErgebnis:
    """Speichert Resultate einer numerischen Integration.

    Zweck: Wert, Methode, Schrittweite und Zusatzinfos gemeinsam zurueckgeben.
    Formel/Verfahren: Abhaengig von der Quadraturregel, z. B. Trapez oder Simpson.
    Input: Wert, Methode, Schrittweite, Anzahl Intervalle, Fehlergrenze, Zusatzinfos.
    Output: Dataclass mit Dictionary- und Tabellenzugriff.
    Voraussetzungen: Schrittweite und Intervallanzahl passen zur Methode.
    Numerische Hinweise: Fehlergrenzen sind nur so gut wie die Ableitungsschranken.
    Beispiel:
        >>> QuadraturErgebnis(1.0, "Test", 0.1, 10).als_dict()["wert"]
        1.0
    """

    wert: float
    methode: str
    schrittweite: float | None = None
    anzahl_intervalle: int | None = None
    fehlergrenze: float | None = None
    zusatzinfos: dict | None = None

    def als_dict(self) -> dict:
        """Gibt das Quadraturergebnis als Dictionary zurueck.

        Zweck: Kompakte Ausgabe fuer Skripte.
        Formel/Verfahren: `dataclasses.asdict`.
        Input: keine.
        Output: dict.
        Voraussetzungen: keine.
        Imports: from dataclasses import asdict.
        Numerische Hinweise: keine.
        Beispiel:
            >>> QuadraturErgebnis(2.0, "M").als_dict()["methode"]
            'M'
        """
        return asdict(self)

    def als_tabelle(self) -> pd.DataFrame:
        """Erstellt eine einzeilige Ergebnistabelle.

        Zweck: Vergleich mehrerer Quadraturergebnisse vorbereiten.
        Formel/Verfahren: Felder werden in eine Tabellenzeile geschrieben.
        Input: keine.
        Output: Pandas-DataFrame.
        Voraussetzungen: keine.
        Imports: import pandas as pd.
        Numerische Hinweise: keine.
        Beispiel:
            >>> QuadraturErgebnis(1, "M").als_tabelle().shape
            (1, 5)
        """
        return pd.DataFrame([{
            "methode": self.methode,
            "wert": self.wert,
            "schrittweite": self.schrittweite,
            "anzahl_intervalle": self.anzahl_intervalle,
            "fehlergrenze": self.fehlergrenze,
        }])


@dataclass
class DglErgebnis:
    """Speichert die numerische Loesung eines Anfangswertproblems.

    Zweck: x-Werte, y-Werte, Methode und Tabelle gemeinsam zurueckgeben.
    Formel/Verfahren: Einschrittverfahren y_{i+1}=y_i+h*Phi(...).
    Input: x_werte, y_werte, Methode, Schrittweite, Ordnung, optionale Tabelle.
    Output: Dataclass mit Dictionary- und Tabellenzugriff.
    Voraussetzungen: x_werte und y_werte haben gleiche Laenge.
    Numerische Hinweise: Kleinere Schrittweiten reduzieren meist den globalen Fehler.
    Beispiel:
        >>> x_werte = np.array([0.0, 1.0])
        >>> y_werte = np.array([1.0, 2.0])
        >>> ergebnis = DglErgebnis(x_werte, y_werte, "Euler", 1, 1)
        >>> ergebnis.als_tabelle().shape[0]
        2
    """

    x_werte: Any
    y_werte: Any
    methode: str
    schrittweite: float
    ordnung: int
    tabelle: pd.DataFrame | None = None

    def als_dict(self) -> dict:
        """Gibt das DGL-Ergebnis als Dictionary zurueck.

        Zweck: Weiterverarbeitung der numerischen Loesung.
        Formel/Verfahren: `dataclasses.asdict`.
        Input: keine.
        Output: dict.
        Voraussetzungen: keine.
        Imports: from dataclasses import asdict.
        Numerische Hinweise: keine.
        Beispiel:
            >>> DglErgebnis([0], [1], "M", 1, 1).als_dict()["ordnung"]
            1
        """
        return asdict(self)

    def als_tabelle(self) -> pd.DataFrame:
        """Gibt oder erzeugt eine Loesungstabelle.

        Zweck: x- und y-Werte schnell pruefungsfreundlich anzeigen.
        Formel/Verfahren: Vektorwerte werden in Spalten y_0, y_1, ... zerlegt.
        Input: keine.
        Output: Pandas-DataFrame.
        Voraussetzungen: y_werte ist skalar oder zweidimensional tabellierbar.
        Imports: import pandas as pd.
        Numerische Hinweise: Tabellen helfen beim Vergleich mit exakten Loesungen.
        Beispiel:
            >>> DglErgebnis([0], [1], "M", 1, 1).als_tabelle().columns[0]
            'x'
        """
        if self.tabelle is not None:
            return self.tabelle
        x = np.asarray(self.x_werte, dtype=float)
        y = np.asarray(self.y_werte, dtype=float)
        if y.ndim == 1:
            return pd.DataFrame({"x": x, "y": y})
        daten = {"x": x}
        for index in range(y.shape[1]):
            daten[f"y_{index}"] = y[:, index]
        return pd.DataFrame(daten)


# ===== validierung.py =====
"""Validierungs- und Konvertierungsfunktionen fuer die HM2-Toolbox."""





def als_vektor(eingabe, name="vektor") -> np.ndarray:
    """Konvertiert eine Eingabe in einen eindimensionalen NumPy-Vektor.

    Zweck: Einheitliche Vektorform fuer numerische Verfahren.
    Formel/Verfahren: `np.asarray(...).reshape(-1)` fuer Listen, Tupel und Arrays.
    Input: eingabe als Array-artiger Wert, name fuer Fehlermeldungen.
    Output: NumPy-Array mit Dimension n.
    Voraussetzungen: Eingabe muss numerisch und endlich sein.
    Numerische Hinweise: Es wird auf dtype float konvertiert.
    Beispiel:
        >>> als_vektor([1, 2]).tolist()
        [1.0, 2.0]
    """
    vektor = np.asarray(eingabe, dtype=float).reshape(-1)
    pruefe_endliche_werte(vektor, name)
    return vektor


def als_matrix(eingabe, name="matrix") -> np.ndarray:
    """Konvertiert eine Eingabe in eine zweidimensionale NumPy-Matrix.

    Zweck: Einheitliche Matrixform fuer LGS, QR und Ausgleichsrechnung.
    Formel/Verfahren: `np.asarray` mit dtype float und Dimensionspruefung.
    Input: eingabe als Matrix-artiger Wert.
    Output: NumPy-Array mit Dimension m x n.
    Voraussetzungen: Eingabe muss zweidimensional und endlich sein.
    Numerische Hinweise: Exakte Werte werden fuer numerische Verfahren zu float.
    Beispiel:
        >>> als_matrix([[1, 2]]).shape
        (1, 2)
    """
    matrix = np.asarray(eingabe, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} muss zweidimensional sein.")
    pruefe_endliche_werte(matrix, name)
    return matrix


def als_sympy_matrix(eingabe, name="matrix") -> sp.Matrix:
    """Konvertiert eine Eingabe in eine SymPy-Matrix.

    Zweck: Symbolische Matrixoperationen mit deutscher Wrapper-API.
    Formel/Verfahren: `sp.Matrix(eingabe)`.
    Input: Matrix-artige Eingabe.
    Output: SymPy-Matrix.
    Voraussetzungen: Eingabe muss von SymPy interpretierbar sein.
    Numerische Hinweise: Symbolische Werte bleiben exakt.
    Beispiel:
        >>> als_sympy_matrix([[1, 2]]).shape
        (1, 2)
    """
    return sp.Matrix(eingabe)


def pruefe_endliche_werte(array, name="array") -> None:
    """Prueft, ob alle numerischen Werte endlich sind.

    Zweck: NaN und Inf frueh abfangen.
    Formel/Verfahren: `np.isfinite`.
    Input: Array-artige numerische Werte.
    Output: None, sonst ValueError.
    Voraussetzungen: Eingabe ist numerisch konvertierbar.
    Numerische Hinweise: Nicht-endliche Werte zerstoeren Normen und LGS-Loeser.
    Beispiel:
        >>> pruefe_endliche_werte([1, 2])
    """
    werte = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(werte)):
        raise ValueError(f"{name} enthaelt NaN oder unendliche Werte.")


def pruefe_gleiche_laenge(*arrays) -> None:
    """Prueft gleiche erste Laenge mehrerer Arrays.

    Zweck: Messwerte, Stuetzstellen und Modellwerte passend halten.
    Formel/Verfahren: Vergleich von `len(...)`.
    Input: beliebig viele Arrays.
    Output: None, sonst ValueError.
    Voraussetzungen: Jedes Argument besitzt eine Laenge.
    Numerische Hinweise: Laengenfehler fuehren sonst zu falschen Residuen.
    Beispiel:
        >>> pruefe_gleiche_laenge([1], [2])
    """
    laengen = [len(array) for array in arrays]
    if len(set(laengen)) > 1:
        raise ValueError(f"Arrays muessen gleiche Laenge haben, erhalten: {laengen}.")


def pruefe_quadratische_matrix(matrix) -> None:
    """Prueft, ob eine Matrix quadratisch ist.

    Zweck: Determinante, Inverse und quadratische LGS absichern.
    Formel/Verfahren: Shape-Pruefung m=n.
    Input: Matrix-artige Eingabe.
    Output: None, sonst ValueError.
    Voraussetzungen: Eingabe ist zweidimensional.
    Numerische Hinweise: Nichtquadratische Matrizen haben keine klassische Inverse.
    Beispiel:
        >>> pruefe_quadratische_matrix([[1]])
    """
    matrix_np = np.asarray(matrix, dtype=float)
    if matrix_np.shape[0] != matrix_np.shape[1]:
        raise ValueError(f"Matrix muss quadratisch sein, erhalten: {matrix_np.shape}.")


def pruefe_dimensionen_matrix_vektor(matrix, vektor) -> None:
    """Prueft Kompatibilitaet von A und b in A x = b.

    Zweck: Lineare Gleichungssysteme vorab validieren.
    Formel/Verfahren: Anzahl Zeilen von A muss Laenge von b sein.
    Input: Matrix A und Vektor b.
    Output: None, sonst ValueError.
    Voraussetzungen: A zweidimensional, b eindimensional.
    Numerische Hinweise: Dimensionfehler sind keine numerischen, sondern Modellfehler.
    Beispiel:
        >>> pruefe_dimensionen_matrix_vektor([[1, 2]], [3])
    """
    matrix_np = np.asarray(matrix, dtype=float)
    vektor_np = np.asarray(vektor, dtype=float).reshape(-1)
    if matrix_np.shape[0] != vektor_np.size:
        raise ValueError(f"Dimensionen passen nicht: A{matrix_np.shape}, b{vektor_np.shape}.")


def pruefe_stuetzstellen_paarweise_verschieden(stuetzstellen) -> None:
    """Prueft, ob Stuetzstellen paarweise verschieden sind.

    Zweck: Interpolation ohne Division durch null.
    Formel/Verfahren: Vergleich der Anzahl eindeutiger Werte.
    Input: Stuetzstellen.
    Output: None, sonst ValueError.
    Voraussetzungen: Werte sind numerisch.
    Numerische Hinweise: Sehr nahe Punkte koennen trotzdem schlecht konditionieren.
    Beispiel:
        >>> pruefe_stuetzstellen_paarweise_verschieden([0, 1])
    """
    werte = np.asarray(stuetzstellen, dtype=float).reshape(-1)
    if np.unique(werte).size != werte.size:
        raise ValueError("Stuetzstellen muessen paarweise verschieden sein.")


def pruefe_stuetzstellen_streng_steigend(stuetzstellen) -> None:
    """Prueft streng steigende Stuetzstellen.

    Zweck: Splines und nichtaequidistante Integration absichern.
    Formel/Verfahren: `np.diff(x) > 0`.
    Input: Stuetzstellen.
    Output: None, sonst ValueError.
    Voraussetzungen: Werte sind numerisch.
    Numerische Hinweise: Sortiere Daten bewusst, nicht automatisch, um Messfehler zu sehen.
    Beispiel:
        >>> pruefe_stuetzstellen_streng_steigend([0, 1, 2])
    """
    werte = np.asarray(stuetzstellen, dtype=float).reshape(-1)
    if not np.all(np.diff(werte) > 0):
        raise ValueError("Stuetzstellen muessen streng steigend sein.")


def pruefe_positive_schrittweite(schrittweite) -> None:
    """Prueft eine positive Schrittweite.

    Zweck: DGL- und Quadraturverfahren absichern.
    Formel/Verfahren: h > 0.
    Input: schrittweite als Zahl.
    Output: None, sonst ValueError.
    Voraussetzungen: Wert ist skalar.
    Numerische Hinweise: Zu kleine h koennen Rundungsfehler verstaerken.
    Beispiel:
        >>> pruefe_positive_schrittweite(0.1)
    """
    if not ist_skalar(schrittweite) or float(schrittweite) <= 0:
        raise ValueError("Schrittweite muss positiv sein.")


def pruefe_anzahl_intervalle(anzahl_intervalle) -> None:
    """Prueft eine positive ganzzahlige Intervallanzahl.

    Zweck: Summierte Quadratur- und DGL-Verfahren absichern.
    Formel/Verfahren: n ist Integer und n > 0.
    Input: anzahl_intervalle.
    Output: None, sonst ValueError.
    Voraussetzungen: keine.
    Imports: import numbers.
    Numerische Hinweise: Mehr Intervalle bedeuten oft mehr Genauigkeit, aber mehr Aufwand.
    Beispiel:
        >>> pruefe_anzahl_intervalle(4)
    """
    if not isinstance(anzahl_intervalle, numbers.Integral) or int(anzahl_intervalle) <= 0:
        raise ValueError("Anzahl Intervalle muss eine positive ganze Zahl sein.")


def pruefe_gerade_anzahl_intervalle(anzahl_intervalle) -> None:
    """Prueft eine positive gerade Intervallanzahl.

    Zweck: Summierte Simpson-Regel absichern.
    Formel/Verfahren: n > 0 und n mod 2 = 0.
    Input: anzahl_intervalle.
    Output: None, sonst ValueError.
    Voraussetzungen: n ist ganzzahlig.
    Numerische Hinweise: Simpson braucht paarweise Teilintervalle.
    Beispiel:
        >>> pruefe_gerade_anzahl_intervalle(4)
    """
    if not isinstance(anzahl_intervalle, numbers.Integral) or int(anzahl_intervalle) <= 0:
        raise ValueError("Anzahl Intervalle muss eine positive ganze Zahl sein.")
    if int(anzahl_intervalle) % 2 != 0:
        raise ValueError("Anzahl Intervalle muss fuer Simpson gerade sein.")


def ist_skalar(wert) -> bool:
    """Prueft, ob ein Wert skalar ist.

    Zweck: Parameter wie Schrittweite oder Toleranz validieren.
    Formel/Verfahren: NumPy-Skalarpruefung mit Dimension 0.
    Input: beliebiger Wert.
    Output: bool.
    Voraussetzungen: keine.
    Numerische Hinweise: Skalar bedeutet nicht automatisch endlich.
    Beispiel:
        >>> ist_skalar(1.0)
        True
    """
    return np.isscalar(wert) or np.asarray(wert).ndim == 0


# ===== lineare_algebra.py =====
"""Lineare Algebra, Matrizen, LGS und QR-Zerlegung mit deutscher API."""





def _kurzdoc(name: str, formel: str, beispiel: str) -> str:
    """Erstellt einen kurzen deutschen Standard-Docstring fuer interne Zwecke."""
    return f"""Zweck: {name}.

    Mathematische Formel / Verfahren: {formel}.
    Input: numerische oder symbolische Werte gemaess Signatur.
    Output: berechneter Wert in NumPy- oder SymPy-Form.
    Voraussetzungen: Dimensionen muessen zum Verfahren passen.
    Numerische Hinweise: Bei schlecht konditionierten Matrizen koennen Rundungsfehler gross werden.
    Beispiel:
        >>> {beispiel}
    """


def vektor_norm_1(vektor) -> float:
    """Berechnet die 1-Norm eines Vektors.

    Formel: ||x||_1 = sum_i |x_i|.
    Input: vektor mit Dimension n. Output: float. Voraussetzungen: endliche Werte.
    Numerischer Hinweis: Robust gegen Vorzeichen, aber nicht gegen Skalierung.
    Beispiel:
        >>> vektor_norm_1([1, -2])
        3.0
    """
    return float(np.linalg.norm(np.asarray(vektor, dtype=float).reshape(-1), ord=1))


def vektor_norm_2(vektor) -> float:
    """Berechnet die euklidische 2-Norm.

    Formel: ||x||_2 = sqrt(sum_i x_i^2). Input: Vektor. Output: float.
    Voraussetzungen: endliche Werte. Hinweis: Standardnorm fuer Residuen.
    Beispiel:
        >>> vektor_norm_2([3, 4])
        5.0
    """
    return float(np.linalg.norm(np.asarray(vektor, dtype=float).reshape(-1), ord=2))


def vektor_norm_inf(vektor) -> float:
    """Berechnet die Maximumsnorm eines Vektors.

    Formel: ||x||_inf = max_i |x_i|. Input: Vektor. Output: float.
    Voraussetzungen: endliche Werte. Hinweis: Gut fuer komponentenweise Schranken.
    Beispiel:
        >>> vektor_norm_inf([1, -5, 2])
        5.0
    """
    return float(np.linalg.norm(np.asarray(vektor, dtype=float).reshape(-1), ord=np.inf))


def matrix_norm_1(matrix) -> float:
    """Berechnet die Matrix-1-Norm.

    Formel: ||A||_1 = maximale Spaltensumme. Input: Matrix m x n. Output: float.
    Voraussetzungen: endliche Werte. Hinweis: Wird fuer Konditionszahlen genutzt.
    Beispiel:
        >>> matrix_norm_1([[1, 2], [3, 4]])
        6.0
    """
    return float(np.linalg.norm(np.asarray(matrix, dtype=float), ord=1))


def matrix_norm_2(matrix) -> float:
    """Berechnet die Spektralnorm einer Matrix.

    Formel: ||A||_2 = groesster Singulaerwert. Input: Matrix. Output: float.
    Voraussetzungen: endliche Werte. Hinweis: Teurer, aber aussagekraeftig.
    Beispiel:
        >>> round(matrix_norm_2([[1, 0], [0, 1]]), 6)
        1.0
    """
    return float(np.linalg.norm(np.asarray(matrix, dtype=float), ord=2))


def matrix_norm_inf(matrix) -> float:
    """Berechnet die Matrix-Unendlichnorm.

    Formel: ||A||_inf = maximale Zeilensumme. Input: Matrix. Output: float.
    Voraussetzungen: endliche Werte. Hinweis: Einfach zu pruefen.
    Beispiel:
        >>> matrix_norm_inf([[1, 2], [3, 4]])
        7.0
    """
    return float(np.linalg.norm(np.asarray(matrix, dtype=float), ord=np.inf))


def vektor_laenge(vektor) -> float:
    """Berechnet die Laenge eines Vektors.

    Formel: Laenge = ||x||_2. Input: Vektor. Output: float.
    Voraussetzungen: endliche Werte. Hinweis: Alias fuer die 2-Norm.
    Beispiel:
        >>> vektor_laenge([3, 4])
        5.0
    """
    return vektor_norm_2(vektor)


def einheitsvektor(vektor) -> np.ndarray:
    """Normiert einen Vektor auf Laenge 1.

    Formel: e = v / ||v||_2. Input: Vektor. Output: NumPy-Vektor.
    Voraussetzungen: v darf nicht der Nullvektor sein. Hinweis: Rundungsfehler bei sehr kleinen Normen.
    Beispiel:
        >>> einheitsvektor([2, 0]).tolist()
        [1.0, 0.0]
    """
    vektor_np = np.asarray(vektor, dtype=float).reshape(-1)
    norm = vektor_norm_2(vektor_np)
    if norm == 0:
        raise ValueError("Nullvektor kann nicht normiert werden.")
    return vektor_np / norm


def skalarprodukt(vektor_a, vektor_b) -> float:
    """Berechnet das Skalarprodukt zweier Vektoren.

    Formel: a dot b = sum_i a_i b_i. Input: zwei Vektoren gleicher Laenge. Output: float.
    Voraussetzungen: kompatible Dimensionen. Hinweis: Basis fuer Winkel und Orthogonalitaet.
    Beispiel:
        >>> skalarprodukt([1, 2], [3, 4])
        11.0
    """
    a = np.asarray(vektor_a, dtype=float).reshape(-1)
    b = np.asarray(vektor_b, dtype=float).reshape(-1)
    if a.size != b.size:
        raise ValueError("Vektoren muessen gleiche Laenge haben.")
    return float(np.dot(a, b))


def kreuzprodukt_3d(vektor_a, vektor_b) -> np.ndarray:
    """Berechnet das Kreuzprodukt in R^3.

    Formel: a x b. Input: zwei 3D-Vektoren. Output: 3D-NumPy-Vektor.
    Voraussetzungen: Laenge 3. Hinweis: Ergebnis ist orthogonal zu beiden Eingaben.
    Beispiel:
        >>> kreuzprodukt_3d([1, 0, 0], [0, 1, 0]).tolist()
        [0.0, 0.0, 1.0]
    """
    a = np.asarray(vektor_a, dtype=float).reshape(-1)
    b = np.asarray(vektor_b, dtype=float).reshape(-1)
    if a.size != 3 or b.size != 3:
        raise ValueError("Kreuzprodukt ist hier nur fuer 3D-Vektoren definiert.")
    return np.cross(a, b)


def winkel_zwischen_vektoren(vektor_a, vektor_b, in_grad=False) -> float:
    """Berechnet den Winkel zwischen zwei Vektoren.

    Formel: cos(phi)=<a,b>/(||a|| ||b||). Input: zwei Vektoren. Output: Radiant oder Grad.
    Voraussetzungen: keine Nullvektoren. Hinweis: Quotient wird auf [-1,1] geklemmt.
    Beispiel:
        >>> vektor_a = np.array([1.0, 0.0])
        >>> vektor_b = np.array([0.0, 1.0])
        >>> in_grad = True
        >>> round(winkel_zwischen_vektoren(vektor_a, vektor_b, in_grad), 6)
        90.0
    """
    a = np.asarray(vektor_a, dtype=float).reshape(-1)
    b = np.asarray(vektor_b, dtype=float).reshape(-1)
    nenner = vektor_norm_2(a) * vektor_norm_2(b)
    if nenner == 0:
        raise ValueError("Winkel mit Nullvektor ist nicht definiert.")
    winkel = float(np.arccos(np.clip(np.dot(a, b) / nenner, -1.0, 1.0)))
    return float(np.degrees(winkel)) if in_grad else winkel


def sind_orthogonal(vektor_a, vektor_b, toleranz=1e-10) -> bool:
    """Prueft Orthogonalitaet zweier Vektoren.

    Formel: a dot b = 0. Input: zwei Vektoren. Output: bool.
    Voraussetzungen: gleiche Laenge. Hinweis: Toleranz wegen Rundungsfehlern.
    Beispiel:
        >>> sind_orthogonal([1, 0], [0, 1])
        True
    """
    return abs(skalarprodukt(vektor_a, vektor_b)) <= toleranz


def matrix_addieren(matrix_a, matrix_b) -> np.ndarray:
    """Addiert zwei Matrizen.

    Formel: C=A+B komponentenweise. Input: Matrizen gleicher Form. Output: NumPy-Matrix.
    Voraussetzungen: gleiche Dimension. Hinweis: Reiner Wrapper fuer sichere API.
    Beispiel:
        >>> matrix_addieren([[1]], [[2]]).tolist()
        [[3.0]]
    """
    a, b = np.asarray(matrix_a, dtype=float), np.asarray(matrix_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Matrizen muessen gleiche Form haben.")
    return a + b


def matrix_subtrahieren(matrix_a, matrix_b) -> np.ndarray:
    """Subtrahiert zwei Matrizen.

    Formel: C=A-B. Input: Matrizen gleicher Form. Output: NumPy-Matrix.
    Voraussetzungen: gleiche Dimension. Hinweis: keine.
    Beispiel:
        >>> matrix_subtrahieren([[3]], [[2]]).tolist()
        [[1.0]]
    """
    a, b = np.asarray(matrix_a, dtype=float), np.asarray(matrix_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Matrizen muessen gleiche Form haben.")
    return a - b


def matrix_skalieren(skalare_zahl, matrix) -> np.ndarray:
    """Multipliziert eine Matrix mit einem Skalar.

    Formel: C=alpha A. Input: Skalar und Matrix. Output: NumPy-Matrix.
    Voraussetzungen: Skalar numerisch. Hinweis: Skaliert auch Rundungsfehler.
    Beispiel:
        >>> matrix_skalieren(2, [[1, 2]]).tolist()
        [[2.0, 4.0]]
    """
    return float(skalare_zahl) * np.asarray(matrix, dtype=float)


def matrix_multiplizieren(matrix_a, matrix_b) -> np.ndarray:
    """Multipliziert zwei Matrizen.

    Formel: C_ij=sum_k A_ik B_kj. Input: A m x n, B n x p. Output: m x p.
    Voraussetzungen: innere Dimensionen passen. Hinweis: Wrapper fuer `@`.
    Beispiel:
        >>> matrix_multiplizieren([[1,2]], [[3],[4]]).tolist()
        [[11.0]]
    """
    a, b = np.asarray(matrix_a, dtype=float), np.asarray(matrix_b, dtype=float)
    if a.shape[1] != b.shape[0]:
        raise ValueError("Innere Matrixdimensionen passen nicht.")
    return a @ b


def matrix_transponieren(matrix):
    """Transponiert eine Matrix.

    Formel: (A^T)_ij=A_ji. Input: NumPy-/Listen-/SymPy-Matrix. Output: transponierte Matrix.
    Voraussetzungen: Matrix-artige Eingabe. Hinweis: SymPy-Eingaben bleiben symbolisch.
    Beispiel:
        >>> matrix_transponieren([[1,2],[3,4]]).tolist()
        [[1.0, 3.0], [2.0, 4.0]]
    """
    if isinstance(matrix, sp.MatrixBase):
        return matrix.T
    return np.asarray(matrix, dtype=float).T


def einheitsmatrix(dimension, symbolisch=False):
    """Erzeugt eine Einheitsmatrix.

    Formel: I_ij=1 fuer i=j, sonst 0. Input: Dimension n. Output: n x n Matrix.
    Voraussetzungen: n positiv. Hinweis: symbolisch=True liefert SymPy.
    Beispiel:
        >>> einheitsmatrix(2).tolist()
        [[1.0, 0.0], [0.0, 1.0]]
    """
    if int(dimension) <= 0:
        raise ValueError("Dimension muss positiv sein.")
    return sp.eye(int(dimension)) if symbolisch else np.eye(int(dimension), dtype=float)


def diagonalmatrix(diagonalwerte, symbolisch=False):
    """Erzeugt eine Diagonalmatrix.

    Formel: D_ii=d_i, D_ij=0 fuer i != j. Input: Diagonalwerte. Output: Matrix.
    Voraussetzungen: Werte sind tabellierbar. Hinweis: symbolisch=True erhaelt exakte Werte.
    Beispiel:
        >>> diagonalmatrix([1, 2]).tolist()
        [[1.0, 0.0], [0.0, 2.0]]
    """
    return sp.diag(*diagonalwerte) if symbolisch else np.diag(np.asarray(diagonalwerte, dtype=float).reshape(-1))


def ist_symmetrisch(matrix, toleranz=1e-10) -> bool:
    """Prueft, ob eine Matrix symmetrisch ist.

    Formel: A=A^T. Input: quadratische Matrix. Output: bool.
    Voraussetzungen: quadratisch. Hinweis: Toleranz wegen Rundungsfehlern.
    Beispiel:
        >>> ist_symmetrisch([[1,2],[2,1]])
        True
    """
    a = np.asarray(matrix, dtype=float)
    pruefe_quadratische_matrix(a)
    return bool(np.allclose(a, a.T, atol=toleranz, rtol=0))


def permutationsmatrix(dimension, zeile_i, zeile_j) -> np.ndarray:
    """Erzeugt eine Matrix zum Vertauschen zweier Zeilen.

    Formel: P entsteht aus I durch Zeilentausch. Input: Dimension und zwei Zeilenindizes.
    Output: Permutationsmatrix. Voraussetzungen: Indizes 0-basiert gueltig. Hinweis: P@A tauscht Zeilen.
    Beispiel:
        >>> permutationsmatrix(2, 0, 1).tolist()
        [[0.0, 1.0], [1.0, 0.0]]
    """
    p = np.eye(int(dimension), dtype=float)
    p[[zeile_i, zeile_j]] = p[[zeile_j, zeile_i]]
    return p


def determinante_1x1(matrix) -> float:
    """Berechnet die Determinante einer 1x1-Matrix.

    Formel: det([a])=a. Input: 1x1-Matrix. Output: float.
    Voraussetzungen: Form 1x1. Hinweis: exakt nur bei symbolischer Variante `determinante`.
    Beispiel:
        >>> determinante_1x1([[5]])
        5.0
    """
    a = np.asarray(matrix, dtype=float)
    if a.shape != (1, 1):
        raise ValueError("Matrix muss 1x1 sein.")
    return float(a[0, 0])


def determinante_2x2(matrix) -> float:
    """Berechnet die Determinante einer 2x2-Matrix.

    Formel: det([[a,b],[c,d]]) = ad - bc. Input: 2x2-Matrix. Output: float.
    Voraussetzungen: Form 2x2. Hinweis: Nahe singulaere Matrizen koennen Rundungsfehler zeigen.
    Beispiel:
        >>> determinante_2x2([[1, 2], [3, 4]])
        -2.0
    """
    a = np.asarray(matrix, dtype=float)
    if a.shape != (2, 2):
        raise ValueError("Matrix muss 2x2 sein.")
    return float(a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0])


def determinante_3x3(matrix) -> float:
    """Berechnet die Determinante einer 3x3-Matrix nach Sarrus.

    Formel: aei+bfg+cdh-ceg-bdi-afh. Input: 3x3-Matrix. Output: float.
    Voraussetzungen: Form 3x3. Hinweis: Fuer grosse Matrizen `determinante` nutzen.
    Beispiel:
        >>> determinante_3x3([[1,0,0],[0,1,0],[0,0,1]])
        1.0
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3):
        raise ValueError("Matrix muss 3x3 sein.")
    a, b, c, d, e, f, g, h, i = m.reshape(-1)
    return float(a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h)


def determinante(matrix, symbolisch=False):
    """Berechnet die Determinante einer quadratischen Matrix.

    Formel: det(A) ueber Leibniz/LU intern. Input: quadratische Matrix. Output: float oder SymPy-Ausdruck.
    Voraussetzungen: quadratisch. Hinweis: symbolisch=True fuer exakte Algebra.
    Beispiel:
        >>> determinante([[1, 2], [3, 4]])
        -2.0000000000000004
    """
    if symbolisch:
        return sp.Matrix(matrix).det()
    pruefe_quadratische_matrix(matrix)
    return float(np.linalg.det(np.asarray(matrix, dtype=float)))


def inverse_2x2(matrix) -> np.ndarray:
    """Berechnet die inverse 2x2-Matrix.

    Formel: A^-1 = 1/(ad-bc) [[d,-b],[-c,a]]. Input: 2x2-Matrix. Output: NumPy-Matrix.
    Voraussetzungen: Determinante nicht null. Hinweis: Bei kleiner Determinante instabil.
    Beispiel:
        >>> inverse_2x2([[1, 0], [0, 2]]).tolist()
        [[1.0, -0.0], [-0.0, 0.5]]
    """
    m = np.asarray(matrix, dtype=float)
    det = determinante_2x2(m)
    if abs(det) < 1e-15:
        raise ValueError("Matrix ist singulaer oder nahezu singulaer.")
    a, b, c, d = m.reshape(-1)
    return (1.0 / det) * np.array([[d, -b], [-c, a]], dtype=float)


def matrix_inverse(matrix, symbolisch=False):
    """Berechnet die inverse Matrix.

    Formel: A A^-1 = I. Input: quadratische regulaere Matrix. Output: NumPy- oder SymPy-Matrix.
    Voraussetzungen: Matrix regulaer. Hinweis: Loese LGS meist stabiler als explizite Inverse.
    Beispiel:
        >>> matrix_inverse([[1, 0], [0, 2]]).tolist()
        [[1.0, 0.0], [0.0, 0.5]]
    """
    if symbolisch:
        return sp.Matrix(matrix).inv()
    pruefe_quadratische_matrix(matrix)
    return np.linalg.inv(np.asarray(matrix, dtype=float))


def rang_matrix(matrix, toleranz=1e-10) -> int:
    """Berechnet den numerischen Rang einer Matrix.

    Formel: Anzahl Singulaerwerte groesser Toleranz. Input: Matrix. Output: int.
    Voraussetzungen: endliche Werte. Hinweis: Toleranz beeinflusst Rang bei fast abhaengigen Spalten.
    Beispiel:
        >>> rang_matrix([[1, 2], [2, 4]])
        1
    """
    return int(np.linalg.matrix_rank(np.asarray(matrix, dtype=float), tol=toleranz))


def konditionszahl_matrix(matrix, norm_ord=2) -> float:
    """Berechnet die Konditionszahl einer Matrix.

    Formel: cond(A)=||A||*||A^{-1}||. Input: quadratische Matrix. Output: float.
    Voraussetzungen: Matrix regulaer. Hinweis: Grosse Konditionszahl bedeutet empfindliches LGS.
    Beispiel:
        >>> konditionszahl_matrix([[1, 0], [0, 1]])
        1.0
    """
    pruefe_quadratische_matrix(matrix)
    return float(np.linalg.cond(np.asarray(matrix, dtype=float), p=norm_ord))


def loese_lgs(matrix_a, vektor_b, methode="numpy") -> np.ndarray:
    """Loest ein lineares Gleichungssystem A x = b.

    Formel: A x = b, intern NumPy oder SymPy-LU. Input: A n x n, b n. Output: Loesungsvektor.
    Voraussetzungen: A quadratisch und regulaer. Hinweis: QR kann fuer Ausgleichsprobleme stabiler sein.
    Beispiel:
        >>> loese_lgs([[2,0],[0,2]], [4,6]).tolist()
        [2.0, 3.0]
    """
    pruefe_dimensionen_matrix_vektor(matrix_a, vektor_b)
    pruefe_quadratische_matrix(matrix_a)
    if methode == "sympy":
        return np.array(sp.Matrix(matrix_a).LUsolve(sp.Matrix(vektor_b)), dtype=float).reshape(-1)
    if methode != "numpy":
        raise ValueError("methode muss 'numpy' oder 'sympy' sein.")
    return np.linalg.solve(np.asarray(matrix_a, dtype=float), np.asarray(vektor_b, dtype=float).reshape(-1))


def loese_lgs_2x2(matrix_a, vektor_b) -> np.ndarray:
    """Loest ein 2x2-LGS.

    Formel: x=A^-1 b mit 2x2-Inversenformel. Input: A 2x2, b 2. Output: Vektor.
    Voraussetzungen: det(A) != 0. Hinweis: Demonstrationsfreundlich, fuer Praxis `loese_lgs`.
    Beispiel:
        >>> loese_lgs_2x2([[2,0],[0,2]], [4,6]).tolist()
        [2.0, 3.0]
    """
    a = np.asarray(matrix_a, dtype=float)
    b = np.asarray(vektor_b, dtype=float).reshape(-1)
    if a.shape != (2, 2) or b.size != 2:
        raise ValueError("Erwartet A 2x2 und b mit Laenge 2.")
    return inverse_2x2(a) @ b


def loese_lgs_3x3(matrix_a, vektor_b) -> np.ndarray:
    """Loest ein 3x3-LGS.

    Formel: A x=b per numerischem Solver. Input: A 3x3, b 3. Output: Vektor.
    Voraussetzungen: A regulaer. Hinweis: nutzt denselben stabilen Solver wie `loese_lgs`.
    Beispiel:
        >>> loese_lgs_3x3(np.eye(3), [1,2,3]).tolist()
        [1.0, 2.0, 3.0]
    """
    a = np.asarray(matrix_a, dtype=float)
    b = np.asarray(vektor_b, dtype=float).reshape(-1)
    if a.shape != (3, 3) or b.size != 3:
        raise ValueError("Erwartet A 3x3 und b mit Laenge 3.")
    return loese_lgs(a, b)


def gauss_elimination(matrix_a, vektor_b=None, pivotisierung=True) -> dict:
    """Fuehrt Gauss-Elimination aus.

    Formel: Zeilenoperationen erzeugen obere Dreiecksform. Input: A und optional b. Output: dict.
    Voraussetzungen: kompatible Dimensionen. Hinweis: Pivotisierung verbessert Stabilitaet.
    Beispiel:
        >>> gauss_elimination([[1, 2], [3, 4]])["stufenmatrix"].shape
        (2, 2)
    """
    a = np.asarray(matrix_a, dtype=float).copy()
    b = None if vektor_b is None else np.asarray(vektor_b, dtype=float).reshape(-1).copy()
    if b is not None and b.size != a.shape[0]:
        raise ValueError("b passt nicht zu A.")
    zeilentausche = []
    m, n = a.shape
    for k in range(min(m, n)):
        if pivotisierung:
            pivot = k + int(np.argmax(np.abs(a[k:, k])))
            if pivot != k:
                a[[k, pivot]] = a[[pivot, k]]
                if b is not None:
                    b[[k, pivot]] = b[[pivot, k]]
                zeilentausche.append((k, pivot))
        if abs(a[k, k]) < 1e-15:
            continue
        for i in range(k + 1, m):
            faktor = a[i, k] / a[k, k]
            # Subtrahiere das passende Vielfache der Pivotzeile.
            a[i, k:] -= faktor * a[k, k:]
            if b is not None:
                b[i] -= faktor * b[k]
    return {"stufenmatrix": a, "rechte_seite": b, "zeilentausche": zeilentausche}


def zeilenstufenform(matrix, symbolisch=False):
    """Berechnet eine Zeilenstufenform.

    Formel: Elementare Zeilenoperationen. Input: Matrix. Output: Matrix in Stufenform.
    Voraussetzungen: Matrix-artige Eingabe. Hinweis: symbolisch=True liefert exakte SymPy-Form.
    Beispiel:
        >>> zeilenstufenform([[1,2],[2,4]]).shape
        (2, 2)
    """
    if symbolisch:
        return sp.Matrix(matrix).echelon_form()
    return gauss_elimination(matrix)["stufenmatrix"]


def reduzierte_zeilenstufenform(matrix, symbolisch=True):
    """Berechnet die reduzierte Zeilenstufenform.

    Formel: RREF mit normierten Pivots. Input: Matrix. Output: RREF und Pivotspalten bei SymPy.
    Voraussetzungen: Matrix-artige Eingabe. Hinweis: Gut fuer Parameterdarstellungen.
    Beispiel:
        >>> reduzierte_zeilenstufenform([[1,2],[2,4]])[1]
        (0,)
    """
    if symbolisch:
        return sp.Matrix(matrix).rref()
    rref, _ = sp.Matrix(np.asarray(matrix, dtype=float)).rref()
    return np.array(rref, dtype=float)


def lgs_loesbarkeit_pruefen(matrix_a, vektor_b, toleranz=1e-10) -> dict:
    """Prueft Loesbarkeit eines LGS.

    Formel: Rang(A)=Rang([A|b]) fuer mindestens eine Loesung. Input: A,b. Output: dict.
    Voraussetzungen: Dimensionen passen. Hinweis: Rang mit Toleranz ist numerisch.
    Beispiel:
        >>> lgs_loesbarkeit_pruefen([[1]], [2])["eindeutig"]
        True
    """
    a = np.asarray(matrix_a, dtype=float)
    b = np.asarray(vektor_b, dtype=float).reshape(-1)
    pruefe_dimensionen_matrix_vektor(a, b)
    rang_a = rang_matrix(a, toleranz)
    rang_erweitert = rang_matrix(np.column_stack([a, b]), toleranz)
    return {
        "rang_a": rang_a,
        "rang_erweitert": rang_erweitert,
        "loesbar": rang_a == rang_erweitert,
        "eindeutig": rang_a == rang_erweitert == a.shape[1],
        "anzahl_unbekannte": a.shape[1],
    }


def parameterdarstellung_lgs(matrix_a, vektor_b) -> dict:
    """Berechnet eine symbolische Parameterdarstellung eines LGS.

    Formel: SymPy `linsolve` fuer A x=b. Input: A,b. Output: dict mit Loesungsmenge.
    Voraussetzungen: Dimensionen passen. Hinweis: Freie Parameter erscheinen als tau-Variablen.
    Beispiel:
        >>> "loesungsmenge" in parameterdarstellung_lgs([[1, 1]], [2])
        True
    """
    a = sp.Matrix(matrix_a)
    b = sp.Matrix(vektor_b)
    variablen = sp.symbols(f"x0:{a.shape[1]}")
    return {"variablen": variablen, "loesungsmenge": sp.linsolve((a, b), variablen)}


def homogenes_lgs_loesen(matrix_a) -> dict:
    """Loest das homogene LGS A x = 0.

    Formel: Nullraum N(A). Input: Matrix A. Output: Basis des Nullraums.
    Voraussetzungen: Matrix-artige Eingabe. Hinweis: Nichttriviale Loesungen bei Rang < n.
    Beispiel:
        >>> len(homogenes_lgs_loesen([[1, 1]])["basis"])
        1
    """
    a = sp.Matrix(matrix_a)
    return {"basis": a.nullspace(), "dimension": len(a.nullspace())}


def qr_zerlegung(matrix_a, modus="complete") -> tuple[np.ndarray, np.ndarray]:
    """Berechnet die QR-Zerlegung.

    Formel: A=QR, Q^T Q=I, R rechtsobere Dreiecksmatrix. Input: Matrix A. Output: (Q,R).
    Voraussetzungen: Matrix numerisch. Hinweis: Stabiler als Normalgleichungen fuer Ausgleich.
    Beispiel:
        >>> A = np.array([[1, 0], [0, 1]], dtype=float)
        >>> q, r = qr_zerlegung(A)
        >>> q.shape
        (2, 2)
    """
    mode = "complete" if modus == "complete" else "reduced"
    return np.linalg.qr(np.asarray(matrix_a, dtype=float), mode=mode)


def qr_zerlegung_reduziert(matrix_a) -> tuple[np.ndarray, np.ndarray]:
    """Berechnet die reduzierte QR-Zerlegung.

    Formel: A=Q_R R_R mit Q_R^T Q_R=I. Input: A m x n. Output: reduziertes (Q,R).
    Voraussetzungen: numerische Matrix. Hinweis: Fuer Ausgleichsrechnung speicherschonend.
    Beispiel:
        >>> qr_zerlegung_reduziert([[1], [2]])[0].shape
        (2, 1)
    """
    return np.linalg.qr(np.asarray(matrix_a, dtype=float), mode="reduced")


def householder_vektor(vektor) -> np.ndarray:
    """Berechnet den Householder-Vektor.

    Formel: v = x + sign(x_1)||x|| e_1, normiert. Input: Vektor x. Output: v.
    Voraussetzungen: x nicht Nullvektor. Hinweis: Vermeidet Ausloeschung durch Vorzeichenwahl.
    Beispiel:
        >>> round(vektor_norm_2(householder_vektor([1, 2])), 6)
        1.0
    """
    x = np.asarray(vektor, dtype=float).reshape(-1)
    norm = vektor_norm_2(x)
    if norm == 0:
        raise ValueError("Householder fuer Nullvektor nicht definiert.")
    e1 = np.zeros_like(x)
    e1[0] = 1.0
    vorzeichen = 1.0 if x[0] >= 0 else -1.0
    v = x + vorzeichen * norm * e1
    return v / np.linalg.norm(v)


def householder_matrix(vektor) -> np.ndarray:
    """Berechnet die Householder-Spiegelungsmatrix.

    Formel: H = I - 2 v v^T. Input: Householder-Richtungsvektor. Output: H.
    Voraussetzungen: v nicht Null. Hinweis: H ist orthogonal und symmetrisch.
    Beispiel:
        >>> householder_matrix([1, 0]).tolist()
        [[-1.0, 0.0], [0.0, 1.0]]
    """
    v = einheitsvektor(vektor)
    return np.eye(v.size) - 2.0 * np.outer(v, v)


def qr_zerlegung_householder(matrix_a) -> tuple[np.ndarray, np.ndarray]:
    """Berechnet QR mit Householder-Spiegelungen.

    Formel: A=QR durch sukzessive H_k, Q^T Q=I, R obere Dreiecksmatrix.
    Input: Matrix A. Output: (Q,R). Voraussetzungen: numerische Matrix.
    Numerischer Hinweis: Householder ist stabiler als klassisches Gram-Schmidt.
    Beispiel:
        >>> matrix_a = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> q, r = qr_zerlegung_householder(matrix_a)
        >>> r.shape
        (2, 2)
    """
    a = np.asarray(matrix_a, dtype=float)
    m, n = a.shape
    r = a.copy()
    q = np.eye(m)
    for k in range(min(m - 1, n)):
        x = r[k:, k]
        if np.linalg.norm(x) < 1e-15:
            continue
        v = householder_vektor(x)
        h_klein = np.eye(m - k) - 2.0 * np.outer(v, v)
        h = np.eye(m)
        h[k:, k:] = h_klein
        r = h @ r
        q = q @ h.T
    return q, r


def loese_lgs_qr(matrix_a, vektor_b) -> np.ndarray:
    """Loest A x=b ueber QR.

    Formel: A=QR, R x=Q^T b. Input: quadratische A und b. Output: x.
    Voraussetzungen: A regulaer. Hinweis: Orthogonale Q veraendern die 2-Norm nicht.
    Beispiel:
        >>> loese_lgs_qr([[2,0],[0,2]], [4,6]).tolist()
        [2.0, 3.0]
    """
    a = np.asarray(matrix_a, dtype=float)
    b = np.asarray(vektor_b, dtype=float).reshape(-1)
    pruefe_dimensionen_matrix_vektor(a, b)
    q, r = qr_zerlegung_reduziert(a)
    return np.linalg.solve(r, q.T @ b)


def loese_ausgleich_qr(matrix_a, vektor_y) -> np.ndarray:
    """Loest ein lineares Ausgleichsproblem per QR.

    Formel: min ||A lambda-y||_2, QR: R lambda=Q^T y. Input: Designmatrix A, y.
    Output: Koeffizienten. Voraussetzungen: voller Spaltenrang empfohlen.
    Numerischer Hinweis: QR ist stabiler als Normalgleichungen A^T A.
    Beispiel:
        >>> loese_ausgleich_qr([[1,0],[1,1],[1,2]], [1,2,3]).round(6).tolist()
        [1.0, 1.0]
    """
    a = np.asarray(matrix_a, dtype=float)
    y = np.asarray(vektor_y, dtype=float).reshape(-1)
    pruefe_dimensionen_matrix_vektor(a, y)
    q, r = qr_zerlegung_reduziert(a)
    return np.linalg.solve(r, q.T @ y)

# ===== nichtlineare_gleichungen.py =====
"""Nichtlineare Gleichungen, Jacobi-Matrizen, Linearisierung und Newton-Verfahren."""





def funktionswert_vektor(funktion, punkt) -> np.ndarray:
    """Wertet eine vektorwertige Funktion aus.

    Zweck: Einheitliche numerische Auswertung f(x).
    Formel/Verfahren: Funktionsaufruf und Konvertierung zu NumPy-Vektor.
    Input: funktion callable, punkt mit Dimension n.
    Output: NumPy-Vektor mit Dimension m.
    Voraussetzungen: Funktion liefert numerische Werte.
    Numerische Hinweise: Nicht-endliche Werte werden von `als_vektor` abgefangen.
    Beispiel:
        >>> def funktion(x):
        ...     return [x[0] + 1]
        >>> punkt = np.array([2.0])
        >>> funktionswert_vektor(funktion, punkt).tolist()
        [3.0]
    """
    punkt_array = np.asarray(punkt, dtype=float).reshape(-1)
    return np.asarray(funktion(punkt_array), dtype=float).reshape(-1)


def jacobi_matrix_symbolisch(funktionen, variablen) -> sp.Matrix:
    """Berechnet die symbolische Jacobi-Matrix.

    Formel: J_ij = d f_i / d x_j.
    Input: Liste von Funktionen und Variablen; eine einzelne skalare Funktion wird als [f] uebergeben.
    Output: SymPy-Matrix.
    Voraussetzungen: SymPy-Ausdruecke.
    Numerische Hinweise: Symbolische Ableitungen vermeiden Differenzenfehler.
    Beispiel:
        >>> x, y = sp.symbols('x y')
        >>> f = x*y
        >>> jacobi_matrix_symbolisch([f], [x, y])
        Matrix([[y, x]])
    """
    return sp.Matrix(funktionen).jacobian(list(variablen))


def jacobi_matrix_auswerten(jacobi_matrix, variablen, punkt) -> np.ndarray:
    """Wertet eine symbolische Jacobi-Matrix numerisch aus.

    Formel: J(x_0) durch Einsetzen der Koordinaten.
    Input: SymPy-Matrix, Variablen, Punkt.
    Output: NumPy-Matrix.
    Voraussetzungen: Laenge von Variablen und Punkt passt.
    Numerische Hinweise: Ergebnis wird zu float konvertiert.
    Beispiel:
        >>> x = sp.symbols('x')
        >>> J = sp.Matrix([[2*x]])
        >>> jacobi_matrix_auswerten(J, [x], [3]).tolist()
        [[6.0]]
    """
    subs = dict(zip(variablen, np.asarray(punkt, dtype=float).reshape(-1)))
    return np.array(sp.Matrix(jacobi_matrix).subs(subs), dtype=float)


def jacobi_matrix_numerisch(funktion, punkt, schrittweite=1e-6, methode="zentral") -> np.ndarray:
    """Berechnet eine numerische Jacobi-Matrix.

    Formel: vorwaerts (f(x+h e_j)-f(x))/h oder zentral
    (f(x+h e_j)-f(x-h e_j))/(2h).
    Input: Funktion R^n->R^m, Punkt, Schrittweite, Methode.
    Output: Matrix m x n.
    Voraussetzungen: Schrittweite positiv, Funktion glatt genug.
    Numerische Hinweise: Zu kleine Schrittweite fuehrt zu Ausloeschung.
    Beispiel:
        >>> def funktion(x):
        ...     return [x[0]**2]
        >>> punkt = np.array([3.0])
        >>> jacobi_matrix_numerisch(funktion, punkt).round(4).tolist()
        [[6.0]]
    """
    x = np.asarray(punkt, dtype=float).reshape(-1)
    h = float(schrittweite)
    if h <= 0:
        raise ValueError("schrittweite muss positiv sein.")
    f0 = funktionswert_vektor(funktion, x)
    jacobi = np.zeros((f0.size, x.size), dtype=float)
    for j in range(x.size):
        einheit = np.zeros_like(x)
        einheit[j] = 1.0
        if methode == "vorwaerts":
            jacobi[:, j] = (funktionswert_vektor(funktion, x + h * einheit) - f0) / h
        elif methode == "zentral":
            jacobi[:, j] = (
                funktionswert_vektor(funktion, x + h * einheit)
                - funktionswert_vektor(funktion, x - h * einheit)
            ) / (2.0 * h)
        else:
            raise ValueError("methode muss 'zentral' oder 'vorwaerts' sein.")
    return jacobi


def linearisierung(funktion, jacobi, entwicklungs_punkt):
    """Erzeugt die Linearisierung einer vektorwertigen Funktion.

    Formel: L(x)=f(x0)+J(x0)(x-x0).
    Input: Funktion, Jacobi callable oder Matrix, Entwicklungspunkt.
    Output: callable fuer L(x).
    Voraussetzungen: Jacobi passt zur Funktion.
    Numerische Hinweise: Nur lokal nahe x0 verlaesslich.
    Beispiel:
        >>> def funktion(punkt):
        ...     return [punkt[0]**2]
        >>> def jacobi(punkt):
        ...     return [[2*punkt[0]]]
        >>> entwicklungs_punkt = np.array([2.0])
        >>> L = linearisierung(funktion, jacobi, entwicklungs_punkt)
        >>> L([2.1]).round(6).tolist()
        [4.4]
    """
    x0 = np.asarray(entwicklungs_punkt, dtype=float).reshape(-1)
    f0 = funktionswert_vektor(funktion, x0)
    j0 = np.asarray(jacobi(x0) if callable(jacobi) else jacobi, dtype=float)
    def linearisierte_funktion(punkt):
        """Wertet die Linearisierung an einem Punkt aus."""
        return f0 + j0 @ (np.asarray(punkt, dtype=float).reshape(-1) - x0)

    return linearisierte_funktion


def tangentialebene_symbolisch(funktion, variablen, punkt) -> sp.Expr:
    """Berechnet die symbolische Tangentialebene.

    Formel: T=f(p)+grad f(p)^T (x-p).
    Input: SymPy-Ausdruck, Variablen, Punkt.
    Output: SymPy-Ausdruck.
    Voraussetzungen: Dimensionen passen.
    Numerische Hinweise: Symbolisch exakt bis zur spaeteren Auswertung.
    Beispiel:
        >>> x, y = sp.symbols('x y')
        >>> f = x**2 + y
        >>> tangentialebene_symbolisch(f, [x, y], [1, 2])
        2*x + y - 1
    """
    p = list(punkt)
    subs = dict(zip(variablen, p))
    wert = funktion.subs(subs)
    ausdruck = wert
    for variable, koordinate in zip(variablen, p):
        ausdruck += sp.diff(funktion, variable).subs(subs) * (variable - koordinate)
    return sp.expand(ausdruck)


def tangentialebene(funktion, gradient, punkt):
    """Erzeugt eine Tangentialebene fuer z=f(x,y).

    Formel: T(x)=f(p)+grad f(p) dot (x-p).
    Input: skalare Funktion, Gradient callable oder Vektor, Punkt.
    Output: callable fuer die Ebene.
    Voraussetzungen: Gradient hat gleiche Dimension wie Punkt.
    Numerische Hinweise: Lokale lineare Naeherung.
    Beispiel:
        >>> def funktion(punkt):
        ...     return punkt[0]**2 + punkt[1]
        >>> def gradient(punkt):
        ...     return [2*punkt[0], 1]
        >>> punkt = np.array([1.0, 2.0])
        >>> T = tangentialebene(funktion, gradient, punkt)
        >>> T([1, 2])
        3.0
    """
    p = np.asarray(punkt, dtype=float).reshape(-1)
    f0 = float(funktion(p))
    g = np.asarray(gradient(p) if callable(gradient) else gradient, dtype=float).reshape(-1)
    def ebene(x):
        """Wertet die Tangentialebene an einem Punkt aus."""
        return float(f0 + np.dot(g, np.asarray(x, dtype=float).reshape(-1) - p))

    return ebene


def _newton_resultat(loesung, iterationen, konvergiert, residualnorm, schrittnorm, verlauf, nachricht, rueckgabe_verlauf):
    """Baut ein IterationsErgebnis und blendet den Verlauf optional aus."""
    return IterationsErgebnis(
        loesung=loesung,
        iterationen=iterationen,
        konvergiert=konvergiert,
        residualnorm=residualnorm,
        schrittnorm=schrittnorm,
        verlauf=verlauf if rueckgabe_verlauf else None,
        nachricht=nachricht,
    )


def newton_verfahren(funktion, jacobi, startwert, toleranz=1e-10, maximale_iterationen=50, rueckgabe_verlauf=False) -> IterationsErgebnis:
    """Loest ein nichtlineares Gleichungssystem mit Newton.

    Formel: Df(x_k) delta_k=-f(x_k), x_{k+1}=x_k+delta_k.
    Input: Funktion, Jacobi callable, Startvektor, Toleranz, Iterationslimit.
    Output: IterationsErgebnis.
    Voraussetzungen: Jacobi quadratisch und regulaer nahe der Loesung.
    Numerische Hinweise: Konvergenz haengt stark vom Startwert ab.
    Beispiel:
        >>> x, y = sp.symbols("x y")
        >>> f1 = x**2 + y**2 - 4
        >>> f2 = x - y
        >>> f = sp.Matrix([f1, f2])
        >>> variablen = (x, y)
        >>> Df = f.jacobian(variablen)
        >>> funktion = sp.lambdify([variablen], f, modules="numpy")
        >>> jacobi = sp.lambdify([variablen], Df, modules="numpy")
        >>> x0 = np.array([1.5, 1.0])
        >>> r = newton_verfahren(funktion, jacobi, x0)
        >>> np.round(r.loesung, 6).tolist()
        [1.414214, 1.414214]
    """
    x = np.array(startwert, dtype=float)
    verlauf = []
    for k in range(maximale_iterationen):
        fwert = np.array(funktion(x), dtype=float).reshape(-1)
        jwert = np.asarray(jacobi(x), dtype=float)
        delta = np.linalg.solve(jwert, -fwert)
        schrittnorm = float(np.linalg.norm(delta))
        residualnorm = float(np.linalg.norm(fwert))
        verlauf.append({"iteration": k, "x": x.copy(), "funktionswert": fwert.copy(), "residualnorm": residualnorm, "delta": delta.copy(), "schrittnorm": schrittnorm, "p": 0, "faktor": 1.0})
        x = x + delta
        residual_neu = float(np.linalg.norm(np.array(funktion(x), dtype=float).reshape(-1)))
        if schrittnorm <= toleranz or residual_neu <= toleranz:
            return _newton_resultat(x, k + 1, True, residual_neu, schrittnorm, verlauf, "konvergiert", rueckgabe_verlauf)
    residual_final = float(np.linalg.norm(np.array(funktion(x), dtype=float).reshape(-1)))
    return _newton_resultat(x, maximale_iterationen, False, residual_final, schrittnorm, verlauf, "maximale Iterationen erreicht", rueckgabe_verlauf)


def vereinfachtes_newton_verfahren(funktion, jacobi, startwert, toleranz=1e-10, maximale_iterationen=50, rueckgabe_verlauf=False) -> IterationsErgebnis:
    """Loest ein NGS mit vereinfachtem Newton-Verfahren.

    Formel: Df(x_0) delta_k=-f(x_k), x_{k+1}=x_k+delta_k.
    Input: Funktion, Jacobi, Startwert. Output: IterationsErgebnis.
    Voraussetzungen: Df(x0) regulaer. Numerische Hinweise: Pro Iteration billiger, oft langsamer.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2 - 4
        >>> funktionen = [f_expr]
        >>> variablen = [x]
        >>> J_expr = jacobi_matrix_symbolisch(funktionen, variablen)
        >>> funktion = sp.lambdify([variablen], sp.Matrix(funktionen), modules="numpy")
        >>> jacobi = sp.lambdify([variablen], J_expr, modules="numpy")
        >>> x0 = [3]
        >>> r = vereinfachtes_newton_verfahren(funktion, jacobi, x0)
        >>> r.konvergiert
        True
    """
    x = np.asarray(startwert, dtype=float).reshape(-1)
    j0 = np.asarray(jacobi(x), dtype=float)
    verlauf = []
    schrittnorm = None
    for k in range(maximale_iterationen):
        fwert = funktionswert_vektor(funktion, x)
        delta = np.linalg.solve(j0, -fwert)
        schrittnorm = float(np.linalg.norm(delta))
        residualnorm = float(np.linalg.norm(fwert))
        verlauf.append({"iteration": k, "x": x.copy(), "funktionswert": fwert.copy(), "residualnorm": residualnorm, "delta": delta.copy(), "schrittnorm": schrittnorm, "p": 0, "faktor": 1.0})
        x = x + delta
        if schrittnorm <= toleranz or np.linalg.norm(funktionswert_vektor(funktion, x)) <= toleranz:
            return _newton_resultat(x, k + 1, True, float(np.linalg.norm(funktionswert_vektor(funktion, x))), schrittnorm, verlauf, "konvergiert", rueckgabe_verlauf)
    return _newton_resultat(x, maximale_iterationen, False, float(np.linalg.norm(funktionswert_vektor(funktion, x))), schrittnorm, verlauf, "maximale Iterationen erreicht", rueckgabe_verlauf)


def gedaempftes_newton_verfahren(funktion, jacobi, startwert, toleranz=1e-10, maximale_iterationen=50, p_max=20, rueckgabe_verlauf=False) -> IterationsErgebnis:
    """Loest ein NGS mit gedaempftem Newton-Verfahren.

    Formel: x_{k+1}=x_k+delta_k/2^p mit kleinstem p, das die Residualnorm senkt.
    Input: Funktion, Jacobi, Startwert, p_max. Output: IterationsErgebnis.
    Voraussetzungen: Jacobi regulaer. Hinweis: Daempfung erhoeht Robustheit bei schlechten Startwerten.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2 - 4
        >>> funktionen = [f_expr]
        >>> variablen = [x]
        >>> J_expr = jacobi_matrix_symbolisch(funktionen, variablen)
        >>> funktion = sp.lambdify([variablen], sp.Matrix(funktionen), modules="numpy")
        >>> jacobi = sp.lambdify([variablen], J_expr, modules="numpy")
        >>> x0 = [3]
        >>> r = gedaempftes_newton_verfahren(funktion, jacobi, x0)
        >>> round(r.loesung[0], 6)
        2.0
    """
    x = np.asarray(startwert, dtype=float).reshape(-1)
    verlauf = []
    schrittnorm = None
    for k in range(maximale_iterationen):
        fwert = funktionswert_vektor(funktion, x)
        residual_alt = float(np.linalg.norm(fwert))
        delta = np.linalg.solve(np.asarray(jacobi(x), dtype=float), -fwert)
        p_gewaehlt = p_max
        faktor = 2.0 ** (-p_max)
        for p in range(p_max + 1):
            kandidat = x + delta / (2.0 ** p)
            if np.linalg.norm(funktionswert_vektor(funktion, kandidat)) < residual_alt:
                p_gewaehlt = p
                faktor = 2.0 ** (-p)
                break
        schritt = faktor * delta
        schrittnorm = float(np.linalg.norm(schritt))
        verlauf.append({"iteration": k, "x": x.copy(), "funktionswert": fwert.copy(), "residualnorm": residual_alt, "delta": delta.copy(), "schrittnorm": schrittnorm, "p": p_gewaehlt, "faktor": faktor})
        x = x + schritt
        residual_neu = float(np.linalg.norm(funktionswert_vektor(funktion, x)))
        if schrittnorm <= toleranz or residual_neu <= toleranz:
            return _newton_resultat(x, k + 1, True, residual_neu, schrittnorm, verlauf, "konvergiert", rueckgabe_verlauf)
    return _newton_resultat(x, maximale_iterationen, False, float(np.linalg.norm(funktionswert_vektor(funktion, x))), schrittnorm, verlauf, "maximale Iterationen erreicht", rueckgabe_verlauf)


def newton_verfahren_2d(funktion_1, funktion_2, jacobi_eintraege, startwert, **optionen) -> IterationsErgebnis:
    """Komfort-Wrapper fuer Newton in zwei Variablen.

    Formel: 2x2-Jacobi in Df(x) delta=-f(x). Input: zwei Funktionen und vier Jacobi-Eintraege.
    Output: IterationsErgebnis. Voraussetzungen: Startwert Laenge 2.
    Numerische Hinweise: Praktisch fuer handschriftliche HM2-Systeme.
    Beispiel:
        >>> def funktion_1(x, y):
        ...     return x**2 - 4
        >>> def funktion_2(x, y):
        ...     return y - 1
        >>> def j11(x, y):
        ...     return 2*x
        >>> def j12(x, y):
        ...     return 0
        >>> def j21(x, y):
        ...     return 0
        >>> def j22(x, y):
        ...     return 1
        >>> jacobi_eintraege = [j11, j12, j21, j22]
        >>> x0 = [3, 0]
        >>> r = newton_verfahren_2d(funktion_1, funktion_2, jacobi_eintraege, x0)
        >>> round(r.loesung[0], 6)
        2.0
    """
    def f(p):
        """Wertet das zweidimensionale Gleichungssystem als Vektor aus."""
        return [funktion_1(p[0], p[1]), funktion_2(p[0], p[1])]

    def j(p):
        """Wertet die vier uebergebenen Jacobi-Eintraege als 2x2-Matrix aus."""
        j11, j12, j21, j22 = jacobi_eintraege
        return [[j11(p[0], p[1]), j12(p[0], p[1])], [j21(p[0], p[1]), j22(p[0], p[1])]]

    return newton_verfahren(f, j, startwert, **optionen)


def newton_verfahren_skalar(funktion, ableitung, startwert, toleranz=1e-10, maximale_iterationen=50) -> IterationsErgebnis:
    """Loest eine skalare Gleichung mit Newton.

    Formel: x_{k+1}=x_k-f(x_k)/f'(x_k). Input: f, f', Startwert. Output: IterationsErgebnis.
    Voraussetzungen: Ableitung nicht null nahe Iterationen. Hinweis: Quadratische Konvergenz nahe einfacher Nullstelle.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2 - 4
        >>> df_expr = ableitung_symbolisch(f_expr, x)
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> ableitung = sp.lambdify(x, df_expr, modules="numpy")
        >>> x0 = 3
        >>> r = newton_verfahren_skalar(funktion, ableitung, x0)
        >>> round(r.loesung, 6)
        2.0
    """
    verlauf = []
    x = float(startwert)
    schrittnorm = None
    for k in range(maximale_iterationen):
        fwert = float(funktion(x))
        awert = float(ableitung(x))
        if awert == 0:
            raise ValueError("Ableitung ist null.")
        delta = -fwert / awert
        schrittnorm = abs(delta)
        verlauf.append({"iteration": k, "x": x, "funktionswert": fwert, "residualnorm": abs(fwert), "delta": delta, "schrittnorm": schrittnorm})
        x += delta
        if schrittnorm <= toleranz or abs(funktion(x)) <= toleranz:
            return IterationsErgebnis(x, k + 1, True, abs(float(funktion(x))), schrittnorm, verlauf, "konvergiert")
    return IterationsErgebnis(x, maximale_iterationen, False, abs(float(funktion(x))), schrittnorm, verlauf, "maximale Iterationen erreicht")


def nullstellenfehler_abschaetzen(funktion, naeherung, epsilon) -> dict:
    """Schaetzt einen Nullstellenfehler ueber Vorzeichenwechsel ab.

    Formel: f(x_n-eps)*f(x_n+eps)<0 => |x_n-xi|<eps.
    Input: Funktion, Naeherung, epsilon. Output: dict.
    Voraussetzungen: stetige Funktion im Intervall. Hinweis: Kein Vorzeichenwechsel bedeutet nicht keine Nullstelle.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2 - 4
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> naeherung = 2.1
        >>> epsilon = 0.2
        >>> nullstellenfehler_abschaetzen(funktion, naeherung, epsilon)["garantiert"]
        True
    """
    links = float(naeherung) - float(epsilon)
    rechts = float(naeherung) + float(epsilon)
    produkt = float(funktion(links)) * float(funktion(rechts))
    return {"intervall": (links, rechts), "produkt": produkt, "garantiert": produkt < 0, "fehlergrenze": float(epsilon) if produkt < 0 else None}


def vorzeichenwechsel_pruefen(funktion, links, rechts) -> bool:
    """Prueft einen Vorzeichenwechsel auf einem Intervall.

    Formel: f(a)*f(b)<0. Input: Funktion, links, rechts. Output: bool.
    Voraussetzungen: Funktion an Randpunkten definiert. Hinweis: Sichert eine Nullstelle bei Stetigkeit.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2 - 4
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> links = 1
        >>> rechts = 3
        >>> vorzeichenwechsel_pruefen(funktion, links, rechts)
        True
    """
    return float(funktion(links)) * float(funktion(rechts)) < 0

# ===== ausgleichsrechnung.py =====
"""Lineare und nichtlineare Ausgleichsrechnung mit deutschen Wrappern."""





def designmatrix_erstellen(stuetzstellen, basisfunktionen) -> np.ndarray:
    """Erstellt die Designmatrix der linearen Ausgleichsrechnung.

    Formel: A_ij = f_j(x_i). Input: Stuetzstellen und Basisfunktionen. Output: Matrix.
    Voraussetzungen: Basisfunktionen sind fuer alle x definiert. Hinweis: Spalten sollten nicht fast linear abhaengig sein.
    Beispiel:
        >>> stuetzstellen = np.array([0.0, 1.0])
        >>> def basis_0(x):
        ...     return 1
        >>> def basis_1(x):
        ...     return x**2
        >>> basisfunktionen = [basis_0, basis_1]
        >>> designmatrix_erstellen(stuetzstellen, basisfunktionen).tolist()
        [[1.0, 0.0], [1.0, 1.0]]
    """
    x = np.asarray(stuetzstellen, dtype=float).reshape(-1)
    return np.array([[funktion(xi) for funktion in basisfunktionen] for xi in x], dtype=float)


def residuen_berechnen(matrix_a, koeffizienten, vektor_y) -> np.ndarray:
    """Berechnet Residuen y-A lambda.

    Formel: r = y - A lambda. Input: Designmatrix, Koeffizienten, y. Output: Residuenvektor.
    Voraussetzungen: Dimensionen passen. Hinweis: Vorzeichenkonvention hier Messwert minus Modellwert.
    Beispiel:
        >>> residuen_berechnen([[1]], [2], [3]).tolist()
        [1.0]
    """
    return np.asarray(vektor_y, dtype=float).reshape(-1) - np.asarray(matrix_a, dtype=float) @ np.asarray(koeffizienten, dtype=float).reshape(-1)


def fehlerfunktional_berechnen(residuen) -> float:
    """Berechnet das Fehlerfunktional.

    Formel: E=||r||_2^2=sum_i r_i^2. Input: Residuen. Output: float.
    Voraussetzungen: Residuen endlich. Hinweis: Kleine Werte bedeuten gute Anpassung.
    Beispiel:
        >>> fehlerfunktional_berechnen([1, 2])
        5.0
    """
    r = np.asarray(residuen, dtype=float).reshape(-1)
    return float(np.dot(r, r))


def r_quadrat_berechnen(stuetzwerte, prognosewerte) -> float:
    """Berechnet das Bestimmtheitsmass R^2.

    Formel: R^2=1-SS_res/SS_tot. Input: Messwerte und Prognosen. Output: float.
    Voraussetzungen: gleiche Laenge. Hinweis: Kann bei schlechten Modellen negativ sein.
    Beispiel:
        >>> r_quadrat_berechnen([1, 2], [1, 2])
        1.0
    """
    y = np.asarray(stuetzwerte, dtype=float).reshape(-1)
    p = np.asarray(prognosewerte, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("stuetzwerte und prognosewerte muessen gleich lang sein.")
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else float("nan")


def residuentabelle_erstellen(stuetzstellen, stuetzwerte, prognosewerte) -> pd.DataFrame:
    """Erstellt eine Residuentabelle.

    Formel: residuum = y_i - yhat_i. Input: x, y, yhat. Output: DataFrame.
    Voraussetzungen: gleiche Laengen. Hinweis: Residuenmuster zeigen Modellfehler.
    Imports: import pandas as pd.
    Beispiel:
        >>> residuentabelle_erstellen([0], [1], [0.5])["residuum"].iloc[0]
        0.5
    """
    x = np.asarray(stuetzstellen, dtype=float).reshape(-1)
    y = np.asarray(stuetzwerte, dtype=float).reshape(-1)
    p = np.asarray(prognosewerte, dtype=float).reshape(-1)
    if not (len(x) == len(y) == len(p)):
        raise ValueError("stuetzstellen, stuetzwerte und prognosewerte muessen gleich lang sein.")
    return pd.DataFrame({"x": x, "messwert": y, "modellwert": p, "residuum": y - p})


def _ausgleichs_ergebnis(matrix_a, y, koeffizienten, methode):
    """Erstellt ein AusgleichsErgebnis aus Designmatrix, Daten und Koeffizienten.
    
    Imports: import pandas as pd.
    """
    prognose = np.asarray(matrix_a, dtype=float) @ koeffizienten
    residuen = y - prognose
    return AusgleichsErgebnis(
        koeffizienten=koeffizienten,
        residuen=residuen,
        fehlerfunktional=fehlerfunktional_berechnen(residuen),
        rang=int(np.linalg.matrix_rank(matrix_a)),
        konditionszahl=float(np.linalg.cond(matrix_a)),
        methode=methode,
        tabelle=pd.DataFrame({"messwert": y, "modellwert": prognose, "residuum": residuen}),
    )


def lineare_ausgleichsrechnung_normalgleichungen(stuetzstellen, stuetzwerte, basisfunktionen) -> AusgleichsErgebnis:
    """Loest lineare Ausgleichsrechnung mit Normalgleichungen.

    Formel: A^T A lambda=A^T y fuer f=sum lambda_j f_j. Input: Daten und Basisfunktionen.
    Output: AusgleichsErgebnis. Voraussetzungen: A^T A regulaer.
    Numerische Hinweise: Normalgleichungen quadrieren die Konditionszahl.
    Beispiel:
        >>> stuetzstellen = np.array([0.0, 1.0])
        >>> stuetzwerte = np.array([1.0, 2.0])
        >>> def basis_0(x):
        ...     return 1
        >>> def basis_1(x):
        ...     return x**2
        >>> basisfunktionen = [basis_0, basis_1]
        >>> ergebnis = lineare_ausgleichsrechnung_normalgleichungen(stuetzstellen, stuetzwerte, basisfunktionen)
        >>> ergebnis.koeffizienten.round(6).tolist()
        [1.0, 1.0]
    """
    a = designmatrix_erstellen(stuetzstellen, basisfunktionen)
    y = np.asarray(stuetzwerte, dtype=float).reshape(-1)
    lambda_vec = np.linalg.solve(a.T @ a, a.T @ y)
    return _ausgleichs_ergebnis(a, y, lambda_vec, "Normalgleichungen")


def lineare_ausgleichsrechnung_qr(stuetzstellen, stuetzwerte, basisfunktionen) -> AusgleichsErgebnis:
    """Loest lineare Ausgleichsrechnung per QR.

    Formel: A=QR, R lambda=Q^T y. Input: Daten und Basisfunktionen. Output: AusgleichsErgebnis.
    Voraussetzungen: voller Spaltenrang empfohlen. Hinweis: Numerisch stabiler als Normalgleichungen.
    Beispiel:
        >>> stuetzstellen = np.array([0.0, 1.0])
        >>> stuetzwerte = np.array([1.0, 2.0])
        >>> def basis_0(x):
        ...     return 1
        >>> def basis_1(x):
        ...     return x**2
        >>> basisfunktionen = [basis_0, basis_1]
        >>> ergebnis = lineare_ausgleichsrechnung_qr(stuetzstellen, stuetzwerte, basisfunktionen)
        >>> ergebnis.koeffizienten.round(6).tolist()
        [1.0, 1.0]
    """
    a = designmatrix_erstellen(stuetzstellen, basisfunktionen)
    y = np.asarray(stuetzwerte, dtype=float).reshape(-1)
    return _ausgleichs_ergebnis(a, y, loese_ausgleich_qr(a, y), "QR")


def polynom_ausgleich(stuetzstellen, stuetzwerte, grad, methode="qr") -> AusgleichsErgebnis:
    """Passt ein Polynom im kleinsten-Quadrate-Sinn an.

    Formel: p(x)=a_0+...+a_g x^g. Input: Daten, Grad, Methode. Output: AusgleichsErgebnis.
    Voraussetzungen: genug Stuetzstellen. Hinweis: Hohe Grade koennen schlecht konditionieren.
    Beispiel:
        >>> stuetzstellen = np.array([0.0, 1.0])
        >>> stuetzwerte = np.array([1.0, 2.0])
        >>> grad = 1
        >>> ergebnis = polynom_ausgleich(stuetzstellen, stuetzwerte, grad)
        >>> ergebnis.koeffizienten.round(6).tolist()
        [1.0, 1.0]
    """
    basis = []
    for potenz in range(int(grad) + 1):
        def basisfunktion(x, potenz=potenz):
            """Wertet die Monombasis x**potenz aus."""
            return x ** potenz
        basis.append(basisfunktion)
    if methode == "normalgleichungen":
        return lineare_ausgleichsrechnung_normalgleichungen(stuetzstellen, stuetzwerte, basis)
    return lineare_ausgleichsrechnung_qr(stuetzstellen, stuetzwerte, basis)


def ausgleichsgerade(stuetzstellen, stuetzwerte, methode="qr") -> AusgleichsErgebnis:
    """Berechnet eine Ausgleichsgerade.

    Formel: y=a_0+a_1 x. Input: Daten. Output: AusgleichsErgebnis.
    Voraussetzungen: mindestens zwei Datenpunkte. Hinweis: Spezialfall von Polynomgrad 1.
    Beispiel:
        >>> stuetzstellen = np.array([0.0, 1.0])
        >>> stuetzwerte = np.array([1.0, 2.0])
        >>> ergebnis = ausgleichsgerade(stuetzstellen, stuetzwerte)
        >>> ergebnis.koeffizienten.round(6).tolist()
        [1.0, 1.0]
    """
    return polynom_ausgleich(stuetzstellen, stuetzwerte, 1, methode)


def ausgleichsparabel(stuetzstellen, stuetzwerte, methode="qr") -> AusgleichsErgebnis:
    """Berechnet eine Ausgleichsparabel.

    Formel: y=a_0+a_1 x+a_2 x^2. Input: Daten. Output: AusgleichsErgebnis.
    Voraussetzungen: mindestens drei Datenpunkte empfohlen. Hinweis: Spezialfall von Polynomgrad 2.
    Beispiel:
        >>> stuetzstellen = np.array([0.0, 1.0, 2.0])
        >>> stuetzwerte = np.array([1.0, 2.0, 5.0])
        >>> ergebnis = ausgleichsparabel(stuetzstellen, stuetzwerte)
        >>> len(ergebnis.koeffizienten)
        3
    """
    return polynom_ausgleich(stuetzstellen, stuetzwerte, 2, methode)


def ausgleichsfunktion_aus_koeffizienten(koeffizienten, basisfunktionen):
    """Erzeugt die Ausgleichsfunktion aus Koeffizienten.

    Formel: f(x)=sum lambda_j f_j(x). Input: Koeffizienten, Basisfunktionen. Output: callable.
    Voraussetzungen: gleiche Anzahl Koeffizienten und Basisfunktionen. Hinweis: Reiner Auswertungswrapper.
    Beispiel:
        >>> koeffizienten = np.array([1.0, 2.0])
        >>> def basis_0(x):
        ...     return 1
        >>> def basis_1(x):
        ...     return x**2
        >>> basisfunktionen = [basis_0, basis_1]
        >>> f = ausgleichsfunktion_aus_koeffizienten(koeffizienten, basisfunktionen)
        >>> f(3)
        19.0
    """
    koeff = np.asarray(koeffizienten, dtype=float).reshape(-1)
    if len(koeff) != len(basisfunktionen):
        raise ValueError("Anzahl Koeffizienten und Basisfunktionen muss gleich sein.")
    def ausgleichsfunktion(x):
        """Wertet die zusammengesetzte Ausgleichsfunktion aus."""
        return float(sum(k * f(x) for k, f in zip(koeff, basisfunktionen)))

    return ausgleichsfunktion


def residuenfunktion_erstellen(modellfunktion, x_daten, y_daten):
    """Erzeugt die Residuenfunktion fuer nichtlineare Ausgleichsrechnung.

    Formel: g(lambda)=y-f(x,lambda). Input: Modell, x, y. Output: callable.
    Voraussetzungen: Modell liefert Werte fuer alle x. Hinweis: Vorzeichen passt zur Gauss-Newton-Formel unten.
    Beispiel:
        >>> def modellfunktion(x, parameter):
        ...     return parameter[0] * x
        >>> x_daten = np.array([2.0])
        >>> y_daten = np.array([5.0])
        >>> r = residuenfunktion_erstellen(modellfunktion, x_daten, y_daten)
        >>> r([2]).tolist()
        [1.0]
    """
    x = np.asarray(x_daten, dtype=float).reshape(-1)
    y = np.asarray(y_daten, dtype=float).reshape(-1)
    if len(x) != len(y):
        raise ValueError("x_daten und y_daten muessen gleich lang sein.")
    def residuenfunktion(parameter):
        """Wertet die Residuen y_i - f(x_i, parameter) aus."""
        return y - np.array([modellfunktion(xi, parameter) for xi in x], dtype=float)

    return residuenfunktion


def jacobi_residuen_numerisch(modellfunktion, x_daten, parameter, schrittweite=1e-6) -> np.ndarray:
    """Berechnet die numerische Jacobi-Matrix der Residuen.

    Formel: (g(p+h e_j)-g(p-h e_j))/(2h). Input: Modell, Daten, Parameter. Output: Matrix.
    Voraussetzungen: Schrittweite positiv. Hinweis: Zentraldifferenz ist genauer als Vorwaerts.
    Beispiel:
        >>> def modellfunktion(x, parameter):
        ...     return parameter[0] * x
        >>> x_daten = np.array([2.0])
        >>> parameter = np.array([3.0])
        >>> jacobi_residuen_numerisch(modellfunktion, x_daten, parameter).round(4).tolist()
        [[-2.0]]
    """
    x = np.asarray(x_daten, dtype=float).reshape(-1)
    p = np.asarray(parameter, dtype=float).reshape(-1)
    h = float(schrittweite)
    jac = np.zeros((x.size, p.size), dtype=float)
    for j in range(p.size):
        e = np.zeros_like(p)
        e[j] = 1.0
        plus = np.array([modellfunktion(xi, p + h * e) for xi in x], dtype=float)
        minus = np.array([modellfunktion(xi, p - h * e) for xi in x], dtype=float)
        jac[:, j] = -(plus - minus) / (2.0 * h)
    return jac


def gauss_newton_verfahren(modellfunktion, jacobi_residuen, x_daten, y_daten, startparameter, toleranz=1e-10, maximale_iterationen=50, rueckgabe_verlauf=False) -> IterationsErgebnis:
    """Loest nichtlineare Ausgleichsrechnung mit Gauss-Newton.

    Formel: (Dg^T Dg) delta=-Dg^T g, lambda_{k+1}=lambda_k+delta.
    Input: Modell, Jacobi der Residuen, Daten, Startparameter. Output: IterationsErgebnis.
    Voraussetzungen: Dg hat vollen Spaltenrang. Hinweis: Nur lokal robust.
    Beispiel:
        >>> def modellfunktion(x, parameter):
        ...     return parameter[0] * x
        >>> x_daten = np.array([1.0, 2.0])
        >>> y_daten = np.array([2.0, 4.0])
        >>> def jacobi_residuen(parameter):
        ...     return np.array([[-1.0], [-2.0]])
        >>> startparameter = np.array([1.0])
        >>> r = gauss_newton_verfahren(modellfunktion, jacobi_residuen, x_daten, y_daten, startparameter)
        >>> round(r.loesung[0], 6)
        2.0
    """
    p = np.asarray(startparameter, dtype=float).reshape(-1)
    gfun = residuenfunktion_erstellen(modellfunktion, x_daten, y_daten)
    verlauf = []
    schrittnorm = None
    for k in range(maximale_iterationen):
        g = gfun(p)
        dg = np.asarray(jacobi_residuen(p), dtype=float)
        delta = np.linalg.solve(dg.T @ dg, -dg.T @ g)
        schrittnorm = float(np.linalg.norm(delta))
        verlauf.append({"iteration": k, "parameter": p.copy(), "residualnorm": float(np.linalg.norm(g)), "delta": delta.copy(), "schrittnorm": schrittnorm, "p": 0, "faktor": 1.0})
        p = p + delta
        residualnorm = float(np.linalg.norm(gfun(p)))
        if schrittnorm <= toleranz or residualnorm <= toleranz:
            return IterationsErgebnis(p, k + 1, True, residualnorm, schrittnorm, verlauf if rueckgabe_verlauf else None, "konvergiert")
    return IterationsErgebnis(p, maximale_iterationen, False, float(np.linalg.norm(gfun(p))), schrittnorm, verlauf if rueckgabe_verlauf else None, "maximale Iterationen erreicht")


def gedaempftes_gauss_newton_verfahren(modellfunktion, jacobi_residuen, x_daten, y_daten, startparameter, toleranz=1e-10, maximale_iterationen=50, p_max=20, rueckgabe_verlauf=False) -> IterationsErgebnis:
    """Loest nichtlineare Ausgleichsrechnung mit gedaempftem Gauss-Newton.

    Formel: lambda_{k+1}=lambda_k+delta/2^p mit sinkender Residualnorm.
    Input: Modell, Jacobi, Daten, Startparameter. Output: IterationsErgebnis.
    Voraussetzungen: Dg^T Dg loesbar. Hinweis: Daempfung verbessert Robustheit.
    Beispiel:
        >>> def modellfunktion(x, parameter):
        ...     return parameter[0] * x
        >>> x_daten = np.array([1.0, 2.0])
        >>> y_daten = np.array([2.0, 4.0])
        >>> def jacobi_residuen(parameter):
        ...     return np.array([[-1.0], [-2.0]])
        >>> startparameter = np.array([1.0])
        >>> r = gedaempftes_gauss_newton_verfahren(modellfunktion, jacobi_residuen, x_daten, y_daten, startparameter)
        >>> round(r.loesung[0], 6)
        2.0
    """
    p = np.asarray(startparameter, dtype=float).reshape(-1)
    gfun = residuenfunktion_erstellen(modellfunktion, x_daten, y_daten)
    verlauf = []
    schrittnorm = None
    for k in range(maximale_iterationen):
        g = gfun(p)
        dg = np.asarray(jacobi_residuen(p), dtype=float)
        delta = np.linalg.solve(dg.T @ dg, -dg.T @ g)
        alt = float(np.linalg.norm(g))
        faktor = 2.0 ** (-p_max)
        p_gewaehlt = p_max
        for d in range(p_max + 1):
            kandidat = p + delta / (2.0 ** d)
            if np.linalg.norm(gfun(kandidat)) < alt:
                faktor = 2.0 ** (-d)
                p_gewaehlt = d
                break
        schritt = faktor * delta
        schrittnorm = float(np.linalg.norm(schritt))
        verlauf.append({"iteration": k, "parameter": p.copy(), "residualnorm": alt, "delta": delta.copy(), "schrittnorm": schrittnorm, "p": p_gewaehlt, "faktor": faktor})
        p = p + schritt
        residualnorm = float(np.linalg.norm(gfun(p)))
        if schrittnorm <= toleranz or residualnorm <= toleranz:
            return IterationsErgebnis(p, k + 1, True, residualnorm, schrittnorm, verlauf if rueckgabe_verlauf else None, "konvergiert")
    return IterationsErgebnis(p, maximale_iterationen, False, float(np.linalg.norm(gfun(p))), schrittnorm, verlauf if rueckgabe_verlauf else None, "maximale Iterationen erreicht")


def exponentialmodell(x, parameter):
    """Wertet das Exponentialmodell aus.

    Formel: f(x)=a*exp(-b*x)+c. Input: x und Parameter [a,b,c]. Output: float.
    Voraussetzungen: drei Parameter. Hinweis: Startwerte beeinflussen Gauss-Newton stark.
    Beispiel:
        >>> round(exponentialmodell(0, [2, 1, 3]), 6)
        5.0
    """
    a, b, c = np.asarray(parameter, dtype=float).reshape(-1)
    return float(a * np.exp(-b * x) + c)


def logistisches_modell(x, parameter):
    """Wertet das logistische Modell aus.

    Formel: f(x)=K/(1+A*exp(-r*x)). Input: x und Parameter [K,A,r]. Output: float.
    Voraussetzungen: drei Parameter. Hinweis: K ist Kapazitaet.
    Beispiel:
        >>> round(logistisches_modell(0, [10, 1, 1]), 6)
        5.0
    """
    k, a, r = np.asarray(parameter, dtype=float).reshape(-1)
    return float(k / (1.0 + a * np.exp(-r * x)))


def logistische_linearisierung(stuetzstellen, stuetzwerte, kapazitaet_k) -> AusgleichsErgebnis:
    """Linearisiert ein logistisches Wachstum bei bekannter Kapazitaet.

    Formel: ln(K/y-1)=ln(A)-r x. Input: x, y, K. Output: AusgleichsErgebnis.
    Voraussetzungen: 0<y<K. Hinweis: Liefert Parameter fuer A und r als lineare Regression.
    Beispiel:
        >>> stuetzstellen = np.array([0.0, 1.0])
        >>> stuetzwerte = np.array([2.0, 4.0])
        >>> kapazitaet_k = 10
        >>> ergebnis = logistische_linearisierung(stuetzstellen, stuetzwerte, kapazitaet_k)
        >>> ergebnis.koeffizienten.shape[0]
        2
    """
    x = np.asarray(stuetzstellen, dtype=float).reshape(-1)
    y = np.asarray(stuetzwerte, dtype=float).reshape(-1)
    if np.any(y <= 0) or np.any(y >= kapazitaet_k):
        raise ValueError("Fuer die Linearisierung muss 0 < y < K gelten.")
    z = np.log(kapazitaet_k / y - 1.0)
    return ausgleichsgerade(x, z, methode="qr")

# ===== interpolation.py =====
"""Interpolation und natuerliche kubische Splines."""






def lagrange_polynom_symbolisch(stuetzstellen, stuetzwerte, variable=None) -> sp.Expr:
    """Berechnet das Lagrange-Polynom symbolisch.

    Formel: P_n(x)=sum_i y_i prod_{j!=i}(x-x_j)/(x_i-x_j). Input: Daten und Variable.
    Output: SymPy-Ausdruck. Voraussetzungen: paarweise verschiedene Stuetzstellen.
    Numerische Hinweise: Symbolische Vereinfachung kann bei vielen Punkten langsam sein.
    Beispiel:
        >>> x = sp.symbols('x')
        >>> stuetzstellen = [0, 1]
        >>> stuetzwerte = [0, 2]
        >>> lagrange_polynom_symbolisch(stuetzstellen, stuetzwerte, x)
        2*x
    """
    variable = variable or sp.symbols("x")
    xwerte = list(stuetzstellen)
    ywerte = list(stuetzwerte)
    if len(xwerte) != len(ywerte):
        raise ValueError("stuetzstellen und stuetzwerte muessen gleich lang sein.")
    pruefe_stuetzstellen_paarweise_verschieden(xwerte)
    polynom = 0
    for i, xi in enumerate(xwerte):
        basis = 1
        for j, xj in enumerate(xwerte):
            if i != j:
                basis *= (variable - xj) / (xi - xj)
        polynom += ywerte[i] * basis
    return sp.expand(polynom)


def lagrange_basiswert(stuetzstellen, index, x_wert) -> float:
    """Berechnet einen Lagrange-Basiswert.

    Formel: l_i(x)=prod_{j!=i}(x-x_j)/(x_i-x_j). Input: Stuetzstellen, Index, x. Output: float.
    Voraussetzungen: Stuetzstellen paarweise verschieden. Hinweis: Fuer viele Punkte numerisch instabil.
    Beispiel:
        >>> stuetzstellen = np.array([0.0, 1.0])
        >>> index = 0
        >>> x_wert = 0.5
        >>> lagrange_basiswert(stuetzstellen, index, x_wert)
        0.5
    """
    x = np.asarray(stuetzstellen, dtype=float).reshape(-1)
    pruefe_stuetzstellen_paarweise_verschieden(x)
    i = int(index)
    wert = 1.0
    for j in range(x.size):
        if j != i:
            wert *= (float(x_wert) - x[j]) / (x[i] - x[j])
    return float(wert)


def lagrange_interpolation(stuetzstellen, stuetzwerte, x_wert) -> float:
    """Wertet das Lagrange-Interpolationspolynom aus.

    Formel: P_n(x)=sum_i l_i(x)y_i. Input: Stuetzstellen, Werte, x. Output: float.
    Voraussetzungen: gleiche Laenge, paarweise verschiedene Stuetzstellen. Hinweis: Baryzentrische Formen sind stabiler.
    Beispiel:
        >>> stuetzstellen = np.array([0.0, 1.0])
        >>> stuetzwerte = np.array([0.0, 2.0])
        >>> x_wert = 0.5
        >>> lagrange_interpolation(stuetzstellen, stuetzwerte, x_wert)
        1.0
    """
    x = np.asarray(stuetzstellen, dtype=float).reshape(-1)
    y = np.asarray(stuetzwerte, dtype=float).reshape(-1)
    if len(x) != len(y):
        raise ValueError("stuetzstellen und stuetzwerte muessen gleich lang sein.")
    return float(sum(y[i] * lagrange_basiswert(x, i, x_wert) for i in range(x.size)))


def interpolationsfehler_abschaetzen(stuetzstellen, x_wert, max_ableitung_n_plus_1) -> float:
    """Schaetzt den Interpolationsfehler ab.

    Formel: |f-P| <= |prod_i(x-x_i)|/(n+1)! * max|f^(n+1)|.
    Input: Stuetzstellen, Auswertungspunkt, Ableitungsschranke. Output: float.
    Voraussetzungen: passende Ableitungsschranke bekannt. Hinweis: Schranke kann konservativ sein.
    Imports: import math.
    Beispiel:
        >>> stuetzstellen = np.array([0.0, 1.0])
        >>> x_wert = 0.5
        >>> max_ableitung_n_plus_1 = 2
        >>> interpolationsfehler_abschaetzen(stuetzstellen, x_wert, max_ableitung_n_plus_1)
        0.25
    """
    x = np.asarray(stuetzstellen, dtype=float).reshape(-1)
    produkt = float(np.prod([float(x_wert) - xi for xi in x]))
    return abs(produkt) / math.factorial(x.size) * abs(float(max_ableitung_n_plus_1))


def natuerlicher_kubischer_spline_koeffizienten(stuetzstellen, stuetzwerte) -> SplineKoeffizienten:
    """Berechnet Koeffizienten eines natuerlichen kubischen Splines.

    Formel: S_i=a_i+b_i dx+c_i dx^2+d_i dx^3 mit c_0=c_n=0 und tridiagonalem LGS.
    Input: streng steigende Stuetzstellen und Werte. Output: SplineKoeffizienten.
    Voraussetzungen: mindestens zwei Punkte. Hinweis: Natuerliche Randbedingungen setzen zweite Ableitung am Rand null.
    Beispiel:
        >>> stuetzstellen = np.array([0.0, 1.0])
        >>> stuetzwerte = np.array([0.0, 1.0])
        >>> koeffizienten = natuerlicher_kubischer_spline_koeffizienten(stuetzstellen, stuetzwerte)
        >>> koeffizienten.b_werte.tolist()
        [1.0]
    """
    x = np.asarray(stuetzstellen, dtype=float).reshape(-1)
    y = np.asarray(stuetzwerte, dtype=float).reshape(-1)
    if len(x) != len(y):
        raise ValueError("stuetzstellen und stuetzwerte muessen gleich lang sein.")
    pruefe_stuetzstellen_streng_steigend(x)
    n = x.size - 1
    h = np.diff(x)
    a = y[:-1].copy()
    c = np.zeros(n + 1)
    if n > 1:
        matrix = np.zeros((n - 1, n - 1), dtype=float)
        rechte_seite = np.zeros(n - 1, dtype=float)
        for i in range(1, n):
            zeile = i - 1
            if zeile - 1 >= 0:
                matrix[zeile, zeile - 1] = h[i - 1]
            matrix[zeile, zeile] = 2.0 * (h[i - 1] + h[i])
            if zeile + 1 < n - 1:
                matrix[zeile, zeile + 1] = h[i]
            rechte_seite[zeile] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
        c[1:n] = np.linalg.solve(matrix, rechte_seite)
    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        # Koeffizientenformeln folgen direkt aus Stetigkeit von S, S' und S''.
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])
    return SplineKoeffizienten(x, a, b, c[:-1], d)


def spline_abschnitt_finden(stuetzstellen, x_wert) -> int:
    """Findet den Spline-Abschnitt fuer einen x-Wert.

    Formel: suche i mit x_i <= x <= x_{i+1}. Input: Stuetzstellen, x. Output: Index.
    Voraussetzungen: Stuetzstellen streng steigend. Hinweis: Randwerte werden dem Randabschnitt zugeordnet.
    Beispiel:
        >>> stuetzstellen = np.array([0.0, 1.0, 2.0])
        >>> x_wert = 1.5
        >>> spline_abschnitt_finden(stuetzstellen, x_wert)
        1
    """
    x = np.asarray(stuetzstellen, dtype=float).reshape(-1)
    pruefe_stuetzstellen_streng_steigend(x)
    wert = float(x_wert)
    if wert < x[0] or wert > x[-1]:
        raise ValueError("x_wert liegt ausserhalb des Spline-Bereichs.")
    return int(min(max(np.searchsorted(x, wert, side="right") - 1, 0), x.size - 2))


def natuerlicher_kubischer_spline_auswerten(koeffizienten, x_wert) -> float:
    """Wertet einen natuerlichen kubischen Spline aus.

    Formel: S_i(x)=a_i+b_i dx+c_i dx^2+d_i dx^3. Input: Koeffizienten, x. Output: float.
    Voraussetzungen: x im Intervall. Hinweis: Keine Extrapolation.
    Beispiel:
        >>> stuetzstellen = [0, 1]
        >>> stuetzwerte = [0, 1]
        >>> k = natuerlicher_kubischer_spline_koeffizienten(stuetzstellen, stuetzwerte)
        >>> natuerlicher_kubischer_spline_auswerten(k, 0.5)
        0.5
    """
    i = spline_abschnitt_finden(koeffizienten.stuetzstellen, x_wert)
    dx = float(x_wert) - np.asarray(koeffizienten.stuetzstellen, dtype=float)[i]
    return float(koeffizienten.a_werte[i] + koeffizienten.b_werte[i] * dx + koeffizienten.c_werte[i] * dx**2 + koeffizienten.d_werte[i] * dx**3)


def natuerlicher_kubischer_spline_ableitung_auswerten(koeffizienten, x_wert, ordnung=1) -> float:
    """Wertet die erste oder zweite Ableitung eines natuerlichen kubischen Splines aus.

    Formel: Fuer S_i(x)=a_i+b_i dx+c_i dx^2+d_i dx^3 gilt
    S_i'(x)=b_i+2c_i dx+3d_i dx^2 und S_i''(x)=2c_i+6d_i dx.
    Input: SplineKoeffizienten, x-Wert im Spline-Bereich, ordnung 1 oder 2.
    Output: float mit Geschwindigkeit bzw. Beschleunigung.
    Voraussetzungen: x_wert liegt innerhalb der Stuetzstellen; ordnung ist 1 oder 2.
    Numerische Hinweise: An Stuetzstellen ist die Ableitung eindeutig, aber der Abschnitt
    wird per rechter Suchregel dem linken Abschnitt zugeordnet.
    Beispiel:
        >>> stuetzstellen = np.array([0.0, 1.0])
        >>> stuetzwerte = np.array([0.0, 1.0])
        >>> koeffizienten = natuerlicher_kubischer_spline_koeffizienten(stuetzstellen, stuetzwerte)
        >>> x_wert = 0.5
        >>> natuerlicher_kubischer_spline_ableitung_auswerten(koeffizienten, x_wert)
        1.0
    """
    i = spline_abschnitt_finden(koeffizienten.stuetzstellen, x_wert)
    dx = float(x_wert) - np.asarray(koeffizienten.stuetzstellen, dtype=float)[i]
    if int(ordnung) == 1:
        return float(koeffizienten.b_werte[i] + 2.0 * koeffizienten.c_werte[i] * dx + 3.0 * koeffizienten.d_werte[i] * dx**2)
    if int(ordnung) == 2:
        return float(2.0 * koeffizienten.c_werte[i] + 6.0 * koeffizienten.d_werte[i] * dx)
    raise ValueError("ordnung muss 1 oder 2 sein.")


def spline_abschnitte_als_text(koeffizienten) -> list[str]:
    """Gibt Spline-Abschnitte als Textformeln aus.

    Formel: S_i=a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3. Input: SplineKoeffizienten. Output: Liste.
    Voraussetzungen: Koeffizienten konsistent. Hinweis: Praktisch zum Abschreiben.
    Beispiel:
        >>> stuetzstellen = np.array([0.0, 1.0])
        >>> stuetzwerte = np.array([0.0, 1.0])
        >>> koeffizienten = natuerlicher_kubischer_spline_koeffizienten(stuetzstellen, stuetzwerte)
        >>> len(spline_abschnitte_als_text(koeffizienten))
        1
    """
    x = np.asarray(koeffizienten.stuetzstellen, dtype=float)
    texte = []
    for i in range(len(koeffizienten.a_werte)):
        texte.append(f"{x[i]} <= x <= {x[i+1]}: {koeffizienten.a_werte[i]} + {koeffizienten.b_werte[i]}*(x-{x[i]}) + {koeffizienten.c_werte[i]}*(x-{x[i]})^2 + {koeffizienten.d_werte[i]}*(x-{x[i]})^3")
    return texte


def spline_tabelle_erstellen(koeffizienten) -> pd.DataFrame:
    """Erstellt eine Tabelle der Spline-Koeffizienten.

    Formel: Ein Abschnitt pro Zeile. Input: SplineKoeffizienten. Output: DataFrame.
    Voraussetzungen: Koeffizienten konsistent. Hinweis: Tabellen sind ideal fuer Pruefungsoutputs.
    Imports: import pandas as pd.
    Beispiel:
        >>> stuetzstellen = np.array([0.0, 1.0])
        >>> stuetzwerte = np.array([0.0, 1.0])
        >>> koeffizienten = natuerlicher_kubischer_spline_koeffizienten(stuetzstellen, stuetzwerte)
        >>> tabelle = spline_tabelle_erstellen(koeffizienten)
        >>> tabelle.shape[0]
        1
    """
    return koeffizienten.als_tabelle()

# ===== integration.py =====
"""Numerische Integration: Newton-Cotes, Fehlergrenzen, Romberg und Gauss-Legendre."""






def rechteckregel(funktion, untere_grenze, obere_grenze) -> float:
    """Berechnet die einfache Mittelpunkt-Rechteckregel.

    Formel: (b-a) f((a+b)/2). Input: Funktion und Grenzen. Output: float.
    Voraussetzungen: Funktion am Mittelpunkt definiert. Hinweis: Exakt fuer lineare Funktionen symmetrisch um den Mittelpunkt.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> untere_grenze = 0
        >>> obere_grenze = 2
        >>> rechteckregel(funktion, untere_grenze, obere_grenze)
        2.0
    """
    a, b = float(untere_grenze), float(obere_grenze)
    return float((b - a) * funktion((a + b) / 2.0))


def trapezregel(funktion, untere_grenze, obere_grenze) -> float:
    """Berechnet die einfache Trapezregel.

    Formel: (b-a)(f(a)+f(b))/2. Input: Funktion und Grenzen. Output: float.
    Voraussetzungen: Funktion an Randpunkten definiert. Hinweis: Exakt fuer lineare Funktionen.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2 - 7
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> untere_grenze = 0
        >>> obere_grenze = 2
        >>> trapezregel(funktion, untere_grenze, obere_grenze)
        -10.0
    """
    a, b = float(untere_grenze), float(obere_grenze)
    return float((b - a) * (funktion(a) + funktion(b)) / 2.0)


def simpsonregel(funktion, untere_grenze, obere_grenze) -> float:
    """Berechnet die einfache Simpson-Regel.

    Formel: (b-a)/6*(f(a)+4f((a+b)/2)+f(b)). Input: Funktion und Grenzen. Output: float.
    Voraussetzungen: Funktion an a,m,b definiert. Hinweis: Exakt fuer Polynome bis Grad 3.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> untere_grenze = 0
        >>> obere_grenze = 1
        >>> round(simpsonregel(funktion, untere_grenze, obere_grenze), 12)
        0.333333333333
    """
    a, b = float(untere_grenze), float(obere_grenze)
    m = (a + b) / 2.0
    return float((b - a) / 6.0 * (funktion(a) + 4.0 * funktion(m) + funktion(b)))


def summierte_rechteckregel(funktion, untere_grenze, obere_grenze, anzahl_intervalle) -> QuadraturErgebnis:
    """Berechnet die summierte Mittelpunkt-Rechteckregel.

    Formel: h sum_{i=0}^{n-1} f(x_i+h/2). Input: Funktion, Grenzen, n. Output: QuadraturErgebnis.
    Voraussetzungen: n positiv. Hinweis: Fehlerordnung O(h^2).
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2 - 7
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> untere_grenze = 0
        >>> obere_grenze = 2
        >>> anzahl_intervalle = 2
        >>> summierte_rechteckregel(funktion, untere_grenze, obere_grenze, anzahl_intervalle).wert
        -11.5
    """
    if int(anzahl_intervalle) <= 0:
        raise ValueError("anzahl_intervalle muss positiv sein.")
    a, b, n = float(untere_grenze), float(obere_grenze), int(anzahl_intervalle)
    h = (b - a) / n
    mittelpunkte = a + (np.arange(n) + 0.5) * h
    werte = np.array([funktion(x) for x in mittelpunkte], dtype=float)
    return QuadraturErgebnis(float(h * np.sum(werte)), "summierte Rechteckregel", h, n, None, {"stuetzstellen": mittelpunkte, "funktionswerte": werte})


def summierte_trapezregel(funktion, untere_grenze, obere_grenze, anzahl_intervalle) -> QuadraturErgebnis:
    """Berechnet die summierte Trapezregel.

    Formel: h((f(a)+f(b))/2+sum_{i=1}^{n-1}f(x_i)). Input: Funktion, Grenzen, n.
    Output: QuadraturErgebnis. Voraussetzungen: n positiv. Hinweis: Fehlerordnung O(h^2).
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2 - 7
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> untere_grenze = 0
        >>> obere_grenze = 2
        >>> anzahl_intervalle = 4
        >>> summierte_trapezregel(funktion, untere_grenze, obere_grenze, anzahl_intervalle).wert
        -11.25
    """
    if int(anzahl_intervalle) <= 0:
        raise ValueError("anzahl_intervalle muss positiv sein.")
    a, b, n = float(untere_grenze), float(obere_grenze), int(anzahl_intervalle)
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([funktion(xi) for xi in x], dtype=float)
    wert = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return QuadraturErgebnis(float(wert), "summierte Trapezregel", h, n, None, {"stuetzstellen": x, "funktionswerte": y})


def trapezregel_nicht_aequidistant(stuetzstellen, stuetzwerte) -> float:
    """Integriert tabellierte nichtaequidistante Werte mit Trapezen.

    Formel: sum_i (y_i+y_{i+1})/2*(x_{i+1}-x_i). Input: x,y. Output: float.
    Voraussetzungen: x streng steigend, gleiche Laenge. Hinweis: Keine Interpolation hoeherer Ordnung.
    Beispiel:
        >>> stuetzstellen = np.array([0.0, 1.0])
        >>> stuetzwerte = np.array([0.0, 2.0])
        >>> trapezregel_nicht_aequidistant(stuetzstellen, stuetzwerte)
        1.0
    """
    x = np.asarray(stuetzstellen, dtype=float).reshape(-1)
    y = np.asarray(stuetzwerte, dtype=float).reshape(-1)
    if len(x) != len(y):
        raise ValueError("stuetzstellen und stuetzwerte muessen gleich lang sein.")
    pruefe_stuetzstellen_streng_steigend(x)
    return float(np.sum((y[:-1] + y[1:]) * 0.5 * np.diff(x)))


def summierte_simpsonregel(funktion, untere_grenze, obere_grenze, anzahl_intervalle) -> QuadraturErgebnis:
    """Berechnet die summierte Simpson-Regel.

    Formel: h/3*(f0+f_n+4 sum ungerade f_i+2 sum gerade f_i), aequivalent S=1/3(T+2R).
    Input: Funktion, Grenzen, gerade n. Output: QuadraturErgebnis.
    Voraussetzungen: n positiv und gerade. Hinweis: Fehlerordnung O(h^4).
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> untere_grenze = 0
        >>> obere_grenze = 1
        >>> anzahl_intervalle = 2
        >>> round(summierte_simpsonregel(funktion, untere_grenze, obere_grenze, anzahl_intervalle).wert, 12)
        0.333333333333
    """
    if int(anzahl_intervalle) <= 0 or int(anzahl_intervalle) % 2 != 0:
        raise ValueError("anzahl_intervalle muss positiv und gerade sein.")
    a, b, n = float(untere_grenze), float(obere_grenze), int(anzahl_intervalle)
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([funktion(xi) for xi in x], dtype=float)
    wert = h / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]))
    return QuadraturErgebnis(float(wert), "summierte Simpsonregel", h, n, None, {"stuetzstellen": x, "funktionswerte": y})


def simpson_als_mittel(trapezwert, rechteckwert) -> float:
    """Berechnet Simpson als gewichtetes Mittel aus Trapez und Rechteck.

    Formel: S=1/3*(T+2R). Input: T und R. Output: float.
    Voraussetzungen: T und R zur gleichen Schrittweite. Hinweis: HM2-Form der Simpsonregel.
    Beispiel:
        >>> simpson_als_mittel(1, 2)
        1.6666666666666667
    """
    return float((float(trapezwert) + 2.0 * float(rechteckwert)) / 3.0)


def integriere_tabelle_nicht_aequidistant(stuetzstellen, werte) -> QuadraturErgebnis:
    """Integriert eine nichtaequidistante Tabelle.

    Formel: nichtaequidistante Trapezsumme. Input: x,y. Output: QuadraturErgebnis.
    Voraussetzungen: x streng steigend. Hinweis: Ergebnis haengt nur von Tabellenwerten ab.
    Beispiel:
        >>> stuetzstellen = np.array([0.0, 1.0])
        >>> stuetzwerte = np.array([0.0, 2.0])
        >>> ergebnis = integriere_tabelle_nicht_aequidistant(stuetzstellen, stuetzwerte)
        >>> ergebnis.wert
        1.0
    """
    wert = trapezregel_nicht_aequidistant(stuetzstellen, werte)
    return QuadraturErgebnis(wert, "nichtaequidistante Trapezregel", None, len(stuetzstellen) - 1, None, {"stuetzstellen": np.asarray(stuetzstellen, dtype=float).reshape(-1), "werte": np.asarray(werte, dtype=float).reshape(-1)})


def funktion_aus_stuetzstellen_linear(stuetzstellen, stuetzwerte):
    """Erzeugt aus tabellierten Stuetzstellen eine linear interpolierte Funktion.

    Zweck: Quadraturfunktionen wie `summierte_trapezregel` oder `romberg_extrapolation`
    erwarten eine Funktion. Mit diesem Wrapper koennen tabellierte Werte trotzdem als
    Funktionsinput genutzt werden.
    Formel: Zwischen zwei Stuetzstellen wird linear interpoliert, intern mit `np.interp`.
    Input: stuetzstellen x_i und stuetzwerte y_i gleicher Laenge.
    Output: Funktion f(x), die skalare x-Werte linear interpoliert auswertet.
    Voraussetzungen: Stuetzstellen streng steigend; ausserhalb des Bereichs extrapoliert
    `np.interp` konstant mit dem Randwert.
    Numerische Hinweise: Linearinterpolation ist nur stueckweise glatt; Romberg setzt
    eigentlich glatte Funktionen voraus und ist dann eher als technischer Vergleich zu lesen.
    Beispiel:
        >>> stuetzstellen = np.array([0.0, 0.5, 1.0])
        >>> stuetzwerte = np.array([0.0, 1.0, 4.0])
        >>> funktion = funktion_aus_stuetzstellen_linear(stuetzstellen, stuetzwerte)
        >>> funktion(0.75)
        2.5
    """
    x = np.asarray(stuetzstellen, dtype=float).reshape(-1)
    y = np.asarray(stuetzwerte, dtype=float).reshape(-1)
    if len(x) != len(y):
        raise ValueError("stuetzstellen und stuetzwerte muessen gleich lang sein.")
    def interpolierte_funktion(x_wert):
        """Wertet die lineare Interpolation an einem skalaren x-Wert aus."""
        return float(np.interp(float(x_wert), x, y))

    return interpolierte_funktion


def fehlergrenze_rechteckregel(schrittweite, untere_grenze, obere_grenze, max_zweite_ableitung) -> float:
    """Berechnet die Fehlergrenze der summierten Rechteckregel.

    Formel: <= h^2/24*(b-a)*max|f''|. Input: h,a,b,M2. Output: float.
    Voraussetzungen: Schranke M2 bekannt. Hinweis: Konservative Abschaetzung.
    Beispiel:
        >>> fehlergrenze_rechteckregel(0.1, 0, 1, 2)
        0.0008333333333333335
    """
    return float(float(schrittweite) ** 2 / 24.0 * abs(float(obere_grenze) - float(untere_grenze)) * abs(float(max_zweite_ableitung)))


def fehlergrenze_trapezregel(schrittweite, untere_grenze, obere_grenze, max_zweite_ableitung) -> float:
    """Berechnet die Fehlergrenze der summierten Trapezregel.

    Formel: <= h^2/12*(b-a)*max|f''|. Input: h,a,b,M2. Output: float.
    Voraussetzungen: Schranke M2 bekannt. Hinweis: Doppelt so gross wie Rechteck-Schranke.
    Beispiel:
        >>> fehlergrenze_trapezregel(0.1, 0, 1, 2)
        0.001666666666666667
    """
    return float(float(schrittweite) ** 2 / 12.0 * abs(float(obere_grenze) - float(untere_grenze)) * abs(float(max_zweite_ableitung)))


def fehlergrenze_simpsonregel(schrittweite, untere_grenze, obere_grenze, max_vierte_ableitung) -> float:
    """Berechnet die Fehlergrenze der summierten Simpson-Regel.

    Formel: <= h^4/2880*(b-a)*max|f^(4)|. Input: h,a,b,M4. Output: float.
    Voraussetzungen: vierte Ableitung beschraenkt. Hinweis: Sehr schnelle Fehlerabnahme bei Halbierung.
    Beispiel:
        >>> fehlergrenze_simpsonregel(0.1, 0, 1, 24)
        8.333333333333336e-07
    """
    return float(float(schrittweite) ** 4 / 2880.0 * abs(float(obere_grenze) - float(untere_grenze)) * abs(float(max_vierte_ableitung)))


def schrittweite_fuer_trapezregel(epsilon, untere_grenze, obere_grenze, max_zweite_ableitung) -> float:
    """Waehlt eine Schrittweite fuer die Trapezregel.

    Formel: h <= sqrt(12 eps/((b-a)M2)). Input: eps,a,b,M2. Output: h.
    Voraussetzungen: eps und M2 positiv. Hinweis: Falls M2=0 ist die Funktion linear.
    Imports: import math.
    Beispiel:
        >>> round(schrittweite_fuer_trapezregel(0.01, 0, 1, 2), 6)
        0.244949
    """
    return float(math.sqrt(12.0 * float(epsilon) / (abs(float(obere_grenze) - float(untere_grenze)) * abs(float(max_zweite_ableitung)))))


def schrittweite_fuer_simpsonregel(epsilon, untere_grenze, obere_grenze, max_vierte_ableitung) -> float:
    """Waehlt eine Schrittweite fuer die Simpson-Regel.

    Formel: h <= fourth_root(2880 eps/((b-a)M4)). Input: eps,a,b,M4. Output: h.
    Voraussetzungen: eps und M4 positiv. Hinweis: Rundung auf gerade Intervallanzahl danach noetig.
    Beispiel:
        >>> round(schrittweite_fuer_simpsonregel(0.01, 0, 1, 24), 6)
        1.046635
    """
    return float((2880.0 * float(epsilon) / (abs(float(obere_grenze) - float(untere_grenze)) * abs(float(max_vierte_ableitung)))) ** 0.25)


def anzahl_intervalle_aus_schrittweite(untere_grenze, obere_grenze, schrittweite, gerade_erforderlich=False) -> int:
    """Berechnet eine Intervallanzahl aus einer maximalen Schrittweite.

    Formel: n=ceil((b-a)/h), optional auf gerade erhoeht. Input: a,b,h. Output: int.
    Voraussetzungen: h>0. Hinweis: Tatsaechliche Schrittweite ist danach hoechstens h.
    Imports: import math.
    Beispiel:
        >>> anzahl_intervalle_aus_schrittweite(0, 1, 0.3)
        4
    """
    n = int(math.ceil(abs(float(obere_grenze) - float(untere_grenze)) / float(schrittweite)))
    if gerade_erforderlich and n % 2:
        n += 1
    return max(n, 1)


def romberg_tabelle_erstellen(trapezwerte) -> pd.DataFrame:
    """Erstellt eine Romberg-Tabelle aus Trapezwerten.

    Formel: T_{j,k}=(4^k T_{j+1,k-1}-T_{j,k-1})/(4^k-1). Input: Trapezwerte. Output: DataFrame.
    Voraussetzungen: Werte fuer halbierte Schrittweiten. Hinweis: Rechte untere Ecke ist genauester Wert.
    Imports: import pandas as pd.
    Beispiel:
        >>> romberg_tabelle_erstellen([1.0, 0.5]).shape
        (2, 2)
    """
    werte = [float(w) for w in trapezwerte]
    m = len(werte)
    tabelle = [[np.nan] * m for _ in range(m)]
    for j, wert in enumerate(werte):
        tabelle[j][0] = wert
    for k in range(1, m):
        for j in range(m - k):
            tabelle[j][k] = (4**k * tabelle[j + 1][k - 1] - tabelle[j][k - 1]) / (4**k - 1)
    return pd.DataFrame(tabelle, columns=[f"T_{k}" for k in range(m)])


def romberg_extrapolation(funktion, untere_grenze, obere_grenze, ordnung_m) -> tuple[float, pd.DataFrame]:
    """Berechnet Romberg-Extrapolation.

    Formel: T_{j,0}=T_f((b-a)/2^j), dann Richardson mit 4^k.
    Input: Funktion, Grenzen, Ordnung m. Output: genauester Wert und Tabelle.
    Voraussetzungen: glatte Funktion. Hinweis: Nutzt Trapezwerte mit halbierter Schrittweite.
    Imports: import pandas as pd.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2 - 7
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> untere_grenze = 0
        >>> obere_grenze = 2
        >>> ordnung_m = 2
        >>> wert, tabelle = romberg_extrapolation(funktion, untere_grenze, obere_grenze, ordnung_m)
        >>> round(wert, 6)
        -11.333333
    """
    m = int(ordnung_m)
    trapezwerte = [summierte_trapezregel(funktion, untere_grenze, obere_grenze, 2**j).wert for j in range(m + 1)]
    tabelle = romberg_tabelle_erstellen(trapezwerte)
    return float(tabelle.iloc[0, m]), tabelle


def romberg_extrapolation_aus_stuetzstellen(stuetzstellen, stuetzwerte, ordnung_m) -> tuple[float, pd.DataFrame]:
    """Berechnet Romberg-Extrapolation aus tabellierten Stuetzstellen.

    Zweck: Praktische Variante, wenn in der Pruefung nur Messwerte oder Tabellenwerte
    gegeben sind, aber die Romberg-Funktion eine auswertbare Funktion erwartet.
    Formel: Zuerst wird mit `funktion_aus_stuetzstellen_linear` eine lineare
    Interpolationsfunktion gebaut, danach wird `romberg_extrapolation` angewendet.
    Input: stuetzstellen x_i, stuetzwerte y_i, Romberg-Ordnung m.
    Output: genauester Wert und Romberg-Tabelle.
    Voraussetzungen: Stuetzstellen streng steigend und ausreichend dicht fuer die
    gewaehlte Romberg-Ordnung.
    Imports: import pandas as pd.
    Numerische Hinweise: Romberg ist fuer glatte Funktionen gedacht. Bei linear
    interpolierten Tabellenwerten ist das Ergebnis eine Naeherung auf Basis der
    Interpolation, nicht der unbekannten Originalfunktion.
    Beispiel:
        >>> stuetzstellen = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        >>> stuetzwerte = np.array([0.0, 1.0, 4.0, 10.0, 18.0])
        >>> ordnung_m = 2
        >>> wert, tabelle = romberg_extrapolation_aus_stuetzstellen(stuetzstellen, stuetzwerte, ordnung_m)
        >>> isinstance(wert, float)
        True
    """
    x = np.asarray(stuetzstellen, dtype=float).reshape(-1)
    y = np.asarray(stuetzwerte, dtype=float).reshape(-1)
    funktion = funktion_aus_stuetzstellen_linear(x, y)
    return romberg_extrapolation(funktion, x[0], x[-1], ordnung_m)


def _gauss_transformiere(knoten, gewichte, funktion, a, b):
    """Transformiert Gauss-Legendre-Knoten von [-1,1] auf [a,b] und summiert."""
    mitte = (a + b) / 2.0
    halb = (b - a) / 2.0
    wert = halb * sum(w * funktion(mitte + halb * x) for x, w in zip(knoten, gewichte))
    return float(wert)


def gauss_legendre_1(funktion, untere_grenze, obere_grenze) -> QuadraturErgebnis:
    """Berechnet Gauss-Legendre mit einem Stuetzpunkt.

    Formel auf [-1,1]: Integral ca. 2 f(0). Input: Funktion und Grenzen. Output: QuadraturErgebnis.
    Voraussetzungen: Funktion auswertbar. Hinweis: Exakt bis Polynomgrad 1.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2 - 7
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> untere_grenze = 0
        >>> obere_grenze = 2
        >>> ergebnis = gauss_legendre_1(funktion, untere_grenze, obere_grenze)
        >>> ergebnis.wert
        -12.0
    """
    a, b = float(untere_grenze), float(obere_grenze)
    return QuadraturErgebnis(_gauss_transformiere([0.0], [2.0], funktion, a, b), "Gauss-Legendre n=1", None, 1, None, {"knoten": [0.0], "gewichte": [2.0]})


def gauss_legendre_2(funktion, untere_grenze, obere_grenze) -> QuadraturErgebnis:
    """Berechnet Gauss-Legendre mit zwei Stuetzpunkten.

    Formel auf [-1,1]: f(-1/sqrt(3))+f(1/sqrt(3)). Input: Funktion, Grenzen. Output: QuadraturErgebnis.
    Voraussetzungen: Funktion auswertbar. Hinweis: Exakt bis Polynomgrad 3.
    Imports: import math.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> untere_grenze = 0
        >>> obere_grenze = 1
        >>> ergebnis = gauss_legendre_2(funktion, untere_grenze, obere_grenze)
        >>> round(ergebnis.wert, 12)
        0.333333333333
    """
    k = 1.0 / math.sqrt(3.0)
    a, b = float(untere_grenze), float(obere_grenze)
    return QuadraturErgebnis(_gauss_transformiere([-k, k], [1.0, 1.0], funktion, a, b), "Gauss-Legendre n=2", None, 2, None, {"knoten": [-k, k], "gewichte": [1.0, 1.0]})


def gauss_legendre_3(funktion, untere_grenze, obere_grenze) -> QuadraturErgebnis:
    """Berechnet Gauss-Legendre mit drei Stuetzpunkten.

    Formel auf [-1,1]: 5/9 f(-sqrt(3/5))+8/9 f(0)+5/9 f(sqrt(3/5)).
    Input: Funktion, Grenzen. Output: QuadraturErgebnis. Voraussetzungen: Funktion auswertbar.
    Imports: import math.
    Numerische Hinweise: Exakt bis Polynomgrad 5.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**4
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> untere_grenze = 0
        >>> obere_grenze = 1
        >>> ergebnis = gauss_legendre_3(funktion, untere_grenze, obere_grenze)
        >>> round(ergebnis.wert, 12)
        0.2
    """
    k = math.sqrt(3.0 / 5.0)
    knoten = [-k, 0.0, k]
    gewichte = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
    a, b = float(untere_grenze), float(obere_grenze)
    return QuadraturErgebnis(_gauss_transformiere(knoten, gewichte, funktion, a, b), "Gauss-Legendre n=3", None, 3, None, {"knoten": knoten, "gewichte": gewichte})


def gauss_legendre_allgemein(funktion, untere_grenze, obere_grenze, anzahl_stuetzstellen) -> QuadraturErgebnis:
    """Berechnet allgemeine Gauss-Legendre-Quadratur.

    Formel: Knoten und Gewichte aus Legendre-Polynomen, transformiert von [-1,1] auf [a,b].
    Input: Funktion, Grenzen, Anzahl n. Output: QuadraturErgebnis.
    Voraussetzungen: n positiv. Hinweis: Nutzt NumPy `leggauss` hinter deutscher API.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2 - 7
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> untere_grenze = 0
        >>> obere_grenze = 2
        >>> anzahl_stuetzstellen = 4
        >>> ergebnis = gauss_legendre_allgemein(funktion, untere_grenze, obere_grenze, anzahl_stuetzstellen)
        >>> round(ergebnis.wert, 12)
        -11.333333333333
    """
    n = int(anzahl_stuetzstellen)
    if n <= 0:
        raise ValueError("anzahl_stuetzstellen muss positiv sein.")
    knoten, gewichte = np.polynomial.legendre.leggauss(n)
    a, b = float(untere_grenze), float(obere_grenze)
    return QuadraturErgebnis(_gauss_transformiere(knoten, gewichte, funktion, a, b), f"Gauss-Legendre n={n}", None, n, None, {"knoten": knoten, "gewichte": gewichte})

# ===== differentialgleichungen.py =====
"""Differentialgleichungen, Einschrittverfahren, Systeme und Standardmodelle."""






def plot_basis(x_label="x", y_label="y", titel="", grid=True):
    """Erstellt eine einfache Matplotlib-Achse mit Beschriftung, Titel und Grid.

    Zweck: Minimaler Startpunkt fuer Pruefungsplots, ohne komplexe Wrapper-Logik.
    Formel/Verfahren: `plt.subplots()`, danach `set_xlabel`, `set_ylabel`, `set_title`, `grid`.
    Input: Achsenbeschriftungen, Titel und Grid-Flag.
    Output: Matplotlib-Achse `ax`.
    Voraussetzungen: `matplotlib.pyplot as plt` ist importiert.
    Beispiel:
        >>> ax = plot_basis(x_label="Zeit t", y_label="Wert y", titel="Einfacher Plot")
        >>> ax.get_xlabel()
        'Zeit t'
    """
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(titel)
    ax.grid(grid)
    return ax


def nullniveau_zwei_funktionen_plotten(f1, f2, x_bereich=(0, 1), y_bereich=(0, 1), anzahl=100, labels=("f1", "f2"), farben=("r", "b"), titel="Grafische Loesung von f1=0 und f2=0"):
    """Plottet die Nullniveau-Konturen zweier Funktionen f1(x,y)=0 und f2(x,y)=0.

    Zweck: Direkte grafische Loesung eines nichtlinearen Gleichungssystems mit zwei
    Gleichungen, nahe am Standard-Pruefungscode mit `meshgrid` und `plt.contour`.
    Formel/Verfahren: X,Y-Gitter erzeugen, f1(X,Y) und f2(X,Y) auswerten, Level 0 plotten.
    Input: zwei Funktionen f1(X,Y), f2(X,Y), x-/y-Bereich, Gitteranzahl, Labels und Farben.
    Output: Matplotlib-Achse `ax`.
    Voraussetzungen: f1 und f2 akzeptieren NumPy-Arrays X,Y.
    Beispiel:
        >>> def f1(X, Y):
        ...     return X**2 + Y**2 - 1
        >>> def f2(X, Y):
        ...     return X - Y
        >>> x_bereich = (-1.5, 1.5)
        >>> y_bereich = (-1.5, 1.5)
        >>> ax = nullniveau_zwei_funktionen_plotten(f1, f2, x_bereich, y_bereich)
        >>> ax.get_title()
        'Grafische Loesung von f1=0 und f2=0'
    """
    x_werte = np.linspace(float(x_bereich[0]), float(x_bereich[1]), int(anzahl))
    y_werte = np.linspace(float(y_bereich[0]), float(y_bereich[1]), int(anzahl))
    X, Y = np.meshgrid(x_werte, y_werte)

    ax = plot_basis("x", "y", titel, grid=True)
    c1 = ax.contour(X, Y, f1(X, Y), levels=[0], colors=farben[0])
    h1, _ = c1.legend_elements()
    c2 = ax.contour(X, Y, f2(X, Y), levels=[0], colors=farben[1])
    h2, _ = c2.legend_elements()
    ax.legend([h1[0], h2[0]], [labels[0], labels[1]])
    return ax


def dgl_ergebnis_plotten(ergebnis, titel=None, label=None, marker="o"):
    """Plottet das Ergebnis eines DGL-Verfahrens wie Euler, Heun oder Runge-Kutta.

    Zweck: Ergebnisobjekte von `euler_verfahren`, `heun_verfahren`, `runge_kutta_4`
    oder den Systemvarianten schnell sichtbar machen.
    Formel/Verfahren: `ergebnis.x_werte` gegen `ergebnis.y_werte` plotten.
    Input: DglErgebnis, optionaler Titel, Label und Marker.
    Output: Matplotlib-Achse `ax`.
    Voraussetzungen: `ergebnis` besitzt `x_werte`, `y_werte` und `methode`.
    Beispiel:
        >>> x, y = sp.symbols("x y")
        >>> f_expr = x**2 - y + 1
        >>> funktion = sp.lambdify((x, y), f_expr, modules="numpy")
        >>> x_start = 0
        >>> y_start = 1
        >>> x_ende = 1
        >>> anzahl_schritte = 10
        >>> ergebnis = euler_verfahren(funktion, x_start, y_start, x_ende, anzahl_schritte)
        >>> ax = dgl_ergebnis_plotten(ergebnis, titel="Euler-Verfahren")
        >>> ax.get_ylabel()
        'y'
    """
    plot_titel = titel if titel is not None else ergebnis.methode
    ax = plot_basis("x", "y", plot_titel, grid=True)
    x_werte = np.asarray(ergebnis.x_werte, dtype=float)
    y_werte = np.asarray(ergebnis.y_werte, dtype=float)
    beschriftung = label if label is not None else ergebnis.methode
    if y_werte.ndim == 1:
        ax.plot(x_werte, y_werte, marker=marker, label=beschriftung)
    else:
        for spalte in range(y_werte.shape[1]):
            ax.plot(x_werte, y_werte[:, spalte], marker=marker, label=f"{beschriftung} y_{spalte}")
    ax.legend()
    return ax


def _dgl_ergebnis(x_werte, y_werte, methode, schrittweite, ordnung):
    """Erstellt ein DglErgebnis inklusive passender Pandas-Tabelle.
    
    Imports: import pandas as pd.
    """
    y = np.asarray(y_werte, dtype=float)
    daten = {"x": x_werte}
    if y.ndim == 1:
        daten["y"] = y
    else:
        for i in range(y.shape[1]):
            daten[f"y_{i}"] = y[:, i]
    return DglErgebnis(np.asarray(x_werte, dtype=float), y, methode, schrittweite, ordnung, pd.DataFrame(daten))


def euler_verfahren(funktion, x_start, y_start, x_ende, anzahl_schritte) -> DglErgebnis:
    """Loest ein AWP mit dem expliziten Euler-Verfahren.

    Formel: y_{i+1}=y_i+h f(x_i,y_i). Input: f,x0,y0,xEnde,n. Output: DglErgebnis.
    Voraussetzungen: n positiv. Hinweis: Globaler Fehler Ordnung 1.
    Beispiel:
        >>> x, y = sp.symbols("x y")
        >>> f_expr = x**2 - y + 1
        >>> funktion = sp.lambdify((x, y), f_expr, modules="numpy")
        >>> x_start = 0
        >>> y_start = 1
        >>> x_ende = 1
        >>> anzahl_schritte = 10
        >>> ergebnis = euler_verfahren(funktion, x_start, y_start, x_ende, anzahl_schritte)
        >>> ergebnis.y_werte[-1] > 1
        True
    """
    if int(anzahl_schritte) <= 0:
        raise ValueError("anzahl_schritte muss positiv sein.")
    n = int(anzahl_schritte)
    x = np.linspace(float(x_start), float(x_ende), n + 1)
    h = (float(x_ende) - float(x_start)) / n
    y = np.zeros(n + 1, dtype=float)
    y[0] = float(y_start)
    for i in range(n):
        y[i + 1] = y[i] + h * funktion(x[i], y[i])
    return _dgl_ergebnis(x, y, "Euler", h, 1)


def mittelpunkt_verfahren(funktion, x_start, y_start, x_ende, anzahl_schritte) -> DglErgebnis:
    """Loest ein AWP mit dem Mittelpunktverfahren.

    Formel: y_h/2=y_i+h/2 f(x_i,y_i), y_{i+1}=y_i+h f(x_i+h/2,y_h/2).
    Input: f,x0,y0,xEnde,n. Output: DglErgebnis. Voraussetzungen: n positiv.
    Numerische Hinweise: Globaler Fehler Ordnung 2.
    Beispiel:
        >>> x, y = sp.symbols("x y")
        >>> f_expr = x**2 - y + 1
        >>> funktion = sp.lambdify((x, y), f_expr, modules="numpy")
        >>> x_start = 0
        >>> y_start = 1
        >>> x_ende = 1
        >>> anzahl_schritte = 10
        >>> ergebnis = mittelpunkt_verfahren(funktion, x_start, y_start, x_ende, anzahl_schritte)
        >>> ergebnis.ordnung
        2
    """
    if int(anzahl_schritte) <= 0:
        raise ValueError("anzahl_schritte muss positiv sein.")
    n = int(anzahl_schritte)
    x = np.linspace(float(x_start), float(x_ende), n + 1)
    h = (float(x_ende) - float(x_start)) / n
    y = np.zeros(n + 1, dtype=float)
    y[0] = float(y_start)
    for i in range(n):
        k1 = funktion(x[i], y[i])
        y_mitte = y[i] + 0.5 * h * k1
        y[i + 1] = y[i] + h * funktion(x[i] + 0.5 * h, y_mitte)
    return _dgl_ergebnis(x, y, "Mittelpunkt", h, 2)


def heun_verfahren(funktion, x_start, y_start, x_ende, anzahl_schritte) -> DglErgebnis:
    """Loest ein AWP mit dem Heun-Verfahren.

    Formel: k1=f(x_i,y_i), k2=f(x_i+h,y_i+h k1), y_{i+1}=y_i+h(k1+k2)/2.
    Input: f,x0,y0,xEnde,n. Output: DglErgebnis. Voraussetzungen: n positiv.
    Numerische Hinweise: Explizites Verfahren zweiter Ordnung.
    Beispiel:
        >>> x, y = sp.symbols("x y")
        >>> f_expr = x**2 - y + 1
        >>> funktion = sp.lambdify((x, y), f_expr, modules="numpy")
        >>> x_start = 0
        >>> y_start = 1
        >>> x_ende = 1
        >>> anzahl_schritte = 10
        >>> ergebnis = heun_verfahren(funktion, x_start, y_start, x_ende, anzahl_schritte)
        >>> ergebnis.ordnung
        2
    """
    if int(anzahl_schritte) <= 0:
        raise ValueError("anzahl_schritte muss positiv sein.")
    n = int(anzahl_schritte)
    x = np.linspace(float(x_start), float(x_ende), n + 1)
    h = (float(x_ende) - float(x_start)) / n
    y = np.zeros(n + 1, dtype=float)
    y[0] = float(y_start)
    for i in range(n):
        k1 = funktion(x[i], y[i])
        k2 = funktion(x[i] + h, y[i] + h * k1)
        y[i + 1] = y[i] + h * (k1 + k2) / 2.0
    return _dgl_ergebnis(x, y, "Heun", h, 2)


def runge_kutta_4(funktion, x_start, y_start, x_ende, anzahl_schritte) -> DglErgebnis:
    """Loest ein AWP mit klassischem Runge-Kutta 4.

    Formel: k1=f(x,y), k2=f(x+h/2,y+h k1/2), k3=f(x+h/2,y+h k2/2),
    k4=f(x+h,y+h k3), y_{i+1}=y_i+h(k1+2k2+2k3+k4)/6.
    Input: f,x0,y0,xEnde,n. Output: DglErgebnis. Voraussetzungen: n positiv.
    Imports: import math.
    Numerische Hinweise: Globaler Fehler Ordnung 4.
    Beispiel:
        >>> x, y = sp.symbols("x y")
        >>> f_expr = x**2 - y + 1
        >>> funktion = sp.lambdify((x, y), f_expr, modules="numpy")
        >>> x_start = 0
        >>> y_start = 1
        >>> x_ende = 1
        >>> anzahl_schritte = 10
        >>> ergebnis = runge_kutta_4(funktion, x_start, y_start, x_ende, anzahl_schritte)
        >>> ergebnis.y_werte[-1] > 1.2
        True
    """
    if int(anzahl_schritte) <= 0:
        raise ValueError("anzahl_schritte muss positiv sein.")
    n = int(anzahl_schritte)
    x = np.linspace(float(x_start), float(x_ende), n + 1)
    h = (float(x_ende) - float(x_start)) / n
    y = np.zeros(n + 1, dtype=float)
    y[0] = float(y_start)
    for i in range(n):
        k1 = funktion(x[i], y[i])
        k2 = funktion(x[i] + h / 2.0, y[i] + h * k1 / 2.0)
        k3 = funktion(x[i] + h / 2.0, y[i] + h * k2 / 2.0)
        k4 = funktion(x[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return _dgl_ergebnis(x, y, "Runge-Kutta 4", h, 4)


def explizites_runge_kutta_verfahren(funktion, x_start, y_start, x_ende, anzahl_schritte, butcher_a, butcher_b, butcher_c, methode_name="explizites Runge-Kutta") -> DglErgebnis:
    """Loest ein AWP mit einem frei vorgegebenen expliziten Runge-Kutta-Schema.

    Formel: k_i=f(x_n+c_i h, y_n+h sum_{j<i} a_ij k_j),
    y_{n+1}=y_n+h sum_i b_i k_i.
    Input: Funktion f(x,y), Startdaten, Schrittzahl und Butcher-Koeffizienten A,b,c.
    Output: DglErgebnis fuer skalare oder vektorielle y-Werte.
    Voraussetzungen: A ist s x s, b und c haben Laenge s; das Schema ist explizit,
    also werden nur bereits berechnete Stufen j<i verwendet.
    Numerische Hinweise: Die Genauigkeitsordnung haengt vom Butcher-Schema ab; das
    Verfahren validiert nicht automatisch alle Ordnungbedingungen.
    Beispiel:
        >>> x, y = sp.symbols("x y")
        >>> f_expr = x**2 - y + 1
        >>> funktion = sp.lambdify((x, y), f_expr, modules="numpy")
        >>> x_start = 0
        >>> y_start = 1
        >>> x_ende = 1
        >>> anzahl_schritte = 10
        >>> butcher_a = [[0, 0], [1, 0]]
        >>> butcher_b = [0.5, 0.5]
        >>> butcher_c = [0, 1]
        >>> erg = explizites_runge_kutta_verfahren(
        ...     funktion, x_start, y_start, x_ende, anzahl_schritte,
        ...     butcher_a, butcher_b, butcher_c
        ... )
        >>> erg.y_werte[-1] > 1.2
        True
    """
    if int(anzahl_schritte) <= 0:
        raise ValueError("anzahl_schritte muss positiv sein.")
    a_mat = np.asarray(butcher_a, dtype=float)
    b_vec = np.asarray(butcher_b, dtype=float).reshape(-1)
    c_vec = np.asarray(butcher_c, dtype=float).reshape(-1)
    if a_mat.shape != (b_vec.size, b_vec.size) or c_vec.size != b_vec.size:
        raise ValueError("Butcher-Koeffizienten haben inkompatible Dimensionen.")
    n = int(anzahl_schritte)
    x = np.linspace(float(x_start), float(x_ende), n + 1)
    h = (float(x_ende) - float(x_start)) / n
    y0_array = np.asarray(y_start, dtype=float)
    skalar = y0_array.ndim == 0
    y0 = np.array([float(y_start)]) if skalar else np.asarray(y_start, dtype=float).reshape(-1)
    y = np.zeros((n + 1, y0.size), dtype=float)
    y[0] = y0
    for schritt in range(n):
        stufen = []
        for i in range(b_vec.size):
            gewichtete_summe = np.zeros_like(y0)
            for j in range(i):
                gewichtete_summe += a_mat[i, j] * stufen[j]
            argument_y = y[schritt] + h * gewichtete_summe
            funktionswert = funktion(x[schritt] + c_vec[i] * h, argument_y[0] if skalar else argument_y)
            stufen.append(np.array([float(funktionswert)]) if skalar else np.asarray(funktionswert, dtype=float).reshape(-1))
        kombination = np.zeros_like(y0)
        for gewicht, stufe in zip(b_vec, stufen):
            kombination += gewicht * stufe
        y[schritt + 1] = y[schritt] + h * kombination
    y_rueckgabe = y[:, 0] if skalar else y
    return _dgl_ergebnis(x, y_rueckgabe, methode_name, h, 0)


def euler_verfahren_system(funktion, x_start, y_startvektor, x_ende, anzahl_schritte) -> DglErgebnis:
    """Loest ein DGL-System mit Euler.

    Formel: y_{i+1}=y_i+h f(x_i,y_i). Input: f, x0, Startvektor, xEnde, n. Output: DglErgebnis.
    Voraussetzungen: f liefert Vektor gleicher Dimension. Hinweis: Ordnung 1.
    Beispiel:
        >>> x, y1 = sp.symbols("x y1")
        >>> f_expr = sp.Matrix([x**2 - y1 + 1])
        >>> funktion_basis = sp.lambdify((x, y1), f_expr, modules="numpy")
        >>> def funktion(x_wert, y_vektor):
        ...     return np.asarray(funktion_basis(x_wert, y_vektor[0]), dtype=float).reshape(-1)
        >>> x_start = 0
        >>> y_startvektor = np.array([1.0])
        >>> x_ende = 1
        >>> anzahl_schritte = 10
        >>> ergebnis = euler_verfahren_system(funktion, x_start, y_startvektor, x_ende, anzahl_schritte)
        >>> ergebnis.y_werte[-1,0] > 1
        True
    """
    return _system_verfahren(funktion, x_start, y_startvektor, x_ende, anzahl_schritte, "Euler System", 1)


def mittelpunkt_verfahren_system(funktion, x_start, y_startvektor, x_ende, anzahl_schritte) -> DglErgebnis:
    """Loest ein DGL-System mit Mittelpunktverfahren.

    Formel: y_m=y_i+h/2 f(x_i,y_i), y_{i+1}=y_i+h f(x_i+h/2,y_m).
    Input: f,x0,Startvektor,xEnde,n. Output: DglErgebnis. Voraussetzungen: f vektorwertig.
    Numerische Hinweise: Ordnung 2.
    Beispiel:
        >>> x, y1 = sp.symbols("x y1")
        >>> f_expr = sp.Matrix([x**2 - y1 + 1])
        >>> funktion_basis = sp.lambdify((x, y1), f_expr, modules="numpy")
        >>> def funktion(x_wert, y_vektor):
        ...     return np.asarray(funktion_basis(x_wert, y_vektor[0]), dtype=float).reshape(-1)
        >>> x_start = 0
        >>> y_startvektor = np.array([1.0])
        >>> x_ende = 1
        >>> anzahl_schritte = 10
        >>> ergebnis = mittelpunkt_verfahren_system(funktion, x_start, y_startvektor, x_ende, anzahl_schritte)
        >>> ergebnis.ordnung
        2
    """
    return _system_verfahren(funktion, x_start, y_startvektor, x_ende, anzahl_schritte, "Mittelpunkt System", 2)


def heun_verfahren_system(funktion, x_start, y_startvektor, x_ende, anzahl_schritte) -> DglErgebnis:
    """Loest ein DGL-System mit Heun.

    Formel: k1=f(x,y), k2=f(x+h,y+h k1), y_{i+1}=y_i+h(k1+k2)/2.
    Input: f,x0,Startvektor,xEnde,n. Output: DglErgebnis. Voraussetzungen: f vektorwertig.
    Numerische Hinweise: Ordnung 2.
    Beispiel:
        >>> x, y1 = sp.symbols("x y1")
        >>> f_expr = sp.Matrix([x**2 - y1 + 1])
        >>> funktion_basis = sp.lambdify((x, y1), f_expr, modules="numpy")
        >>> def funktion(x_wert, y_vektor):
        ...     return np.asarray(funktion_basis(x_wert, y_vektor[0]), dtype=float).reshape(-1)
        >>> x_start = 0
        >>> y_startvektor = np.array([1.0])
        >>> x_ende = 1
        >>> anzahl_schritte = 10
        >>> ergebnis = heun_verfahren_system(funktion, x_start, y_startvektor, x_ende, anzahl_schritte)
        >>> ergebnis.ordnung
        2
    """
    return _system_verfahren(funktion, x_start, y_startvektor, x_ende, anzahl_schritte, "Heun System", 2)


def runge_kutta_4_system(funktion, x_start, y_startvektor, x_ende, anzahl_schritte) -> DglErgebnis:
    """Loest ein DGL-System mit RK4.

    Formel: Vektorielle RK4-Kombination k1,k2,k3,k4. Input: f,x0,Startvektor,xEnde,n.
    Output: DglErgebnis. Voraussetzungen: f liefert passenden Vektor.
    Numerische Hinweise: Ordnung 4 und meist gute Pruefungswahl.
    Beispiel:
        >>> x, y1 = sp.symbols("x y1")
        >>> f_expr = sp.Matrix([x**2 - y1 + 1])
        >>> funktion_basis = sp.lambdify((x, y1), f_expr, modules="numpy")
        >>> def funktion(x_wert, y_vektor):
        ...     return np.asarray(funktion_basis(x_wert, y_vektor[0]), dtype=float).reshape(-1)
        >>> x_start = 0
        >>> y_startvektor = np.array([1.0])
        >>> x_ende = 1
        >>> anzahl_schritte = 10
        >>> ergebnis = runge_kutta_4_system(funktion, x_start, y_startvektor, x_ende, anzahl_schritte)
        >>> ergebnis.ordnung
        4
    """
    return _system_verfahren(funktion, x_start, y_startvektor, x_ende, anzahl_schritte, "Runge-Kutta 4 System", 4)


def _system_verfahren(funktion, x_start, y_startvektor, x_ende, anzahl_schritte, methode, ordnung):
    """Gemeinsamer Integrationskern fuer Euler, Mittelpunkt, Heun und RK4 bei Systemen."""
    if int(anzahl_schritte) <= 0:
        raise ValueError("anzahl_schritte muss positiv sein.")
    n = int(anzahl_schritte)
    x = np.linspace(float(x_start), float(x_ende), n + 1)
    h = (float(x_ende) - float(x_start)) / n
    y0 = np.asarray(y_startvektor, dtype=float).reshape(-1)
    y = np.zeros((n + 1, y0.size), dtype=float)
    y[0] = y0
    for i in range(n):
        if ordnung == 1:
            y[i + 1] = y[i] + h * np.asarray(funktion(x[i], y[i]), dtype=float).reshape(-1)
        elif methode.startswith("Mittelpunkt"):
            k1 = np.asarray(funktion(x[i], y[i]), dtype=float).reshape(-1)
            y[i + 1] = y[i] + h * np.asarray(funktion(x[i] + h / 2.0, y[i] + h * k1 / 2.0), dtype=float).reshape(-1)
        elif methode.startswith("Heun"):
            k1 = np.asarray(funktion(x[i], y[i]), dtype=float).reshape(-1)
            k2 = np.asarray(funktion(x[i] + h, y[i] + h * k1), dtype=float).reshape(-1)
            y[i + 1] = y[i] + h * (k1 + k2) / 2.0
        else:
            k1 = np.asarray(funktion(x[i], y[i]), dtype=float).reshape(-1)
            k2 = np.asarray(funktion(x[i] + h / 2.0, y[i] + h * k1 / 2.0), dtype=float).reshape(-1)
            k3 = np.asarray(funktion(x[i] + h / 2.0, y[i] + h * k2 / 2.0), dtype=float).reshape(-1)
            k4 = np.asarray(funktion(x[i] + h, y[i] + h * k3), dtype=float).reshape(-1)
            y[i + 1] = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return _dgl_ergebnis(x, y, methode, h, ordnung)


def dgl_hoeherer_ordnung_zu_system(funktion_hoechste_ableitung, ordnung) -> callable:
    """Formt eine DGL hoeherer Ordnung in ein System erster Ordnung um.

    Formel: z_1=y, z_2=y', ..., z_n=y^(n-1), z_1'=z_2, ..., z_n'=f(x,z_1,...,z_n).
    Input: Funktion der hoechsten Ableitung und Ordnung. Output: callable f(x,z).
    Voraussetzungen: Ordnung >= 1. Hinweis: Danach mit RK4-System loesbar.
    Beispiel:
        >>> def hoechste_ableitung(x, zustand):
        ...     return -zustand[0]
        >>> system = dgl_hoeherer_ordnung_zu_system(hoechste_ableitung, 2)
        >>> system(0, [1, 0]).tolist()
        [0.0, -1.0]
    """
    n = int(ordnung)
    if n < 1:
        raise ValueError("ordnung muss mindestens 1 sein.")
    def system(x, zustand):
        """Wertet das aus einer hoeheren DGL erzeugte System erster Ordnung aus."""
        z = np.asarray(zustand, dtype=float).reshape(-1)
        return np.concatenate([z[1:], [funktion_hoechste_ableitung(x, z)]])
    return system


def stabilitaetsfunktion_euler(z) -> float:
    """Berechnet die Euler-Stabilitaetsfunktion.

    Formel: R(z)=1+z, fuer y'=-alpha y ist z=-alpha h. Input: z. Output: float.
    Voraussetzungen: skalar. Hinweis: Stabil wenn |R(z)|<1.
    Beispiel:
        >>> stabilitaetsfunktion_euler(-1)
        0.0
    """
    return float(1.0 + z)


def ist_euler_stabil(alpha, schrittweite) -> bool:
    """Prueft Euler-Stabilitaet fuer y'=-alpha y.

    Formel: |1-alpha h|<1. Input: alpha,h. Output: bool.
    Voraussetzungen: alpha>0, h>0. Hinweis: Rand |.|=1 ist nicht asymptotisch stabil.
    Beispiel:
        >>> ist_euler_stabil(2, 0.5)
        True
    """
    return abs(1.0 - float(alpha) * float(schrittweite)) < 1.0


def stabilitaetsintervall_euler(alpha) -> tuple[float, float]:
    """Gibt das stabile h-Intervall fuer Euler an.

    Formel: 0<h<2/alpha. Input: alpha. Output: (0,2/alpha).
    Voraussetzungen: alpha>0. Hinweis: Je groesser alpha, desto kleiner h.
    Beispiel:
        >>> stabilitaetsintervall_euler(2)
        (0.0, 1.0)
    """
    return (0.0, 2.0 / float(alpha))


def globalen_fehler_berechnen(exakte_loesung, x_werte, y_werte) -> np.ndarray:
    """Berechnet globale Fehler an Gitterpunkten.

    Formel: e_i=y_exakt(x_i)-y_i. Input: exakte Loesung, x, y. Output: Fehlerarray.
    Voraussetzungen: gleiche Laengen. Hinweis: Betrag nehmen fuer Fehlernormen.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> exakt_expr = x**2 - 1
        >>> exakte_loesung = sp.lambdify(x, exakt_expr, modules="numpy")
        >>> x_werte = np.array([0.0, 1.0])
        >>> y_werte = np.array([-1.0, -0.5])
        >>> globalen_fehler_berechnen(exakte_loesung, x_werte, y_werte).tolist()
        [0.0, 0.5]
    """
    x = np.asarray(x_werte, dtype=float).reshape(-1)
    y = np.asarray(y_werte, dtype=float)
    exakt = np.array([exakte_loesung(xi) for xi in x], dtype=float)
    return exakt - y


def konvergenzordnung_schaetzen(fehler_grob, fehler_fein, faktor=2) -> float:
    """Schaetzt die Konvergenzordnung.

    Formel: p=log(E_grob/E_fein)/log(faktor). Input: zwei Fehler und Faktor. Output: float.
    Voraussetzungen: Fehler positiv. Hinweis: Sinnvoll bei asymptotischem Verhalten.
    Imports: import math.
    Beispiel:
        >>> konvergenzordnung_schaetzen(0.25, 0.0625)
        2.0
    """
    return float(math.log(float(fehler_grob) / float(fehler_fein)) / math.log(float(faktor)))


def boeing_brems_system(masse=97000, konstante_bremskraft=570000, luftterm=5):
    """Erzeugt ein Bremsmodell fuer eine Boeing.

    Formel: s'=v, v'=-(F+c v^2)/m. Input: Parameter. Output: Systemfunktion.
    Voraussetzungen: Masse positiv. Hinweis: Vereinfachtes physikalisches Modell.
    Beispiel:
        >>> system = boeing_brems_system()
        >>> zustand = np.array([0.0, 70.0])
        >>> len(system(0, zustand))
        2
    """
    def system(t, zustand):
        """Wertet das Boeing-Bremsmodell fuer Zustand [s, v] aus."""
        s, v = zustand
        return np.array([v, -(konstante_bremskraft + luftterm * v * abs(v)) / masse])
    return system


def raketen_beschleunigung(t, v_rel=2600, startmasse=300000, endmasse=80000, brenndauer=190, g=9.81):
    """Berechnet eine einfache Raketenbeschleunigung.

    Formel: a=v_rel*(-m'(t))/m(t)-g. Input: Zeit und Parameter. Output: float.
    Voraussetzungen: 0<=t<=Brenndauer sinnvoll. Hinweis: Nach Brennschluss nur -g in diesem Modell.
    Beispiel:
        >>> isinstance(raketen_beschleunigung(0), float)
        True
    """
    t = float(t)
    if t > brenndauer:
        return -float(g)
    massenstrom = (startmasse - endmasse) / brenndauer
    masse = startmasse - massenstrom * t
    return float(v_rel * massenstrom / masse - g)


def raketen_system(v_rel=2600, startmasse=300000, endmasse=80000, brenndauer=190, g=9.81):
    """Erzeugt ein Raketenmodell fuer Hoehe und Geschwindigkeit.

    Formel: h'=v, v'=a(t). Input: Parameter. Output: Systemfunktion.
    Voraussetzungen: Parameter physikalisch sinnvoll. Hinweis: Vereinfachtes Vertikalmodell.
    Beispiel:
        >>> len(raketen_system()(0, [0, 0]))
        2
    """
    def system(t, z):
        """Wertet das Raketen-System fuer Zustand [h, v] aus."""
        return np.array([z[1], raketen_beschleunigung(t, v_rel, startmasse, endmasse, brenndauer, g)])

    return system


def lotka_volterra_system(t, zustand, a=1.0, b=0.5, c=0.75, d=0.25):
    """Wertet das Lotka-Volterra-System aus.

    Formel: x'=a x-bxy, y'=-c y+dxy. Input: t,zustand,[Parameter]. Output: Vektor.
    Voraussetzungen: Zustand Laenge 2. Hinweis: Klassisches Raeuber-Beute-Modell.
    Beispiel:
        >>> lotka_volterra_system(0, [1, 1]).tolist()
        [0.5, -0.5]
    """
    x, y = zustand
    return np.array([a * x - b * x * y, -c * y + d * x * y], dtype=float)


def logistisches_wachstum_mit_stoerung(t, umsatz, stoer_amplitude=20):
    """Berechnet logistisches Wachstum mit sinusfoermiger Stoerung.

    Formel: u'=r u(1-u/K)+A sin(t) mit r=0.2,K=1000. Input: t, Umsatz. Output: float.
    Voraussetzungen: skalarer Umsatz. Hinweis: Beispielmodell fuer Pruefungs-DGL.
    Imports: import math.
    Beispiel:
        >>> isinstance(logistisches_wachstum_mit_stoerung(0, 100), float)
        True
    """
    return float(0.2 * umsatz * (1.0 - umsatz / 1000.0) + stoer_amplitude * math.sin(t))

# ===== fehleranalyse.py =====
"""Fehleranalyse, Kondition und LGS-Fehlerabschaetzung."""





def absoluter_fehler(naeherung, exakt) -> float:
    """Berechnet den absoluten Fehler.

    Formel: |x_tilde-x|. Input: Naeherung und exakter Wert. Output: float.
    Voraussetzungen: skalare Werte. Hinweis: Einheit entspricht der Groesse selbst.
    Beispiel:
        >>> absoluter_fehler(1.1, 1.0)
        0.10000000000000009
    """
    return abs(float(naeherung) - float(exakt))


def relativer_fehler(naeherung, exakt) -> float:
    """Berechnet den relativen Fehler.

    Formel: |x_tilde-x|/|x|. Input: Naeherung und exakt. Output: float.
    Voraussetzungen: exakt != 0. Hinweis: Dimensionslose Fehlergroesse.
    Beispiel:
        >>> relativer_fehler(1.1, 1.0)
        0.10000000000000009
    """
    if float(exakt) == 0:
        raise ValueError("Relativer Fehler bei exakt=0 nicht definiert.")
    return absoluter_fehler(naeherung, exakt) / abs(float(exakt))


def konditionszahl_funktion(funktion, ableitung, x_wert) -> float:
    """Berechnet die Konditionszahl einer skalaren Funktion.

    Formel: K=|f'(x)| |x| / |f(x)|. Input: f, f', x. Output: float.
    Voraussetzungen: f(x) != 0. Hinweis: Grosse K bedeutet empfindliche Auswertung.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> ableitung_expr = sp.diff(f_expr, x)
        >>> ableitung = sp.lambdify(x, ableitung_expr, modules="numpy")
        >>> konditionszahl_funktion(funktion, ableitung, 2)
        2.0
    """
    fx = float(funktion(x_wert))
    if fx == 0:
        return float("inf")
    return abs(float(ableitung(x_wert))) * abs(float(x_wert)) / abs(fx)


def absoluter_fehler_fortpflanzung(ableitung, x_wert, eingabefehler) -> float:
    """Schaetzt absolute Fehlerfortpflanzung.

    Formel: |Delta f| ca. |f'(x)| |Delta x|. Input: Ableitung, x, Eingabefehler. Output: float.
    Voraussetzungen: lineare Naeherung gueltig. Hinweis: Lokale Abschaetzung.
    Beispiel:
        >>> ableitung_expr = sp.diff(f_expr, x)
        >>> ableitung = sp.lambdify(x, ableitung_expr, modules="numpy")
        >>> absoluter_fehler_fortpflanzung(ableitung, 3, 0.1)
        0.6000000000000001
    """
    return abs(float(ableitung(x_wert))) * abs(float(eingabefehler))


def relativer_fehler_fortpflanzung(funktion, ableitung, x_wert, relativer_eingabefehler) -> float:
    """Schaetzt relative Fehlerfortpflanzung.

    Formel: relativer Ausgabefehler ca. K_f(x) * relativer Eingabefehler.
    Input: Funktion, Ableitung, x, relativer Fehler. Output: float.
    Voraussetzungen: f(x)!=0. Hinweis: Nutzt Konditionszahl.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> ableitung_expr = sp.diff(f_expr, x)
        >>> ableitung = sp.lambdify(x, ableitung_expr, modules="numpy")
        >>> relativer_fehler_fortpflanzung(funktion, ableitung, 2, 0.01)
        0.02
    """
    return konditionszahl_funktion(funktion, ableitung, x_wert) * abs(float(relativer_eingabefehler))


def lgs_absoluter_fehler_abschaetzung(matrix_a, stoerung_b, norm_ord=2) -> float:
    """Schaetzt absoluten LGS-Fehler bei Stoerung in b.

    Formel: ||x-x_tilde|| <= ||A^{-1}|| ||b-b_tilde||. Input: A, Stoerung b. Output: float.
    Voraussetzungen: A regulaer. Hinweis: Nur Stoerung der rechten Seite betrachtet.
    Beispiel:
        >>> lgs_absoluter_fehler_abschaetzung([[2]], [0.1])
        0.05
    """
    inv = matrix_inverse(matrix_a)
    db = np.asarray(stoerung_b, dtype=float).reshape(-1)
    return float(np.linalg.norm(inv, ord=norm_ord) * np.linalg.norm(db, ord=norm_ord))


def lgs_relativer_fehler_abschaetzung(matrix_a, vektor_b, stoerung_b, norm_ord=2) -> float:
    """Schaetzt relativen LGS-Fehler.

    Formel: ||dx||/||x|| <= cond(A) ||db||/||b||. Input: A,b,db. Output: float.
    Voraussetzungen: A regulaer, b != 0. Hinweis: Klassische Konditionsabschaetzung.
    Beispiel:
        >>> lgs_relativer_fehler_abschaetzung([[2]], [2], [0.1])
        0.05
    """
    b = np.asarray(vektor_b, dtype=float).reshape(-1)
    db = np.asarray(stoerung_b, dtype=float).reshape(-1)
    return float(konditionszahl_matrix(matrix_a, norm_ord) * np.linalg.norm(db, ord=norm_ord) / np.linalg.norm(b, ord=norm_ord))


# ===== analysis_hilfen.py =====
"""Analysis-Hilfen: Ableiten, Integrieren, Grenzwerte und Anwendungen."""





def ableitung_symbolisch(funktion, variable, ordnung=1) -> sp.Expr:
    """Berechnet eine symbolische Ableitung.

    Formel: d^n f / dx^n. Input: SymPy-Ausdruck, Variable, Ordnung. Output: SymPy-Ausdruck.
    Voraussetzungen: SymPy-Ausdruck. Hinweis: Exakt, solange SymPy vereinfachen kann.
    Beispiel:
        >>> x = sp.symbols('x')
        >>> f = x**2
        >>> ableitung_symbolisch(f, x)
        2*x
    """
    return sp.diff(funktion, variable, int(ordnung))


def partielle_ableitung_symbolisch(funktion, variable, ordnung=1) -> sp.Expr:
    """Berechnet eine partielle Ableitung.

    Formel: partial^n f / partial variable^n. Input: Funktion, Variable, Ordnung. Output: Ausdruck.
    Voraussetzungen: Symbolische Variablen. Hinweis: Wrapper fuer `sp.diff`.
    Beispiel:
        >>> x, y = sp.symbols('x y')
        >>> f = x*y
        >>> partielle_ableitung_symbolisch(f, x)
        y
    """
    return sp.diff(funktion, variable, int(ordnung))


def gradient_symbolisch(funktion, variablen) -> sp.Matrix:
    """Berechnet den symbolischen Gradienten.

    Formel: grad f=(df/dx_1,...,df/dx_n)^T. Input: Funktion, Variablen. Output: SymPy-Matrix.
    Voraussetzungen: skalarer Ausdruck. Hinweis: Grundlage fuer Linearisierung.
    Beispiel:
        >>> x, y = sp.symbols('x y')
        >>> f = x*y
        >>> gradient_symbolisch(f, [x, y])
        Matrix([[y], [x]])
    """
    return sp.Matrix([sp.diff(funktion, variable) for variable in variablen])


def hessenmatrix_symbolisch(funktion, variablen) -> sp.Matrix:
    """Berechnet die Hesse-Matrix.

    Formel: H_ij=d^2 f/(dx_i dx_j). Input: Funktion, Variablen. Output: SymPy-Matrix.
    Voraussetzungen: zweimal differenzierbar symbolisch. Hinweis: Fuer Extremstellen.
    Beispiel:
        >>> x, y = sp.symbols('x y')
        >>> f = x**2 + y**2
        >>> hessenmatrix_symbolisch(f, [x, y])
        Matrix([[2, 0], [0, 2]])
    """
    return sp.hessian(funktion, variablen)


def stammfunktion_symbolisch(funktion, variable) -> sp.Expr:
    """Berechnet eine Stammfunktion.

    Formel: F'=f. Input: f und Variable. Output: SymPy-Ausdruck.
    Voraussetzungen: Integral symbolisch bestimmbar. Hinweis: Konstante wird weggelassen.
    Beispiel:
        >>> x = sp.symbols('x')
        >>> f = 2*x
        >>> stammfunktion_symbolisch(f, x)
        x**2
    """
    return sp.integrate(funktion, variable)


def bestimmtes_integral_symbolisch(funktion, variable, untere_grenze, obere_grenze) -> sp.Expr:
    """Berechnet ein bestimmtes Integral symbolisch.

    Formel: int_a^b f(x) dx. Input: f, Variable, Grenzen. Output: SymPy-Wert.
    Voraussetzungen: Integral existiert. Hinweis: Exakte Werte moeglich.
    Beispiel:
        >>> x = sp.symbols('x')
        >>> f = x**2 - 7
        >>> untere_grenze = 0
        >>> obere_grenze = 1
        >>> bestimmtes_integral_symbolisch(f, x, untere_grenze, obere_grenze)
        -20/3
    """
    return sp.integrate(funktion, (variable, untere_grenze, obere_grenze))


def uneigentliches_integral_symbolisch(funktion, variable, untere_grenze, obere_grenze) -> sp.Expr:
    """Berechnet ein uneigentliches Integral symbolisch.

    Formel: Grenzwert des bestimmten Integrals mit unendlichen/singulaeren Grenzen.
    Input: f, Variable, Grenzen. Output: SymPy-Wert. Voraussetzungen: Konvergenz.
    Numerische Hinweise: Divergente Integrale bleiben oo/zoo oder unevaluated.
    Beispiel:
        >>> x = sp.symbols('x')
        >>> f = sp.exp(-x)
        >>> uneigentliches_integral_symbolisch(f, x, 0, sp.oo)
        1
    """
    return sp.integrate(funktion, (variable, untere_grenze, obere_grenze))


def grenzwert_symbolisch(funktion, variable, punkt, richtung="+-") -> sp.Expr:
    """Berechnet einen symbolischen Grenzwert.

    Formel: lim_{x->a} f(x), optional einseitig. Input: f, Variable, Punkt, Richtung. Output: Ausdruck.
    Voraussetzungen: SymPy kann den Grenzwert bestimmen. Hinweis: '+-' bedeutet zweiseitig.
    Beispiel:
        >>> x = sp.symbols('x')
        >>> f = sp.sin(x)/x
        >>> grenzwert_symbolisch(f, x, 0)
        1
    """
    return sp.limit(funktion, variable, punkt, dir=richtung)


def polynomdivision_symbolisch(polynom, divisor, variable) -> tuple[sp.Expr, sp.Expr]:
    """Fuehrt symbolische Polynomdivision durch.

    Formel: polynom = quotient*divisor + rest. Input: Polynom, Divisor, Variable. Output: (q,r).
    Voraussetzungen: Polynome in derselben Variable. Hinweis: Exakte SymPy-Rechnung.
    Beispiel:
        >>> x = sp.symbols('x')
        >>> polynom = x**2 - 1
        >>> divisor = x - 1
        >>> polynomdivision_symbolisch(polynom, divisor, x)
        (x + 1, 0)
    """
    return sp.div(polynom, divisor, domain="EX")


def mitternachtsformel(a, b, c) -> tuple[complex, complex]:
    """Loest eine quadratische Gleichung.

    Formel: x=(-b +- sqrt(b^2-4ac))/(2a). Input: a,b,c. Output: zwei komplexe Loesungen.
    Voraussetzungen: a != 0. Hinweis: Komplexe Loesungen werden unterstuetzt.
    Imports: import cmath.
    Beispiel:
        >>> mitternachtsformel(1, 0, -1)
        ((1+0j), (-1+0j))
    """
    diskriminante = b * b - 4 * a * c
    wurzel = cmath.sqrt(diskriminante)
    return ((-b + wurzel) / (2 * a), (-b - wurzel) / (2 * a))


def funktion_verkettung(funktion_f, funktion_g):
    """Erzeugt die Verkettung g o f.

    Formel: (g o f)(x)=g(f(x)). Input: zwei callables. Output: callable.
    Voraussetzungen: Wertebereich von f passt zum Definitionsbereich von g. Hinweis: Reihenfolge beachten.
    Beispiel:
        >>> def funktion_f(x):
        ...     return x + 1
        >>> def funktion_g(x):
        ...     return 2*x
        >>> verkettet = funktion_verkettung(funktion_f, funktion_g)
        >>> verkettet(3)
        8
    """
    def verkettete_funktion(x):
        """Wertet g(f(x)) aus."""
        return funktion_g(funktion_f(x))

    return verkettete_funktion


def ist_gerade_funktion_symbolisch(funktion, variable) -> bool:
    """Prueft symbolisch gerade Symmetrie.

    Formel: f(-x)=f(x). Input: Funktion, Variable. Output: bool.
    Voraussetzungen: SymPy kann vereinfachen. Hinweis: False kann bei schwer vereinfachbaren Ausdruecken auftreten.
    Beispiel:
        >>> x = sp.symbols('x')
        >>> f = x**2
        >>> ist_gerade_funktion_symbolisch(f, x)
        True
    """
    return bool(sp.simplify(funktion.subs(variable, -variable) - funktion) == 0)


def ist_ungerade_funktion_symbolisch(funktion, variable) -> bool:
    """Prueft symbolisch ungerade Symmetrie.

    Formel: f(-x)=-f(x). Input: Funktion, Variable. Output: bool.
    Voraussetzungen: SymPy kann vereinfachen. Hinweis: Nuetzlich fuer Integrale auf [-a,a].
    Beispiel:
        >>> x = sp.symbols('x')
        >>> f = x**3
        >>> ist_ungerade_funktion_symbolisch(f, x)
        True
    """
    return bool(sp.simplify(funktion.subs(variable, -variable) + funktion) == 0)


def trigonometrische_nullstellen_sinus(k_min, k_max) -> list:
    """Gibt Sinus-Nullstellen an.

    Formel: x=k*pi. Input: ganzzahlige Grenzen k_min,k_max. Output: Liste.
    Voraussetzungen: k_min<=k_max. Hinweis: Symbolische Vielfache von pi.
    Beispiel:
        >>> trigonometrische_nullstellen_sinus(0, 2)
        [0, pi, 2*pi]
    """
    return [k * sp.pi for k in range(int(k_min), int(k_max) + 1)]


def trigonometrische_nullstellen_cosinus(k_min, k_max) -> list:
    """Gibt Cosinus-Nullstellen an.

    Formel: x=pi/2+k*pi. Input: k-Grenzen. Output: Liste.
    Voraussetzungen: k_min<=k_max. Hinweis: Symbolische Vielfache von pi.
    Beispiel:
        >>> trigonometrische_nullstellen_cosinus(0, 1)
        [pi/2, 3*pi/2]
    """
    return [sp.pi / 2 + k * sp.pi for k in range(int(k_min), int(k_max) + 1)]


def logarithmus_basis(wert, basis) -> float:
    """Berechnet einen Logarithmus zu beliebiger Basis.

    Formel: log_b(x)=ln(x)/ln(b). Input: Wert und Basis. Output: float.
    Voraussetzungen: x>0, b>0, b!=1. Hinweis: Nutzt natuerlichen Logarithmus intern.
    Imports: import math.
    Beispiel:
        >>> logarithmus_basis(8, 2)
        3.0
    """
    return float(math.log(float(wert)) / math.log(float(basis)))


def _integriere_numerisch(funktion, a, b, n=2000):
    """Integriert eine skalare Funktion intern mit der summierten Trapezregel."""
    x = np.linspace(float(a), float(b), n + 1)
    y = np.array([funktion(xi) for xi in x], dtype=float)
    return float(np.trapezoid(y, x))


def funktionsmittelwert(funktion, untere_grenze, obere_grenze, numerisch=True) -> float:
    """Berechnet den Funktionsmittelwert.

    Formel: mu=1/(b-a) int_a^b f(x) dx. Input: Funktion, Grenzen. Output: float.
    Voraussetzungen: b!=a. Hinweis: numerisch=True nutzt Trapezapproximation.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2 - 7
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> untere_grenze = 0
        >>> obere_grenze = 1
        >>> round(funktionsmittelwert(funktion, untere_grenze, obere_grenze), 6)
        -6.666667
    """
    a, b = float(untere_grenze), float(obere_grenze)
    return _integriere_numerisch(funktion, a, b) / (b - a)


def flaeche_zwischen_kurven(funktion_f, funktion_g, schnittpunkte) -> float:
    """Berechnet die Flaeche zwischen zwei Kurven.

    Formel: A=sum int |f-g| dx zwischen aufeinanderfolgenden Schnittpunkten.
    Input: zwei Funktionen und sortierte Schnittpunkte. Output: float.
    Voraussetzungen: Schnittpunkte decken Flaechenabschnitte ab. Hinweis: Betrag vermeidet Vorzeichenfehler.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2 - 1
        >>> g_expr = 1 - x**2
        >>> funktion_f = sp.lambdify(x, f_expr, modules="numpy")
        >>> funktion_g = sp.lambdify(x, g_expr, modules="numpy")
        >>> schnittpunkte = np.array([-1.0, 1.0])
        >>> round(flaeche_zwischen_kurven(funktion_f, funktion_g, schnittpunkte), 6)
        2.666666
    """
    punkte = list(schnittpunkte)
    def integrand(x):
        """Betrag der Differenz der beiden Kurven."""
        return abs(funktion_f(x) - funktion_g(x))

    return float(sum(_integriere_numerisch(integrand, punkte[i], punkte[i + 1]) for i in range(len(punkte) - 1)))


def rotationsvolumen_um_x_achse(funktion, untere_grenze, obere_grenze) -> float:
    """Berechnet ein Rotationsvolumen um die x-Achse.

    Formel: V=pi int_a^b f(x)^2 dx. Input: Funktion, Grenzen. Output: float.
    Voraussetzungen: Funktion integrierbar. Hinweis: Vorzeichen von f verschwindet durch Quadrat.
    Imports: import math.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2 + 1
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> untere_grenze = 0
        >>> obere_grenze = 1
        >>> round(rotationsvolumen_um_x_achse(funktion, untere_grenze, obere_grenze), 6)
        5.864307
    """
    def integrand(x):
        """Integrand pi*f(x)^2 ohne den Faktor pi."""
        return funktion(x) ** 2

    return float(math.pi * _integriere_numerisch(integrand, untere_grenze, obere_grenze))


def bogenlaenge(funktion, ableitung, untere_grenze, obere_grenze, numerisch=True) -> float:
    """Berechnet die Bogenlaenge eines Graphen.

    Formel: L=int_a^b sqrt(1+f'(x)^2) dx. Input: Funktion, Ableitung, Grenzen. Output: float.
    Voraussetzungen: Ableitung integrierbar. Hinweis: Funktion selbst wird fuer die Formel nicht benoetigt.
    Imports: import math.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> ableitung_expr = sp.diff(f_expr, x)
        >>> ableitung = sp.lambdify(x, ableitung_expr, modules="numpy")
        >>> untere_grenze = 0
        >>> obere_grenze = 1
        >>> round(bogenlaenge(funktion, ableitung, untere_grenze, obere_grenze), 6)
        1.478943
    """
    def integrand(x):
        """Bogenlaengenintegrand."""
        return math.sqrt(1.0 + ableitung(x) ** 2)

    return _integriere_numerisch(integrand, untere_grenze, obere_grenze)


def mantelflaeche_rotation(funktion, ableitung, untere_grenze, obere_grenze, numerisch=True) -> float:
    """Berechnet die Mantelflaeche eines Rotationskoerpers.

    Formel: M=2pi int_a^b f(x)sqrt(1+f'(x)^2) dx. Input: Funktion, Ableitung, Grenzen.
    Output: float. Voraussetzungen: f nichtnegativ fuer geometrische Interpretation.
    Imports: import math.
    Numerische Hinweise: Numerische Trapezregel mit feinem Gitter.
    Beispiel:
        >>> x = sp.symbols("x")
        >>> f_expr = x**2 + 1
        >>> funktion = sp.lambdify(x, f_expr, modules="numpy")
        >>> ableitung_expr = sp.diff(f_expr, x)
        >>> ableitung = sp.lambdify(x, ableitung_expr, modules="numpy")
        >>> untere_grenze = 0
        >>> obere_grenze = 1
        >>> round(mantelflaeche_rotation(funktion, ableitung, untere_grenze, obere_grenze), 6)
        13.102203
    """
    def integrand(x):
        """Mantelflaechenintegrand ohne Faktor 2*pi."""
        return funktion(x) * math.sqrt(1.0 + ableitung(x) ** 2)

    return float(2.0 * math.pi * _integriere_numerisch(integrand, untere_grenze, obere_grenze))

# ===== daten_tabellen.py =====
"""Pandas-Tabellen fuer schnelle Pruefungsoutputs."""





def datenrahmen_aus_messwerten(x_werte, y_werte, x_name="x", y_name="y") -> pd.DataFrame:
    """Erstellt einen DataFrame aus Messwerten.

    Formel: tabelliert Paare (x_i,y_i). Input: x,y und Spaltennamen. Output: DataFrame.
    Voraussetzungen: gleiche Laenge. Hinweis: Basis fuer Regressionstabellen.
    Imports: import pandas as pd.
    Beispiel:
        >>> datenrahmen_aus_messwerten([0], [1]).shape
        (1, 2)
    """
    x = np.asarray(x_werte, dtype=float).reshape(-1)
    y = np.asarray(y_werte, dtype=float).reshape(-1)
    if len(x) != len(y):
        raise ValueError("x_werte und y_werte muessen gleich lang sein.")
    return pd.DataFrame({x_name: x, y_name: y})


def residuentabelle(stuetzstellen, messwerte, modellwerte) -> pd.DataFrame:
    """Erstellt eine Residuentabelle.

    Formel: residuum=y-yhat. Input: x, Messwerte, Modellwerte. Output: DataFrame.
    Voraussetzungen: gleiche Laengen. Hinweis: Residuen sollten zufaellig um 0 streuen.
    Imports: import pandas as pd.
    Beispiel:
        >>> residuentabelle([0], [1], [0.5])["residuum"].iloc[0]
        0.5
    """
    x = np.asarray(stuetzstellen, dtype=float).reshape(-1)
    y = np.asarray(messwerte, dtype=float).reshape(-1)
    m = np.asarray(modellwerte, dtype=float).reshape(-1)
    if not (len(x) == len(y) == len(m)):
        raise ValueError("stuetzstellen, messwerte und modellwerte muessen gleich lang sein.")
    return pd.DataFrame({"x": x, "messwert": y, "modellwert": m, "residuum": y - m})


def iterationstabelle(verlauf) -> pd.DataFrame:
    """Erstellt eine Tabelle aus Iterationsverlauf.

    Formel: Liste von Dictionaries -> DataFrame. Input: Verlauf. Output: DataFrame.
    Voraussetzungen: Verlauf tabellarisch. Hinweis: Ideal fuer Newton-Verfahren.
    Imports: import pandas as pd.
    Beispiel:
        >>> iterationstabelle([{"k": 0}]).shape
        (1, 1)
    """
    return pd.DataFrame(verlauf or [])


def quadraturvergleich_tabelle(ergebnisse) -> pd.DataFrame:
    """Vergleicht mehrere Quadraturergebnisse.

    Formel: pro Ergebnis eine Zeile mit Methode und Wert. Input: iterable Ergebnisse. Output: DataFrame.
    Voraussetzungen: Objekte haben passende Attribute oder dict-Struktur. Hinweis: Gut fuer Rechteck/Trapez/Simpson.
    Imports: import pandas as pd.
    Beispiel:
        >>> ergebnis = QuadraturErgebnis(1, "Testmethode")
        >>> quadraturvergleich_tabelle([ergebnis]).shape[0]
        1
    """
    zeilen = []
    for erg in ergebnisse:
        if hasattr(erg, "als_dict"):
            daten = erg.als_dict()
        else:
            daten = dict(erg)
        zeilen.append({k: daten.get(k) for k in ["methode", "wert", "schrittweite", "anzahl_intervalle", "fehlergrenze"]})
    return pd.DataFrame(zeilen)


def dgl_loesungstabelle(x_werte, y_werte, spaltennamen=None) -> pd.DataFrame:
    """Erstellt eine Tabelle fuer DGL-Loesungen.

    Formel: x_i und y_i bzw. Komponenten y_{i,j}. Input: x,y. Output: DataFrame.
    Voraussetzungen: Laengen passen. Hinweis: Systeme werden in mehrere Spalten zerlegt.
    Imports: import pandas as pd.
    Beispiel:
        >>> dgl_loesungstabelle([0], [1]).columns.tolist()
        ['x', 'y']
    """
    x = np.asarray(x_werte, dtype=float).reshape(-1)
    y = np.asarray(y_werte, dtype=float)
    if y.ndim == 1:
        return pd.DataFrame({"x": x, (spaltennamen[0] if spaltennamen else "y"): y})
    namen = spaltennamen or [f"y_{i}" for i in range(y.shape[1])]
    daten = {"x": x}
    for i, name in enumerate(namen):
        daten[name] = y[:, i]
    return pd.DataFrame(daten)


def exportiere_tabelle_csv(tabelle, dateiname) -> None:
    """Exportiert eine Tabelle als CSV.

    Formel: DataFrame.to_csv ohne Index. Input: Tabelle und Dateiname. Output: None.
    Voraussetzungen: Schreibrechte im Zielordner. Hinweis: UTF-8 fuer Umlaute.
    Imports: import pandas as pd.
    Beispiel:
        >>> exportiere_tabelle_csv(pd.DataFrame({"x":[1]}), "test.csv")
    """
    tabelle.to_csv(dateiname, index=False, encoding="utf-8")


# ===== physik_beispiele.py =====
"""Physik- und Standardbeispiele auf Basis der allgemeinen Toolbox-Funktionen."""






def fall_mit_luftwiderstand_geschwindigkeit(t, masse=80, g=9.81, k=0.25) -> float:
    """Berechnet Fallgeschwindigkeit mit quadratischem Luftwiderstand.

    Formel: v(t)=sqrt(mg/k)*tanh(sqrt(gk/m)t). Input: Zeit und Parameter. Output: float.
    Voraussetzungen: positive Parameter. Hinweis: Vereinfachtes Modell ab Ruhe.
    Imports: import math.
    Beispiel:
        >>> fall_mit_luftwiderstand_geschwindigkeit(0)
        0.0
    """
    return float(math.sqrt(masse * g / k) * math.tanh(math.sqrt(g * k / masse) * t))


def strecke_fall_mit_luftwiderstand_simpson(zeit_ende=10, intervalle=10) -> QuadraturErgebnis:
    """Berechnet Fallstrecke durch Simpson-Integration der Geschwindigkeit.

    Formel: s=int_0^T v(t) dt. Input: Endzeit und gerade Intervallzahl. Output: QuadraturErgebnis.
    Voraussetzungen: intervalle gerade. Hinweis: Nutzt allgemeine Simpson-Funktion.
    Beispiel:
        >>> strecke_fall_mit_luftwiderstand_simpson(1, 10).wert > 0
        True
    """
    return summierte_simpsonregel(fall_mit_luftwiderstand_geschwindigkeit, 0, zeit_ende, intervalle)


def erdmasse_aus_dichtetabelle(radius_km, dichte_kg_pro_m3) -> QuadraturErgebnis:
    """Schaetzt die Erdmasse aus radialer Dichtetabelle.

    Formel: M=4*pi int rho(r) r^2 dr. Input: Radius in km, Dichte kg/m^3. Output: QuadraturErgebnis.
    Voraussetzungen: Radien streng steigend. Hinweis: km werden in m umgerechnet.
    Imports: import math.
    Beispiel:
        >>> erdmasse_aus_dichtetabelle([0, 1], [1, 1]).wert > 0
        True
    """
    r_m = np.asarray(radius_km, dtype=float) * 1000.0
    rho = np.asarray(dichte_kg_pro_m3, dtype=float)
    werte = 4.0 * math.pi * rho * r_m**2
    return integriere_tabelle_nicht_aequidistant(r_m, werte)


def boeing_landung_beispiel(schrittweite=0.1, endzeit=20) -> DglErgebnis:
    """Berechnet ein Boeing-Bremsbeispiel.

    Formel: s'=v, v'=-(F+c v^2)/m. Input: Schrittweite, Endzeit. Output: DglErgebnis.
    Voraussetzungen: positive Schrittweite. Hinweis: Startgeschwindigkeit 80 m/s.
    Beispiel:
        >>> boeing_landung_beispiel(1, 2).y_werte.shape[1]
        2
    """
    schritte = int(round(endzeit / schrittweite))
    return runge_kutta_4_system(boeing_brems_system(), 0, [0, 80], endzeit, schritte)


def rakete_beispiel(endzeit=190, schritte=1900) -> DglErgebnis:
    """Berechnet ein Raketenbeispiel.

    Formel: h'=v, v'=a(t). Input: Endzeit und Schritte. Output: DglErgebnis.
    Voraussetzungen: Schritte positiv. Hinweis: Vereinfachtes Modell ohne Luftwiderstand.
    Beispiel:
        >>> rakete_beispiel(1, 10).y_werte.shape[1]
        2
    """
    return runge_kutta_4_system(raketen_system(), 0, [0, 0], endzeit, schritte)


def raeuber_beute_beispiel(endzeit=15, schritte=150) -> DglErgebnis:
    """Berechnet ein Raeuber-Beute-Beispiel.

    Formel: Lotka-Volterra-System. Input: Endzeit und Schritte. Output: DglErgebnis.
    Voraussetzungen: positive Startpopulationen. Hinweis: Qualitatives Standardmodell.
    Beispiel:
        >>> raeuber_beute_beispiel(1, 10).y_werte.shape[1]
        2
    """
    return runge_kutta_4_system(lotka_volterra_system, 0, [10, 5], endzeit, schritte)


def motorleistung_polynomfit_beispiel() -> AusgleichsErgebnis:
    """Fuehrt einen Polynomfit fuer Motorleistungsdaten aus.

    Formel: p_2(x) per kleinste Quadrate. Input: keine. Output: AusgleichsErgebnis.
    Voraussetzungen: Beispieldaten fest eingebaut. Hinweis: Zeigt QR-Polynomfit.
    Beispiel:
        >>> len(motorleistung_polynomfit_beispiel().koeffizienten)
        3
    """
    drehzahl = np.array([1000, 2000, 3000, 4000, 5000], dtype=float)
    leistung = np.array([20, 55, 88, 105, 110], dtype=float)
    return polynom_ausgleich(drehzahl, leistung, 2)


def exponentialfit_gauss_newton_beispiel() -> IterationsErgebnis:
    """Fuehrt einen Exponentialfit mit Gauss-Newton aus.

    Formel: a exp(-b x)+c. Input: keine. Output: IterationsErgebnis.
    Voraussetzungen: Beispieldaten fest eingebaut. Hinweis: Jacobi wird numerisch gebildet.
    Beispiel:
        >>> exponentialfit_gauss_newton_beispiel().loesung.shape[0]
        3
    """
    x = np.array([0, 1, 2, 3, 4], dtype=float)
    y = np.array([5.1, 3.8, 3.0, 2.5, 2.2], dtype=float)
    def jacobi_residuen(parameter):
        """Berechnet die numerische Residuen-Jacobi fuer das Beispiel."""
        return jacobi_residuen_numerisch(exponentialmodell, x, parameter)

    return gauss_newton_verfahren(
        exponentialmodell,
        jacobi_residuen,
        x,
        y,
        [3.0, 0.5, 2.0],
        maximale_iterationen=20,
    )

