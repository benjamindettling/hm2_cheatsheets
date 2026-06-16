# -*- coding: utf-8 -*-
"""
Polynom-Ausgleich mit Vandermonde-Matrix, Normalgleichung und QR
================================================================

Aufgabenstellung
----------------
Gegeben sind Messpunkte, die ungefähr auf einer kubischen Kurve liegen.
Für verschiedene Polynomgrade m soll eine Ausgleichsfunktion

    p_m(x) = lambda_0 + lambda_1*x + ... + lambda_m*x**m

bestimmt werden.

Berechnet werden pro Grad:
- Vandermonde-Matrix A
- Lösung mit Normalgleichung
- Lösung mit QR-Zerlegung
- Fehlerfunktional E = ||y - A lambda||_2^2
- Vergleich mit np.polyfit als Kontrolle

Geplottet werden:
- Messdaten und QR-Ausgleichspolynome verschiedener Grade
- Fehlerfunktional in Abhängigkeit vom Polynomgrad
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)

# %% 1) Gegebene Werte

xdat = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
ydat = np.array([3.30, 2.25, 1.92, 1.42, 1.03, 0.82, 0.86, 1.27, 2.18])

# Zu untersuchende Polynomgrade
grade = [1, 2, 3, 4]

x = sp.symbols("x")

# %% 2) Hilfsfunktionen für Vandermonde-Matrix und Auswertung

def vandermonde_matrix(xwerte, grad):
    """A_ij = x_i**j für j = 0, ..., grad."""
    return np.vander(xwerte, N=grad + 1, increasing=True)


def polynom_auswerten(xwerte, lam):
    """Wertet lambda_0 + lambda_1*x + ... + lambda_m*x**m aus."""
    ywerte = np.zeros_like(xwerte, dtype=float)
    for j in range(len(lam)):
        ywerte += lam[j] * xwerte**j
    return ywerte


def polynom_symbolisch(lam):
    """Erzeugt eine lesbare symbolische Darstellung des Polynoms."""
    return sp.expand(sum(lam[j] * x**j for j in range(len(lam))))

# %% 3) Polynom-Ausgleich für verschiedene Grade

resultate = []

print("Polynom-Ausgleich für verschiedene Grade")
print("=" * 55)

for grad in grade:
    A = vandermonde_matrix(xdat, grad)
    y = ydat.reshape(-1, 1)

    # Normalgleichung: A.T A lambda = A.T y
    lam_normal = np.linalg.solve(A.T @ A, A.T @ y).reshape(-1)

    # QR-Zerlegung: A = Q R, danach R lambda = Q.T y
    Q, R = np.linalg.qr(A, mode="reduced")
    lam_qr = np.linalg.solve(R, Q.T @ y).reshape(-1)

    # np.polyfit gibt Koeffizienten in absteigender Reihenfolge zurück.
    # Für den Vergleich drehen wir sie in die Reihenfolge lambda_0, ..., lambda_m.
    lam_polyfit = np.polyfit(xdat, ydat, deg=grad)[::-1]

    # Fehlerfunktional für die QR-Lösung
    yfit_qr = A @ lam_qr.reshape(-1, 1)
    residuen = y.reshape(-1) - yfit_qr.reshape(-1)
    E = np.linalg.norm(residuen, 2)**2

    resultate.append((grad, lam_qr, E))

    print(f"\nGrad m = {grad}")
    print("lambda Normalgleichung:", lam_normal)
    print("lambda QR:             ", lam_qr)
    print("lambda np.polyfit:     ", lam_polyfit)
    print("E = ||y - A lambda||_2^2:", E)
    print("p_m(x) =", polynom_symbolisch(lam_qr))

# %% 4) Plot der Ausgleichspolynome

xx = np.linspace(xdat.min() - 0.2, xdat.max() + 0.2, 500)

plt.figure()
plt.plot(xdat, ydat, "o", label="Messdaten")

for grad, lam_qr, E in resultate:
    yy = polynom_auswerten(xx, lam_qr)
    plt.plot(xx, yy, label=f"Grad {grad}, E={E:.3e}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Polynom-Ausgleich mit Vandermonde-Matrix und QR")
plt.grid(True)
plt.legend()
plt.show()

# %% 5) Plot Fehlerfunktional gegen Polynomgrad

grade_array = np.array([eintrag[0] for eintrag in resultate])
fehler_array = np.array([eintrag[2] for eintrag in resultate])

plt.figure()
plt.semilogy(grade_array, fehler_array, "o-")
plt.xlabel("Polynomgrad m")
plt.ylabel("Fehlerfunktional E")
plt.title("Fehlerfunktional gegen Polynomgrad")
plt.grid(True)
plt.show()
