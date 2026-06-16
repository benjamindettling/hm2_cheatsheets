# -*- coding: utf-8 -*-
"""
Trapezregel fuer nicht-aequidistante Stuetzstellen
==================================================

Aufgabenstellung
----------------
Die Masse eines Planeten soll aus tabellierten Dichtewerten berechnet werden.
Die Dichte rho(r) ist an nicht-aequidistanten Radien gegeben.

Gesucht ist naeherungsweise

    M = integral_0^R rho(r) * 4*pi*r^2 dr.

Da die Stuetzstellen nicht gleichmaessig verteilt sind, wird die summierte
Trapezregel fuer nicht-aequidistante Daten verwendet:

    I ~= Summe_i (y_i + y_{i+1})/2 * (x_{i+1} - x_i).

Wichtig: Die Radien werden von km in m umgerechnet, weil rho in kg/m^3 gegeben ist.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% Angaben: Messdaten / Tabellenwerte

# Radius in km, nicht aequidistant
r_km = np.array([0, 800, 1200, 1400, 2000, 3000, 3500, 4000, 5000, 5700, 6370], dtype=float)

# Dichte in kg/m^3, einfache plausible Dichteverteilung nach aussen abnehmend
rho = np.array([13000, 12900, 12700, 12000, 11650, 10000, 7500, 5500, 4500, 4000, 3300], dtype=float)

# Literaturwert als Orientierung / Plausibilitaetsvergleich
M_literatur = 5.97e24


# %% Umrechnung und Integrand aufstellen

r_m = r_km * 1000.0                         # km -> m
integrand = rho * 4 * np.pi * r_m**2        # y_i = rho(r_i) * 4*pi*r_i^2

print("Trapezregel fuer nicht-aequidistante Stuetzstellen")
print("==================================================")
print("Gegeben: r in km, rho in kg/m^3")
print("Gesucht: M = integral rho(r) * 4*pi*r^2 dr")
print()

print("Tabelle")
print("------")
print(" i      r [km]      rho [kg/m^3]        integrand")
for i in range(len(r_km)):
    print(f"{i:2d}  {r_km[i]:10.1f}  {rho[i]:14.1f}  {integrand[i]:16.6e}")
print()


# %% Summierte Trapezregel fuer nicht-aequidistante Daten

# Einzelbeitraege der Trapeze:
# I_i = (y_i + y_{i+1})/2 * (x_{i+1} - x_i)
beitraege = (integrand[:-1] + integrand[1:]) / 2 * (r_m[1:] - r_m[:-1])
M_approx = np.sum(beitraege)
relativer_fehler = abs(M_approx - M_literatur) / M_literatur

print("Trapezbeitraege")
print("---------------")
for i in range(len(beitraege)):
    print(f"Intervall [{r_km[i]:.0f}, {r_km[i+1]:.0f}] km: {beitraege[i]:.6e} kg")
print()

print("Resultat")
print("--------")
print(f"M_approx       = {M_approx:.6e} kg")
print(f"M_literatur    = {M_literatur:.6e} kg")
print(f"rel. Abweichung = {relativer_fehler:.4%}")


# %% Plot 1: Dichteverteilung

plt.figure()
plt.plot(r_km, rho, "o-", label="Dichtewerte")
plt.xlabel("r [km]")
plt.ylabel("rho(r) [kg/m^3]")
plt.title("Nicht-aequidistante Dichtewerte")
plt.grid(True)
plt.legend()
plt.show()


# %% Plot 2: Integrand und Trapezflaechen

plt.figure()
plt.plot(r_km, integrand, "o-", label="rho(r) * 4*pi*r^2")
plt.fill_between(r_km, integrand, alpha=0.25, label="Trapeznaeherung")
plt.xlabel("r [km]")
plt.ylabel("Integrand [kg/m]")
plt.title("Trapezregel fuer nicht-aequidistante Stuetzstellen")
plt.grid(True)
plt.legend()
plt.show()


# %% Kontrollrechnung mit numpy.trapz / np.trapezoid

# np.trapezoid macht hier genau dieselbe nicht-aequidistante Trapezregel.
M_np = np.trapezoid(integrand, r_m)
print()
print("Kontrolle mit np.trapezoid:")
print(f"M_np = {M_np:.6e} kg")
