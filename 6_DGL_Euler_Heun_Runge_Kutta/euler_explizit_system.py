# -*- coding: utf-8 -*-
"""
Explizites Eulerverfahren fuer ein DGL-System
=============================================

Aufgabenstellung
----------------
Eine ungedaempfte Federschwingung wird beschrieben durch

    m*x''(t) + k*x(t) = 0.

Als System 1. Ordnung:

    x'(t) = v(t)
    v'(t) = -(k/m)*x(t)

Gegeben sind

    m = 1.5 kg,     k = 6.0 N/m,
    x(0) = 1.0 m,   v(0) = 0.0 m/s.

Gesucht ist die numerische Loesung mit dem expliziten Eulerverfahren.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 1. Angaben der Aufgabe
# -----------------------------------------------------------------------------

t_sym, x_sym, v_sym, m_sym, k_sym = sp.symbols("t x v m k")
F_sym = sp.Matrix([v_sym, -(k_sym / m_sym) * x_sym])
print("Symbolisches DGL-System F(t, [x,v]) =")
sp.pprint(F_sym)

m = 1.5                                           # Masse
k = 6.0                                           # Federkonstante

a = 0.0                                           # Startzeit
b = 8.0                                           # Endzeit
h = 0.02                                          # Schrittweite
n = int((b - a) / h)

u0 = np.array([1.0, 0.0])                          # Anfangswert [x0, v0]


def F(t, u):
    """Rechte Seite des DGL-Systems. u[0]=x, u[1]=v."""
    x = u[0]
    v = u[1]
    return np.array([v, -(k / m) * x])


def energie(u):
    """Mechanische Energie E = 1/2*m*v^2 + 1/2*k*x^2."""
    x = u[:, 0]
    v = u[:, 1]
    return 0.5 * m * v**2 + 0.5 * k * x**2


# -----------------------------------------------------------------------------
# 2. Explizites Eulerverfahren fuer Systeme
# -----------------------------------------------------------------------------

def euler_system(F, a, u0, b, n):
    """Euler explizit fuer u' = F(t,u)."""
    t = np.zeros(n + 1)
    u = np.zeros((n + 1, len(u0)))

    t[0] = a
    u[0, :] = u0
    h = (b - a) / n

    for i in range(n):
        u[i + 1, :] = u[i, :] + h * F(t[i], u[i, :])
        t[i + 1] = t[i] + h

    return t, u


t, u = euler_system(F, a, u0, b, n)
E = energie(u)


# -----------------------------------------------------------------------------
# 3. Resultatausgabe
# -----------------------------------------------------------------------------

print("\nExplizites Eulerverfahren fuer die Federschwingung")
print("==================================================")
print("m = {:.2f}, k = {:.2f}, h = {:.4f}, n = {}".format(m, k, h, n))
print("Startwert:  x(0) = {:.4f}, v(0) = {:.4f}".format(u0[0], u0[1]))
print("Endwert:    x({:.1f}) = {:.8f}, v({:.1f}) = {:.8f}".format(
    b, u[-1, 0], b, u[-1, 1]
))
print("Energie Start: {:.8f}".format(E[0]))
print("Energie Ende:  {:.8f}".format(E[-1]))
print("Bemerkung: Explizites Euler erhaelt die Energie bei Schwingungen nicht gut.")


# -----------------------------------------------------------------------------
# 4. Plots
# -----------------------------------------------------------------------------

plt.figure()
plt.plot(t, u[:, 0], label="x(t) Auslenkung")
plt.plot(t, u[:, 1], label="v(t) Geschwindigkeit")
plt.xlabel("t")
plt.ylabel("Wert")
plt.title("Federschwingung mit explizitem Eulerverfahren")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(u[:, 0], u[:, 1])
plt.xlabel("x")
plt.ylabel("v")
plt.title("Phasenportrait: explizites Eulerverfahren")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t, E)
plt.xlabel("t")
plt.ylabel("Energie")
plt.title("Energieverlauf beim expliziten Eulerverfahren")
plt.grid(True)
plt.show()
