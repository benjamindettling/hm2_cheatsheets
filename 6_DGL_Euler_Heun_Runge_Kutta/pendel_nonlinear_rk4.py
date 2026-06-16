# -*- coding: utf-8 -*-
"""
Nichtlineares Pendel mit RK4
============================

Aufgabenstellung
----------------
Ein mathematisches Pendel wird beschrieben durch

    theta'(t) = omega(t)
    omega'(t) = -(g/L)*sin(theta(t)).

Zum Vergleich wird auch das linearisierte Pendel berechnet:

    theta'(t) = omega(t)
    omega'(t) = -(g/L)*theta(t).

Gegeben sind

    g = 9.81 m/s^2,   L = 1.2 m,
    theta(0) = 0.8 rad,   omega(0) = 0.

Gesucht ist die Loesung mit dem klassischen RK4-Verfahren.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 1. Angaben der Aufgabe
# -----------------------------------------------------------------------------

t_sym, theta_sym, omega_sym, g_sym, L_sym = sp.symbols("t theta omega g L")
F_nl_sym = sp.Matrix([omega_sym, -(g_sym / L_sym) * sp.sin(theta_sym)])
F_lin_sym = sp.Matrix([omega_sym, -(g_sym / L_sym) * theta_sym])

print("Nichtlineares Pendel-System:")
sp.pprint(F_nl_sym)
print("\nLinearisiertes Pendel-System:")
sp.pprint(F_lin_sym)

g = 9.81
L = 1.2

a = 0.0
b = 10.0
h = 0.02
n = int((b - a) / h)

u0 = np.array([0.8, 0.0])                         # [theta0, omega0]


def F_nonlinear(t, u):
    theta = u[0]
    omega = u[1]
    return np.array([omega, -(g / L) * np.sin(theta)])


def F_linear(t, u):
    theta = u[0]
    omega = u[1]
    return np.array([omega, -(g / L) * theta])


def energie_pendel(u):
    """Mechanische Energie des nichtlinearen Pendels."""
    theta = u[:, 0]
    omega = u[:, 1]
    return 0.5 * L**2 * omega**2 + g * L * (1 - np.cos(theta))


# -----------------------------------------------------------------------------
# 2. RK4 fuer Systeme
# -----------------------------------------------------------------------------

def rk4_system(F, a, u0, b, n):
    t = np.zeros(n + 1)
    u = np.zeros((n + 1, len(u0)))
    t[0] = a
    u[0, :] = u0
    h = (b - a) / n

    for i in range(n):
        k1 = F(t[i], u[i, :])
        k2 = F(t[i] + h / 2, u[i, :] + h / 2 * k1)
        k3 = F(t[i] + h / 2, u[i, :] + h / 2 * k2)
        k4 = F(t[i] + h, u[i, :] + h * k3)

        u[i + 1, :] = u[i, :] + h / 6 * (k1 + 2*k2 + 2*k3 + k4)
        t[i + 1] = t[i] + h

    return t, u


# -----------------------------------------------------------------------------
# 3. Berechnung
# -----------------------------------------------------------------------------

t_nl, u_nl = rk4_system(F_nonlinear, a, u0, b, n)
t_lin, u_lin = rk4_system(F_linear, a, u0, b, n)
E_nl = energie_pendel(u_nl)


# -----------------------------------------------------------------------------
# 4. Resultatausgabe
# -----------------------------------------------------------------------------

print("\nNichtlineares Pendel mit RK4")
print("============================")
print("g = {:.2f}, L = {:.2f}, h = {:.4f}, n = {}".format(g, L, h, n))
print("Startwert: theta(0) = {:.4f}, omega(0) = {:.4f}".format(u0[0], u0[1]))
print("Nichtlinear: theta(b) = {:.8f}, omega(b) = {:.8f}".format(u_nl[-1, 0], u_nl[-1, 1]))
print("Linear:      theta(b) = {:.8f}, omega(b) = {:.8f}".format(u_lin[-1, 0], u_lin[-1, 1]))
print("Energie Start nichtlinear: {:.8f}".format(E_nl[0]))
print("Energie Ende  nichtlinear: {:.8f}".format(E_nl[-1]))


# -----------------------------------------------------------------------------
# 5. Plots
# -----------------------------------------------------------------------------

plt.figure()
plt.plot(t_nl, u_nl[:, 0], label="nichtlinear theta(t)")
plt.plot(t_lin, u_lin[:, 0], "--", label="linearisiert theta(t)")
plt.xlabel("t")
plt.ylabel("theta")
plt.title("Pendel: nichtlinear vs. linearisiert")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(u_nl[:, 0], u_nl[:, 1], label="nichtlinear")
plt.plot(u_lin[:, 0], u_lin[:, 1], "--", label="linearisiert")
plt.xlabel("theta")
plt.ylabel("omega")
plt.title("Pendel: Phasenportrait")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(t_nl, E_nl)
plt.xlabel("t")
plt.ylabel("Energie")
plt.title("Energie des nichtlinearen Pendels mit RK4")
plt.grid(True)
plt.show()
