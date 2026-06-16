# -*- coding: utf-8 -*-
"""
Federschwingung: Vergleich Euler, Heun und RK4
==============================================

Aufgabenstellung
----------------
Eine schwach gedaempfte Feder-Masse-Schwingung wird modelliert durch

    m*x''(t) + d*x'(t) + k*x(t) = 0.

Als System 1. Ordnung:

    x'(t) = v(t)
    v'(t) = -(d/m)*v(t) - (k/m)*x(t)

Gegeben sind

    m = 1.0 kg,     d = 0.15 Ns/m,     k = 4.0 N/m,
    x(0) = 1.0 m,   v(0) = 0.0 m/s.

Gesucht ist ein Vergleich von Euler, Heun und RK4.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 1. Angaben der Aufgabe
# -----------------------------------------------------------------------------

t_sym, x_sym, v_sym, m_sym, d_sym, k_sym = sp.symbols("t x v m d k")
F_sym = sp.Matrix([v_sym, -(d_sym / m_sym) * v_sym - (k_sym / m_sym) * x_sym])
print("Symbolisches System:")
sp.pprint(F_sym)

m = 1.0
d = 0.15
k = 4.0

a = 0.0
b = 12.0
h = 0.04
n = int((b - a) / h)

u0 = np.array([1.0, 0.0])


def F(t, u):
    x = u[0]
    v = u[1]
    return np.array([v, -(d / m) * v - (k / m) * x])


def energie(u):
    x = u[:, 0]
    v = u[:, 1]
    return 0.5 * m * v**2 + 0.5 * k * x**2


# -----------------------------------------------------------------------------
# 2. Verfahren fuer Systeme
# -----------------------------------------------------------------------------

def euler_system(F, a, u0, b, n):
    t = np.zeros(n + 1)
    u = np.zeros((n + 1, len(u0)))
    t[0] = a
    u[0, :] = u0
    h = (b - a) / n
    for i in range(n):
        u[i + 1, :] = u[i, :] + h * F(t[i], u[i, :])
        t[i + 1] = t[i] + h
    return t, u


def heun_system(F, a, u0, b, n):
    t = np.zeros(n + 1)
    u = np.zeros((n + 1, len(u0)))
    t[0] = a
    u[0, :] = u0
    h = (b - a) / n
    for i in range(n):
        k1 = F(t[i], u[i, :])
        u_pred = u[i, :] + h * k1
        k2 = F(t[i] + h, u_pred)
        u[i + 1, :] = u[i, :] + h / 2 * (k1 + k2)
        t[i + 1] = t[i] + h
    return t, u


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

t_eu, u_eu = euler_system(F, a, u0, b, n)
t_he, u_he = heun_system(F, a, u0, b, n)
t_rk, u_rk = rk4_system(F, a, u0, b, n)

E_eu = energie(u_eu)
E_he = energie(u_he)
E_rk = energie(u_rk)


# -----------------------------------------------------------------------------
# 4. Resultatausgabe
# -----------------------------------------------------------------------------

print("\nFederschwingung: Verfahrenvergleich")
print("===================================")
print("m = {:.2f}, d = {:.2f}, k = {:.2f}, h = {:.4f}, n = {}".format(m, d, k, h, n))
print("Startwert: x(0) = {:.4f}, v(0) = {:.4f}".format(u0[0], u0[1]))
print("")
print("Verfahren       x(b)          v(b)          Energie(b)")
print("Euler        {:12.8f} {:12.8f} {:12.8f}".format(u_eu[-1, 0], u_eu[-1, 1], E_eu[-1]))
print("Heun         {:12.8f} {:12.8f} {:12.8f}".format(u_he[-1, 0], u_he[-1, 1], E_he[-1]))
print("RK4          {:12.8f} {:12.8f} {:12.8f}".format(u_rk[-1, 0], u_rk[-1, 1], E_rk[-1]))


# -----------------------------------------------------------------------------
# 5. Plots
# -----------------------------------------------------------------------------

plt.figure()
plt.plot(t_eu, u_eu[:, 0], label="Euler")
plt.plot(t_he, u_he[:, 0], label="Heun")
plt.plot(t_rk, u_rk[:, 0], label="RK4")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Federschwingung: Auslenkung")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(u_eu[:, 0], u_eu[:, 1], label="Euler")
plt.plot(u_he[:, 0], u_he[:, 1], label="Heun")
plt.plot(u_rk[:, 0], u_rk[:, 1], label="RK4")
plt.xlabel("x")
plt.ylabel("v")
plt.title("Federschwingung: Phasenportrait")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(t_eu, E_eu, label="Euler")
plt.plot(t_he, E_he, label="Heun")
plt.plot(t_rk, E_rk, label="RK4")
plt.xlabel("t")
plt.ylabel("Energie")
plt.title("Federschwingung: Energieverlauf")
plt.grid(True)
plt.legend()
plt.show()
