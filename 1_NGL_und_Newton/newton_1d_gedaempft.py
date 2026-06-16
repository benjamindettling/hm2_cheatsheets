# -*- coding: utf-8 -*-
"""
newton_1d_gedaempft.py

Aufgabenstellung
================
Gegeben ist die nichtlineare Gleichung

    f(x) = x**3 - 2*x + 2 = 0.

Gesucht ist eine Nullstelle mit dem gedämpften Newton-Verfahren.
Die Dämpfung wählt pro Schritt den kleinsten Wert p aus {0, ..., pmax},
sodass

    |f(x_k + delta / 2**p)| < |f(x_k)|.

Dieses Beispiel zeigt gut, warum Dämpfung nützlich ist: Vom Startwert x0 = 0
würde das ungedämpfte Newton-Verfahren zwischen zwei Punkten hin und her springen.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% 1) Gegebene Werte und Funktion eintragen

x = sp.symbols("x")

f_sp = x**3 - 2*x + 2
df_sp = sp.diff(f_sp, x)

f = sp.lambdify(x, f_sp, modules="numpy")
df = sp.lambdify(x, df_sp, modules="numpy")

x0 = 0.0
tol = 1e-10
max_iter = 30
pmax = 12


# %% 2) Gedämpftes Newton-Verfahren

xs = [x0]
ps = []                                      # Speichert den gewählten Dämpfungsexponenten p

for k in range(max_iter):
    xk = xs[-1]
    fx = f(xk)
    delta = -fx / df(xk)                    # Ungedämpfte Newton-Korrektur

    # Dämpfung: Schritt solange halbieren, bis der Funktionsbetrag kleiner wird
    p = 0
    while p <= pmax and abs(f(xk + delta / 2**p)) >= abs(fx):
        p += 1

    if p > pmax:
        p = 0                               # Falls keine Dämpfung akzeptiert wird: Newton-Schritt verwenden

    damp = 1 / 2**p
    x_next = xk + damp * delta

    xs.append(float(x_next))
    ps.append(p)

    print(
        "k={:2d}: x_k={: .12f}, f(x_k)={: .3e}, "
        "delta={: .3e}, p={}, Schrittfaktor={:.5f}".format(
            k, xk, fx, delta, p, damp
        )
    )

    if abs(f(x_next)) < tol or abs(damp * delta) < tol:
        break

xs = np.array(xs)


# %% 3) Resultat ausgeben

print("\nSymbolische Funktion:")
print("f(x)  =", f_sp)
print("f'(x) =", df_sp)

print("\nResultat:")
print("Nullstelle x ≈ {:.12f}".format(xs[-1]))
print("f(x)        ≈ {:.3e}".format(f(xs[-1])))
print("Iterationen  =", len(xs) - 1)
print("Gewählte p-Werte:", ps)


# %% 4) Plot: Funktion und gedämpfte Iterationspunkte

x_plot = np.linspace(-2.3, 1.3, 700)
y_plot = f(x_plot)

plt.figure()
plt.plot(x_plot, y_plot, label="f(x)")
plt.axhline(0, linewidth=1, label="y = 0")
plt.plot(xs, f(xs), "o", label="gedämpfte Newton-Iterationen")
plt.plot(xs[-1], f(xs[-1]), "x", markersize=10, label="Nullstelle")

for i, xi in enumerate(xs):
    plt.annotate(str(i), (xi, f(xi)), textcoords="offset points", xytext=(5, 5))

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gedämpftes Newton-Verfahren in 1D")
plt.grid(True)
plt.legend()
plt.show()


# %% 5) Zusatzplot: Residuum pro Iteration

residuen = np.abs(f(xs))

plt.figure()
plt.semilogy(np.arange(len(residuen)), residuen, "o-")
plt.xlabel("Iteration k")
plt.ylabel("|f(x_k)|")
plt.title("Abnahme des Residuums durch Dämpfung")
plt.grid(True)
plt.show()
