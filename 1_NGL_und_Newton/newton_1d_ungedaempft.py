# -*- coding: utf-8 -*-
"""
newton_1d_ungedaempft.py

Aufgabenstellung
================
Gegeben ist die nichtlineare Gleichung

    f(x) = x**3 - 2*x - 5 = 0.

Gesucht ist eine Nullstelle von f mit dem ungedämpften Newton-Verfahren.
Zusätzlich soll der Iterationsverlauf geplottet werden.

Hinweis für die Prüfung:
- f(x) oben austauschen
- Startwert x0, Toleranz und max_iter anpassen
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# %% 1) Gegebene Werte und Funktion eintragen

x = sp.symbols("x")

# Nichtlineare Funktion. Das Beispiel ist bewusst nicht trivial, aber gut lösbar.
f_sp = x**3 - 2*x - 5

df_sp = sp.diff(f_sp, x)                 # Ableitung für Newton automatisch berechnen

# Numerische Funktionen aus den symbolischen Ausdrücken erzeugen
f = sp.lambdify(x, f_sp, modules="numpy")
df = sp.lambdify(x, df_sp, modules="numpy")

x0 = 2.0                                  # Startwert nahe der gesuchten Nullstelle
tol = 1e-10                               # Abbruchschranke
max_iter = 20                             # Maximale Anzahl Iterationen


# %% 2) Newton-Verfahren ohne Dämpfung

xs = [x0]                                 # Alle Iterationswerte für Tabelle und Plot speichern

for k in range(max_iter):
    xk = xs[-1]
    fx = f(xk)
    dfx = df(xk)

    delta = -fx / dfx                     # Newton-Korrektur: df(x_k)*delta = -f(x_k)
    x_next = xk + delta                   # Update: x_{k+1} = x_k + delta
    xs.append(float(x_next))

    print(
        "k={:2d}: x_k={: .12f}, f(x_k)={: .3e}, delta={: .3e}".format(
            k, xk, fx, delta
        )
    )

    if abs(f(x_next)) < tol or abs(delta) < tol:
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


# %% 4) Plot: Funktion und Iterationspunkte

x_plot = np.linspace(1.4, 2.5, 500)
y_plot = f(x_plot)

plt.figure()
plt.plot(x_plot, y_plot, label="f(x)")
plt.axhline(0, linewidth=1, label="y = 0")
plt.plot(xs, f(xs), "o", label="Newton-Iterationen")
plt.plot(xs[-1], f(xs[-1]), "x", markersize=10, label="Nullstelle")

# Iterationsnummern direkt neben die Punkte schreiben
for i, xi in enumerate(xs):
    plt.annotate(str(i), (xi, f(xi)), textcoords="offset points", xytext=(5, 5))

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Ungedämpftes Newton-Verfahren in 1D")
plt.grid(True)
plt.legend()
plt.show()
