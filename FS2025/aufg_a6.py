# ----------------------------------------------------------------------------
# SEP HM2 FS25 / Pythonaufgabe Kapitel 8 Version A / adel
# ----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Angaben

def f(x, y):
    return 2*(1-x)*y
a = 0.
y0 = 1
b = 3.
def yexakt(x):
    return np.exp(2*x - x**2)

# TEILAUFGABE a)

# Berechnung numerische Loesung (gemaess Aufgabe 2 von Serie 12)

def RK4(f, a, y0, b, n):
    x = np.zeros(n+1)
    x[0] = a
    y = np.zeros(n+1)
    y[0] = y0
    h = (b - a)/n
    # Achtung: a und b werden lokal umbezeichnet!
    c = np.array([0., 1/3., 1/3., 2/3.])
    a = np.array([[0.  , 0.  , 0.  ],
                  [1/3., 0.  , 0.  ],
                  [0.  , 1/3., 0.  ],
                  [0   , 1/3., 1/3.]])
    b = np.array( [1/4., 0., 0., 3/4.])
    for i in range(n):
        k1 = f(x[i] + c[0]*h, y[i])
        k2 = f(x[i] + c[1]*h, y[i] + h* a[1,0]*k1)
        k3 = f(x[i] + c[2]*h, y[i] + h*(a[2,0]*k1 + a[2,1]*k2))
        k4 = f(x[i] + c[3]*h, y[i] + h*(a[3,0]*k1 + a[3,1]*k2 + a[3,2]*k3))
        k = b[0]*k1 + b[1]*k2 + b[2]*k3 + b[3]*k4
        x[i+1] = x[i] + h
        y[i+1] = y[i] + h*k
    return x, y

h = 0.1
n = int((b - a)/h)
xnum, ynum = RK4(f, a, y0, b, n)                                   # 4 Punkte

# Plot
plt.plot(xnum, ynum, marker='o', linestyle=' ', color='blue',
         label='numerische Loesung')
xplot = np.arange(a, b, 0.01)
yplot = yexakt(xplot)
plt.plot(xplot, yplot, color='red', label='exakte Loesung')
plt.legend()
plt.grid()
plt.show()                                                         # 1 Punkt

# TEILAUFGABE b)

# Berechnung globale Fehler

error = np.zeros(3)
for i in range(3):
    h = 0.1*10**-i
    n = int((b - a)/h)
    xnum, ynum = RK4(f, a, y0, b, n)
    error[i] = np.abs(ynum[-1] - yexakt(b))
print(error)
# [1.25007841e-05 1.49523304e-08 1.49021698e-11]                    # 3 Punkte

# Bestimmung Konvergenzordnung

# Schlussfolgerung: Das Verfahren hat Konvergenzordnung 3.
# Begruendung: Wird Schrittweite um Faktor 10 verkleinert, so verkleinert
# sich der globale Fehler um rund Faktor 1000 = 10**3.
                                                                    # 2 Punkte
    
    
