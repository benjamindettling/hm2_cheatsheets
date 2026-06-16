# -*- coding: utf-8 -*-
"""
Aufgabe 5 (Version A)

@author: beer
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


""" Aufgabe a) """

# Symbolischer Teil
# =================
A, B, p, q = sp.symbols('A B p q')
lam = sp.Matrix([A, B, p, q])

xdat = np.array([0.,    1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.])
ydat = np.array([0.39,  0.89, 1.45, 1.23, 0.86, 0.83, 0.58, 0.50, 0.44, 0.34])

def T_sp(x, lam):                                     # Modellfunktion symbolisch
    return (lam[0]*x + lam[1])/(x**2 + lam[2]*x + lam[3])

g = sp.Matrix([y - T_sp(x, lam) for x, y in zip(xdat, ydat)])
Dg = g.jacobian(lam)


# Numerischer Teil
# ================

g = sp.lambdify([([A], [B], [p], [q])], g, 'numpy')
Dg = sp.lambdify([([A], [B], [p], [q])], Dg, 'numpy')
                                                                    # 2 Punkte

def T_np(x, lam):                                     # Modellfunktion numerisch
    return (lam[0]*x + lam[1])/(x**2 + lam[2]*x + lam[3])



# Gedämpftes Gauss-Newton Verfahren mit Normalgleichung in Iteration
def gauss_newton_normal(g, Dg, lam0, tol, max_iter, pmax, verbose):

    def E(lam):             # Fehlerfunktional
        return np.linalg.norm(g(lam))**2

    num_iter = 0
    lam = lam0
    increment = tol+1
    
    while increment > tol:

        delta = np.linalg.solve(Dg(lam).T@Dg(lam), -Dg(lam).T@g(lam))

        # Dämpfung                                                             
        damp = 1; p = 0
        while (E(lam) < E(lam + damp*delta)) and (pmax > 0):
            damp /= 2
            p += 1
            if (p > pmax):
                damp = 1; p = 0
                break

        if verbose:
            print("Dämpfungsfaktor: {:f}".format(p))
        
        # Update des Vektors lambda        
        lam = lam + damp*delta
        increment = np.linalg.norm(damp*delta)

        # Überprüfung der maximalen Anzahl Iterationen
        num_iter += 1
        if num_iter > max_iter:
            raise Exception("Divergenz!")

    return(lam, num_iter)



# Gedämpftes Gauss-Newton Verfahren mit numpy.linalg.lstsq in Iteration
def gauss_newton_lstsq(g, Dg, lam0, tol, max_iter, pmax, verbose):

    def E(lam):             # Fehlerfunktional
        return np.linalg.norm(g(lam))**2

    num_iter = 0
    lam = lam0
    increment = tol+1
    
    while increment > tol:

        delta = np.linalg.lstsq(Dg(lam), -g(lam), rcond=None)[0]           

        # Dämpfung                                                             
        damp = 1; p = 0
        while (E(lam) < E(lam + damp*delta)) and (pmax > 0):
            damp /= 2
            p += 1
            if (p > pmax):
                damp = 1; p = 0
                break

        if verbose:
            print("Dämpfungsfaktor: {:f}".format(p))
        
        # Update des Vektors lambda        
        lam = lam + damp*delta
        increment = np.linalg.norm(damp*delta)

        # Überprüfung der maximalen Anzahl Iterationen
        num_iter += 1
        if num_iter > max_iter:
            raise Exception("Divergenz!")

    return(lam, num_iter)



# Gedämpftes Gauss-Newton Verfahren mit QR-Zerlegung in Iteration
def gauss_newton_qr(g, Dg, lam0, tol, max_iter, pmax, verbose):

    def E(lam):             # Fehlerfunktional
        return np.linalg.norm(g(lam))**2

    num_iter = 0
    lam = lam0
    increment = tol+1
    
    while increment > tol:

        # QR-Zerlegung von Dg(lam)
        [Q,R] = np.linalg.qr(Dg(lam))
        delta = np.linalg.solve(R,-Q.T @ g(lam))
        
        # Dämpfung                                                             
        damp = 1; p = 0
        while (E(lam) < E(lam + damp*delta)) and (pmax > 0):
            damp /= 2
            p += 1
            if (p > pmax):
                damp = 1; p = 0
                break

        if verbose:
            print("Dämpfungsfaktor: {:f}".format(p))
        
        # Update des Vektors lambda        
        lam = lam + damp*delta
        increment = np.linalg.norm(damp*delta)

        # Überprüfung der maximalen Anzahl Iterationen
        num_iter += 1
        if num_iter > max_iter:
            raise Exception("Divergenz!")

    return(lam, num_iter)
                                                                    # 2 Punkte


# Aufruf Gauss-Newton
tol = 1e-5
max_iter = 4000
pmax = 1        # Fuer pmax >= 1 sinkt die Anzahl Iterationen von 14 auf 12 ab!

lam0 = np.array([[8., 8., 0., 8.]], dtype=np.float64).T

""" Die Aufrufe der drei Varianten fuer Gauss-Newton gedaempft dienen nur der
    Stabilitaetsueberpruefung! Es sollte ueberall dasselbe herauskommen """

[lamn, n] = gauss_newton_normal(g, Dg, lam0, tol, max_iter, pmax, True)
print('lam_' + str(n) + ' = ' + str(np.reshape(lamn,(-1))))
print('')

[lamsq, n] = gauss_newton_lstsq(g, Dg, lam0, tol, max_iter, pmax, True)
print('lam_' + str(n) + ' = ' + str(np.reshape(lamsq,(-1))))
print('')

[lamqr, n] = gauss_newton_qr(g, Dg, lam0, tol, max_iter, pmax, True)
print('lam_' + str(n) + ' = ' + str(np.reshape(lamqr,(-1))))
print('')

""" Ergebnis: lam_12 = [ 2.10815158  2.22636264 -2.84703435  6.37416154] """
                                                                    # 2 Punkte

""" Aufgabe b) """
""" Falls sich hier studentische Loesungen von der Musterloesung unterscheiden
    sollten, so ist bei der Korrektur eine gewisse Kulanz angezeigt! """
""" Die Dämpfung muss pmax >= 1 gesetzt werden, damit der Algorithmus
    am schnellsten konvergiert (nur noch 12 statt 14 Iterationen). """
                                                                    # 2 Punkte
#Plot
plt.figure()
plt.clf()
xf = np.arange(xdat.min(), xdat.max(), 0.001)
yf = T_np(xf, lamn)
plt.semilogy(xdat, ydat, '*', label='data')
plt.semilogy(xf, yf, label='fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()                                                          # 2 Punkte
