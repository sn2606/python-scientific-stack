# 8.4. Use numerical integration to verify or refute each of the following conjectures.
# (c)integral(0, 1, ((e^(−9x^2) + e^(−1024(x − 1/4)^2))/√π)dx) = 0.2
# (d)integral(0, 10, (50/(π(2500x^2 + 1)))dx) = 0.5

import numpy as np
import scipy.integrate as spi
import scipy.linalg as spla
from math import e, pi, sqrt

def fc(x):
    '''
    Calculates value integrating function for part c at x
    '''
    res = ((e**(-9*x**2)) + (e**(-1024*(x-0.25)**2)))/sqrt(pi)
    return res

def fd(x):
    '''
    Calculates value integrating function for part c at x
    '''
    res = 50/(pi*(2500*x**2 + 1))
    return res

def findWeights(a, b, x, n):
    '''
    Calculates weights for numerical quadrature
    '''
    # x = np.linspace(a, b, x, n) # nodes for f
    b = [(b**i - a**i)/i for i in range(1, n+1)]
    b = np.array(b).reshape((n, 1))
    A = np.zeros((n,n))
    for i in range(n):
        A[i, :] = x**i
    
    lu, piv = spla.lu_factor(A)
    w = spla.lu_solve((lu, piv), b)
    return w.reshape((1,n))

def integralQuadrature(f, a, b):
    '''
    Function to approximate integral by Newton-Cotes numerical quadrature method
    '''
    n = 2 # n larger => h smaller i.e. I(n)->Q(n) as n->inf
    x = np.linspace(a, b, n)
    w = findWeights(a, b, x, n)
    fi = np.array([f(i) for i in x]).reshape((n,1))
    return np.dot(w,fi)    
    

if __name__ == '__main__':
    ic = round(spi.quad(fc, 0, 1)[0], 2)
    print(ic == 0.2)
    id = round(spi.quad(fd, 0, 10)[0], 2)
    print(id == 0.5)
    
    # print(integralQuadrature(fd, 0, 10)) : not working