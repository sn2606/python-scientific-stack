# 7.2. (a) Write a routine for computing the Newton polynomial interpolant for a given set of data
# points, and a second routine for evaluating the Newton interpolant at a given argument value using 
# Horner’s rule.
# (b) Write a routine for computing the new Newton polynomial interpolant when a new data point is
# added.
# (c) If your programming language supports recursion, write a recursive routine that implements
# part a by calling your routine for part b recursively. Compare its performance with that of your
# original implementation.

import numpy as np
import scipy.linalg as spla

def newtonInterpolation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Function to find newton interpolating polynomial from points X, Y
    X -> X coordinates
    Y -> Y coordinates
    '''
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("X, Y should be of equal dimension.")
    
    n = len(x)
    A = np.zeros((10,10))
    A[:,0] = 1
    
    for i in range(n):
        for j in range(1, i+1):
            pi = 1
            for k in range(0,j-1):
                pi *= (x[i] - x[k])
            A[i,j] = pi
            
    b = y.reshape((n,1))
    
    lu, piv = spla.lu_factor(A)
    x = spla.lu_solve((lu, piv), b)
    
    return x

def evaluateNewton(coeff: np.ndarray, x: np.ndarray, t: float) -> float:
    '''
    Evaluate newton interpolant at point t
    '''
    n = len(coeff)
    
    if n < 1:
        raise ValueError("coeff: List of polynomial coefficients in order of decreasing power of x")
    
    result = coeff[n-1]
    for i in range(n-1, 0, -1):
        result = result*(t-x[i]) + coeff[i-1]
        
    return result

def dividedDiff(x, y):
    '''
    function to calculate the divided
    differences table
    '''
    n = len(y)
    coef = np.zeros((n, n))
    # the first column is y
    coef[:,0] = y
    
    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j]-x[i])
            
    return coef[0, :]

def newtonPolyEval(coef, x_data, x):
    '''
    evaluate the newton polynomial 
    at x
    '''
    n = len(x_data) - 1 
    p = coef[n]
    for k in range(1,n+1):
        p = coef[n-k] + (x -x_data[n-k])*p
    return p

if __name__ == '__main__':
    x = np.array([-9, -8, -7, 8])
    y = np.array([-2, 6, 7, 3])
    # get the divided difference coef
    ai = dividedDiff(x, y)
    print(ai)

    # evaluate on new data points
    x_new = np.arange(-5, 5, .1)
    y_new = newtonPolyEval(ai, x, x_new)