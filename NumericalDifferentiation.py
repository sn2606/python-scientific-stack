# 8.18. In this exercise we will experiment with numerical differentiation using data from Computer
# Problem 3.1:
# t | 0.0 1.0 2.0 3.0 4.0 5.0
# y | 1.0 2.7 5.8 6.6 7.5 9.9
# For each of the following methods for estimating the derivative, compute the derivative of the
# original data and also experiment with randomly perturbing the y values to determine the sensitivity 
# of the resulting derivative estimates. For each method, comment on both the reasonableness of
# the derivative estimates and their sensitivity to perturbations. Note that the data are monotonically 
# increasing, so one might expect the derivative always to be positive.
# (a) For n = 0, 1, . . . , 5, fit a polynomial of degree n by least squares to the data, 
# then differentiate the resulting polynomial and evaluate the derivative at each of the given t values.

import numpy as np
import scipy.linalg as sp

if __name__ == '__main__':
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 2.7, 5.8, 6.6, 7.5, 9.9])
    
    for i in range(5):    
        diff = []
        for j in range(5):
            poly = np.poly1d(np.polyfit(t, y, i))
            h = 0.1
            diff.append((poly(t+h)-poly(t))/h)
            y = sorted(np.random.randint(0,10,(6,)) + np.random.rand(6)) #random y to check sensitivity?
            # print(y)
        print(diff)