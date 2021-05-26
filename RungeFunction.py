# 7.4. Compute both polynomial and cubic spline interpolants to Runge’s function, 
# f(t) = 1/(1 + 25t2), using both n = 11 and 21 equally spaced points on the interval 
# [−1, 1]. Compare your results graphically by plotting both interpolants and
# the original function for each value of n

import numpy as np
import scipy.interpolate as spin
import matplotlib.pyplot as plt

def RungePoints(n: int):
    x = np.linspace(-1, 1, n)
    y = [(1/(1 + 25*(i**2))) for i in x]
    return x, y


if __name__ == '__main__':
    x11, y11 = RungePoints(11)
    x21, y21 = RungePoints(21)
    poly11 = spin.KroghInterpolator(x11, y11)
    poly21 = spin.KroghInterpolator(x21, y21)
    cusp11 = spin.CubicSpline(x11, y11)
    cusp21 = spin.CubicSpline(x21, y21)
    
    x = np.arange(-1, 1, 0.1)
    y = [(1/(1 + 25*(i**2))) for i in x]
    
    yp11 = poly11(x)
    yp21 = poly21(x)
    yc11 = cusp11(x)
    yc21 = cusp21(x)
    
    fig, ax = plt.subplots(2, 1, sharex='col', sharey='row')
    ax[0].plot(x, yp11)
    ax[0].plot(x, yp21)
    ax[0].plot(x, y)
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Polynomial Interpolation")
    ax[0].legend(["Polynomial 11 points", "Polynomial 21 points", "Actual Function"],  bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax[1].plot(x, yc11)
    ax[1].plot(x, yc21)
    ax[1].plot(x, y)
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].set_title("Cubic Spline Interpolation")
    ax[1].legend(["Cubic Spline 11 points", "Cubic Spline 21 points", "Actual Function"],  bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    fig.savefig("Output/7-4.png", bbox_inches="tight")
    plt.show()