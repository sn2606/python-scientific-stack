import scipy.integrate as spi
import scipy.misc as spm

# 7.1. (a) Write a routine that uses Horner’s rule to evaluate a polynomial p(t) given its degree n, an
# array x containing its coefficients, and the value t of the independent variable at which it is to be
# evaluated.
# (b) Add options to your routine to evaluate the derivative p0(t) or the integral (a,b,p(t)dt), given a
# and b.

def horner(polyCoeff, x, derivative=None, integral=None):
    '''
    polyCoeff: List of polynomial coefficients in order of decreasing power of x
    x: the value at which the polynomial is to be computed
    derivative: Takes input as integer n - to calculate nth derivative of given polynomial
    integral: Takes input as tuple (a,b) - to calculate integral (a,b,p(t)dt)
    '''
    n = len(polyCoeff)
    
    if n < 1:
        raise ValueError("polyCoeff: List of polynomial coefficients in order of decreasing power of x")
    
    result = polyCoeff[0]
    for i in range(1, n):
        result = result*x + polyCoeff[i]
    
    if derivative:
        dx = spm.derivative(lambda k: horner(polyCoeff, k), x, n=derivative)
        print("Derivative at ", x, " is ", dx)
        
    if integral:
        (a,b) = integral
        integ = spi.quad(lambda k: horner(polyCoeff, k), a, b)
        print("Integral in range (", a, ", ", b, ") is ", integ)
    return result


if __name__ == '__main__':
    # 3x^3 + 2x^2 + x + 1
    x = 1
    coeff = [3, 2, 1, 1]    
    res = horner(coeff, x, derivative=2, integral=(0,2))
    print("The value of polynomial at x = " + str(x) + " is " + str(res))