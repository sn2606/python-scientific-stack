import numpy as np
from numpy.linalg import cond
from scipy.linalg import lu_factor, lu_solve

# An n × n Hilbert matrix H has entries hij = 1/(i + j − 1), so it has the form
# 1   1/2 1/3 · · ·
# 1/2 1/3 1/4 · · ·
# 1/3 1/4 1/5 · · ·
# .   .   .   . . .
# .   .   .   . . .
# .   .   .   . . .
# For n = 2, 3, . . ., generate the Hilbert matrix of order n, and also generate the n-vector b = Hx,
# where x is the n-vector with all of its components equal to 1. Use a library routine for Gaussian
# elimination (or Cholesky factorization, since the Hilbert matrix is symmetric and positive definite)
# to solve the resulting linear system Hx = b, obtaining an approximate solution xˆ. Compute the
# ∞-norm of the residual r = b − Hxˆ and of the error ∆x = xˆ − x, where x is the vector of all
# ones. How large can you take n before the error is 100 percent (i.e., there are no significant digits in
# the solution)? Also use a condition estimator to obtain cond(H) for each value of n. 
# Try to characterize the condition number as a function of n.
# As n varies, how does the number of correct digits in the components of the computed solution relate
# to the condition number of the matrix?

# Class to implement a general matrix
class Matrix(object):    
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix    
    
    # multiply matrix and and n-vector b
    def generateB(self, x: np.ndarray) -> np.ndarray:
        '''
        Performs the matrix multiplication Hx to generate b.
        '''
        self.b = np.dot(self.matrix, x)
        return self.b

    # method to solve the linear equation Ax = b
    def gaussianElimination(self) -> np.ndarray:
        '''
        Solve Ax = b using Gaussian elimination via LU decomposition.
        A = LU   decompose H into lower and upper triangular matrices
        LUx = b  substitute into original equation for A
        Let y = Ux and solve:
        Ly = b --> y = (L^-1)b  solve for y using forward substitution
        Ux = y --> x = (U^-1)y  solve for x using backward substitution
        '''
        lu, piv = lu_factor(self.matrix)
        x = lu_solve((lu, piv), self.b)
        
        return x
    
    # method to compute error delta x
    def computeErrorX(self, x: np.ndarray, xcap: np.ndarray) -> np.ndarray:
        '''
        Method to compute the error delta x.
        '''
        deltaX = np.subtract(xcap, x)
        return deltaX
    
    # method to compute residual
    def computeResidual(self, xcap: np.ndarray) -> np.ndarray:
        '''
        Method to compute residual r = b - A*xcap.
        '''
        return np.subtract(self.b, np.dot(self.matrix, xcap))
    
    # method to compute infinity norm of the residual
    def computeInfNorm(self, A: np.ndarray) -> float:
        '''
        Method to compute infinity norm of a given ndarray A.
        '''
        Anew = np.abs(A).sum(axis = 1);
        return np.max(Anew, axis=0)
    
    # method to compute condition of the matrix
    def computeCond(self, A: np.ndarray) -> float:
        '''
        Method to compute condition number of a given ndarray A.
        '''
        return cond(A, np.inf)

# Class to implement a Hilbert matrix
class HilbertMatrix(Matrix):
    def __init__(self, n: int):
        '''
        Generates a Hilbert matrix of order n.
        '''
        hmatrix = []
        for i in range(1, n+1):
            row = [1/(i+j-1) for j in range(1, n+1)]
            hmatrix.append(row)
        self.matrix = np.array(hmatrix)


if __name__ == '__main__':
    # relError = 0
    n = 13
    # while relError < 0.5:
    hm = HilbertMatrix(n)
    # print(hm.matrix)
    x = np.ones((n, 1))
    # print(x)
    b = hm.generateB(x)
    # print(b)
    xcap = hm.gaussianElimination()
    # print(xcap)
    delX = hm.computeErrorX(x, xcap)
    # print(delX)
    res = hm.computeResidual(xcap)
    # print(res)
    infn1 = hm.computeInfNorm(res)
    infn2 = hm.computeInfNorm(xcap)
    # relError = hm.computeInfNorm(delX)/hm.computeInfNorm(x)
    relError = np.linalg.norm(delX, np.inf)/np.linalg.norm(x, np.inf)
    # print(infn1)
    # print(infn2)
    c = cond(hm.matrix, np.inf)
    print(c)
    n += 100
    print(relError)