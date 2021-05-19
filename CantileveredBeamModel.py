import numpy as np
from numpy.linalg import cond, norm
from scipy.linalg import lu_factor, lu_solve, solve_banded, lu
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from math import log10
import sys
import time

# Consider a horizontal cantilevered beam
# that is clamped at one end but free along the remainder of its length. A discrete model of the
# forces on the beam yields a system of linear equations Ax = b, where the n × n matrix A has the
# banded form
#  9 −4  1  0 · · · · ·  ·  0
# −4  6 −4  1 . . . . .  .  .
#  1 −4  6 −4 1 . . . .  .  .
#  0  .  .  . . . . . .  .  0
#  .  .  .  . . 1 −4  6 −4  1
#  .  .  .  . . .  1 −4  5 −2
#  0  ·  ·  · · ·  0  1 −2  1
# the n-vector b is the known load on the bar (including its own weight), 
# and the n-vector x represents the resulting deflection of the bar that is
# to be determined. We will take the bar to be uniformly loaded, with bi = 1/n4
# for each component of the load vector.
# (a) Letting n = 100, solve this linear system using both a standard library routine for dense linear
# systems and a library routine designed for banded (or more general sparse) systems. How do the two
# routines compare in the time required to compute the solution? How well do the answers obtained
# agree with each other?

class BeamMatrix:
    def __init__(self, n):
        if n <= 4 or type(n) != int:
            sys.exit()
        
        load = np.ones((n, 1))
        load *= (1 / (n**4))
        self.load = load
        # print(load.shape)
        
        arr = np.zeros((n, n), dtype='int64')
        arr[0][0] = 9
        arr[1][1] = 5
        arr[n-1][n-1] = 2
        arr[n-2][n-2] = 5
        arr[n-1][n-2] = arr[n-2][n-1] = -2
        arr[n-2][n-3] = arr[0][1] = arr[1][0] = arr[1][2] = -4        
        arr[n-1][n-3] = arr[n-2][n-4] = arr[0][2] = arr[1][3] = 1
        
        for i in range(0, n-4):
            arr[i+2][0+i] = 1
            arr[i+2][1+i] = -4
            arr[i+2][2+i] = 6
            arr[i+2][3+i] = -4
            arr[i+2][4+i] = 1
            
        self.matrix = arr
        self.dimension = n
        
    # solve Ax = b by algorithm to solve banded systems
    def solveBanded(self) -> np.ndarray:
        '''
        Method to solve the beam matrix by solve_banded.
        The matrix constructed by this class is a banded matrix
        '''
        x = solve_banded((49, 50), self.matrix, self.load)
        return x
    
    # Solve the sparse linear system Ax=b, where b may be a vector or a matrix.
    def solveSparse(self, b) -> np.ndarray:
        '''
        Method to solve the beam matrix by spsolve.
        The matrix constructed by this class can be considered as a sparse matrix if n >> 5
        '''
        x = spsolve(csc_matrix(self.matrix), b)
        return x
    
    # method to solve the linear equation Ax = b where A is considered as a dense matrix
    def solveDense(self, b) -> np.ndarray:
        '''
        Solve Ax = b using Gaussian elimination via LU decomposition.
        A = LU   decompose H into lower and upper triangular matrices
        LUx = b  substitute into original equation for A
        Let y = Ux and solve:
        Ly = b --> y = (L^-1)b  solve for y using forward substitution
        Ux = y --> x = (U^-1)y  solve for x using backward substitution
        '''
        lu, piv = lu_factor(self.matrix)
        x = lu_solve((lu, piv), b)
        
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
        residual = self.load - np.dot(self.matrix, xcap)
        # print(residual.shape)
        return residual
    
    # method to compute infinity norm of the residual
    def computeInfNorm(self, A: np.ndarray) -> float:
        '''
        Method to compute infinity norm of a given ndarray A.
        '''
        # Anew = np.abs(A).sum(axis = 1);
        # return np.max(Anew, axis=0)
        return norm(A, np.inf)
    
    # method to compute condition of the matrix
    def computeCond(self, A: np.ndarray) -> float:
        '''
        Method to compute condition number of a given ndarray A.
        '''
        return cond(A, np.inf)
    
    # method to compute error bound for a numerically calculated xcap
    def computeErrorBound(self, xcap: np.ndarray) -> float:
        '''
        Method to calculate error bound on xcap
        Error bound is given by condition number of matrix times the relative residual
        infNorm(deltaX)/infNorm(xcap) is considered as relative error
        '''
        nx = self.computeInfNorm(xcap)
        nA = self.computeInfNorm(self.matrix)
        nr = self.computeInfNorm(self.computeResidual(xcap))
        cA = self.computeCond(self.matrix)
        errorBound = cA * (nr / (nA * nx))
        return errorBound 
    
# repeated block of code in main
def executeForN(n: int):
    bm = BeamMatrix(n)
    A = bm.matrix
    b = bm.load
    
    # start_time = time.time()
    # xcap_banded = bm.solveBanded()
    # end_time = time.time()
    # time_banded = end_time - start_time
    # error_bound_banded = bm.computeErrorBound(xcap_banded)
    
    start_time = time.time()
    xcap_sparse = bm.solveSparse(b)
    end_time = time.time()
    time_sparse = end_time - start_time
    xcap_sparse = np.reshape(xcap_sparse, (n, 1))
    error_bound_sparse = bm.computeErrorBound(xcap_sparse)
    
    start_time = time.time()
    xcap_dense = bm.solveDense(b)
    end_time = time.time()
    time_dense = end_time - start_time
    error_bound_dense = bm.computeErrorBound(xcap_dense)
    
    print("n = ", n)
    cA = bm.computeCond(A)
    print("cond(A) = ", cA)
    print("log10(cond(A))", log10(cA))
    print("A is ill-conditioned")
    
    print("Type      Time                       ErrorBound")
    print("__________________________________________________________")
    # print("Banded   " + time_banded + "     " + error_bound_banded)
    print("Sparse   " , time_sparse , "     " , error_bound_sparse)
    print("Dense    " , time_dense , "     " , error_bound_dense, "\n")
    
    return bm, xcap_sparse, xcap_dense
            

if __name__ == '__main__':
    
    bm, xs, xd = executeForN(100)
    
    n = 6
    bm = BeamMatrix(n)
    A = bm.matrix
    L, U = lu(A, permute_l=True)
    # print(res)
    # print(A2)
    # print(L)
    # print(U)
    
    # Letting n = 1000, solve the linear system using this factorization (two triangular solves will be required). 
    # Also solve the system in its original form using a banded (or general sparse) system solver
    # as in part a. How well do the answers obtained agree with each other? Which approach seems
    # more accurate? What is the condition number of A, and what accuracy does it suggest that you
    # should expect? Try iterative refinement to see if the accuracy or residual improves for the less accurate method.
    
    bm, xs, xd = executeForN(1000)
    
    # sparse is less accurate
    # lets consider 5 iterations of iterative refinement
    
    m = 5
    for i in range(m):
        r = bm.computeResidual(xd)
        c = bm.solveDense(r)
        # c = np.reshape(c, (1000, 1))
        xd += c
    
    # after m iterations, the error bound satisfies relerror <= eb^m
    print(bm.computeErrorBound(xd)**m)