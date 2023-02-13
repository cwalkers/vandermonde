import numpy as np
import matplotlib.pyplot as plt

class ComputationError:
    
    def __init__(self):
        print("There was an error in solving the linear system.")


class Vandermonde:

    def __init__(self, x: list):
        '''
        Given a list of n real numbers, creates a Vandermonde matrix (n x n)
        representation of n n-1th degree polynomials.
        '''

        assert sorted(x) == sorted(list(set(x))), "x's must be unique"

        v = []
        p = [1]

        for k in x:
            for i in range(1, len(x)):
                p.append(k**i)

            v.extend([p])
            p = [1]

        self.matrix = np.array(v)
        
    def __str__(self): 
        
        return str(self.matrix)

    def solve(self, f: list) -> tuple:
        '''       
        Given a Vandermode Object, and a list of values for f(x),
        solves the linear system to produce the coefficients of the
        interpolant polynomial.
        
        Paramaters:
            self: Vandermode Object: the Vandermode matrix
            f: lst: the values of the polynomial we seek to interpolate
        
        Returns: tuple (A, x):
            A: the Vandermode Matrix after Gaussian Elimination
            x: the coefficients of the interpolant
        '''
        A = self.matrix
        n = len(f)
        
        assert len(A) == n, 'Dimensions must match'

        for i in range(n - 1):
            
            #pivoting
            max = abs(A[i, i])
            maxidx = i

            for z in range(i + 1, n):
                if abs(A[z, i]) > max:

                    max = abs(A[z, i])
                    maxidx = z
            
            if maxidx != i:

                A[maxidx], A[i] = A[i].copy(), A[maxidx].copy()
                f[maxidx], f[i] = f[i].copy(), f[maxidx].copy()

            #elimination
            for j in range(i + 1, n): 
                ratio = A[j, i] / A[i, i]

                for k in range(i, n):
                    A[j, k] -= ratio*A[i, k]

                f[j] -= ratio*f[i]

        #back subsitution
        c = np.zeros(n)
        c[n - 1] = f[n - 1] / A[n - 1, n - 1]

        for i in range(n - 2, -1, -1): 
            y = f[i]
            for j in range(i + 1, n):
                y -= A[i, j]*c[j]

            c[i] = y/A[i, i]

        #check the computation
        if np.isclose([np.dot(A, c)], [f], atol=1e-10).all(): 
            print("Interpolated")
        else:
            raise ComputationError

        return (A, c)


class Polynomial: 
    
    def __init__(self, coefficients): 
        '''
        '''
        self.coefficients = coefficients
        self.f = lambda x: sum(c*x**i for i, c in enumerate(coefficients))
        
    def __str__(self):

        st = ''

        for i, c in enumerate(self.coefficients):

            c = round(c, 4)
            
            if i == 0:
                st += str(c)

            elif i == 1:
                if c < 0: 
                    st += ' - ' + str(c)[1:] + 'x'
                else:
                    st += ' + ' + str(c) + 'x'

            else:
                if c < 0:
                    st += ' - ' + f'{c}x^{i}'[1:]
                else:
                    st += ' + ' + f'{c}x^{i}'

        return st

if __name__ == "__main__":
    X = [-3, -2, -1, 0, 1, 2, 3]
    f = [2, 3, -1, -2, -4, -1 ,0]

    V = Vandermonde(X)
    print(V)

    A, c = V.solve(f)

    print(A)
    print(c)

    poly = Polynomial(c)
    print(f"Interpolant: {poly}")
    
    xs = np.arange(-3, 3, 0.01)
    
    plt.plot(xs, poly.f(xs))
    plt.scatter([-3, -2, -1, 0, 1, 2, 3], [2, 3, -1, -2, -4, -1 ,0])
    plt.ylabel("f(x)")
    plt.xlabel("x")
    plt.show()