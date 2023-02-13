import numpy as np

class ComputationError:
    
    def __init__(self):
        print("There was an error in solving the linear system.")


class Vandermode:
    '''
    Given X, a set of n real numbers, creates a Vandermode matrix
    representation of n n-1 degree polynomials. 
    '''

    def __init__(self, X):
        
        #TODO: add assertions

        n = len(X)
        v = []
        p = [1]

        for k in X:
            for i in range(1, n):
                p.append(k**i)

            v.extend([p])
            p = [1]

        self.matrix = np.array(v)
        
    def __str__(self): 
        
        return str(self.matrix)

    def solve(self, f):
        '''
        TODO: add docstring
        '''
        A = self.matrix
        n = len(f)

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
        x = np.zeros(n)
        x[n - 1] = f[n - 1] / A[n - 1, n - 1]

        for i in range(n - 2, -1, -1): 
            y = f[i]
            for j in range(i + 1, n):
                y -= A[i, j]*x[j]

            x[i] = y/A[i, i]

        #check the computation
        if np.isclose([np.dot(A, x)], [f], atol=1e-10).all(): 
            print("Interpolated")
        else:
            raise ComputationError
        
        return (A, x)

if __name__ == "__main__":
    X = [-3, -2, -1, 0, 1, 2, 3]
    f = [2, 3, -1, -2, -4, -1 ,0]
        
    V = Vandermode(X)
    
    print(V)

    c = V.solve(f)

    print(c[0])
    print(c[1])