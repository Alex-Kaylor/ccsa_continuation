from ccsa import CCSA
import numpy as np

#Test a function with a discontinuous derivative at the origin
def test_0():
    def f(y,x,grad):
        print(x)
        y[0] = np.sqrt(x[1]) # objective function
        y[1] = (2*x[0] + 0)**3 - x[1] # constraint 1
        y[2] = (-1*x[0] + -1)**3 - x[1] # constraint 2
        if grad is not None:
            grad[0,0] = 0.0
            grad[1,0] = 0.5 / np.sqrt(x[1])
            grad[0,1] = 3 * 2 * (2*x[0] + 0)**2
            grad[1,1] = -1.0
            grad[0,2] = 3 * -1 * (-1*x[0] + -1)**2
            grad[1,2] = -1.0
    
    test = CCSA(2,2,f,
        lb=[-np.inf,0],ub=[np.inf,np.inf],
        verbose=True,max_eval=50)
    x0 = np.array([-1.234, 15.678])
    x = test(x0)
    print(x)

#Test a function with a minimum at the origin
def test_1():
    def f(y,x,grad):
        print(x)
        y[0] = (x[1])**3 # objective function
        y[1] = (2*x[0] + 0)**3 - x[1] # constraint 1
        y[2] = (-1*x[0] + -1)**3 - x[1] # constraint 2
        if grad is not None:
            grad[0,0] = 0.0
            grad[1,0] = 3*(x[1])**2
            grad[0,1] = 3 * 2 * (2*x[0] + 0)**2
            grad[1,1] = -1.0
            grad[0,2] = 3 * -1 * (-1*x[0] + -1)**2
            grad[1,2] = -1.0
    
    test = CCSA(2,2,f,
        lb=[-np.inf,0],ub=[np.inf,np.inf],
        verbose=True,max_eval=50)
    x0 = np.array([1.234, 5.678])
    x = test(x0)
    print(x)
    
#OK I'll save this one for later - it's pretty
#sensitive to initial conditions
#Test a function with an infinite number of minima
#Note if this one is initialized with x[1] close to
#0 you're getting close to the mess of minima and
#may get unexpected results
def test_2():
    def f(y,x,grad):
        print(x)
        y[0] = np.cos(1/x[1]) # objective function
        y[1] = (2*x[0] + 0)**3 - x[1] # constraint 1
        y[2] = (-1*x[0] + -1)**3 - x[1] # constraint 2
        if grad is not None:
            grad[0,0] = 0.0
            grad[1,0] = np.sin(1/x[1])/(x[1]**2)
            grad[0,1] = 3 * 2 * (2*x[0] + 0)**2
            grad[1,1] = -1.0
            grad[0,2] = 3 * -1 * (-1*x[0] + -1)**2
            grad[1,2] = -1.0
    
    test = CCSA(2,2,f,
        lb=[-np.inf,0],ub=[np.inf,np.inf],
        verbose=True,max_eval=50)
    x0 = np.array([1.234, 5])
    x = test(x0)
    print(x)
    
#Test a function that should run to infinity
#
def test_3():
    def f(y,x,grad):
        print(x)
        y[0] = 1/x[1] # objective function
        y[1] = 0*(2*x[0] + 0)**3 - 1/x[1] # constraint 1
        y[2] = 0*(-1*x[0] + -1)**3 - 1/x[1] # constraint 2
        if grad is not None:
            grad[0,0] = 0.0
            grad[1,0] = -1/(x[1]**2)
            grad[0,1] = 0*3 * 2 * (2*x[0] + 0)**2
            grad[1,1] = 1.0/(x[1]**2)
            grad[0,2] = 0*3 * -1 * (-1*x[0] + -1)**2
            grad[1,2] = 1.0/(x[1]**2)
    
    test = CCSA(2,2,f,
        lb=[-np.inf,0],ub=[np.inf,np.inf],
        verbose=True,max_eval=50)
    x0 = np.array([1.234, 0.1])
    x = test(x0)
    print(x)

if __name__ == "__main__":
    test_0()
