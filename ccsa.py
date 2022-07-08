import numpy as np


_RHO_SCALE       = 0.1
_RHO_DEFAULT     = 1.0
_RHO_MIN         = 1e-5 # smallest value rho can accept
_SIGMA_DEFAULT   = 1.0
_SIGMA_MAX_SCALE = 10.0
_SIGMA_MIN_SCALE = 1e-8 # much smaller than the svanberg paper default of 0.01
_GAMMA_INCREASE  = 1.2
_GAMMA_DECREASE  = 0.7
sub_solver_defaults = {
}

class CCSA():
    def __init__(self,
        n,                                      # number of DOF
        nc,                                     # number of constraints
        f,                                      # objective function
        stop_val = None,
        ftol_rel: float = 0.0,
        ftol = None,
        xtol = None,
        xtol_rel: float = 0.0,
        max_inner: int = np.inf,                # maximum number of inner iterations
        max_outer: int = np.inf,                # maximum number of outer iterations
        max_eval:  int = np.inf,                # maximum number of function evaluations
        lb: int = -np.inf,                      # lower bounds
        ub: int = np.inf,                       # upper bounds
        sub_solver_params: dict = None,         # dictionary of subsolver parameters
        verbose: bool = False                   # verbosity setting
        ):
        self.n = n 
        self.nc = nc
        
        '''the objective function oracle must have arguments of the form:
            def f(x,result,grad):
                ...
        where x is the input array, size [n]
        result is [m+1] where m is the number of nonlinear constraints
        grad is [n,m+1]
        '''
        self.f = f
        self.inner  = 0 # inner iteration number
        self.outer  = 0 # outer iteration number
        self.max_inner = max_inner
        self.max_outer = max_outer
        self.max_eval  = max_eval
        self.stop_val = stop_val
        self.ftol_rel = ftol_rel
        self.ftol = ftol
        self.xtol = xtol
        self.xtol_rel = xtol_rel
        self.total_iters = 0 # running count of our function calls
        self.verbose = verbose
        
        # Clean the upper and lower bounds
        if isinstance(lb, (list, tuple, np.ndarray)):
            self.lb = np.asarray(lb).flatten()
        elif np.isscalar(lb):
            self.lb = np.ones((self.n,)) * lb
        else:
            raise ValueError("Lower bounds must either be a scalar or numpy array")
        if isinstance(ub, (list, tuple, np.ndarray)):
            self.ub = np.asarray(ub).flatten()
        elif np.isscalar(lb):
            self.ub = np.ones((self.n,)) * ub
        else:
            raise ValueError("Upper bounds must either be a scalar or numpy array")

        # Establish the subsolver default parameters
        if sub_solver_params:
            self.sub_solver_params = sub_solver_params
        else: 
            self.sub_solver_params = sub_solver_defaults

        self.y = None

    def __call__(self,x0):
        self.x   = x0.copy()   # current DOF
        self.x_1 = x0.copy()   # previous DOF
        self.x_2 = x0.copy()   # k-2 previous DOF
        self.df = np.zeros((self.n,self.nc+1))
        self.y = np.zeros((self.nc+1,))
        self.sigma = 0.5 * (self.ub-self.lb)
        self.sigma[self.sigma==-np.inf] = _SIGMA_DEFAULT
        self.sigma[self.sigma==np.inf]  = _SIGMA_DEFAULT
        self.rho = np.ones((self.nc+1,)) * _RHO_DEFAULT

        self.y_prev = np.zeros((self.nc+1,))

        self.needs_outer = True
        self.needs_inner = True

        while ((self.outer < self.max_outer) and (self.needs_outer) and (self.total_iters < self.max_eval)):
            if self.verbose:
                print("Current outer iteration: ",self.outer)
            
            # reset the inner iteration data
            self.needs_inner = True
            self.inner = 0

            # evaluate the current point and gradient
            self.y_prev[:] = self.y[:]
            self.f(self.y,self.x,self.df)
            self.total_iters += 1

            # perform inner iterations
            while ((self.inner < self.max_inner) and (self.needs_inner) and (self.total_iters <= self.max_eval)):
                if self.verbose:
                    print("Current inner iteration: ",self.inner)

                # solve the subproblem
                x_hat, y_hat = self.solve_subproblem()
                print("xhat",x_hat)

                # now call f using x_hat to check if it's conservative
                self.y_prev[:] = self.y[:]
                self.f(self.y,x_hat,None)
                self.total_iters += 1

                if self.verbose:
                    print("y_hat: {} | y: {} ".format(y_hat,self.y))
                if (self._check_if_conservative(self.y,y_hat)):
                    if self.verbose:
                        print("CONSERVATIVE")
                    # subproblem was sufficiently conservative
                    self.needs_inner = False
                    self.x_2[:] = self.x_1[:]   # rollback
                    self.x_1[:] = self.x[:]     # rollback
                    self.x[:]   = x_hat         # only cache the update if it looks good
                else:
                    # we need to make our subproblem more conservative...
                    delta = self._calculate_delta(x_hat,self.y,y_hat)
                    self._make_more_conservative(delta)
                
                self.inner += 1
                if self.check_termination():
                    return self.x
                
            
            # inner iterations are done
            self._make_less_conservative()
            self.outer += 1
            if self.check_termination():
                return self.x

        return self.x
    
    def check_termination(self):
        if self.total_iters >= self.max_eval:
            print("Reached Maximum Evaluation Limit")
            return True
        diff = np.abs(self.y[0] - self.y_prev[0])
        #if (self.xtol_rel is not None) and (np.norm()):
        #    xtol_rel
        #The recursive subproblem invariably converges to self.y[0] == 0 and self.y[0] - self.y_prev[0] == 0
        #The normal evaluation method below would give a numpy warning
        if(diff == 0 and self.outer > 0):
            return True
        
        
        if(self.outer > 0 and (diff/ self.y_prev[0]) < self.ftol_rel):
            print(self.y[0],self.y_prev[0])
            print("val: {} | {}".format((np.abs(self.y[0] - self.y_prev[0]) / self.y_prev[0]),self.ftol_rel))
            return True
        return False       
    
    '''Approximating functions used to model
    the subproblem'''
    def w(self,x0,x):
        return 0.5 * np.sum(((x-x0) / self.sigma) ** 2)
    def v(self,f0,x0,x,grad):
        return f0 + np.dot(grad,(x-x0))
    def g(self,rho,f0,x0,x,grad):
        return self.v(f0,x0,x,grad) + rho*self.w(x0,x)
    
    def dual(self,λ,y,grad,x):
        # calculate x from dual equations
        # assume x is initialized to current step
        # vectorize later
        for j in range(x.size):
            u = self.sigma[j]**2 * self.df[j,0]
            v = self.rho[0]
            if λ is not None:
                print("λ: {} df: {}".format(λ,self.df[j,1:]))
                u += self.sigma[j]**2 * np.dot(λ,self.df[j,1:])
                v += np.dot(λ,self.rho[1:])
            x[j] += -(u/v)
            x[j] = self.lb[j] if (x[j] < self.lb[j]) else x[j]
            x[j] = self.ub[j] if (x[j] > self.ub[j]) else x[j]

        # calculate dual from x
        y = self.g(self.rho[0],self.y[0],self.x,x,self.df[:,0])
        grad_temp = []
        if λ is not None:
            for r in range(λ.size):
                grad_temp.append(self.g(self.rho[1+r],self.y[1+r],self.x,x,self.df[:,1+r]))
                y += λ[r] * grad_temp[r]
        if (grad is not None):
            print("grad, ",grad)
            # we are maximizing, so negate function and gradient
            grad[:,0] = -np.asarray(grad_temp)
        
        # we are maximizing, so negate function and gradient
        return -y, grad

    def solve_subproblem(self):
        ''' approximate the function with a set of
        convex quadratics. In the rare case where there
        are no nonlinear constraints, we can solve the
        dual problem using just one iteration/evaluation.
        Otherwise, the dual is a maximization problem,
        which we can approximate (again) using a the CCSA
        algorithm.
        '''
        x = np.zeros(self.x.shape)
        # no constraints
        if self.y.size == 1:
            y = 0 # dummy in place (not needed)
            grad = None # dummy in place (not needed)
            λ0 = None
            x = self.x.copy()
            self.dual(λ0,y,grad,x)
            y = self.g(self.rho,self.y,self.x,x,np.squeeze(self.df))
            return x, y
        # constraints
        else:
            n = self.y.size-1
            λ0 = np.ones((n,))
            sub_opt = CCSA(
                n, # number of DOF
                0, # number of constraints
                lambda y,λ,grad: self.dual(λ,y,grad,x),
                lb=np.zeros(λ0.shape),
                ub=np.ones(λ0.shape)*np.inf,
                max_eval=100,
                **self.sub_solver_params
            )
            # find the lagrange multipliers that work
            λ_hat = sub_opt(λ0)

            # evaluate dual with final multipliers to get primal variables
            y = 0
            grad = np.zeros((n,1))
            self.dual(λ_hat,y,grad,x)
            y = np.zeros((self.y.shape))
            y[0] = self.g(self.rho[0],self.y[0],self.x,x,self.df[:,0])
            for k in range(n):
                y[k+1] = self.g(self.rho[k+1],self.y[k+1],self.x,x,self.df[:,k+1])
            return x, y
    
    def _check_if_conservative(self,y,y_hat):
        ''' ensure that the dual objective function and
        all of the dual constraint functions are strictly
        conservative wrt the actual ob func and constraints'''
        result = True
        y = np.asarray(y)
        y_hat = np.asarray(y_hat)        
        for y_i,y_hat_i in zip(y,y_hat):
            result = result & (y_hat_i >= y_i)
        return result
    
    def _calculate_delta(self,x_hat,y,y_hat):
        delta = (y - y_hat) / self.w(self.x,x_hat)
        return delta
    
    def _make_more_conservative(self,delta):
        for k in range(self.nc+1):
            if delta[k] > 0:
                self.rho[k] = min(10*self.rho[k],1.1*(self.rho[k]+delta[k]))
            else:
                continue
    
    def _make_less_conservative(self):
        # update rho array
        for k in range(self.nc+1):
            self.rho[k] = max(_RHO_SCALE*self.rho[k],_RHO_MIN)
        
        # update sigma array
        for j in range(self.n):
            if (self.ub[j]<np.inf) and (self.lb[j]>-np.inf):
                max_min = self.ub[j]-self.lb[j]
                if (self.outer < 2):
                    self.sigma[j] = 0.5*max_min
                else:
                    gamma = 0
                    cond = (self.x[j]-self.x_1[j])*(self.x_1[j]-self.x_2[j])
                    if cond > 0:
                        gamma = _GAMMA_DECREASE
                    elif cond < 0:
                        gamma = _GAMMA_INCREASE
                    else:
                        gamma = 1
                    self.sigma[j] = gamma*self.sigma[j]
                    self.sigma[j] = min(self.sigma[j],_SIGMA_MAX_SCALE*max_min)
                    self.sigma[j] = max(self.sigma[j],_SIGMA_MIN_SCALE*max_min)

if __name__ == "__main__":
    def f(result, x, grad):
        print("ALEC ",x)
        result[:] = np.dot(x,x) + 2
        if grad is not None:
            grad[:] = 2*x.reshape((x.size,1))
    
    #x = np.ones((4,))
    #x = np.random.rand(4)
    x = np.asarray([1,2,3,4])
    test = CCSA(4,0,f,verbose = True,max_eval=20)
    test(x)
