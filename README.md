# ccsa

This repository contains a Python implementation of a nonlinear optimization algorithm based on Conservative Convex Separable Approximations [1]. So far only the quadratic approximation is implemented.

# Be warned: 
This code does not work as of 7/8/22. It's currently in the testing phase. The tests.py file will be updated steadily to hold mathematical functions that verify the convergence of the method. 

After fixing all the bugs, I will implement more convergence criteria (absolute tolerance, etc.) and modify some of the function handles to make things easier to use.

Many thanks to [@smartalecH](https://github.com/smartalecH?tab=overview/) for much of the initial implementation.

[1] K. Svanberg, A class of globally convergent optimization methods based on conservative convex separable approximations, SIAM Journal of Optimization, 2002, 12, 555-573.
