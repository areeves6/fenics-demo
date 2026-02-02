import dolfinx 
from dolfinx import mesh, fem, plot, io, la 
import ufl 
from dolfinx.io import gmshio
import numpy as np 
import sympy
import mpi4py
from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import os 
import sys 
import dolfinx.fem.petsc
from solvers import LinearSolver, AlphaSolver, NonlinearSolver

# The form_prob fuction forms the functions used to solve the problem, the functionals to solve the problem, and establishes the solvers
def form_prob(state, conditions, properties, tdim, dx, domain, matl_type, spaces, bcs, bounds, ):

    u = state["u"]
    alpha = state["alpha"]
    alpha_low = bounds["alpha_low"]
    alpha_up = bounds["alpha_up"]
    E_0 = conditions["E_0"]
    Gc_0 = conditions["Gc_0"]
    ell_ = conditions["ell_"]
    V_u = spaces["V_u"]
    V_alpha = spaces["V_alpha"]
    E = properties["E"]
    nu = properties["nu"]
    Gc = properties["Gc"]
    bcu = bcs["bcu"]
    bcalpha = bcs["bcalpha"]
        
    if matl_type ["hyperelastic"]:

    #Defining the functions for the energy functional
        def w(alpha): # Continoiusly increasing function 
            return alpha ** 2
        def a(alpha, k_res = 1.e-6): # Degradation function (continually decreasing) 
            return (1 - alpha) ** 2 + k_res
        # Lame parameters 
        mu = E / (2.0 * (1.0 + nu))
        lmbda = E * nu / (1.0 - nu ** 2)
        I = (ufl.Identity(tdim)) # Identity matrix 
        F = (I + ufl.grad(u)) # Deformation gradient 
        C = (F.T * F) # Right Cauchy Green Tensor 
        Ic = (ufl.tr(C)) # Invariant dependent on C 
        J = (ufl.det(F)) # Jacobian 

        #Defining the different energy functionals 
        if matl_type ["hypermodel"] == "neoHookean1":
            def W_0(u):
                return (mu / 2) * (J**(-2/3)*Ic - 3) + (lmbda/2)*(J - 1)**2
        elif matl_type ["hypermodel"] == "neoHookean2":
            def W_0(u):
                return (mu / 2) * (Ic - 3 - 2 * ufl.ln(J)) + (lmbda / 2) * (J - 1) ** 2
                
        def W(u, alpha):
            return a(alpha) * W_0(u)

        #Defining the necessary constants for solving the problem
        b = sympy.Symbol("b") #integration symbol 
        c_w = 4*sympy.integrate(sympy.sqrt(w(b)),(b,0,1)) # constant associated with w(alpha)
        ell = fem.Constant(domain, PETSc.ScalarType(ell_)) # Characteristic length 
        init = 0.00017160499171694368 / 2 # Initial guess 

        print("c_w = ", c_w)
        print("ell = ", ell.value)
        print("init = ", init)

        f = fem.Constant(domain, PETSc.ScalarType((0.0,0.0))) # magnitude of the external forces (x,y)
        elastic_energy = W(u,alpha) * dx #elastic energy
        fracture_energy = Gc / float(c_w) * (w(alpha) / ell + ell * ufl.dot(ufl.grad(alpha), ufl.grad(alpha))) * dx #fracture energy
        external_work = ufl.dot(f,u) * dx #external work 
        total_energy = elastic_energy + fracture_energy - external_work #total enery functional 

        
        E_u = ufl.derivative(total_energy, u, ufl.TestFunction(V_u)) # Residual form for the displacement 
        E_u_u = ufl.derivative(E_u, u, ufl.TrialFunction(V_u)) # Jacobian for the nonlinear solver 

        solver_u = NonlinearSolver(E_u, u, bcu, J=E_u_u) 

        E_alpha = ufl.derivative(total_energy, alpha, ufl.TestFunction(V_alpha)) # Residual form for the displacement 
        E_alpha_alpha = ufl.derivative(E_alpha, alpha, ufl.TrialFunction(V_alpha)) # Jacobian for the nonlinear solver 
        
        bounds = (alpha_low, alpha_up)
        
        damage_problem = AlphaSolver(E_alpha, alpha, bcalpha, J=E_alpha_alpha, bounds=bounds)

        return total_energy, init, solver_u, damage_problem

    # The else case is for the linear elasatic case 
    else:
        def w(alpha): # continious strictly increasing function 
            return alpha
        
        def a(alpha, k_res = 1.e-6): # degradation function (continious strictly decreasing)
            return (1 - alpha) ** 2 + k_res
        
        def eps(u): # linearized strain function 
            return ufl.sym(ufl.grad(u))
        
        def sigma_0(u): # stress tensor of the undamaged material 
            mu = E / (2.0 * (1.0 + nu))
            lmbda = E * nu / (1.0 - nu ** 2)
            return 2.0 * mu  * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(tdim)
        
        def sigma(u,alpha): #stress tensor of the damaged material 
            return a(alpha) * sigma_0(u)

        b = sympy.Symbol("b") # Integration symbol 
        c_w = 4*sympy.integrate(sympy.sqrt(w(b)),(b,0,1)) # Constant associated with w(alpha)
        ell = fem.Constant(domain, PETSc.ScalarType(ell_)) # Characteristic Length 
        init = 0.00017160499171694368 # Initial Guess 

        print("c_w = ", c_w)
        print("ell = ", ell.value)
        print("init = ", init)

        f = fem.Constant(domain, PETSc.ScalarType((0.0,0.0))) # magnitude of the external forces (x,y)
        elastic_energy = 0.5 * ufl.inner(sigma(u,alpha), eps(u)) * dx #elastic energy
        fracture_energy = Gc / float(c_w) * (w(alpha) / ell + ell * ufl.dot(ufl.grad(alpha), ufl.grad(alpha))) * dx #fracture energy
        external_work = ufl.dot(f,u) * dx #external work 
        total_energy = elastic_energy + fracture_energy - external_work #total enery functional 

        solver_u = LinearSolver(total_energy, V_u, u, bcu)

        E_alpha = ufl.derivative(total_energy, alpha, ufl.TestFunction(V_alpha)) # Residual form for the damage 
        E_alpha_alpha = ufl.derivative(E_alpha, alpha, ufl.TrialFunction(V_alpha)) # Jacobian for the nonlinear solver 
        
        bounds = (alpha_low, alpha_up)
        
        damage_problem = AlphaSolver(E_alpha, alpha, bcalpha, J=E_alpha_alpha, bounds=bounds)
    
        return total_energy, init, solver_u, damage_problem