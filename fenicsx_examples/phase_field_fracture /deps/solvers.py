import dolfinx 
from dolfinx import mesh, fem, plot, io, la 
import ufl 
import petsc4py
from petsc4py import PETSc
import dolfinx.fem.petsc

# The LinearSolver class contians a linear solver for displacement (u)
class LinearSolver:
    def __init__(self, total_energy, V_u, u, bcu=[], petsc_options={}):
        self.u = u 
        self.V_u = V_u 
        self.total_energy = total_energy 
        self.bcu = bcu
        self.petsc_options = {"ksp_type": "preonly", "pc_type": "lu"} # Setting ksp and pc options for the linear solver s
        self.SolveLinearProblem()

    # Setting up the the forms for solving 
    def SolveLinearProblem(self):
        E_u = ufl.derivative(self.total_energy,self.u,ufl.TestFunction(self.V_u))
        E_du = ufl.replace(E_u,{self.u: ufl.TrialFunction(self.V_u)}) 

        a = ufl.lhs(E_du) # Extracts the bilinear form 
        L = ufl.rhs(E_du) # Extracts the  linear form 

        # Creates the PETSC Linear Problem 
        self.problem_u = dolfinx.fem.petsc.LinearProblem(a, L, bcs = self.bcu, u = self.u, petsc_options = self.petsc_options) 
        
    # Using the forms to solve for the displacement
    def DefSolver(self):
        # Solve the linear system 
        self.problem_u.solve()

# The NonlinearSolver class contains a nonlinear solver for displacement (u) 
class NonlinearSolver:
    def __init__(self, F, u, bcu, J=None, bounds = None):
        self.V = u.function_space
        du = ufl.TrialFunction(self.V)
        self.L = fem.form(F)
        if J is None:
            self.a = fem.form(F)
        else: 
            self.a = fem.form(J)
        self.bcu = bcu
        self._F, self._J = None, None
        self.u = u 

        self.b_u = la.create_petsc_vector(self.V.dofmap.index_map, self.V.dofmap.index_map_bs)
        self.J_u = fem.petsc.create_matrix(self.a)

        # The solver settings for the PETSc SNES nonlinear solver 
        self.solver = PETSc.SNES().create()
        self.solver.setType("newtonls")
        self.solver.setFunction(self.F, self.b_u)
        self.solver.setJacobian(self.J, self.J_u)
        self.solver.setTolerances(rtol=1.0e-9, max_it=50)
        self.solver.getKSP().setType("preonly")
        self.solver.getKSP().setTolerances(rtol=1.0e-9)
        self.solver.getKSP().getPC().setType("lu")

    def F(self, snes, x, F):
        # Assembling the resudual vector 
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        with F.localForm() as f_local:
            f_local.set(0.0)
        fem.petsc.assemble_vector(F, self.L)
        fem.petsc.apply_lifting(F, [self.a], bcs=[self.bcu], x0=[x], alpha=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(F, self.bcu, x, -1.0)

    def J(self, snes, x, J, P):
        # Assembling the Jacobian matrix 
        J.zeroEntries()
        fem.petsc.assemble_matrix(J, self.a, bcs=self.bcu)
        J.assemble()
        
    def DefSolver(self):
        # Solving the PETSc SNES solver set up above 
        self.solver.solve(None, self.u.x.petsc_vec)

# The AlphaSolver class contains a nonlinear solver for the damage (alpha)
class AlphaSolver:
    def __init__(self, F, u, bcalpha, J=None, bounds=None):
        self.V = u.function_space
        du = ufl.TrialFunction(self.V)
        self.L = fem.form(F)
        if J is None:
            self.a = fem.form(ufl.derivative(F, u, du))
        else:
            self.a = fem.form(J)
        self.bcalpha = bcalpha
        self._F, self._J = None, None
        self.u = u

        self.b_alpha = la.create_petsc_vector(self.V.dofmap.index_map, self.V.dofmap.index_map_bs)
        self.J_alpha = fem.petsc.create_matrix(self.a)

        # The solver settings for the PETSc SNES nonlinear solver 
        self.solver = PETSc.SNES().create()
        self.solver.setType("vinewtonrsls")
        self.solver.setFunction(self.F, self.b_alpha)
        self.solver.setJacobian(self.J, self.J_alpha)
        self.solver.setTolerances(rtol=1.0e-9, max_it=50)
        self.solver.getKSP().setType("preonly")
        self.solver.getKSP().setTolerances(rtol=1.0e-9)
        self.solver.getKSP().getPC().setType("lu")

        # Setting up the variable bounds 
        if bounds is not None:
            alpha_low, alpha_up = bounds
            self.solver.setVariableBounds(alpha_low.x.petsc_vec,alpha_up.x.petsc_vec)
            
    def F(self, snes, x, F):
        # Assembling the resudual vector 
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        with F.localForm() as f_local:
            f_local.set(0.0)
        fem.petsc.assemble_vector(F, self.L)
        fem.petsc.apply_lifting(F, [self.a], bcs=[self.bcalpha], x0=[x], alpha=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(F, self.bcalpha, x, -1.0)

    def J(self, snes, x, J, P):
        # Assembling the Jacobian matrix 
        J.zeroEntries()
        fem.petsc.assemble_matrix(J, self.a, bcs=self.bcalpha)
        J.assemble()
        
    def AlphaSolve(self):
        self.solver.solve(None, self.u.x.petsc_vec)


