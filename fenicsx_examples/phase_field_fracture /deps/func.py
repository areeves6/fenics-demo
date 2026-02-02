import dolfinx 
from dolfinx import mesh, fem, plot, io, la 
import ufl
import numpy as np 
import pyvista 
from typing import Optional
import petsc4py
from petsc4py import PETSc

# The MeshPlotter class creates a plot of the geometry and mesh 
class MeshPlotter:
    def __init__(self, domain, tdim, cell_tags = None):
        pyvista.OFF_SCREEN = False
        self.tdim = tdim 
        self.domain = domain 
        self.cell_tags = cell_tags
        topology, cell_types, points = dolfinx.plot.vtk_mesh (domain, self.tdim)
        self.grid = pyvista.UnstructuredGrid(topology, cell_types, points)

    # The plot is visualized using pyvista using the domain and the cell tags to display the size of the mesh, the location of the nodes, and any     areas with unique materials. The plot is displayed in the notebook when run. 
    def plot (self):
        num_local_cells = self.domain.topology.index_map(self.tdim).size_local

        if self.cell_tags is not None:
            self.grid.cell_data["Physical Groups"] = self.cell_tags.values[self.cell_tags.indices < num_local_cells]
            self.grid.set_active_scalars("Physical Groups")
            
        p = pyvista.Plotter(window_size=[800, 800])
        p.add_mesh(self.grid, show_edges=True)
        p.view_xy()
        p.show()

# The create_functions function creates the function spaces and functions as well as the limits on alpha and the integration measures        
def create_functions(domain):
    V_u = fem.functionspace(domain, ("Lagrange", 1, (2,))) # Function space for the displacement 
    V_alpha = fem.functionspace(domain, ("Lagrange",1)) # Function space for the damage 

    spaces = {"V_u": V_u, "V_alpha": V_alpha}

    u = fem.Function (V_u , name = "Displacement") # Displacement function 
    alpha = fem.Function (V_alpha, name = "Damage") # Damage function 

    state = {"u": u, "alpha": alpha}
   
    # Creates and sets up the upper and lower limits of alpha 
    alpha_low = fem.Function(V_alpha); alpha_low.x.array[:] = 0
    alpha_up = fem.Function(V_alpha); alpha_up.x.array[:] = 1

    bounds = {"alpha_low": alpha_low, "alpha_up": alpha_up}

    dx = ufl.Measure("dx",domain=domain) # Creates an integration measure for the integration over the domain 
    ds = ufl.Measure("ds",domain=domain) # Creates an integration measure for the integration over the surface
    return state, spaces, bounds, dx, ds

# The unpack_functions function brings the variables and functions out of their respective dictionaries (this is done as a function to streamline the process of using them in the main code) 
def unpack_functions(state, spaces, bounds):
    u = state["u"]
    alpha = state["alpha"]
    alpha_low = bounds["alpha_low"]
    alpha_up = bounds["alpha_up"]
    V_u = spaces["V_u"]
    V_alpha = spaces["V_alpha"]
    return u, alpha, alpha_low, alpha_up, V_u, V_alpha

# The material_properties function collects the values of the materials assigned in the main code to be assigned to arrays which hold the assignement for the location of each material (crated this way to allow for cases with two materials)    
def material_properties(material_1, material_2, domain, cell_tags, top_tag):

    Q = fem.functionspace (domain, ("DG", 0)) # Creating a function space for the materials 
    material_tags = np.unique(cell_tags.values) # Creating material tags to identify the locations of each material
    # Creating functions for each material property 
    E = fem.Function(Q)
    nu = fem.Function(Q)
    Gc = fem.Function(Q)

    # Iterating through the material_tags to assign the material property at each loacion in the model 
    for tag in material_tags:
        cells = cell_tags.find(tag)
        print(f"Tag {tag} has {len(cells)} cells")
        if tag == top_tag:
            E_ = material_1["E1"]
            nu_ = material_1["nu1"]
            Gc_ = material_1["Gc1"]
        else: 
            E_ = material_2["E2"]
            nu_ = material_2["nu2"]
            Gc_ = material_2["Gc2"]

        # The matrial properties at each location are stored in and used from the following arrays
        E.x.array[cells] = np.full_like(cells, E_, dtype = PETSc.ScalarType) # Young's Modulus 
        nu.x.array[cells] = np.full_like(cells, nu_, dtype = PETSc.ScalarType) # Poisson Ratio 
        Gc.x.array[cells] = np.full_like(cells, Gc_, dtype = PETSc.ScalarType) # Critical Fracture Energy 

    properties = {"E": E, "nu": nu, "Gc": Gc}
    return properties

# The AltMin class contains the alternate minimization process used for solving the displacement and damage problems     
class AltMin:
    def __init__ (self, state, alt_min_parameters, dx, solver_u, damage_problem, file_results, load=None):
        self.u = state["u"] 
        self.alpha = state["alpha"] 
        self.state = state
        self.dx = dx 
        self.solver_u = solver_u 
        self.damage_problem = damage_problem
        self.alt_min_parameters = alt_min_parameters
        self.file_results = file_results
        self.load = load

    # The simple_monitor function tracks and displays the number of iterations as well as the resulting displacement and damage vectors 
    def monitor(self, state, iteration, error_L2):
        print(f"Load: {self.load:.6e}, Iteration: {iteration:3d}, Error: {error_L2:3.4e}")
        u_arr = self.u.x.array
        alpha_arr = self.alpha.x.array
        # this works as a check to make sure that displacement and damage are occuring 
        print(f"  max(alpha) = {self.alpha.x.array.max():.4e}, min(alpha) = {self.alpha.x.array.min():.4e}")
        print(f"  max(u) = {self.u.x.array.max():.4e}, min(u) = {self.u.x.array.min():.4e}")
        
    # The alternate_minimization function iterates between solving for the displacement and the damage and calculates the error until it is           within tolerance for all load steps  
    def set_load(self, load):
        self.load = load
        
    def alternate_minimization(self, parameters, monitor=None):

        self.alpha_backup = fem.Function(self.alpha.function_space) # Function to store the previous damage 
        self.alpha.x.petsc_vec.copy(result=self.alpha_backup.x.petsc_vec) # Copy current damage values to the backup 

        iteration = 0
        error_L2 = np.inf
        
        while error_L2 > parameters["atol"] and iteration < parameters["max_iter"]:
                              
            # solve displacement
            self.solver_u.DefSolver()
        
            # solve damage
            self.damage_problem.AlphaSolve()
        
            # use the current alpha and previous alpha to calculate the L2 norm 
            diff = self.alpha - self.alpha_backup
            L2_error = ufl.inner(diff, diff) * self.dx
            error_L2 = np.sqrt(fem.assemble_scalar(fem.form(L2_error)))
            self.alpha.x.petsc_vec.copy(self.alpha_backup.x.petsc_vec) # Updating the new damage values to the backup 
        
            if monitor is not None:
                monitor(self.state, iteration, error_L2) # If the monitor is not none it uses and returns the simple monitor 

            iteration += 1 
            
        return (error_L2, iteration)

