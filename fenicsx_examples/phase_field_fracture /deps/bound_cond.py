import dolfinx 
from dolfinx import mesh, fem, plot, io, la 
import ufl
import numpy as np 
import pyvista 
from typing import Optional
import petsc4py
from petsc4py import PETSc

# The prescribed_disp_x function establishes the boundary conditions such that the part fixed on the left and the bottom and a prescribed displacement in the on the right edge in the positive x-direction 
def prescribed_disp_x (domain, V_u, V_alpha, facet_tags):
    fdim = domain.topology.dim-1 # creating the dim-1 for the facetes
    
    xBot_facets = facet_tags.find(2) # left facets 
    xTop_facets = facet_tags.find(3) # right facets
    yBot_facets = facet_tags.find(4) # bottom facets
    yTop_facets = facet_tags.find(5) # top facets
    
    xBot_dofs_ux = fem.locate_dofs_topological(V_u.sub(0), fdim, xBot_facets) # x-displacement DOF's on xBot 
    xTop_dofs_ux = fem.locate_dofs_topological(V_u.sub(0), fdim, xTop_facets) # x-displacement DOF's on xTop
    yBot_dofs_uy = fem.locate_dofs_topological(V_u.sub(1), fdim, yBot_facets) # y-displacement DOF's on yBot
    
    u_D = fem.Constant(domain,PETSc.ScalarType(0.)) # prescribed dispacement of constant value 1 
    bcu_l = fem.dirichletbc(0.0, xBot_dofs_ux, V_u.sub(0)) #  Dirichlet BC on left edge in x-direction 
    bcu_r = fem.dirichletbc(u_D, xTop_dofs_ux, V_u.sub(0)) # Dirichlet BC on rigt edge in x-direction 
    bcu_b = fem.dirichletbc(0.0, yBot_dofs_uy, V_u.sub(1)) # Dirichlet BC on bottom edge in y-direction 
    bcu = [bcu_l, bcu_r, bcu_b] # Vector of displacement boundary conditions 
    
    xBot_dofs_alphax = fem.locate_dofs_topological(V_alpha, fdim, xBot_facets) # Damage DOF's on left edge 
    xTop_dofs_alphax = fem.locate_dofs_topological(V_alpha, fdim, xTop_facets) # Damage DOF's on right edge 

    # The damage BC's do not have to be applied but are an option 
    bcalpha_l = fem.dirichletbc(0.0, xBot_dofs_alphax, V_alpha) # Damage BC on left edge 
    bcalpha_r = fem.dirichletbc(0.0, xTop_dofs_alphax, V_alpha) # Damage BC on right edge 
    bcalpha = [] # Vector of damage boundary conditions 

    return bcu, bcalpha, u_D

# The prescribed_disp_y function establishes the boundary conditions such that the part is fixed on the left edge and a prescribed displacment is applied to the right edge in the positive y-direction 
def prescribed_disp_y (domain, V_u, V_alpha, facet_tags):
    fdim = domain.topology.dim-1 # creating the dim-1 for the facetes
    
    xBot_facets = facet_tags.find(2) # left facets 
    xTop_facets = facet_tags.find(3) # right facets
    yBot_facets = facet_tags.find(4) # bottom facets
    yTop_facets = facet_tags.find(5) # top facets
    
    xBot_dofs_ux = fem.locate_dofs_topological(V_u.sub(0), fdim, xBot_facets) # x-displacement DOF's on xBot 
    xBot_dofs_uy = fem.locate_dofs_topological(V_u.sub(1), fdim, xBot_facets) # y-displacement DOF's on xBot 
    xTop_dofs_ux = fem.locate_dofs_topological(V_u.sub(0), fdim, xTop_facets) # x-displacement DOF's on xTop
    xTop_dofs_uy = fem.locate_dofs_topological(V_u.sub(1), fdim, xTop_facets) # y-displacement DOF's on xTop 
    
    u_D = fem.Constant(domain,PETSc.ScalarType(1.)) # prescribed dispacement of constant value 1 
    bcu_lx = fem.dirichletbc(0.0, xBot_dofs_ux, V_u.sub(0)) # Dirichlet BC on left edge in x-direction 
    bcu_ly = fem.dirichletbc(0.0, xBot_dofs_uy, V_u.sub(1)) # Dirichlet BC on left edge in y-direction 
    bcu_r = fem.dirichletbc(u_D, xTop_dofs_uy, V_u.sub(1)) # Dirichlet BC on right edge in y-direction 
    bcu = [bcu_lx, bcu_ly, bcu_r] # Vector of displacement boundary conditions 

    # The damage BC's do not have to be applied but are an option 
    xBot_dofs_alphax = fem.locate_dofs_topological(V_alpha, fdim, xBot_facets) # Damage DOF's on left edge 
    xTop_dofs_alphax = fem.locate_dofs_topological(V_alpha, fdim, xTop_facets) # Damage DOF's on riht edge 
    
    bcalpha_l = fem.dirichletbc(0.0, xBot_dofs_alphax, V_alpha) # Damage BC on left edge 
    bcalpha_r = fem.dirichletbc(0.0, xTop_dofs_alphax, V_alpha) # Damage BC on right edge 
    bcalpha = [] # Vector of damage boundary conditions

    return bcu, bcalpha, u_D