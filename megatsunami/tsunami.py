import gmsh
import numpy as np


import meshio
import scipy.sparse as sp
import scipy.sparse.linalg as spla
# Initialize Gmsh
gmsh.initialize()
gmsh.model.add("Mega-Tsunami")
gmsh.write("mega_tsunami_mesh.msh", format="msh2")
# Define Geometry: Outer ocean and inner coastal lines
outer_square = gmsh.model.occ.addRectangle(0, 0, 0, 10, 10)  # Large ocean domain
inner_square = gmsh.model.occ.addRectangle(3, 3, 0, 4, 4)    # Small coastal region
gmsh.model.occ.cut([(2, outer_square)], [(2, inner_square)], removeTool=False)
gmsh.model.occ.synchronize()

# Define Physical Groups for Boundary Conditions
outer_boundary = gmsh.model.getBoundary([(2, outer_square)], oriented=False)
inner_boundary = gmsh.model.getBoundary([(2, inner_square)], oriented=False)

outer_boundary_tags = [b[1] for b in outer_boundary]
inner_boundary_tags = [b[1] for b in inner_boundary]

gmsh.model.addPhysicalGroup(1, outer_boundary_tags, 1)  # Absorbing boundary
gmsh.model.addPhysicalGroup(1, inner_boundary_tags, 2)  # Dirichlet boundary
gmsh.model.setPhysicalName(1, 1, "Absorbing Boundary")
gmsh.model.setPhysicalName(1, 2, "Dirichlet Boundary")

# Generate and export the mesh
gmsh.model.mesh.generate(2)
gmsh.write("mega_tsunami_mesh.msh")
gmsh.finalize()


# Load the mesh using meshio
mesh = meshio.read("mega_tsunami_mesh.msh")

# Extract nodes and elements
nodes = mesh.points[:, :2]  # 2D coordinates
triangles = mesh.cells_dict["triangle"]

# Create a global stiffness matrix
num_nodes = len(nodes)
K = sp.lil_matrix((num_nodes, num_nodes), dtype=np.complex128)
b = np.zeros(num_nodes, dtype=np.complex128)

# Define wave number
k = 2 * np.pi / 4000  # Wavelength = 4000 km

# Function to compute local stiffness matrix
def compute_element_stiffness(coords, k):
    x = coords[:, 0]
    y = coords[:, 1]
    area = 0.5 * abs(x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1]))
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / (2 * area)
    c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / (2 * area)
    grad_term = np.outer(b, b) + np.outer(c, c)
    mass_term = (1 / 12) * np.ones((3, 3))
    np.fill_diagonal(mass_term, 1 / 6)
    return area * grad_term - k**2 * area * mass_term

# Assemble the global stiffness matrix
for tri in triangles:
    indices = tri
    coords = nodes[indices]
    Ke = compute_element_stiffness(coords, k)
    for i in range(3):
        for j in range(3):
            K[indices[i], indices[j]] += Ke[i, j]

# Apply Dirichlet Boundary Conditions
dirichlet_nodes = np.unique(mesh.cell_data["gmsh:physical"]["line"][inner_boundary_tags])
for node in dirichlet_nodes:
    K[node, :] = 0
    K[node, node] = 1
    b[node] = 0

# Apply Absorbing Boundary Conditions
absorbing_nodes = np.unique(mesh.cell_data["gmsh:physical"]["line"][outer_boundary_tags])
for node in absorbing_nodes:
    K[node, node] += 1j * k


# Solve the linear system
u = spla.spsolve(K.tocsr(), b)

# Visualize the solution
import matplotlib.pyplot as plt
plt.tricontourf(nodes[:, 0], nodes[:, 1], triangles, np.real(u), cmap="viridis")
plt.colorbar(label="Re(u)")
plt.title("Wave Pressure Distribution (Real Part)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


