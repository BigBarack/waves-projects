import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
# Load the .msh file
mesh = meshio.read("tsunami.msh")

# Extract triangle elements
triangles = mesh.cells_dict.get("triangle", [])
nodes = mesh.points[:, :2]  # 2D coordinates

# Create a global stiffness matrix
num_nodes = len(nodes)
K = sp.lil_matrix((num_nodes, num_nodes), dtype=np.complex128)
b = np.zeros(num_nodes, dtype=np.complex128)

# Define wave number
k = 2 * np.pi / 4000  # Wavelength = 4000 km

# Function to compute local stiffness matrix




