import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Load the mesh file
mesh = meshio.read("tsunami.msh")

# Extract triangle elements and node coordinates
triangles = mesh.cells_dict.get("triangle", [])
nodes = mesh.points[:, :2]  # Extract only 2D coordinates

# Define boundaries
x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()

# Boundary node indices
left_boundary = np.where(nodes[:, 0] == x_min)[0]
right_boundary = np.where(nodes[:, 0] == x_max)[0]
bottom_boundary = np.where(nodes[:, 1] == y_min)[0]
top_boundary = np.where(nodes[:, 1] == y_max)[0]

# Dirichlet boundary condition on Γ1 (e.g., inner square in the middle of the domain)
inner_boundary = np.where((nodes[:, 0] > 0.4) & (nodes[:, 0] < 0.6) &
                          (nodes[:, 1] > 0.4) & (nodes[:, 1] < 0.6))[0]

# Map periodic boundaries
left_to_right = {l: r for l, r in zip(left_boundary, right_boundary)}
bottom_to_top = {b: t for b, t in zip(bottom_boundary, top_boundary)}

# Global parameters
k = 2 * np.pi / 4000  # Wavenumber, wavelength = 4000 km
num_nodes = len(nodes)
K = sp.lil_matrix((num_nodes, num_nodes), dtype=np.complex128)
b = np.zeros(num_nodes, dtype=np.complex128)

def local_stiffness(vertices):
    """Compute the local stiffness matrix for a triangle."""
    x = vertices[:, 0]
    y = vertices[:, 1]

    # Compute area of the triangle
    A = 0.5 * abs(
        x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1])
    )
    B = np.array([
        [y[1] - y[2], y[2] - y[0], y[0] - y[1]],
        [x[2] - x[1], x[0] - x[2], x[1] - x[0]]
    ])
    G = np.dot(B.T, B) / (4 * A)  # Gradient contribution
    M = (A / 12) * (np.ones((3, 3)) + 2 * np.eye(3))  # Mass matrix
    return G - k**2 * M, A

def assemble_global_system():
    """Assemble the global stiffness matrix and load vector."""
    for tri in triangles:
        vertices = nodes[tri]
        local_K, _ = local_stiffness(vertices)
        for i in range(3):
            for j in range(3):
                K[tri[i], tri[j]] += local_K[i, j]
    # Add source term to the load vector
    source = np.exp(-np.linalg.norm(nodes - np.array([3, 3]), axis=1)**2 / 0.1**2)
    b[:] = source

def apply_dirichlet_boundary(boundary_nodes):
    """Apply Dirichlet boundary conditions."""
    for node in boundary_nodes:
        K[node, :] = 0
        K[:, node] = 0
        K[node, node] = 1
        b[node] = 0

def apply_periodic_boundary():
    """Apply periodic boundary conditions."""
    # Handle left-to-right periodicity
    for left, right in left_to_right.items():
        K[right, :] += K[left, :]
        K[:, right] += K[:, left]
        b[right] += b[left]
        K[left, :] = 0
        K[:, left] = 0
        K[left, left] = 1
        b[left] = 0

    # Handle bottom-to-top periodicity
    for bottom, top in bottom_to_top.items():
        K[top, :] += K[bottom, :]
        K[:, top] += K[:, bottom]
        b[top] += b[bottom]
        K[bottom, :] = 0
        K[:, bottom] = 0
        K[bottom, bottom] = 1
        b[bottom] = 0

def solve():
    """Solve the linear system Ku = b."""
    A = sp.csr_matrix(K)
    u = spla.spsolve(A, b)
    return u

# Main assembly process
assemble_global_system()
apply_dirichlet_boundary(inner_boundary)  # Apply u=0 on Γ1
apply_periodic_boundary()  # Connect Γ2

# Solve the system
u = solve()

# Visualization (Optional)
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

triangulation = Triangulation(nodes[:, 0], nodes[:, 1], triangles)
plt.tricontourf(triangulation, np.real(u), cmap="viridis")
plt.colorbar(label="Wave amplitude")
plt.title("Wave Propagation with Periodic and Dirichlet Boundary Conditions")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


