import numpy as np
from scipy.special import hankel2
import matplotlib.pyplot as plt


def green_function(k, x, y):
    r = np.linalg.norm(x - y)
    if r==0:
        r =eps
    return -1j / 4 * hankel2(0, k * r)

def green_function_approx(k, x, y, gamma=0.5772156649):  # γ: Euler constant
    r = np.linalg.norm(x - y)
    if r==0 :
        r =eps
    return -1j / 4 - (1 / (2 * np.pi)) * (gamma + np.log(k * r / 2))
    
def incident_field(k, d, x):
    #return np.exp(-1j * k * np.dot(d, x))
    return np.dot(x,x**4)


# Parameters
k = 0.001  # Wavenumber
N = 200 # Number of source points
M = 200 # Number of observation points
xs = np.linspace(-1, 1, N, dtype=complex)  # Observation points
xp = np.linspace(-1, 1, M, dtype=complex)  # Source points



"""since in the diagonal elements the spaces are so close to eachother  you need to apply the close approximation
for it to work""""


for i in range(N):
    # Midpoint of xp segment
    xp_mid = (xp[i] + xp[i- 1]) / 2 if 0 < i < max(N,M) else (xp[i] + xp[i]) / 2
        
    # Length of xp segment 
    li = xp[i] - xp[i - 1] if i > 0 else xp[i + 1] - xp[i]

    
    b[i] = li * np.exp(k * -1j * xp[i])
    
    
    for j in range(M):  
        # Length of xs segment
        lj = xs[j] - xs[j - 1] if j > 0 else xs[j + 1] - xs[j]
        z1= max(np.abs(xp_mid - xs[j]), eps)

        z2 = max(np.abs(xp_mid - xs[j + 1]) if j < M - 1 else eps, eps)
        
        # Hankel function contributions at segment endpoints
        G1 = -1j/4*scp.special.hankel2(0, k*z1) 
        G2 = -1j/4*scp.special.hankel2(0, k*z2)

        

        # Update the element of A
        A[i, j] += 0.5 * li * lj * (G1 + G2)
    

print(A,b)
# Solve the linear system
q = np.linalg.solve(A, b)

qn=np.fft.ifft(q)


# Plotting the results
plt.plot(xs, np.abs(q), label="Real Part", color="blue", linewidth=2)
#plt.plot(xs.real, qn.imag, label="Imaginary Part", color="red", linewidth=2)

# Adding labels and legend
plt.xlabel("xs")
plt.ylabel("q")
plt.title("q values")
plt.legend()
plt.grid(True)

N = 50  # Number of segments
k = 2 * np.pi  # Wavenumber
length = 2.0  # Length of the segment Γ
dx = length / N
segments = [(-1 + i * dx, -1 + (i + 1) * dx) for i in range(N)]

eps= 10**(-8)

A = np.zeros((N, N), dtype=complex)

for i in range(N):
    xi_center = np.array([segments[i][0] + dx / 2, 0])  # Midpoint of Si
    for j in range(N):
        yj_left = np.array([segments[j][0], 0])  # Left endpoint of Sj
        yj_right = np.array([segments[j][1], 0])  # Right endpoint of Sj
        
        if i == j:  # Diagonal element correction
            A[i, j] = dx * green_function_approx(k, xi_center, xi_center)
        else:  # Off-diagonal elements
            A[i, j] = 0.5 * dx * dx * (green_function(k, xi_center, yj_left) +
                                        green_function(k, xi_center, yj_right))



d = np.array([1, 0])  # Direction of wave propagation
b = np.zeros(N, dtype=complex)

for i in range(N):
    xi_center = np.array([segments[i][0] + dx / 2, 0])  # Midpoint of Si
    b[i] = -incident_field(k, d, xi_center) * dx


q = np.linalg.solve(A, b)
x_values = np.array([segments[i][0] + dx / 2 for i in range(N)])  # Midpoints for plotting


plt.plot(x_values,np.abs(q) ,label='Re(q)')
#plt.plot(x_values, q.real, label='Re(q)')
#plt.plot(x_values, q.imag, label='Im(q)')
plt.legend()
plt.title("Solution q along the segment Γ")
plt.xlabel("x")
plt.ylabel("q")
plt.show()



def compute_electric_field(k, q, grid_x, grid_y, segments):
    field = np.zeros_like(grid_x, dtype=complex)
    dx = segments[0][1] - segments[0][0]
    
    for i in range(len(segments)):
        xi_center = np.array([segments[i][0] + dx / 2, 0])  # Midpoint of Si
        for gx, gy in np.ndindex(grid_x.shape):
            point = np.array([grid_x[gx, gy], grid_y[gx, gy]])
            field[gx, gy] += q[i] * dx * green_function(k, xi_center, point)
    
    return field


# Create a grid
x_grid = np.linspace(-1, 1, 100)
y_grid = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x_grid, y_grid)

# Compute electric field
E_field = compute_electric_field(k, q, X, Y, segments)

# Plot the field
plt.imshow(np.abs(E_field), extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
plt.colorbar(label='|Electric Field|')
plt.title("Electric Field Near the Scatterer")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
