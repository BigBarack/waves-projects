# Importing necessary modules
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

# Parameters
k = 1  # Wavenumber
N = 200 # Number of source points
M = 200 # Number of observation points
xs = np.linspace(-1, 1, N, dtype=complex)  # Observation points
xp = np.linspace(-1, 1, M, dtype=complex)  # Source points
# Initializing A matrix and b vector as complex-valued arrays
A = np.zeros((N, M), dtype=complex)
b = np.zeros(N, dtype=complex)
#for when it becomes zero make it become epsilon
eps=10**(-5)
print(eps)
# Loop to compute A matrix and b vector
for i in range(N):
    # Midpoint of xp segment
    xp_mid = (xp[i] + xp[i- 1]) / 2 if i > 0 else (xp[i] + xp[i]) / 2
        
    # Length of xp segment 
    li = xp[i] - xp[i - 1] if i > 0 else xp[i + 1] - xp[i]

    
    b[i] += li * np.exp(k * -1j * xp[i])
    
    
    for j in range(M):  
        # Length of xs segment
        lj = xs[j] - xs[j - 1] if j > 0 else xs[j + 1] - xs[j]
        if (xp_mid - xs[j])!=0:

            z1=np.abs(xp_mid - xs[j]) 
        else:

            z1=eps
        if (xp_mid - xs[j])!=0:

            z2=np.abs(xp_mid - xs[j])
        else:
            z2=eps
        # Hankel function contributions at segment endpoints
        G1 = scp.special.hankel2(0, z1) 
        G2 = scp.special.hankel2(0, z2)

        

        # Update the element of A
        A[i, j] += 0.5 * li * lj * (G1 + G2)
    

print(A,b)

# Solve the linear system
q = np.linalg.solve(A, b)

qn=np.fft.ifft(q)


# Plotting the results
plt.plot(xs.real, qn.real, label="Real Part", color="blue", linewidth=2)
plt.plot(xs.real, qn.imag, label="Imaginary Part", color="red", linewidth=2)

# Adding labels and legend
plt.xlabel("xs")
plt.ylabel("q")
plt.title("q values")
plt.legend()
plt.grid(True)
plt.show()




'''
The heavy oscillations occur because when Xs and Xp become arbitraroly close so the diagonal elements and near diagonal elements
if we now use the small argument approximation we get the following'''




A = np.zeros((N, M), dtype=complex)
b = np.zeros(N, dtype=complex)
#for when it becomes zero make it become epsilon
eps=10**(-5)
print(eps)
# Loop to compute A matrix and b vector
for i in range(N):
    # Midpoint of xp segment
    xp_mid = (xp[i] + xp[i- 1]) / 2 if i > 0 else (xp[i] + xp[i]) / 2
        
    # Length of xp segment 
    li = xp[i] - xp[i - 1] if i > 0 else xp[i + 1] - xp[i]

    
    b[i] += li * np.exp(k * -1j * xp[i])
    
    
    for j in range(M):  
        # Length of xs segment
        lj = xs[j] - xs[j - 1] if j > 0 else xs[j + 1] - xs[j]
        if (xp_mid - xs[j])!=0:

            z1=np.abs(xp_mid - xs[j]) 
        else:

            z1=eps
        if (xp_mid - xs[j])!=0:

            z2=np.abs(xp_mid - xs[j])
        else:
            z2=eps
        # Hankel function contributions at segment endpoints
        G1 = -1/4*1j-1/(2*np.pi)*(np.euler_gamma+np.log10(k*z1/2)) 
        G2 = -1/4*1j-1/(2*np.pi)*(np.euler_gamma+np.log10(k*z2/2)) 

        

        # Update the element of A
        A[i, j] += 0.5 * li * lj * (G1 + G2)
    

print(A,b)

# Solve the linear system
q = np.linalg.solve(A, b)

qn=np.fft.ifft(q)


# Plotting the results
plt.plot(xs.real, qn.real, label="Real Part", color="blue", linewidth=2)
plt.plot(xs.real, qn.imag, label="Imaginary Part", color="red", linewidth=2)

# Adding labels and legend
plt.xlabel("xs")
plt.ylabel("q")
plt.title("q values")
plt.legend()
plt.grid(True)
plt.show()