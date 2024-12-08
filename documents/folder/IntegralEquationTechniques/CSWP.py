# Importing necessary modules
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import seaborn as sns


# Parameters
k = 0.001  # Wavenumber
N = 200 # Number of source points
M = 200 # Number of observation points
xs = np.linspace(-1, 1, N, dtype=complex)  # Observation points
xp = np.linspace(-1, 1, M, dtype=complex)  # Source points
# Initializing A matrix and b vector as complex-valued arrays
A = np.zeros((N, M), dtype=complex)
b = np.zeros(N, dtype=complex)
#for when it becomes zero make it become epsilon
eps=10**(-10)
print(eps)
# Loop to compute A matrix and b vector
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
A= (A+A.T)/2
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
plt.show()


'''
The heavy oscillations occur because when Xs and Xp become arbitraroly close so the diagonal elements and near diagonal elements
if we now use the small argument approximation we get the following'''




D = np.zeros((N, M), dtype=complex)
c = np.zeros(N, dtype=complex)
#for when it becomes zero make it become epsilon
eps=10**(-5)
print(eps)
# Loop to compute A matrix and b vector
for i in range(N):
    # Midpoint of xp segment
    xp_mid = (xp[i] + xp[i- 1]) / 2 if i > 0 else (xp[i] + xp[i]) / 2
        
    # Length of xp segment 
    li = xp[i] - xp[i - 1] if i > 0 else xp[i + 1] - xp[i]

    
    c[i] += li * np.exp(k * 1j * xp[i])
    
    
    for j in range(M):  
        # Length of xs segment
        lj = xs[j] - xs[j - 1] if j > 0 else xs[j + 1] - xs[j]
        
        z1= max(np.abs(xp_mid - xs[j]), eps)

        z2 = max(np.abs(xp_mid - xs[j + 1]) if j < M - 1 else eps, eps)
        
        # Hankel function contributions at segment endpoints
        G1 = -1/4*1j-1/(2*np.pi)*(np.euler_gamma+np.log(k*z1/2)) 
        G2 = -1/4*1j-1/(2*np.pi)*(np.euler_gamma+np.log(k*z2/2)) 

        

        # Update the element of A
        D[i, j] += 0.5 * li * lj * (G1 + G2)
    

print(A,b)
D= (D+D.T)/2
# Solve the linear system
r = np.linalg.solve(D, c)




# Plotting the results
plt.plot(r.real, r.imag, label="Real Part", color="blue", linewidth=2)

# Adding labels and legend
plt.xlabel("xs")
plt.yscale("log")
plt.ylabel("log scaled xs")
plt.title("q values")
plt.legend()
plt.grid(True)
plt.show()