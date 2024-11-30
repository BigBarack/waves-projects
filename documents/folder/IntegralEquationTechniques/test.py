# Importing necessary modules
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

# Parameters
k = 1  # Wavenumber
N = 5 # Number of source points
M = 5 # Number of observation points
xs = np.linspace(-1, 1, N, dtype=complex)  # Observation points
xp = np.linspace(-1, 1, M, dtype=complex)  # Source points

# Initializing A matrix and b vector as complex-valued arrays
b = np.zeros(N, dtype=complex)

# Loop to compute A matrix and b vector
for i in range(N):
    # Midpoint of xp segment
    xp_mid = (xp[i] + xp[i- 1]) / 2 if i > 0 else (xp[i] + xp[i]) / 2
        
    # Length of xp segment 
    li = xp[i] - xp[i - 1] if i > 0 else xp[i + 1] - xp[i]

    
    b[i] += li * np.exp(k * 1j * xp[i])
    

print(b)


print(q)

# Plotting the results
plt.plot(xs.real, q.real, label="Real Part", color="blue", linewidth=3)
plt.plot(xs.real, q.imag, label="Imaginary Part", color="red", linewidth=3)

# Adding labels and legend
plt.xlabel("lambda")
plt.ylabel("q")
plt.title("Condition Number Plot")
plt.legend()
plt.grid(True)
plt.show()