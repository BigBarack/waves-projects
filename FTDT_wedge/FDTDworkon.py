import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


def initialize_pml(nx, ny, thickness, sigma_max):
    """
    Initializes PML profiles for both velocity components and pressure fields.
    
    Parameters:
    - nx, ny: Grid dimensions.
    - thickness: Thickness of the PML in grid points.
    - sigma_max: Maximum conductivity in the PML.

    Returns:
    - sigma_x, sigma_y: Conductivity profiles for the x and y directions.
    """
    sigma_x = np.zeros((nx + 1, ny))
    sigma_y = np.zeros((nx, ny + 1))

    for i in range(thickness):
        absorption = sigma_max * ((thickness - i) / thickness) ** 2
        sigma_x[i, :] = sigma_x[-i - 1, :] = absorption
        sigma_y[:, i] = sigma_y[:, -i - 1] = absorption

    return sigma_x, sigma_y


def step_SIT_SIP_PML(nx, ny, c, dx, dy, dt, sigma_x, sigma_y):
    """
    Updates the fields in the simulation, including the PML terms.
    
    Parameters:
    - nx, ny: Grid dimensions.
    - c: Wave propagation speed.
    - dx, dy: Spatial steps.
    - dt: Time step.
    - sigma_x, sigma_y: Conductivity profiles for x and y directions.
    """
    global ox, oy, p

    # Update ox field
    ox[1:-1, :] = (1 - sigma_x[1:-1, :] * dt) * ox[1:-1, :] - dt / dx * (p[1:, :] - p[:-1, :])

    # Update oy field
    oy[:, 1:-1] = (1 - sigma_y[:, 1:-1] * dt) * oy[:, 1:-1] - dt / dy * (p[:, 1:] - p[:, :-1])

    # Update pressure field
    p += -c**2 * dt * (
        (1 / dx) * (ox[1:, :] - ox[:-1, :]) +
        (1 / dy) * (oy[:, 1:] - oy[:, :-1])
    )

    # Apply boundary conditions (reflection-free)
    p[0, :] = p[-1, :] = 0
    p[:, 0] = p[:, -1] = 0


# Constants and Initialization
d = 1
c = 340  # Speed of sound
A = 1  # Source amplitude
f_min = c * 0.1 / (np.pi * d)
f_max = c * 4 / (np.pi * d)
fc = (f_min + f_max) / 2
bandwidth = f_max - f_min
l_min = np.pi * d / 2
l_max = np.pi * d * 20

t0 = 1e-3
sigma = (1 / (2 * bandwidth))**2
dx = 0.1 * l_min
dy = dx
nx = int(l_max / dx)
ny = nx
CFL = 1
dt = CFL / (c * np.sqrt((1 / dx**2) + (1 / dy**2)))
nt = int(nx // CFL)

# Initialize Fields and PML
thickness = int(0.1 * nx)  # 10% of grid size as PML thickness
sigma_max = 50 # Maximum conductivity
sigma_x, sigma_y = initialize_pml(nx, ny, thickness, sigma_max)

global ox, oy, p
ox = np.zeros((nx + 1, ny), dtype=np.float64)
oy = np.zeros((nx, ny + 1), dtype=np.float64)
p = np.zeros((nx, ny), dtype=np.float64)

# Source and Receivers
x_source, y_source = nx // 2, ny // 2
x_rec1, y_rec1 = int(x_source + d / dx), int(y_source + 1.5 * d / dx)

# Initialize Visualization
fig, ax = plt.subplots()
plt.axis("equal")
plt.xlim([0, nx])
plt.ylim([0, ny])
movie = []

# FDTD Iteration
for it in range(nt):
    t = it * dt
    source = A * np.sin(2 * np.pi * fc * (t - t0)) * np.exp(-((t - t0)**2) / sigma)
    p[x_source, y_source] += source

    # Update Fields
    step_SIT_SIP_PML(nx, ny, c, dx, dy, dt, sigma_x, sigma_y)

    # Visualization
    artists = [
        ax.text(0.5, 1.05, f"Step {it}/{nt}", size=12, ha="center", transform=ax.transAxes),
        ax.imshow(p, vmin=-0.02 * A, vmax=0.02 * A),
    ]
    movie.append(artists)

# Create Animation
print("Simulation complete.")
my_anim = ArtistAnimation(fig, movie, interval=10, repeat_delay=1000, blit=True)
plt.show()


#this was our fft at the end last year, to be seen if we will need any parts of it
"""
####post processing
def fourier_transform_frequency_band(delta_t, f, f_min, f_max, axis=0):     # function to analyze the frequency responce for signal f
    F = np.fft.fft(f, axis=axis)                                            # Compute the Fourier Transform
    N = f.shape[axis]                                                       # Compute the frequency bins
    freq = np.fft.fftfreq(N, delta_t)                                       # Take only the positive frequencies up to N//2 (due to symmetry in the FFT of real signals)
    F_positive = F[:N//2]
    freq_positive = freq[:N//2]
    mask = (freq_positive >= f_min) & (freq_positive <= f_max)              # Limit the spectrum to frequencies within the specified band [f_min, f_max]
    F_band = F_positive[mask]
    freq_band = freq_positive[mask]
    return freq_band, F_band

freq_receiver, F_receiver = fourier_transform_frequency_band(dt, receiver, f_min, f_max)
freq_reference, F_reference = fourier_transform_frequency_band(dt, reference, f_min, f_max)
ratio_amplitudes = np.abs(F_receiver / F_reference)
plt.plot(freq_receiver, ratio_amplitudes)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Pressure Ratio (Receiver / Reference)')
plt.title(f'P-field Ratio for d = {undulation} * a')
plt.show()
"""

"""
issues:
(1) slice properly
(2) implement PML
(3) nx is too low, getting stuck on something stupid probably 
"""


