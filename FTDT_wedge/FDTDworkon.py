import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


def step_SIT_SIP(nx, ny, c, dx, dy, dt):
    global ox, oy, p

    #need to change these to update slicing upto 1 cell before the wedge (1)
    ox[1:-1, :] += -dt/dx*(p[1:, :] - p[:-1, :])
    oy[:, 1:-1] += -dt/dy*(p[:, 1:] - p[:, :-1])
    p += -c**2*dt*(1/dx*(ox[1:, :] - ox[:-1, :]) + 1/dy*(oy[:, 1:] - oy[:, :-1]))

    ox[0,:]=0
    oy[-1,:]=0
    ox[0,:]=0
    oy[-1,:]=0
    p[0,:]=0
    p[-1,:]=0
    return None

    
    return ox,oy,p


d = 1           #arbitrary value for d
c = 340         #speed of sound
A = 1           #source amplitude (arbitrary)

f_min=c * 0.1 /(np.pi*d)
f_max=c * 4 /(np.pi*d)
fc = (f_min + f_max ) / 2               # central frequency
bandwidth = f_max - f_min
l_min = np.pi * d / 2
l_max = np.pi * d * 20

t0 = 1E-3                                    # time delay, determines when pulse is emitted, in continuous source (no exp windowing) it changes the phase
sigma = (1 / (2 * bandwidth)) **2               # determines width of pulse. according to literature, bandwidth=1/(2*np.sqrt(sigma))
dx = 0.1 * l_min                           # spatial discretisation step = 1/10 of lamda_min
dy = dx
#can reduce dx a bit further to correct diagonal propagation

#nw  will need to be expanded due to pml thickness once implemented
nx = int( l_max  / dx)
ny = nx
CFL = 1


dt = CFL / (c * np.sqrt((1 / dx ** 2)+(1 / dy ** 2)))      # time step from spatial disc. & CFL
nt = int(nx // CFL)

####PML implementation####--------------------------------------------------------------------------------------------------------------------
## we need to create a thicknesllayer that absorbs it so much that when it comes back we can reduce it to 0

#(2)

####initialize grid fields####-----------------------------------------------------------------------------------------------------------------
global ox, oy, p
ox = np.zeros((nx+1, ny),dtype=np.float64)
oy = np.zeros((nx, ny+1),dtype=np.float64)                 # when memory problems arise, can try float32
p = np.zeros((nx, ny),dtype=np.float64)                    # can sometimes lead to errors in the animation and overflow...

#### source and reciever positions####---------------------------------------------------------------------------------------------------------
#x_source=int(nx/8)
#y_source=int(ny/2)         # source d/2 from the left, center in y means 1 d on top air, the rest 3d are blocked
x_source=int(nx/2)
y_source=int(ny/2)


x_rec1 = int(x_source + d / dx)
y_rec1 = y_source + int( 1.5 * d / dx)
x_rec2 = int(x_source + 2 * d / dx)
y_rec2 = y_rec1
x_rec3 = int(x_source + 3 * d / dx)
y_rec3 = y_rec1

#### initialize source variable, time series and tables of receiver & reference
receiver1 = np.zeros((nt,1))
receiver2 = np.zeros((nt,1))
receiver3 = np.zeros((nt,1))

tijdreeks=np.zeros((nt,1))       # time series, for the axis in post_processing, name of variable unchanged to help with using the tools import if needed
source=0

####The FDTD iteration, can be made into a function to run for different frequencies and other changed parameters------------------------------------------
fig, ax = plt.subplots()
plt.axis('equal')
plt.xlim([1, nx+1])
plt.ylim([1, ny+1])
movie = []

for it in range(0, nt):
    t = (it-1)*dt
    tijdreeks[it, 0]=t
    print('%d/%d' % (it, nt))                                             # works like a loading bar while sim is running
                                                                        # helpful to identify problematic iterations during debugging
    source = A*np.sin(2*np.pi*fc*(t-t0))*np.exp(-((t-t0)**2)/(sigma))     # update source for new time

    p[x_source,y_source] += source                    # adding source term to propagation
    step_SIT_SIP(nx,ny,c,dx,dy,dt)                                        # propagate over one time step

    receiver1[it] = p[x_rec1,y_rec1]                                         # store p field at receiver locations
    receiver2[it] = p[x_rec2,y_rec2]                                      
    receiver3[it] = p[x_rec3, y_rec3]
      # presenting the p field

    artists = [
        ax.text(0.5,1.05,'%d/%d' % (it, nt),
                      size=plt.rcParams["axes.titlesize"],
                      ha="center", transform=ax.transAxes, ),
        ax.imshow(p, vmin=-0.02*A, vmax=0.02*A),                                   #original animation code
        #ax.imshow(np.clip(p, -0.02*A, 0.02*A)),                                   #by clipping the values inside this range, the error no longer persists
        ax.plot(x_source,y_source,'rs',fillstyle="none")[0],
        ax.plot(x_rec1,y_rec1,'ko',fillstyle="none")[0],
        ax.plot(x_rec2,y_rec2,'ko',fillstyle="none")[0],
        ax.plot(x_rec3,y_rec3,'ko',fillstyle="none")[0],

          ]
    movie.append(artists)

print('iterations done')
my_anim = ArtistAnimation(fig, movie, interval=10, repeat_delay=1000,
                          blit=True)
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


