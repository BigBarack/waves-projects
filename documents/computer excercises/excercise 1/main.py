import numpy as np
import matplotlib.pyplot as plt
from tools import post_Afout_Pfout

from matplotlib.animation import ArtistAnimation
def step_SIT_SIP(nx,ny,c,dx,dy,dt):
    global ox ,oy,p 
    ox[1:-1, :] += -dt/dx*(p[1:,:]-p[:-1,:]) 
    oy[:, 1:-1] += -dt/dy*(p[:,1:]-p[:,:-1]) 
    p[1:-1, 1:-1] += -c**2 * dt * (1/dx * (ox[2:-1, 1:-1] - ox[1:-2, 1:-1]) + 1/dy * (oy[1:-1, 2:-1] - oy[1:-1, 1:-2])     # Difference for `oy` excluding edges
)
    p[0,1:-1]  += -c**2 * dt * ( 1/dx * (ox[1,1:-1]-ox[-1,1:-1]) + 1/dy* (  oy[0,2:-1]  -oy[0,1:-2]))
    p[-1,1:-1] += -c**2 * dt * ( 1/dx * (ox[0,1:-1]-ox[-2,1:-1]) + 1/dy* (  oy[-1,2:-1]  -oy[-1,1:-2]))
    p[1:-1,0]  += -c**2 * dt * ( 1/dx * (ox[2:-1,0]-ox[1:-2,0]) + 1/dy* (  oy[1:-1,1]  -oy[1:-1,0]))
    p[1:-1,-1]  += -c**2 * dt * ( 1/dx * (ox[2:-1,-1]-ox[1:-2,-1]) + 1/dy* (  oy[1:-1,0]  -oy[1:-1,-2]))
    #p[1:-1,1:-1] += -c**2 * dt * ( 1/dx * (ox[1:,:]-ox[:-1,:]) + 1/dy* (oy[:,1:]-oy[:,:-1]))


    return None
    # implementeer deze functie // implement this function
 


#INITIALISATIE 2D-GRID EN SIMULATIEPARAMETERS-----------------------------
#INITIALISATION 2D-GRID AND SIMULATION PARAMETERS-------------------------

c=340 #geluidssnelheid - speed of sound (wave speed)
dx=0.2 #ruimtelijke discretisatiestap - spatial discretisation step
dy=dx

nx=175 #aantal cellen in x richting - number of cells in x direction
ny=175 #aantal cellen in y richting - number of cells in y direction

CFL=1 #Courant getal - Courant number

dt=CFL/(c*np.sqrt((1/dx**2)+(1/dy**2))) #tijdstap - time step

nt=175//CFL #aantal tijdstappen in simulatie - number of time steps

#locatie bron (midden van grid) en ontvangers
#location of source(central) and receivers
x_bron=int(nx/2)
y_bron=int(ny/2)

x_recorder=x_bron+30
y_recorder=y_bron+30 #plaats recorder 1 - location receiver 1
x_ref=x_bron+20
y_ref=y_bron+20 #plaats referentie 1 - location reference receiver 1

x_recorder2=x_bron+30
y_recorder2=y_bron #plaats recorder 2 - location receiver 2
x_ref2=x_bron+20
y_ref2=y_bron #plaats referentie 2 - location reference receiver 2

#pulse gegevens 
#source pulse information
A=10
fc=100
t0=2.5E-2
sigma=5E-5

#initialisatie snelheids- en drukvelden
#initialisation of o and p fields 
global ox, oy, p
ox = np.zeros((nx+1, ny))
oy = np.zeros((nx, ny+1))
p = np.zeros((nx, ny)) 

#film
#movie

#initialisatie tijdsreeks recorders
#initialisation time series receivers
recorder = np.zeros((nt,1))
recorder_ref = np.zeros((nt,1))

recorder2 = np.zeros((nt,1))
recorder2_ref = np.zeros((nt,1))

bront = np.zeros((nt,1))
tijdreeks=np.zeros((nt,1))
bron=0

#TIJDSITERATIE------------------------------------------------------
#TIME ITTERATION----------------------------------------------------
fig, ax = plt.subplots()
plt.axis('equal')
plt.xlim([1, nx+1])
plt.ylim([1, ny+1])
movie = []
for it in range(0, nt):
    t = (it-1)*dt
    tijdreeks[it, 0]=t
    print('%d/%d' % (it, nt))

    bron=A*np.sin(2*np.pi*fc*(t-t0))*np.exp(-((t-t0)**2)/(sigma)) #bron updaten bij nieuw tijd - update source for new time

    p[x_bron,y_bron] = p[x_bron,y_bron]+bron #druk toevoegen bij de drukvergelijking op bronlocatie - adding source term to propagation
    # p = np.random.random(nx * ny).reshape((nx, ny)) * 0.02*A
    step_SIT_SIP(nx,ny,c,dx,dy,dt)   #propagatie over 1 tijdstap - propagate over one time step

    recorder[it] = p[x_recorder,y_recorder] #druk opnemen op recorders en referentieplaatsen - store p field at receiver locations
    recorder_ref[it] = p[x_ref,y_ref]

    recorder2[it] = p[x_recorder2,y_recorder2]
    recorder2_ref[it] = p[x_ref2,y_ref2]

    #voorstellen drukveld
    #presenting the p field  
    
    #view(0,90)
    #shading interp
    artists = [
        ax.text(0.5,1.05,'%d/%d' % (it, nt), 
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes, ),
        ax.imshow(p, vmin=-0.02*A, vmax=0.02*A),
        ax.plot(x_bron,y_bron,'ks',fillstyle="none")[0],
        ax.plot(x_recorder,y_recorder,'ro',fillstyle="none")[0],
        ax.plot(x_ref,y_ref,'ko',fillstyle="none")[0],
        ax.plot(x_recorder2,y_recorder2,'ro',fillstyle="none")[0],
        ax.plot(x_ref2,y_ref2,'ko',fillstyle="none")[0]
        ]
    movie.append(artists)
    
    #mov(it) = getframe #wegcommentarieren voor simulatie vlugger te laten lopen - if this line is removed simulation runs much faster
my_anim = ArtistAnimation(fig, movie, interval=50, repeat_delay=1000,
                                   blit=True)
plt.show()
#movie(mov) #laten afspelen opgenomen simulatie - play back of stored movie

#NAVERWERKING : BEREKENING FASEFOUT en AMPLITUDEFOUT---------------------------------
#POST PROCESSING : CALCULATE PHASE and AMPLITUDE ERROR-------------------------------
n_of_samples=8192

post_Afout_Pfout(dx,dy,c,dt,x_ref,x_recorder,y_ref,y_recorder,x_bron,y_bron,recorder,recorder_ref,n_of_samples,tijdreeks,fc)
post_Afout_Pfout(dx,dy,c,dt,x_ref2,x_recorder2,y_ref2,y_recorder2,x_bron,y_bron,recorder2,recorder2_ref,n_of_samples,tijdreeks,fc)
