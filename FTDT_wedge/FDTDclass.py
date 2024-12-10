import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


class FDTD():

    def __init__(self):
             #all the physical constants we need
            self.c = 340 #[m/s] speed of sound


            #miscelaneous
            self.t0 =1E-3 #time delay
        
           
           
            #initialize the fields
            self.plane_wave_init()
            self.discritization()
            self.ox = np.zeros((self.nx+1, self.ny),dtype=np.float64)
            self.oy = np.zeros((self.nx, self.ny+1),dtype=np.float64)                 # when memory problems arise, can try float32
            self.p = np.zeros((self.nx, self.ny),dtype=np.float64)     

            #all the physical constants we need
            self.c = 340 #[m/s] speed of sound


            #receivers and sources 

            self.x_source = int(self.nx / 2)  # Source is at the center
            self.y_source = int(self.ny / 2)


            self.A = 1  # Source amplitude
            self.x_rec1 = int(self.x_source + self.d / self.dx)
            self.y_rec1 = self.y_source + int(1.5 * self.d / self.dx)
            self.x_rec2 = int(self.x_source + 2 * self.d / self.dx)
            self.y_rec2 = self.y_rec1
            self.x_rec3 = int(self.x_source + 3 * self.d / self.dx)
            self.y_rec3 = self.y_rec1

            
    def plane_wave_init(self,d=20):

        """
        do this before discritization
        also this also initializng the parameters of the wavesource 
        """

        self.d=d

        self.f_min=self.c * 0.1 /(np.pi*d)
        self.f_max=self. c * 4 /(np.pi*d)
        self.fc = (self.f_min + self.f_max ) / 2               # central frequency
        self.bandwidth = self.f_max - self.f_min

        self.l_min = np.pi * self.d / 2
        self.l_max = np.pi * self.d * 20
        self.sigma = (1 / (2 * self.bandwidth)) **2            # width of pulse according to literature
        return None
    

    def discritization(self):
        self.dx = 0.1 * self.l_min                          #spatial discretisation of lambda_min
        self.dy = self.dx 

        # can still reduce it

        self.nx = int( self.l_max / self.dx)
        self.ny = self.nx

        self.CFL = 1

        self.dt = self.CFL / (self.c * np.sqrt((1 / self.dx ** 2) + (1 / self.dy ** 2)))    # time step from spatial disc. & CFL
        self.nt = int(self.nx // self.CFL)
        return None
    

    def step_SIT_SIP(self):
        

        #need to change these to update slicing upto 1 cell before the wedge (1)
        self.ox[1:-1, :] += -self.dt / self.dx * ( self.p[1:, :] - self.p[:-1, :] ) * np.exp( -self.dampx[1:-1,None])
        self.oy[:, 1:-1] += -self.dt / self.dy * ( self.p[:, 1:] - self.p[:, :-1] ) * np.exp( -self.dampx[None, 1:-1,])
        self.p += -self.c**2*self.dt*(1/self.dx*(self.ox[1:, :] - self.ox[:-1, :])  * np.exp( -self.dampx[:-1,None]) +
                  + 1/self.dy*(self.oy[:, 1:] - self.oy[:, :-1]) *  np.exp( -self.dampy[None,1:])
                  )
        self.ox = self.ox * np.exp( -self.dampx[:,None])
        self.oy = self.oy * np.exp( -self.dampy[None:,])
        self.p = self.p * np.exp(- self.dampx[:-1,None ] )
        return None
    

    def PML(self,sigmamax=0.17):
         #thickness needs to be small relatively to nx
         self.thickness=  max(1,self.nx//10)


        # damping coëfficients   
         
         
         self.dampx=np.zeros(self.nx + 1) 
         self.dampy=np.zeros(self.ny + 1)


         def profile(index,thickness,sigmamax):
              return sigmamax * ( (index /thickness) )


        
        # only uses the values in the pml layer so that the non pml layer doesnt get affected
         for i in range(self.thickness):
            sigma_value= profile(i,self.thickness,sigmamax)


            self.dampx[i] = sigma_value
            self.dampx[-(i + 1)] = sigma_value
            self.dampy[i] = sigma_value
            self.dampy[-(i + 1)] = sigma_value
         return None




    def iterate(self):
        receiver1 = np.zeros((self.nt,1))
        receiver2 = np.zeros((self.nt,1))
        receiver3 = np.zeros((self.nt,1))

        timeseries = np.zeros((self.nt,1))

        fig, ax = plt.subplots()
        plt.axis('equal')
        plt.xlim([1, self.nx+1])
        plt.ylim([1, self.ny+1])
        movie = []

        for it in range(0, self.nt):
            t = (it-1)*self.dt
            timeseries[it, 0] = t
            print('%d/%d' % (it, self.nt))  # Loading bar while sim is running

            source = self.A * np.sin(2 * np.pi * self.fc * (t - self.t0)) * np.exp(-((t - self.t0) ** 2) / (self.sigma))  # Update source for new time

            self.p[self.x_source, self.y_source] += source  # Adding source term to propagation
            self.step_SIT_SIP()  # Propagate over one time step

            receiver1[it] = self.p[self.x_rec1, self.y_rec1]  # Store p field at receiver locations
            receiver2[it] = self.p[self.x_rec2, self.y_rec2]
            receiver3[it] = self.p[self.x_rec3, self.y_rec3]

            # Presenting the p field
            artists = [
                ax.text(0.5, 1.05, '%d/%d' % (it, self.nt),
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes),
                ax.imshow(self.p, vmin=-0.02*self.A, vmax=0.02*self.A),  # Original animation code
                ax.plot(self.x_source, self.y_source, 'rs', fillstyle="none")[0],
                ax.plot(self.x_rec1, self.y_rec1, 'ko', fillstyle="none")[0],
                ax.plot(self.x_rec2, self.y_rec2, 'ko', fillstyle="none")[0],
                ax.plot(self.x_rec3, self.y_rec3, 'ko', fillstyle="none")[0],
            ]
            movie.append(artists)
        print('iterations done')
        my_anim = ArtistAnimation(fig, movie, interval=10, repeat_delay=1000,
                                blit=True)
        plt.show()
        return None




problem=FDTD()

problem.PML()
problem.iterate()
