import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy import ndimage
from scipy.special import jv, hankel1

class FDTD:

    def __init__(self,resolution = 0.1, domain_to_lmax = 1):
        """
        initialize instance of FDTD
        :param resolution: decides dx size compared to lmin
        :param domain_to_lmax: decides nx
        """
        # constants needed
        self.c = 340  # [m/s] speed of sound
        self.t0 = 1E-3  # time delay of pulse
        self.d=1
        self.A = 1  # Source amplitude
        # frequency band of interest
        self.f_min = self.c * 0.1 / (np.pi * self.d)
        self.f_max = self.c * 4 / (np.pi * self.d)
        self.fc = (self.f_min + self.f_max) / 2  # central frequency
        self.bandwidth = self.f_max - self.f_min
        self.l_min = np.pi * self.d / 2
        self.l_max = np.pi * self.d * 20
        #self.sigma = (1 / (2 * self.bandwidth)) ** 2  # width of pulse according to literature
        self.sigma = (4 * np.log(2)) / (np.pi * self.bandwidth)**2
        # discretization
        self.dx = resolution * self.l_min          # can still reduce dx to fix diagonal propagation
        self.dy = self.dx
        self.nx = int(domain_to_lmax * self.l_max / self.dx)
        self.ny = self.nx
        self.CFL = 1
        self.dt = self.CFL / (
                    self.c * np.sqrt((1 / self.dx ** 2) + (1 / self.dy ** 2)))  # time step from spatial disc. & CFL
        self.nt = int(self.nx // self.CFL)
        # initialize the fields
        self.oy = np.zeros((self.ny + 1, self.nx), dtype=np.float64)
        self.ox = np.zeros((self.ny, self.nx + 1), dtype=np.float64)  # if memory problems arise, can try float32
        self.p = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.wedge = None
        self.mask_normal_p = np.ones((self.ny, self.nx))
        self.mask_normal_ox = np.ones_like(self.ox, dtype=int)
        self.mask_normal_oy = np.ones_like(self.oy, dtype=int)
        # receivers and sources
        self.x_source = int(self.nx / 2)  # Source is at the center
        self.y_source = int(self.ny / 2)

        self.x_rec1 = int(self.x_source + self.d / self.dx)
        self.y_rec1 = self.y_source + int(1.5 * self.d / self.dx)
        self.x_rec2 = int(self.x_source + 2 * self.d / self.dx)
        self.y_rec2 = self.y_rec1
        self.x_rec3 = int(self.x_source + 3 * self.d / self.dx)
        self.y_rec3 = self.y_rec1

    def step_sit_sip(self):
        #oy
        self.oy[1:-1, :] += (-self.dt / self.dy * (self.p[1:, :] - self.p[:-1, :])
                             * self.mask_normal_oy[1:-1, :].astype(bool))
        if self.wedge is not None and self.BC == 'reflecting':
            self.oy[self.mask_perimeter_oy == 1] += 2 * self.dt * self.p[self.mask_perimeter_p == 1] / self.dx
        self.oy = self.oy * np.exp(-self.dampy[:, None])
        #ox
        self.ox[:, 1:-1] += (-self.dt / self.dx * (self.p[:, 1:] - self.p[:, :-1])
                             * self.mask_normal_ox[:, 1:-1].astype(bool))
        if self.wedge is not None and self.BC == 'reflecting':
            self.ox[self.mask_perimeter_ox == 1] += 2 * self.dt * self.p[self.mask_perimeter_p == 1] / self.dx
        self.ox = self.ox * np.exp(-self.dampx[None:, ])
        #p
        self.p += -self.c ** 2 * self.dt * (
                1 / self.dy * (self.oy[1:, :] - self.oy[:-1, :])  +
                + 1 / self.dx * (self.ox[:, 1:] - self.ox[:, :-1])
                    ) * self.mask_normal_p.astype(bool)
        if self.wedge is not None and self.BC == 'reflecting':
            masked_oy = self.oy * self.mask_perimeter_oy    # consider 0 outside of perimeter
            masked_ox = self.ox * self.mask_perimeter_ox    # consider 0 outside of perimeter
            self.p[self.mask_perimeter_p == 1] += -self.c ** 2 * self.dt * (
                1 / self.dy * (masked_oy[1:, :][self.mask_perimeter_oy[1:, :]==1] - masked_oy[:-1, :][self.mask_perimeter_oy[:-1, :]==1]) +
                + 1 / self.dx * (masked_ox[:, 1:][self.mask_perimeter_ox[:, 1:]==1] - masked_ox[:, :-1][self.mask_perimeter_ox[:, 1:]==1] )
                    )

        self.p = self.p
        return None


    def pml(self, sigmamax=0.35,thickness_denom = 7, pml_order = 3,prof_type = 'polynomial',scale=1):
        """
        pml_order: profile of pml; higher order means less reflections from inner side
        prof_type: either polynomial or exponential
        scale: to further reduce lattice mismatch of inner side of PML
        """
        # thickness needs to be small relatively to nx
        self.pml_thickness = max(1, self.nx // thickness_denom)
        # damping coëfficient grid in x and y
        self.dampx = np.zeros(self.nx + 1)
        self.dampy = np.zeros(self.ny + 1)
        def profile(index, thickness, sigmamax,pml_order,prof_type='polynomial',scale=1):
            return sigmamax * ((index / thickness) ** pml_order) if prof_type == 'polynomial' else (
                    sigmamax * (1 - np.exp(-pml_order * (index / thickness)**scale)))

        # only uses the values in the pml layer so that the non pml layer doesnt get affected
        for i in range(self.pml_thickness):
            #sigma_value = profile(i, self.pml_thickness, sigmamax, pml_order, prof_type, scale)
            sigma_value = profile(self.pml_thickness - 1 - i, self.pml_thickness, sigmamax, pml_order)

            self.dampx[i] = sigma_value
            self.dampx[-(i + 1)] = sigma_value
            self.dampy[i] = sigma_value
            self.dampy[-(i + 1)] = sigma_value
        #print(self.dampx)          #debugger
        return None

    def create_wedge(self,angle=0,surface='reflecting'):
        self.wedge = 1
        self.mask_wedge = np.zeros((self.ny, self.nx))
        self.BC = surface   #type of wedge surface
        wedge_start_x = self.x_rec1
        wedge_start_y = self.y_source - int(1 * self.d / self.dx)+3
        self.mask_wedge[wedge_start_y:, wedge_start_x:] = 1       #not including perimeter, just the wedge
        self.mask_wedge = np.flipud(self.mask_wedge)



        if angle != 0:
            # Extend the mask to prevent information loss during rotation
            xt = 1.4  # Scaling factor for the extended mask size
            new_nx = int(self.nx * xt)
            new_ny = int(self.ny * xt)
            extended_mask = np.zeros((new_ny, new_nx))

            # Position the wedge mask in the top-left of the extended array
            offset_x = (new_nx - self.nx) // 2
            offset_y = (new_ny - self.ny) // 2
            extended_mask[offset_y:offset_y + self.ny, offset_x:offset_x + self.nx] = self.mask_wedge

            # Ensure the wedge shape fills the grid up to its edges
            wedge_start_y = offset_y + self.y_source - int(self.d / self.dx)
            wedge_start_x = offset_x + self.x_rec1
            extended_mask[wedge_start_y + 1:, wedge_start_x + 1:] = 1
            extended_mask[:wedge_start_y + 1, :] = 0  # Clear parts above the wedge
            extended_mask = np.flipud(extended_mask)

            # Rotate the mask using ndimage.rotate
            rotated_mask = ndimage.rotate(extended_mask, angle, reshape=False, order=0)

            # Crop the rotated mask back to the original size
            crop_x_start = offset_x
            crop_x_end = offset_x + self.nx
            crop_y_start = offset_y
            crop_y_end = offset_y + self.ny
            self.mask_wedge = rotated_mask[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

            # Threshold to ensure binary mask (values can be fractional after rotation)
            np.clip(self.mask_wedge, 0, 1)


            # code_block to fix locations of source and receiver
            def rotate_xy(x,y,theta,nx,ny):
                c_x = nx / 2
                c_y = ny / 2
                shifted_x = x - c_x
                shifted_y = y - c_y
                rad_angle = np.deg2rad(theta)
                rotation_matrix = np.array([[np.cos(rad_angle), -np.sin(rad_angle)], [np.sin(rad_angle), np.cos(rad_angle)]])
                original_vector = np.array([shifted_y, shifted_x])   #position vector; to be rotated
                rotated = rotation_matrix @ original_vector
                new_x = rotated[0] + c_x
                new_y = rotated[1] + c_y
                return tuple(np.round([new_y, new_x]).astype(int))

            self.x_source, self.y_source = rotate_xy(self.x_source, self.y_source, angle,self.nx,self.ny)
            self.x_rec1, self.y_rec1 = rotate_xy(self.x_rec1, self.y_rec1, angle,self.nx,self.ny)
            self.x_rec2, self.y_rec2 = rotate_xy(self.x_rec2, self.y_rec2, angle,self.nx,self.ny)
            self.x_rec3, self.y_rec3 = rotate_xy(self.x_rec3, self.y_rec3, angle,self.nx,self.ny)
        #boundary cells
        struct = np.ones((3, 3))  # 8-connected neighborhood, used for the dilation
        self.mask_perimeter_p = ndimage.binary_dilation(self.mask_wedge, structure=struct).astype(int) - self.mask_wedge
        self.mask_perimeter_ox = np.zeros_like(self.ox, dtype=int)
        self.mask_perimeter_oy = np.zeros_like(self.oy, dtype=int)
        self.mask_perimeter_ox[:, 1:] = self.mask_perimeter_p
        self.mask_perimeter_oy[1:, :] = self.mask_perimeter_p
        self.mask_normal_ox[:, 1:] = 1- self.mask_wedge
        self.mask_normal_oy[1:, :] = 1 - self.mask_wedge
        self.mask_normal_ox -= self.mask_perimeter_ox
        self.mask_normal_oy -= self.mask_perimeter_oy
        self.mask_normal_p += - self.mask_wedge
        self.mask_normal_p += - self.mask_perimeter_p

    def debugger(self):
        fig, ax1 = plt.subplots(figsize=(8, 8))
        #plt.imshow(self.mask_wedge, origin='lower', cmap='gray')
        # Plot source and receivers with same style as animation
        ax1.contourf(self.mask_wedge, levels=[0.5, 1], colors='black', linestyles='--')
        #ax1.contour(self.mask_perimeter_p, levels=[0.5], colors='yellow', linestyles='--')
        #ax1.contour(self.rot, levels=[0.5], colors='black', linestyles='--')   #inbetween stages for mask rotation
        #ax1.contourf(self.extended, levels=[0.5, 1], colors='yellow', linestyles='--') # //
        ax1.plot(self.x_source, self.y_source, 'rs', fillstyle="none", label='Source')
        ax1.plot(self.x_rec1, self.y_rec1, 'ko', fillstyle="none", label='Receiver 1')
        ax1.plot(self.x_rec2, self.y_rec2, 'ko', fillstyle="none", label='Receiver 2')
        ax1.plot(self.x_rec3, self.y_rec3, 'ko', fillstyle="none", label='Receiver 3')
        ax1.plot(self.x_rec3, self.y_rec3, 'ko', fillstyle="none", label='Receiver 3')
        plt.title('Mask Placement with source and receivers')
        plt.show()
        print(f'fmin={self.f_min}, fc={self.fc}, fmax={self.f_max}')



    def iterate(self,reference='No'):
        receiver1 = np.zeros((self.nt, ))
        receiver2 = np.zeros((self.nt, ))
        receiver3 = np.zeros((self.nt, ))
        if reference == 'Yes':
            free1 = np.zeros((self.nt, ))
            free2 = np.zeros((self.nt, ))
            free3 = np.zeros((self.nt, ))
            dist1 = np.sqrt(13)*self.d / 2
            dist2 = 2.5 * self.d
            dist3 = 1.5 * self.d * np.sqrt(5)
            def p_free(dist,time): #approximation of free_space
               time_delayed = time - dist/ self.c - self.t0
               return (self.A / dist) * np.sin(2 * np.pi * self.fc * time_delayed) * np.exp(-(time_delayed ** 2) / self.sigma )
            # def p_free(dist,time):    #free)space if hankel function was used, issue with postprocessing expecting real-valued
            #     time_delayed = time - self.t0
            #     source_t = (self.A / dist) * np.sin(2 * np.pi * self.fc * time_delayed) * np.exp(-(time_delayed ** 2) / self.sigma )
            #     return np.real(hankel1(0, (2 * np.pi * self.fc / self.c)*dist) * source * (1j/4))

        timeseries = np.zeros((self.nt, ))

        fig, ax = plt.subplots()
        plt.axis('equal')
        plt.xlim([1, self.nx + 1])
        plt.ylim([1, self.ny + 1])
        movie = []

        for it in range(0, self.nt):
            t = (it - 1) * self.dt
            timeseries[it, ] = t
            print('%d/%d' % (it, self.nt))  # Loading bar while sim is running

            source = self.A * np.sin(2 * np.pi * self.fc * (t - self.t0)) * np.exp(
                -((t - self.t0) ** 2) / self.sigma)  # Update source for new time

            self.p[self.y_source, self.x_source] += source  # Adding source term to propagation
            self.step_sit_sip()  # Propagate over one time step

            receiver1[it] = self.p[self.y_rec1, self.x_rec1]  # Store p field at receiver locations
            receiver2[it] = self.p[self.y_rec2, self.x_rec2]
            receiver3[it] = self.p[self.y_rec3, self.x_rec3]
            if reference == 'Yes':
                #store p field at receiver locations - free space
                free1[it] = p_free(dist1,t)
                free2[it] = p_free(dist2,t)
                free3[it] = p_free(dist3,t)
            # Presenting the p field
            artists = [
                ax.text(0.5, 1.05, '%d/%d' % (it, self.nt),
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes),
                ax.imshow(self.p, vmin=-0.02 * self.A, vmax=0.02 * self.A,origin='lower'),
                ax.plot(self.x_source, self.y_source, 'rs', fillstyle="none")[0],
                ax.plot(self.x_rec1, self.y_rec1, 'ko', fillstyle="none")[0],
                ax.plot(self.x_rec2, self.y_rec2, 'ko', fillstyle="none")[0],
                ax.plot(self.x_rec3, self.y_rec3, 'ko', fillstyle="none")[0]
            ]
            if self.wedge != None:
                 artists.append(ax.contourf(self.mask_wedge, levels=[0.5,1], colors='black', linestyles='--'))
            movie.append(artists)
        print('iterations done')
        my_anim = ArtistAnimation(fig, movie, interval=10, repeat_delay=1000,
                                  blit=True)
        plt.show()
        if reference == 'Yes':
            return timeseries, receiver1, receiver2, receiver3, free1, free2, free3
        return timeseries, receiver1, receiver2, receiver3


def task1():
    problem = FDTD(0.1, 1)
    problem.pml(sigmamax=0.35, thickness_denom=7, pml_order=3)
    problem.create_wedge(0)
    timeseries, r1, r2, r3, free1, free2, free3 = problem.iterate(reference='Yes')
    parameters = np.array([problem.dt, problem.t0, problem.A, problem.sigma, problem.c, problem.d, problem.f_min, problem.f_max, problem.d])

    #----arrays were saved externally during development to avoid rerunning the sim -----
    # np.savez('receivers',timeseries=timeseries, r1=r1, r2=r2, r3=r3, free1=free1, free2=free2, free3=free3,parameters=parameters)
    # data = np.load('receivers.npz')
    # r1, free1, parameters = data['r1'], data['free1'], data['parameters']
    # r2, free2 = data['r2'], data['free2']
    # r3, free3 = data['r3'], data['free3']
    # timeseries = data['timeseries']
    def frequency_ratio(rec, free, i, param, show_full=True ):
        """
        performs FFT on 2 real-valued timeseries and plots relevant data
        :param rec: timeseries array of receiver
        :param free: timeseries array at free space
        :param i: receiver numbering, used for title
        :param param: parameters of the sim
        :param show_full: without cutoff after limit frequency
        :return: None
        """
        dt = param[0]
        N = rec.size
        assert N == free.size, 'different length arrays'
        p_wedge_F = np.fft.rfft(rec)
        p_free_F = np.fft.rfft(free)
        freqs = np.fft.rfftfreq(N, dt)
        epsilon = 0.001 #avoid deviding by 0
        ratio = np.abs(p_wedge_F) / np.abs(p_free_F + epsilon)
        f_limit = 2 * param[7]
        f_limit_index = np.searchsorted(freqs, f_limit)
        # if show_full:
        #     f_limit_index = -1
        print(freqs.size,p_wedge_F.size)    #debugger
        # ----p(f) PLOT----
        # plt.plot(freqs[:-1], p_wedge_F[:-1], label="receiver")
        # plt.plot(freqs[:-1], p_free_F[:-1], label="free space")
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Pressure")
        # plt.title(f"Frequency Spectrum of Receiver {i} and Free Space at same distance")
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        
        # #----|p(f)| PLOT----
        plt.plot(freqs[:-1], np.abs(p_wedge_F)[:-1], label="receiver")
        plt.plot(freqs[:-1], np.abs(p_free_F)[:-1], label="free space")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title(f"Frequency Spectrum of Receiver {i} and Free Space at same distance")
        plt.legend()
        plt.grid(True)
        plt.show()
        # ----RATIO PLOT----
        plt.plot(freqs[1:f_limit_index], ratio[1:f_limit_index])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(" $|p_{wedge}(f)| / |p_{free}(f)|$")
        plt.title(f"Magnitude Ratio of Receiver {i} to Free Space versus frequency")
        plt.grid(True)
        plt.show()
        return None

    frequency_ratio(r1, free1, 1, parameters, show_full=True)
    frequency_ratio(r2, free2, 2, parameters, show_full=False)
    frequency_ratio(r3, free3, 3, parameters, show_full=False)

    # ----ANALYTIC PART----
    # parameters
    dt = parameters[0]
    t0 = parameters[1]
    c = parameters[4]
    A = parameters[2]
    d = parameters[8]
    sigma = parameters[3]
    fmin = parameters[6]
    fmax = parameters[7]
    fc = ( fmin + fmax ) / 2
    kc = 2 * np.pi * fc / c
    timesteps = timeseries.size
    wedge_angle = 3 / 2 * np.pi #the wedge angle (or alpha) is defined as the outside angle
    # distances of source and receivers from wedge edge
    dist0 = np.sqrt(2) * d
    dist1 = 0.5 * d
    dist2 = np.sqrt(3 / 4) * d
    dist3 = np.sqrt(5 / 4) * d
    # angles of source and receivers to top_side of wedge
    phi0 = 5 / 4 * np.pi
    phi1 = 1 / 2 * np.pi
    phi2 = np.arcsin(d / (2 * dist2))
    phi3 = np.arcsin(d / (2 * dist3))
    # analytic computation
    def source_term(t, t0=t0, A=A, fc=fc, sigma=sigma):
        return A * np.sin(2 * np.pi * fc * (t - t0)) * np.exp(-((t - t0) ** 2) / sigma)
    sourceF = np.fft.rfft(np.array([source_term(tt) for tt in timeseries])) #DFT of source term to be used for analytical
    frequencies = np.fft.rfftfreq(timeseries.size,timeseries[2]-timeseries[1])

    def compute_uh(r, pfi, r0=dist0, pfi0=phi0, k=kc, alpha=wedge_angle, terms=100):
        u_h = 0
        for l in range(terms):
            e_l = 1 / 2 if l == 0 else 1
            v_l = (2 * l / 3) if alpha - (
                        3 * np.pi / 2) < 0.001 else l * np.pi / alpha  # for 90deg wedge, avoiding pi calc
            J_vl = jv(v_l, k * r) if r <= r0 else jv(v_l, k * r0)
            H_vl = hankel1(v_l, k * r0) if r <= r0 else hankel1(v_l, k * r)
            # accumulate for number of terms with angular part
            u_h += e_l * H_vl * J_vl * np.cos(v_l * pfi) * np.cos(v_l * pfi0)
        u_h *= np.pi / (alpha * 1j)
        return u_h * (1/4)

    rec1_analytical = np.zeros((frequencies.size,), dtype=complex)
    rec2_analytical = np.zeros((frequencies.size,), dtype=complex)
    rec3_analytical = np.zeros((frequencies.size,), dtype=complex)
    for i, f in enumerate(frequencies):
        If = sourceF[i]
        rec1_analytical[i] = compute_uh(dist1, phi1, k=(2*np.pi*f/c)) * If
        rec2_analytical[i] = compute_uh(dist2, phi2, k=(2*np.pi*f/c)) * If
        rec3_analytical[i] = compute_uh(dist3, phi3, k=(2*np.pi*f/c)) * If
    #plots did not turn out as expected
    plt.plot( frequencies,rec1_analytical, label="receiver 1 analytical")
    plt.plot( frequencies,rec2_analytical, label="receiver 2 analytical")
    plt.plot( frequencies,rec3_analytical, label="receiver 3 analytical")
    plt.xlabel("Frequency ")
    plt.ylabel("Pressure")
    plt.legend()
    plt.grid(True)
    plt.show()
    return None

#task1()        #run for task1

#task2 we did not have time to implement the complex alternative. However the way the code was written, the implementation
#would have been relatively simple by changing self.BC='complex' and using the correct update equation for that condition

def task3():
    """
    rotates the wedge and repeats the simulation for the same grid and for half gridcell
    :return:
    """
    problem = FDTD(0.1, 1)
    problem.pml(sigmamax=0.35, thickness_denom=7, pml_order=3)
    problem.create_wedge(30)
    _, r1, r2, r3 = problem.iterate(reference='No')

    problem2 = FDTD(0.05, 1)
    problem2.pml(sigmamax=0.35, thickness_denom=7, pml_order=3)
    problem2.create_wedge(30)
    _, r1, r2, r3 = problem2.iterate(reference='No')
#task3()    #run for task3

#task4 we did not implement this. We would have adjusted the grid cells into trapezoids matching the geometry of the
#rotated wedge to avoid the spurious reflections
