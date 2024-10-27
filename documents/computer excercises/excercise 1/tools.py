import numpy as np
from scipy.special import hankel1, hankel2
from numpy.fft import fft
import matplotlib.pyplot as plt
def besselh(nu,K,Z):
    if K  ==  1:
        return hankel1(nu, Z)
    elif K  ==  2:
        return hankel2(nu, Z)
    else:
        raise ValueError()

def post_Afout_Pfout(dx,dy,c,dt,x_ref,x_recorder,y_ref,y_recorder,x_bron,y_bron,recorder,recorder_ref,n_of_samples,tijdreeks,fc):

    #generatie frequentie-as - generate frequency axes
    maxf = 1/dt

    df = maxf/n_of_samples

    fas = np.zeros((n_of_samples))

    for loper in range(0, n_of_samples):
       fas[loper] = df*loper

    fas[0] = 0.00001 # avoiding phase problem at k=0 in analytical solution

    #amplitudeverhouding en faseverhouding analytisch - analytical amplitude
    #ratio and phase difference
    r1 = np.sqrt(((x_ref-x_bron)*dx)**2+((y_ref-y_bron)*dy)**2)

    r2 = np.sqrt(((x_recorder-x_bron)*dx)**2+((y_recorder-y_bron)*dy)**2)

    aantalcellengepropageerd = np.sqrt((x_recorder-x_ref)**2+(y_recorder-y_ref)**2)


    k = 2*np.pi*fas/c

    Averhouding_theorie = np.abs(besselh(0,1,k*r1)/besselh(0,1,k*r2))

    Pverschil_theorie = np.unwrap(np.angle(1j*np.pi*besselh(0,1,k*r1)))-np.unwrap(np.angle(1j*np.pi*besselh(0,1,k*r2))) 


    #amplitudeverhouding en faseverhouding FDTD - amplitude ratio and phase
    #difference from FDTD
    print(recorder.shape)
    fftrecorder = fft(recorder.flatten(),n_of_samples)
    print(fftrecorder.shape)
    fftrecorder_ref = fft(recorder_ref.flatten(),n_of_samples)

    Averhouding_FDTD = np.abs(fftrecorder_ref/fftrecorder)

    Pverschil_FDTD = np.unwrap(np.angle(fftrecorder_ref))-np.unwrap(np.angle(fftrecorder)) 


    plt.subplots()
    plt.subplot(2,3,1)
    plt.plot(tijdreeks,recorder)
    plt.title('t recorder')
    plt.subplot(2,3,2)
    plt.plot(fas,np.abs(fftrecorder))
    plt.title('fft recorder abs')
    plt.xlim([0.5*fc,1.5*fc])

    plt.subplot(2,3,3)
    plt.plot(fas, np.unwrap(np.angle(fftrecorder)))
    plt.title('fft recorder phase')
    plt.xlim([0.5*fc, 1.5*fc])
    plt.ylim([-50, 0.0])

    plt.subplot(2,3,4)
    plt.plot(tijdreeks,recorder_ref)
    plt.title('t recorder ref')
    plt.subplot(2,3,5)
    plt.plot(fas,np.abs(fftrecorder_ref))
    plt.title('fft recorder ref abs')
    plt.xlim([0.5*fc, 1.5*fc])

    plt.subplot(2,3,6)
    plt.plot(fas,np.unwrap(np.angle(fftrecorder_ref)))
    plt.title('fft recorder phase')
    plt.xlim([0.5*fc, 1.5*fc])
    plt.ylim([-50, 0.0])


    #vergelijking analytisch-FDTD - comparison analytical versus FDTD
    lambdaoverdx = (c/fas)/dx

    Averhoudingrel = (Averhouding_FDTD/Averhouding_theorie.flatten())

    Averhouding = 1+((Averhoudingrel-1)/aantalcellengepropageerd)

    plt.show()
    plt.close()
    print('Averhouding_FDTD')
    print(Averhouding_FDTD.shape)
    print('Averhouding_theorie')
    print(Averhouding_theorie.shape)
    print('lambdaoverdx')
    print(lambdaoverdx.shape)
    print('Averhouding')
    print(Averhouding.shape)
    plt.subplots()
    plt.subplot(2,1,1)
    plt.plot(lambdaoverdx,Averhouding)
    
    plt.xlim([5, 20])
    plt.title('Amplitude ratio FDTD/analyt. per cel')
    plt.ylabel('ratio')
    plt.xlabel('number of cells per wavelength')
    plt.ylim([0.99, 1.01])

    #Pdifference = np.unwrap((Pverschil_FDTD+Pverschil_theorie)/aantalcellengepropageerd)
    Pdifference = (Pverschil_FDTD+Pverschil_theorie)/aantalcellengepropageerd

    plt.subplot(2,1,2)
    plt.plot(lambdaoverdx,Pdifference)
    plt.xlim([5, 20])
    plt.title('Phase difference FDTD - analyt. per cel')
    plt.ylabel('Difference')
    plt.xlabel('number of cells per wavelength')
    plt.ylim([-0.03, 0.03])
    plt.show()

