import numpy as np
## we subtract and add back 1/x since its principle values can be solved analytically
## the principle value of 1/x becomes -ln(2)
#our function becomes f(x)-1/x which we will ue laguerre for
N=20
s,w= np.polynomial.legendre.leggauss(N//3)
q,p= np.polynomial.legendre.leggauss(N-N//3)
sn,wn= (-1+1*s),w
sp, wp= (1/2+1/2*q),1/2*p

def int(x):
    intx= (1+np.abs(x))/np.tan(np.exp(x)-1)-1/x
    return intx

ans=np.sum(wp*int(sp))+np.sum(wn*int(sn)) -np.log(2)
print(ans)