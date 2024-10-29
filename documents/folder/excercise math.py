import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


N= 100 #amount of quadrature points

## upperand under bounds integral
a=0
b=1
lmbda =1

## vraag 1 phi(x)= x + 7*lmbda/(28-8*lmbda)*sqrt(x)

## question 2: the kernel is not symmetric and since that QG-method 
# relies on symmetric positive matrices. we can't use this
## therefore we shall continue with gmres iterative method to solve the linear system
s,w= np.polynomial.legendre.leggauss(N)

sn,wn= (1/2+1/2*s),1/2*w

Kernel = np.outer(np.sqrt(sn),sn**2)
Weight= np.diag(wn)

A = np.identity(N)-lmbda*np.dot(Kernel,Weight)
print(np.shape(A))

u= sp.sparse.linalg.gmres(A,sn,maxiter=100)
y_numerical=u[0]
print(y_numerical)


y_analytical= sn + lmbda*7*np.sqrt(sn)/(28-8*lmbda)
x_values= sn

print(np.shape(y_numerical[0]))


def nystrom(Lambda, N):
    s,w= np.polynomial.legendre.leggauss(N)

    sn,wn= (1/2+1/2*s),1/2*w

    Kernel = np.outer(np.sqrt(sn),sn**2)
    Weight= np.diag(wn)

    A = np.identity(N)-Lambda*np.dot(Kernel,Weight)

    u= sp.sparse.linalg.gmres(A,sn,maxiter=100)
    y_numerical=u[0]
    x_values= sn
    return y_numerical ,x_values,A





## question 3

# Plotting the solutions
y_numerical, x_values,A=nystrom(1,N)

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_analytical, label="Analytical Solution", color="blue", linewidth=2)
plt.plot(x_values, y_numerical, 'o-', label="Numerical Solution (Euler)", color="red", markersize=4)

# Adding labels and legend
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison of Numerical Solution (nystrom Method) vs Analytical Solution")
plt.legend()
plt.grid(True)
plt.show()
## question 4:when the denominator becomes 0 that is when lmbfa
## becomes 3.5

## question 5: computing the condition number

invA=np.linalg.inv(A)
normA=np.linalg.norm(A, ord=2)
normIA= np.linalg.norm(invA,ord=2)


LAMBDA= np.arange(1, 4 + 0.05, 0.05)
cond=[]
for i in LAMBDA:
    y_numerical1,x_values1,A1=nystrom(i,N)
    cond=np.append(cond,np.linalg.cond(A1))
    


plt.plot(LAMBDA,cond, label="condition number", color="blue", linewidth=2)

# Adding labels and legend
plt.xlabel("lambda")
plt.ylabel("y")
plt.title("condition number")
plt.legend()
plt.grid(True)
plt.show()
#we see a peak around 3.5 which is also the eigenvalue we found in question 4 for which explodes

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