#these are the modules that I will be using
import numpy as np
import scipy as scp
import matplotlib as plt


#creating the matrices A and b
#remember that

x=np.linspace(0,50,1)
#trying with the scipy
LAMBDA=scp.special.hankel2(x)


plt.plot(LAMBDA,x, label="condition number", color="blue", linewidth=2)

# Adding labels and legend
plt.xlabel("lambda")
plt.ylabel("y")
plt.title("condition number")
plt.legend()
plt.grid(True)
plt.show()
#we see a peak