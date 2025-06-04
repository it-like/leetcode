import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
L = np.pi
N = 2048
dx = 0.001
x = L * np.arange(-1 + dx, 1 + dx, dx)
n = len(x)
nquart = int(np.floor(N/4))

# hat fct
#f = np.zeros_like(x)
#f[nquart: 2 * nquart] =(4/N)*np.arange(1, nquart + 1)
#f[2*nquart: 3 * nquart] =np.ones(nquart) - (4/N)*np.arange(0,nquart)


# block fct
f = np.zeros_like(x)
f[nquart: 3 * nquart] = 1

fig, ax = plt.subplots()
ax.plot(x,f,'_', color = 'k')

#plt.plot()
#plt.show()


cmap = get_cmap('tab10')
colors = cmap.colors
ax.set_prop_cycle(color=colors)



A0 = np.sum(f * np.ones_like(x)) * dx
fFS = A0/2

depth = 300
A = np.zeros(depth)
B = np.zeros(depth)
name = 'Fourier_gibbs'

for k in range(depth):
    A[k] = np.sum(f *np.cos(np.pi * (k+1)* x/L)) * dx
    B[k] = np.sum(f *np.sin(np.pi * (k+1)* x/L)) * dx
    fFS = fFS  +  A[k]*np.cos(np.pi * (k+1)* x/L) + B[k] * np.sin(np.pi * (k+1)* x/L)
    if k == depth -1 :
        ax.plot(x,fFS, '-')



#path = 'fourier/images/'
#plt.savefig(path + name)
plt.show()
    

