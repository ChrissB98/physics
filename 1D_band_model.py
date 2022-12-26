import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import linalg as LA
import time

#Defining constants
hbar = (1/(2*np.pi))*6.62607015*10**(-34)
m = 9.1093837015*10**(-31)
E_r = ((hbar**2) / (2*m)) * (2*np.pi)**2
G = 4*np.pi

def kronecker_delta(n_prime,n):
    if(n_prime==n):
        return 1
    else:
        return 0

def M(n_prime,n,s,q): #Matrix element
    return ( hbar**2 * (q+n*G)**2)/(2*m) * kronecker_delta(n_prime, n) + (E_r*s/2)*(kronecker_delta(n_prime, n)-kronecker_delta(n_prime, n+1)/2 -kronecker_delta(n_prime+1, n)/2)

def hamiltonian(s,q):
    values = [M(n1,n2,s,q) for n1 in range(-5,6,1) for n2 in range(-5,6,1)]
    matrix = np.array(values).reshape((11,11))
    return matrix

def eigenvalues(s,q):
    return np.sort(LA.eigvals(hamiltonian(s,q)))


q = np.linspace(-G,G,1000)
bands = np.array([])
s_vals = range(1,101,1)
start = time.time()

#Computing energy eigenvalues for five lowest bands & saving in 'bands' array
for s in s_vals:
    print("Computing s="+str(s))
    bands_save = [[],[],[],[],[]]
    for qs in q:
        val = eigenvalues(s, qs)/E_r
        for index in range(0,5,1):
            bands_save[index].append(val[index])
    bands = np.append(bands,bands_save)
print("Simulation time: {duration:.2f} seconds".format(duration = time.time()-start))

#Animation
fig, ax = plt.subplots()
ax.set(xlabel='q',ylabel='E/E_r')
ax.set_xlim(-G,G)
ax.set_ylim(0,np.max(bands)+1)
line0, = ax.plot(0,0)
line1, = ax.plot(0,0)
line2, = ax.plot(0,0)
line3, = ax.plot(0,0)
line4, = ax.plot(0,0)

def animation_frame(i):
    line0.set_data(q,bands[i*5000+0*1000:i*5000+0*1000+1000])
    line1.set_data(q,bands[i*5000+1*1000:i*5000+1*1000+1000])
    line2.set_data(q,bands[i*5000+2*1000:i*5000+2*1000+1000])
    line3.set_data(q,bands[i*5000+3*1000:i*5000+3*1000+1000])
    line4.set_data(q,bands[i*5000+4*1000:i*5000+4*1000+1000])
    ax.set_title("1D Band Structure s="+str(i+1))

animation = FuncAnimation(fig,func=animation_frame,frames=np.arange(0,len(s_vals),1),interval=10)
plt.show()