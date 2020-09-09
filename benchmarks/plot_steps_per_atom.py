import matplotlib.pylab as plt
import matplotlib
import numpy as np
import sys
matplotlib.style.use('default')
from matplotlib import rc
plt.style.use('../../fig.mplstyle')
plt.rc('font', family='sans-serif')
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12 
rc('font', size=SMALL_SIZE)          # controls default text sizes
rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
lcs=['k','r','g','b','y','m']



size=(3,2)
plt.figure(1,figsize=size)

N     = int(float(sys.argv[1]))
STEPS = int(sys.argv[2])
data=np.loadtxt('benchmark_time.dat')
ind=np.argsort(data[:,0])
data=data[ind,:]
plt.plot(data[:,0],STEPS/N/data[:,1],'k')
plt.ylabel('step/(atom*time)')

plt.savefig('steps_per_atom_per_time.png')

plt.xlabel('# cpu')
plt.tight_layout()

plt.figure(2,figsize=size)

plt.plot(data[:,0],data[0,1]*data[0,0]/data[:,1],'k')

plt.plot(data[:,0],data[:,0],'r')
plt.ylabel('speedup')
plt.xlabel('# cpu')
plt.tight_layout()
plt.savefig('speedup.png')



