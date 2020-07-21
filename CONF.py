import numpy
import sympy

Np     = 10000      # Number of particles
#tau    = 0.7       # ps⁻1
dt = 0.001
NSTEPS = 1000
#T0      = 300         # K 
Nv = 80
sigma = 0.5
nprint = 1000  # Printing frequency
mass   = 72   # g/mol
kappa  = 0.05 # kj/mol
L      = [10.61,10.61,10.61] # nm
quasi  = 1       # Number of steps between updates
press_print = 1
#read_start = '300.dat'
types=1
rho0 = Np/(L[0]*L[1]*L[2])
dV=L[0]*L[1]*L[2]/Nv




# hPF potential defined
phi=sympy.var('phi:%d'%(types))

#Potential
def w(phi):
    return 0.5/(kappa*rho0)*(sum(phi)-rho0)**2


V_EXT = [sympy.lambdify([phi], sympy.diff(w(phi),'phi%d'%(i))) for i in range(types)]
w     = sympy.lambdify([phi], w(phi))

 
# Filter

#H(k,v) = v * exp(-0.5*sigma**2*k.normp(p=2))
def H(k, v):
    return v * numpy.exp(-0.5*sigma**2*k.normp(p=2))







