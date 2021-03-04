import numpy as np
import sympy

Np = 10000  # Number of particles
tau = 0.7  # ps⁻1
dt = 0.0005
NSTEPS = 100
T0 = 300  # K
Nv = 100
sigma = 0.5
nprint = 10  # Printing frequency
mass = 72  # g/mol
kappa = 0.05  # kj/mol
L = [10.61, 10.61, 10.61]  # nm
quasi = 1  # Number of steps between updates
press_print = 1
# read_start = '300.dat'
types = 1
ntypes = 1
rho0 = Np / (L[0] * L[1] * L[2])
dV = L[0] * L[1] * L[2] / Nv
NAMES = np.array([np.string_(a) for a in ["A", "B"]])
domain_decomp = True
# uniform_start=True
_start = 300
# hPF potential defined
phi = sympy.var("phi:%d" % (types))

# Potential
def w(phi):
    return 0.5 / (kappa * rho0) * (sum(phi) - rho0) ** 2


V_EXT = [sympy.lambdify([phi], sympy.diff(w(phi), "phi%d" % (i))) for i in range(types)]
w = sympy.lambdify([phi], w(phi))


# Filter

# H(k,v) = v * exp(-0.5*sigma**2*k.normp(p=2))

k = sympy.var("k:%d" % (3))


def H1(k):
    return sympy.functions.elementary.exponential.exp(
        -0.5 * sigma ** 2 * (k0 ** 2 + k1 ** 2 + k2 ** 2)
    )


kdHdk = [
    k0 * sympy.diff(H1(k), "k0"),
    k1 * sympy.diff(H1(k), "k1"),
    k2 * sympy.diff(H1(k), "k2"),
]

kdHdk = [sympy.lambdify([k], kdHdk[i]) for i in range(3)]

H1 = sympy.lambdify([k], H1(k))


def H(k, v):
    return v * H1(k)  # numpy.exp(-0.5*sigma**2*k.normp(p=2))


# def H(k,v):

#     return v * numpy.exp(-0.5*sigma**2*k.normp(p=2))
