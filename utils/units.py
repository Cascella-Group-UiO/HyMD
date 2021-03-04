# CONVERSIONS FACTOR FOR UNITLESS RESULTS
# a'= a/a0, where a and a' are the quantity with and without units.

mass = 72
phi0 = 10000 / 10.61 ** 3
kappa = 0.05
kappainv = 0.05 ** (-1)
kb = 2.479 / 298.0

l0 = phi0 ** (-1.0 / 3)
E0 = kappainv
F0 = kappainv * l0
m0 = mass
v0 = l0 ** -1 * (kappainv / mass) ** (0.5)
t0 = (kappainv / mass) ** (0.5)
T0 = kappainv / kb
D0 = l0 ** 2 / t0 * 1e5 / 100
