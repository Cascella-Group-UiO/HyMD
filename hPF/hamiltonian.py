import numpy as np
import sympy


class W:
    def __init__(self, config):
        self.config = config
        self._setup()

    def _setup(self):
        if not hasattr(self.config, 'simulation_volume'):
            self.config.simulation_volume = np.prod(
                np.asarray(self.config.box_size)
            )
        self.config.rho0 = (
            self.config.n_particles / self.config.simulation_volume
        )
        self.phi = sympy.var('phi:%d' % (len(self.config.unique_names)))


class DefaultNoChi(W):
    def __init__(self, config):
        super(DefaultNoChi, self).__init__(config)
        super(DefaultNoChi, self)._setup()
        self.setup()

    def setup(self):
        def w(phi, kappa=self.config.kappa, rho0=self.config.rho0):
            return 0.5 / (kappa * rho0) * (sum(phi) - rho0)**2

        self.v_ext = [
            sympy.lambdify([self.phi], sympy.diff(w(self.phi), 'phi%d' % (i)))
            for i in range(len(self.config.unique_names))
        ]
        self.w = sympy.lambdify([self.phi], w(self.phi))


class DefaultWithChi(W):
    def __init__(self, config):
        super(DefaultNoChi, self).__init__(config)
        super(DefaultNoChi, self)._setup()
        self.setup()

    def setup(self):
        def w(phi, kappa=self.config.kappa, rho0=self.config.rho0,
              chi=self.config.chi):
            triu_ind = np.triu_indices(types, k=1)
            interaction = 0.0
            for i, j in zip(triu_ind[0], triu_ind[1]):
                interaction += chi[i, j] * phi[i] * phi[j]
            interaction *= 0.5 / rho0
            incompressibility = 0.5 / (kappa * rho0) * (sum(phi) - rho0)**2
            return incompressibility + interaction

        def w(phi, kappa=self.config.kappa, rho0=self.config.rho0):
            return (0.5 / (kappa * rho0) * (sum(phi) - rho0)**2 +
                    0.5 / rho0 * (chi * phi0 * phi1))

        self.v_ext = [
            sympy.lambdify([self.phi], sympy.diff(w(self.phi), 'phi%d' % (i)))
            for i in range(len(self.config.unique_names))
        ]
        self.w = sympy.lambdify([self.phi], w(self.phi))
