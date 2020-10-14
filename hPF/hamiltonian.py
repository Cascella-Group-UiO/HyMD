import numpy as np
import sympy


class Hamiltonian:
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
        k = sympy.var('k:%d' % (3))

        def fourier_space_window_function(k):
            return sympy.functions.elementary.exponential.exp(
                - 0.5 * self.config.sigma**2 * (k0**2 + k1**2 + k2**2)  # noqa: F821, E501
            )

        self.window_function_lambda = sympy.lambdify(
            [k], fourier_space_window_function(k)
        )

        def H(k, v):
            return v * self.window_function_lambda(k)

        self.H = H


class SquaredPhi(Hamiltonian):
    def __init__(self, config):
        super(SquaredPhi, self).__init__(config)
        super(SquaredPhi, self)._setup()
        self.setup()

    def setup(self):
        def w(phi, kappa=self.config.kappa, rho0=self.config.rho0):
            return 0.5 / (kappa * rho0) * (sum(phi))**2

        self.v_ext = [
            sympy.lambdify([self.phi], sympy.diff(w(self.phi), 'phi%d' % (i)))
            for i in range(len(self.config.unique_names))
        ]
        self.w = sympy.lambdify([self.phi], w(self.phi))


class DefaultNoChi(Hamiltonian):
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


class DefaultWithChi(Hamiltonian):
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

        self.v_ext = [
            sympy.lambdify([self.phi], sympy.diff(w(self.phi), 'phi%d' % (i)))
            for i in range(len(self.config.unique_names))
        ]
        self.w = sympy.lambdify([self.phi], w(self.phi))
