import numpy as np
import sympy


class Hamiltonian:
    def __init__(self, config):
        self.config = config
        self._setup()

    def _setup(self):
        if not hasattr(self.config, "simulation_volume"):
            self.config.simulation_volume = np.prod(
                np.asarray(self.config.box_size)
            )
        self.config.rho0 = (
            self.config.n_particles / self.config.simulation_volume
        )
        self.phi = sympy.var("phi:%d" % (len(self.config.unique_names)))
        k = sympy.var("k:%d" % (3))

        def fourier_space_window_function(k):
            return sympy.functions.elementary.exponential.exp(
                -0.5
                * self.config.sigma ** 2
                * (k0 ** 2 + k1 ** 2 + k2 ** 2)  # noqa: F821, E501
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
            return 0.5 / (kappa * rho0) * (sum(phi)) ** 2

        self.v_ext = [
            sympy.lambdify([self.phi], sympy.diff(w(self.phi), "phi%d" % (i)))
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
            return 0.5 / (kappa * rho0) * (sum(phi) - rho0) ** 2

        self.v_ext = [
            sympy.lambdify([self.phi], sympy.diff(w(self.phi), "phi%d" % (i)))
            for i in range(len(self.config.unique_names))
        ]
        self.w = sympy.lambdify([self.phi], w(self.phi))


class DefaultWithChi(Hamiltonian):
    def __init__(self, config, unique_names, type_to_name_map):
        super(DefaultWithChi, self).__init__(config)
        super(DefaultWithChi, self)._setup()
        self.setup(unique_names, type_to_name_map)

    def setup(self, unique_names, type_to_name_map):
        self.type_to_name_map = type_to_name_map
        self.chi_type_dictionary = {
            tuple(sorted([c.atom_1, c.atom_2])): c.interaction_energy
            for c in self.config.chi
        }

        def w(
            phi,
            kappa=self.config.kappa,
            rho0=self.config.rho0,
            chi=self.config.chi,
            type_to_name_map=self.type_to_name_map,
            chi_type_dictionary=self.chi_type_dictionary,
        ):

            interaction = 0
            for i in range(self.config.n_types):
                for j in range(i + 1, self.config.n_types):
                    ni = type_to_name_map[i]
                    nj = type_to_name_map[j]
                    names = sorted([ni, nj])
                    c = chi_type_dictionary[tuple(names)]

                    interaction += c * phi[i] * phi[j] / rho0
            incompressibility = 0.5 / (kappa * rho0) * (sum(phi) - rho0) ** 2
            return incompressibility + interaction

        self.v_ext = [
            sympy.lambdify([self.phi], sympy.diff(w(self.phi), "phi%d" % (i)))
            for i in range(len(self.config.unique_names))
        ]
        self.w = sympy.lambdify([self.phi], w(self.phi))
