import numpy as np
import sympy


class Hamiltonian:
    def __init__(self, config):
        self.config = config
        self._setup()

    def _setup(self):
        if not hasattr(self.config, "simulation_volume"):
            self.config.simulation_volume = np.prod(np.asarray(self.config.box_size))
        self.config.rho0 = self.config.n_particles / self.config.simulation_volume
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

        def V_bar(
                phi,
                k,
                kappa=self.config.kappa,
                rho0=self.config.rho0,
        ):
            V_incompressibility = 1/(kappa*rho0)*sum(phi)
            V_interaction = 0
            return (V_interaction,V_incompressibility)

        self.V_bar = [
            sympy.lambdify([self.phi], V_bar(self.phi, k))
            for k in range(len(self.config.unique_names))
        ]

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
        def w(
                phi,
                kappa=self.config.kappa,
                rho0=self.config.rho0,
                a=self.config.a
        ):
            return 0.5 / (kappa * rho0) * (sum(phi) - a) ** 2

        def V_bar(
                phi,
                k,
                kappa=self.config.kappa,
                rho0=self.config.rho0,
                a=self.config.a,
        ):
            V_incompressibility = 1/(kappa*rho0)*(sum(phi) - a)
            V_interaction = 0
            return (V_interaction,V_incompressibility)

        self.V_bar = [
            sympy.lambdify([self.phi], V_bar(self.phi, k))
            for k in range(len(self.config.unique_names))
        ]
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
        self.K_coupl_type_dictionary = {
            tuple(sorted([c.atom_1, c.atom_2])): c.squaregradient_energy
            for c in self.config.K_coupl
        }
        self.phi_laplacian = [ sympy.var("phi_laplacian%d(0:%d)" %(t,3)) for t in range(len(self.config.unique_names)) ]

        def w(
            phi,
            kappa=self.config.kappa,
            rho0=self.config.rho0,
            a=self.config.a,
            chi=self.config.chi,
            type_to_name_map=self.type_to_name_map,
            chi_type_dictionary=self.chi_type_dictionary,
        ):
            interaction = 0
            for i in range(self.config.n_types):
                for j in range(i + 1, self.config.n_types):

                # use line below instead to count diagonal chi:
                #for j in range(i, self.config.n_types):

                    ni = type_to_name_map[i]
                    nj = type_to_name_map[j]
                    names = sorted([ni, nj])
                    c = chi_type_dictionary[tuple(names)]
                    interaction += c * phi[i] * phi[j] / rho0
            incompressibility = 0.5 / (kappa * rho0) * (sum(phi) - a) ** 2
            return incompressibility + interaction

        def V_bar(
                phi,
                k,
                kappa=self.config.kappa,
                rho0=self.config.rho0,
                a=self.config.a,
                chi=self.config.chi,
                type_to_name_map=self.type_to_name_map,
                chi_type_dictionary=self.chi_type_dictionary,
        ):
            V_incompressibility = 1/(kappa*rho0)*(sum(phi) - a)

            V_interaction = 0
            nk = type_to_name_map[k]
            for i in range(self.config.n_types):
                ni = type_to_name_map[i]
                names = sorted([nk, ni])
                if ni!=nk:
                    c = chi_type_dictionary[tuple(names)] 
                else:
                    c = 0
                #uncomment to count diagonal chi terms:
                #c = chi_type_dictionary[tuple(names)]
                V_interaction += c * phi[i] / rho0
            return (V_interaction,V_incompressibility)

        self.V_bar = [
            sympy.lambdify([self.phi], V_bar(self.phi, k))
            for k in range(len(self.config.unique_names))
        ]

        self.v_ext = [
            sympy.lambdify([self.phi], sympy.diff(w(self.phi), "phi%d" % (i)))
            for i in range(len(self.config.unique_names))
        ]
        self.w = sympy.lambdify([self.phi], w(self.phi))

    def w1(
        self,
        phi_gradient,
    ):
        rho0=self.config.rho0
        type_to_name_map=self.type_to_name_map
        K_coupl_type_dictionary = self.K_coupl_type_dictionary
        squaregradient = 0.0
        for t1 in range(self.config.n_types):
            for t2 in range(t1, self.config.n_types):
                nt1 = type_to_name_map[t1]
                nt2 = type_to_name_map[t2]
                names = sorted([nt1, nt2])
                if nt1!=nt2:
                    c = K_coupl_type_dictionary[tuple(names)]
                else:
                    c = 0
                for d in range(3):
                    squaregradient += c * phi_gradient[t1][d] * phi_gradient[t2][d] / rho0 
        return squaregradient

    def v_ext1(
            self,
            phi_lap_filtered,
            v_ext1
    ):
        rho0 = self.config.rho0
        type_to_name_map=self.type_to_name_map
        K_coupl_type_dictionary = self.K_coupl_type_dictionary
        for k in range(self.config.n_types):
            for l in range(self.config.n_types):
                nk = type_to_name_map[k]
                nl = type_to_name_map[l]
                names = sorted([nk, nl])
                if nk!=nl:
                    c = K_coupl_type_dictionary[tuple(names)] 
                else:
                    c = 0
                #uncomment to include diagonal K_coupl terms:
                #c = K_coupl_type_dictionary[tuple(names)]
                v_ext1[k] += -1 * c / rho0 * phi_lap_filtered[l]

