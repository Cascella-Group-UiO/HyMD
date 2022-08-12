"""Interaction energy functionals.

Implements field--particle interaction energy functionals depending on the
field configuration. Extending HyMD with a new Hamiltonian is done by
subclassing the :code:`Hamiltonian` superclass with logic intialized in the
constructor.

The grid-independent filtering function is defined in the :code:`Hamiltonian`
superclass constructor. Hijack :code:`self.H` in a subclass to change the
filter.
"""
import numpy as np
import sympy


class Hamiltonian:
    """Interaction energy functional superclass
    """
    def __init__(self, config):
        """Constructor

        Parameters
        ----------
        config : Config
            Configuration object.

        See also
        --------
        hymd.input_parser.Config : Configuration dataclass handler.
        """
        self.config = config
        self._setup()

    def _setup(self):
        """Superclass setup

        Sets up the grid-independent filtering function :code:`H`, and the
        SymPy logic for symbolically differentiating interaction energy
        functionals in Hamiltonian subclasses.
        """
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
    """Simple squared density interaction energy functional

    The interaction energy density takes the form

    .. math::

        w[\\tilde\\phi] = \\frac{1}{2\\kappa\\rho_0}
            \\left(
                \\sum_k \\tilde\\phi_k
            \\right)^2,

    where :math:`\\kappa` is the compressibility and :math:`rho_0` is the
    average density of the fully homogenous system. Expressing the species
    densities in terms of fluctuations from the average,

    .. math::

        \\mathrm{d}\\tilde\\phi_k = \\rho_0 - \\tilde\\phi_k,

    it is evident that this interaction energy functional is a slight change of
    the :code:`DefaultNoChi` Hamiltonian, as the expanded interaction energy
    density becomes

    .. math::

        w[\\tilde\\phi] = \\frac{1}{2\\kappa\\rho_0}
            \\left(
                6\\rho_0\\left[
                    \\sum_k \\mathrm{d}\\tilde\\phi_k
                \\right]
                +
                9\\rho_0^2
                +
                \\sum_k \\mathrm{d}\\tilde\\phi_k^2
                +
                \\prod_{k\\not=l}
                    \\mathrm{d}\\tilde\\phi_k \\mathrm{d}\\tilde\\phi_l
            \\right),

    identical to :code:`DefaultNoChi` apart from a constant energy shift
    :math:`9\\rho_0^2`(which does not impact dynamics) and the
    :math:`\\rho_0`--:math:`\\mathrm{d}\\tilde\\phi_k` cross-terms. These
    cross-terms constitute contributions to the energy linear in
    :math:`\\mathrm{d}\\tilde\\phi_k` absent in :code:`DefaultNoChi`, which has
    only quadratic terms present.

    See also
    --------
    hymd.hamiltonian.DefaultNoChi
    """
    def __init__(self, config):
        """Constructor

        Parameters
        ----------
        config : Config
            Configuration object.

        See also
        --------
        hymd.input_parser.Config : Configuration dataclass handler.
        """
        super(SquaredPhi, self).__init__(config)
        super(SquaredPhi, self)._setup()
        self.setup()

    def setup(self):
        """Setup the interaction energy potential and the external potential
        """
        def w(phi, kappa=self.config.kappa, rho0=self.config.rho0):
            return 0.5 / (kappa * rho0) * (sum(phi)) ** 2

        self.v_ext = [
            sympy.lambdify([self.phi], sympy.diff(w(self.phi), self.phi[i]))
            for i in range(len(self.config.unique_names))
        ]
        self.w = sympy.lambdify([self.phi], w(self.phi))


class DefaultNoChi(Hamiltonian):
    """Incompressibility-only interaction energy functional

    The interaction energy density takes the form

    .. math::

        w[\\tilde\\phi] = \\frac{1}{2\\kappa} \\left(
            \\sum_k \\tilde\\phi_k - \\rho_0
        \\right)^2,

    where :math:`\\kappa` is the compressibility and :math:`\\rho_0` is the
    average density of the fully homogenous system. The :code:`SquaredPhi`
    Hamiltonian implements a similar functional with an additional linear term
    component depending on

    .. math::

        \\mathmr{d}\\tilde\\phi_k = \\tilde\\phi_k - \\rho_0

    and not :math:`\\mathrm{d}\\tilde\\phi_k^2`. No explicit inter-species
    interaction is present apart from the indirect interaction through the
    incompressibility.

    See also
    --------
    hymd.hamiltonian.DefaultNoChi
    hymd.input_parser.Config :
        Configuration dataclass handler.
    """
    def __init__(self, config):
        """Constructor

        Parameters
        ----------
        config : Config
            Configuration object.

        See also
        --------
        hymd.input_parser.Config : Configuration dataclass handler.
        """
        super(DefaultNoChi, self).__init__(config)
        super(DefaultNoChi, self)._setup()
        self.setup()

    def setup(self):
        """Setup the interaction energy potential and the external potential
        """
        def w(phi, kappa=self.config.kappa, rho0=self.config.rho0):
            return 0.5 / (kappa * rho0) * (sum(phi) - rho0) ** 2

        self.v_ext = [
            sympy.lambdify([self.phi], sympy.diff(w(self.phi), self.phi[i]))
            for i in range(len(self.config.unique_names))
        ]
        self.w = sympy.lambdify([self.phi], w(self.phi))


class DefaultWithChi(Hamiltonian):
    """Incompressibility and :math:`\\chi`-interactions energy functional

    The interaction energy density takes the form

    .. math::

        w[\\tilde\\phi] =
            \\frac{1}{2\\rho_0}
                \\sum_{k,l}\\chi_{kl} \\tilde\\phi_k \\tilde\\phi_l
            +
            \\frac{1}{2\\kappa} \\left(
                \\sum_k \\tilde\\phi_k - \\rho_0
            \\right)^2,

    where :math:`\\kappa` is the incompressibility, :math:`\\rho_0` is the
    average density of the fully homogenous system and :math:`\\chi_{ij}` is
    the Flory-Huggins-like inter-species mixing energy.
    """
    def __init__(self, config, unique_names, type_to_name_map):
        """Constructor

        Parameters
        ----------
        config : Config
            Configuration object.
        unique_names : numpy.ndarray
            Sorted array of all distinct names of different species present in
            the simulation. Result of :code:`numpy.unique(all_names)`, where
            :code:`all_names` is the gathered union of all individual MPI
            ranks' :code:`names` arrays.
        type_to_name_map : dict[int, str]
            Dictionary of the mapping from type indices (integers) to type
            names.

        See also
        --------
        hymd.input_parser.Config : Configuration dataclass handler.
        """
        super(DefaultWithChi, self).__init__(config)
        super(DefaultWithChi, self)._setup()
        self.setup(unique_names, type_to_name_map)

    def setup(self, unique_names, type_to_name_map):
        """Setup the interaction energy potential and the external potential

        Parameters
        ----------
        unique_names : numpy.ndarray
            Sorted array of all distinct names of different species present in
            the simulation. Result of :code:`numpy.unique(all_names)`, where
            :code:`all_names` is the gathered union of all individual MPI
            ranks' :code:`names` arrays.
        type_to_name_map : dict[int, str]
            Dictionary of the mapping from type indices (integers) to type
            names.
        """
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
            sympy.lambdify([self.phi], sympy.diff(w(self.phi), self.phi[i]))
            for i in range(len(self.config.unique_names))
        ]
        self.w = sympy.lambdify([self.phi], w(self.phi))


def get_hamiltonian(config):
    """Return appropriate Hamiltonian object based on the 
    config.hamiltonian string.

    Parameters
    ----------
    config : Config
        Configuration object.

    Returns
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object.
    """
    if config.hamiltonian.lower() == "defaultnochi":
        hamiltonian = DefaultNoChi(config)
    elif config.hamiltonian.lower() == "defaultwithchi":
        hamiltonian = DefaultWithChi(
            config, config.unique_names, config.type_to_name_map
        )
    elif config.hamiltonian.lower() == "squaredphi":
        hamiltonian = SquaredPhi(config)

    return hamiltonian