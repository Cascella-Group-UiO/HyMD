import numpy as np
import sympy


class Hamiltonian:
    def __init__(self, config):
        self.config = config
        self._setup()

    def _setup(self):
        if not hasattr(self.config, "simulation_volume"):
            self.config.simulation_volume = np.prod(np.asarray(self.config.box_size))
        
        if self.config.vitual_charge_types:
            self.config.rho0 = (self.config.n_particles - self.config.vitual_charge_types_num) / self.config.simulation_volume # xinmeng 
        else:
            self.config.rho0 = self.config.n_particles / self.config.simulation_volume
        
        if self.config.preset_rho: 
            self.config.rho0 = self.config.preset_rho
        
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
    """
    2021-07-29 with self-interaction, if does not work, then use the 
    old BACK_DefaultWithChi
    only modification is:

    for j in range(i + 1, self.config.n_types):
    ===> 
    for j in range(i    , self.config.n_types):


    """
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
            ### _len = len(self.config.kai_types_id) # 2021-12-01 old protocol with kai_types_id is not correct.. 
            for i in range(self.config.n_types):
                for j in range(i + 1, self.config.n_types): ## for j in range(i , self.config.n_types): 
                    ni = type_to_name_map[i]
                    nj = type_to_name_map[j]
                    names = sorted([ni, nj])
                    c = chi_type_dictionary[tuple(names)]
                    
                    if i in self.config.kai_types_id and j in self.config.kai_types_id:  ### only consider the kai when types are not vitual type 
                        #if i in self.config.kai_types_id and j in self.config.kai_types_id and not (i in self.config.freez_ids and j in self.config.freez_ids) :  ### only consider the kai when types are not vitual type; also exclude freez self-interaction 
                        interaction += c * phi[i] * phi[j] / rho0
                        #print(i, j, ni, nj, c, self.config.kai_types_id)
            #print(sum)
            #print(sum([phi[index] for index in self.config.kai_types_id]))
            
            ### this way some how give error 
            ### File "/Users/lixinmeng/Desktop/working/md-scf/HyMD-2021/hymd/field.py", line 242, in update_field
            ### hamiltonian.v_ext[t](phi).r2c(out=v_ext_fourier[0])
            ### AttributeError: 'int' object has no attribute 'r2c'
            #if self.config.freez_ids: 
            #    _keep_list = [item for item in self.config.kai_types_id if item not in self.config.freez_ids ]  
            #    print(_keep_list)
            #    incompressibility = 0.5 / (kappa * rho0) * (sum( [phi[index] for index in _keep_list]) - rho0) ** 2 
            #else:
            #    incompressibility = 0.5 / (kappa * rho0) * (sum( [phi[index] for index in self.config.kai_types_id]) - rho0) ** 2 
            #    print(incompressibility)
            
            ####  Now just do not exlude the freez self-interaction...
            incompressibility = 0.5 / (kappa * rho0) * (sum( [phi[index] for index in self.config.kai_types_id]) - rho0) ** 2 
            
            return incompressibility + interaction
        
        self.v_ext = [
            sympy.lambdify([self.phi], sympy.diff(w(self.phi), "phi%d" % (i)))
            for i in range(len(self.config.unique_names))
        ]

        self.w = sympy.lambdify([self.phi], w(self.phi))




class BACK_DefaultWithChi(Hamiltonian):
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
