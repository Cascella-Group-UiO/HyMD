"""Class to wrap PLUMED for use within HyMD.
"""

import numpy as np
from mpi4py import MPI
from .logger import Logger

try:
    import plumed
    has_plumed = True
except ImportError:
    has_plumed = False

class PlumedBias:
    """Deals with PLUMED interface
    """
    def __init__(self, config, plumeddat, logfile, comm=MPI.COMM_WORLD):
        """Constructor

        Parameters
        ----------
        config : Config
            Configuration object.
        plumeddat : str
            Path file containing PLUMED input.
        logfile : str
            Path to PLUMED's output file
        comm : mpi4py.Comm
            MPI communicator to use for rank communication.

        See also
        --------
        hymd.input_parser.Config : Configuration dataclass handler.
        """
        if not has_plumed:
            err_str = (
                "You are trying to use PLUMED "
                "but HyMD could not import py-plumed."
            )
            Logger.rank0.log(logging.ERROR, err_str)
            if comm.Get_rank() == 0:
                raise ImportError(err_str)

        self.plumed_obj = plumed.Plumed()
        self.plumed_version = np.zeros(1, dtype=np.intc)
        self.plumed_forces = np.zeros((0,3), dtype=np.double)
        self.plumed_bias = np.zeros(1, np.double)
        self.comm = comm
        self.ready = False

        self.plumed_obj.cmd("getApiVersion", self.plumed_version)
        if self.plumed_version[0] <= 3:
            err_str = (
                "HyMD requires a PLUMED API > 3. "
                "Use a newer PLUMED kernel."
            )
            Logger.rank0.log(logging.ERROR, err_str)
            if comm.Get_rank() == 0:
                raise AssertionError(err_str)

        self.plumed_obj.cmd("setMDEngine", "HyMD")

        self.plumed_obj.cmd("setMPIComm", comm)
        self.plumed_obj.cmd("setPlumedDat", plumeddat)
        self.plumed_obj.cmd("setLogFile", logfile)
        self.plumed_obj.cmd("setTimestep", 
            config.respa_inner * config.time_step
        )
        self.plumed_obj.cmd("setNatoms", config.n_particles)
        self.plumed_obj.cmd("setKbT", 
            config.gas_constant * config.target_temperature
        )
        self.plumed_obj.cmd("setNoVirial")
        self.plumed_obj.cmd("init")


    def prepare(self, step, forces, positions, indices, config, charges):
        """Setup all the PLUMED pointers and returns if PLUMED will use
        the potential energy as well.
        """
        self.plumed_forces = forces.astype(np.double)

        needs_energy = np.zeros(1, np.intc)
        # plumed_virial = np.zeros((3,3), dtype=np.double)
        masses = np.full_like(indices, config.mass, dtype=np.double)
        box = np.diag(config.box_size)

        self.plumed_obj.cmd("setAtomsNlocal", indices.shape[0]);
        self.plumed_obj.cmd("setAtomsGatindex", indices);
        self.plumed_obj.cmd("setStep", step)
        self.plumed_obj.cmd("setForces", self.plumed_forces)
        # no need to worry about Fortran order because PLUMED wrapper ravels
        self.plumed_obj.cmd("setPositions", positions)
        self.plumed_obj.cmd("setCharges", charges)
        self.plumed_obj.cmd("setMasses", masses)
        self.plumed_obj.cmd("setBox", box)
        # plumed_obj.cmd("setVirial", plumed_virial)

        # check if PLUMED needs energy and returns
        self.plumed_obj.cmd("prepareCalc")
        self.plumed_obj.cmd("isEnergyNeeded", needs_energy)

        self.ready = True
        return True if needs_energy[0] != 0 else False


    def calc(self, forces, poteng):
        """After setting up with prepare, set the potential energy
        and runs performCalc to get the bias energy and forces.
        """
        if not self.ready:
            err_str = (
                "Trying to calculate PLUMED biased forces "
                "without first calling prepare method."
            )
            Logger.rank0.log(logging.ERROR, err_str)
            if self.comm.Get_rank() == 0:
                raise RuntimeError(err_str)      

        # set the energy and calc
        self.plumed_obj.cmd("setEnergy", poteng)
        self.plumed_obj.cmd("performCalc")
        self.plumed_obj.cmd("getBias", self.plumed_bias)

        # subtract forces to get only the bias' extra force
        self.plumed_forces -= forces

        self.ready = False
        return self.plumed_forces, self.plumed_bias[0]

