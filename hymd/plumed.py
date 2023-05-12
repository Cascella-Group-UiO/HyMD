"""Class to wrap PLUMED for use within HyMD.
"""

import numpy as np
import logging
import os
from mpi4py import MPI
from .logger import Logger


class PlumedBias:
    """PLUMED handler class

    Notes
    -----
    This wraps the :code:`Plumed()` class, see
    `https://github.com/plumed/plumed2/tree/master/python`.

    The :code:`PlumedBias()` object is created with the arguments from
    :code:`__init__` and at everystep the methods :code:`prepare()` and
    :code:`calc()` should be called.

    Attributes
    ----------
    plumed_obj : plumed.Plumed
        Plumed object used to pass and request information from PLUMED.
    plumed_forces : (N, D) numpy.ndarray
        Array of forces of :code:`N` particles in :code:`D` dimensions.
        After :code:`calc()`, this array contains only the forces due
        to the PLUMED bias.
    positions : (N, D) numpy.ndarray
        Array of positions for :code:`N` particles in :code:`D` dimensions.
        Local for each MPI rank.
        Needed because we need C-contiguous array for passing to PLUMED.
    plumed_bias : (1,) numpy.ndarray
        Used as a pointer to an :code:`double` to store the bias energy.
    plumed_version : (1,) numpy.ndarray
        Used as a pointer to an :code:`int` to store PLUMED API version.
    comm : mpi4py.Comm
        MPI communicator to use for rank communication.
    ready : bool
        Stores wether the :code:`calc()` method can be called or not.
    """

    plumed_obj = None
    plumed_forces = None
    positions = None
    charges = None
    plumed_bias = np.zeros(1, np.double)
    plumed_version = np.zeros(1, dtype=np.intc)
    comm = None
    ready = False
    verbose = None

    def __init__(self, config, plumeddat, logfile, comm=MPI.COMM_WORLD, verbose=1):
        """Constructor

        Parameters
        ----------
        config : Config
            Configuration object.
        plumeddat : str
            Path file containing PLUMED input.
        logfile : str
            Path to PLUMED's output file
        comm : mpi4py.Comm, optional
            MPI communicator to use for rank communication.
        verbose : int, optional
            Specify the logging event verbosity of this object.

        See also
        --------
        hymd.input_parser.Config : Configuration dataclass handler.
        """
        try:
            import plumed
        except ImportError:
            err_str = (
                "You are trying to use PLUMED " "but HyMD could not import py-plumed."
            )
            Logger.rank0.log(logging.ERROR, err_str)
            raise ImportError(err_str)

        self.comm = comm
        self.verbose = verbose

        try:
            kernel_str = "Using PLUMED_KERNEL={}".format(os.environ["PLUMED_KERNEL"])
        except:
            kernel_str = "The PLUMED_KERNEL environment variable is not set."

        Logger.rank0.log(logging.INFO, kernel_str)

        try:
            self.plumed_obj = plumed.Plumed()
        except:
            err_str = (
                "HyMD was not able to create a PLUMED object. "
                "Maybe there is a problem with your PLUMED_KERNEL?"
            )
            Logger.rank0.log(logging.ERROR, err_str)
            raise RuntimeError(err_str)

        self.plumed_obj.cmd("getApiVersion", self.plumed_version)
        if self.plumed_version[0] <= 3:
            err_str = "HyMD requires a PLUMED API > 3. " "Use a newer PLUMED kernel."
            Logger.rank0.log(logging.ERROR, err_str)
            raise AssertionError(err_str)

        self.plumed_obj.cmd("setMDEngine", "HyMD")
        self.plumed_obj.cmd("setMPIComm", comm)

        Logger.rank0.log(
            logging.INFO, f"Attempting to read PLUMED input from {plumeddat}"
        )

        self.plumed_obj.cmd("setPlumedDat", plumeddat)
        self.plumed_obj.cmd("setLogFile", logfile)
        self.plumed_obj.cmd("setTimestep", config.respa_inner * config.time_step)
        self.plumed_obj.cmd("setNatoms", config.n_particles)
        self.plumed_obj.cmd("setKbT", config.gas_constant * config.target_temperature)
        self.plumed_obj.cmd("setNoVirial")
        self.plumed_obj.cmd("init")

        Logger.rank0.log(
            logging.INFO,
            f"Successfully read PLUMED input. PLUMED output file is {logfile}",
        )

    def finalize(self):
        """Finalize object"""
        self.plumed_obj.finalize()

    def __enter__(self):
        """Allow usage in 'with' context"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Finalize on exit of context"""
        self.plumed_obj.__exit__(exc_type, exc_value, traceback)

    @property
    def api_version(self):
        """Returns the API version got from the PLUMED kernel."""
        return self.plumed_version[0]

    def prepare(self, step, forces, positions, indices, config, charges):
        """Set the pointers to positions and forces, and returns
        wether the potential energy is being requested by PLUMED or not.
        """
        if self.verbose > 1:
            Logger.rank0.log(logging.INFO, f"Setting PLUMED pointers for step {step}")
        self.plumed_forces = forces.ravel(order="C").astype(np.double)
        self.charges = charges.astype(np.double)
        self.positions = positions.ravel(order="C").astype(
            np.double
        )  # get C-contiguous array

        needs_energy = np.zeros(1, np.intc)
        # plumed_virial = np.zeros((3,3), dtype=np.double)
        masses = np.full_like(indices, config.mass, dtype=np.double)
        box = np.diag(config.box_size).astype(np.double)

        self.plumed_obj.cmd("setAtomsNlocal", indices.shape[0])
        self.plumed_obj.cmd("setAtomsGatindex", indices)
        self.plumed_obj.cmd("setStep", step)
        self.plumed_obj.cmd("setForces", self.plumed_forces)
        self.plumed_obj.cmd("setPositions", self.positions)
        self.plumed_obj.cmd("setCharges", self.charges)
        self.plumed_obj.cmd("setMasses", masses)
        self.plumed_obj.cmd("setBox", box)
        # self.plumed_obj.cmd("setVirial", plumed_virial)

        # check if PLUMED needs energy and returns
        self.plumed_obj.cmd("prepareCalc")
        self.plumed_obj.cmd("isEnergyNeeded", needs_energy)

        self.ready = True
        return True if needs_energy[0] != 0 else False

    def calc(self, forces, poteng):
        """Passes the energy (which can be set to any value in case
        :code:`prepare()` returns PLUMED doesn't need it) to get
        the forces and energy from the bias.
        """
        if not self.ready:
            err_str = (
                "Trying to calculate PLUMED biased forces "
                "without first calling prepare method."
            )
            Logger.rank0.log(logging.ERROR, err_str)
            raise RuntimeError(err_str)

        if self.verbose > 1:
            Logger.rank0.log(logging.INFO, "Calculating PLUMED forces")

        # set the energy and calc
        self.plumed_obj.cmd("setEnergy", poteng)
        self.plumed_obj.cmd("performCalc")
        self.plumed_obj.cmd("getBias", self.plumed_bias)

        # check if the returned forces are valid
        if np.isnan(self.plumed_forces).any():
            err_str = (
                "Forces returned by PLUMED are not valid. "
                "It means there's a NaN in the computed forces, "
                "and your input should be checked. "
            )
            Logger.rank0.log(logging.ERROR, err_str)
            raise RuntimeError(err_str)

        # subtract forces to get only the bias' extra force
        self.plumed_forces -= forces.ravel(order="C")

        if self.verbose > 1:
            Logger.rank0.log(logging.INFO, "Done calculating PLUMED forces")

        self.ready = False
        return (
            np.asfortranarray(np.reshape(self.plumed_forces, forces.shape, order="C")),
            self.plumed_bias[0],
        )
