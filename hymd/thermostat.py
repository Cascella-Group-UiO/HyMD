import numpy as np


def velocity_rescale(velocity, config, comm):
    """ Velocity rescale thermostat, see
        https://doi.org/10.1063/1.2408420
            Parameters
            ----------
            vel : [N_mpi,3] float array beloning to MPI-task
            tau : float, relaxation time of thermostat.
            Returns:
            ----------
            out : vel
            Thermostatted velocity array.
    """
    # Initial kinetic energy
    kinetic_energy = comm.allreduce(0.5 * config.mass * np.sum(velocity**2))

    # Berendsen-like term
    Ek0 = (3 / 2) * (2.479 / 298.0) * config.n_particles * config.target_temperature  # noqa: E501
    d1 = (Ek0 - kinetic_energy) * config.time_step / config.tau

    # Wiener noise
    dW = np.sqrt(config.time_step) * np.random.normal()

    # Stochastic term
    d2 = (2 * np.sqrt(kinetic_energy * Ek0 / (3 * config.n_particles))
          * dW / np.sqrt(config.tau))

    # Target kinetic energy
    target_kinetic_energy = kinetic_energy + d1 + d2

    # Velocity scaling
    alpha = np.sqrt(target_kinetic_energy / kinetic_energy)
    velocity = velocity * alpha
    return velocity
