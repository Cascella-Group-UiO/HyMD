import numpy as np


def velocity_rescale(velocity, config, comm, R1=None, Ri2_sum=None):
    K = comm.allreduce(0.5 * config.mass * np.sum(velocity ** 2))
    K_target = (
        (3 / 2) * (2.479 / 298.0) * config.n_particles * config.target_temperature
    )
    N_f = 3 * config.n_particles
    exp1 = np.exp(-config.time_step / config.tau)
    exp2 = np.exp(-config.time_step / (2 * config.tau))

    if R1 is None and Ri2_sum is None:
        if comm.rank == 0:
            R1 = np.random.normal()

            # (degrees of freedom - 1) even
            if np.mod(N_f - 1, 2) == 0:
                Ri2_sum = 2 * np.random.gamma((N_f - 1) / 2, scale=1.0)

            # (degrees of freedom - 1) odd
            else:
                Ri2_sum = 2 * np.random.gamma((N_f - 2) / 2, scale=1.0) + R1 ** 2
            print(R1, Ri2_sum)
        R1 = comm.bcast(R1, root=0)
        Ri2_sum = comm.bcast(Ri2_sum, root=0)

    alpha2 = (
        +exp1
        + K_target / (K * N_f) * (1 - exp1) * (R1 * R1 + Ri2_sum)
        + 2 * exp2 * R1 * np.sqrt((1 - exp1) * K_target / (K * N_f))
    )
    alpha = np.sqrt(alpha2)
    velocity = velocity * alpha
    config.thermostat_work += K * (alpha2 - 1.0)
    return velocity
