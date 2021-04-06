import numpy as np
from mpi4py import MPI


def velocity_rescale(velocity, names, config, comm, R1=None, Ri2_sum=None):
    if not any(config.thermostat_coupling_groups):
        config.thermostat_coupling_groups = [config.unique_names.copy()]
    for i, group in enumerate(config.thermostat_coupling_groups):
        ind = np.where(
            np.logical_or.reduce(list(names == np.string_(t) for t in group))
        )
        group_n_particles = len(ind[0])
        K = comm.allreduce(0.5 * config.mass * np.sum(velocity[ind, :]**2))
        K_target = (
            (3 / 2) * (2.479 / 298.0) * group_n_particles * config.target_temperature
        )
        N_f = 3 * group_n_particles
        exp1 = np.exp(-config.time_step / config.tau)
        exp2 = np.exp(-config.time_step / (2 * config.tau))

        if R1 is None and Ri2_sum is None:
            R1_ = None
            Ri2_sum_ = None
            if comm.rank == 0:
                R1_ = np.random.normal()

                # (degrees of freedom - 1) even
                if np.mod(N_f - 1, 2) == 0:
                    Ri2_sum_ = 2 * np.random.gamma((N_f - 1) / 2, scale=1.0)

                # (degrees of freedom - 1) odd
                else:
                    Ri2_sum_ = 2 * np.random.gamma((N_f - 2) / 2, scale=1.0) + R1**2
            R1_ = comm.bcast(R1_, root=0)
            Ri2_sum_ = comm.bcast(Ri2_sum_, root=0)

        else:
            try:
                R1_ = R1[i]
            except TypeError as e:
                n_groups = len(config.thermostat_coupling_groups)
                err_str = (
                    f"Could not interpret the provided R1 ({R1}) and/or "
                    f"Ri2_sum ({Ri2_sum}) as lists of random numbers (of "
                    f"size equal the number of thermostat coupling groups "
                    f"[{n_groups}]) or as single random numbers."
                )
                if isinstance(R1, (int, float)):
                    R1_ = R1
                else:
                    raise ValueError(err_str) from e
            try:
                Ri2_sum_ = Ri2_sum[i]
            except TypeError as e:
                if isinstance(Ri2_sum, (int, float)):
                    Ri2_sum_ = Ri2_sum
                else:
                    raise ValueError(err_str) from e

        thermostat_work_ = 0.0
        if group_n_particles > 0:
            alpha2 = (
                +exp1
                + K_target / (K * N_f) * (1 - exp1) * (R1_ * R1_ + Ri2_sum_)
                + 2 * exp2 * R1_ * np.sqrt((1 - exp1) * K_target / (K * N_f))
            )
            kinetic_energy_before = 0.5 * config.mass * np.sum(velocity[ind, :]**2)
            alpha = np.sqrt(alpha2)
            velocity[ind, :] = velocity[ind, ...] * alpha
            kinetic_energy_after = 0.5 * config.mass * np.sum(velocity[ind, :]**2)
            # thermostat_work_ = K * (alpha2 - 1.0)
            thermostat_work_ = kinetic_energy_after - kinetic_energy_before

        thermostat_work = comm.allreduce(thermostat_work_, MPI.SUM)
        config.thermostat_work += thermostat_work
    return velocity
