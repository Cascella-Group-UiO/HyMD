"""Integrates equations of motion.

Implements the Velocity Verlet integrator. The *reversible reference system
propagator algorithm* (rRESPA) integrator uses the Velocity Verlet functions
inside the MD loop in the main module.
"""


def integrate_velocity(velocities, accelerations, time_step):
    """Velocity update step of the Velocity Verlet integration algorithm.

    Computes the velocity update step of the Velocity Verlet algorithm. The
    :code:`velocities` argument is **not** changed in place.

    Parameters
    ----------
    velocities : (N, D) numpy.ndarray
        Array of velocities of :code:`N` particles in :code:`D` dimensions.
    accelerations : (N, D) numpy.ndarray
        Array of accelerations of :code:`N` particles in :code:`D` dimensions.
    time_step : float
        The time step used in the integration.

    Notes
    -----
    The Velocity Verlet algorithm contains two steps, first the velocties are
    integrated one half step forward in time by applying the forces from the
    previous step

    .. math:: \\mathbf{v}_{\\text{new}} = \\mathbf{v} + \\frac{\\Delta t}{2m}
              \\mathbf{f}

    before the positions are moved a full step using the updated half-step
    velocities.

    See Also
    --------
    integrate_position : The position update step of the Velocity Verlet
                         algorithm.
    """
    return velocities + 0.5 * time_step * accelerations


def integrate_position(positions, velocities, time_step):
    """Position update step of the Velocity Verlet integration algorithm.

    Computes the position update step of the Velocity Verlet algorithm. The
    :code:`positions` argument is **not** changed in place.

    Parameters
    ----------
    positions : (N, D) numpy.ndarray
        Array of positions of :code:`N` particles in :code:`D` dimensions.
    velocities : (N, D) numpy.ndarray
        Array of velocities of :code:`N` particles in :code:`D` dimensions.
    time_step : float
        The time step used in the integration.

    Notes
    -----
    The Velocity Verlet algorithm contains two steps, first the velocties are
    integrated one half step forward in time by applying the forces from the
    previous step, then the positions are updated a full step using the updated
    velocities

    .. math:: \\mathbf{x}_{\\text{new}} = \\mathbf{x} + \\Delta t \\mathbf{v}

    See Also
    --------
    integrate_velocity : The velocity update step of the Velocity Verlet
                         algorithm.
    """
    return positions + time_step * velocities
