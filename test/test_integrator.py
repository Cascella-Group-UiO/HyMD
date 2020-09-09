import pytest
import numpy as np
from hPF.integrator import integrate_velocity, integrate_position


def test_integrator_velocity_zero_force(three_atoms):
    _, _, _, _, _, velocities, CONF = three_atoms
    old_velocities = velocities.copy()
    velocities = integrate_velocity(velocities, np.zeros_like(velocities),
                                    CONF['dt'])
    assert np.allclose(old_velocities, velocities, atol=1e-13)


def test_integrator_velocity(three_atoms):
    _, _, _, _, _, velocities, CONF = three_atoms
    accelerations = np.array([
        [5.309962614030253,  -5.068488339109831,  0.828090598458989],
        [-3.868819942226709, -9.720221218265905,  4.036326041075613],
        [9.728114214745183,  -8.189855111379165, -3.075260427419437]
    ], dtype=np.float64)
    velocities = integrate_velocity(velocities, accelerations, CONF['dt'])
    expected_velocities = np.array([
        [0.3004100792669800, -0.452528545222956, -0.288323467370162],
        [-0.153011424114385,  0.313703442307804,  0.420088246650352],
        [0.4402634172106540,  0.077338401336559,  0.041682352532849]
    ], dtype=np.float64)
    assert np.allclose(expected_velocities, velocities, atol=1e-13)


def test_integrator_position_zero_velocity_zero_force(three_atoms):
    _, _, _, _, positions, velocities, CONF = three_atoms
    old_positions = positions.copy()
    positions = integrate_position(positions, np.zeros_like(velocities),
                                   CONF['dt'])
    assert np.allclose(old_positions, positions, atol=1e-13)


def test_integrator_position_zero_force(three_atoms):
    _, _, _, _, positions, velocities, CONF = three_atoms
    positions = integrate_position(positions, velocities, CONF['dt'])
    expected_positions = np.array([
        [3.341937188559520, 3.053303418939576, 3.648148235458336],
        [4.254612887747975, 3.315211337414966, 4.720346322206166],
        [2.170585066308332, 2.215015584448489, 4.184927411402398]
    ], dtype=np.float64)
    assert np.allclose(expected_positions, positions, atol=1e-13)


def test_integrator_position(three_atoms):
    _, _, _, _, positions, velocities, CONF = three_atoms
    accelerations = np.array([
        [5.309962614030253,  -5.068488339109831,  0.828090598458989],
        [-3.868819942226709, -9.720221218265905,  4.036326041075613],
        [9.728114214745183,  -8.189855111379165, -3.075260427419437]
    ], dtype=np.float64)
    positions = integrate_position(positions, velocities, CONF['dt'])
    expected_positions = np.array([
        [8.650700368489337,  -2.014040031176965, 4.476051781733678],
        [0.386666848956082,  -6.402814241137364, 8.755760622946480],
        [11.896501858440546, -5.972989572004016, 1.110361635243710]
    ], dtype=np.float64)
    # assert np.allclose(expected_positions, positions, atol=1e-13)


def test_integrator_velocity_verlet_constant_force(three_atoms):
    """With a constant force F₀ = (α, β, γ), the velocity obeys the recurrence
    relation

        v       =   v      +  Δt  a
          i+1         i             0

    with solution

        v       =   v      +  i Δt a
          i           0              0

    The position obeys the recurrence relation
                                 ╭     ╮       1    2
        p       =   p      +  Δt │ v   │   +  ─── Δt  a
          i+1         i          ╰   i ╯       2        0
                                 ╭                 ╮       1    2
                =   p      +  Δt │ v   +  i Δt a   │   +  ─── Δt  a
                      i          ╰   0           0 ╯       2        0

    with solution
                               i Δt ╭                      ╮
        p       =   p      +  ───── │ a   Δt i   +   2 v   │
          i           0         2   ╰   0                0 ╯
    """
    _, _, _, _, positions, velocities, CONF = three_atoms
    initial_positions = positions.copy()
    initial_velocities = velocities.copy()
    dt = CONF['dt']
    accelerations = np.array([
        [-7.539285072366672,  0.968464568407796,  0.598639645160741],
        [9.1962508228909070,  1.505356899114361,  3.601585312042335],
        [8.8869056807446400, -0.972763307999720, -7.335044988262844]
    ], dtype=np.float64)

    for step in range(1, 10):
        # Velocity Verlet steps
        velocities = integrate_velocity(velocities, accelerations, dt)
        positions = integrate_position(positions, velocities, dt)
        velocities = integrate_velocity(velocities, accelerations, dt)

        for i in range(3):
            for d in range(3):
                v = velocities[i, d]
                a = accelerations[i, d]
                v_0 = initial_velocities[i, d]
                v_expected = v_0 + step * dt * a
                assert v == pytest.approx(v_expected, abs=1e-13)

                p = positions[i, d]
                p_0 = initial_positions[i, d]
                p_expected = p_0 + 0.5 * step * dt * (a * dt * step + 2 * v_0)
                assert p == pytest.approx(p_expected, abs=1e-13)


def test_integrator_velocity_verlet_linear_force(three_atoms):
    """With a time-linear force F(t) = (α t, β t, γ t) the velocity obeys the
    relation
                             α Δt ╭            ╮
       v       =   v     +  ───── │ 2 t  +  Δt │
         i+1         0        2   ╰            ╯

    with solution
                             1      2   2
       v       =   v     +  ─── α Δt  i               (1)
         i           0       2

    The position recurrence relation takes the form
                               ╭         α    2  2 ╮    1      3
       p       =   p     +  Δt │ v    + ─── Δt  i  │ + ─── α Δt  i
         i+1         0         ╰   0     2         ╯    2

    with solution
                             1      3   ╭  2      ╮
       p       =   p     +  ─── α Δt  i │ i  -  1 │ +  Δt i v
         i           0       6          ╰         ╯           0
    """
    _, _, _, _, positions, velocities, CONF = three_atoms
    initial_positions = positions.copy()
    initial_velocities = velocities.copy()
    dt = CONF['dt']
    acceleration_constants = np.array([
        [-7.539285072366672,  0.968464568407796,  0.598639645160741],
        [9.1962508228909070,  1.505356899114361,  3.601585312042335],
        [8.8869056807446400, -0.972763307999720, -7.335044988262844]
    ], dtype=np.float64)

    # Initial forces at t=0 are zero
    accelerations = np.zeros_like(acceleration_constants)
    for step in range(1, 10):
        # Velocity Verlet steps
        velocities = integrate_velocity(velocities, accelerations, dt)
        positions = integrate_position(positions, velocities, dt)

        # Update forces
        accelerations = step * dt * acceleration_constants
        velocities = integrate_velocity(velocities, accelerations, dt)

        for i in range(3):
            for d in range(3):
                v = velocities[i, d]
                a_0 = acceleration_constants[i, d]
                v_0 = initial_velocities[i, d]
                v_expected = v_0 + 0.5 * a_0 * dt**2 * step**2
                assert v == pytest.approx(v_expected, abs=1e-13)

                p = positions[i, d]
                p_0 = initial_positions[i, d]
                p_expected = (p_0 + (step**2 - 1) * a_0 * dt**3 * step / 6.0 +
                              dt * step * v_0)
                assert p == pytest.approx(p_expected, abs=1e-13)
