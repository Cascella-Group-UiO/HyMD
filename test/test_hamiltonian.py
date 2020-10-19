from mpi4py import MPI
import pytest
import logging
import pmesh
from types import ModuleType
import numpy as np
from hamiltonian import Hamiltonian, DefaultNoChi, DefaultWithChi, SquaredPhi
from input_parser import _find_unique_names, convert_CONF_to_config
from force import Chi


def test_DefaultNoChi_window_function(dppc_single, config_CONF, caplog):
    caplog.set_level(logging.INFO)
    indices, _, names, _, r, _ = dppc_single
    conf_file_name, _ = config_CONF
    CONF = {}
    CONF_ = {}
    exec(open(conf_file_name).read(), CONF)
    exec(open(conf_file_name).read(), CONF_)
    CONF = {k: v for k, v in CONF.items() if (not k.startswith('_') and
                                              not isinstance(v, ModuleType))}
    CONF_ = {k: v for k, v in CONF_.items() if (not k.startswith('_') and
                                                not isinstance(v, ModuleType))}
    with pytest.warns(Warning) as recorded_warning:
        config_conf = convert_CONF_to_config(CONF_)
        assert recorded_warning[0].message.args[0]
        assert caplog.text
    config_conf = _find_unique_names(config_conf, names)
    W = DefaultNoChi(config_conf)

    k = np.array(
        [[0.5321315378106508, -0.6711591309063634,  0.8362051282174443],
         [0.2853046917286570, -0.6542962742862817,  0.3174805390299977],
         [0.6999748102259762, -0.9385345654631219, -0.7383831543700541]]
    )
    v = np.array(
        [[0.6106860760556785, -0.5406324770662296, 0.6388756736156205],
         [-0.7348831910188103, 0.2808258965802970, 0.7446817693106476],
         [0.6458163432308923, -0.0526126093343278, 0.7065510160449484]]
    )
    for kk, vv in zip(k, v):
        assert np.allclose(W.H(kk, vv), CONF['H'](kk, vv), atol=1e-14)


def test_DefaultNoChi_energy_functional(config_CONF, caplog):
    caplog.set_level(logging.INFO)
    conf_file_name, _ = config_CONF
    CONF = {}
    CONF_ = {}
    exec(open(conf_file_name).read(), CONF)
    exec(open(conf_file_name).read(), CONF_)
    CONF = {k: v for k, v in CONF.items() if (not k.startswith('_') and
                                              not isinstance(v, ModuleType))}
    CONF_ = {k: v for k, v in CONF_.items() if (not k.startswith('_') and
                                                not isinstance(v, ModuleType))}
    with pytest.warns(Warning) as recorded_warning:
        config_conf = convert_CONF_to_config(CONF_)
        assert recorded_warning[0].message.args[0]
        assert caplog.text
    config_conf = _find_unique_names(config_conf, list(CONF['NAMES']))
    W = DefaultNoChi(config_conf)

    r = np.array(
        [[5.0856957005460090, 2.6917109262984917, 2.1680859811719504],
         [0.0592355479043435, 7.4705053874595030, 0.7169198724514365],
         [6.8662344320308580, 3.1065309079252390, 3.5621049784729850],
         [5.5405900785250210, 6.5585020954226065, 1.3155113935992528],
         [1.6400925628903045, 2.1325438112867725, 3.4149968975443680]]
    )
    types = np.array([0, 0, 1, 1, 1], dtype=int)

    assert config_conf.mesh_size == CONF['Nv']
    assert np.allclose(np.asarray(config_conf.box_size), np.asarray(CONF['L']),
                       atol=1e-14)
    mesh_size = config_conf.mesh_size
    pm = pmesh.ParticleMesh((mesh_size, mesh_size, mesh_size),
                            BoxSize=config_conf.box_size, dtype='f8',
                            comm=MPI.COMM_WORLD)
    layouts = [pm.decompose(r[t == types]) for t in
               range(len(config_conf.unique_names))]
    painted = [
        pm.paint(r[t == types], layout=layouts[t])
        for t in range(len(config_conf.unique_names))
    ]
    painted_ = [
        pm.paint(r[t == types], layout=layouts[t])
        for t in range(CONF['ntypes'])
    ]
    # Particles -> grid operation should conserve total mass
    assert np.sum(np.sum(painted[0])) == pytest.approx(2.0, abs=1e-14)
    assert np.sum(np.sum(painted_[0])) == pytest.approx(2.0, abs=1e-14)
    assert np.sum(np.sum(painted[1])) == pytest.approx(3.0, abs=1e-14)
    assert np.sum(np.sum(painted_[1])) == pytest.approx(3.0, abs=1e-14)

    pm = pmesh.ParticleMesh([6, 6, 6],
                            BoxSize=np.array([5, 5, 5]), dtype='f8',
                            comm=MPI.COMM_WORLD)

    for p in painted:
        p.r2c(out=Ellipsis).apply(W.H, out=Ellipsis).c2r(out=Ellipsis)
    for p_ in painted_:
        p_.r2c(out=Ellipsis).apply(CONF['H'], out=Ellipsis).c2r(out=Ellipsis)
    for p, p_ in zip(painted, painted_):
        assert np.allclose(p, p_, atol=1e-14)

    # Particles -> grid -> FFT -> convolution with filter H -> iFFT operation
    # should conserve total mass
    assert np.sum(np.sum(painted[0])) == pytest.approx(2.0, abs=1e-14)
    assert np.sum(np.sum(painted_[0])) == pytest.approx(2.0, abs=1e-14)
    assert np.sum(np.sum(painted[1])) == pytest.approx(3.0, abs=1e-14)
    assert np.sum(np.sum(painted_[1])) == pytest.approx(3.0, abs=1e-14)

    filtered = painted
    filtered_ = painted_

    for t in range(len(config_conf.unique_names)):
        v_ext_fourier_space = (W.v_ext[t](filtered)
                                .r2c(out=Ellipsis)
                                .apply(W.H, out=Ellipsis))
        v_ext_fourier_space_ = (CONF['V_EXT'][t](filtered_)
                                .r2c(out=Ellipsis)
                                .apply(CONF['H'], out=Ellipsis))
        assert np.allclose(v_ext_fourier_space, v_ext_fourier_space_,
                           atol=1e-14)

    CONF['L'] = np.array(CONF['L'])
    CONF['V'] = CONF['L'][0] * CONF['L'][1] * CONF['L'][2]
    CONF['dV'] = CONF['V'] / (CONF['Nv']**3)
    V = np.prod(config_conf.box_size)
    n_mesh__cells = np.prod(np.full(3, config_conf.mesh_size))
    volume_per_cell = V / n_mesh__cells
    painted = [
        pm.paint(r[t == types], layout=layouts[t]) / volume_per_cell
        for t in range(len(config_conf.unique_names))
    ]
    painted_ = [
        pm.paint(r[t == types], layout=layouts[t]) / volume_per_cell
        for t in range(CONF['ntypes'])
    ]
    for p in painted:
        p.r2c(out=Ellipsis).apply(W.H, out=Ellipsis).c2r(out=Ellipsis)
    for p_ in painted_:
        p_.r2c(out=Ellipsis).apply(CONF['H'], out=Ellipsis).c2r(out=Ellipsis)
    w = W.w(painted) * volume_per_cell
    w_ = CONF['w'](painted_) * CONF['dV']

    assert np.allclose(w, w_, atol=1e-14)
    assert w.csum() == pytest.approx(w_.csum(), abs=1e-14)


def test_Hamiltonian_no_chi_gaussian_core(config_CONF, caplog):
    caplog.set_level(logging.INFO)
    conf_file_name, _ = config_CONF
    CONF = {}
    exec(open(conf_file_name).read(), CONF)
    CONF = {k: v for k, v in CONF.items() if (not k.startswith('_') and
                                              not isinstance(v, ModuleType))}
    CONF['Np'] = 3
    with pytest.warns(Warning) as recorded_warning:
        config_conf = convert_CONF_to_config(CONF)
        assert recorded_warning[0].message.args[0]
        assert caplog.text

    names = np.array([np.string_(s) for s in ['A', 'A']])
    config = _find_unique_names(config_conf, names)
    config.box_size = np.array([15.0, 15.0, 15.0])
    config.mesh_size = np.array([160, 160, 160])
    config.n_particles = 3
    r = np.array(
        [[1.50, 0.75, 2.25],
         [2.25, 0.00, 3.00],
         [4.50, 1.50, 2.25]]
    )
    pm = pmesh.ParticleMesh(config.mesh_size, BoxSize=config.box_size,
                            dtype='f8', comm=MPI.COMM_WORLD)
    V = np.prod(config.box_size)
    n_mesh__cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh__cells

    layout = pm.decompose(r)
    phi = pm.paint(r, layout=layout) / volume_per_cell
    assert phi.csum() == pytest.approx(config.n_particles / volume_per_cell,
                                       abs=1e-14)
    hamiltonian = Hamiltonian(config)
    phi = (phi.r2c(out=Ellipsis)
              .apply(hamiltonian.H, out=Ellipsis)
              .c2r(out=Ellipsis))
    assert phi.csum() == pytest.approx(config.n_particles / volume_per_cell,
                                       abs=1e-11)

    for W in [SquaredPhi(config), DefaultNoChi(config)]:
        w = (W.w([phi]) * volume_per_cell).csum()

        # Gaussian core model energy
        pi32 = np.arccos(-1.0)**(3.0 / 2.0)
        c = 16.0 * pi32 * config.kappa * config.sigma**3 * config.rho0
        diag = config.n_particles / c
        offdiag = 0
        for i in range(config.n_particles):
            for j in range(i + 1, config.n_particles):
                rij = r[i, :] - r[j, :]
                rij2 = np.dot(rij, rij)
                offdiag += 2 * np.exp(- rij2 / (4.0 * config.sigma**2)) / c
        if isinstance(W, DefaultNoChi):
            E = diag + offdiag - 0.5 * config.n_particles / config.kappa
        else:
            E = diag + offdiag
        assert w == pytest.approx(E, abs=1e-6)

    # three particles, W=phi^2/(2 phi0 kappa)
    # 27046.447859682066 + 160.09645705998503 = 27206.544316742053 -->const+exp
    # 27206.544316742053 --> mathematica
    # 27206.544316741987 --> grid 640
    # 27206.544316741983 --> grid 320
    # 27206.544316741983 --> grid 160

    # two particles, W=phi^2/(2 phi0 kappa)
    # 27046.44785968207 + 240.14467831817544 = 27286.592538000245 -->const+exp
    # 27286.592538000245 --> mathematica
    # 27286.592538000346 --> grid 640
    # 27286.592538000346 --> grid 320
    # 27286.59253800035 --> grid 160

    # two particles, W=phi^2
    # 1.6823405640794202 + 0.014937456322488748 = 1.6972780204 -->const+exp
    # 1.697278020401909 --> mathematica
    # 1.6972780204019127 --> grid 640
    # 1.6972780204019127 --> grid 320
    # 1.6972780204019127 --> grid 160

    # one particle, W=phi^2/(2 phi0 kappa)
    # 27046.44785968207 --> mathematica
    # 27046.447859682085 --> grid 640
    # 27046.447859682085 --> grid 320
    # 27046.447859682088 --> grid 160

    # one particle, W=phi^2
    # 0.8411702820397101 --> mathematica
    # 0.841170282039712 --> grid 640
    # 0.841170282039712  --> grid 320
    # 0.8411702820397121 --> grid 160


def test_Hamiltonian_with_chi_gaussian_core(config_CONF, caplog):
    caplog.set_level(logging.INFO)
    conf_file_name, _ = config_CONF
    CONF = {}
    exec(open(conf_file_name).read(), CONF)
    CONF = {k: v for k, v in CONF.items() if (not k.startswith('_') and
                                              not isinstance(v, ModuleType))}
    with pytest.warns(Warning) as recorded_warning:
        config_conf = convert_CONF_to_config(CONF)
        assert recorded_warning[0].message.args[0]
        assert caplog.text

    names = np.array([np.string_(s) for s in ['A', 'A', 'B', 'C', 'C']])
    types = np.array([0, 0, 1, 2, 2])
    config = _find_unique_names(config_conf, names)
    config.box_size = np.array([15.0, 15.0, 15.0])
    config.mesh_size = np.array([160, 160, 160])
    config.n_particles = 5
    r = np.array(
        [[1.50, 0.75, 2.25],
         [2.25, 0.00, 3.00],
         [4.50, 1.50, 2.25],
         [1.50, 1.50, 0.75],
         [3.00, 4.50, 1.50]]
    )
    pm = pmesh.ParticleMesh(config.mesh_size, BoxSize=config.box_size,
                            dtype='f8', comm=MPI.COMM_WORLD)
    V = np.prod(config.box_size)
    n_mesh__cells = np.prod(np.full(3, config.mesh_size))
    volume_per_cell = V / n_mesh__cells

    layouts = [pm.decompose(r[types == t]) for t in range(config.n_types)]
    phi = [pm.paint(r[types == t], layout=layouts[t]) / volume_per_cell
           for t in range(config.n_types)]
    # assert sum(phi).csum() == pytest.approx(config.n_particles/volume_per_cell, abs=1e-14)    <<<<<---- FIXME
    hamiltonian = Hamiltonian(config)
    for t in range(config.n_types):
        phi[t] = (phi[t].r2c(out=Ellipsis)
                        .apply(hamiltonian.H, out=Ellipsis)
                        .c2r(out=Ellipsis))
    # assert sum(phi).csum() == pytest.approx(config.n_particles/volume_per_cell, abs=1e-14)      <<<<<---- FIXME

    chi_ = [
      [['A', 'B'],   [9.6754032616815161]],
      [['A', 'C'], [-13.2596290315913623]],
      [['B', 'C'],   [0.3852001771213374]]
    ]
    chi_dict = {("A", "B"):   9.6754032616815161,
                ("A", "C"): -13.2596290315913623,
                ("B", "C"):   0.3852001771213374}
    chi = [None] * ((config.n_types - 1) * config.n_types // 2)
    for i, c in enumerate(chi_):
        chi[i] = Chi(
            atom_1=c[0][0], atom_2=c[0][1], interaction_energy=c[1][0]
        )
    config.chi = chi
    type_to_name_map = {0: 'A', 1: 'B', 2: 'C'}

    W = DefaultWithChi(config, config.unique_names, type_to_name_map)
    W_ = DefaultNoChi(config)
    w = (W.w(phi) * volume_per_cell).csum()
    w_ = (W_.w(phi) * volume_per_cell).csum()

    # Gaussian core model energy
    pi32 = np.arccos(-1.0)**(3.0 / 2.0)
    c = 16.0 * pi32 * config.kappa * config.sigma**3 * config.rho0
    diag = config.n_particles / c
    offdiag = 0
    for i in range(config.n_particles):
        for j in range(i + 1, config.n_particles):
            rij = r[i, :] - r[j, :]
            rij2 = np.dot(rij, rij)
            offdiag += 2 * np.exp(- rij2 / (4.0 * config.sigma**2)) / c
    E = diag + offdiag - 0.5 * config.n_particles / config.kappa
    assert w_ == pytest.approx(E, abs=1e-6)

    interaction_energy = 0
    for i in range(config.n_particles):
        for j in range(i + 1, config.n_particles):
            ni = names[i].decode('utf-8')
            nj = names[j].decode('utf-8')
            if ni != nj:
                rij = r[i, :] - r[j, :]
                rij2 = np.dot(rij, rij)
                c_ = 2 * config.kappa * chi_dict[tuple(sorted([ni, nj]))] / c
                interaction_energy += (
                    c_ * np.exp(- rij2 / (4.0 * config.sigma**2))
                )
    E = E + interaction_energy
    assert w == pytest.approx(E, abs=1e-6)

    # five particles, three types, W=sum chi(i,j) phi(i) phi(j)
    # -2.8663135769749566  --> mathematica
    # -2.8663135769747057 --> grid 640
    # -2.866313576974921 --> grid 320
    # -2.866313576974796 --> grid 160
