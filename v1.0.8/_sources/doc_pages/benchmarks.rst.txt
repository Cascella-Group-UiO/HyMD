.. _benchmarks-label:

Benchmarks
##########
Homopolymer melt test systems of :math:`30^2`, :math:`40^2`, and :math:`50^2`
nanometers containing 224 k, 533 k, and 1.04 M particles, respectively.

Bonded forces
^^^^^^^^^^^^^
Legend label denotes the number of particles in the system.

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   nodes_small = np.array([1, 2, 5, 8, 12, 15, 17, 18], dtype=np.float64)
   time_small = np.array([
      4.187099, 2.403876, 1.604452, 1.104753, 1.102880, 1.102931, 1.091972,
      1.090594
   ], dtype=np.float64)
   nodes_medium = np.array([
      1, 3, 4, 7, 13, 14, 16, 25, 30, 31, 35, 40, 42, 46, 47, 50
   ], dtype=np.float64,)
   time_medium = np.array([
      11.201839, 5.281660, 3.693661, 2.317845, 1.566609, 1.425823, 1.437634,
      0.962915, 0.955779, 0.958936, 0.966228, 0.973874, 0.974221, 0.976087,
      0.982746, 0.950908,
   ], dtype=np.float64)
   nodes_large = np.array([
      1, 2, 3, 4, 5, 7, 12, 13, 16, 25, 26, 30, 37, 49, 55, 71, 80
   ], dtype=np.float64)
   time_large = np.array([
      20.77469, 12.257692, 9.430474, 6.560756, 5.340657, 3.760844, 3.655468,
      2.623202, 2.315075, 1.668369, 1.497107, 1.533885, 1.414385, 1.417828,
      1.404457, 1.411190, 1.435383,
   ], dtype=np.float64)
   plt.plot(
      nodes_small * 128, 1e6 * time_small / 10000.0, "r-o", label="224 k"
   )
   plt.plot(
      nodes_medium * 128, 1e6 * time_medium / 10000.0, "b-x", label="533 k"
   )
   plt.plot(
      nodes_large * 128, 1e6 * time_large / 10000.0, "k-^", label="1.04 M"
   )
   plt.xlabel("CPUs", fontsize=15)
   plt.ylabel("Time per step, us", fontsize=15)
   plt.legend(fontsize=15)
   plt.xscale('log', base=2)
   plt.yscale('log', base=2)
   plt.show()


Particle--field forces
^^^^^^^^^^^^^^^^^^^^^^
Legend label denotes the number of mesh grid points used in the FFT, essentially
the energy conservation precision of the method.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt

    mesh_2048_cpus = [2048,     1024,     512,       256,       128,       64,        32,        16,        8,          4         , ]
    mesh_2048_time = [3.375108, 5.977880, 10.414429, 16.859782, 31.692761, 50.496983, 51.508302, 76.465812, 114.956267, 200.790592, ]

    mesh_1024_cpus = [2048,     1024,     512,      256,      128,      64,       32,       16,       8,         4,         2,         1        , ]
    mesh_1024_time = [0.439009, 0.662988, 1.172991, 2.024713, 3.625324, 7.148897, 7.330420, 9.117424, 13.209331, 22.204145, 41.150021, 71.836402, ]

    mesh_512_cpus = [1024,    512,      256,       128,      64,       32,       16,       8,        4,        2,       1        , ]
    mesh_512_time = [0.137271, 0.148182, 0.161536, 0.271535, 0.798458, 0.817093, 1.114978, 1.712410, 2.870377, 5.151649, 8.589524, ]

    mesh_256_cpus = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    mesh_256_time = [0.033834, 0.037837, 0.053191, 0.070217, 0.115923, 0.200090, 0.336702, 0.665111, 1.127391]

    mesh_128_cpus = [64,       32,       16,       8,        4,        2,        1,        ]
    mesh_128_time = [0.012815, 0.018430, 0.022436, 0.037648, 0.055399, 0.093687, 0.169120, ]

    mesh_64_cpus = [64,       32,       16,       8,        4,        2,        1,        ]
    mesh_64_time = [0.012815, 0.018162, 0.018430, 0.037648, 0.037949, 0.055399, 0.093687, ]

    plt.loglog(mesh_2048_cpus, mesh_2048_time, "r-x", label="2048³")
    plt.loglog(mesh_1024_cpus, mesh_1024_time, "b-o", label="1024³")
    plt.loglog(mesh_512_cpus, mesh_512_time, "k-^", label="512³")
    plt.loglog(mesh_256_cpus, mesh_256_time, "c-v", label="256³")
    plt.loglog(mesh_128_cpus, mesh_128_time, "y-8", label="128³")
    plt.loglog(mesh_64_cpus, mesh_64_time, "g->", label="64³")
    plt.xlabel("CPUs", fontsize=15)
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.ylabel("Time per step, s", fontsize=15)
    plt.legend(loc="upper left", fontsize=15)
    plt.show()
