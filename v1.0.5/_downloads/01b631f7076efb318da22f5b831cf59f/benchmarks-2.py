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