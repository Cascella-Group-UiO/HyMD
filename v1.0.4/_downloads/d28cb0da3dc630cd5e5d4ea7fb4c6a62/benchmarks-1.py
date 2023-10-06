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