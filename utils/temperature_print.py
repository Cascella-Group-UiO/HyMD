import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
#f = h5py.File('sim_binary100ps.h5', 'r')
f = h5py.File(str(sys.argv[1]), 'r')
first_frame = int(sys.argv[2])
last_frame = int(sys.argv[3])
print(list(f['observables/temperature']))
value = list(f['observables/temperature/value'])[first_frame:last_frame]
time = list(f['observables/temperature/time'])[first_frame:last_frame]
#p_kin = [value[i][0] for i in range(len(value))]
#p0 = [value[i][1] for i in range(len(value))]
#p1 = [value[i][2] for i in range(len(value))]
#p2x = [value[i][3] for i in range(len(value))]
#p2y = [value[i][4] for i in range(len(value))]
#p2z = [value[i][5] for i in range(len(value))]
#for t, p in zip(time, value):
#    print(t,'\t',p)
print('Average:',np.average(value))
#PLOTS
plt.xlabel('Time (ps)')
plt.plot(list(time), value, label='Temp')
#plt.plot(list(time), p0, label='p0')
#plt.plot(list(time), p1, label='p1')
#plt.plot(list(time), p2x, label='p2x')
#plt.plot(list(time), p2y, label='p2y')
#plt.plot(list(time), p2z, label='p2z')
plt.legend()
plt.savefig("temperature.png")
plt.show()

