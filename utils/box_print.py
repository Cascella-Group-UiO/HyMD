import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys

f = h5py.File(str(sys.argv[1]), "r")
first_frame = int(sys.argv[2])
last_frame = int(sys.argv[3])
# f = h5py.File('sim_binary100ps.h5', 'r')
x_value = [_[0] for _ in list(f["particles/all/box/edges/value"])[first_frame:last_frame]]
y_value = [_[1] for _ in list(f["particles/all/box/edges/value"])[first_frame:last_frame]]
z_value = [_[2] for _ in list(f["particles/all/box/edges/value"])[first_frame:last_frame]]
volume = [x_value[i] * y_value[i] * z_value[i] for i in range(len(x_value))]
N = len(list(f["particles/all/species"]))
density = [0.11955 * N / _ for _ in volume] #gm/cc
time = list(f["particles/all/box/edges/time"])[first_frame:last_frame]
#time = np.arange(first_frame,last_frame)
# p_kin = [value[i][0] for i in range(len(value))]
# p0 = [value[i][1] for i in range(len(value))]
# p1 = [value[i][2] for i in range(len(value))]
# p2x = [value[i][3] for i in range(len(value))]
# p2y = [value[i][4] for i in range(len(value))]
# p2z = [value[i][5] for i in range(len(value))]
# for t, p in zip(time, value):
#    print(t,'\t',p)
#

stillbox=[]
print('x_value:',x_value)
for i in range(len(x_value)-1):
    if(x_value[i] == x_value[i+1]):
        stillbox.append("(%s)"%(str(i)))
if(stillbox): print("unchanged box frames:",stillbox)
print('Last frame:', last_frame, '; box_size: [',x_value[-1],' ',y_value[-1],' ',z_value[-1],']')

##PLOTS
plt.xlabel("Time (ps)")
plt.ylabel("Length (nm)")
plt.plot(time, x_value, label='Box length in x')
plt.plot(list(time), y_value, label='Box length in y')
plt.plot(list(time), z_value, marker='o', label='Box length in z')
plt.legend()
plt.savefig("box.png")
plt.show()
plt.plot(time, density, label="Density (gm/cc)")
plt.legend()
plt.show()
