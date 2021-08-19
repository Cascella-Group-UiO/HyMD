import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
f = h5py.File(str(sys.argv[1]), 'r')
value = f['observables/pressure/value']
start_frame = int(sys.argv[2])
last_frame = int(sys.argv[3])
time = list(f['observables/pressure/time'] )[start_frame:last_frame]
#time = np.arange(start_frame,last_frame)

value = np.array(list(value))[start_frame:last_frame]
if(len(value[0])==9):
    pr = [
        p_kin, p0, p1,
        p2x, p2y, p2z,
        p_tot_x, p_tot_y, p_tot_z
    ]  = [value[:,i] for i in range(len(value[0]))]
else:
    pr = [
        p_kin, p0, p1,
        p2x, p2y, p2z,
        p_bond_x, p_bond_y, p_bond_z,
        p_angle_x, p_angle_y, p_angle_z,
        p_dihedral_x, p_dihedral_y, p_dihedral_z,
        p_tot_x, p_tot_y, p_tot_z
    ]  = [value[:,i] for i in range(len(value[0]))]


#print('p_tot_z values from start_frame to last_frame:',p_tot_z[start_frame:last_frame])

#AVERAGE VALUE PRINTS
print('Average p_tot_x:',np.average(p_tot_x[round(0.5*last_frame):last_frame]))
print('Average p_tot_y:',np.average(p_tot_y[round(0.5*last_frame):last_frame]))
print('Average p_tot_z:',np.average(p_tot_z[round(0.5*last_frame):last_frame]))

#PLOTS
color = ['b','g','r','c','m','y','k','brown','gray','orange','purple']
#plt.ylim(-18,15)
#plt.xlabel('Time (ps)')
#plt.plot(list(time), p_kin, label='p_kin', color=color[0])
#plt.plot(list(time), p0, label='p0', color=color[1])
#plt.plot(list(time), p1, label='p1', color=color[2])

#plt.plot(list(time), p2x, label='p2x', color=color[3])
#plt.plot(list(time), p2y, label='p2y', color=color[4])
#plt.plot(list(time), p2z, label='p2z', color=color[5])

#plt.plot(list(time), p2z - (p2x+p2y)/2, label='p_field_N - p_field_L', color=color[2])

#plt.plot(list(time), (p_bond_x+p_bond_y)/2, label='Avg p_bond in x,y', color=color[6])
#plt.plot(list(time), p_bond_x, label='p_bond_x', color=color[6])
#plt.plot(list(time), p_bond_y, label='p_bond_y', color=color[7])
#plt.plot(list(time), p_bond_z, label='p_bond_z', color=color[8])

#plt.plot(list(time), (p_angle_x+p_angle_y)/2, label='Avg p_angle in x,y', color=color[9])
#plt.plot(list(time), p_angle_x, label='p_angle_x', color=color[9])
#plt.plot(list(time), p_angle_y, label='p_angle_y')
#plt.plot(list(time), p_angle_z, label='p_angle_z', color=color[10])

#plt.plot(list(time), p_bond_z - (p_bond_x+p_bond_y)/2, label='p_bond_N - p_bond_L', color=color[0])
#plt.plot(list(time), p_angle_z - (p_angle_x+p_angle_y)/2, label='p_angle_N - p_angle_L', color=color[1])

#plt.plot(list(time), p_bond_z + p_angle_z - (p_bond_x+p_bond_y)/2 - (p_angle_x+p_angle_y)/2, label='p_(bond+angle)_N +  - p_(bond+angle)_L', color=color[9])

#plt.plot(list(time), (p_tot_x+p_tot_y)/2, label='Avg p_total in x,y')
#plt.plot(list(time), p_tot_x, label='p_total in x')
#plt.plot(list(time), p_tot_y, label='p_total in y')
plt.plot(list(time), p_tot_z, label='p_total in z')
plt.legend()
plt.savefig("pressure.png")
plt.show()

