#Determine lower limits of barostat call frequencies by estimating
#the significance of the box- and position- rescaling factors.
import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
from decimal import Decimal
f = h5py.File(sys.argv[1],'r')
start_frame = int(sys.argv[2])
last_frame = int(sys.argv[3])
alpha_0 = float(sys.argv[4])
value = f['observables/pressure/value']
value = np.array(list(value))[start_frame:last_frame]
x_value = list(f['particles/all/box/edges/value'])[start_frame:last_frame]
time = list(f['observables/pressure/time'] )[start_frame:last_frame]
pressure = [
    p_kin, p0, p1,
    p2x, p2y, p2z,
    p_bond_x, p_bond_y, p_bond_z,
    p_angle_x, p_angle_y, p_angle_z,
    p_dihedral_x, p_dihedral_y, p_dihedral_z,
    p_tot_x, p_tot_y, p_tot_z
]  = [value[:,i] for i in range(len(value[0]))]
[PL, PN] = [0, 0]
PL = (pressure[-3] + pressure[-2])/2
PN = pressure[-1]

#scaling factor
time_step = 0.3
tau_p = 1
beta = 4.6 * 10**(-5) #bar^(-1) #isothermal compressibility of water
skip = 0; call = 0
eps_alpha = abs(1 - alpha_0)
print('eps_alpha:',eps_alpha)
alphaL = 1 - time_step / tau_p * beta * (1.0 - PL)
alphaN = 1 - time_step / tau_p * beta * (1.0 - PN)
eps_alphaL = abs(1-alphaL[:])
eps_alphaN = abs(1-alphaN[:])
for _ in range(len(eps_alphaL)):
    print('%.0E,%.0E'%(eps_alphaL[_],eps_alphaN[_]))
    if(eps_alphaL[_]>eps_alpha or eps_alphaN[_]>eps_alpha):
        print('Call barostat. Box x:', x_value[_])
        call += 1
    else:
        print('Skip barostat. Box x:', x_value[_])
        skip += 1

print('No. of skips:%i/%i'%(skip,len(eps_alphaL)))
print('No. of calls:%i/%i'%(call,len(eps_alphaL)))

#eps_L = abs(1 - alphaL[:])
#print('eps_alpha_L:',eps_L)
#eps_N = abs(1 - alphaN[:])
#print('eps_alpha_N:',eps_N)
#plt.ylim(-0.0001,0.0001)
#plt.plot(time, eps_L, label='eps_L')
#plt.plot(time, eps_N, label='eps_N')
#plt.legend()
#plt.show()

#c4=0;c5=0;c6=0; c7=0; c8=0; c9=0
#for eps_value_N in eps_N:
#    if eps_value_N > 10**(-4):
#        c4 += 1
#    elif eps_value_N > 10**(-5):
#        c5 += 1
#    elif eps_value_N > 10**(-6):
#        c6 += 1
#    elif eps_value_N > 10**(-7):
#        c7 += 1
#    elif eps_value_N > 10**(-8):
#        c8 += 1
#    elif eps_value_N > 10**(-9):
#        c9 += 1
#print('c4:',c4/len(eps))
#print('c5:',c5/len(eps))
#print('c6:',c6/len(eps))
#print('c7:',c7/len(eps))
#print('c8:',c8/len(eps))
#print('c9:',c9/len(eps))
#print('total:',(c4+c5+c6+c7+c8+c9)/len(eps))

#x = ['-4','-5','-6','-7','-8','-9']
#plt.xlabel('log(eps)')
#plt.ylabel('# counts (normalized to 1)')
#plt.bar(x,np.array([c4,c5,c6,c7,c8,c9])/len(eps), align='edge', width=1)
#plt.show()
