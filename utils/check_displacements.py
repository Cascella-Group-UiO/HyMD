import numpy as np
import sys
import h5py

"""
Currently, HyMD can have the error of sending random particles to the origin in specific frames.
This does not affect the MD but can affect post-MD analysis. To check if the trajectory suffers
from this error, run this util as:
    python3 utils/check_displacements.py sim.h5 all

sys.argv[1]: sim.h5: trajectory in h5md format
sys.argv[2:]:  i)  all : checks for displacements across all frames in the trajectory
              ii) 32 : checks in frames 32
             iii) 32 34 36 .... m: checks in frames 32, 34, 36, .... , mth frame
"""


flag = False
values = h5py.File(sys.argv[1], 'r')
steps = values['particles/all/position/step']
if(sys.argv[2] == 'all'):
    frames = list(range(0,len(steps)))
else:
    frames = list(sys.argv[2:])
    frames = list(map(int, frames))
    print(frames)
for frame in frames:
    positions = values['particles/all/position/value'][frame, :, :]
    displaced_index = []
    for i in range(len(positions)):
        if(positions[i][0] == 0.000):
            displaced_index.append(i)
    if(len(displaced_index)>0):
        flag = True
        print('--- Frame : ', frame, '---')
        print('Number of particles displaced to origin: ',len(displaced_index))
        print('Index of particles displaced to origin: ', displaced_index)

if(flag == False):
    print('Number of particles displaced origin is 0 in all frames')

