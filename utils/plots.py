import matplotlib
import matplotlib.pyplot as plt
import fileinput
import numpy as np
import math
from argparse import ArgumentParser
import os
import h5py
from scipy.optimize import curve_fit
import warnings

def PLOT_aVSdensity(parameter, a, folder_path, no_show, out, ax):
    '''
    folder_path:   path to folder containing parameter (eg:sigma) and a valued folders.
                   Example: /Users/samiransen23/aparametrization
                   >ls /Users/samiransen23/aparametrization
                   sigma=0.238 sigma=0.338 sigma=0.438
                   >ls /Users/samiransen23/aparametrization/sigma\=0.238
                   a=9.25 a=9.30 a=9.35
                   a=9.25 should contain the trajectory sim.h5
    '''
    if len(parameter) == 1:
        p1 = []
        for f in os.listdir(folder_path):
            if f.startswith(parameter[0]):
                p1.append(str(f.split(parameter[0]+'=')[1]))
        dp = len(p1[0].split('.')[-1])
        p1 = list(map(float, p1))
        p1.sort()
        form = "{:."+str(dp)+"f}"
        parameter.extend([form.format(val) for val in list(map(float, p1))])
    if a == 'all':
        p1 = []
        for f in os.listdir(folder_path+"/"+parameter[0]+"="+parameter[1]):
            p1.append(str(f.split('a=')[1]))
        dp = len(p1[0].split('.')[-1])
        p1 = list(map(float, p1))
        p1.sort()
        form = "{:."+str(dp)+"f}"
        a = [form.format(val) for val in list(map(float, p1))]
    V = np.zeros((len(parameter) - 1,len(a)))
    density = np.zeros((len(parameter) - 1,len(a)))
    for ii in range(len(parameter)-1):
        for i in range(len(a)):
            #file_path = os.path.abspath("/Users/samiransen23/hymdtest/test_aparametrization/a="+a[i]+"/sim.h5")
            #h5md_file = open_h5md_file(file_path)
            file_path = os.path.abspath(folder_path+"/"+parameter[0]+"="+parameter[ii+1]+"/a="+a[i]+"/sim.h5")
            f = h5py.File(file_path, "r")
            N = len(list(f["particles/all/species"]))
            try:
                V[ii,i] = np.prod(f["particles/all/box/edges/value"][-1].diagonal())
            except:
                V[ii,i] = np.prod(f["particles/all/box/edges/value"][-1])
            density[ii,i] = N/V[ii,i] * 0.11955 #gm/cc
            f.close()

    ##PLOTS
    #fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    ax1 = ax[0]
    ax2 = ax[1]
    fontsize=36

    #a vs density for different parameter values
    a = [float(i) for i in a]
    ax1.tick_params(axis='both', labelsize=fontsize-2)
    ax1.set_xlabel(r'$a \, \mathrm{(nm^{-3})}$', fontsize=fontsize+2)
    ax1.set_ylabel("Density"+ r"$\, \mathrm{(g \, cm^{-3} )}$", fontsize=fontsize+2)
    ax1.axhline(y=1, color='0.8', linestyle='--')
    for ii in range(len(parameter)-1):
        ax1.plot(a, density[ii], marker='.',
                 linewidth=2.6, label='$\%s=%s$'%(parameter[0], parameter[ii+1]))
    ax1.legend(loc='center right', bbox_to_anchor=(-0.26, 0.5), fontsize=fontsize+2)

    #parameter vs a
    p = [float(i) for i in parameter[1:]]
    a_star = np.zeros(len(parameter) -1)
    for ii in range(len(parameter)-1):
        m, c = np.polyfit(a, density[ii], deg=1)
        a_star[ii] = (1 - c)/m
    if (parameter[0]=='kappa'):
        ax2.set_xlabel('$\%s$'%(parameter[0] + "\, \mathrm{(mol \, kJ^{-1})}"),
                fontsize=fontsize+2)
    elif (parameter[0]=='sigma'):
        ax2.set_xlabel('$\%s$'%(parameter[0] + "\, \mathrm{(nm)}"),
                fontsize=fontsize+2)
    else:
        ax2.set_xlabel(parameter[0], fontsize=fontsize+2)
    ax2.tick_params(axis='both', labelsize=fontsize-2)
    ax2.set_ylabel(r'$a^* \, \mathrm{(nm^{-3})}$', fontsize=fontsize+2)
    ax2.plot(p, a_star, marker='o',
            linewidth=2.6)

    if no_show:
        if out:
            out = os.path.abspath(out)
        else:
            out = os.getcwd()+'/fig_temp.pdf'
        plt.savefig(out, format='pdf')
    else:
        plt.show()
    

def PLOT_mVSvolume(m, folder_path):
    '''
    This function plots a list of chosen density factors m versus
    the equilibrium volume under constant ext pressure.
    folder_path:   path to folder containing m valued folders.
                   Example: /Users/samiransen23/out
                   >ls /Users/samiransen23/out
                   m=1.0 m=1.1 m=1.2 m=1.3
                   m=1.0 should contain the trajectory sim.h5
    '''
    volume = np.zeros(len(m))
    density = np.zeros(len(m))
    areapl = np.zeros(len(m))
    numperleaf = 264
    for ii in range(len(m)):
        file_path = os.path.abspath(folder_path+"/m="+m[ii]+"/sim.h5")
        f = h5py.File(file_path, "r")
        volume[ii] = np.prod(f["particles/all/box/edges/value"][-1])
        N = len(list(f["particles/all/species"]))
        density[ii] = N/volume[ii] * 0.11955 #gm/cc
        areapl[ii] = np.prod(f["particles/all/box/edges/value"][-1][0:2])/numperleaf
        f.close()

    ##PLOTS
    [float(i) for i in m]
    plt.xlabel("m")
    plt.ylabel("volume (nm^3)")
    plt.plot(m, volume, marker='o', label="m vs volume")
    plt.legend()
    plt.show()
    plt.xlabel("m")
    plt.ylabel("density (gm/cm^3)")
    plt.plot(m, density, marker='o', label="m vs density")
    plt.legend()
    plt.show()
    plt.xlabel("m")
    plt.ylabel("area per lipid (nm^2)")
    plt.plot(m, areapl, marker='o', label="m vs area per lipid")
    plt.legend()
    plt.show()

def PLOT_tVSbox(trajectories, first_frame, last_frame, no_show, out, extract_data,
        no_density=False, label=None, styles=None):

    def PLOT_single(trajectory, first_frame, last_frame, no_show, out, extract_data,
            fontsize, ax, plot2dlist, c_iter, no_density=False, label=None, style=None):
        f = h5py.File(trajectory, "r")
        box_value = list(f["particles/all/box/edges/value"])
        if last_frame<0:
            last_frame = len(list(f['observables/pressure/value']))+last_frame+1
        
        try:
            x_value = [_[0][0] for _ in box_value[first_frame:last_frame]]
            y_value = [_[1][1] for _ in box_value[first_frame:last_frame]]
            z_value = [_[2][2] for _ in box_value[first_frame:last_frame]]
            time = list(f["particles/all/box/edges/time"])[first_frame:last_frame]
        except:
            #before commit 36330cec518a7f2f5ee6bcdcebb9e32c6d6b3d93
            x_value = [_[0] for _ in list(f["particles/all/box/edges/value"])[first_frame:last_frame]]
            y_value = [_[1] for _ in list(f["particles/all/box/edges/value"])[first_frame:last_frame]]
            z_value = [_[2] for _ in list(f["particles/all/box/edges/value"])[first_frame:last_frame]]
            time = list(f["particles/all/box/edges/time"])[first_frame:last_frame]
        
        volume = [x_value[i] * y_value[i] * z_value[i] for i in range(len(x_value))]
        N = len(list(f["particles/all/species"]))
        f.close()
    
        density = [0.11955 * N / _ for _ in volume] #gm/cc
        stillbox=[]
        for i in range(len(x_value)-1):
            if(x_value[i] == x_value[i+1]):
                stillbox.append("(%s)"%(str(i)))
        average_final_box_size = [np.average(x_value[round(0.75*(last_frame-first_frame)):last_frame]),
                np.average(y_value[round(0.75*(last_frame-first_frame)):last_frame]),
                np.average(z_value[round(0.75*(last_frame-first_frame)):last_frame])]
        figtext = "Average final box_size: {0:.3f}, {1:.3f}, {2:.3f}".format(*average_final_box_size)
        ##PLOTS
        if not no_density:
            ax[1].plot(time, density)
            ax[1].annotate(figtext,
                    xy = (1.4, 24), xycoords='figure points', fontsize=11)
        if any(val > 9000 for val in time):
            time = [val / 1000 for val in time] #ps --> ns
            try:
                ax[0].set_xlabel("Time (ns)", fontsize=fontsize)
            except:
                ax.set_xlabel("Time (ns)", fontsize=fontsize)
        if not None in styles:
            if 'scr' in style.lower():
                cx = cz = '#808080' #gray
                ls = 'solid'; lw = 2.0
            else:
                ls = 'solid'; lw = 3.4
                if 'nmlberendsen' in style.lower():
                    cx = cz = '#002dcd' #bright blue
                elif 'omlberendsen' in style.lower():
                    cx = cz = '#da0e00' #bright red
                else:
                    cx = next(c_iter)
                    cz = next(c_iter)
        else:
            ls = 'solid'; lw = 3.4; cx = next(c_iter); cz = next(c_iter)

        try:
            ax[0].plot(time, x_value, label=label[0], color=cx)
            ax[0].plot(time, z_value, label=label[1], color=cz)
        except:
            plot2dlist.append( ax.plot(time, x_value, label=label[0], color=cx, ls=ls, lw=lw,) )
            plot2dlist.append( ax.plot(time, z_value, label=label[1], color=cz, ls=ls, lw=lw,) )
    
        print('File: ',trajectory)
        print('Last frame:', last_frame, '; box_size: [',x_value[-1],' ',y_value[-1],' ',z_value[-1],']')
        print(figtext)
        print("Area per lipid for 264/layer:", average_final_box_size[0]*average_final_box_size[1]/264)
        if no_show:
            if extract_data:
                index = trajectory.split('constA_f')[1].split('/')[0]
                with open(extract_data,'a') as f:
                    f.write("{ind}\t{a0:.2f}\t{a1:.2f}\t{a2:.2f}\n".format(ind=index, 
                        a0=average_final_box_size[0], a1=average_final_box_size[1], a2=average_final_box_size[2]))
        else:
            if(stillbox): print("unchanged box frames:",stillbox)
        print('---------')

    #PLOT SETUP
    #using matplotlibstyle: /Users/samiransen23/anaconda3/envs/py38/lib/python3.8/site-packages/matplotlib/mpl-data/stylelib/samiran-signature.mplstyle
    c = {'blue':'#001C7F','green':'#017517','golden':'#B8860B','purple':'#633974',
            'red':'#8C0900','teal':'#006374','brown':'#573D1C','orange':'#DC901D',
            'bright red': '#da0e00'}
    matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler('color',c.values())
    c_iter = iter(matplotlib.rcParams['axes.prop_cycle'].by_key()['color'])

    fontsize = 30
    if no_density:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        ax.set_xlabel("Time (ps)", fontsize=fontsize+2)
        ax.set_ylabel("Length of box (nm)", fontsize=fontsize+2)
        ax.tick_params(axis='both', labelsize=fontsize-2)
        ax.set_ylim(11.5,18.5)
    else:
        fig, ax = plt.subplots(2)
        ax[0].set_ylabel("Length of box (nm)", fontsize=fontsize)
        ax[1].set_xlabel("Time (ps)", fontsize=fontsize)
        ax[1].set_ylabel('Density (gm/cc)', fontsize=fontsize)

    if not label:
        label = np.array([None, None])

    plot2dlist = []
    if styles is None:
        styles = [None for _ in trajectories]
    for trajectory, style in zip(trajectories, styles):
        PLOT_single(trajectory, first_frame, last_frame, no_show, out, extract_data,
                fontsize, ax, plot2dlist, c_iter, no_density, label, style,)
    try:
        ax[0].legend(fontsize=fontsize)
    except:
        ax.legend(["1", "2", "3", "4", "5"], fontsize=fontsize, loc='center left', bbox_to_anchor=(1, 0.5))
        #ax.legend(fontsize=fontsize, loc='center left', bbox_to_anchor=(1, 0.5))
    if no_show:
        if out:
            out = os.path.abspath(out)
        else:
            out = os.getcwd()+'/fig_temp.pdf'
        plt.tight_layout()
        plt.savefig(out, format='pdf')
        print('Saving plot to: ',out)
    else:
        plt.show()


def PLOT_tVSpressure(trajectory, first_frame, last_frame, no_show=False):
    if len(trajectory) == 1:
        trajectory = trajectory[0]
    f = h5py.File(trajectory, "r")
    value = f['observables/pressure/value']
    if last_frame<0:
        last_frame = len(list(f['observables/pressure/value']))+last_frame
    time = list(f['observables/pressure/time'] )[first_frame:last_frame]
    box_z = [_[2] for _ in list(f["particles/all/box/edges/value"])[first_frame:last_frame]]
    
    value = np.array(list(value))[first_frame:last_frame]
    f.close()
    if(len(value[0])==9):
        pr = [
            p_kin, p0, p1,
            p2x, p2y, p2z,
            p_tot_x, p_tot_y, p_tot_z
        ]  = [value[:,i] for i in range(len(value[0]))]
    elif( len(value[0])==18 ):
        pr = [
            p_kin, p0, p1,
            p2x, p2y, p2z,
            p_bond_x, p_bond_y, p_bond_z,
            p_angle_x, p_angle_y, p_angle_z,
            p_dihedral_x, p_dihedral_y, p_dihedral_z,
            p_tot_x, p_tot_y, p_tot_z
        ]  = [value[:,i] for i in range(len(value[0]))]
    elif( len(value[0])==22 ):
        pr = [
            p_kin, p0, p1,
            p2x, p2y, p2z,
            p_w1_0,
            p_w1_x, p_w1_y, p_w1_z,
            p_bond_x, p_bond_y, p_bond_z,
            p_angle_x, p_angle_y, p_angle_z,
            p_dihedral_x, p_dihedral_y, p_dihedral_z,
            p_tot_x, p_tot_y, p_tot_z
        ]  = [value[:,i] for i in range(len(value[0]))]
    elif( len(value[0])==25 ):
        pr = [
            p_kin, p0, p1,
            p2x, p2y, p2z,
            p_w1_0,
            p_w1_1_x, p_w1_2_x, p_w1_1_y, p_w1_2_y, p_w1_1_z, p_w1_2_z,
            p_bond_x, p_bond_y, p_bond_z,
            p_angle_x, p_angle_y, p_angle_z,
            p_dihedral_x, p_dihedral_y, p_dihedral_z,
            p_tot_x, p_tot_y, p_tot_z
        ]  = [value[:,i] for i in range(len(value[0]))]

    #AVERAGE VALUE PRINTS
    p_avg = [ np.average(val[round(0.25*(last_frame-first_frame)):last_frame]) for val in [p_tot_x, p_tot_y, p_tot_z] ]
    box_z_avg = np.average( box_z[round(0.75*(last_frame-first_frame)):last_frame] )
    factor = 0.1 * 16.6 # 1nm kJ/(mol nm^3) = 1nm × 16.6 bar = 16.6 × 10⁻⁹ m × 10⁸ mN/m^2 = 0.1 × 16.6 mN/m
    st = 0.5 * box_z_avg * (p_avg[2] - (p_avg[0]+p_avg[1])/2 ) * factor
    p_avg = [val*16.6 for val in p_avg] # 1 kJ/(mol nm^3) = 16.6 bar
    print("p_tot_x: {0:.3f}, p_tot_y: {1:.3f}, p_tot_z: {2:.3f} {3:s}".format(*p_avg, "bar"))
    #print("p_tot_x: {0:.3f}, p_tot_y: {1:.3f}, p_tot_z: {2:.3f} {3:s}".format(*p_avg, "kJ mol^(-1) nm^(-3)"))
    print('Avg final surface tension: ',st, '(mN / m)')
    
    #PLOTS
    color = ['b','g','r','c','m','y','k','brown','gray','orange','purple']
    plt.xlabel('Time (ps)')
    #plt.plot(list(time), p_kin, label='p_kin', color=color[0])
    #plt.plot(list(time), p0, label='p0', color=color[1])
    #plt.plot(list(time), p1, label='p1', color=color[2])
    
    #plt.plot(list(time), (p2x + p2y)/2, label='p2 in x,y', color=color[3])
    #plt.plot(list(time), p2x, label='p2x', color=color[3])
    #plt.plot(list(time), p2y, label='p2y', color=color[4])
    #plt.plot(list(time), p2z, label='p2z', color=color[5])
    
    if len(value[0])==25:
        #plt.plot(list(time), p_w1_0, label='p_w1_0', color=color[6])
        #plt.plot(list(time), (p_w1_1_x + p_w1_1_y)/2, label='p_w1_1 in x,y', color=color[0])
        #plt.plot(list(time), (p_w1_2_x + p_w1_2_y)/2, label='p_w1_2 in x,y', color=color[1])
        #plt.plot(list(time), p_w1_1_z, label='p_w1_1 in z', color=color[2])
        #plt.plot(list(time), p_w1_2_z, label='p_w1_2 in z', color=color[3])
        pass
    elif len(value[0])==22:
        #plt.plot(list(time), p_w1_0, label='p_w1_0', color=color[6])
        #plt.plot(list(time), (p_w1_x + p_w1_y)/2 , label='p_w1 in x,y', color=color[7])
        #plt.plot(list(time), p_w1_z, label='p_w1_z', color=color[9])
        pass
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
    
    #plt.plot(list(time), (p_tot_x+p_tot_y)/2, label='p_total in x,y')
    #plt.plot(list(time), p_tot_z, label='p_total in z')
    plt.plot(list(time), (p_tot_x+p_tot_y)/2*16.6, label='p_total in x,y (bar)')
    plt.plot(list(time), p_tot_z*16.6, label='p_total in z (bar)')
    plt.legend(loc='upper center', ncol = 3, fontsize = 'small')
    if no_show:
        plt.savefig('tVSp.pdf', format='pdf')
    else:
        plt.show()

def PLOT_areaVSsurfacetension(trajectory, first_frame, last_frame, no_show, out,
        extract_data, index=1):
    f = h5py.File(trajectory, "r")
    pressure = f['observables/pressure/value']
    if last_frame<0:
        last_frame = len(list(pressure))+last_frame
    time = list(f['observables/pressure/time'] )[first_frame:last_frame]
    box_value = list(f["particles/all/box/edges/value"])
    pressure = np.array(list(pressure))[first_frame:last_frame]
    f.close()
    
    numperleaf = 264
    try:
        x_value = [_[0][0] for _ in box_value[first_frame:last_frame]]
        y_value = [_[1][1] for _ in box_value[first_frame:last_frame]]
        z_value = [_[2][2] for _ in box_value[first_frame:last_frame]]
    except:
        #before commit 36330cec518a7f2f5ee6bcdcebb9e32c6d6b3d93
        x_value = [_[0] for _ in box_value[first_frame:last_frame]]
        y_value = [_[1] for _ in box_value[first_frame:last_frame]]
        z_value = [_[2] for _ in box_value[first_frame:last_frame]]

    [
        p_tot_x, p_tot_y, p_tot_z
        ]  = [pressure[:,i] for i in range(-3, 0)]
    p_N = p_tot_z
    p_L = ( p_tot_x + p_tot_y )/2
    #gamma: surface tension per interface
    factor = 0.1 * 16.6 # 1nm kJ/(mol nm^3) = 1nm × 16.6 bar = 16.6 × 10⁻⁹ m × 10⁸ mN/m^2 = 0.1 × 16.6 mN/m
    gamma = [0.5 * z_value[i] * (p_N[i] - p_L[i]) * factor for i in range(len(z_value))]
    #A: area per lipid
    A = [   x_value[_] * y_value[_] / numperleaf for _ in range(len(x_value)) ]
    #PLOTS
    ax = plt.figure().add_subplot(111)
    plt.xlabel('Area per lipid ( nm^2 )')
    plt.ylabel('Surface tension ( mN/m )')
    plt.scatter(A, gamma)
    #fit_straight(A, gamma, ax)
    if no_show:
        if out:
            out = os.path.abspath(out)
        else:
            out = os.getcwd()+'fig_temp.pdf'
        plt.savefig(out, format='pdf')
        if extract_data:
            print('Writing data from:',trajectory)
            with open(extract_data,'a') as f:
                f.write("{0:d}\t{1:.3f}\t{2:.3f}\t{3:.3f}\n".format(index,np.average(A), np.average(gamma), np.std(gamma)))
    else:
        #PRINTS
        print('%s\t%s%.3f%s'%('Avg area per lipid', r'A=', np.average(A), r' (nm^2)'))
        print('%s\t%s%.3f%s'%('Avg surface tension', r'<γ>=', np.average(gamma), r' (mN/m)'))
        print('%s%s\t%.3f%s'%('Std deviation of ', r'γ', np.std(gamma),  r' (mN/m)'))
        plt.show()

def fit_straight(x, y, ax, ind, col):
    y_coord = [0.2, 0.1]
    label = ['Fit set 1', 'Fit set 2']
    #col = ['#da0e00','#002dcd']
    m, c  = np.polyfit( x, y, deg=1)
    line = [(m * xval + c) for xval in x]
    figtext=""
    #print('hardcoded 0.64 multiplying factor')
    #figtext = '%s%d%s%.3f'%(r'$m$',ind,':', m)
    #text = 'm=%.2f c=%.2f\n %s'%(m, c, figtext)
    #ax.text(0.6, 0.9, text , ha='center', va='center', transform = ax.transAxes)
    #ax.vlines(-c/m, plt.ylim()[0], 0, linestyle = 'dotted', color = 'y')
    ax.plot(x, line, linewidth = 1.6, linestyle = '--',
            color = col, label=label[ind])
    #ax.text(0.95, y_coord[ind], figtext,
    #    verticalalignment='bottom', horizontalalignment='right',
    #    transform=ax.transAxes,
    #    fontsize=14)
    print('Slope for ',label[ind],': ',m)
    return m,c

def fit_sigmoid(x, y, ax, species):
    def sigmoid(x, a, b, y_m):
        return y_m / (1.0 + np.exp(-a*(x-b)))
    start_x = 0
    end_x = 10
    start_index = np.where(np.round(x,0) == start_x)[0][0]
    end_index = np.where(np.round(x,0) == end_x)[0][-1]
    if species == 'C':
        y_m_guess = np.average(y[(end_index - 2) : end_index])
    elif species == 'W':
        y_m_guess = np.average(y[start_index : (start_index+2)])
    b_guess = x[int((start_index + end_index) / 2)]
    popt, pcov = curve_fit(sigmoid, x[start_index : end_index], y[start_index : end_index], p0 = [1, b_guess, y_m_guess])
    print('popt: a=',popt[0],'; b=',popt[1],'; y_m=',popt[2])
    xfit = np.linspace(x[start_index] , x[end_index], 100)
    yfit = sigmoid( xfit , popt[0], popt[1], popt[2]) 
    ax.plot(xfit, yfit, label='fit')
    #ax.plot( x[start_index : end_index] , yfit )
    #write fitted data to file
    with open('fit_%s.dat'%(species),'w') as f:
        [ f.write(str(xfit[i])+' '+str(yfit[i])+'\n') for i in range(len(xfit)) ]

def fit_fromfile(ax, file, own_scale, ind):
    #c = ['#5b5b5b','#375bcf']
    c = ['#002dcd', '#B8860B'] 
    label = ['$hPF_\mathrm{lit}$','$UA_\mathrm{ref}$',]
    linestyle = ["dashed","dotted"]
    with open(file, 'r') as f:
        data = f.read()
    data = data.split('\n')[0:-1]
    index = []; x = []; y = []
    pr_unit = data[0].split()[1].split('(')[1].split(')')[0] 
    for line in data[1:]:
        f_list = [float(i) for i in line.split()]
        x.append(f_list[0])
        y.append(f_list[1])
    if own_scale=='True':
        ax2 = ax.twinx()
        ax2.set_ylabel('('+pr_unit+')', color='#006374')
        ax2.scatter(x, y, marker='.', color='#006374', s=8.2, label='fit')
    else:
        if ind == 0:
            ax.plot(x, y, linestyle=linestyle[ind], linewidth=2.7, color=c[ind],  label=label[ind])
        else:
            ax.scatter(x, y, marker='.', color=c[ind], s=36, label=label[ind])
    



def PLOT_pressure_profile(trajectory, first_frame, last_frame, path, terms, box_scale, config, own_scale,
        dir='z', no_show=False, out=None, sqgrad=False, labels=None, axis=None, fit=None,
        legend_loc = None, yaxis=None,):
    """
    Parameters
    ----------
    terms: list, str
        Names of pressure terms to plot:
            'p_kin', 'p0', 'p1', etc. as named in list pr_names;
            Other possible terms:
            'diff-field': plots the difference between normal and lateral components of field pressure
            'diff-bond': plots the difference between normal and lateral components of bonded pressure
            'diff-angle': plots the difference between normal and lateral components of angle (bonded) pressure
            'diff-total': plots the difference between normal and lateral components of total pressure
            'all':  plots all terms in list pr_names
    """
    def plot(box_avg, pr_avg, no_show, sqgrad, terms, box_scale, yaxis, label='',):
        #units
        #pr_avg is in kJ/(mol nm^3)
        #to plot pr in bar:
        #1 kJ/(mol nm^3) = 16.61 bar
        pr_avg = 16.61 * pr_avg
        pr_unit = 'bar'
        #make pressure dictionary
        pr_names = ['p_kin','p0','p1',
                'p2x','p2y','p2z',
                'pW1_0','pW1_1x','pW1_2x',
                'pW1_1y','pW1_2y', 'pW1_1z','pW1_2z',
                'p_bond_x', 'p_bond_y', 'p_bond_z',
                'p_angle_x', 'p_angle_y', 'p_angle_z',
                'p_dihedral_x', 'p_dihedral_y', 'p_dihedral_z',
                'p_tot_x', 'p_tot_y', 'p_tot_z'
                ]
        pr_dict = {}
        terms_individual = []
        terms_derived = []
        for i in range(25):
            pr_dict[pr_names[i]] = pr_avg[i]
        for term in terms:
            if term in pr_dict:
                terms_individual.append(term)
            else:
                terms_derived.append(term)
        if 'all' in terms_derived:
            terms_individual = pr_names

        #SIMPLE CALCULATIONS TO PLOT
        if sqgrad:
            pr_iso = np.sum(pr_avg[0:3], axis=0) + pr_avg[6]
            pr_field_N = pr_iso + pr_avg[5] + pr_avg[11] + pr_avg[12]
            pr_field_L = pr_iso + (pr_avg[3] + pr_avg[4])/2 + np.sum(pr_avg[7:11], axis=0)/2
        else:
            pr_iso = np.sum(pr_avg[0:3], axis=0)
            pr_field_N = pr_iso + pr_avg[5]
            pr_field_L = pr_iso + (pr_avg[3] + pr_avg[4])/2

        #PLOTS
        #using matplotlibstyle: /Users/samiransen23/anaconda3/envs/py38/lib/python3.8/site-packages/matplotlib/mpl-data/stylelib/samiran-signature.mplstyle
        c = {'blue':'#001C7F',
           'green':'#017517',
           'golden':'#B8860B',
           'purple':'#633974',
           'red':'#8C0900',
           'teal':'#006374',
           'brown':'#573D1C',
          'orange':'#DC901D'}
        matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler('color',c.values())
        
        if box_scale is not None:
            xaxis = np.linspace(-int(box_scale)/2, int(box_scale)/2, len(pr_avg[0]))
        else:
            xaxis = np.linspace(-box_avg[2]/2, box_avg[2]/2, len(pr_avg[0]))
        if yaxis:
            ax.set_ylim([float(yaxis[0]), float(yaxis[1])])

        for term in terms_individual:
            ax.plot(xaxis, pr_dict[term], label=term)
        #ax.plot(xaxis, pr_avg[1], color=c['darkblue'], label='p0')
        #ax.plot(xaxis, pr_avg[2], color=c['B'], label='p1')
        #ax.plot(xaxis, pr_avg[3], color='r', label='p2x')
        #ax.plot(xaxis, pr_avg[5], color=c['C'], label='p2z')
        #ax.plot(xaxis, pr_avg[6], color=c['D'], label='pW1_0')
        #ax.plot(xaxis, pr_z_sum[7]+pr_z_sum[8], label='pW1_1x+pW1_2x')
        #ax.plot(xaxis, pr_z_sum[9]+pr_z_sum[10], label='pW1_1y+pW1_2y')
        #ax.plot(xaxis, pr_avg[11]+pr_avg[12], color=c['E'], label='pW1_1z+pW1_2z')
        #plt.legend()
        #plt.show()
        #ax = plt.figure().add_subplot(111)
        #ax.plot(xaxis, (pr_avg[13]+pr_avg[14])/2, color=c['blue'], label=r'$P_{x,y}^{bond}$')
        #ax.plot(xaxis, pr_avg[15], color=c['orange'], label=r'$P_{z}^{bond}$')
        #ax.plot(xaxis, (pr_avg[16]+pr_avg[17])/2, color=c['blue'], label=r'$P_{x,y}^{angle}$')
        #ax.plot(xaxis, pr_avg[18], color=c['orange'], label=r'$P_{z}^{angle}$')
        #ax.plot(xaxis, (pr_avg[22]+pr_avg[23])/2, color=c['blue'], label='Total p in x,y') 
        #ax.plot(xaxis, pr_avg[24], c['orange'], label='Total p in z') 
        for term in terms_derived:
            if term=='diff-field':
                ax.plot(xaxis, pr_field_N - pr_field_L, color=c['blue'], linewidth=3.2,
                        #label=r'$(P_{N}^{field} - P_{L}^{field})$ '+label)
                        label = r'$Δ P_{\mathrm{field}}$ ')
            if term=='diff-bond':
                ax.plot(xaxis, pr_avg[15] - (pr_avg[13]+pr_avg[14])/2, color=c['green'], linewidth=3.2,
                        #label=r'$(P_{N}^{bond} - P_{L}^{bond})$')
                        label = r'$Δ P_{\mathrm{bond}}$ ')
            if term=='diff-angle':
                ax.plot(xaxis, pr_avg[18] - (pr_avg[16]+pr_avg[17])/2, color=c['golden'], linewidth=3.2,
                        #label=r'$(P_{N}^{angle} - P_{L}^{angle})$')
                        label = r'$Δ P_{\mathrm{angle}}$ ')
            if term=='diff-total':
                label = 'HhPF' if fit else r'$\Delta P_{\mathrm{tot}}$'
                ax.plot(xaxis, pr_avg[24] - (pr_avg[22]+pr_avg[23])/2, 
                        color = '#da0e00', #bright red
                        #color=c['red'],
                        lw=3.2, label=label)
            if term=='pr_field_L':
                ax.plot(xaxis, pr_field_L, #color=c['blue'],
                        linestyle='dashed', label=r'$P_{L}^{field}$'+label)
            if term=='pr_field_N':
                ax.plot(xaxis, pr_field_N, #color=c['orange'],
                        linestyle='dashed', label=r'$P_N^{field}$'+label)
        ax.tick_params(axis='both', labelsize=fontsize-2)
        if fit:
            ax.set_xlabel('Position along normal '+r"$\mathrm{(nm)}$", fontsize=fontsize)
        else:
            ax.set_xlabel('Position along normal '+r"$\mathrm{(nm)}$", fontsize=fontsize)
        if(len(terms)==1):
            if(terms[0]=='diff-total'):
                ax.set_ylabel(r'$\Delta P_{\mathrm{tot}}$'+ r'$\;\mathrm{( %s )}$'%(str(pr_unit)), fontsize=fontsize)
            else:
                ax.set_ylabel(terms[0]+ ' \mathrm{( '+ pr_unit + ' )}', fontsize=fontsize)
        else:
            ax.set_ylabel(r'$\Delta P \;\mathrm{( '+ pr_unit + ' )}$', fontsize=fontsize)

    def pr_profile_single(trajectory, first_frame, last_frame,
            sqgrad, dir='z', config=None):
        if dir == 'z':
            dir = 2
        f = h5py.File(trajectory, "r")
        pressure = f['observables/pressure/value']
        if last_frame<0:
            last_frame = len(list(pressure))+last_frame
        box_value = list(f["particles/all/box/edges/value"])
        pressure = np.array(list(pressure))[first_frame:last_frame+1]
        f.close()

        #selecting first frame only
        pressure = pressure[0]
        pr_avg = np.zeros([25,pressure.shape[1]])
        box_value = [box_value[0][i][i] for i in range(3)]
        #sum over all x,y
        #NON-BONDED TERMS
        for i in range(13):
            pr_avg[i] = np.sum(pressure[i], axis=(0,1))
        #BONDED TERMS
        for i in range(13,22):
            pr_avg[i] = pressure[i][0,0,:]
        #TOTAL PRESSURE
        if sqgrad:
            pr_iso = np.sum(pr_avg[0:3], axis=0) + pr_avg[6]
            x_ind = [3,7,8, 13,16,19]
            y_ind = [4,9,10, 14,17,20]
            z_ind = [5,11,12, 15,18,21]
        else:
            pr_iso = np.sum(pr_avg[0:3], axis=0)
            x_ind = [3,13,16,19]
            y_ind = [4,14,17,20]
            z_ind = [5,15,18,21]
        pr_avg[22] += pr_iso 
        pr_avg[23] += pr_iso 
        pr_avg[24] += pr_iso 
        for i,j,k in zip(x_ind,y_ind,z_ind):
            pr_avg[22] += pr_avg[i]
            pr_avg[23] += pr_avg[j]
            pr_avg[24] += pr_avg[k]

        #Normalization when slicing z (to maintain intensiveness)
        if config:
            import toml
            config = toml.load(config)
            N_z = config['field']['mesh_size'][dir]
        else:
            warnings.warn(
                r"No config file provided. Grid points could not "\
                "be determined",
                )

        pr_avg = N_z * pr_avg

        return pr_avg, box_value

    fontsize = 32
    if path:
        paths=[]
        if isinstance(path, str):
            paths.append(path)
        else:
            paths = path
        #samiran_signature = plt.style.use('samiran-signature')
        if not axis:
            fig = plt.figure(figsize=(10,7))
            ax = fig.add_subplot(111)
        pr_trajs = []
        box_trajs = []
        for path in paths:
            extracted_frames = [int(f.split('pressure')[1].split('.h5')[0])
                for f in os.listdir(path)
                ]
            pr_traj = []
            box_traj = []
            for val in extracted_frames:
                trajectory = os.path.join(os.path.abspath(path),'pressure'+str(val)+'.h5')
                pr_avg, box_value = pr_profile_single(trajectory, first_frame, last_frame, sqgrad, config=config)
                pr_traj.append(pr_avg)
                box_traj.append(box_value)
            pr_trajs.append(np.average( np.array(pr_traj), axis=0 ))
            box_trajs.append(np.average( np.array(box_traj), axis=0 ))
        for i in range(len(paths)):
            plot(box_trajs[i], pr_trajs[i], no_show=no_show, sqgrad=sqgrad, terms=terms,
                    box_scale=box_scale, yaxis=yaxis,)
    elif trajectory:
        pr_avg, box_value = pr_profile_single(trajectory, first_frame, last_frame, sqgrad)
        plot(box_value, pr_avg, no_show=no_show, sqgrad=sqgrad, terms=terms,
                box_scale=box_scale)

    if fit:
        fits = []
        if isinstance(fit, str):
            fits.append(fit)
        else:
            fits = fit
        for fit,own_scale_val in zip(fits,own_scale):
            fit_fromfile(ax, fit, own_scale_val, fits.index(fit))
    if fit:
        lgnd = fig.legend(loc=legend_loc, fontsize=fontsize-2, markerscale = 4)
    else:
        lgnd = fig.legend(loc=legend_loc, fontsize=fontsize, markerscale = 4)
    plt.tight_layout()
    if no_show:
        if out:
            out = os.path.abspath(out)
        else:
            out = os.getcwd()+'fig_temp.pdf'
        plt.savefig(out, format='pdf')
    else:
        plt.show()

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--traj", default=None, nargs='+', help="sim.h5")
    ap.add_argument("--first", type=int, default=0, help="first frame for plot. Eg: 0")
    ap.add_argument("--last", type=int, default=-1, help="last frame for plot. Eg: -1")
    ap.add_argument("--config", default=None, help="config.toml")
    ap.add_argument("--path", default=None, help="path of folder containing folders/files as reqd. See inside function")
    ap.add_argument("--plot_aVSdensity", action='store_true')
    ap.add_argument("--plot_mVSvolume", action='store_true')
    ap.add_argument("--plot_tVSpressure", action='store_true')
    ap.add_argument("--plot_tVSbox", action='store_true')
    ap.add_argument("--plot_areaVSsurfacetension", action='store_true')
    ap.add_argument("--plot_pressure_profile", action='store_true')
    ap.add_argument("--no_show", action='store_true', help='Do not show plot. Save figure instead')
    ap.add_argument("--no_density", action='store_true', help='Do not show time vs density plot.')
    ap.add_argument("--parameter", type=str, nargs="+", default=None, help="If parameter is sigma: sigma 0.238 0.338 0.438")
    ap.add_argument("--a", type=str, nargs="+", default='all', help="9.25 9.30 9.35")
    ap.add_argument("--m", type=str, nargs="+", default=None, help="1.0 1.1 1.2")
    ap.add_argument("--out", default=None, help="pathname of figure to be saved")
    ap.add_argument("--sqgrad", action='store_true', help='include sq grad pressure terms in plot')
    ap.add_argument("--terms", type=str, nargs="+", default='all', help="See description of PLOT_pressure_profile")
    ap.add_argument("--yaxis", type=str, nargs="+", default=None, help="-600 400")
    ap.add_argument("--box_scale", type=float, default=None, help='plot pr profiles with box length in z scaled to this value')
    ap.add_argument("--own_scale", nargs="+", default='False', help='plot the fitting fn(s) with its own vertical axis')
    ap.add_argument("--extract_data", type=str, default=None, help='save average final box in a file in this path')
    ap.add_argument("--fit", default=None, nargs="+", help='path(str) or paths(ndarray) to file with data to fit with')
    ap.add_argument("--legend_loc", type=float, default=None, nargs="+", help='legend location given by (x,y) coordinates in ([0,1],[0,1]) limits')
    ap.add_argument("--label", default=None, nargs="+", help='custom label in order of plots')
    ap.add_argument("--style", default=None, nargs='+', help='berendsen scr: list of barostat types for tVSbox plot for multiple files')
    args = ap.parse_args()
    if args.plot_aVSdensity: PLOT_aVSdensity(args.parameter, args.a, args.path, args.no_show, args.out)
    if args.plot_mVSvolume: PLOT_mVSvolume(args.m, args.path)
    if args.plot_tVSpressure: PLOT_tVSpressure(args.traj, args.first, args.last, no_show=args.no_show)
    if args.plot_tVSbox:
        PLOT_tVSbox(args.traj, args.first, args.last, no_show=args.no_show, out=args.out, extract_data=args.extract_data,
                no_density=args.no_density, label=args.label, styles=args.style)
    if args.plot_areaVSsurfacetension: PLOT_areaVSsurfacetension(args.traj, args.first, args.last, no_show=args.no_show, out=args.out, extract_data=args.extract_data)
    if args.plot_pressure_profile:
        PLOT_pressure_profile(args.traj, args.first, args.last, args.path, 
            no_show=args.no_show, out=args.out, sqgrad=args.sqgrad, terms=args.terms,
            box_scale=args.box_scale, config=args.config, fit=args.fit, own_scale=args.own_scale,
            legend_loc=args.legend_loc, yaxis=args.yaxis,)
