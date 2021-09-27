"""
                                             ,----,                                                                                                                               
          ____                             ,/   .`|                                                                 ,-.----.                                                      
        ,'  , `.    ,---,                ,`   .'  :                              ,--,                               \    /  \                                                     
     ,-+-,.' _ |  .'  .' `\            ;    ;     /        ,-.----.            ,--.'|                               |   :    \                                                    
  ,-+-. ;   , ||,---.'     \         .'___,/    ,'  ,---.  \    /  \    ,---.  |  | :     ,---.                     |   |  .\ :            __  ,-.                        __  ,-. 
 ,--.'|'   |  ;||   |  .`\  |        |    :     |  '   ,'\ |   :    |  '   ,'\ :  : '    '   ,'\   ,----._,.        .   :  |: |          ,' ,'/ /|  .--.--.             ,' ,'/ /| 
|   |  ,', |  '::   : |  '  |        ;    |.';  ; /   /   ||   | .\ : /   /   ||  ' |   /   /   | /   /  ' /        |   |   \ : ,--.--.  '  | |' | /  /    '     ,---.  '  | |' | 
|   | /  | |  |||   ' '  ;  :        `----'  |  |.   ; ,. :.   : |: |.   ; ,. :'  | |  .   ; ,. :|   :     |        |   : .   //       \ |  |   ,'|  :  /`./    /     \ |  |   ,' 
'   | :  | :  |,'   | ;  .  |            '   :  ;'   | |: :|   |  \ :'   | |: :|  | :  '   | |: :|   | .\  .        ;   | |`-'.--.  .-. |'  :  /  |  :  ;_     /    /  |'  :  /   
;   . |  ; |--' |   | :  |  '            |   |  ''   | .; :|   : .  |'   | .; :'  : |__'   | .; :.   ; ';  |        |   | ;    \__\/: . .|  | '    \  \    `. .    ' / ||  | '    
|   : |  | ,    '   : | /  ;             '   :  ||   :    |:     |`-'|   :    ||  | '.'|   :    |'   .   . |        :   ' |    ," .--.; |;  : |     `----.   \'   ;   /|;  : |    
|   : '  |/     |   | '` ,/              ;   |.'  \   \  / :   : :    \   \  / ;  :    ;\   \  /  `---`-'| |        :   : :   /  /  ,.  ||  , ;    /  /`--'  /'   |  / ||  , ;    
;   | |`-'      ;   :  .'                '---'     `----'  |   | :     `----'  |  ,   /  `----'   .'__/\_: |        |   | :  ;  :   .'   \---'    '--'.     / |   :    | ---'     
|   ;/          |   ,.'                                    `---'.|              ---`-'            |   :    :        `---'.|  |  ,     .-./          `--'---'   \   \  /           
'---'           '---'                                        `---`                                 \   \  /           `---`   `--`---'                          `----'            
                                                                                                    `--`-'                                                                        
MD Topology Parser 
adapted from gmxParse.py 
xinmengli2020@gmal.com 
noted 2021-04-23 Oslo 
'ascii art from http://patorjk.com/software/taag/#p=display&f=Graffiti&t=Type%20Something%20'


- here the class is more like struct as not method is defined 
- could define a general class with maximum information; less efficient than xx_convert2_yy()
"""

import h5py
import sys
import re
import numpy as np
import math as m
import time 
import collections ## ordered dictionary
import pandas as pd 
import os 
import numpy.ma as ma ## mask array to get the bond indices 
from itertools import combinations, product
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import toml
from pathlib import Path
from glob import glob
import shutil

class ItpMolecule:
    def __init__(self, molname,  molnum):
        self.molname      = molname
        self.molnum       = molnum

class ItpAtom:
    def __init__(self,index,atomtype,resnr,resname,atomname,cgnr, charge,mass):
        self.index     = index
        self.atomtype  = atomtype
        self.resnr     = resnr
        self.resname   = resname
        self.atomname  = atomname
        self.cgnr      = cgnr
        self.charge    = charge
        self.mass      = mass
        

class ItpBond:
    def __init__(self, head,tail,func,length,strength):
        self.head      = head
        self.tail      = tail
        self.func      = func
        self.length    = length
        self.strength  = strength

class ItpAngle:
    def __init__(self, up, middle, down, num, angle0,strength,nameA,nameB,nameC):
        self.up        = up
        self.middle    = middle
        self.down      = down
        self.num       = num
        self.angle0    = angle0
        self.strength  = strength
        self.nameA     = nameA
        self.nameB     = nameB
        self.nameC     = nameC


class ItpVdwPair:
    """
    e.g. martini [ nonbond_params ] 
    """
    def __init__(self, vdw_head, vdw_tail, vdw_func, vdw_c6, vdw_c12 ):
        self.vdwHeadType   =  vdw_head  
        self.vdwTailType   =  vdw_tail  
        self.vdwFunc       =  vdw_func
        self.vdwC6         =  vdw_c6 
        self.vdwC12        =  vdw_c12
        
        if self.vdwHeadType  == self.vdwTailType:
            self.sametype  =  True
        else:
            self.sametype  =  False
        
        self.epsilon = self.vdwC6 **2 / (4*self.vdwC12)
        self.sigma   = (self.vdwC12/self.vdwC6)**(1/6) 





    


#class ItpMartiniVdw:
#    """
#    added 06-07-2021 
#    df = pd.DataFrame({ 'vdwHeadType':vdw_head, 
#                        'vdwTailType':vdw_tail,
#                        'vdwSigma'   :vdw_sigma , 
#                        'vdwEpsilon' :vdw_epsilon,
#                        'vdwC6'      :vdw_c6,
#                        'vdwC12'     :vdw_c12,
#                        'vdwKai'     :vdw_kai
#                    })
#    """
#    def __init__(self, vdw_head, vdw_tail, vdw_sigma, vdw_epsilon, vdw_c6, vdw_c12, vdw_kai):
#        self.vdwHeadType   =  vdw_head  
#        self.vdwTailType   =  vdw_tail  
#        self.vdwSigma      =  vdw_sigma   
#        self.vdwEpsilon    =  vdw_epsilon 
#        self.vdwC6         =  vdw_c6 
#        self.vdwC12        =  vdw_c12  
#        self.vdwKai        =  vdw_kai
#    
    


def get_parameter_angle(index1,index2,index3,itpAtoms_F,atoms_F):
    for itpAtom in itpAtoms_F:
        if itpAtom.index == index1:
            nameA_F  = itpAtom.name
            massA_F  = itpAtom.mass
        if itpAtom.index == index2:
            nameB_F  = itpAtom.name
            massB_F  = itpAtom.mass
        if itpAtom.index == index3:
            nameC_F  = itpAtom.name
            massC_F  = itpAtom.mass

    if (massA_F > 1.008000) and (massB_F > 1.008000) and (massC_F > 1.008000):
        strength_F = STRENGTH2
    else:
        strength_F = STRENGTH1

    for atom in atoms_F:
        if atom.index == index1:
            up_x = atom.x
            up_y = atom.y
            up_z = atom.z
        if atom.index == index2:
            middle_x = atom.x
            middle_y = atom.y
            middle_z = atom.z
        if atom.index == index3:
            down_x = atom.x
            down_y = atom.y
            down_z = atom.z
    
    xx = np.array([up_x-middle_x, up_y-middle_y, up_z-middle_z])
    yy = np.array([down_x-middle_x, down_y-middle_y, down_z-middle_z])

    lxx = np.sqrt(xx.dot(xx))
    lyy = np.sqrt(yy.dot(yy))
    
    cos_angle = xx.dot(yy)/(lxx*lyy)
    angle_F = np.arccos(cos_angle)*180/np.pi
    
    return angle_F, strength_F,nameA_F,nameB_F,nameC_F


def get_parameter_angle_from_typecsv(up,middle,down,itpAtom_list,df_atom,df_angle):
    """
    - the trouble some here is the angle types are define accoridg to the atomtypeID, which is missing in the itpAtom_list and has to get the atomtypeID using the df_atom information
    #### df_atom  atomtypeID,atomName,atomMass,atomCharge
    #### df_angle angleUpType,angleMiddleType,angleDownType,angleTheta,angleStrength
    #### ---> up, middle, down, num, angle0,strength,nameA,nameB,nameC
    """
    _FUNC = 1
    func = 1
    up_atomname     = itpAtom_list[up-1].atomname
    middle_atomname = itpAtom_list[middle-1].atomname
    down_atomname   = itpAtom_list[down-1].atomname
    
    [up_atomtypeID, middle_atomtypeID, down_atomtypeID] = [  df_atom.copy()[df_atom["atomName"].isin([x])].atomtypeID.values.tolist()[0] for x in [up_atomname,  middle_atomname, down_atomname ]  ]
    ## fas operation for the below; see e.g. https://www.programiz.com/python-programming/list
    #_item_atom = df_atom.copy()[df_atom["atomName"].isin([up_atomname])]
    #up_atomtypeID   = _item_atom.atomtypeID.values.tolist()[0]
    #print(up_atomname,  middle_atomname, down_atomname, up_atomtypeID, middle_atomtypeID, down_atomtypeID)
    
    _angleitem = df_angle.loc[(df_angle['angleUpType']==up_atomtypeID) & (df_angle['angleMiddleType']==middle_atomtypeID)  & (df_angle['angleDownType']==down_atomtypeID) ]
    if _angleitem.empty:
        _angleitem = df_angle.loc[(df_angle['angleUpType']==down_atomtypeID) & (df_angle['angleMiddleType']==middle_atomtypeID)  & (df_angle['angleDownType']==up_atomtypeID) ]
        if _angleitem.empty:
            raise Exception("ERROR, angle parameters not found")  
    
    angle0 = _angleitem.angleTheta.values.tolist()[0]
    strength = _angleitem.angleStrength.values.tolist()[0]
    #itpAngle_list.append(ItpAngle(up, middle, down, func, angle0,strength,up_atomname,middle_atomname,down_atomname))
    
    return ItpAngle(up, middle, down, func, angle0,strength,up_atomname,middle_atomname,down_atomname)
    

def write_singlebead_itp( itp_file,molName, atomtype, atomcsv):

    itp_lines = []
    _nrexcl   = 1
    ###### section moleculetype
    itp_lines.append("[ moleculetype ]") 
    itp_lines.append("; molname      nrexcl") 
    itp_lines.append(f"{molName}    {_nrexcl}") 
    itp_lines.append('')

    ###### section atoms
    itp_lines.append("[ atoms ]") 
    itp_lines.append("; id 	type 	resnr 	residu 	atom 	cgnr 	charge    mass")

    df_atomtype  =  pd.read_csv( atomcsv )   
    _source_item =  df_atomtype[ df_atomtype["atomName"].isin([ atomtype ]) ]
    charge       =  _source_item.atomCharge.values.tolist()[0]
    mass         =  _source_item.atomMass.values.tolist()[0]
    itp_lines.append(f"    {1}    {atomtype}    {1}    {molName} 	{atomtype}    {1}    {charge}    {mass}")
    itp_lines.append('')

    ##for line in itp_lines:
    ##    print(line)
    ###### write out
    
    f=open( itp_file ,'w')
    s1='\n'.join(itp_lines)
    f.write(s1)
    f.write('\n')
    f.write('\n')
    f.close()


    



def write_molecule_itp( itp_file,molName, itpAtoms,itpBonds,itpAngles):

    itp_lines = []
    _nrexcl   = 1
    ###### section moleculetype
    itp_lines.append("[ moleculetype ]") 
    itp_lines.append("; molname      nrexcl") 
    itp_lines.append(f"{molName}    {_nrexcl}") 
    itp_lines.append('')
    
    ###### section atoms
    itp_lines.append("[ atoms ]") 
    itp_lines.append("; id 	type 	resnr 	residu 	atom 	cgnr 	charge  mass")
    for item in itpAtoms: 
        ## index,atomtype,resnr,resname,atomname,cgnr, charge,mass
        itp_lines.append(f"    {item.index}    {item.atomtype}    {item.resnr}    {item.resname} 	{item.atomname}    {item.cgnr}    {item.charge}    {item.mass}")
    itp_lines.append('')
    
    ###### section bonds 
    itp_lines.append("[ bonds ]") 
    itp_lines.append("; i  j 	funct 	length 	strength")
    for item in itpBonds: 
        ## head,tail,func,length,strength
        itp_lines.append(f"    {item.head}    {item.tail}    {item.func}    {item.length} 	  {item.strength}   ")
    itp_lines.append('')

    ###### section angles 
    itp_lines.append("[ angles ]") 
    itp_lines.append("; i  j  k 	funct 	angle 	strength")
    for item in itpAngles: 
        ##  up, middle, down, num, angle0,strength,nameA,nameB,nameC
        itp_lines.append(f"    {item.up}    {item.middle}    {item.down}    {item.num}    {item.angle0}    {item.strength}     ;   {item.nameA}  {item.nameB}  {item.nameC}  ")
    itp_lines.append('')
    
    ##for line in itp_lines:
    ##    print(line)
    ###### write out
    
    f=open( itp_file ,'w')
    s1='\n'.join(itp_lines)
    f.write(s1)
    f.write('\n')
    f.write('\n')
    f.close()
    
    



def write_top_file(topfile,casename, ffpath, topcsv):
    
    top_lines = []

    df_molecules =  pd.read_csv( topcsv )   ## molName, molNum
    molecules = df_molecules.molName.values.tolist()
    nums = df_molecules.molNum.values.tolist()
    print( molecules, nums )
    
    ##### section: include the forcefield.itp
    top_lines.append(f"; include forcefield.tip")
    ff_file = os.path.join(ffpath, 'forcefield.itp')
    top_lines.append(f"#include \"{ff_file}\"") 
    top_lines.append('')

    ##### section: include separate molecule itp files 
    for molecule in molecules:
        molecule_itp_file = os.path.join(ffpath, f"{molecule}.itp")
        top_lines.append(f"#include \"{molecule_itp_file }\"") 
    top_lines.append('')    
    
    #### section: [ system ]
    top_lines.append("[ system ]") 
    top_lines.append(f"{casename}") 
    top_lines.append('')
    
    #### section: [ molecules ]
    top_lines.append("[ molecules ]") 
    for name, num in zip(molecules, nums):
        top_lines.append(f"{name}    {num}") 
    top_lines.append('')

    for line in top_lines:
        print(line)
    
    #in_top_file = _df1.emTop.values[0]
    #out_top_file = os.path.join(work_folder, "md.top") 
    #change_line_list = [] ## single chain!!
    #for chain in chains:
    #    line = f'Protein_chain_{chain}' 
    #    change_line_list.append(line) 
    #list_top_file = [] 
    #with open(in_top_file,'r') as f:
    #    data = f.readlines()
    #    for line in data:
    #        line_new = line.strip('\n') 
    #        try:
    #            if line_new.split()[0] in change_line_list and int(line_new.split()[1]) == 1: ## some may not have the second thus do not merge to a same condition
    #                #print(line_new)
    #                #### == 1 , excludes the  
    #                #### [ moleculetype ]
    #                #### ; Name            nrexcl
    #                #### Protein_chain_A  60
    #                list_top_file.append(line_new.split()[0] + '    ' + str(N_repeat))
    #            else:
    #                list_top_file.append(line_new)  
    #        except:
    #            list_top_file.append(line_new)     
    ##list_top_file.append( "\n " )
    ############################### write out top file 
    for _line in top_lines:
        print(_line)
    f=open(topfile,'w')
    s1='\n'.join(top_lines)
    f.write(s1)
    f.write('\n')
    f.close() 



     



def write_ff_itp_file(path, ffname, vdwcsv, atomcsv, bondcsv, anglecsv=False, dihedralcsv=False):
    '''
    ----- this one follow the martini type e.g. http://cgmartini.nl/images/parameters/ITP/martini_v2.2.itp; ffnonbonded.itp and ffbonded.itp are include inside the file explicltiy 
    - about the nonbond_params: cit[1] https://manual.gromacs.org/documentation/2019.1/reference-manual/topologies/parameter-files.html
    - _nbfunc=1
    - _comb_rule=1   ; use the v(c6) and w(c12)
    '''
    _nbfunc=1
    _comb_rule=1 
    _element_num = 36 ## NOW the element number is set to a fix value 
    _ptype = 'A' ##By default
    _hymd_label = ';===>HymdKai'
    _func_nonbond = 1 
    _func_bond    = 1
    _func_angle   = 1
    float_accu = 8
    ######
    itp_lines = []
    ###### note 
    #_note = '; DEMOOOOOO '
    #itp_lines.append(_note)
    #itp_lines.append('')
    ###### define ff name 
    itp_lines.append(f"#define _FF_{ffname}") 
    itp_lines.append('')
    ###### section: default 
    itp_lines.append(f"[ defaults ]") 
    itp_lines.append("; nbfunc       comb-rule") 
    itp_lines.append(f"{_nbfunc}    {_comb_rule}") 
    itp_lines.append('')
    ###### section ffnonbonded.itp [ atomtypes ]
    ## name   at.num      mass      charge   ptype         V(c6)        W(c12)
    ## in the martini itp the the columns are treated as 
    ## name    mass      charge          ptype c6 c12
    ## following the cit[1], add the at.num all as 36 
    itp_lines.append("[ atomtypes ]") 
    itp_lines.append("; name   at.num      mass      charge   ptype         V(c6)        W(c12)") 
    df_vdw      =  pd.read_csv( vdwcsv )  ## contains the vdw self-and pair lj interaction 
    ## extraBurden, has to read the  atomcsv to get the atom type names; need to get the names of atomtypes in atomcsv 
    df_atomtype =  pd.read_csv( atomcsv )   
    
    ## loop and add the atomtypes
    for index, row in df_vdw.iterrows():
        ## print(index, int(row["vdwHeadType"]), type(row["vdwHeadType"]))
        if int(row["vdwHeadType"]) == int(row["vdwTailType"]):
            check_id = int(row["vdwHeadType"])
            _source_item    = df_atomtype[ df_atomtype["atomtypeID"].isin([ check_id ]) ]
            check_id_name   = _source_item.atomName.values.tolist()[0]
            check_id_mass   = _source_item.atomMass.values.tolist()[0]
            check_id_charge = _source_item.atomCharge.values.tolist()[0]
            check_id_vdwC6  = np.round(row["vdwC6"], float_accu)
            check_id_vdwC12 = np.round(row["vdwC12"],float_accu)
            check_id_kai    = np.round(row["vdwKai"],float_accu)
            #print( check_id, check_id_name )
            itp_lines.append(f" {check_id_name}    {_element_num}    {check_id_mass}    {check_id_charge}    {_ptype}    {check_id_vdwC6}    {check_id_vdwC12}    {_hymd_label} {check_id_kai}")        
    
    ###### section ffnonbonded.itp nonbonded pair intereacitons
    ###   ; i    j func       V(c6)        W(c12)
    itp_lines.append('')
    itp_lines.append("[ nonbond_params ]")
    itp_lines.append("; i    j     func       V(c6)        W(c12)")
    for index, row in df_vdw.iterrows():
        if int(row["vdwHeadType"]) != int(row["vdwTailType"]):
            check_id_i        = int(row["vdwHeadType"])
            check_id_j        = int(row["vdwTailType"])
            _source_item_i    = df_atomtype[ df_atomtype["atomtypeID"].isin([ check_id_i ]) ]
            _source_item_j    = df_atomtype[ df_atomtype["atomtypeID"].isin([ check_id_j ]) ]
            check_id_i_name   = _source_item_i.atomName.values.tolist()[0]
            check_id_j_name   = _source_item_j.atomName.values.tolist()[0]
            ##print(check_id_i_name, check_id_j_name)
            check_vdwC6  = np.round(row["vdwC6"], float_accu)
            check_vdwC12 = np.round(row["vdwC12"],float_accu)
            check_kai    = np.round(row["vdwKai"],float_accu)
            itp_lines.append(f" {check_id_i_name}    {check_id_j_name}    {_func_nonbond}    {check_vdwC6}    {check_vdwC12}    {_hymd_label} {check_kai}")  

    ###### section [ bondtypes ] e.g. from ffbonded.itp 
    df_bondtype =  pd.read_csv( bondcsv )  
    itp_lines.append('')
    itp_lines.append("[ bondtypes ]")
    itp_lines.append("; i    j    func        b0          kb   ")
    for index, row in df_bondtype.iterrows():
        check_id_i        = int(row["bondHeadType"])
        check_id_j        = int(row["bondTailType"])
        _source_item_i    = df_atomtype[ df_atomtype["atomtypeID"].isin([ check_id_i ]) ]
        _source_item_j    = df_atomtype[ df_atomtype["atomtypeID"].isin([ check_id_j ]) ]
        check_id_i_name   = _source_item_i.atomName.values.tolist()[0]
        check_id_j_name   = _source_item_j.atomName.values.tolist()[0]
        check_length      = np.round(row["bondLength"], float_accu)
        check_strength    = np.round(row["bondStrength"],float_accu)
        itp_lines.append(f" {check_id_i_name}    {check_id_j_name}    {_func_bond}    {check_length}    {check_strength}")
   
    ###### section [ angletypes ] 
    if anglecsv:
        df_angletype =  pd.read_csv( anglecsv )  
        itp_lines.append('')
        itp_lines.append("[ angletypes ]")
        itp_lines.append("; i    j    k func       th0         cth   ")
        for index, row in df_angletype.iterrows():
            check_id_i        = int(row["angleUpType"])
            check_id_j        = int(row["angleMiddleType"])
            check_id_k        = int(row["angleDownType"])
            _source_item_i    = df_atomtype[ df_atomtype["atomtypeID"].isin([ check_id_i ]) ]
            _source_item_j    = df_atomtype[ df_atomtype["atomtypeID"].isin([ check_id_j ]) ]
            _source_item_k    = df_atomtype[ df_atomtype["atomtypeID"].isin([ check_id_j ]) ]
            check_id_i_name   = _source_item_i.atomName.values.tolist()[0]
            check_id_j_name   = _source_item_j.atomName.values.tolist()[0]
            check_id_k_name   = _source_item_k.atomName.values.tolist()[0]
            check_theta       = np.round(row["angleTheta"], float_accu)
            check_strength    = np.round(row["angleStrength"],float_accu)
            itp_lines.append(f" {check_id_i_name}    {check_id_j_name}    {check_id_j_name}    {_func_angle}    {check_theta}    {check_strength}")

    #for _line in itp_lines:
    #    print(_line)
    
    out_itp_file = os.path.join(path, 'forcefield.itp')
    f=open( out_itp_file ,'w')
    s1='\n'.join(itp_lines)
    f.write(s1)
    f.write('\n')
    f.write('\n')
    f.close()



def extract_itp_from_fort5_fort3csv( fort5_atoms, resname, atomcsv, bondcsv, vdwcsv, anglecsv=False, dihedralcsv=False):
    """
    - ITP reference: from https://manual.gromacs.org/documentation/current/reference-manual/topologies/topology-file-formats.html 
    - ===> itp atoms seciton: index,atomtype,resnr,resname,atomname,cgnr, charge ,mass 
      ==== e.g.                  1    C        1      URE      C      1     0.880229  12.01000     
      fort5_atoms exp: atomid,atomname,atomtypeID,bondnum,x,y,z,vx,vy,vz,bond1,bond2,bond3,bond4,bond5,bond6
      RESNR = 1 ## here only process one molecule/residue
    
    - ===> itp bonds seciton:      head, tail,   func, length,strength
      ==== e.g.                      5    21      1   0.68031 500.00000
      FUNC = 1 ## here by default only use bond type = 1
    """
    df_atom   =  pd.read_csv( atomcsv) ##  atomtypeID,atomName,atomMass,atomCharge
    df_bond   =  pd.read_csv( bondcsv) ##  bondHeadType,bondTailType,bondLength,bondStrength
    df_vdw    =  pd.read_csv( vdwcsv ) 
    ##print(df_vdw)
    
    #### atoms section 
    RESNR = 1  
    itpAtom_list = []
    for atom in fort5_atoms:
        _item_atom = df_atom.copy()[df_atom["atomtypeID"].isin([atom.atomtypeID])]
        index     = atom.atomid
        atomtype  = atom.atomname #atom.atomtypeID; 
        resnr     = RESNR
        resname   = resname
        atomname  = atom.atomname
        cgnr      = atom.atomid
        charge    = _item_atom.atomCharge.values.tolist()[0]
        mass      = _item_atom.atomMass.values.tolist()[0]
        itpAtom_list.append (ItpAtom(index,atomtype,resnr,resname,atomname,cgnr, charge ,mass ))
    #for itp in itpAtom_list:
    #    print('itpatom',itp.__dict__)
    
    #### bonds secttion
    FUNC = 1 
    itpBond_list = [] # head, tail,   func, length,strength

    fort5_bond_list = []
    for atom in fort5_atoms:
        head = atom.atomid
        head_type = atom.atomtypeID   
        ##print([ atom.bond1,atom.bond2,atom.bond3,atom.bond4,atom.bond5,atom.bond6])
        for j in [atom.bond1,atom.bond2,atom.bond3,atom.bond4,atom.bond5,atom.bond6]:
            if j != 0: 
                tail = j 
                tail_type = fort5_atoms[j-1].atomtypeID 
                ##print(head, tail)
                if ([head, tail] in fort5_bond_list) or ([tail, head] in fort5_bond_list):
                    pass
                else:
                    fort5_bond_list.append([head, tail])
                    ##print('keep --- ', head, tail) 
                    func = FUNC
                    #print(head, tail, head_type, tail_type)
                    #for bond_item in df_bond
                    #_item_record = df_bond[df_bond["bondHeadType"].isin([head_type])] and df_bond[df_bond["bondTailType"].isin([tail_type])]
                    _bonditem = df_bond.loc[(df_bond['bondHeadType']==head_type) & (df_bond['bondTailType']==tail_type)]
                    if _bonditem.empty:
                        _bonditem = df_bond.loc[(df_bond['bondHeadType']==tail_type) & (df_bond['bondTailType']==head_type)]
                        if _bonditem.empty:
                            raise Exception("ERROR, bond parameters not found")  
                    #print(_bonditem)
                    length = _bonditem.bondLength.values.tolist()[0]
                    strength = _bonditem.bondStrength.values.tolist()[0]
                    itpBond_list.append(ItpBond(head,tail,func,length,strength))
    
    #for itp in itpBond_list:
    #    print('itpbond', itp.__dict__)
    
    ########################### angle part  ################################################### 
    if not anglecsv:
        pass
    else:
        df_angle = pd.read_csv( anglecsv) ## angleUpType,angleMiddleType,angleDownType,angleTheta,angleStrength
        FUNC = 1 
        itpAngle_list = [] # head, tail,   func, length,strength

        #### generate angle list: from bonds to angles
        wholeindex = np.array(itpBond_list).tolist() 
        ###wholeindex = np.arange(len(itpBond_list)).tolist()
        checklist = list(zip(wholeindex, wholeindex[1:] + wholeindex[:1])) ## ## reference https://www.geeksforgeeks.org/python-pair-iteration-in-list/
        for pair in checklist:
            #print(pair[0].__dict__, pair[1].__dict__)
            item1 = pair[0]
            item2 = pair[1]
            if (item2.head == item1.tail):
                up        = item1.head
                middle    = item1.tail
                down      = item2.tail
                itpAngle = get_parameter_angle_from_typecsv(up,middle,down,itpAtom_list,df_atom, df_angle) 
                itpAngle_list.append(itpAngle) 
                    
        #for itp in itpAngle_list:
        #    print('itpangle',itp.__dict__)
    ##############
    if dihedralcsv :
        return (itpAtom_list, itpBond_list, itpAngle_list, itpDihedral_list )
    elif anglecsv :
        return (itpAtom_list, itpBond_list, itpAngle_list)
    else:
        return (itpAtom_list, itpBond_list)
    
        



            

        
    


class PdbAtom:
    '''the basic class for atom in pdb file, basic format
    see in http://deposit.rcsb.org/adit/docs/pdb_atom_format.html
    '''
    def __init__(self,label,index, name,indicator,residue, chain,resid,insert,x,y,z,occu,temp,seg,element,charge):
        self.label     = label
        self.index     = index 
        self.name      = name 
        self.indicator = indicator 
        self.residue   = residue
        self.chain     = chain
        self.resid     = resid 
        self.insert    = insert
        self.x         = x 
        self.y         = y
        self.z         = z
        self.occu      = occu
        self.temp      = temp
        self.seg       = seg
        self.element   = element
        self.charge    = charge

class GroAtom:
    def __init__(self,resid,residuename, atomname, index, x, y, z):
        self.resid        = resid
        self.residuename  = residuename
        self.atomname     = atomname
        self.index        = index
        self.x         = x
        self.y         = y
        self.z         = z


class Fort5Atom:
    """
    atom line example:
          1 T        1      1   18.417    9.410   27.788    0 0 0    2      0      0      0      0      0
          2 C        2      2   18.381    9.084   27.791    0 0 0    1      3      0      0      0      0
    --> parsing information 
     index atomname  atomtypeID bondnum  x  y         z   vx vy vz  bond1 bond2   bond3  bond4  bond5  bond6
    """
    def __init__(self,atomid,atomname,atomtypeID,bondnum,x,y,z,vx,vy,vz,bond1,bond2,bond3,bond4,bond5,bond6):
        
        self.atomid       =    atomid
        self.atomname     =    atomname
        self.atomtypeID   =    atomtypeID
        self.bondnum      =    bondnum
        self.x            =    x
        self.y            =    y 
        self.z            =    z 
        self.vx           =    vx
        self.vy           =    vy
        self.vz           =    vz
        self.bond1        =    bond1
        self.bond2        =    bond2
        self.bond3        =    bond3
        self.bond4        =    bond4
        self.bond5        =    bond5
        self.bond6        =    bond6


class Fort5AtomWhole:
    """
    extended from Fort5Atom; with self.resid
    """
    def __init__(self,resid, atomid,atomname,atomtypeID,bondnum,x,y,z,vx,vy,vz,bond1,bond2,bond3,bond4,bond5,bond6):
        
        self.resid        =    resid  ## resid starts from 1 ok 
        self.atomid       =    atomid
        self.atomname     =    atomname
        self.atomtypeID   =    atomtypeID
        self.bondnum      =    bondnum
        self.x            =    x
        self.y            =    y 
        self.z            =    z 
        self.vx           =    vx
        self.vy           =    vy
        self.vz           =    vz
        self.bond1        =    bond1
        self.bond2        =    bond2
        self.bond3        =    bond3
        self.bond4        =    bond4
        self.bond5        =    bond5
        self.bond6        =    bond6

def loadGroPosition(inputfilename):
    """
    # 2021-06-10  
    BOX SIZE WITH A SPECIFIC GAP  
    _BOX_ITEM_GAP = '   '
    # 2021-06-16 
    just use .split() instead of .split(_BOX_ITEM_GAP)
    
    """
    _REMARK = 2
    _BOX_ITEM_GAP = '   '

    with open(inputfilename) as f:
        lines = f.readlines()
        num_atoms = int(lines[1])
        box_size = np.array( lines[_REMARK+num_atoms].strip('\n').lstrip().split(), dtype=float)
        #box_size = np.array( lines[_REMARK+num_atoms].strip('\n').lstrip().split(_BOX_ITEM_GAP), dtype=float) 
    #print(nun_atoms)
    #print(box_size)
    demo=[]
    GroAtom_list = []
    ##    1ETH     CB    1   1.460   1.559   1.491  0.2285  0.2711 -0.7051
    ##    0         1    2    3       4       5       6
    with open(inputfilename,'r') as f:
        data = f.readlines()
        count = 0
        for line in data[_REMARK:_REMARK+num_atoms]:
            count += 1
            #print (line)
            list = line.split()
            ## the resid and residuename belong to list[0],split it
            resid         = int(line[0:5])#int( (count-1) / 121 ) +1 
            residuename   = line[5:10]
            atomname      = line[10:15]
            index         = int(line[15:20])
            x             = float(line[20:28])  ## double in c, float64 in numpy --> doesn't matter in here
            y             = float(line[28:36])
            z             = float(line[36:44])
            GroAtom_list.append(GroAtom(resid,residuename, atomname, index, x, y, z))
    return GroAtom_list, box_size


def loadfort5_whole(inputfilename):
    """
    here reads an example file fort.5 file 
    -  reference fort5_to_hdf5.py "hymd/utils"
    -  
    """
    with open(inputfilename, "r") as f:
        data = f.readlines()
    
    box_array = np.array( [float(data[1].split()[i]) for i in range(3)] )
    n_atoms  = int(data[-1].split()[0])
    n_molecules = int(data[4].split()[0])
    skip_head = 4  ## 5-1 ## after this will be molecule blockes 
    skip_head_mol = 2

    line_count = skip_head
    atom_list  = []
    for i in np.arange( n_molecules) :
        resid = int(data[line_count+1].split()[-1]) ## starts from 1 ok 
        num_atom_in_resid = int(data[line_count+2])
        ## print(i, resid, num_atom_in_resid) ## time.sleep(1)
        for line in data[line_count+skip_head_mol+1:line_count+skip_head_mol+num_atom_in_resid+1]: #rightside need + 1
            ## print(line) ## time.sleep(1)
            list = line.split()
            atomid      =  int(list[0])
            atomname    =  list[1]
            atomtypeID  =  int(list[2])
            bondnum     =  int(list[3])
            x           =  float(list[4])
            y           =  float(list[5])
            z           =  float(list[6])
            vx          =  float(list[7])
            vy          =  float(list[8])
            vz          =  float(list[9])
            bond1       =  int(list[10])
            bond2       =  int(list[11])
            bond3       =  int(list[12])
            bond4       =  int(list[13])
            bond5       =  int(list[14])
            bond6       =  int(list[15])
            atom_list.append(Fort5AtomWhole(resid,atomid,atomname,atomtypeID,bondnum,x,y,z,vx,vy,vz,bond1,bond2,bond3,bond4,bond5,bond6))
        
        line_count += (num_atom_in_resid + skip_head_mol)
    
    return (atom_list, box_array)
    

def fort5whole_write2_gro(in_fort5_atoms, out_gro_file, case_name, box_array,top_mol):

    resname_list = []
    
    ## add judge information 
    if isinstance(top_mol, collections.OrderedDict):
        print('top_mol input as OrderedDict')

        for k, v in top_mol.items():
            #print(k, v)
            resname_list += [k] * v
        ##print(resname_list)
    elif isinstance(top_mol, pd.DataFrame):
        print('top_mol input is data frame, to be added')
    
    
    with open( out_gro_file,'w') as fo:
        fo.write(case_name+'\n')
        fo.write("%5d "%( len( in_fort5_atoms) )+'\n')
        for atom in in_fort5_atoms:
            if (atom.atomid > 99999):
                atom.atomid = (atom.atomid % 99999)
            _resid = atom.resid 
            if (atom.resid > 99999):
                atom.resid = (atom.resid % 99999)
            #fo.write( "%5d%5s%5s%5d%8.3f%8.3f%8.3f"%(atom.resid,atom.residuename,atom.element,atom.index,atom.x,atom.y,atom.z)+'\n')
            fo.write( "%5d%-5s%5s%5d%8.3f%8.3f%8.3f"%(atom.resid, resname_list[_resid-1], atom.atomname, atom.atomid,atom.x,atom.y,atom.z)+'\n')
        fo.write(" %8.5f %8.5f %8.5f"%(box_array[0], box_array[1], box_array[2]) )
        fo.write('\n')    
    

def loadfort5_simple(inputfilename):
    """
    here reads an example file solute.5 file 
    - clean refers to that the first line is the number of atoms
    - skip = 1   
    """
    skip = 1 
    demo=[]
    atom_list = []
    with open(inputfilename,'r') as f:
        data = f.readlines()
        atom_num = int(data[0])
        ##print(atom_num)
        count = 0
        for line in data[skip:skip+atom_num]:
            count += 1
            ##print (line)
            list = line.split()
            ####   1 T        1      1   18.417    9.410   27.788    0 0 0    2      0      0      0      0      0
            ###->  0 1        2      3     4        5        6       7 8 9    10     11    12     13      14     15 
            atomid      =  int(list[0])
            atomname    =  list[1]
            atomtypeID  =  int(list[2])
            bondnum     =  int(list[3])
            x           =  float(list[4])
            y           =  float(list[5])
            z           =  float(list[6])
            vx          =  float(list[7])
            vy          =  float(list[8])
            vz          =  float(list[9])
            bond1       =  int(list[10])
            bond2       =  int(list[11])
            bond3       =  int(list[12])
            bond4       =  int(list[13])
            bond5       =  int(list[14])
            bond6       =  int(list[15])
            atom_list.append(Fort5Atom(atomid,atomname,atomtypeID,bondnum,x,y,z,vx,vy,vz,bond1,bond2,bond3,bond4,bond5,bond6))
    return atom_list

def fort5_write2_gro(in_fort5_atoms, out_gro_file, case_name, molecule_name='MOL', box_x=5,box_y=5,box_z=5):
    with open( out_gro_file,'w') as fo:
        fo.write(case_name+'\n')
        fo.write("%5d "%( len( in_fort5_atoms) )+'\n')
        for atom in in_fort5_atoms:
            if (atom.atomid > 99999):
                atom.atomid = (atom.atomid % 99999)
            #fo.write( "%5d%5s%5s%5d%8.3f%8.3f%8.3f"%(atom.resid,atom.residuename,atom.element,atom.index,atom.x,atom.y,atom.z)+'\n')
            try:
                resid  = atom.resid 
            except:
                resid = 1 
            fo.write( "%5d%-5s%5s%5d%8.3f%8.3f%8.3f"%(resid, molecule_name, atom.atomname, atom.atomid,atom.x,atom.y,atom.z)+'\n')
        fo.write(" %8.5f %8.5f %8.5f"%(box_x, box_y, box_z) )
        fo.write('\n')    
        


def write_pdb_file(atoms,bonds,outputfilename):
    with open(outputfilename, 'w') as fo:
        #remarks
        fo.write("REMARK    Regenerated by local python code"+'\n')
        fo.write("REMARK    SOME MOLECULE"+'\n')
        for atom in atoms:
            fo.write( "%-6s%5d %4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f      %4s%2s%2s"%(atom.label,atom.index, atom.name,atom.indicator,atom.residue, atom.chain,atom.resid,atom.insert,atom.x,atom.y,atom.z,atom.occu,atom.temp,atom.seg,atom.element,atom.charge)+'\n')
            ## for the '\n',when split if use [78:] will include the \n to the last element, avoid that

        #TER MIDDLE 
        fo.write("TER     %d"%(ATOMNUM+1))
        fo.write('\n')
        for bondgroup in bonds:
            fo.write("CONECT")
            for item in bondgroup:
                fo.write("%5d"%item)
            fo.write('\n')
        #END
        fo.write("END"+'\n')

def write_numpy_to_mol2_file_nocharge(array,outputfilename):   
    unit_nm_to_ang = 1 #10 
    N_MOL = len(array)
    with open(outputfilename, 'w') as fo:
        fo.write("@<TRIPOS>MOLECULE"+'\n')
        fo.write("MOL"+'\n')
        fo.write("%10d  0  0  0   "%(N_MOL) +'\n')
        fo.write("SMALL"+'\n')
        fo.write("USER_CHARGES"+'\n')
        fo.write('\n')

        fo.write("@<TRIPOS>ATOM"+'\n')
        for i in np.arange( N_MOL ):
            fo.write( "%8d %8s %10.4f%10.4f%10.4f c3 %8d MOL  %13.6f"%( i+1,'C', array[i][0] * unit_nm_to_ang , array[i][1]* unit_nm_to_ang, array[i][2]*unit_nm_to_ang, i+1, 0)+'\n')
        fo.write('\n')
        fo.write('\n')



def write_numpy_to_mol2_file(array,charges, outputfilename):   
    unit_nm_to_ang = 10 #10 
    N_MOL = len(array)
    with open(outputfilename, 'w') as fo:
        fo.write("@<TRIPOS>MOLECULE"+'\n')
        fo.write("MOL"+'\n')
        fo.write("%10d  0  0  0   "%(N_MOL) +'\n')
        fo.write("SMALL"+'\n')
        fo.write("USER_CHARGES"+'\n')
        fo.write('\n')

        fo.write("@<TRIPOS>ATOM"+'\n')
        for i in np.arange( N_MOL ):
            fo.write( "%8d %8s %10.4f%10.4f%10.4f c3 %8d MOL  %13.6f"%( i+1,'C', array[i][0] * unit_nm_to_ang , array[i][1]* unit_nm_to_ang, array[i][2]*unit_nm_to_ang, i+1,charges[i])+'\n')
        fo.write('\n')
        fo.write('\n')




def WRITE_TRJ_GRO(fp, x, vel,t,nparticles,box):    
    fp.write('MD of %d mols, t=%.3f\n'%(nparticles,t))
    fp.write('%-10d\n'%(nparticles))
    for i in range(len(x)):
        fp.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f\n"%(i//10+1,'A','A',i+1,x[i,0],x[i,1],x[i,2],vel[i,0],vel[i,1],vel[i,2]))
    fp.write("%-5.5f\t%5.5f\t%5.5f\n"%(box[0],box[1],box[2]))
    fp.flush()
    return fp


def gro_to_hdf5( groAtoms, box, top_mol_dict, charges=False):
    """
    - reference: fort5_to_hdf5.py  
    UNFINISHED
    """
    box = box
    n_atoms = len(groAtoms)
    n_molecules = 0 
    for k, v in top_mol_dict.items():
        n_molecules += v 
    #print(n_molecules)
    _ndim = 3
    f_hd5 = h5py.File(out_path, "w")

    dset_pos = f_hd5.create_dataset("coordinates", (1, n_atoms, _ndim), dtype="float64")
    dset_vel = f_hd5.create_dataset("velocities", (1, n_atoms, _ndim), dtype="float64")
    dset_types = f_hd5.create_dataset("types", (n_atoms,), dtype="i")
    dset_molecule_index = f_hd5.create_dataset("molecules", (n_atoms,), dtype="i")
    dset_indices = f_hd5.create_dataset("indices", (n_atoms,), dtype="i")
    dset_names = f_hd5.create_dataset("names", (n_atoms,), dtype="S5")
    dset_bonds = f_hd5.create_dataset("bonds", (n_atoms, _ndim), dtype="i")
    
    f_hd5.attrs["box"] = box
    f_hd5.attrs["n_molecules"] = n_molecules
    molecule_index = 0

    """    
    def write_to_dataset(molecule_lines, molecule_index):
        atom_indices = [int(s.split()[0]) for s in molecule_lines]
        type_indices = [int(s.split()[2]) for s in molecule_lines]
        names = [s.split()[1] for s in molecule_lines]

        for i, line in enumerate(molecule_lines):
            atom_index = atom_indices[i] - 1
            type_index = type_indices[i] - 1
            pos_vel = [float(s.replace("D", "E")) for s in line.split()[4:10]]
            dset_pos[0, atom_index, :] = np.array(pos_vel[:3])
            dset_vel[0, atom_index, :] = np.array(pos_vel[3:])
            dset_types[atom_index] = type_index
            dset_molecule_index[atom_index] = molecule_index
            dset_names[atom_index] = np.string_(names[i])
            dset_indices[atom_index] = atom_index

            bonds = [int(s) - 1 for s in line.split()[10:]]
            dset_bonds[atom_index] = bonds[:3]

    for i, line in enumerate(lines):
        split_line = line.split()
        if i > 4 and len(split_line) == 1 and is_int(split_line[0]):
            molecule_start_line = i
            atoms_in_molecule = int(split_line[0])

            molecules_lines = lines[
                molecule_start_line + 1 : molecule_start_line + atoms_in_molecule + 1
            ]
            write_to_dataset(molecules_lines, molecule_index)
            molecule_index += 1
    """

def load_kai_matrix_gen_toml(input_kai_full_csv,toml_template, out_toml_folder,out_toml_label):

    df_kai_full =  pd.read_csv( input_kai_full_csv ) 
    #print(df_kai_full)
    ### get a list without repeats 
    df_kai_full_norpt = []
    for indexi, rowi in df_kai_full.iterrows():
        found = False
        name_set_i = { rowi['HName'], rowi['TName'] }
        for item in df_kai_full_norpt:
            name_set_j = { item['HName'], item['TName'] }
            if name_set_i == name_set_j:
                found = True 
                break 
        if found:
            continue 
        else:
            df_kai_full_norpt.append(rowi ) 

    #for item in df_kai_full_norpt:
    #    print( item['HName'], item['TName'] )
    ###

    Path(out_toml_folder).mkdir(parents=True, exist_ok=True)

    ori_toml = toml.load(toml_template)
    #print(ori_toml['bonds'])

    _Cut_Melt = 5 
    print(f'WARNING, _Cut_Melt in this function is set to {_Cut_Melt} by default')
    column_names =  list(df_kai_full.columns)
    ##['HName', 'TName', 'MtnHeadName', 'MtnTailName', 'BiasDist', 'MtnKai', 'MtnKaiNoS', ...]
    case_list = column_names[_Cut_Melt:]
    case_id = 0 
    
    for case in case_list:
        kai_toml = []  #chi = [ [ ["T", "C"],[-10.800] ], ... ] 
        print(case) ## each case generate a toml file 
        ### OLD use the full csv dataframe  
        ##kai_value_list = df_kai_full[case].values.tolist()
        ##kai_head_list = df_kai_full['HName'].values.tolist()
        ##kai_tail_list = df_kai_full['TName'].values.tolist()
        #### NEW use the list of series  
        for item in df_kai_full_norpt:
            kai_head = item['HName']
            kai_tail = item['TName']
            kai_value = item[case]
            record  = [ [kai_head, kai_tail],[round(kai_value, 3)]] 
            #print(record)
            kai_toml.append(record)
        
        out_toml = ori_toml
        out_toml['field']['chi'] = kai_toml
        out_toml_file = os.path.join(out_toml_folder, f'id{case_id}_{out_toml_label}_{case}.toml') 
        f = open(out_toml_file,'w')
        toml.dump(out_toml, f)
        f.close()

        case_id += 1 
    
    ####### a simple change and write out 
    #out_toml = ori_toml
    #out_toml['field']['chi'] = [1,2,3]
    #case = 'test'
    #out_toml_file = os.path.join(out_toml_folder, f'{out_toml_label}_{case}.toml') 
    #f = open(out_toml_file,'w')
    #toml.dump(out_toml, f)
    #f.close()
    ######## 
    

def gen_single_slurm_multitasks2(work_folder, slurm_label, toml_label,  num_case=1):
    """
    """
    ####################################### setup could from a toml file 
    job_name = 'HyMD'
    budget_account='nn4654k'
    run_timelen = '1-0:0:0'
    num_nodes = num_case # 1 node per case 
    cores_pernode = 128
    #######################################
    IN_H5 = 'all.h5'
    seed  = 10 
    #######################################

    slurm_file = os.path.join(work_folder, f'{slurm_label}.slurm')
    with open(slurm_file, 'w') as fo:
        fo.write(f'#!/bin/bash'+'\n')
        fo.write(f'#SBATCH --job-name={job_name}'+'\n')
        fo.write(f'#SBATCH --account={budget_account}'+'\n')
        fo.write(f'#SBATCH --time={run_timelen}'+'\n')
        fo.write(f'#SBATCH --nodes={num_nodes}'+'\n')
        fo.write(f'#SBATCH --ntasks-per-node={cores_pernode}'+'\n')
        fo.write("\n")

        fo.write(f'module --quiet purge'+'\n')
        fo.write(f'set -o errexit # exit on errors'+'\n')
        fo.write(f'module load h5py/2.10.0-foss-2020a-Python-3.8.2'+'\n')
        fo.write(f'module load pfft-python/0.1.21-foss-2020a-Python-3.8.2'+'\n')
        fo.write(f'set -x'+'\n')
        fo.write("\n")

        fo.write(f'OUT_DIR=${{SLURM_SUBMIT_DIR}}'+'\n')
        
        for i in range(num_case):
            fo.write(f'CASE_{i}=${{OUT_DIR}}/case_{i}'+'\n')
        fo.write("\n")
        
        for i in range(num_case):
            fo.write(f'mkdir -p ${{CASE_{i}}}' +'\n')
        fo.write("\n")
        
        #### rename the toml file and put in separate folders 
        for i in range(num_case):
            target_file = os.path.join(f'{work_folder}', f'case_{i}.toml')  
            files = glob(f'{work_folder}/{toml_label}{i}_*') 
            if len(files) != 1:
                raise ValueError('error: non unique toml file!!!')
            ori_file =  files[0]
            shutil.copyfile(ori_file, target_file)
        #########################################################

        fo.write(f'date'+'\n')
        for i in range(num_case):
            fo.write(f'srun --exclusive --nodes 1 --ntasks {cores_pernode} python3 ${{HOME}}/HyMD-2021/hymd/main.py case_{i}.toml  {IN_H5} --logfile=log_{i}.txt --verbose 2 --velocity-output --destdir ${{CASE_{i}}} --seed {seed}  ')
            if i == num_case-1: 
                endstr = f'&> /dev/null'+'\n'
            else:
                endstr = f'&'+'\n'
            fo.write(endstr)
        fo.write("\n")
        


def gen_single_slurm_multitasks(work_folder, slurm_label, toml_label,  num_case=1):
    """
    run all the tasks in single slurm 
    see example: template-run-betzy-multiple.slurm
    
    """
    ####################################### setup could from a toml file 
    job_name = 'HyMD'
    budget_account='nn4654k'
    run_timelen = '1-0:0:0'
    num_nodes = num_case # 1 node per case 
    cores_pernode = 128
    #######################################
    IN_H5 = 'all.h5'
    seed  = 10 
    #######################################

    slurm_file = os.path.join(work_folder, f'{slurm_label}.slurm')
    with open(slurm_file, 'w') as fo:
        fo.write(f'#!/bin/bash'+'\n')
        fo.write(f'#SBATCH --job-name={job_name}'+'\n')
        fo.write(f'#SBATCH --account={budget_account}'+'\n')
        fo.write(f'#SBATCH --time={run_timelen}'+'\n')
        fo.write(f'#SBATCH --nodes={num_nodes}'+'\n')
        fo.write(f'#SBATCH --ntasks-per-node={cores_pernode}'+'\n')
        fo.write("\n")

        fo.write(f'module --quiet purge'+'\n')
        fo.write(f'set -o errexit # exit on errors'+'\n')
        fo.write(f'module load h5py/2.10.0-foss-2020a-Python-3.8.2'+'\n')
        fo.write(f'module load pfft-python/0.1.21-foss-2020a-Python-3.8.2'+'\n')
        fo.write(f'set -x'+'\n')
        fo.write("\n")

        fo.write(f'OUT_DIR=${{SLURM_SUBMIT_DIR}}'+'\n')
        
        for i in range(num_case):
            fo.write(f'CASE_{i}=${{OUT_DIR}}/case_{i}'+'\n')
        fo.write("\n")
        
        for i in range(num_case):
            fo.write(f'mkdir -p ${{CASE_{i}}}' +'\n')
        fo.write("\n")
        
        #### rename the toml file and put in separate folders 
        for i in range(num_case):
            target_file = os.path.join(f'{work_folder}', f'case_{i}.toml')  
            files = glob(f'{work_folder}/{toml_label}{i}_*') 
            if len(files) != 1:
                raise ValueError('error: non unique toml file!!!')
            ori_file =  files[0]
            shutil.copyfile(ori_file, target_file)
        #########################################################

        fo.write(f'date'+'\n')
        for i in range(num_case):
            fo.write(f'srun --exclusive --nodes 1 --ntasks {cores_pernode} python3 ${{HOME}}/HyMD-2021/hymd/main.py case_{i}.toml  {IN_H5} --logfile=log_{i}.txt --destdir ${{CASE_{i}}} --seed {seed}  ')
            if i == num_case-1: 
                endstr = f'&> /dev/null'+'\n'
            else:
                endstr = f'&'+'\n'
            fo.write(endstr)
        fo.write("\n")
        
        """
        group = ' '
        count = 0
        for i in xrange(total_num_atom):
            #if i%121 == 26:
            group += str(i+1) + ' '
            count += 1
            if count % 15 == 0:
                group += "\n"
        fo.write( group + "\n")
        """


def gen_multi_slurm_singletask_saga(work_folder, slurm_label, toml_label,  num_case=1):
    """
    run all the tasks in seperate slurm files on saga

    see example: template-run-betzy-multiple.slurm
    """
    ####################################### setup could from a toml file 
    job_name = 'HyMD'
    budget_account='nn4654k'
    run_timelen = '6-0:0:0'
    #num_nodes = num_case # 1 node per case 
    cores_pernode = 32
    mem_per_cpu ='1G'
    #######################################
    IN_H5 = 'all.h5'
    seed  = 10

    #######################################
    for i in np.arange(num_case):
        slurm_file = os.path.join(work_folder, f'{slurm_label}_{i}.slurm')
        with open(slurm_file, 'w') as fo:
            fo.write(f'#!/bin/bash'+'\n')
            fo.write(f'#SBATCH --job-name={job_name}'+'\n')
            fo.write(f'#SBATCH --account={budget_account}'+'\n')
            fo.write(f'#SBATCH --time={run_timelen}'+'\n')
            fo.write(f'#SBATCH --ntasks={cores_pernode}'+'\n')
            fo.write(f'#SBATCH --mem-per-cpu={mem_per_cpu}'+'\n')
            fo.write("\n")


            fo.write(f'module --quiet purge'+'\n')
            fo.write(f'set -o errexit # exit on errors'+'\n')
            fo.write(f'module load h5py/2.10.0-foss-2020a-Python-3.8.2'+'\n')
            fo.write(f'module load pfft-python/0.1.21-foss-2020a-Python-3.8.2'+'\n')
            fo.write(f'set -x'+'\n')
            fo.write("\n")

            fo.write(f'OUT_DIR=${{SLURM_SUBMIT_DIR}}'+'\n')
        
             
            fo.write(f'CASE_{i}=${{OUT_DIR}}/case_{i}'+'\n')
            fo.write("\n")
        
             
            fo.write(f'mkdir -p ${{CASE_{i}}}' +'\n')
            fo.write("\n")
            
            #### rename the toml file and put in separate folders 
            target_file = os.path.join(f'{work_folder}', f'case_{i}.toml')  
            files = glob(f'{work_folder}/{toml_label}{i}_*') 
            if len(files) != 1:
                raise ValueError('error: non unique toml file!!!')
            ori_file =  files[0]
            shutil.copyfile(ori_file, target_file)
            #########################################################

            fo.write(f'date'+'\n')
            
            fo.write(f'srun -n {cores_pernode} python3 ${{HOME}}/HyMD-2021/hymd/main.py case_{i}.toml  {IN_H5} --logfile=log_{i}.txt --verbose 2 --velocity-output --destdir ${{CASE_{i}}} --seed {seed}  ')
            
            fo.write("\n")
        





def add_lift_binary_and_plot(water_types, interface_types, base_colums, kai_input_csv, kai_full_csv, kai_full_pdf, weight, num_lift_try=3, kai_limit=25, kai_neg_limit=-3):
    """
    avoid the case that head parts stay inside the cluster
    """
    df_kai =  pd.read_csv( kai_input_csv ) 
    column_names =  list(df_kai.columns)
    _Cut_Melt = 5 
    keep_list = column_names[:_Cut_Melt]

    df_kai_out = df_kai[keep_list].copy()
    
    ### waterish_phase = water_types+ interface_types 
    ### for simpliciy just add the water_types, interface tune together as others  
    ### other wise, can/ need to further tune the interface with water phase 
    waterish_phase = water_types 

    for column in base_colums[1:]:
        for i in range(num_lift_try): 
            new_array = []    
            for index, row in df_kai.iterrows():
                head = row['HName']
                tail = row['TName']
                kai_value = row[column]
                if (head in waterish_phase and tail not in waterish_phase) or (tail in waterish_phase and head not in waterish_phase):
                    print(f'tune       {head}  {tail}')
                    kai_value = kai_value + (i+1)*weight 
                else: 
                    print(f'not change {head}  {tail}')
                new_array.append(kai_value)

            df_kai_out[f'{column}Lift{i+1}'] = new_array
            
    df_kai_out.to_csv(kai_full_csv, index=False)
    
     
    column_names =  list(df_kai_out.columns)
    print(column_names)
    # ['HName', 'TName', 'MtnHeadName', 'MtnTailName', 'BiasDist', 'MtnKai', 'MtnKaiNoS']
    _Cut_Melt = 5 
    print(f'WARNING, _Cut_Melt in this function is set to {_Cut_Melt} by default')
    keep_list = column_names[:_Cut_Melt]
    melt_list = column_names[_Cut_Melt:]
    #print(keep_list)
    #print(melt_list)
    df_kai_melt = pd.melt(df_kai_out, id_vars=keep_list, value_vars=melt_list, var_name='KaiType',value_name='KaiValue')
    #print(df_kai_melt)
    
    """
    ############## two labels: add bias and add calm
    ### bias label 
    label_array = [] 
    for item in df_kai_melt['KaiType'].to_numpy():
        if item == 'MtnKai':
            label_array.append('MtnKai')
        elif item == 'MtnKaiNoS':
            label_array.append('MtnKaiNoS')
        else:
            label_array.append( item[:-8]) ## remove the CalmNeg* 
    df_kai_melt['BiasLabel'] = label_array
    
    
    ### calm label 
    label_array = [] 
    for item in df_kai_melt['KaiType'].to_numpy():
        if item == 'MtnKai':
            label_array.append('')# -2
        elif item == 'MtnKaiNoS':
            label_array.append('')# -1 
        else:
            label_array.append( int( item[-1]) ) 
    df_kai_melt['CalmNegLabel'] = label_array
    """

    g = sns.FacetGrid(df_kai_melt, col="HName", hue="KaiType")
    g.map(sns.scatterplot, "BiasDist", "KaiValue",  alpha=.7) # MtnKai MtnKaiNoS
    g.add_legend()
    g.set(xticks=[-6, -4, -2, 0, 2, 4, 6], yticks=[-12, 0, 25])
    
    #print(g.axes)
    #print(shape(g.axes))
    for item in g.axes:
        for ax in item: 
            ax.axhline(0, ls='-.')
            ax.axhline(kai_limit, ls='-.')
            ax.axhline(kai_neg_limit, ls='-.')

    g.savefig(kai_full_pdf)
    
    
    
    ##################### Simple initial load and plot ################
    #df_kai =  pd.read_csv( input_kai_csv )
    ### df_atomtype_list   =  df_atomtype_record.to_numpy().tolist()
    #column_names =  list(df_kai.columns)
    #print(column_names)
    ## ['HName', 'TName', 'MtnHeadName', 'MtnTailName', 'BiasDist', 'MtnKai', 'MtnKaiNoS']
    #_Cut_Melt = 5 
    #print(f'WARNING, _Cut_Melt in this function is set to {_Cut_Melt} by default')
    #keep_list = column_names[:_Cut_Melt]
    #melt_list = column_names[_Cut_Melt:]
    ##print(keep_list)
    ##print(melt_list)
    #df_kai_melt = pd.melt(df_kai, id_vars=keep_list, value_vars=melt_list, var_name='KaiType',value_name='KaiValue')
    ##print(df_kai_melt)
    #g = sns.FacetGrid(df_kai_melt, col="HName", hue="KaiType",)
    #g.map(sns.scatterplot, "BiasDist", "KaiValue",  alpha=.7) # MtnKai MtnKaiNoS
    #g.add_legend()
    #g.set(xticks=[-6, -4, -2, 0, 2, 4, 6], yticks=[-12, 0, 12])
    #g.savefig(kai_full_pdf)
    #################################################################


def add_lift_binary2_and_plot(water_types, interface_types, base_colums, kai_input_csv, kai_full_csv, kai_full_pdf, weight, num_lift_try=3, kai_limit=25, kai_neg_limit=-3):
    """
    THIS ONE compared with add_lift_binary_and_plot
    increase the repulsion between intereace types with the core part 
    """
    df_kai =  pd.read_csv( kai_input_csv ) 
    column_names =  list(df_kai.columns)
    _Cut_Melt = 5 
    keep_list = column_names[:_Cut_Melt]

    df_kai_out = df_kai[keep_list].copy()
    
    waterish_phase = water_types+ interface_types 
    

    for column in base_colums[1:]:
        for i in range(num_lift_try): 
            new_array = []    
            for index, row in df_kai.iterrows():
                head = row['HName']
                tail = row['TName']
                kai_value = row[column]
                if (head in water_types and tail not in water_types) or (tail in water_types and head not in water_types):
                    print(f'tune       {head}  {tail}')
                    kai_value = kai_value + (i+1)*weight 

                if (head in interface_types and tail not in waterish_phase) or (tail in interface_types and head not in waterish_phase):
                    kai_value = kai_value + (i+1)*weight 

                new_array.append(kai_value)

            df_kai_out[f'{column}LiftB{i+1}'] = new_array
            
    df_kai_out.to_csv(kai_full_csv, index=False)
    
     
    column_names =  list(df_kai_out.columns)
    print(column_names)
    # ['HName', 'TName', 'MtnHeadName', 'MtnTailName', 'BiasDist', 'MtnKai', 'MtnKaiNoS']
    _Cut_Melt = 5 
    print(f'WARNING, _Cut_Melt in this function is set to {_Cut_Melt} by default')
    keep_list = column_names[:_Cut_Melt]
    melt_list = column_names[_Cut_Melt:]
    #print(keep_list)
    #print(melt_list)
    df_kai_melt = pd.melt(df_kai_out, id_vars=keep_list, value_vars=melt_list, var_name='KaiType',value_name='KaiValue')
    #print(df_kai_melt)
    

    g = sns.FacetGrid(df_kai_melt, col="HName", hue="KaiType")
    g.map(sns.scatterplot, "BiasDist", "KaiValue",  alpha=.7) # MtnKai MtnKaiNoS
    g.add_legend()
    g.set(xticks=[-6, -4, -2, 0, 2, 4, 6], yticks=[-12, 0, 25])
    
    #print(g.axes)
    #print(shape(g.axes))
    for item in g.axes:
        for ax in item: 
            ax.axhline(0, ls='-.')
            ax.axhline(kai_limit, ls='-.')
            ax.axhline(kai_neg_limit, ls='-.')

    g.savefig(kai_full_pdf)


def add_scale_kai_and_plot(input_kai_csv, out_kai_csv, kai_full_pdf, numera_list, denom=6):
    """
    [2021-08-12]
    based on add_bias_kai_and_plot

    # We select a refence eps_ij, e.g. eps_ref = eps_ww = P4 P4 4.999864058869093 
    # then others are defined as 
    # -6*( eps_ij - eps_ref)

    denom is the 6
    """

    
    df_kai =  pd.read_csv( input_kai_csv ) 
    df_kai_out = df_kai.copy()

    ### generate new columns that add the bias 
    data = df_kai[['BiasDist', 'MtnKaiNoS']].to_numpy() 
    for numera in numera_list: 
        new_array = data[:,1]*(numera/denom)
        df_kai_out[f'AddScale{numera}Over{denom}'] = new_array 
    
    df_kai_out.to_csv(out_kai_csv, index=False)
    
    
    column_names =  list(df_kai_out.columns)
    print(column_names)
    # ['HName', 'TName', 'MtnHeadName', 'MtnTailName', 'BiasDist', 'MtnKai', 'MtnKaiNoS']
    _Cut_Melt = 5 
    print(f'WARNING, _Cut_Melt in this function is set to {_Cut_Melt} by default')
    keep_list = column_names[:_Cut_Melt]
    melt_list = column_names[_Cut_Melt:]
    #print(keep_list)
    #print(melt_list)
    df_kai_melt = pd.melt(df_kai_out, id_vars=keep_list, value_vars=melt_list, var_name='KaiType',value_name='KaiValue')
    #print(df_kai_melt)
     
    ############## one label: add scale
    ### scale label 
    label_array = [] 
    for item in df_kai_melt['KaiType'].to_numpy():
        if item == 'MtnKai':
            label_array.append('MtnKai')
        elif item == 'MtnKaiNoS':
            label_array.append('MtnKaiNoS')
        else:
            label_array.append( item) ## remove the CalmNeg* 
    df_kai_melt['ScaleLabel'] = label_array
    
    
    g = sns.FacetGrid(df_kai_melt, col="HName", hue='ScaleLabel')
    g.map(sns.scatterplot, "BiasDist", "KaiValue",  alpha=.7) # MtnKai MtnKaiNoS
    g.add_legend()
    g.set(xticks=[-6, -4, -2, 0, 2, 4, 6], yticks=[-12, 0, 25])
    
    #print(g.axes)
    #print(shape(g.axes))
    for item in g.axes:
        for ax in item: 
            ax.axhline(0, ls='-.')
            #ax.axhline(kai_limit, ls='-.')
            #ax.axhline(kai_neg_limit, ls='-.')

    g.savefig(kai_full_pdf)
    
    
    

def add_bias_kai_and_plot(input_kai_csv,out_kai_csv, kai_full_pdf, weight,  weightcalm=0, num_bias_try = 3, kai_limit=35, num_calm_neg_try=3, kai_neg_limit=-5.0, linearity=1):
    """
    - The weight tunes how strong the bias is added
    - the nonlinearity (exponential) tunes the bias length

    - input_kai_csv is the input without adding bias
    - out_kai_csv add extra columns with the bias 

    - num_calm_neg_try ==> this one infact tunes the surface tension 
    - kai_neg_limit=-3.0 ===> very negative kai mess up the interacton matrix 

    """
    
    df_kai =  pd.read_csv( input_kai_csv ) 
    df_kai_out = df_kai.copy()

    ### generate new columns that add the bias 
    data = df_kai[['BiasDist', 'MtnKaiNoS']].to_numpy() 
    for i in range(num_bias_try):
        new_array = data[:,1] + np.abs(data[:,0])*( weight*(i+1) )
        new_array[new_array > kai_limit] = kai_limit 

        #df_kai_out[f'Add{linearity}Bias{i+1}'] = new_array ## this is the one without calm (taking care of very negative values)
        for j in range(num_calm_neg_try):
            new_array[new_array < kai_neg_limit] += weight*j
            df_kai_out[f'Add{linearity}Bias{i+1}CalmNeg{j}'] = new_array        
            



    df_kai_out.to_csv(out_kai_csv, index=False)
    
    
    column_names =  list(df_kai_out.columns)
    print(column_names)
    # ['HName', 'TName', 'MtnHeadName', 'MtnTailName', 'BiasDist', 'MtnKai', 'MtnKaiNoS']
    _Cut_Melt = 5 
    print(f'WARNING, _Cut_Melt in this function is set to {_Cut_Melt} by default')
    keep_list = column_names[:_Cut_Melt]
    melt_list = column_names[_Cut_Melt:]
    #print(keep_list)
    #print(melt_list)
    df_kai_melt = pd.melt(df_kai_out, id_vars=keep_list, value_vars=melt_list, var_name='KaiType',value_name='KaiValue')
    #print(df_kai_melt)
     
    ############## two labels: add bias and add calm
    ### bias label 
    label_array = [] 
    for item in df_kai_melt['KaiType'].to_numpy():
        if item == 'MtnKai':
            label_array.append('MtnKai')
        elif item == 'MtnKaiNoS':
            label_array.append('MtnKaiNoS')
        else:
            label_array.append( item[:-8]) ## remove the CalmNeg* 
    df_kai_melt['BiasLabel'] = label_array
    
    
    ### calm label 
    label_array = [] 
    for item in df_kai_melt['KaiType'].to_numpy():
        if item == 'MtnKai':
            label_array.append('')# -2
        elif item == 'MtnKaiNoS':
            label_array.append('')# -1 
        else:
            label_array.append( int( item[-1]) ) 
    df_kai_melt['CalmNegLabel'] = label_array
    
    g = sns.FacetGrid(df_kai_melt, col="HName", row='CalmNegLabel', hue="BiasLabel")
    g.map(sns.scatterplot, "BiasDist", "KaiValue",  alpha=.7) # MtnKai MtnKaiNoS
    g.add_legend()
    g.set(xticks=[-6, -4, -2, 0, 2, 4, 6], yticks=[-12, 0, 25])
    
    #print(g.axes)
    #print(shape(g.axes))
    for item in g.axes:
        for ax in item: 
            ax.axhline(0, ls='-.')
            ax.axhline(kai_limit, ls='-.')
            ax.axhline(kai_neg_limit, ls='-.')

    g.savefig(kai_full_pdf)
    
    
    ##################### Simple initial load and plot ################
    #df_kai =  pd.read_csv( input_kai_csv )
    ### df_atomtype_list   =  df_atomtype_record.to_numpy().tolist()
    #column_names =  list(df_kai.columns)
    #print(column_names)
    ## ['HName', 'TName', 'MtnHeadName', 'MtnTailName', 'BiasDist', 'MtnKai', 'MtnKaiNoS']
    #_Cut_Melt = 5 
    #print(f'WARNING, _Cut_Melt in this function is set to {_Cut_Melt} by default')
    #keep_list = column_names[:_Cut_Melt]
    #melt_list = column_names[_Cut_Melt:]
    ##print(keep_list)
    ##print(melt_list)
    #df_kai_melt = pd.melt(df_kai, id_vars=keep_list, value_vars=melt_list, var_name='KaiType',value_name='KaiValue')
    ##print(df_kai_melt)
    #g = sns.FacetGrid(df_kai_melt, col="HName", hue="KaiType",)
    #g.map(sns.scatterplot, "BiasDist", "KaiValue",  alpha=.7) # MtnKai MtnKaiNoS
    #g.add_legend()
    #g.set(xticks=[-6, -4, -2, 0, 2, 4, 6], yticks=[-12, 0, 12])
    #g.savefig(kai_full_pdf)
    #################################################################
    


    





def pull_kai_matrix_from_martini(atomtype_csv, martini_lj_pairs,  out_kai_csv):
    """
    2021-07-08
    template kai in HyMD toml file
    template_kai  = [
                      [["T", "C"],[-10.800]],
                      [["T", "B"],[-10.800]],
                      [["T", "S"],[-0.225]]
                    ]


    epsilon sigma from c6 and c12 
    ;; gmx sigeps  -c6 0.15091 -cn 0.16267E-02  
    ;; --- c6    =  1.50910e-01, c12    =  1.62670e-03
    ;; --- sigma =  0.47000, epsilon =  3.50000

    !!! converison 
    https://en.wikipedia.org/wiki/Lennard-Jones_potential A = c12 B = c6
    sigma   =  (c12/c6)**(1/6) 
    epsilon =  c6**2 / 4c12 

    !!! calcuate kai 
    ## -6*( eps_ij - 0.5(eps_i + eps_j) ) 

    """
    
    kai_list = []
    

    df_atomtype_record =  pd.read_csv( atomtype_csv)
    df_atomtype_list   =  df_atomtype_record.to_numpy().tolist()
    #list of records print(df_atomtype_list)
    
    ## loop the atomtype_list 
    ##pair_list = list(combinations(df_atomtype_list, 2)) ### note 2021-07-19 this is to loop without of repeat
    pair_list = list(product(df_atomtype_list, repeat=2))  ### note 2021-07-19 this can loop all, full matrix
    for (item1, item2) in pair_list: 
        #print(item1, item2)
        #print(item1[4], type(item1[4]))
        #_kai_record = [item1[1], item2[1],  item1[4], item2[4], abs(item2[5]-item1[5])] 
        _kai_record = [item1[1], item2[1],  item1[4], item2[4], item2[5]-item1[5]] ## not abs() then easier to find pairs 


        ################################### THIS IS THE RAW 
        for martini_lj in martini_lj_pairs:
            if set( [item1[4], item2[4]] ) == set( [ martini_lj.vdwHeadType,  martini_lj.vdwTailType ] ):
                #print(f'found  {item1[4]}   {item2[4]}')
                ##c6  = martini_lj.vdwC6
                ##c12 = martini_lj.vdwC12
                ##epsilon = c6**2 / (4*c12)
                ##sigma   = (c12/c6)**(1/6) 
                ##print(item1[1], item2[1],  item1[4], item2[4], c6, c12, epsilon, sigma ) ## ok 
                # print(item1[1], item2[1],  item1[4], item2[4], martini_lj.vdwC6, martini_lj.vdwC12, martini_lj.epsilon, martini_lj.sigma)


                ## -6*( eps_ij - 0.5(eps_i + eps_j) ) 
                eps_ij = martini_lj.epsilon 

                for martini_lj in martini_lj_pairs:
                    if martini_lj.sametype and item1[4] == martini_lj.vdwHeadType : 
                        eps_i = martini_lj.epsilon
                        break
                
                for martini_lj in martini_lj_pairs:
                    if martini_lj.sametype and item2[4] == martini_lj.vdwHeadType : 
                        eps_j = martini_lj.epsilon
                        break
                
                kai =  -6.0 * ( eps_ij - 0.5*(eps_i + eps_j))
                ##print(item1[1], item2[1],  item1[4], item2[4], martini_lj.vdwC6, martini_lj.vdwC12, martini_lj.epsilon, martini_lj.sigma, kai)
                _kai_record.append(kai)
                ##kai_list.append(_kai_record)

                break 
        
        ##################################  remove S 
        for martini_lj in martini_lj_pairs:
            typeHead = item1[4]
            typeTail = item2[4] 
            
            ### remove S 
            if typeHead[0]=='S':
                typeHead = typeHead[1:]
            if typeTail[0]=='S':
                typeTail = typeTail[1:]
            
            if set( [typeHead , typeTail] ) == set( [ martini_lj.vdwHeadType,  martini_lj.vdwTailType ] ):

                ## -6*( eps_ij - 0.5(eps_i + eps_j) ) 
                eps_ij = martini_lj.epsilon 

                for martini_lj in martini_lj_pairs:
                    if martini_lj.sametype and typeHead == martini_lj.vdwHeadType : 
                        eps_i = martini_lj.epsilon
                        break
                
                for martini_lj in martini_lj_pairs:
                    if martini_lj.sametype and typeTail == martini_lj.vdwHeadType : 
                        eps_j = martini_lj.epsilon
                        break
                
                kai =  -6.0 * ( eps_ij - 0.5*(eps_i + eps_j))
                _kai_record.append(kai)
                kai_list.append(_kai_record)

                break 
        
        #for item in kai_list:
        #    print('the chi:   ', item)
        
        ############ out csv 
        ## see https://www.geeksforgeeks.org/make-a-pandas-dataframe-with-two-dimensional-list-python/
        ## data line e.g. ['S', 'N', 'N0', 'SQ0', -2.6247347690813374, -0.0]
        df_out = pd.DataFrame(kai_list, 
             columns =['HName', 'TName', 'MtnHeadName','MtnTailName','BiasDist', 'MtnKai', 'MtnKaiNoS']) 
        #print(df_out)
        df_out.to_csv(out_kai_csv, index=False)

        
        

def pull_kai_matrix_from_martini_new(atomtype_csv, martini_lj_pairs,  out_kai_csv, eps_ref = 4.999864):
    """
    [2021-08-12]
    based on the function pull_kai_matrix_from_martini
    checking the issue about self-interaction terms 

    !!! calcuate kai 
    ## -6*( eps_ij - 0.5(eps_i + eps_j) )  Eq  
    The fact is that Martini already give the pair interaction strenghes 
    The above Eq twists the interaction scales from Martini FF, e.g.: 
      W mix with C 
      W-W deep attractive well 
      C-C shallow attractive well 
      W-C can be zero 
      then purely the self-interaction terms 

    We select a refence eps_ij, e.g. eps_ref = eps_ww = P4 P4 4.999864058869093 
    then others are defined as 
    -6*( eps_ij - eps_ref)

    """
    
    kai_list = []
    

    df_atomtype_record =  pd.read_csv( atomtype_csv)
    df_atomtype_list   =  df_atomtype_record.to_numpy().tolist()
    #list of records print(df_atomtype_list)
    
    ## loop the atomtype_list 
    ##pair_list = list(combinations(df_atomtype_list, 2)) ### note 2021-07-19 this is to loop without of repeat
    pair_list = list(product(df_atomtype_list, repeat=2))  ### note 2021-07-19 this can loop all, full matrix
    for (item1, item2) in pair_list: 
        #print(item1, item2)
        #print(item1[4], type(item1[4]))
        #_kai_record = [item1[1], item2[1],  item1[4], item2[4], abs(item2[5]-item1[5])] 
        _kai_record = [item1[1], item2[1],  item1[4], item2[4], item2[5]-item1[5]] ## not abs() then easier to find pairs 


        ################################### THIS IS THE RAW 
        for martini_lj in martini_lj_pairs:
            if set( [item1[4], item2[4]] ) == set( [ martini_lj.vdwHeadType,  martini_lj.vdwTailType ] ):
                #print(f'found  {item1[4]}   {item2[4]}')
                ##c6  = martini_lj.vdwC6
                ##c12 = martini_lj.vdwC12
                ##epsilon = c6**2 / (4*c12)
                ##sigma   = (c12/c6)**(1/6) 
                ##print(item1[1], item2[1],  item1[4], item2[4], c6, c12, epsilon, sigma ) ## ok 
                # print(item1[1], item2[1],  item1[4], item2[4], martini_lj.vdwC6, martini_lj.vdwC12, martini_lj.epsilon, martini_lj.sigma)


                ## -6*( eps_ij - 0.5(eps_i + eps_j) ) 
                eps_ij = martini_lj.epsilon 
                #print(item1[4], item2[4],eps_ij ) 

                kai =  -6.0 * ( eps_ij - eps_ref )
                ##print(item1[1], item2[1],  item1[4], item2[4], martini_lj.vdwC6, martini_lj.vdwC12, martini_lj.epsilon, martini_lj.sigma, kai)
                _kai_record.append(kai)
                ##kai_list.append(_kai_record)

                break 
        
        ##################################  remove S 
        for martini_lj in martini_lj_pairs:
            typeHead = item1[4]
            typeTail = item2[4] 
            
            ### remove S 
            if typeHead[0]=='S':
                typeHead = typeHead[1:]
            if typeTail[0]=='S':
                typeTail = typeTail[1:]
            
            if set( [typeHead , typeTail] ) == set( [ martini_lj.vdwHeadType,  martini_lj.vdwTailType ] ):

                ## -6*( eps_ij - 0.5(eps_i + eps_j) ) 
                eps_ij = martini_lj.epsilon 

                #kai =  -6.0 * ( eps_ij - 0.5*(eps_i + eps_j))
                kai =  -6.0 * ( eps_ij - eps_ref) # _new 

                _kai_record.append(kai)
                kai_list.append(_kai_record)

                break 
        
        #for item in kai_list:
        #    print('the chi:   ', item)
        
        ############ out csv 
        ## see https://www.geeksforgeeks.org/make-a-pandas-dataframe-with-two-dimensional-list-python/
        ## data line e.g. ['S', 'N', 'N0', 'SQ0', -2.6247347690813374, -0.0]
        df_out = pd.DataFrame(kai_list, 
             columns =['HName', 'TName', 'MtnHeadName','MtnTailName','BiasDist', 'MtnKai', 'MtnKaiNoS']) 
        #print(df_out)
        df_out.to_csv(out_kai_csv, index=False)
        
    


def gmx_record_match(strg, search=re.compile(r'[^a-zA-Z0-9]').search):
    """
    requires the strg is (line start with) a-z or number 
    """
    return not bool(search(strg))

def load_martini_ff_vdw(itp_file):
    """
    2021-07-07 before, in all the load gmx itp files, 
    loop lines and break with empty line; 
    which requires no unnecessary gap lines 
    === Now then loop the until the next section  [ ] or end  

    target section: [ nonbond_params ]
    
    """
    itpVdwAtom_list = []
    
    itpVdwPair_list = []
    with open(itp_file,'r') as f:
        data = f.readlines()
        index_nonbond = [x for x in range(len(data)) if '[ nonbond_params ]' in data[x].lower()]
        index_section = [x for x in range(len(data)) if '[ ' in data[x].lower()]
        #print(index_nonbond)
        #print(index_section)
        start = index_nonbond[0] # section [ nonbond_params ] 
        try:
            end   =  index_section[  index_section.index(start) + 1 ] # section next to [ nonbond_params ]
            data_target = data[start:end]
        except: 
            end   =  -1 
            data_target = data[start:] ### checked already that if put -1; then the last line -1 is not included     
        #print( start, end)
        
        ############### access [ nonbond_params ] section 
        ### for line in data[start:end]: ## to index_pairs[0] or -1 
        for line in data_target: 
            ##print(line )
            demoline = line.split()
            #print(demoline)
            if demoline: ### this is a list,  will filter the empty list, i.e. empty line  
                #print(demoline)
                if gmx_record_match(demoline[0]):
                    #print(demoline) ### test ok 
                    vdw_head   = demoline[0] 
                    vdw_tail   = demoline[1] 
                    vdw_func   = int(demoline[2])
                    vdw_c6     = float(demoline[3])
                    vdw_c12    = float(demoline[4])
                    itpVdwPair_list.append (ItpVdwPair(vdw_head, vdw_tail, vdw_func, vdw_c6, vdw_c12))    
                    #print(vdw_head, vdw_tail, vdw_func, vdw_c6, vdw_c12) # test ok 
    return itpVdwPair_list




def read_top_molecules(topfile):
    """
    only search for the [ molecules ] seciton

    ! currently this funciton does work for: no empty line in the end 
    ! the problem is the -1 in the data[index_molecules[0]:-1]
    ! add -1 means that the last item -1 is not included
    ! e.g. a = [1,2,3]
    !      a[1:-1] # [2]
    !      a[1:] # [2,3]
    ! 
    ! Thus remove the -1  
    ! 


    """
    # if file exists 
    itpMolecules = [] 
    with open(topfile,'r') as f:
        data = f.readlines()
        #print('data', data)
        index_molecules = [x for x in range(len(data)) if '[ molecules ]' in data[x].lower()]
        for line in data[index_molecules[0]:]:
            #print('here,,,', line)
            demoline = line.split()
            if line == "\n": # cutoff at the first empty line 
                break
            elif demoline[1].isdigit():
                demolist    =  line.split()
                molname   =  demolist[0]
                molnum    =  demolist[1]
                itpMolecules.append(ItpMolecule( molname,   molnum))
            else:
                continue
    return itpMolecules



def load_molecule_itp_atoms(itp_file):
    """
    HERE only read the [ atoms ] section 

    FIX: [x:-1] does not include the last item in a list 

    ! Notice that the seciton name should be gaped with space in both left and right side, e.g. [ atoms ]
    ! otherwise, will not be able to locate 
    """
    itpAtom_list = []
    
    print('now, reading the itp file:', itp_file)
    with open(itp_file,'r') as f:
        data = f.readlines()
        index_atoms = [x for x in range(len(data)) if '[ atoms ]' in data[x].lower()]
        ############### access atoms 
        for line in data[index_atoms[0]:]: 
            demoline = line.split()
            if line == "\n": # cutoff at the first empty line 
                break
            elif demoline[0].isdigit():
                demolist  =  line.split()
                index     =  int(demolist[0])
                atomtype  =  demolist[1]
                resnr     =  int(demolist[2])
                resname   =  demolist[3]
                atomname  =  demolist[4]
                cgnr      =  int(demolist[5])
                charge    =  float(demolist[6])
                mass      =  float(demolist[7])
                #print(index, atomtype)
                itpAtom_list.append (ItpAtom(index,atomtype,resnr,resname,atomname, cgnr,charge,mass))
            else:
                continue
    return itpAtom_list 


    
def load_molecule_itp_bonds(itp_file):
    """
    MissingLabel = -999
    """
    MissingLabel = -999
    
    itpBond_list = []
    with open(itp_file,'r') as f:
        data = f.readlines()
        index_bonds= [x for x in range(len(data)) if '[ bonds ]' in data[x].lower()]
        ############### access bonds
        for line in data[index_bonds[0]:-1]: ## to index_pairs[0] or -1 
            demoline = line.split()
            if line == "\n": # cutoff at the first empty line 
                break
            elif demoline[0].isdigit():
                demolist  =  line.split()
                head      =  int(demolist[0])
                tail      =  int(demolist[1])
                func      =  int(demolist[2])
                try:
                    length = float(demolist[3])
                except IndexError:
                    length = MissingLabel 
                try:
                    strength = float(demolist[4])
                except IndexError:
                    strength = MissingLabel 
                itpBond_list.append (ItpBond(head,tail,func,length,strength))
            else:
                continue
    return itpBond_list 


def access_top_molecule_itps_gen_whole_atomtypeID( itpMolecules, atomcsv, top_to_itp_path=''):
    """
    redundent thing here, has to access the atomcsv to convert the typename to typeID ...
    """
    _SHIFT = 1 ### IF the id in h5 starts from 0 

    # if file exists 
    ######## generate the whole list of typename 
    whole_name_list = []  

    for molecule_group in itpMolecules:
        molecule_name = molecule_group.molname
        molecule_num  = int(molecule_group.molnum)
        ### better to get the path of top file, then add the top_to_itp_path 
        ### --> here assume the atomcsv give the absolute path 
        abs_path = os.path.dirname(atomcsv)
        _path = os.path.join(abs_path, top_to_itp_path)
        #print(_path)
        molecule_itp_file  = os.path.join(_path, f"{molecule_name}.itp" )  
        #print(  molecule_name , molecule_num, molecule_itp_file )
        itp_atom_list = load_molecule_itp_atoms( molecule_itp_file )
        
        #atomtypeName_list = [x.atomname for x in itp_atom_list]
        atomtypeName_list = [x.atomtype for x in itp_atom_list] ## 2021-06-11
        
        print('inside molecule itp:  ' , atomtypeName_list)
        whole_name_list += atomtypeName_list * molecule_num ## duplicate list https://stackoverflow.com/questions/33046980/duplicating-a-list-n-number-of-times
        ### test 
        ### a = [ 1, 2]
        ### b = a * 3 
    
    
    ####### map the name list to typeID list 
    ####### construct mapping dict from the dataframe, to convert the typename to typeID 
    df_atomtype =  pd.read_csv( atomcsv )
    atom_typename_typeID_dict = dict(zip( df_atomtype.atomName.values.tolist(), df_atomtype.atomtypeID.values.tolist() ))
    print( atom_typename_typeID_dict )
    ####### mapping oepration 
    whole_atomtypeID_list = np.vectorize(atom_typename_typeID_dict.get)(np.array(whole_name_list)) ## option1 
    #print(whole_atomtypeID_list)
    for _ in whole_atomtypeID_list-_SHIFT:
        print(_)
     
    return (np.array(whole_name_list,dtype="S5"), np.array(whole_atomtypeID_list)-_SHIFT) 
     

def access_top_molecule_itps_gen_whole_atomBondIdx( itpMolecules, atomcsv, top_to_itp_path=''):
    """
    - editted 2021-06-14 add _MAX_N_BONDS variable 

    """
    _SHIFT = -1 ### IF the id in h5 starts from 0 
    # if file exists 
    ######## generate the whole list of bonded indices 
    whole_atom_bonded_indices_list = []  
    
    _MAX_N_BONDS = 4 
    
    _continue = 0 ## continue from different type of molecules 
    for molecule_group in itpMolecules:
        molecule_name = molecule_group.molname
        molecule_num  = int(molecule_group.molnum)
        print('--------------------------------', molecule_name, molecule_num)
        abs_path = os.path.dirname(atomcsv)
        _path = os.path.join(abs_path, top_to_itp_path)
        molecule_itp_file  = os.path.join(_path, f"{molecule_name}.itp" )  
        #print(  molecule_name , molecule_num, molecule_itp_file )
        itp_atom_list = load_molecule_itp_atoms( molecule_itp_file )
        atom_id_list = [x.index for x in itp_atom_list]
        
        molecule_bonded_indices = []
        if len(atom_id_list)==1:
            print('one bead molecule')
            itp_bond_list = []
            _array = [0]*_MAX_N_BONDS #[0,0,0]
            molecule_bonded_indices.append(_array)
        else:
            itp_bond_list = load_molecule_itp_bonds( molecule_itp_file )
            for id in atom_id_list:
                #_array = [0,0,0]
                _array = [0]*_MAX_N_BONDS
                for bond in itp_bond_list:
                    pair = [ int(bond.head), int(bond.tail) ]
                    if id in pair :
                        ##_array.extend(pair)
                        ##_array = list(set(_array)) 
                        ##_array.remove(id)
                        pair.remove(id)
                        _array = pair + _array
                        _array = _array[:_MAX_N_BONDS] ##_array = _array[:3] ####

                ##molecule_bonded_indices.append( np.array(_array))
                molecule_bonded_indices.append( _array )


        molecule_bonded_indices = np.array(molecule_bonded_indices) + _SHIFT
        print('***** molecule type *****')
        print(molecule_bonded_indices)

        ### mask 
        _filter_value = -1 
        molecule_bonded_indices_mask = ma.masked_values(molecule_bonded_indices, _filter_value)
        print(molecule_bonded_indices_mask)

        for _duplicate in np.arange(molecule_num):
            ##duplidate_ma = (molecule_bonded_indices_mask + _duplicate*len(atom_id_list)).data
            ##print(type(duplidate_ma))
            _shift_indices =  (molecule_bonded_indices_mask + _duplicate*len(atom_id_list) + _continue).data   
            ##print(molecule_name, _shift_indices)
            whole_atom_bonded_indices_list.extend( list(_shift_indices) ) 
            ##time.sleep(1)
        
        _continue += molecule_num * len(atom_id_list)
    
    return np.array(whole_atom_bonded_indices_list)


def access_top_molecule_itps_gen_whole_atomcharges( itpMolecules, atomcsv, top_to_itp_path=''):
    ## after write the code use the cmd+shift+2 to active the autoDocstring ###
    
    # if file exists 
    ######## generate the whole list of typename 
    whole_charge_list = []  
    
    for molecule_group in itpMolecules:
        molecule_name = molecule_group.molname
        molecule_num  = int(molecule_group.molnum)
        ### better to get the path of top file, then add the top_to_itp_path 
        ### --> here assume the atomcsv give the absolute path 
        abs_path = os.path.dirname(atomcsv)
        _path = os.path.join(abs_path, top_to_itp_path)
        #print(_path)
        molecule_itp_file  = os.path.join(_path, f"{molecule_name}.itp" ) 
        #print(  molecule_name , molecule_num, molecule_itp_file )
        itp_atom_list = load_molecule_itp_atoms( molecule_itp_file )
        atomcharge_list = [x.charge for x in itp_atom_list]
        print('inside molecule itp:  ' , atomcharge_list)
        whole_charge_list += atomcharge_list * molecule_num ## duplicate list https://stackoverflow.com/questions/33046980/duplicating-a-list-n-number-of-times
    
    return np.array(whole_charge_list)
    

def access_top_molecule_itps_gen_whole_moleculeindex( itpMolecules, atomcsv, top_to_itp_path=''):
    """
    #
    # This one generate the molecule index using numpy array thus starts from  0 
    #- no need to add -1 shift 
    
    ### 2021-06-14 BEEEE careful about this funciton see the changes in 2021-06-14
    
    """

    # if file exists 
    ######## generate the whole list of typename 
    whole_molecule_index_list = []  
    
    def duplicate(testList, n): ## ## duplicate list https://stackoverflow.com/questions/33046980/duplicating-a-list-n-number-of-times
        return [ele for ele in testList for _ in range(n)]

    _continue = 0 
    for molecule_group in itpMolecules:
        molecule_name = molecule_group.molname
        molecule_num  = int(molecule_group.molnum)
        ### better to get the path of top file, then add the top_to_itp_path 
        ### --> here assume the atomcsv give the absolute path 
        abs_path = os.path.dirname(atomcsv)
        _path = os.path.join(abs_path, top_to_itp_path)
        #print(_path)
        molecule_itp_file  = os.path.join(_path, f"{molecule_name}.itp" )  
        #print(  molecule_name , molecule_num, molecule_itp_file )
        itp_atom_list = load_molecule_itp_atoms( molecule_itp_file )
        atom_num_in_mol = len( itp_atom_list )
        
        #_array_sgl = np.arange( molecule_num ) + _continue 
        _array_sgl = np.arange( molecule_num ) +1+ _continue  ## editted 2021-06-14; careful about from one type molecule to another 
        _array_dpl = duplicate (_array_sgl, atom_num_in_mol)
        whole_molecule_index_list += _array_dpl

        _continue = _array_sgl[-1]
        
        print(_array_dpl[0],_array_dpl[-1])

    return np.array(whole_molecule_index_list)-1 # -1 is the molecule id starts from zero # editted 2021-06-14;
    


def gmx_to_h5( out_h5_file, in_gro_file, top, atomcsv, electric_label=False): ## groAtoms, box, top_mol_dict, charges=False
    """
    - reference: fort5_to_hdf5.py
                  
    """
    groAtoms, box = loadGroPosition(in_gro_file)
    n_atoms = len(groAtoms)

    n_molecules = 0 
    #### access [ molecules ] in the top file 
    if isinstance(top, str):
        print('top input as string, may judge whether file exist')
        itpMolecule_list = read_top_molecules(top)
        for molecule in itpMolecule_list:
            k, v = molecule.molname, int(molecule.molnum)
            print( k, v)
            n_molecules += v 
    
    #for k, v in top_mol.items():
    #        #print(k, v)
    #        resname_list += [k] * v
    #for k, v in top_mol_dict.items():
    #    n_molecules += v 
    print(n_molecules)

    _ndim = 3
    f_h5 = h5py.File(out_h5_file, "w")
    
    dset_pos = f_h5.create_dataset("coordinates", (1, n_atoms, _ndim), dtype="float64")
    dset_vel = f_h5.create_dataset("velocities", (1, n_atoms, _ndim), dtype="float64")
    dset_types = f_h5.create_dataset("types", (n_atoms,), dtype="i")
    dset_molecule_index = f_h5.create_dataset("molecules", (n_atoms,), dtype="i")
    dset_indices = f_h5.create_dataset("indices", (n_atoms,), dtype="i")
    dset_names = f_h5.create_dataset("names", (n_atoms,), dtype="S5")
    dset_bonds = f_h5.create_dataset("bonds", (n_atoms, 3), dtype="i")

    if electric_label:
        ## can get the charge array first, then assign directly 
        ## hf.create_dataset('charge', data=charges, dtype='float32')
        dset_charges = f_h5.create_dataset("charge", (n_atoms,), dtype="float32")
    ###### get box size from all.gro
    ### h5dump -N "box" all.h5
    f_h5.attrs["box"] = box
    
    ###### get number molecules from top 
    ### h5dump -N "n_molecules" all.h5
    f_h5.attrs["n_molecules"] = n_molecules
    
    
    ###### get coordinates from the all.gro 
    ### h5dump -d "coordinates" all.h5
    _frame = 0 
    for idx, atom in enumerate(groAtoms):
        if idx % 10 == 0:
            print(idx)
        dset_pos[_frame, idx, :] = np.array( [atom.x, atom.y, atom.z] )
    
    
    ##### get indices  
    ### h5dump -d "indices" all.h5
    dset_indices[:] = np.arange(len(groAtoms))
    ### could also assing value when create_dataset 

    ##### get types # atomtypes
    ##### CAN DO IT via loopling the gromAtoms, 
    ##### 
    ### NTOE the types here are defined by the numbers 
    ### e.g. as the atomtypeID 
    ### h5dump -d "types" all.h5
    ###
    ##### get names 
    ### h5dump -d "names" all.h5
    (dset_names[:], dset_types[:])= access_top_molecule_itps_gen_whole_atomtypeID(itpMolecule_list, atomcsv)
    
    ##### get molecule index ## this can be done via using the groAtoms, which need to take care of the 99999 limits
    ##### now mimicking the types, loop the top and itp
    ### h5dump -d "molecules" all.h5
    ### dset_molecule_index[:] = access_top_molecule_itps_gen_whole_moleculeindex( itpMolecule_list, atomcsv)
    dset_molecule_index[...] = access_top_molecule_itps_gen_whole_moleculeindex( itpMolecule_list, atomcsv)
    ##### get bonds 
    ##### h5dump -d "bonds" all.h5
    ##### WARNING -- bonded terms is done like
    ##### _index: x y -1 
    ##### e.g. following the occam way, not necessarily. also in git issue:  #Make HyMD output valid HyMD input #79
    ##### the input h5 should be further determined 
    ##### NOW maximumlly follows what is used 
    ##### 
    ##### ASSIGN dataset data 1,in create stage 2. assign values item by items 3. assign a new/same type using ellipisis?
    ##### https://docs.h5py.org/en/stable/high/dataset.html
    ##### https://www.slideshare.net/HDFEOS/dan-kahnpythonhdf5-2
    ##### 
    dset_bonds[...]  = access_top_molecule_itps_gen_whole_atomBondIdx(itpMolecule_list, atomcsv)
    
    ##### Get charges ; add the decleration above 
    ##### Then work with run.sh config.toml  and try to run 
    ##### h5dump -d "charge" all.h5
    ##### charges = np.vectorize(particle_type_charge_dict.get)(np.array(names)) ## option1 
    #####
    if electric_label:
        dset_charges[...] = access_top_molecule_itps_gen_whole_atomcharges(itpMolecule_list, atomcsv)
        





def gmx_to_h5_from_more_hand( out_h5_file, in_gro_file, top, atomcsv, alias_mol_dict, electric_label=False): ## groAtoms, box, top_mol_dict, charges=False
    """
    - reference: fort5_to_hdf5.py

    In this function, we will read the top file and gro file from the GMX martini
    top file: we only access the molecule name and molecule num 
       - as the itp in hpf is redefiend, then we have to give the correpinding itp files for different molecule types;
         which can be done providing an dictionary==  alias_mol_dict 
       - 
    gro file, we get the coordiantes 
       

                  
    """

    groAtoms, box = loadGroPosition(in_gro_file)
    n_atoms = len(groAtoms)
    
    print('----------- load gro file: number of atoms , box size ')
    print(n_atoms, box)
    
    
    n_molecules = 0 
    #### access [ molecules ] in the top file 
    if isinstance(top, str):
        print('----------- load top file: ')
        itpMolecule_list = read_top_molecules(top)
        for molecule in itpMolecule_list:
            k, v = molecule.molname, int(molecule.molnum)
            print( k, v)
            n_molecules += v 
    print('----------- total number of molecules: ')
    print(n_molecules)
    
    
    _ndim = 3
    f_h5 = h5py.File(out_h5_file, "w")
    
    dset_pos = f_h5.create_dataset("coordinates", (1, n_atoms, _ndim), dtype="float64")
    dset_vel = f_h5.create_dataset("velocities", (1, n_atoms, _ndim), dtype="float64")
    dset_types = f_h5.create_dataset("types", (n_atoms,), dtype="i")
    dset_molecule_index = f_h5.create_dataset("molecules", (n_atoms,), dtype="i")
    dset_indices = f_h5.create_dataset("indices", (n_atoms,), dtype="i")
    dset_names = f_h5.create_dataset("names", (n_atoms,), dtype="S5")
    MAX_N_BONDS = 4
    dset_bonds = f_h5.create_dataset("bonds", (n_atoms, MAX_N_BONDS), dtype="i")
    
    
    if electric_label:
        ## can get the charge array first, then assign directly 
        ## hf.create_dataset('charge', data=charges, dtype='float32')
        dset_charges = f_h5.create_dataset("charge", (n_atoms,), dtype="float32")
    ###### get box size from all.gro
    ### h5dump -N "box" all.h5
    f_h5.attrs["box"] = box
    
    ###### get number molecules from top 
    ### h5dump -N "n_molecules" all.h5
    f_h5.attrs["n_molecules"] = n_molecules
    
    
    ###### get coordinates from the all.gro 
    ### h5dump -d "coordinates" all.h5
    _frame = 0 
    for idx, atom in enumerate(groAtoms):
        #if idx % 10 == 0:
            #print(idx)
        dset_pos[_frame, idx, :] = np.array( [atom.x, atom.y, atom.z] )
    
    
    ##### get indices  
    ### h5dump -d "indices" all.h5
    dset_indices[:] = np.arange(len(groAtoms))
    ### could also assing value when create_dataset 
    

    ##### get types # atomtypes
    ##### CAN DO IT via loopling the gromAtoms, 
    ##### 
    ### NTOE the types here are defined by the numbers 
    ### e.g. as the atomtypeID 
    ### h5dump -d "types" all.h5
    ###
    ##### get names 
    ### h5dump -d "names" all.h5
     
    #################### 2021-06-11 
    # ---> add name alias 
    
    # test
    #print(itpMolecule_list)
    for item in itpMolecule_list :
        #print(item.__dict__)
        try: 
            item.molname = alias_mol_dict[item.molname] 
        except ValueError:
            print('corresponding itp is not found !!!! ')
    
    #for item in itpMolecule_list :
    #    print(item.__dict__)

    #for item in itpMolecule_list:
    #   print(item.__dict__)
    
    (dset_names[:], dset_types[:])= access_top_molecule_itps_gen_whole_atomtypeID(itpMolecule_list, atomcsv)
    #access_top_molecule_itps_gen_whole_atomtypeID(itpMolecule_list, atomcsv)
    #for _ in zip(dset_names, dset_types):
    #    print(_)

    ##### get molecule index ## this can be done via using the groAtoms, which need to take care of the 99999 limits
    ##### now mimicking the types, loop the top and itp
    ### h5dump -d "molecules" all.h5
    ### dset_molecule_index[:] = access_top_molecule_itps_gen_whole_moleculeindex( itpMolecule_list, atomcsv)
    dset_molecule_index[...] = access_top_molecule_itps_gen_whole_moleculeindex( itpMolecule_list, atomcsv)
    
    

    
    ##### get bonds 
    ##### h5dump -d "bonds" all.h5
    ##### WARNING -- bonded terms is done like
    ##### _index: x y -1 
    ##### e.g. following the occam way, not necessarily. also in git issue:  #Make HyMD output valid HyMD input #79
    ##### the input h5 should be further determined 
    ##### NOW maximumlly follows what is used 
    ##### 
    ##### ASSIGN dataset data 1,in create stage 2. assign values item by items 3. assign a new/same type using ellipisis?
    ##### https://docs.h5py.org/en/stable/high/dataset.html
    ##### https://www.slideshare.net/HDFEOS/dan-kahnpythonhdf5-2
    ##### 
    dset_bonds[...]  = access_top_molecule_itps_gen_whole_atomBondIdx(itpMolecule_list, atomcsv)
    

    ##### Get charges ; add the decleration above 
    ##### Then work with run.sh config.toml  and try to run 
    ##### h5dump -d "charge" all.h5
    ##### charges = np.vectorize(particle_type_charge_dict.get)(np.array(names)) ## option1 
    #####
    if electric_label:
        dset_charges[...] = access_top_molecule_itps_gen_whole_atomcharges(itpMolecule_list, atomcsv)
     

