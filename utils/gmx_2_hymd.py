import numpy as np
import os
from .topologyParser import gmx_to_h5_from_more_hand
import pandas as pd

print("------------------- starting ")

# provide file names, type names ("W" for water), masses (72 is default), charges
work_folder = "./"
out_h5_filename = "converted.h5"
in_gro_file = "start.gro"
in_top_name = "topol.top"
atomtype_id = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
atomtype_name = np.array(["T", "C", "B", "S", "O", "M", "N", "W", "F"])
atomtype_mass = np.array([72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0])
atomtype_charge = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

print("---- generating atomtype.csv ")
df = pd.DataFrame(
    {
        "atomtypeID": atomtype_id,
        "atomName": atomtype_name,
        "atomMass": atomtype_mass,
        "atomCharge": atomtype_charge,
    }
)
atomtype_csv = os.path.join("./", "atomtype.csv")
df.to_csv(atomtype_csv, index=False)


out_h5_file = os.path.join(work_folder, out_h5_filename)
in_gro_file = os.path.join(work_folder, in_gro_file)
in_top_file = os.path.join(work_folder, in_top_name)

# Set to false if the system isn't charged
electric_label = True

# the key is the molecule resname from the gmx topology
# the value is the hPF itp file for that molecule
# eg "AZT": "AZT" --> it will read AZT.itp
alias_mol_dict = {
    "AZT": work_folder + "AZT",
    "W": work_folder + "SOL",
    "CL-": work_folder + "CL",
}

print("--------- generating h5 file ")
gmx_to_h5_from_more_hand(
    out_h5_file, in_gro_file, in_top_file, atomtype_csv, alias_mol_dict, electric_label
)
print("----------------------- done ")
