import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import sys

### process read file 
inFile = './'+'check-energy.txt'
with open(inFile) as f:
    lines = (line for line in f if not ( line.startswith('2021-') or line.startswith('           step')) )
    data  = np.loadtxt(lines )
    df = pd.DataFrame({'step':data[:,0],'totalE':data[:,3],'potE':data[:,3],'fieldE':data[:,6], 'fieldQE':data[:,7]})
    #df = pd.DataFrame({'totalE':data[:,3],'potE':data[:,3],'fieldE':data[:,6], 'fieldQE':data[:,7]})


# Plot y1 vs x in blue on the left vertical axis.
ax = plt.subplot(1, 1, 1)
plt.xlabel("x")
#plt.xlim(0, 10)
#plt.xticks(range(11))
#plt.tick_params(axis="x", pad=8)
#plt.ylabel("Blue", color="b")
#plt.ylim(0, 500)
#plt.yticks(range(0, 501, 100))
#ax.yaxis.set_minor_locator(AutoMinorLocator(4))
#plt.tick_params(axis="y", labelcolor="b", pad=8)
plt.ylabel('total E')
plt.plot(df.step.values, df.totalE.values, "k-.", linewidth=2, label='toal E')

plt.legend()

## Plot y2 vs x in red on the right vertical axis.
ax2 = plt.twinx()
plt.ylabel("Red", color="r")
#plt.ylim(0, 100)
#plt.yticks(range(0, 101, 20))
#ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
#plt.tick_params(axis="y", labelcolor="r", pad=8)
plt.plot(df.step.values, df.fieldQE.values, "r-s", linewidth=2,label='elec E')
plt.plot(df.step.values, df.fieldE.values, "g-", linewidth=2, label='field E')

plt.ylabel('elec E / field E')

plt.legend()

#plt.savefig("Two axes with minor ticks.png", dpi=75, format="png")
#plt.close()

plt.savefig('check-energy.pdf',bbox_inches='tight')
### 

