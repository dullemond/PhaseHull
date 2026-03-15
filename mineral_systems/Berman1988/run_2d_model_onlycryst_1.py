import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpltern   # https://mpltern.readthedocs.io/en/latest/index.html
import phasehull as ph
from model_Berman1988 import Berman88

relevel    = False
#relevel    = True

T          = 800.+273.15

nrefine    = 4

endmembers = ["SiO2","CaO","Al2O3"]
#endmembers = ["SiO2","CaO","MgO"]
#endmembers = ["SiO2","MgO","Al2O3"]
#endmembers = ["CaO","MgO","Al2O3"]
endmnames  = endmembers
b          = Berman88(endmembers,T)

crystaldb  = ph.CrystalDatabase(b.mdb)

phull      = ph.PhaseHull(endmembers,crystaldb,None)

isel_allc  = phull.select_simplices_of_a_given_kind('allcryst')

# Plot them however you like:

fig = plt.figure()
ax  = fig.add_subplot(projection="ternary")
fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)
# Plot the crystal coexistence triangles
for isim in isel_allc:
    x = phull.thesimplices[-1]['x'][isim]
    x = np.vstack((x,x[0,:]))
    ax.fill(x[:,0],x[:,1],x[:,2],color='C3',alpha=0.2)
    ax.plot(x[:,0],x[:,1],x[:,2],color='C3')
# Plot the crystals
db=phull.crystals[0].dbase
db=db[db['stable']]
x = np.stack(db['x'])
ax.scatter(x[:,0],x[:,1],x[:,2],s=64.0, c='C3', edgecolors="k",zorder=100)
size   = 8
voff   = 0.06
for i in range(len(x)):
    if x[i,0]>0.7:
        off=-voff
    else:
        off=voff
    ax.text(x[i,0]+off,x[i,1]-off/2,x[i,2]-off/2,db['Abbrev'].iloc[i],ha='center',va='center',size=size)
ax.set_tlabel(endmnames[0])
ax.set_llabel(endmnames[1])
ax.set_rlabel(endmnames[2])
plt.savefig(f'fig_ternary_T={T:.0f}.pdf')

plt.show()
