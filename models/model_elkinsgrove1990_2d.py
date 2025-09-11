import numpy as np
import mpltern   # https://mpltern.readthedocs.io/en/latest/index.html
import phasehull as ph
from mineral_systems.ElkinsGrove1990 import ElkinsGrove90
import matplotlib.pyplot as plt

T          = 900. + 273.15
P          = 1.

endmembers = ["CaAl2Si2O8","NaAlSi3O8","KAlSi3O8"]  #   Anorthite, Albite, Orthoclase
endmnames  = ["Anorthite", "Albite",   "Orthclase"]

f          = ElkinsGrove90(endmembers,T,P)
def Gfunc(x):
    return f.Gfunc(x)

liquid     = ph.Liquid('feldspar',endmembers,Gfunc)

phull      = ph.PhaseHull(endmembers,None,liquid,nres0=30,nrefine=4)

# Request the binodal curve coordinates:

xb = phull.get_x_values_of_binodal_curves(stride=1)

# Request the tie line coordinates, with a stride

xt = phull.get_x_values_of_tie_lines(stride=64)

# Plot them however you like:

fig = plt.figure()
ax  = fig.add_subplot(projection="ternary")
fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)
ax.fill([1,0,0],[0,1,0],[0,0,1],color='#F0D080')
for stype in xb:
    for igroup in range(len(xb[stype])):
        x = phull.complete_x(xb[stype][igroup])
        ax.plot(x[:,0],x[:,1],x[:,2],color='#900000')
        #ax.fill(x[:,0],x[:,1],x[:,2],color='#F0D0A0')
        ax.fill(x[:,0],x[:,1],x[:,2],color='#F0E0B0')
        ax.fill(x[:,0],x[:,1],x[:,2],color='#F0F0C0')
for stype in xt:
    for igroup in range(len(xt[stype])):
        for itie in range(len(xt[stype][igroup])):
            x = phull.complete_x(xt[stype][igroup][itie])
            ax.plot([x[0,0],x[1,0]],[x[0,1],x[1,1]],[x[0,2],x[1,2]],color='C2',marker='o')
ax.set_tlabel(endmnames[0])
ax.set_llabel(endmnames[1])
ax.set_rlabel(endmnames[2])
plt.savefig(f'fig_feldspar_T={T:.0f}.pdf')
plt.show()
