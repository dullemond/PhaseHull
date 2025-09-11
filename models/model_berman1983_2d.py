import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpltern   # https://mpltern.readthedocs.io/en/latest/index.html
import phasehull as ph
from phasehull.phasehull_colors import linecolors,fillcolors
from mineral_systems.Berman1983 import Berman83

relevel    = False
#relevel    = True

T          = 1400.+273.15

nrefine    = 4

endmembers = ["SiO2","CaO","Al2O3"]
#endmembers = ["SiO2","CaO","MgO"]
#endmembers = ["SiO2","MgO","Al2O3"]
#endmembers = ["CaO","MgO","Al2O3"]
endmnames  = endmembers
b          = Berman83(endmembers,T)
def Gfunc(x):
    return b.Gfunc(x)

crystaldb  = ph.CrystalDatabase(b.mdb)
liquid     = ph.Liquid('magma',endmembers,Gfunc)

nres0      = 30
phull      = ph.PhaseHull(endmembers,crystaldb,liquid,nres0=nres0,nrefine=nrefine)

isel_allc  = phull.select_simplices_of_a_given_kind('allcryst')
isel_liq   = phull.select_simplices_of_a_given_kind('liquid')
isel_2c1l  = phull.select_simplices_of_a_given_kind('cryst_2_liq_1')
isel_1c2l  = phull.select_simplices_of_a_given_kind('cryst_1_liq_2')
isel_inm3  = phull.select_simplices_of_a_given_kind('inmisc_liquids_3phase')

# Request the binodal curve coordinates:

xb = phull.get_x_values_of_binodal_curves(stride=1)

# Request the tie line coordinates, with a stride

xt = phull.get_x_values_of_tie_lines(stride=4)

# Request the liquidus

liquidus_ipts,liquidus_x = phull.get_liquidus_ipts_and_x_2d()

# Plot them however you like:

h = 0.07  # Vertical spacing of annotations
fig = plt.figure()
ax  = fig.add_subplot(projection="ternary")
fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)
for x in liquidus_x:
    ax.fill(x[:,0],x[:,1],x[:,2],color=fillcolors['liquid'])
# Plot the binodal lines
for stype in xb:
    for igroup in range(len(xb[stype])):
        x = phull.complete_x(xb[stype][igroup])
        ax.plot(x[:,0],x[:,1],x[:,2],color=linecolors['binodal'])
# Plot the tie lines
for stype in xt:
    for igroup in range(len(xt[stype])):
        for itie in range(len(xt[stype][igroup])):
            x = phull.complete_x(xt[stype][igroup][itie])
            ax.plot([x[0,0],x[1,0]],[x[0,1],x[1,1]],[x[0,2],x[1,2]],color=linecolors['tieline_c1l2'],marker='o',ms=2,linewidth=0.5)
# Plot the 3 liquid inmiscibility triangles
for isim in isel_inm3:
    x = phull.thesimplices[-1]['x'][isim]
    x = np.vstack((x,x[0,:]))
    ax.fill(x[:,0],x[:,1],x[:,2],color=fillcolors['inmisc_liquids_3phase'])
    ax.plot(x[:,0],x[:,1],x[:,2],color=linecolors['inmisc_liquids_3phase'])
# Plot the 2 crystal 1 liquid coexistence triangles
for isim in isel_2c1l:
    x = phull.thesimplices[-1]['x'][isim]
    x = np.vstack((x,x[0,:]))
    ax.fill(x[:,0],x[:,1],x[:,2],color=fillcolors['cryst_2_liq_1'])
    ax.plot(x[:,0],x[:,1],x[:,2],color=linecolors['cryst_2_liq_1'])
# Plot the 1 crystal 2 liquid coexistence triangles
for isim in isel_1c2l:
    x = phull.thesimplices[-1]['x'][isim]
    x = np.vstack((x,x[0,:]))
    ax.fill(x[:,0],x[:,1],x[:,2],color=fillcolors['cryst_1_liq_2'])
    #ax.plot(x[:,0],x[:,1],x[:,2],color=linecolors['cryst_1_liq_2'])
# Plot the crystal coexistence triangles
for isim in isel_allc:
    x = phull.thesimplices[-1]['x'][isim]
    x = np.vstack((x,x[0,:]))
    ax.fill(x[:,0],x[:,1],x[:,2],color=fillcolors['allcryst'])
    ax.plot(x[:,0],x[:,1],x[:,2],color=linecolors['allcryst'])
# Plot the crystals
db=phull.crystals[0].dbase
db=db[db['stable']]
x = np.stack(db['x'])
ax.scatter(x[:,0],x[:,1],x[:,2],s=64.0, c=linecolors['crystal'], edgecolors="k",zorder=100)
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
#ax.text(1,0.5,-0.5,f'T = {T} K',ha='left')
ax.text(1,0.5,-0.5,f'T = {T:.0f} K ({T-273.15:.0f} C)',ha='left')
ax.text(1,-0.5,0.5,f'nres0 = {nres0}',ha='right')
ax.text(1-h,-0.5+0.5*h,0.5+0.5*h,f'Refinement levels: {nrefine}',ha='right')
plt.savefig(f'fig_ternary_tielines_T={T:.0f}.pdf')

fig = plt.figure()
ax  = fig.add_subplot(projection="ternary")
fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)
for stype in xb:
    for igroup in range(len(xb[stype])):
        x = phull.complete_x(xb[stype][igroup])
        ax.plot(x[:,0],x[:,1],x[:,2],color='C0')
x   = phull.complete_x(phull.thepoints[-1][:,:-1])
ax.scatter(x[:,0],x[:,1],x[:,2],marker='.',s=1,color='C1')
ax.set_tlabel(endmnames[0])
ax.set_llabel(endmnames[1])
ax.set_rlabel(endmnames[2])
ax.text(1,0.5,-0.5,f'T = {T:.0f} K ({T-273.15:.0f} C)',ha='left')
ax.text(1,-0.5,0.5,f'nres0 = {nres0}',ha='right')
ax.text(1-h,-0.5+0.5*h,0.5+0.5*h,f'Refinement levels: {nrefine}',ha='right')
plt.savefig(f'fig_ternary_refinement_T={T:.0f}.pdf')

plt.show()
