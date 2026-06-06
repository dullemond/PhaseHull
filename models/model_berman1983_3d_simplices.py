import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpltern   # https://mpltern.readthedocs.io/en/latest/index.html
import phasehull as ph
from phasehull.phasehull_colors import linecolors,fillcolors
from phasehull.phasehull_compgeom import *
from mineral_systems.Berman1983 import Berman83
from phasehull.phasehull_support import latexify_chemical_formula

relevel    = False
#relevel    = True

T          = 1400.+273.15
#T          = 600.+273.15

nrefine    = 0

components = ["CaO","Al2O3","SiO2","MgO"]
compnames  = components
b          = Berman83(components,T)
def Gfunc(x):
    return b.Gfunc(x)

crystaldb  = ph.CrystalDatabase(b.mdb)
liquid     = ph.Liquid('magma',components,Gfunc)

nres0      = 30
#nres0      = 50
phull      = ph.PhaseHull(components,crystaldb,liquid,nres0=nres0,nrefine=nrefine)


isel_allc  = phull.select_simplices_of_a_given_kind('allcryst')
isel_liq   = phull.select_simplices_of_a_given_kind('liquid')
isel_3c1l  = phull.select_simplices_of_a_given_kind('cryst_3_liq_1')
isel_2c2l  = phull.select_simplices_of_a_given_kind('cryst_2_liq_2')
isel_1c3l  = phull.select_simplices_of_a_given_kind('cryst_1_liq_3')

# Make a perfectly symmetric tetrad
hy      = np.sqrt(0.75)
dy      = -hy/3
hz      = np.sqrt(2/3)
dz      = -hz/4
corners = np.array([[-0.5,dy,dz],[0.5,dy,dz],[0.0,hy+dy,dz],[0.,0.,hz+dz]])
corners[:,:].mean(axis=0)
ifaces  = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
fig  = plt.figure(figsize=(5,5))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax   = fig.add_subplot(projection='3d')
fax  = []
for ifacs in range(len(ifaces)):
    fax.append(ax.plot_trisurf(corners[:,0],corners[:,1],corners[:,2],triangles=ifaces[ifacs:ifacs+1],color='gray',alpha=0.2,edgecolor='black',linewidth=1.2))
fax[0].set_zorder(-100.)
fax[2].set_zorder(-101.)
fax[1].set_zorder(100.)
fax[3].set_zorder(101.)

# === Hide default cartesian ticks, grids and panes ===
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.xaxis.line.set_lw(0.)
ax.yaxis.line.set_lw(0.)
ax.zaxis.line.set_lw(0.)
ax.grid(False)
ax.xaxis.pane.fill = False # Left pane
ax.yaxis.pane.fill = False # Right pane
ax.zaxis.pane.fill = False # Bottom pane
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_zbound(-0.7,0.6)
ax.set_box_aspect([1,1,1],zoom=1.5)
ax.set_xlim(-0.7,0.3)
#ax.set_zlim(-0.65,0.45)
ax.set_zlim(-0.63,0.47)

# Label on a vertical XZ-plane

eps  = 0.03
size = 12
ax.text(corners[0,0]-eps,corners[0,1]-3*eps,corners[0,2],latexify_chemical_formula(components[0]),fontsize=size)
ax.text(corners[1,0]-0.5*eps,corners[1,1]-2.5*eps,corners[1,2],latexify_chemical_formula(components[1]),fontsize=size)
ax.text(corners[2,0]+0.5*eps,corners[2,1]+eps,corners[2,2],latexify_chemical_formula(components[2]),fontsize=size)
ax.text(corners[3,0]-1.5*eps,corners[3,1],corners[3,2]+0.7*eps,latexify_chemical_formula(components[3]),fontsize=size)

# A simplex plotting function
def plot_3d_simplex(ax,corners,x,color,alpha=0.2,linewidth=1.2,shuffle=[0,1,2,3],edgecolor=None):
    if edgecolor is None:
        edgecolor=color
    assert x.shape[0]==4
    assert x.shape[1]==4
    points  = (x[:,:,None]*corners[None,:,:]).sum(axis=1)
    ifaces  = np.array([[shuffle[0],shuffle[1],shuffle[2]],[shuffle[1],shuffle[0],shuffle[3]],
                        [shuffle[2],shuffle[3],shuffle[0]],[shuffle[3],shuffle[2],shuffle[1]]])
    return ax.plot_trisurf(points[:,0],points[:,1],points[:,2],triangles=ifaces,color=color,alpha=alpha,edgecolor=edgecolor,linewidth=linewidth)

# Now plot one
# isim  = isel_allc[6]  ; color = linecolors['allcryst']
# #isim  = isel_1c3l[16] ; color = linecolors['cryst_1_liq_3']
# #isim  = isel_2c2l[16] ; color = linecolors['cryst_2_liq_2']
# #isim  = isel_3c1l[16] ; color = linecolors['cryst_3_liq_1']
# x     = phull.thesimplices[-1]['x'][isim]
# shu   = [1,0,2,3]
# simpl = plot_3d_simplex(ax,corners,x,color=color,alpha=0.2,linewidth=1.2,shuffle=shu)
# #simpl = plot_3d_simplex(ax,corners,x,color=color,alpha=1,linewidth=0,shuffle=shu)

# Plot all of a given type
# color = '#0000FF' # fillcolors['liquid']
# shu   = [1,0,2,3]
# for isim in isel_liq:
#     x     = phull.thesimplices[-1]['x'][isim]
#     simpl = plot_3d_simplex(ax,corners,x,color=color,alpha=0.1,linewidth=0,shuffle=shu)
# plt.savefig(f'fig_quaternary_liquid_T={T:.0f}.pdf')

color = '#FF0000'
shu   = [1,0,2,3]
for isim in isel_allc:
    x     = phull.thesimplices[-1]['x'][isim]
    simpl = plot_3d_simplex(ax,corners,x,color=color,alpha=0.1,linewidth=0,shuffle=shu)
plt.savefig(f'fig_quaternary_allcryst_T={T:.0f}.pdf')
