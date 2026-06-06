import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpltern   # https://mpltern.readthedocs.io/en/latest/index.html
import phasehull as ph
from phasehull.phasehull_colors import linecolors,fillcolors
from phasehull.phasehull_compgeom import *
from mineral_systems.Berman1983 import Berman83

relevel    = False
#relevel    = True

T          = 1400.+273.15
#T          = 600.+273.15

nrefine    = 0

components = ["SiO2","CaO","Al2O3","MgO"]
compnames  = components
b          = Berman83(components,T)
def Gfunc(x):
    return b.Gfunc(x)

crystaldb  = ph.CrystalDatabase(b.mdb)
liquid     = ph.Liquid('magma',components,Gfunc)

#nres0      = 30
nres0      = 50
phull      = ph.PhaseHull(components,crystaldb,liquid,nres0=nres0,nrefine=nrefine)


isel_allc  = phull.select_simplices_of_a_given_kind('allcryst')
isel_liq   = phull.select_simplices_of_a_given_kind('liquid')
isel_3c1l  = phull.select_simplices_of_a_given_kind('cryst_3_liq_1')
isel_2c2l  = phull.select_simplices_of_a_given_kind('cryst_2_liq_2')
isel_1c3l  = phull.select_simplices_of_a_given_kind('cryst_1_liq_3')


#z        = 0.00001
z        = 0.25
#z        = (2/3)
#z        = 0.5
xcorners = np.array([[1.-z,0.,0.,z],[0.,1.-z,0.,z],[0.,0.,1.-z,z]])
o,u,w    = define_2d_plane_inside_nd_space(xcorners[:,:-1])

def plot_2d_crosssection_of_one_nd_simplex(phull,isim,o,u,w,ax,fillcolor,linecolor):
    vertices = phull.thesimplices[-1]['x'][isim][:,:-1]
    pts_2d_ordered, pts_Nd = simplex_plane_cross_section(vertices, o, u, w)
    if pts_2d_ordered is not None:
        x = np.zeros((pts_2d_ordered.shape[0]+1,pts_2d_ordered.shape[1]+1))
        x[:-1,:-1] = pts_2d_ordered
        x[:-1,-1]  = 1-x[:-1,:-1].sum(axis=-1)
        x[-1,:]    = x[0,:]
        ax.fill(x[:,0],x[:,1],x[:,2],color=fillcolor)
        ax.plot(x[:,0],x[:,1],x[:,2],color=linecolor)
    

h = 0.07  # Vertical spacing of annotations
fig = plt.figure()
ax  = fig.add_subplot(projection="ternary",ternary_sum=1-z)
fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)

# Plot the crystal coexistence triangles
for isim in isel_allc:
    plot_2d_crosssection_of_one_nd_simplex(phull,isim,o,u,w,ax,fillcolors['allcryst'],linecolors['allcryst'])
for isim in isel_3c1l:
    plot_2d_crosssection_of_one_nd_simplex(phull,isim,o,u,w,ax,fillcolors['cryst_3_liq_1'],linecolors['cryst_3_liq_1'])
for isim in isel_2c2l:
    plot_2d_crosssection_of_one_nd_simplex(phull,isim,o,u,w,ax,fillcolors['cryst_2_liq_2'],linecolors['cryst_2_liq_2'])
for isim in isel_1c3l:
    plot_2d_crosssection_of_one_nd_simplex(phull,isim,o,u,w,ax,fillcolors['cryst_1_liq_3'],linecolors['cryst_1_liq_3'])
for isim in isel_liq:
    plot_2d_crosssection_of_one_nd_simplex(phull,isim,o,u,w,ax,fillcolors['liquid'],linecolors['liquid'])

ax.set_tlabel(compnames[0])
ax.set_llabel(compnames[1])
ax.set_rlabel(compnames[2])
ax.text(1,0.5,-0.5,f'T = {T:.0f} K ({T-273.15:.0f} C)',ha='left')
ax.text(1,-0.5,0.5,f'nres0 = {nres0}',ha='right')
#ax.text(1-h,-0.5+0.5*h,0.5+0.5*h,f'Refinement levels: {nrefine}',ha='right')
ax.text(1-h,-0.5+0.5*h,0.5+0.5*h,r'$x_{\mathrm{MgO}}$'+f'={z:.2g}',ha='right')
plt.savefig(f'fig_quaternary_z={z:.2g}_T={T:.0f}.pdf')

plt.show()
