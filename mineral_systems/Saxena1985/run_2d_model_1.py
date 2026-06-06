import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpltern   # https://mpltern.readthedocs.io/en/latest/index.html
import phasehull as ph
from phasehull.phasehull_colors import linecolors,fillcolors
from model_Saxena1985 import Saxena85

relevel    = False
#relevel    = True

T          = 1100.
P          = 15e3
s          = Saxena85(T,P)

components = ["Mg2Si2O6","Fe2Si2O6","Ca2Si2O6"]
compnames  = ["En","Fs","Wo"]
comporder  = [2,0,1]

nres0      = 100 #20
nyh        = nres0+4
nyv        = nres0
yh         = np.linspace(0,1,nyh)
yv         = np.linspace(0,1,nyv)
yt_Wo      = yv[None,:]*0.5
yt_En      = (1-yh[:,None])*(1-yt_Wo)
yt_Fs      = yh[:,None]*(1-yt_Wo)
yt_Wo      = yt_Wo + 0*yt_Fs
ygrid      = np.stack((yt_En.flatten(),yt_Fs.flatten(),yt_Wo.flatten())).T

solsols    = [None,None]
solsols[0] = ph.SolidSolution('ortho',components,components,s.Gfunc_ortho,s.mdb,ygrid=ygrid)
solsols[1] = ph.SolidSolution('clino',components,components,s.Gfunc_clino,s.mdb,ygrid=ygrid)

phull      = ph.PhaseHull(components,solsols=solsols)

isel_solsol  = phull.select_simplices_of_a_given_kind('solsol')
isel_sscoex  = phull.select_simplices_of_a_given_kind('solsol_coexist')
isel_ssinms  = phull.select_simplices_of_a_given_kind('solsol_inmisc')

xb = phull.get_x_values_of_binodal_curves(stride=1)
xt = phull.get_x_values_of_tie_lines(stride=8)

# Plot them however you like:

colors = {'ortho':'#F1CDB1','clino':'#A6DEDD', 'coexist':'#C3DE9F', 'ortho_inmisc':'#DDBDA3', 'clino_inmisc':'#9FBDDD'}
labels = {'ortho':'ortho',  'clino':'clino', 'coexist':'coexist', 'ortho_inmisc':'ortho_inmisc', 'clino_inmisc':'clino_inmisc'}
h = 0.07  # Vertical spacing of annotations
fig = plt.figure()
ax  = fig.add_subplot(projection="ternary")
fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)
# Plot the normal solsol simplices
for isim in isel_solsol:
    x = phull.thesimplices[-1]['x'][isim]
    x = np.vstack((x,x[0,:]))
    solname =  phull.thesimplices[-1]['ptnames'][isim,0]
    color = colors[solname[1:]]
    label = labels[solname[1:]]
    ax.fill(x[:,comporder[0]],x[:,comporder[1]],x[:,comporder[2]],color=color,label=label)
    labels[solname[1:]]=None
# Plot the normal solsol inmiscibility gap simplices
for isim in isel_ssinms:
    x = phull.thesimplices[-1]['x'][isim]
    x = np.vstack((x,x[0,:]))
    solname =  phull.thesimplices[-1]['ptnames'][isim,0]
    color = colors[solname[1:]+'_inmisc']
    label = labels[solname[1:]+'_inmisc']
    ax.fill(x[:,comporder[0]],x[:,comporder[1]],x[:,comporder[2]],color=color,label=label)
    labels[solname[1:]+'_inmisc']=None
# Plot the coexisting solsol simplices
for isim in isel_sscoex:
    x = phull.thesimplices[-1]['x'][isim]
    x = np.vstack((x,x[0,:]))
    label = labels['coexist']
    ax.fill(x[:,comporder[0]],x[:,comporder[1]],x[:,comporder[2]],color=colors['coexist'],label=label)
    labels['coexist'] = None
# Plot the binodal lines
for stype in xb:
    for igroup in range(len(xb[stype])):
        x = phull.complete_x(xb[stype][igroup])
        ax.plot(x[:,comporder[0]],x[:,comporder[1]],x[:,comporder[2]],color=linecolors['binodal'])
# Plot the tie lines
for stype in xt:
    for igroup in range(len(xt[stype])):
        for itie in range(len(xt[stype][igroup])):
            x = phull.complete_x(xt[stype][igroup][itie])
            ax.plot([x[0,comporder[0]],x[1,comporder[0]]],[x[0,comporder[1]],x[1,comporder[1]]],[x[0,comporder[2]],x[1,comporder[2]]],color=linecolors['tieline_c1l2'],marker='o',ms=2,linewidth=0.5)
ax.set_tlabel(compnames[comporder[0]])
ax.set_llabel(compnames[comporder[1]])
ax.set_rlabel(compnames[comporder[2]])
ax.text(0.5,0.5,0.0,'Di ',ha='right')
ax.text(0.5,0.0,0.5,' Hd',ha='left')
ax.text(1,0.5,-0.5,f'T = {T:.0f} K, P = {P/1e3:.0f} kbar',ha='left')
ax.text(1,-0.5,0.5,f'nres0 = {nres0}',ha='right')
ax.plot([0.5,0.5],[0,0.5],[0.5,0],color='black',linewidth=0.5)
plt.legend()
plt.savefig(f'fig_saxena85_ternary_solsol_T={T:.0f}_P={P:.3g}.pdf')
