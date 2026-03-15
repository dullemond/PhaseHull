import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ternary
import phasehull as ph
from model_GhiorsoSack1995 import GhiorsoSack95

#relevel    = False
relevel    = True

MELTS      = False
#MELTS      = True

#T          = 2100.+273.15
#T          = 2000.
T          = 1250.+273.15
#T          = 3200.

components = ["CaSiO3","SiO2"]
#components = ["SiO2","Al2O3"]

gs         = GhiorsoSack95(components,T,MELTS=MELTS)
def Gfunc(x):
    return gs.Gfunc(x)

crystaldb  = ph.CrystalDatabase(gs.mdb)
liquid     = ph.Liquid('magma',components,Gfunc)

phull      = ph.PhaseHull(components,crystaldb,liquid,nres0=100)

isel_allc  = phull.select_simplices_of_a_given_kind('allcryst')
isel_liq   = phull.select_simplices_of_a_given_kind('liquid')
isel_cltie = phull.select_simplices_of_a_given_kind('tieline_c1l1')
isel_inmis = phull.select_simplices_of_a_given_kind('tieline_c0l2')

simplices  = phull.thesimplices[-1]

phull.define_a_relevelling_plane(components=components)

G_liq      = phull.thepoints_liq[-1][:,-1]
G_cryst    = phull.thepoints_cryst[-1][:,-1]
ylabel     = 'G [kJ/mol]'

if relevel:
    simplices['G'] -= phull.G_relevelling_plane(simplices['x'])
    G_liq          -= phull.G_relevelling_plane(phull.thepoints_liq[-1][:,:-1])
    G_cryst        -= phull.G_relevelling_plane(phull.thepoints_cryst[-1][:,:-1])
    ylabel          = r'$\bar G$ [kJ/mol]'

#colors     = {'allcryst':'C1','liquid':'C0','cryst_1_liq_1':'C4','crystals':'C3','inmisc_liquids':'C9'}
colors     = {'allcryst':'C1','liquid':'deepskyblue','tieline_c1l1':'C4','crystals':'C3','tieline_c0l2':'C9'}
labels     = {'allcryst':'cryst-cryst tie line','liquid':'liquid','tieline_c1l1':'cryst-liquid tie line','tieline_c0l2':'inmisc liquids'}
unit       = 1e3

plt.figure()
plt.plot(phull.thepoints_liq[-1][:,0],G_liq/unit,':',color=colors['liquid'])
plt.plot(phull.thepoints_cryst[-1][:,0],G_cryst/unit,'D',color=colors['crystals'])
for i in isel_allc:  plt.plot([simplices['x'][i,0,0],simplices['x'][i,1,0]],[simplices['G'][i,0]/unit,simplices['G'][i,1]/unit],'.-',color=colors['allcryst'],label=labels['allcryst']); labels['allcryst']=None
for i in isel_liq:   plt.plot([simplices['x'][i,0,0],simplices['x'][i,1,0]],[simplices['G'][i,0]/unit,simplices['G'][i,1]/unit],color=colors['liquid'],label=labels['liquid']); labels['liquid']=None
for i in isel_cltie: plt.plot([simplices['x'][i,0,0],simplices['x'][i,1,0]],[simplices['G'][i,0]/unit,simplices['G'][i,1]/unit],'.-',color=colors['tieline_c1l1'],label=labels['tieline_c1l1']); labels['tieline_c1l1']=None
for i in isel_inmis: plt.plot([simplices['x'][i,0,0],simplices['x'][i,1,0]],[simplices['G'][i,0]/unit,simplices['G'][i,1]/unit],'.-',color=colors['tieline_c0l2'],label=labels['tieline_c0l2']); labels['tieline_c0l2']=None
plt.xlabel('x')
plt.ylabel(ylabel)
ytxt = plt.gca().get_ylim()[0] + 3
plt.text(0,ytxt,ph.latexify_chemical_formula(components[1]),ha='center')
plt.text(1,ytxt,ph.latexify_chemical_formula(components[0]),ha='center')
ytxt = plt.gca().get_ylim()[1] - 6
plt.text(0.65,ytxt,f'T = {T:.0f} K')
plt.text(0.65,ytxt-4,f'P = 1 bar')
for icr,row in crystaldb.dbase.iterrows():
    plt.text(row['x'][0],(row['mfDfG']-phull.G_relevelling_plane(row['x']))/unit+2,row['Abbrev'],ha='center',size=7)
plt.legend()
if relevel:
    srel='_rel'
else:
    srel=''
plt.savefig(f'fig_binary_{components[0]}_{components[1]}_x_G_T={T:.0f}'+srel+'.pdf')
plt.show()
