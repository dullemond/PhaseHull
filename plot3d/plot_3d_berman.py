import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import mpltern   # https://mpltern.readthedocs.io/en/latest/index.html
import phasehull as ph
from phasehull.phasehull_colors import linecolors,fillcolors
from mineral_systems.Berman1983 import Berman83
from phasehull.phasehull_colors import linecolors,fillcolors
from phasehull.phasehull_support import latexify_chemical_formula
from text_to_3d import *
from copy import deepcopy

relevel    = False
#relevel    = True

azim       = -90
###elev       = 90
elev       = 1

T          = 900.+273.15
#T          = 1600.+273.15
scale      = 1e3

nrefine    = 2

components = ["SiO2","CaO","Al2O3"]
#components = ["SiO2","CaO","MgO"]
#components = ["SiO2","MgO","Al2O3"]
#components = ["CaO","MgO","Al2O3"]
compnames  = components
b          = Berman83(components,T)
def Gfunc(x):
    return b.Gfunc(x)

crystaldb  = ph.CrystalDatabase(b.mdb)
liquid     = ph.Liquid('magma',components,Gfunc)

nres0      = 15
phull      = ph.PhaseHull(components,crystaldb,liquid,nres0=nres0,nrefine=nrefine)
phull.define_a_relevelling_plane(components=components)

idxorder   = [2,0,1]
points     = phull.thepoints[-1]
x          = points.copy()
x[:,2]     = 1-x[:,0]-x[:,1]
x2d        = x[:,idxorder[0]] + 0.5*x[:,idxorder[1]]
y2d        = (np.sqrt(3)/2)*x[:,idxorder[1]]
G          = points[:,-1]-phull.G_relevelling_plane(x)
triangles  = phull.thesimplices[-1]['ipts']

colors     = phull.thesimplices[-1]['stype'].copy()
cset = set(colors)
for c in cset:
    colors[colors==c] = fillcolors[c]
ecolors    = phull.thesimplices[-1]['stype'].copy()
cset = set(ecolors)
linecolors['liquid']='C0'
for c in cset:
    ecolors[ecolors==c] = linecolors[c]
ewidths    = phull.thesimplices[-1]['stype'].copy()
for c in cset:
    if c=='liquid':
        ewidths[ewidths==c] = 0.2
    else:
        ewidths[ewidths==c] = 0.6
ewidths = ewidths.astype(float)

fig = plt.figure(figsize=(5,5))
fig.subplots_adjust(left=0.0,right=1.0,bottom=0,top=1.)
ax  = fig.add_subplot(projection='3d',computed_zorder=False)
ax.set_proj_type('ortho')

aspect = [4,3.5,4]   # Somehow the projection on the screen was not truly square, so hence 3.5
ax.set_box_aspect(aspect, zoom=1.3)
ax.view_init(azim=azim,elev=elev)
plt.draw()

Gvalid = phull.thesimplices[-1]['G']-phull.G_relevelling_plane(phull.thesimplices[-1]['x'])
G_min  = Gvalid.min()
G_max  = Gvalid.max()
z_base = (G_min - 1e4)/scale  # Move triangle 10 kJ/mol below min G
z_top  = (G_max + 1e4)/scale
z_bound= [z_base-10,z_top+10]
Gbasecorners = phull.G_relevelling_plane(np.array([[1,0,0],[0,1,0],[0,0,1]]))

# Define a consistent offset below z_base
z_range = (G_max - G_max)/scale
label_offset = 0.025 * z_range 
z_label = z_base - label_offset

# === Draw triangle at z = z_base ===
triangle_base = np.array([
    [0.0, 0.0, z_base],            # A
    [1.0, 0.0, z_base],            # B
    [0.5, np.sqrt(3)/2, z_base],   # C
    [0.0, 0.0, z_base]             # back to A to close
])
ax.plot(triangle_base[:,0], triangle_base[:,1], triangle_base[:,2], 'k-', lw=1, zorder=-1.)

# === Draw G axis ===
Gax_base = np.array([[[0.0, 0.0, z_base],[0.0, 0.0, z_top]],
                     [[1.0, 0.0, z_base],[1.0, 0.0, z_top]],
                     [[0.5, np.sqrt(3)/2, z_base],[0.5, np.sqrt(3)/2, z_top]]])
Gax_001  = ax.plot(Gax_base[0,:,0], Gax_base[0,:,1], Gax_base[0,:,2], 'k-', lw=1)
Gax_100  = ax.plot(Gax_base[1,:,0], Gax_base[1,:,1], Gax_base[1,:,2], 'k-', lw=1)
Gax_010  = ax.plot(Gax_base[2,:,0], Gax_base[2,:,1], Gax_base[2,:,2], 'k-', lw=1)
Gax_all  = [Gax_100[0],Gax_010[0],Gax_001[0]]

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

# Label on a vertical XZ-plane
ds   = 0.07
size = 0.06
tax_001 = text_to_3d(ax, latexify_chemical_formula(components[idxorder[2]]),
           pos=(-2*ds, -ds, z_label), normal=(0, 0, 1),
           up=(0, 1, 0),size=size, color='black')
tax_100 = text_to_3d(ax, latexify_chemical_formula(components[idxorder[0]]),
           pos=(1+0.5*ds, -ds, z_label), normal=(0, 0, 1),
           up=(0, 1, 0),size=size, color='black')
tax_010 = text_to_3d(ax, latexify_chemical_formula(components[idxorder[1]]),
           pos=(0.5-1*ds, np.sqrt(3)/2+ds, z_label), normal=(0, 0, 1),
           up=(0, 1, 0),size=size, color='black')
tax_all  = [tax_100,tax_010,tax_001]

# Label the z-axis
phi = 90. * np.pi/180
p   = text_to_3d(ax, r'$\hat G$',
           pos=(-ds, -ds, 0.9*z_top), normal=(0, -np.sin(phi), np.cos(phi)),
           up=(0, np.cos(phi), np.sin(phi)),size=size,
           scale=(1.,1.,(z_top-z_label)), color='black')

# Compute zorder

def compute_unitvector(azim=-90,elev=90):
    n = np.zeros(3)
    n[0] = np.cos(elev*np.pi/180.)*np.cos(azim*np.pi/180.)
    n[1] = np.cos(elev*np.pi/180.)*np.sin(azim*np.pi/180.)
    n[2] = np.sin(elev*np.pi/180.)
    return n

def convert_unitvector_to_xG_space(n,idxorder):
    # First take the n[0] and n[1] to compute an x vector (still in the wrong index order)
    dx       = np.zeros(3)
    dx[0]    = n[0] - n[1]/np.sqrt(3) # The first two elements of n are dx
    dx[1]    = n[1]*2/np.sqrt(3)
    dx[2]    = -dx[0]-dx[1]           # The sum of dx must be 0
    nxg      = np.zeros(3)
    nxg[idxorder[0]] = dx[0]
    nxg[idxorder[1]] = dx[1]
    nxg[idxorder[2]] = dx[2]
    nxg[-1] = n[-1]          # Replace the last with G
    return nxg

def shear_unitvector(nxg,Gbase,zscale):
    n     = nxg.copy()
    n[2] += n[0]*(Gbase[0]-Gbase[2])
    n[2] += n[1]*(Gbase[1]-Gbase[2])
    n[2] /= zscale
    return n

def compute_zorder(equations,azim=-90,elev=90,Gbase=[0.,0.,0.],zscale=1e3,idxorder=[0,1,2],axes=None):
    n         = compute_unitvector(azim=azim,elev=elev)
    n         = convert_unitvector_to_xG_space(n,idxorder)
    n         = shear_unitvector(n,Gbase,zscale)
    zorder    = zscale*(equations[:,:-1]*n[None,:]).sum(axis=-1)
    zordermin = zorder.min()
    zordermax = zorder.max()
    zorder    = (zorder-zordermin)/(zordermax-zordermin) - 0.5
    if axes is not None:
        for i,zo in enumerate(zorder):
            axes[i].zorder=zo
    return zorder

def Gax_zorder(azim,axes=None):
    n        = compute_unitvector(azim=azim,elev=0)
    Gax_phi  = np.array([120,0,-120])*np.pi/180.
    Gax_dir  = np.stack((np.sin(Gax_phi),np.cos(Gax_phi))).T
    zorder   = (Gax_dir*n[None,:-1]).sum(axis=-1)
    zorder[zorder<=0] = -1
    zorder[zorder>0]  = 1
    if axes is not None:
        for i,zo in enumerate(zorder):
            axes[i].zorder=zo
    return zorder

# Set the vertical range of the 3D box (important for rotation)
ax.set_zbound(z_bound[0],z_bound[1])

# Get the positions and orientations of the faces of the convex hull.
# Since the original convex hull contains also faces pointing upward,
# the simplices of the phase diagram are a subset of the simplices
# of the full convex hull. Using 'id_qhull' we can find them back.

equations = phull.thehulls[-1].equations[phull.thesimplices[-1]['id_qhull']]

# If requested, add the full liquid surface on top
addliq = False
if addliq:
    phullliq = deepcopy(phull)
    phullliq.recompute_with_selected_phases(incl_liq=True,incl_solsol=False,incl_cryst=False)
    points       = phullliq.thepoints[-1]
    xnew         = points.copy()
    xnew[:,2]    = 1-xnew[:,0]-xnew[:,1]
    x2dnew       = xnew[:,idxorder[0]] + 0.5*xnew[:,idxorder[1]]
    y2dnew       = (np.sqrt(3)/2)*xnew[:,idxorder[1]]
    Gnew         = points[:,-1]-phullliq.G_relevelling_plane(x)
    trianglesnew = phullliq.thesimplices[-1]['ipts']
    x2d          = np.hstack((x2d,x2dnew))
    y2d          = np.hstack((y2d,y2dnew))
    x            = np.vstack((x,xnew))
    G            = np.hstack((G,Gnew))
    triangles    = np.vstack((triangles,trianglesnew))
    eqsnew       = phullliq.thehulls[-1].equations[phullliq.thesimplices[-1]['id_qhull']]
    equations    = np.vstack((equations,eqsnew))

smplx_x2d = x2d[triangles].mean(axis=-1)
smplx_y2d = y2d[triangles].mean(axis=-1)
zscale    = scale*np.abs(z_bound[1]-z_bound[0])

# Now plot
#ax.plot_trisurf(Triangulation(x2d,y2d,triangles=triangles), G/scale, linewidth=1., color=colors[0], edgecolor=ecolors[0], shade=False)
#for it in range(len(triangles)):

iy = np.argsort(smplx_y2d)
simplaxes = []
for it in range(len(iy)):
    simplaxes.append(ax.plot_trisurf(Triangulation(x2d,y2d,triangles=triangles[it:it+1]), G/scale, linewidth=ewidths[it], edgecolor=ecolors[it], color=colors[it], shade=False))

zorder    = compute_zorder(equations,azim=azim,elev=elev,Gbase=Gbasecorners,zscale=zscale,idxorder=idxorder,axes=simplaxes)
Gax_zorder(azim,axes=Gax_all)
Gax_zorder(azim,axes=tax_all)

def reorient(ax,azim,elev,simplaxes,Gax_all,tax_all,Gbase,zscale,idxorder):
    ax.view_init(azim=azim,elev=elev)
    compute_zorder(equations,azim=azim,elev=elev,Gbase=Gbasecorners,zscale=zscale,idxorder=idxorder,axes=simplaxes)
    Gax_zorder(azim,axes=Gax_all)
    Gax_zorder(azim,axes=tax_all)
    plt.draw()

reorient(ax,-20,10,simplaxes,Gax_all,tax_all,Gbasecorners,zscale,idxorder)
