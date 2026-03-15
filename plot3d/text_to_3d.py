# Created by claude.ai 2026-03-03, with some help by C.P. Dullemond

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import pathpatch_2d_to_3d


def rotation_matrix_from_axes(normal, up):
    """
    Build a 3x3 rotation matrix whose columns are the (right, up, normal)
    orthonormal basis vectors.  This maps patch-local XY → world plane.
    """
    n = np.asarray(normal, dtype=float);  n /= np.linalg.norm(n)
    u = np.asarray(up,     dtype=float)
    u -= np.dot(u, n) * n;                u /= np.linalg.norm(u)
    r = np.cross(u, n)                    # right = up × normal
    return np.column_stack([r, u, n])     # columns: x→r, y→u, z→n


def text_to_3d(ax, text, pos, normal, up, size=1.0, color='black', alpha=1.0, scale=None):
    """
    Render text as a 3D-oriented filled patch.

    Parameters
    ----------
    ax     : Axes3D instance
    text   : str
    pos    : (3,) anchor point in 3D world coordinates (bottom-left of text)
    normal : (3,) normal to the text plane
    up     : (3,) up direction for the text within the plane
    size   : float — font size scaling
    color  : fill colour
    alpha  : opacity
    """
    # 0. Define a rescaling if necessary
    if scale is None: scale = (1.,1.,1.)
    scale = np.array(scale)
    
    # 1. Create the patch in the XY plane at the origin
    tp    = TextPath((0, 0), text, size=size)
    patch = PathPatch(tp, facecolor=color, edgecolor=color,
                      alpha=alpha, linewidth=0)
    ax.add_patch(patch)

    # 2. Use pathpatch_2d_to_3d with zdir='z' (identity orientation)
    pathpatch_2d_to_3d(patch, z=0, zdir='z')

    # 3. Now grab the 3D vertices that were just written and rotate + translate them
    R   = rotation_matrix_from_axes(normal, up)
    pos = np.asarray(pos, dtype=float)

    verts3d  = patch._segment3d            # shape (N, 3), z==0 for all points
    verts3d  = (R @ np.array(verts3d).T).T # rotate
    verts3d *= scale[None,:]               # Scale
    verts3d += pos                         # translate
    patch._segment3d = verts3d

    return patch

if __name__=='__main__':

    # ── demo ──────────────────────────────────────────────────────────────────────
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')

    # Sphere surface
    u_ = np.linspace(0, 2*np.pi, 40)
    v_ = np.linspace(0,   np.pi, 20)
    U, V = np.meshgrid(u_, v_)
    X, Y, Z = np.cos(U)*np.sin(V), np.sin(U)*np.sin(V), np.cos(V)
    ax.plot_trisurf(X.ravel(), Y.ravel(), Z.ravel(), alpha=0.2, color='steelblue')

    # Label flat on the XY-plane
    text_to_3d(ax, "Flat on XY",
               pos=(-.5, -1.8, 0),
               normal=(0, 0, 1),
               up=(0, 1, 0),
               size=0.4, color='navy')

    # Label on a vertical XZ-plane
    text_to_3d(ax, "Vertical XZ",
               pos=(-0.5, 1.1, -0.5),
               normal=(0, 1, 0),
               up=(0, 0, 1),
               size=0.4, color='darkred')

    # Label on a 45°-tilted plane
    text_to_3d(ax, "Tilted 45°",
               pos=(-1.8, 0.2, 0.5),
               normal=(1, 1, 0),
               up=(0, 0, 1),
               size=0.4, color='darkgreen')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('3D-oriented text via PathPatch')
    plt.tight_layout()
    plt.show()
