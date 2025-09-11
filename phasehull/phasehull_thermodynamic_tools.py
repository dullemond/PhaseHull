#---------------------------------------------------------------------------
#                  Part of PhaseHull, a simple python package
#                    to compute equilibrium phase diagrams
#
#                           (C) C. P. Dullemond
#                      Heidelberg University, Germany
#                                June 2025
#
# This module contains thermodynamic tools that can be used as an
# alternative to (or a test of) the convex hull algorithm. It uses more
# conventional method of computing, e.g., the liquidus of a system. But
# these tools do not form a complete set that can replace the convex
# hull algorithm.
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import phasehull as ph

def find_liquidus_x_of_a_crystal_given_TP_and_dx(model,mineral,T,P,dx,nitermax=32):
    """
    Given a crystal with fixed composition from the mineral database of the model
    (name of the crystal is mineral), and given a T and P, this function will
    attempt to find the location of the liquidus belonging to this crystal, by
    searching in direction dx starting from the crystal composition, and finding
    the location where the chemical potential of the liquid with respect to the
    stoichiometry of the crystal equals the chemical potential of the crystal.
    This is done with the root-finding algorithm brentq of scipy.optimize.

    Arguments:

      model     The mineral system model class (such as Berman83 class from the
                model_Berman1983.py).

      mineral   The acronym of the mineral for which the liquidus is to be found.
                They can be found in the minerals.fwf file in the column 'Abbrev'.

      T         Temperature [K]

      P         Pressure [bar]

      dx        Direction in mole fraction space. Array of length number of
                endmembers. Must sum to 0. The liquidus will then be sought
                along the line xs + s * dx, where xs is the composition of
                the crystal, and s is a scalar obeying 0<=s<=dist, where dist
                is the distance from xs along this direction to the edge of
                the domain (will be calculated internally). The length of the
                dx vector does not matter.

      nitermax  If the G surface of the liquid has a complex non-convex shape,
                then the brentq may not be able to find a solution at first.
                A maximum of nitermax times the dist will be halved and a new
                attempt will be made. Only if nitermax is reached without
                a successful root found, the algorithm will give up and
                return None.

    Returns:

      xl        A mole fraction vector of the location of the liquidus. If
                it cannot find a liquidus, it will return None.
    """
    from scipy.optimize import brentq
    assert np.abs(dx.sum())<1e-10, 'Error: Direction dx does not sum to 0.'
    nendm   = len(model.endmselect)
    Rgas    = 8.314
    # Make sure that the model is reset to the correct temperature and pressure
    model.reset(T,P)
    # Get the chemical potential and composition of the crystal solid
    mn      = model.mdb.set_index('Abbrev').loc[mineral]
    nu      = ph.dissect_oxide(mn['Formula'],endmembers=model.endmselect)['nuendm']
    xs      = nu/nu.sum()
    mucryst = model.get_mu0_at_T(model.mdb,mineral,T)
    # Find the distance to the edge of the domain
    dist    = 1e99
    iiend   = -1
    for iend in range(nendm):
        if dx[iend]!=0:
            s = -xs[iend]/dx[iend]
            if s==0 and dx[iend]<0:
                return None
            if s>0 and s<dist:
                dist  = s
                iiend = iend
    assert iiend>=0 and dist<1e90, 'Weird error'
    # Get the G values of the liquid end members
    Gliqend = np.zeros(nendm)
    for iend in range(nendm):
        xend = np.zeros((1,nendm))
        xend[0,iend]  = 1.
        Gliqend[iend] = model.compute_G_of_liquid_mixture(model.ldb,T,xend,model.endmselect)[0]
    # Set up the function to find the root of
    def fun(s):
        x     = np.zeros([1,nendm])
        x[0,:]= xs + dx*s
        gamma = model.margules.get_activity_coefficients_of_components(x,T,P).T
        muliq = (nu*(Gliqend+Rgas*T*np.log(x*gamma+1e-99))).sum(axis=-1)[0]
        return muliq-mucryst
    # Check that brentq can solve it
    fs = fun(0.)
    if fs<0:
        return None   # The crystal itsel is molten, no liquidus exists
    for iter in range(nitermax):
        fe = fun(dist)
        if fe*fs<0:
            s  = brentq(fun,0,dist)
            xl = xs + s*dx
            return xl
        dist *= 0.5
    return None
