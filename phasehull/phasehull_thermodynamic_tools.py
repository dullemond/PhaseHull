#---------------------------------------------------------------------------
#                  Part of PhaseHull, a simple python package
#                    to compute equilibrium phase diagrams
#
#                           (C) C. P. Dullemond
#                      Heidelberg University, Germany
#                                June 2025
#
# This module contains thermodynamic tools that may or may not be helpful.
# For the core functionality of PhaseHull they are not necessary.
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import phasehull as ph
import pickle

#---------------------------------------------------------------------------
# Some numeric derivative tools to compute the chemical potentials and
# activity coefficients from a given G(x) or G_excess(x) function.
#
# Note that it is always better to use, instead, analytical expressions
# for the derivatives and activity coefficients. For the Margules formulation
# of the G_excess(x) function, Berman & Brown (1984) gave a full analytic
# expression valid for Margules of arbitrary order. But some papers give
# alternative expression, for which the conversion to activity coefficients
# may not always be easy to formulare. Though in most cases the papers go
# the opposite, and easier, direction by formulating a model for the
# activity coefficients, and compute the G_excess from those (using
# Eq. 4 of Berman & Brown (1984). So it should be rarely necessary to
# use the below numerical derivative tools to compute the activity
# coefficients from the G_excess, but it can be handy for testing
# purposes.
#---------------------------------------------------------------------------

def compute_compensated_derivative_of_hatG(hatGfunc,x,i,j,eps=1e-4):
    """
    The molar Gibbs free energy hatG(x) lives on the plane defined
    by x.sum()==1. If you want to compute the derivative of hatG(x)
    in direction i, you must specify how you wish to keep x.sum()==1.
    In the "compensated derivative" this is done by accompanying the
    step in i-direction by a compensating (opposite sign) step in j
    direction. See definition in Berman & Brown (1984) just above
    their equation 11.

    Arguments:

      hatGfunc     A function accepting x and returning the molar Gibbs
                   free energy at that x.

      x            A 2D array of x values with x.sum(axis=-1)==1, meaning
                   the rightmost index is the component index and the
                   leftmost index allows multiple x positions. Can also
                   be a 1D array, meaning only a single x position.

      i            Integer: The direction in which the derivative is
                   to be computed

      j            Integer: The compensating direction.

      eps          The step size for the numerical derivative. Note
                   that the best value is of the order of 1e-7 times the
                   typical magnitude of the values of hatG. Too big
                   or too small means losing precision.

    Returns:

      dhatGdx      The value(s) of the derivative.
    
    """
    assert not np.isscalar(x), 'Error: x must be at least a full x vector of all components'
    scalar = len(x.shape)==1
    if(scalar): x=np.stack([x])
    x      = x/x.sum(axis=-1)[...,None]
    xp     = x.copy()
    xp[...,i] += eps
    xp[...,j] -= eps  # The compensation direction
    G      = hatGfunc(x)
    Gp     = hatGfunc(xp)
    dGdx   = (Gp-G)/eps
    if(scalar):
        dGdx = dGdx[0]
    return dGdx

def compute_restricted_derivative_of_hatG(hatGfunc,x,i,eps=1e-4,fromcompder=False):
    """
    The molar Gibbs free energy hatG(x) lives on the plane defined
    by x.sum()==1. If you want to compute the derivative of hatG(x)
    in direction i, you must specify how you wish to keep x.sum()==1.
    In the "restricted derivative" this is done by taking first a
    step in i direction without adjusting for x.sum()==1, meaning
    we go out of the x.sum()==1-plane. Then we rescale x to obey
    x.sum()==1 again. We then return to the plane, and moved a step
    in the direction i, without having changed the ratio of the
    mole fractions of the other k!=i components. This derivative
    of hatG would be the same if we did not rescale x, but we
    assured that hatG(factor*x)=hatG(x), so that the hatG at the
    out-of-plane x+dx is the same as that if the rescaled one.

    The restricted derivative of hatG(x) is equal to:

      dhatG|                 dhatG|
      -----|      = Sum  x_k -----|
      dx_i |restr   k!=i     dx_ik|comp

    where the derivative with _comp is the compensated derivative
    of compute_compensated_derivative_of_hatG() above. The r.h.s.
    of the above equality is what is the second term of the r.h.s.
    of Eq. 15 of Berman & Brown 1984.

    Arguments:

      hatGfunc     A function accepting x and returning the molar Gibbs
                   free energy at that x.

      x            A 2D array of x values with x.sum(axis=-1)==1, meaning
                   the rightmost index is the component index and the
                   leftmost index allows multiple x positions. Can also
                   be a 1D array, meaning only a single x position.

      i            Integer: The direction in which the derivative is
                   to be computed

      eps          The step size for the numerical derivative. Note
                   that the best value is of the order of 1e-7 times the
                   typical magnitude of the values of hatG. Too big
                   or too small means losing precision.

      fromcompder  Default False: computing it directly through the
                   definition. If True, then using the sum of the
                   compensated derivatives.

    Returns:

      dhatGdx      The value(s) of the derivative.
    
    """
    assert not np.isscalar(x), 'Error: x must be at least a full x vector of all components'
    scalar = len(x.shape)==1
    if(scalar): x=np.stack([x])
    x      = x/x.sum(axis=-1)[...,None]
    if not fromcompder:
        # The direct computation
        xp     = x.copy()
        xp[...,i] += eps
        xpp    = xp/xp.sum(axis=-1)[...,None]
        G      = hatGfunc(x)
        Gp     = hatGfunc(xpp)
        dGdx   = (Gp-G)/eps
    else:
        # Using the above sum formula
        dGdx = np.zeros_like(x[...,0])
        for k in range(x.shape[-1]):
            if k!=i:
                dGdx += x[...,k]*compute_compensated_derivative_of_hatG(hatGfunc,x,i,k,eps=eps)
    if(scalar):
        dGdx = dGdx[0]
    return dGdx

def numerically_compute_chemical_potential(hatGfunc,x,i,eps=1e-4):
    """
    Compute the values of mu_i = dG/dn_i, which is by definition
    the chemical potential, from numerical derivatives of the hatG(x)
    function.

    Arguments:

      hatGfunc     A function accepting x and returning the molar Gibbs
                   free energy at that x.

      x            A 2D array of x values with x.sum(axis=-1)==1, meaning
                   the rightmost index is the component index and the
                   leftmost index allows multiple x positions. Can also
                   be a 1D array, meaning only a single x position.

      i            Integer: The component for which the R*T*log(gamma) is
                   to be computed

      eps          The step size for the numerical derivative. Note
                   that the best value is of the order of 1e-7 times the
                   typical magnitude of the values of hatG. Too big
                   or too small means losing precision.

    Note: The computation is done here directly by taking n=x and then
    varying n[i] by eps. One can, however, also use the function
    numerically_compute_rtlngamma() below and instead of hatGex give
    it hatG. That function then uses the restricted or compensated
    derivatives of hatG (instead of the dG/dn[i] as done here), but
    the result is the same (apart from numerical errors due to the
    finite difference derivative).
    """
    assert not np.isscalar(x), 'Error: x must be at least a full x vector of all components'
    scalar = len(x.shape)==1
    if(scalar): x=np.stack([x])
    x      = x/x.sum(axis=-1)[...,None]
    xp     = x.copy()
    xp[...,i] += eps
    xpsum  = xp.sum(axis=-1)
    xpp    = xp/xpsum[...,None]
    G      = hatGfunc(x)
    Gp     = xpsum*hatGfunc(xpp)
    dGdn   = (Gp-G)/eps
    if(scalar):
        dGdn = dGdn[0]
    return dGdn

def numerically_compute_rtlngamma(hatGexcessfunc,x,i,eps=1e-4):
    """
    Compute the value of R*T*ln(gamma_i) where gamma_i is the activity
    coefficient, from numerical derivatives of the hatG_excess(x)
    function.

    Arguments:

      hatGexcessfunc  A function accepting x and returning the molar Gibbs
                   free energy excess at that x. The excess is the Gibbs
                   function without the mu^0 terms and without the
                   entropy terms R*T*x_i*log(x_i). For ideal solutions
                   the hatGexcessfunc should be 0, in which case the
                   rtlngamma are also zero.

      x            A 2D array of x values with x.sum(axis=-1)==1, meaning
                   the rightmost index is the component index and the
                   leftmost index allows multiple x positions. Can also
                   be a 1D array, meaning only a single x position.

      i            Integer: The component for which the R*T*log(gamma) is
                   to be computed

      eps          The step size for the numerical derivative. Note
                   that the best value is of the order of 1e-7 times the
                   typical magnitude of the values of hatG. Too big
                   or too small means losing precision.
    """
    assert not np.isscalar(x), 'Error: x must be at least a full x vector of all components'
    scalar = len(x.shape)==1
    if(scalar): x=np.stack([x])
    x      = x/x.sum(axis=-1)[...,None]
    dGdx   = compute_restricted_derivative_of_hatG(hatGexcessfunc,x,i,eps=eps)
    rtlng  = hatGexcessfunc(x) + dGdx
    if(scalar):
        rtlng = rtlng[0]
    return rtlng

#---------------------------------------------------------------------------
# Some thermodynamic tools that can be used as an
# alternative to (or a test of) the convex hull algorithm. It uses more
# conventional method of computing, e.g., the liquidus of a system. But
# these tools do not form a complete set that can replace the convex
# hull algorithm.
#---------------------------------------------------------------------------

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
                components. Must sum to 0. The liquidus will then be sought
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
    ncomp   = len(model.compselect)
    Rgas    = 8.314
    # Make sure that the model is reset to the correct temperature and pressure
    model.reset(T,P)
    # Get the chemical potential and composition of the crystal solid
    mn      = model.mdb.set_index('Abbrev').loc[mineral]
    nu      = ph.dissect_oxide(mn['Formula'],components=model.compselect)['nucomp']
    xs      = nu/nu.sum()
    if hasattr(model,'get_mu0_at_T_P'):
        mucryst = model.get_mu0_at_T_P(model.mdb,mineral,T,P)
    else:
        mucryst = model.get_mu0_at_T(model.mdb,mineral,T)
    # Find the distance to the edge of the domain
    dist    = 1e99
    iicmp   = -1
    for icmp in range(ncomp):
        if dx[icmp]!=0:
            s = -xs[icmp]/dx[icmp]
            if s==0 and dx[icmp]<0:
                return None
            if s>0 and s<dist:
                dist  = s
                iicmp = icmp
    assert iicmp>=0 and dist<1e90, 'Weird error'
    # Get the G values of the liquid components
    Gliqcmp = np.zeros(ncomp)
    for icmp in range(ncomp):
        xcmp = np.zeros((1,ncomp))
        xcmp[0,icmp]  = 1.
        Gliqcmp[icmp] = model.compute_G_of_liquid_mixture(model.ldb,T,P,xcmp,model.compselect)[0]
    # Set up the function to find the root of
    def fun(s):
        x     = np.zeros([1,ncomp])
        x[0,:]= xs + dx*s
        gamma = model.margules.get_activity_coefficients_of_components(x,T,P).T
        muliq = (nu*(Gliqcmp+Rgas*T*np.log(x*gamma+1e-99))).sum(axis=-1)[0]
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

def find_liquidus_T_of_a_crystal_given_P_and_x(model,mineral,P,x,Tmin=100.,Tmax=4000.,Tstart=4000.,method='brentq'):
    """
    Given a crystal with fixed composition from the mineral database of the model
    (name of the crystal is mineral), and given a pressure P, and composition
    x elsewhere in the phase diagram, this function will attempt to find the
    temperature of the liquidus belonging to this crystal at that composition x.
    It does so by finding the temperature T where the chemical potential of the
    liquid with respect to the stoichiometry of the crystal equals the chemical
    potential of the crystal. This is done with the root-finding algorithm brentq
    of scipy.optimize.

    Arguments:

      model     The mineral system model class (such as Berman83 class from the
                model_Berman1983.py).

      mineral   The acronym of the mineral for which the liquidus is to be found.
                They can be found in the minerals.fwf file in the column 'Abbrev'.

      P         Pressure [bar]

      x         Location in mole fraction space. Array of length number of
                components. Must sum to 1. 

      method    If 'brent', use brentq(), else, use root()

    Returns:

      Tliq      The temperature of the liquidus at x belonging to this crystal.
    """
    from scipy.optimize import brentq,root
    x       = x/x.sum()
    ncomp   = len(model.compselect)
    assert ncomp==len(x), 'The vector x is not same length as number of components.'
    Rgas    = 8.314
    # Get the chemical potential and composition of the crystal solid
    mn      = model.mdb.set_index('Abbrev').loc[mineral]
    nu      = ph.dissect_oxide(mn['Formula'],components=model.compselect)['nucomp']
    # Set up the function to find the root of
    def fun(T):
        model.reset(T,P)
        # Get mu of crystal
        if hasattr(model,'get_mu0_at_T_P'):
            mucryst = model.get_mu0_at_T_P(model.mdb,mineral,T,P)
        else:
            mucryst = model.get_mu0_at_T(model.mdb,mineral,T)
        # Get the mu0 values of the liquid components
        Gliqcmp = np.zeros(ncomp)
        for icmp in range(ncomp):
            xcmp = np.zeros((1,ncomp))
            xcmp[0,icmp]  = 1.
            Gliqcmp[icmp] = model.compute_G_of_liquid_mixture(model.ldb,T,P,xcmp,model.compselect)[0]
        gamma = model.margules.get_activity_coefficients_of_components(x,T,P).T  # Should be possible to call model.get_activity_coefficients_of_components(x,T,P)
        muliq = (nu*(Gliqcmp+Rgas*T*np.log(x*gamma+1e-99))).sum(axis=-1)
        if not np.isscalar(muliq):
            muliq = muliq[0]
        return muliq-mucryst
    # Check that brentq can solve it
    if method=='brentq':
        fmin = fun(Tmin)
        fmax = fun(Tmax)
        if fmin*fmax>0:
            return None   # The Tmin,Tmax range is not enough
        Tliq = brentq(fun,Tmin,Tmax)
    else:
        Tliq = root(fun,Tstart,method=method)['x']
    if not np.isscalar(Tliq): Tliq=Tliq[0]
    return Tliq

def compute_melting_temperatures_of_minerals(model,mdb,comp,P=1.):
    """
    For each mineral in the mdb list, compute the melting temperature.
    Since some minerals melt congruently, and other do not, we have to
    compute not just when each mineral melts, but also if the liquidus
    at the location of this mineral belonging to another mineral is at
    a higher temperature. If so, then the melting is eutectic, not
    congruent. So, for a complete check of the liquidus at the
    chemical composition of a mineral, we have to check the full NxN
    matrix of possible minerals. You can save the result with the
    function save_melting_temperatures(), and re-read it with the
    function read_melting_temperatures().
    """
    method = 'hybr'
    ncomp  = len(comp)
    nmin   = len(mdb)
    Tliqu  = np.zeros((nmin,nmin))
    Tmin   = 100.
    Tmax   = 4000.

    # First the diagonal
    minnames = ['' for _ in range(nmin)]
    minabbrs = ['' for _ in range(nmin)]
    minforms = ['' for _ in range(nmin)]
    for im,mm in enumerate(mdb.index):
        print(f'Computing Tliqu for {mm}')
        name         = mdb.loc[mm]['Name']
        abbr         = mm
        formula      = mdb.loc[mm]['Formula']
        minnames[im] = name
        minabbrs[im] = abbr
        minforms[im] = formula
        mn           = mdb.loc[mm]
        Tliqu[im,im] = find_liquidus_T_of_a_crystal_given_P_and_x(model,mm,P,mn['x'],Tmin=Tmin,Tmax=Tmax,Tstart=4000.,method=method)
    
    # Then the rest
    for im,mm in enumerate(mdb.index):
        for io,mo in enumerate(mdb.index):
            if Tliqu[im,im]<Tliqu[io,io]:
                print(f'Computing Tliq for {mm} belonging to {mo}')
                mmn          = mdb.loc[mm]
                Tl           = find_liquidus_T_of_a_crystal_given_P_and_x(model,mo,P,mmn['x'],Tmin=Tmin,Tmax=Tmax,Tstart=4000.,method=method)
                if Tl<Tmax and Tl>Tmin:
                    Tliqu[im,io] = Tl

    # Now classify
    minerals = {}
    for im,mm in enumerate(mdb.index):
        Tmelt        = Tliqu[im,im]
        io           = np.argmax(Tliqu[im,:])
        if io==im:
            melttype = 'congruent'
            Tliq     = Tmelt
            miner    = {'Name':mdb.loc[mm]['Name'],'Formula':mdb.loc[mm]['Formula'],'melttype':melttype,'Tmelt':Tmelt,'Tliq':Tliq}
        else:
            melttype = 'eutectic'
            Tliq     = Tliqu[im,io]
            isort    = np.argsort(-Tliqu[im])
            assert isort[0]==io, 'Weird error'
            liquidi  = []
            for i in isort:
                if Tliqu[im,i]>Tmelt:
                    liquidi.append({'Name':minnames[i],'Abbrev':minabbrs[i],'Formula':minforms[i],'Tliq':Tliqu[im,i]})
            miner    = {'Name':mdb.loc[mm]['Name'],'Formula':mdb.loc[mm]['Formula'],'melttype':melttype,'Tmelt':Tmelt,'Tliq':Tliq,'liquidi':liquidi}
        minerals[mm] = miner
    
    # Eliminate cases where the 'eutectic' melting is just a different polymorph
    # of the same chemical formula
    for im,mm in enumerate(mdb.index):
        if minerals[mm]['melttype']=='eutectic':
            polymorph = False
            formula   = mdb.loc[mm]['Formula']
            for l in minerals[mm]['liquidi']:
                if l['Formula']==formula:
                    polymorph = True
            if polymorph:
                minerals[mm]['melttype']='polymorph'

    return minerals

def save_melting_temperatures(comp,minerals):
    """
    Saving the result of compute_melting_temperatures_of_minerals()
    to a file.
    """
    filename = 'Tmelt_System_'+'_'.join(comp)+'.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(minerals, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_melting_temperatures(comp):
    """
    Reading the melting temperature pickle file for the system
    of components given by comp.
    """
    filename = 'Tmelt_System_'+'_'.join(comp)+'.pickle'
    with open(filename, 'rb') as handle:
        minerals = pickle.load(handle)
    return minerals
