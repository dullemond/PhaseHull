#---------------------------------------------------------------------------
#                  Part of PhaseHull, a simple python package
#                    to compute equilibrium phase diagrams
#
#                           (C) C. P. Dullemond
#                      Heidelberg University, Germany
#                                June 2025
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from phasehull import dissect_oxide
import scipy

class SubSystem(object):
    def __init__(self,endm_prim,endm_comp,G_endm_prim=None,G_endm_comp=None,endm_comp_weights=None):
        """
        This tool set is meant for making it easier to compare and convert between
        two overlapping endmember systems: a primitive one (consisting of the
        most basic units only, e.g. the MgO, Al2O3, SiO2 system), and a composite
        one, (e.g. the Mg2SiO4, SiO2 subsystem or the Mg2SiO4, Al2O3, SiO2 system).

        The composite subsystem can either be lower-dimensional (like in the
        example of the binary Mg2SiO4, SiO2 subsystem embedded in the ternary
        MgO, Al2O3, SiO2 primitive system) or of the same dimension (like in the
        example of the ternary Mg2SiO4, Al2O3, SiO2 subsystem, also embedded in
        the ternary MgO, Al2O3, SiO2 primitive system). But the composite
        endmembers must all lie within the system spanned by the primitive
        endmembers, i.e. the no negative molar fractions in the primitive system.

        Arguments:

          endm_prim     List of names of the primitive endmembers, e.g.
                        ['MgO','SiO2']

          endm_comp     List of names of the composite endmembers, e.g.
                        ['Mg2SiO4','SiO2']

        Optional arguments/keywords:
        
          G_endm_prim   The Gibbs energy at the location of the primitive endmembers,
                        as computed by the model using these primitive endmembers.

          G_endm_comp   The Gibbs energy at the location of the composite endmembers,
                        as computed by the model using thee composite endmembers.

          endm_comp_weights   If set to an array of floats with the same length
                              as endm_comp, this allows you to use fractional
                              composite endmembers, e.g. MnSi0.5O2 which would
                              mean you give Mn2SiO4 as composite endmember,
                              and set the weight to 0.5 for that one. It is not
                              possible to use MnSi0.5O2 directly as an endmember.

          If both G_endm_prim and G_endm_comp are given, then SubSystem can
          compute the difference in the plane spanned by the composite
          system endmembers and the plane spanned by the primitive 
          system endmembers (the latter: values at the location of the
          composite endmembers). This allows easier comparison between
          the mixing Gibbs surfaces of the two systems. For more details: see
          comments in the code.

        Methods/functions:

          convert_from_xcomp_to_xprim(xcomp)
          convert_from_xprim_to_xcomp(xprim)
        
        
        """
        self.endm_prim         = endm_prim
        self.endm_comp         = endm_comp
        if G_endm_prim is None:
            self.G_endm_prim   = None
        else:
            assert len(G_endm_prim)==len(self.endm_prim), 'Error: Nr of G values for primitive system unequal to nr of endmembers'
            self.G_endm_prim   = np.array(G_endm_prim)
        if G_endm_comp is None:
            self.G_endm_comp   = None
        else:
            assert len(G_endm_comp)==len(self.endm_comp), 'Error: Nr of G values for composite system unequal to nr of endmembers'
            self.G_endm_comp   = np.array(G_endm_comp)
        if endm_comp_weights is None:
            endm_comp_weights  = np.ones(len(self.endm_comp))
        assert len(endm_comp_weights)==len(self.endm_comp), 'Error: Nr of weights for composite system unequal to nr of endmembers'
        self.endm_comp_weights = endm_comp_weights
        self.nendm_prim        = len(self.endm_prim)
        self.nendm_comp        = len(self.endm_comp)
        assert self.nendm_comp<=self.nendm_prim, 'Error: Subsystem larger than primitive system'

        # Conversion matrix from composite N_comp vector to primitive N_prim vector
        nu_prim_comp = np.zeros((self.nendm_prim,self.nendm_comp))
        for ic,ec in enumerate(self.endm_comp):
            dis                = dissect_oxide(ec,endmembers=endm_prim)
            nu_prim_comp[:,ic] = dis['nuendm']*self.endm_comp_weights[ic]
        self.nu_prim_comp      = nu_prim_comp

        # The number of moles of primitive endmembers needed to create 1 mole of composite endmember
        self.comp_nprim        = nu_prim_comp.sum(axis=0)

        # Conversion matrix from primitive N_prim vector to composite N_comp vector
        # (only if nendm_comp==nendm_prim)
        if self.nendm_comp==self.nendm_prim:
            self.nu_comp_prim  = scipy.linalg.inv(self.nu_prim_comp)

    def compute_ratio_of_nmoles_in_comp_to_nmoles_in_prim(self,xcomp):
        """
        If you mix 1 mole of composite endmembers with mole fractions xcomp,
        you get M moles of liquid (where M will depend on how you define the
        formula unit of the liquid particles). If you now compute the
        mole fractions in the primitive system, xprim, corresponding to xcomp,
        then the number of moles of endmembers is typically no longer 1 mole.
        Yet, physically, the number of moles M of liquid particles should
        be the same. Or to put it differently, if you now take 1 mole of the
        primitive endmembers at mole fraction xprim, then you get less liquid
        than when you mixed the composite endmembers, because you typically
        need more moles of primitive endmembers than composite endmembers
        for the same liquid formula.

        To compare results from the composite and primitive systems to each
        other, one must correct for this. One way to do this is to use mass
        fractions instead of mole fractions. But here, instead, we compute
        for all given xcomp values, the ratio of the nr of moles of liquid
        (in whatever formula unit, does not matter) from 1 mole of endmembers
        in the composite system to the nr of moles of liquid from 1 mole of
        endmembers in the primitive system.
        """
        xcomp    = np.array(xcomp)
        xcomp    = xcomp/xcomp.sum(axis=-1)[...,None]
        ntotprim = self.convert_from_ncomp_to_nprim(xcomp).sum(axis=-1)
        return ntotprim
        
    def convert_G_value_from_composite_to_primitive_system(self,xcomp,Gcomp):
        """
        If you mix liquids in a composite endmembers sytem and compare
        their Gibbs values to the equivalent mixtures in the primitive
        endmember system, then you get different Gibbs energies, because
        you get different quantities of liquid. For a detailed description
        of this problem, see the doc string of the function

           compute_ratio_of_nmoles_in_comp_to_nmoles_in_prim()

        So if you want to compare the Gibbs energies obtained in these
        two systems, you must correct for this difference in amount of
        liquid. This is done here.

        Arguments:

          xcomp    A 2D array of x values x[nx,nendm] (where nendm is the
                   number of endmembers) for which the conversion should
                   be done.
          Gcomp    Array of Gibbs energies for each of the xcomp values.

        Returns:

          Gprim    The corrected Gibbs energies that now can be safely
                   compared to Gibbs energies in the primitive end member
                   system.
        """
        ntotprim = self.compute_ratio_of_nmoles_in_comp_to_nmoles_in_prim(xcomp)
        Gprim    = Gcomp/ntotprim
        return Gprim

    def compute_G_nomix_in_prim_system_at_locations_of_composite_endmembers(self):
        """
        For convenience, compute the G_endm_comp_inprim_nomix, which is the G computed
        at the location of the composite endmembers, without mixing terms (i.e., just the
        linear mean), but in the system composed of the primitive endmembers. Why is this
        useful? That is because if we want to compare the G curves of the composite system
        to those of the primitive system, we typically want to subtract the linear plane
        of the linear (unmixed) combination of the primitive endmembers, because otherwise
        the figures/plots are dominated by the (uninteresting) sum_i x_i G_i plane, and
        the (interesting) parts sum_i x_i log(x_i) + sum W_ij x_i x_j are too small to
        really see on the plot. If we subtract the sum_i x_i G_i plane (the one from the
        primitive endmembers) then the sum_i x_i log(x_i) + sum W_ij x_i x_j of the
        primitive endmembers will be 0 at all endmembers, but the one of the
        composite endmembers will not be 0 at (their) endmembers. The differences are
        the ones calculated here, so that the sum_i x_i log(x_i) + sum W_ij x_i x_j
        curves of the composite endmembers can be easily overplotted over the ones
        of the primitive endmembers. Note that, in principle, for the locations where the
        composite and primitive endmembers are the same, the G_endm_comp should be
        equal to G_endm_prim, but since we are usually comparing different models
        with each other, this may not be exactly the case.

        Arguments:

           None

        Returns:

           G_nomix    The values of the Gibbs energy at the location of the composite
                      endmembers, computed using linear mean of the primitive endmembers
                      (i.e. no contributions for chemical mixing). And because often
                      more primitive endmembers are needed to build one composite
                      formula, the result is divided by self.comp_nprim, to be able
                      to compare to the G functions living on the primitive space.
        """
        G_nomix  = (np.array(self.G_endm_prim)[:,None]*self.nu_prim_comp/self.comp_nprim[None,:]).sum(axis=0)
        G_nomix /= self.comp_nprim
        return G_nomix

    def convert_from_ncomp_to_nprim(self,ncomp):
        """
        If you have mole in the composite endmember system, ncomp, then
        this function returns the moles in the primitive endmember system.
    
        Arguments:
    
          ncomp    A 1D array of n (moles) values. Or a 2D array of
                   n values n[nx,nendm] (where nendm is the number of endmembers)
                   for which the conversion should be done.
    
        Returns:

          nprim    Array of values of the moles in the primitive system. Note that
                   the sum of nprim.sum() does in general not equal ncomp.sum()
        """
        ncomp = np.array(ncomp)
        nprim = (self.nu_prim_comp[...,:,:]*ncomp[...,None,:]).sum(axis=-1)
        return nprim
    
    def convert_from_xcomp_to_xprim(self,xcomp):
        """
        If you have mole fractions in the composite endmember system, xcomp, then
        this function returns the mole fractions in the primitive endmember system.
    
        Arguments:
    
          xcomp    A 1D array of x values that should add to 1. Or a 2D array of
                   x values x[nx,nendm] (where nendm is the number of endmembers)
                   for which the conversion should be done.
    
        Returns:

          xprim   Array of values of the mole fractions in the primitive system.
        """
        xcomp = np.array(xcomp)
        xcomp = xcomp/xcomp.sum(axis=-1)[...,None]
        nprim = (self.nu_prim_comp[...,:,:]*xcomp[...,None,:]).sum(axis=-1)
        xprim = nprim/nprim.sum(axis=-1)[...,None]
        return xprim
    
    def convert_from_nprim_to_ncomp(self,nprim):
        """
        If you have mole in the primitive endmember system, nprim, then
        this function returns the moles in the composite endmember system.
        Note that this is only possible if the number of endmembers of the composite
        system equals that of the primitive system, and they are not degenerate.
    
        Arguments:
    
          nprim    A 1D array of n (moles) values. Or a 2D array of
                   n values n[nx,nendm] (where nendm is the number of endmembers)
                   for which the conversion should be done.
    
          out_of_bounds_nan   If True, then whereever the resulting ncomp has
                              negative values (i.e. the nprim was outside of
                              the scope of the composite subsystem) these
                              values will be set to np.nan.

        Returns:

          ncomp    Array of values of the moles in the composite system. Note that
                   the sum of ncomp.sum() does in general not equal nprim.sum()
        """
        nprim = np.array(nprim)
        ncomp = (self.nu_comp_prim[...,:,:]*nprim[...,None,:]).sum(axis=-1)
        if out_of_bounds_nan:
            ncomp[np.any(ncomp<0,axis=-1)] = np.nan
        return ncomp
    
    def convert_from_xprim_to_xcomp(self,xprim,out_of_bounds_nan=True):
        """
        If you have mole fractions in the primitive endmember system, xprim, then
        this function returns the mole fractions in the composite endmember system.
        Note that this is only possible if the number of endmembers of the composite
        system equals that of the primitive system, and they are not degenerate.
    
        Arguments:
    
          xprim    A 1D array of x values that should add to 1. Or a 2D array of
                   x values x[nx,nendm] (where nendm is the number of endmembers)
                   for which the conversion should be done.
    
          out_of_bounds_nan   If True, then whereever the resulting xcomp has
                              negative values (i.e. the xprim was outside of
                              the scope of the composite subsystem) these
                              values will be set to np.nan.

        Returns:

          xcomp   Array of values of the mole fractions in the composite system.
        """
        xprim = np.array(xprim)
        xprim = xprim/xprim.sum(axis=-1)[...,None]
        ncomp = (self.nu_comp_prim[...,:,:]*xprim[...,None,:]).sum(axis=-1)
        xcomp = ncomp/(ncomp.sum(axis=-1)[...,None]+1e-99)
        if out_of_bounds_nan:
            xcomp[np.any(xcomp<0,axis=-1)] = np.nan
        return xcomp

# ==================================================================================
# Examples
# ==================================================================================

if __name__ == "__main__":

    # Example: Comparing the model of Berman in his PhD thesis to the MELTS model of Ghiorso & Sack 1995

    endm_prim   = ['SiO2','MgO']
    endm_comp   = ['SiO2','Mg2SiO4']
    G_endm_prim = np.array([-1124829., -718317.])  # The G values at the primitive endmembers from the Berman PhD thesis model
    G_endm_comp = np.array([-1126980.,-2678878.])  # The G values at the composite endmembers from the Ghiorso & Sack 1995 model
    
    subsys      = SubSystem(endm_prim,endm_comp,G_endm_prim,G_endm_comp)

    nendm_prim  = len(endm_prim)
    nendm_comp  = len(endm_comp)
    nx          = 100
    x1d         = np.linspace(0,1,nx+1)
    xcomp       = np.zeros((nx+1,nendm_comp))
    xcomp[:,0]  = x1d
    xcomp[:,1]  = 1-x1d
    xprim       = subsys.convert_from_xcomp_to_xprim(xcomp)

    # Figure showing the conversion from one into another
    plt.figure()
    for i in range(nendm_prim):
        plt.plot(xprim[:,i],xcomp[:,i],label=f'{endm_prim[i]},{endm_comp[i]}')
    plt.xlabel(r'$x_{\mathrm{prim}}$')
    plt.ylabel(r'$x_{\mathrm{comp}}$')
    plt.legend()
    plt.show()

    # Now overplot the two different Gibbs curves of Berman and G&S
    Rgas        = 8.314  # J/mol·K
    T           = 2000.  # T in Kelvin
    P           = 1.     # P in bar
    nx          = 100
    x1d         = np.linspace(0,1,nx+1)
    xprim       = np.zeros((nx+1,nendm_prim))
    xprim[:,0]  = x1d
    xprim[:,1]  = 1-x1d
    xcomp       = subsys.convert_from_xprim_to_xcomp(xprim)  # Will contain NaNs on the left side; no problem

    # Berman Margules parameters in the primitive system
    # m=MgO, s=SiO2
    prim_WH_smmm =   -610906.87
    prim_WH_ssmm =   -270581.70
    prim_WH_sssm =     94145.91
    prim_WS_smmm =      -195.01
    prim_WS_ssmm =       -59.98
    prim_WS_sssm =        52.77
    prim_WG_smmm = prim_WH_smmm - T*prim_WS_smmm
    prim_WG_ssmm = prim_WH_ssmm - T*prim_WS_ssmm
    prim_WG_sssm = prim_WH_sssm - T*prim_WS_sssm
    
    # Ghiorso & Sack 1995 Margules parameters in the composite system
    comp_WH_sm   =      3421.00
    comp_WS_sm   =         0.00
    comp_WG_sm   = comp_WH_sm   - T*comp_WS_sm

    # Berman Gibbs energy of the liquid
    prim_G0      = (xprim*G_endm_prim[None,:]).sum(axis=-1)          # The mean of the endmembers
    prim_Gi      = Rgas*T*(xprim*np.log(xprim+1e-99)).sum(axis=-1)   # The ideal mixing term
    prim_Gw      = prim_WG_smmm*xprim[:,0]*xprim[:,1]**3             # The first of the Margules terms
    prim_Gw     += prim_WG_ssmm*xprim[:,0]**2*xprim[:,1]**2          # The second of the Margules terms
    prim_Gw     += prim_WG_sssm*xprim[:,0]**3*xprim[:,1]             # The third of the Margules terms
    prim_G       = prim_G0 + prim_Gi + prim_Gw
    prim_G0i     = prim_G0 + prim_Gi
    
    # Ghiorso & Sack Gibbs energy of the liquid
    comp_G0      = (xcomp*G_endm_comp[None,:]).sum(axis=-1)          # The mean of the endmembers
    comp_Gi      = Rgas*T*(xcomp*np.log(xcomp+1e-99)).sum(axis=-1)   # The ideal mixing term
    comp_Gw      = comp_WG_sm*xcomp[:,0]*xcomp[:,1]                  # The Margules term (interaction term)
    comp_G       = comp_G0 + comp_Gi + comp_Gw

    # Convert the Ghiorso & Sack Gibbs energy of the liquid to the equivalent moles in the primitive system
    comp_G_prim  = subsys.convert_G_value_from_composite_to_primitive_system(xcomp,comp_G)
    comp_G0i_prim= subsys.convert_G_value_from_composite_to_primitive_system(xcomp,comp_G0+comp_Gi)
    
    # The baseline (plane) to subtract for easier viewing
    #G_nomix_prcm = subsys.compute_G_nomix_in_prim_system_at_locations_of_composite_endmembers()
    G_plane_prim = (xprim*G_endm_prim[None,:]).sum(axis=-1)
    #G_plane_comp = (xcomp*G_nomix_prcm[None,:]).sum(axis=-1)

    # Figure comparing the Gibbs energies of the two models, viewed
    # from the perspective of the primitive x
    plt.figure()
    plt.plot(x1d,(prim_G-G_plane_prim)/1e3,label='Berman (prim)',color='C0')
    plt.plot(x1d,(comp_G_prim-G_plane_prim)/1e3,label='GhiorsoSack (comp)',color='C1')
    plt.plot(x1d,(prim_G0i-G_plane_prim)/1e3,':',label='Berman ideal',color='C0')
    plt.plot(x1d,(comp_G0i_prim-G_plane_prim)/1e3,':',label='GhiorsoSack ideal',color='C1')
    plt.xlabel(r'$x_{\mathrm{prim}}$')
    plt.ylabel(r'$G-G_{\mathrm{base}}$')
    plt.legend()
    plt.title('System MgO/Mg2SiO4 and SiO2')
    plt.show()
