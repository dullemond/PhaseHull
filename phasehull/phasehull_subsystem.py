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
try:
    from phasehull import dissect_oxide
except:
    from phasehull_support import dissect_oxide
import scipy

class SubSystem(object):
    def __init__(self,comp_prim,comp_comp,G_comp_prim=None,G_comp_comp=None,comp_comp_weights=None):
        """
        This tool set is meant for making it easier to compare and convert between
        two overlapping component systems: a primitive one (consisting of the
        most basic units only, e.g. the MgO, Al2O3, SiO2 system), and a composite
        one, (e.g. the Mg2SiO4, SiO2 subsystem or the Mg2SiO4, Al2O3, SiO2 system).

        NOTE: In the literature such composite components are often referred to
              as "complex components", while our primitive components are often
              referred to as "simple oxides". See e.g., Ghiorso & Carmichael 1980
              Contrib. Mineral. Petrol. 71, 323-342; Hastie & Bonnell 1986, Journal
              of Non-Crystalline Solids 84, 151-158.

        The composite subsystem can either be lower-dimensional (like in the
        example of the binary Mg2SiO4, SiO2 subsystem embedded in the ternary
        MgO, Al2O3, SiO2 primitive system) or of the same dimension (like in the
        example of the ternary Mg2SiO4, Al2O3, SiO2 subsystem, also embedded in
        the ternary MgO, Al2O3, SiO2 primitive system). But the composite
        components must all lie within the system spanned by the primitive
        components, i.e. the no negative molar fractions in the primitive system.

        Arguments:

          comp_prim     List of names of the primitive components, e.g.
                        ['MgO','SiO2']

          comp_comp     List of names of the composite components, e.g.
                        ['Mg2SiO4','SiO2']

        Optional arguments/keywords:
        
          G_comp_prim   The Gibbs energy at the location of the primitive components,
                        as computed by the model using these primitive components.

          G_comp_comp   The Gibbs energy at the location of the composite components,
                        as computed by the model using thee composite components.

          comp_comp_weights   If set to an array of floats with the same length
                              as comp_comp, this allows you to use fractional
                              composite components, e.g. MnSi0.5O2 which would
                              mean you give Mn2SiO4 as composite component,
                              and set the weight to 0.5 for that one. It is not
                              possible to use MnSi0.5O2 directly as a component
                              formula (because the formula interpreter cannot
                              handle non-integers), but by choosing Mn2SiO4 as
                              the formula (the string / name) of that component
                              and setting the corresponding value of the
                              comp_comp_weights to 0.5, you have effectively
                              chosen MnSi0.5O2 as a composite component.

          If both G_comp_prim and G_comp_comp are given, then SubSystem can
          compute the difference in the Gibss surface spanned by the composite
          system components and the Gibbs surface spanned by the primitive 
          system components (the latter: values at the location of the
          composite components). This allows easier comparison between
          the mixing Gibbs surfaces of the two systems. For more details: see
          comments in the code.

        Methods/functions:

          convert_from_xcomp_to_xprim(xcomp)
          convert_from_xprim_to_xcomp(xprim)
        
        
        """
        self.comp_prim         = comp_prim
        self.comp_comp         = comp_comp
        if G_comp_prim is None:
            self.G_comp_prim   = None
        else:
            assert len(G_comp_prim)==len(self.comp_prim), 'Error: Nr of G values for primitive system unequal to nr of components'
            self.G_comp_prim   = np.array(G_comp_prim)
        if G_comp_comp is None:
            self.G_comp_comp   = None
        else:
            assert len(G_comp_comp)==len(self.comp_comp), 'Error: Nr of G values for composite system unequal to nr of components'
            self.G_comp_comp   = np.array(G_comp_comp)
        if comp_comp_weights is None:
            comp_comp_weights  = np.ones(len(self.comp_comp))
        assert len(comp_comp_weights)==len(self.comp_comp), 'Error: Nr of weights for composite system unequal to nr of components'
        self.comp_comp_weights = comp_comp_weights
        self.ncomp_prim        = len(self.comp_prim)
        self.ncomp_comp        = len(self.comp_comp)
        assert self.ncomp_comp<=self.ncomp_prim, 'Error: Subsystem larger than primitive system'

        # Conversion matrix from composite N_comp vector to primitive N_prim vector
        nu_prim_comp = np.zeros((self.ncomp_prim,self.ncomp_comp))
        for ic,ec in enumerate(self.comp_comp):
            dis                = dissect_oxide(ec,components=comp_prim)
            nu_prim_comp[:,ic] = dis['nucomp']*self.comp_comp_weights[ic]
        self.nu_prim_comp      = nu_prim_comp

        # The number of moles of primitive components needed to create 1 mole of composite component
        self.comp_nprim        = nu_prim_comp.sum(axis=0)

        # Conversion matrix from primitive N_prim vector to composite N_comp vector
        # (only if ncomp_comp==ncomp_prim)
        if self.ncomp_comp==self.ncomp_prim:
            self.nu_comp_prim  = scipy.linalg.inv(self.nu_prim_comp)

    # Claude addition (start)
    # Claude corrected a typo in the docstring below: "because the primitive system
    # is allowed to be a subspace" -> "because the composite system is allowed ...".
    # Claude addition (end)
    def compute_mole_ratio_from_comp_to_prim(self,xcomp):
        """
        If you mix 1 mole of composite components with mole fractions xcomp,
        and you now translate this into the primitive system components, how 
        many moles of primitive system components does this become? This is 
        what we call the "mole ratio from the composite component system to 
        the primitive component system". For example: the system of composite
        components [SiO2,Mg2SiO4] translated into the system of primitive
        components [SiO2,MgO]. If you have xcomp=[0,1], meaning you have
        only Mg2SiO4, then this translates into xprim=[(1/3),(2/3)] (see
        method convert_from_xcomp_to_xprim() below). One mole worth of
        Mg2SiO4 equals 3 moles worth of SiO2 and MgO (1 mole of SiO2 and
        2 moles of MgO). So the mole ration from [SiO2,Mg2SiO4] to [SiO2,MgO]
        at xcomp=(0,1) is 3, while at xcomp=(1,0) it is 1. It is a linear
        function of the xcomp coordinates (but can be a non-linear function
        of the xprim coordinates). So for xcomp=(0.5,0.5) its value is 2
        in the above example.

        Note: This function is only offered in this direction (composite
        to primitive). The inverse function would simply be that you
        convert xprim into pcomp, call compute_mole_ratio_from_comp_to_prim()
        and then take the inverse (1/...) of the result. The reason why
        this function is not added here is because the composite system
        is allowed to be a subspace (lower dimensional) of the primitive
        system (for solid solutions), in which case the inverse, starting
        from an arbitrary xprim, may not be defined, as the xprim location
        may be out of the plane of the subspace. Yet, when embedding such
        a lower-dimensional solid solution space into a primitive (or non
        primitive, but in the context of this module still 'primitive')
        larger dimensional system, you need to multiply the Gibbs free energy 
        of the composite system by the inverse of the mole ratio from the
        composite (solid solution) system to the primitive (full) system.

        Final note: The result of compute_mole_ratio_from_comp_to_prim()
        at the pure xcomp locations (in the above example [1.,0.] and
        [0.,1.]) are the same as the self.comp_nprim values (in the
        above example these are 1. and 3.).
        """
        xcomp    = np.array(xcomp)
        xcomp    = xcomp/xcomp.sum(axis=-1)[...,None]
        ntotprim = self.convert_from_ncomp_to_nprim(xcomp).sum(axis=-1)
        return ntotprim
        
    def convert_G_value_from_composite_to_primitive_system(self,xcomp,Gcomp):
        """
        If you mix liquids in a composite components sytem and compare
        their Gibbs values to the equivalent mixtures in the primitive
        component system, then you get different Gibbs energies, because
        you get different quantities of liquid. For a detailed description
        of this problem, see the doc string of the function

           compute_mole_ratio_from_comp_to_prim()

        So if you want to compare the Gibbs energies obtained in these
        two systems, you must correct for this difference in amount of
        liquid. This is done here.

        Arguments:

          xcomp    A 2D array of x values x[nx,ncomp] (where ncomp is the
                   number of components) for which the conversion should
                   be done.
          Gcomp    Array of Gibbs energies for each of the xcomp values.

        Returns:

          Gprim    The corrected Gibbs energies that now can be safely
                   compared to Gibbs energies in the primitive end member
                   system.
        """
        Gcomp    = np.array(Gcomp)
        ntotprim = self.compute_mole_ratio_from_comp_to_prim(xcomp)
        Gprim    = Gcomp/ntotprim
        return Gprim

    def convert_G_value_from_primitive_to_composite_system(self,xprim,Gprim,out_of_bounds_nan=True):
        """
        The inverse of convert_G_value_from_composite_to_primitive_system().
        Note that this is only possible if the composite system has the same
        number of independent components as the primitive system.
        Arguments:

          xprim    A 2D array of x values x[nx,nprim] (where nprim is the
                   number of components) for which the conversion should
                   be done.
          Gprim    Array of Gibbs energies for each of the xprim values.

          out_of_bounds_nan   If True, then whereever the resulting ncomp has
                              negative values (i.e. the nprim was outside of
                              the scope of the composite subsystem) these
                              values will be set to np.nan.

        Returns:

          Gcomp    The corrected Gibbs energies that now can be safely
                   compared to Gibbs energies in the primitive end member
                   system.
        """
        assert hasattr(self,'nu_comp_prim'), 'Error: Can only convert G from primitive to composite system if both have the same dimension.'
        Gprim    = np.array(Gprim)
        xcomp    = self.convert_from_xprim_to_xcomp(xprim,out_of_bounds_nan=out_of_bounds_nan)
        ntotcomp = 1/(self.compute_mole_ratio_from_comp_to_prim(xcomp)+1e-99)
        Gcomp    = Gprim/ntotcomp
        return Gcomp

    def compute_G_nomix_in_prim_system_at_locations_of_composite_components(self):
        """
        For convenience, compute the G_comp_comp_inprim_nomix, which is the G computed
        at the location of the composite components, without mixing terms (i.e., just the
        linear mean), but in the system composed of the primitive components. Why is this
        useful? That is because if we want to compare the G curves of the composite system
        to those of the primitive system, we typically want to subtract the linear plane
        of the linear (unmixed) combination of the primitive components, because otherwise
        the figures/plots are dominated by the (uninteresting) sum_i x_i mu^0_i plane, and
        the (interesting) parts sum_i RT x_i log(x_i) + sum W_ij x_i x_j are too small to
        really see on the plot. If we subtract the sum_i x_i mu^0_i plane (the one from the
        primitive components) then the sum_i RT x_i log(x_i) + sum W_ij x_i x_j of the
        primitive components will be 0 at all components, but the one of the
        composite components will not be 0 at (their) components. The differences are
        the ones calculated here, so that the sum_i RT x_i log(x_i) + sum W_ij x_i x_j
        curves of the composite components can be easily overplotted over the ones
        of the primitive components. Note that, in principle, for the locations where the
        composite and primitive components are the same, the G_comp_comp should be
        equal to G_comp_prim, but since we are usually comparing different models
        with each other, this may not be exactly the case.

        Arguments:

           None

        Returns:

           G_nomix    The values of the Gibbs energy at the location of the composite
                      components, computed using linear mean of the mu^0 values of
                      the primitive components (i.e. no contributions for mixing).
                      And because often more primitive components are needed to build
                      one composite component (see compute_mole_ratio_from_comp_to_prim()
                      above), the result is divided by self.comp_nprim, to be able
                      to compare to the G functions living on the primitive space.
        """
        G_nomix  = (np.array(self.G_comp_prim)[:,None]*self.nu_prim_comp/self.comp_nprim[None,:]).sum(axis=0)
        # Claude deleted old code here
        # Claude addition (start)
        # The division by self.comp_nprim was already done inside the sum above
        # (which yields the linear mean per mole of primitive components). The old
        # line "G_nomix /= self.comp_nprim" divided a second time, which was wrong
        # (e.g. it gave (G_SiO2+2*G_MgO)/9 instead of (G_SiO2+2*G_MgO)/3 for Mg2SiO4).
        # Claude addition (end)
        return G_nomix

    def convert_from_ncomp_to_nprim(self,ncomp):
        """
        If you have mole in the composite component system, ncomp, then
        this function returns the moles in the primitive component system.
    
        Arguments:
    
          ncomp    A 1D array of n (moles) values. Or a 2D array of
                   n values n[nx,ncomp] (where ncomp is the number of components)
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
        If you have mole fractions in the composite component system, xcomp, then
        this function returns the mole fractions in the primitive component system.
    
        Arguments:
    
          xcomp    A 1D array of x values that should add to 1. Or a 2D array of
                   x values x[nx,ncomp] (where ncomp is the number of components)
                   for which the conversion should be done.
    
        Returns:

          xprim   Array of values of the mole fractions in the primitive system.
        """
        xcomp = np.array(xcomp)
        xcomp = xcomp/xcomp.sum(axis=-1)[...,None]
        nprim = (self.nu_prim_comp[...,:,:]*xcomp[...,None,:]).sum(axis=-1)
        xprim = nprim/nprim.sum(axis=-1)[...,None]
        return xprim
    
    # Claude deleted old code here
    # Claude addition (start)
    def convert_activity_from_comp_to_prim(self,acomp,T=None,mu0_comp=None,mu0_prim=None,
                                           Rgas=8.314,assume_linear_standard_states=False):
        """
        If you have computed the activities of the composite components, this
        function will return the corresponding activities of the primitive
        components, each relative to the real standard state of that primitive
        component (e.g. pure liquid MgO), not relative to a fictive one.

        Why the default goes through the chemical potentials:

          An activity is only defined together with its standard state:
          mu_i = mu0_i + R*T*ln(a_i). The chemical potentials of the two
          component systems are related exactly by the stoichiometry
          (mu_comp = nu^T mu_prim, see convert_chemical_potential_from_comp_to_prim()),
          but the standard states are in general NOT related this way: the
          standard state of a composite component (e.g. pure liquid Mg2SiO4)
          differs from the stoichiometric combination of the standard states
          of the primitive components (2 MgO(liq) + SiO2(liq), unmixed) by the
          Gibbs energy of formation of the composite from the primitives,
          Delta_f G = mu0_comp - sum_p nu_pc mu0_prim_p, which is usually large.

          Transforming ln(a) directly with the stoichiometric matrix (the old
          method, still available with assume_linear_standard_states=True)
          therefore silently yields the primitive activities relative to a
          fictive standard state. For instance, from a_Mg2SiO4 and a_SiO2 it
          gives a_MgO relative to mu0*_MgO = (mu0_Mg2SiO4 - mu0_SiO2)/2, and
          the true a_MgO differs from it by a factor exp((mu0*_MgO-mu0_MgO)/RT),
          which can be many orders of magnitude. The two methods agree only
          if Delta_f G = 0 for all composite components.

          The default method therefore does:
            mu_comp = mu0_comp + R*T*ln(a_comp)
            mu_prim = (nu^-1)^T mu_comp        (exact)
            a_prim  = exp((mu_prim - mu0_prim)/(R*T))

        Arguments:

          acomp   The activities of the components in the composite system
                  (array [...,ncomp_comp]), relative to the standard states
                  mu0_comp.

        Keywords:

          T       Temperature [K]. Required unless assume_linear_standard_states=True.
                  Scalar or array broadcastable to acomp[...,0].

          mu0_comp   Standard-state chemical potentials of the composite components
                     [J/mol], per (weighted, see comp_comp_weights) formula unit.
                     If None, self.G_comp_comp is used.

          mu0_prim   Standard-state chemical potentials of the primitive components
                     [J/mol]. If None, self.G_comp_prim is used.
                     NOTE: mu0_comp and mu0_prim must refer to the same energy
                     zero point (e.g. both formation from the elements, or both
                     atomization energies), and must be evaluated at the same T and P.
                     They may come from different models (e.g. mu0_comp from
                     Ghiorso & Sack 1995 and mu0_prim from Berman 1983); then
                     aprim is the activity relative to the pure primitive
                     component as described by the second model.

          Rgas    Gas constant [J/(mol K)].

          assume_linear_standard_states   If True, use the old method (direct
                  linear transformation of ln(a), i.e. assuming that
                  mu0_comp = nu^T mu0_prim). Default False.

        Returns:

          aprim   The activities of the components in the primitive system
        """
        assert hasattr(self,'nu_comp_prim'), 'Error: Can only convert activity from composite to primitive system if both have the same dimension.'
        acomp  = np.array(acomp)
        if assume_linear_standard_states:
            # The old method (kept as an option)
            lacomp = np.log(acomp)
            laprim = (self.nu_comp_prim[...,:,:]*lacomp[...,:,None]).sum(axis=-2)
            aprim  = np.exp(laprim)
            return aprim
        if mu0_comp is None: mu0_comp = self.G_comp_comp
        if mu0_prim is None: mu0_prim = self.G_comp_prim
        assert T is not None, 'Error: Temperature T is needed to convert activities via the chemical potentials (or use assume_linear_standard_states=True).'
        assert mu0_comp is not None and mu0_prim is not None, 'Error: Standard-state chemical potentials of both systems are needed (pass mu0_comp and mu0_prim, or G_comp_comp and G_comp_prim at creation of the SubSystem), or use assume_linear_standard_states=True.'
        mu0_comp = np.array(mu0_comp)
        mu0_prim = np.array(mu0_prim)
        assert mu0_comp.shape[-1]==self.ncomp_comp, 'Error: mu0_comp has wrong length.'
        assert mu0_prim.shape[-1]==self.ncomp_prim, 'Error: mu0_prim has wrong length.'
        RT     = Rgas*np.array(T,dtype=float)
        if RT.ndim>0: RT = RT[...,None]
        mucomp = mu0_comp + RT*np.log(acomp)
        muprim = self.convert_chemical_potential_from_comp_to_prim(mucomp)
        aprim  = np.exp((muprim-mu0_prim)/RT)
        return aprim
    # Claude addition (end)

    def convert_chemical_potential_from_comp_to_prim(self,mucomp):
        """
        If you have computed the chemical potentials of the composite components, 
        this function will return the corresponding chemical potentials of the primitive
        components.

        Arguments:

          mucomp   The chemical potentials of the components in the composite system

        Returns:

          muprim   The chemical potentials of the components in the primitive system
        """
        assert hasattr(self,'nu_comp_prim'), 'Error: Can only convert chemical potentials from composite to primitive system if both have the same dimension.'
        mucomp  = np.array(mucomp)
        muprim = (self.nu_comp_prim[...,:,:]*mucomp[...,:,None]).sum(axis=-2)
        return muprim

    def convert_from_nprim_to_ncomp(self,nprim,out_of_bounds_nan=True):
        """
        If you have mole in the primitive component system, nprim, then
        this function returns the moles in the composite component system.
        Note that this is only possible if the number of components of the composite
        system equals that of the primitive system, and they are not degenerate.
    
        Arguments:
    
          nprim    A 1D array of n (moles) values. Or a 2D array of
                   n values n[nx,ncomp] (where ncomp is the number of components)
                   for which the conversion should be done.
    
          out_of_bounds_nan   If True, then whereever the resulting ncomp has
                              negative values (i.e. the nprim was outside of
                              the scope of the composite subsystem) these
                              values will be set to np.nan.

        Returns:

          ncomp    Array of values of the moles in the composite system. Note that
                   the sum of ncomp.sum() does in general not equal nprim.sum()
        """
        assert hasattr(self,'nu_comp_prim'), 'Error: Can only convert n from primitive to composite system if both have the same dimension.'
        nprim = np.array(nprim)
        ncomp = (self.nu_comp_prim[...,:,:]*nprim[...,None,:]).sum(axis=-1)
        if out_of_bounds_nan:
            ncomp[np.any(ncomp<0,axis=-1)] = np.nan
        return ncomp
    
    def convert_from_xprim_to_xcomp(self,xprim,out_of_bounds_nan=True):
        """
        If you have mole fractions in the primitive component system, xprim, then
        this function returns the mole fractions in the composite component system.
        Note that this is only possible if the number of components of the composite
        system equals that of the primitive system, and they are not degenerate.
    
        Arguments:
    
          xprim    A 1D array of x values that should add to 1. Or a 2D array of
                   x values x[nx,ncomp] (where ncomp is the number of components)
                   for which the conversion should be done.
    
          out_of_bounds_nan   If True, then whereever the resulting xcomp has
                              negative values (i.e. the xprim was outside of
                              the scope of the composite subsystem) these
                              values will be set to np.nan.

        Returns:

          xcomp   Array of values of the mole fractions in the composite system.
        """
        assert hasattr(self,'nu_comp_prim'), 'Error: Can only convert x from primitive to composite system if both have the same dimension.'
        xprim = np.array(xprim)
        xprim = xprim/xprim.sum(axis=-1)[...,None]
        ncomp = (self.nu_comp_prim[...,:,:]*xprim[...,None,:]).sum(axis=-1)
        xcomp = ncomp/(ncomp.sum(axis=-1)[...,None]+1e-99)
        if out_of_bounds_nan:
            xcomp[np.any(xcomp<0,axis=-1)] = np.nan
        return xcomp

    # Claude deleted old code here
    # Claude addition (start)
    def convert_activity_from_prim_to_comp(self,aprim,T=None,mu0_comp=None,mu0_prim=None,
                                           Rgas=8.314,assume_linear_standard_states=False):
        """
        If you have computed the activities of the primitive components, this
        function will return the corresponding activities of the composite
        components, each relative to the real standard state of that composite
        component (e.g. pure liquid Mg2SiO4), not relative to a fictive one.

        The default method goes through the chemical potentials:
            mu_prim = mu0_prim + R*T*ln(a_prim)
            mu_comp = nu^T mu_prim             (exact)
            a_comp  = exp((mu_comp - mu0_comp)/(R*T))
        For the reasoning why this is necessary (the standard states of the
        two systems are in general not related by the stoichiometry), see the
        docstring of convert_activity_from_comp_to_prim(). Unlike that
        function, this direction also works if the composite system is a
        lower-dimensional subspace of the primitive system.

        Arguments:

          aprim   The activities of the components in the primitive system
                  (array [...,ncomp_prim]), relative to the standard states
                  mu0_prim.

        Keywords:

          T, mu0_comp, mu0_prim, Rgas, assume_linear_standard_states
                  See convert_activity_from_comp_to_prim(). With
                  assume_linear_standard_states=True the old method (direct
                  linear transformation of ln(a)) is used.

        Returns:

          acomp   The activities of the components in the composite system
        """
        aprim  = np.array(aprim)
        if assume_linear_standard_states:
            # The old method (kept as an option)
            laprim = np.log(aprim)
            lacomp = (self.nu_prim_comp[...,:,:]*laprim[...,:,None]).sum(axis=-2)
            acomp  = np.exp(lacomp)
            return acomp
        if mu0_comp is None: mu0_comp = self.G_comp_comp
        if mu0_prim is None: mu0_prim = self.G_comp_prim
        assert T is not None, 'Error: Temperature T is needed to convert activities via the chemical potentials (or use assume_linear_standard_states=True).'
        assert mu0_comp is not None and mu0_prim is not None, 'Error: Standard-state chemical potentials of both systems are needed (pass mu0_comp and mu0_prim, or G_comp_comp and G_comp_prim at creation of the SubSystem), or use assume_linear_standard_states=True.'
        mu0_comp = np.array(mu0_comp)
        mu0_prim = np.array(mu0_prim)
        assert mu0_comp.shape[-1]==self.ncomp_comp, 'Error: mu0_comp has wrong length.'
        assert mu0_prim.shape[-1]==self.ncomp_prim, 'Error: mu0_prim has wrong length.'
        RT     = Rgas*np.array(T,dtype=float)
        if RT.ndim>0: RT = RT[...,None]
        muprim = mu0_prim + RT*np.log(aprim)
        mucomp = self.convert_chemical_potential_from_prim_to_comp(muprim)
        acomp  = np.exp((mucomp-mu0_comp)/RT)
        return acomp
    # Claude addition (end)
        
    def convert_chemical_potential_from_prim_to_comp(self,muprim):
        """
        If you have computed the chemical potentials of the primitive components, 
        this function will return the corresponding chemical potentials of the composite
        components.

        Arguments:

          muprim   The chemical potentials of the components in the primitive system

        Returns:

          mucomp   The chemical potentials of the components in the composite system
        """
        muprim  = np.array(muprim)
        mucomp = (self.nu_prim_comp[...,:,:]*muprim[...,:,None]).sum(axis=-2)
        return mucomp
        

# ==================================================================================
# Examples
# ==================================================================================

if __name__ == "__main__":

    # Example: Comparing the model of Berman in his PhD thesis to the MELTS model of Ghiorso & Sack 1995

    comp_prim   = ['SiO2','MgO']
    comp_comp   = ['SiO2','Mg2SiO4']
    G_comp_prim = np.array([-1124829., -718317.])  # The G values at the primitive components from the Berman PhD thesis model
    G_comp_comp = np.array([-1126980.,-2678878.])  # The G values at the composite components from the Ghiorso & Sack 1995 model
    
    subsys      = SubSystem(comp_prim,comp_comp,G_comp_prim,G_comp_comp)

    ncomp_prim  = len(comp_prim)
    ncomp_comp  = len(comp_comp)
    nx          = 100
    x1d         = np.linspace(0,1,nx+1)
    xcomp       = np.zeros((nx+1,ncomp_comp))
    xcomp[:,0]  = x1d
    xcomp[:,1]  = 1-x1d
    xprim       = subsys.convert_from_xcomp_to_xprim(xcomp)

    # Figure showing the conversion from one into another
    plt.figure()
    for i in range(ncomp_prim):
        plt.plot(xprim[:,i],xcomp[:,i],label=f'{comp_prim[i]},{comp_comp[i]}')
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
    xprim       = np.zeros((nx+1,ncomp_prim))
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
    prim_G0      = (xprim*G_comp_prim[None,:]).sum(axis=-1)          # The mean of the components
    prim_Gi      = Rgas*T*(xprim*np.log(xprim+1e-99)).sum(axis=-1)   # The ideal mixing term
    prim_Gw      = prim_WG_smmm*xprim[:,0]*xprim[:,1]**3             # The first of the Margules terms
    prim_Gw     += prim_WG_ssmm*xprim[:,0]**2*xprim[:,1]**2          # The second of the Margules terms
    prim_Gw     += prim_WG_sssm*xprim[:,0]**3*xprim[:,1]             # The third of the Margules terms
    prim_G       = prim_G0 + prim_Gi + prim_Gw
    prim_G0i     = prim_G0 + prim_Gi
    
    # Ghiorso & Sack Gibbs energy of the liquid
    comp_G0      = (xcomp*G_comp_comp[None,:]).sum(axis=-1)          # The mean of the components
    comp_Gi      = Rgas*T*(xcomp*np.log(xcomp+1e-99)).sum(axis=-1)   # The ideal mixing term
    comp_Gw      = comp_WG_sm*xcomp[:,0]*xcomp[:,1]                  # The Margules term (interaction term)
    comp_G       = comp_G0 + comp_Gi + comp_Gw

    # Convert the Ghiorso & Sack Gibbs energy of the liquid to the equivalent moles in the primitive system
    comp_G_prim  = subsys.convert_G_value_from_composite_to_primitive_system(xcomp,comp_G)
    comp_G0i_prim= subsys.convert_G_value_from_composite_to_primitive_system(xcomp,comp_G0+comp_Gi)
    
    # The baseline (plane) to subtract for easier viewing
    #G_nomix_prcm = subsys.compute_G_nomix_in_prim_system_at_locations_of_composite_components()
    G_plane_prim = (xprim*G_comp_prim[None,:]).sum(axis=-1)
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
