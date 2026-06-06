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
import pandas as pd
from scipy.spatial import ConvexHull

class CrystalDatabase(object):
    """
    Class for a database of the physical properties of fixed-composition phases, 
    i.e., minerals with a well-defined (typically stoichiometric) composition.
    Each of these will be a single point on the phase diagram. For simplicity
    of nomenclature we call these phases 'crystals' (although of course there
    are many crystal solid solutions with continuous compositions, see the
    class SolidSolution below)

    A note on scaling of the phases (formula units):
    A phase can often be written with any multiple of a formula unit. For example:
    the phase solid crystalline enstatite MgSiO3 can also be written as Mg2Si2O6.
    It is exactly the same. Furthermore, if you mix 0.5 mole of SiO2 with 0.5 mole
    of MgO (in total "1 mole of constituting system component") and form enstatite,
    you will only form 0.5 mole of MgSiO3, which would be 1 mole of the formula
    unit Mg(1/2)Si(1/2)O(3/2). To be able to put all phases correctly into the
    same phase diagam, we must choose the formula unit that corresponds to one
    mole of constituting system component, in this case Mg(1/2)Si(1/2)O(3/2).
    In the usual mineral databases, however, these minerals are listed with their
    more natural formulae, such as MgSiO3. The 'dbase' table given as argument
    to CrystalDatabase (see below) must therefore contain information about how
    to scale from the listed mineral MgSiO3 to the required Mg(1/2)Si(1/2)O(3/2).
    The column 'moles' should contain this multiplicity factor, which is 2 in the
    example of MgSiO3 and 4 if the same mineral is listed as Mg2Si2O6 (of course
    all assuming that the system components are SiO2 and MgO). The column 'mfDfG'
    is then the scaled Gibbs free energy of formation, such that mfDfG=DfG/moles.
    """
    def __init__(self,dbase,resetfunc=None,resetdbfunc=None,components=None,factors=None):
        """
        Provide or read a database of fixed composition phases.

        Arguments:

          dbase        Either a string containing the name of the .csv or fixed-width-format
                       file containing the database, or a Pandas DataFrame of the database.
                       The database must have at least the columns:
                         "Abbrev"     The abbreviated name of the solid
                         "Formula"    The chemical formula, e.g. "Mg2SiO4"
                         "x"          The location in the phase diagram: An array x[0:ncomp]
                                      with ncomp the nr of components.
                         "moles"      How many moles you get if you mix 1 mole of system
                                      components according to x to get this phase.
                                      Example: system components [SiO2,MgO], phase
                                      Mg2SiO4, then moles=3.0 and x=[(1/3),(2/3)].
                         "DfG"        The Delta_f G or Delta_a G Gibbs energy of formation
                         "mfDfG"      As DfG, but scaled to "per mole of system component",
                                      i.e. mfDfG = DfG/moles.
                       Other columns can be added for the reset function (see resetfunc below).
                       Typically the DfG and mfDfG are computed by the reset function for
                       a given T and P.

                       Note: If components (possibly with factors) is given as keyword
                             (see below), then the 'x', 'moles' and 'mfDfG' columns are
                             computed, so you don't have to provide them in the table.

        Optional:

          resetfunc    A function with arguments (T,P), i.e. temperature (in Kelvin) and
                       pressure (in bar) that recomputes the Gibbs energy (in the mfDfG
                       column)

          resetdbfunc  A function with arguments (T,P), i.e. temperature (in Kelvin) and
                       pressure (in bar) that returns a new version of the pandas
                       database. If you do not provide this function, you should ensure
                       that, when resetting to a different T and/or P, the dbase you have
                       passed to this class remains the same instance (just with different
                       values in the 'DfG' and 'mfDfG' columns). If your model creates,
                       instead, a new dbase each time the resetfunc() is called, then
                       you must also provide the resetdbfunc() function to allow PhaseHull
                       to re-link the correct dbase.

          components   List of chemical formulae of the system components. If given,
                       then the 'x', 'moles' and 'mfDfG' columns of the database are
                       automaticlly computed.
        
          factors      If list or array numbers of length len(components), then
                       the components are, in fact, components*factors. Useful
                       for when a component is, e.g., MnSi0.5O2 instead of
                       Mn2SiO4 (the component would then be Mn2SiO4 with a
                       factor of 0.5). Note: Only used when components is given.
        """
        self.components = components
        self.factors    = factors
        if type(dbase) is str:
            if dbase[-4:]=='.csv':
                dbase = pd.read_csv(dbase)
            elif dbase[-4:]=='.fwf':
                dbase = pd.read_fwf(dbase)
            else:
                raise ValueError(f'Do not know how to read {dbase}')
        elif type(dbase)!=pd.DataFrame:
            raise  ValueError(f'Error: dbase must be a pandas DataFrame')
        if components is not None:
            # If components is given, then immediately make selection
            # and compute the 'x', 'mfDfG' and 'moles' columns
            from . import extract_from_mineral_database_based_on_components
            if factors is None:
                factors = np.ones(len(components))
            dbase = extract_from_mineral_database_based_on_components(dbase,components,factors=factors)
        self.dbase  = dbase
        self.reset  = resetfunc
        self.resetdb= resetdbfunc
        # For convenience: A dictionary from abbreviation to integer index
        self.index  = {index: value for value, index in enumerate(self.dbase['Abbrev'])}

    def call_reset(self,T,P):
        if self.reset is not None:
            self.T = T
            self.P = P
            self.reset(T,P)
            if self.resetdb is not None:
                dbase = self.resetdb(T,P)
                if self.components is not None:
                    from . import extract_from_mineral_database_based_on_components
                    factors = self.factors
                    if factors is None:
                        factors = np.ones(len(self.components))
                        dbase = extract_from_mineral_database_based_on_components(dbase,self.components,factors=factors)
                self.dbase = dbase

class Liquid(object):
    """
    Class for 'Solutions': Typically a liquid, or more general a
    non-stoichiometric material (e.g., an alloy or a glas), which has a
    Gibbs energy for any value of fractional abundance vector x.

    The name 'Liquid' was chosen instead of 'Solution' because (1) the
    word 'solution' has too many other meanings, and can be confusing,
    and (2) for the geophysical and astrophysical community the most
    common uses would be for melts. But this class can equally well be
    used for solid solutions. The only condition is that a Gibbs energy
    can be defined on the entire domain of the system (all x positions).
    For restricted solid solutions, such as 'joins' in the phase diagram,
    please use the SolidSolution class instead.
    """
    def __init__(self,name,components,Gfunc,kwforGfunc=None,resetfunc=None,
                 gammafunc=None,factors=None,Hfunc=None):
        """
        Arguments:

          name         The name of this liquid or alloy. Can be as simple as e.g.,
                       'liquid' (for a melt) or 'solid' (for an alloy or glas).
                       Only important if one has more than one continuous
                       non-stoichiometric phases.

          components   List of names of the system components out of which this liquid
                       or alloy is composed. Important to assure the appropriate
                       order of the x-values when this class is used in other
                       parts of the code. For example, if components =
                       ['MgO','SiO2','CaO'] and the fractional abundances are
                       given as [0.2,0.7,0.1], then it means that the liquid has
                       0.2 mole fraction of MgO, 0.7 mole fraction of SiO2 and
                       0.1 mole fraction of CaO. Particularly important if other
                       parts of the code use only a subset of these components,
                       or a different order of them.

          Gfunc        A Python function for Gibbs(x) where x is either a 1D array
                       [x[0],x[1]...x[M-1]] with M = len(components) or x is a
                       2D array x[N,M] with N is the number of sampling points
                       in the M-dimensional phase space. The function Gfunc must
                       be able to accept these 2D x arrays (i.e., 1D array of
                       arrays x[0:M]), for efficiency, so that a single call to
                       Gfunc returns a 1D array or G values at all x-points.

        Optional:

          kwforGfunc   A dictionary of possible keyword arguments for the Gfunc,
                       in case your function needs additional information to be
                       passed on.

          resetfunc    A function with arguments (T,P), i.e. temperature (in Kelvin) and
                       pressure (in bar) that reconfigures the Gfunc to compute
                       the Gibbs energy for the new T and P.

          gammafunc    If set, this is a function gamma(x), the activity
                       coefficient, defined such that the activity a=gamma*x.

          factors      If list or array numbers of length len(components), then
                       the components are, in fact, components*factors. Useful
                       for when a component is, e.g., MnSi0.5O2 instead of
                       Mn2SiO4 (the component would then be Mn2SiO4 with a
                       factor of 0.5).

          Hfunc        If set, this is the enthalpy function. Not necessary
                       for phasehull, but useful for energy considerations.

        IMPORTANT NOTE: In PhaseHull all liquids are mapped onto the same grid.
                        For each grid point, PhaseHull will then check which of
                        the liquids has the lowest Gibbs energy, and will take
                        that liquid for that point. This is different from the
                        SolidSolution class, where each instance has its own
                        set of points (and therefore its own grid). 
        """
        self.name             = name
        self.components       = np.array(components)
        self.factors          = factors
        self.Gfunc            = Gfunc
        self.Hfunc            = Hfunc
        self.kwforGfunc       = kwforGfunc
        self.reset            = resetfunc
        self.gammafunc        = gammafunc

    def call_Gfunc(self,compon,x):
        """
        Interface to the Gibbs function provided by the user, allowing it to
        be used for a subset or different order of components.

        Arguments:

          compon       List of names of the components in the correct order
                       in which they are in x. All compon must be part of the
                       self.components list.

          x            Array x[N,M] of mole fractions such that x.sum(axis=1)==1.
                       Here M is len(compon), and N is the number of sampling
                       points.

        Returns:

          G            An array G[N] of Gibbs energy values.
        """
        assert set(compon)<=set(self.components), f'Error: Not all components {compon} are included in liquid {self.name}'
        if len(x.shape)==1:
            x = np.array([x,])
        assert len(x.shape)==2, 'Error: x must be an array x[N,M].'
        assert len(compon)==x.shape[-1], 'Error: The number of components must equal the x.shape[1].'
        M   = len(self.components)
        N   = x.shape[0]
        idx = []
        for e in compon:
            idx.append(np.where(self.components==e)[0][0])
        xx  = np.zeros((N,M))
        for i in range(len(compon)):
            xx[:,idx[i]] = x[:,i]
        if self.kwforGfunc is not None:
            G   = self.Gfunc(xx,**self.kwforGfunc)
        else:
            G   = self.Gfunc(xx)
        return G

    def call_reset(self,T,P):
        """
        To be able to change temperature T or pressure P of this liquid, you have to provide
        a function called 'reset()'. If that function is part of an external class, and stores
        its information there, then all is fine. But what if you prefer to store the information
        (e.g., updated coefficients of the G(x) function model) in the Liquid class? The way to
        do this is to make your function reset() return a dictionary, that will then be stored
        in the Liquid class as self.kwforGfunc.

        The function call_reset() is a wrapper around your reset() function, allowing to 'catch'
        the output of your reset() function and store it in self.kwforGfunc. 
        """
        if self.reset is not None:
            self.T = T
            self.P = P
            kwforGfunc = self.reset(T,P)
            if kwforGfunc is not None:
                self.kwforGfunc = kwforGfunc

class SolidSolution(object):
    """
    The solid solution crystal families, often called 'joins'. Note: only recommended if 
    this family has a lower dimension than the nr of components. Otherwise: better use 
    an instance of the Liquid class and add to self.liquids, as that class has more features
    such as grid refinement and more. The solid solution class is a rather primitive
    class to add a grid of points independent of the liquid grid, and typically a
    subspace of the full x-space. Example: We have CaO,MgO,SiO2 ternary space, but 
    we want to add the Ca(1-y)Mg(y)MgSi2O6 binary solid solution (the Enstatite-
    Diopside join) inside it.

    This class has a method setup_base_grid() to create a simple regular grid on the
    subspace of this solid solution. That does not mean, however, that only regular
    grids are allowed. You can put your own grid onto this subspace, or add gridpoints
    to the regular grid. This way you can make your own grid refinement. However, an
    automatic grid refinement, as it is implemented in PhaseHull for the Liquid class
    instances, is not offered for SolidSolution class instances.
    """
    def __init__(self,name,components,endmembers,Gfunc,mdb,endmfact=None,kwforGfunc=None,resetfunc=None,
                 ygrid=None,nres=30,factors=None):
        """
        Arguments:

          name          The name you want to give to this solid solution

          components    Array of formulae of the components of the full system, in which
                        you intend to embed this solid solution. E.g. ['CaO','MgO','SiO2']

          endmembers    Array of formulae or names of the endmembers of the solid solution. 
                        This array should have a length smaller or equal to the length of the
                        components array. The formulae or names must be listed in the
                        mineral database mdb (below). NOTE: It can be that the endmember
                        you wish is only listed with another multiplication factor in the
                        mdb database, e.g. MgSiO3 while your model for the solid solution
                        uses e.g. Mg2Si2O6. If that is so, please give (for this example)
                        the 'MgSiO3' in endmembers, and specify endmfact below, setting
                        the corresponding value of endfact to 2. This signals the code
                        that the endmember formula unit is 2x MgSiO3 = Mg2Si2O6.

                        A note on reciprocal solutions: Often the solid solution has
                        endmembers that are not the endpoints of a simplex. Example:
                        the pyroxene solid solution of Enstatite (Mg2Si2O6), Ferrosilite
                        (Fe2Si2O6), Diopside (CaMgSi2O6), and Hedenbergite (CaFeSiO6)
                        are part of the triangle (2D simplex) spanned by Enstatite --
                        Ferrosilite -- Wollastonite (Ca2Si2O6). But Wollastonite is
                        excluded from the solution, and so are all compositional points
                        with more than 50% of Ca2Si2O6 endmember. The way it works here
                        is that you still have to give Mg2Si2O6, Fe2Si2O6 and Ca2Si2O6
                        as "endmembers", and the ensure that the grid only covers the
                        valid part of the triangle. The "reciprocal" internal degree
                        of freedom at a given composition is then something that you
                        (the user) have to fix (so that at each grid point on the
                        En-Fs-Di-Hd quadrilateral only has one unique value of G).
                        Typically this "fixing" is done by finding the minimum G
                        along this internal degree of freedom.

          Gfunc         The function that computes the Gibbs energy as a function of the
                        y coordinate within the solid solution. Must be a function Gfunc(y).
                        It should return the Gibbs energy per mole of endmember. In the
                        example of components = ['CaO','MgO','SiO2'], endmembers =
                        ['MgSiO3','CaMgSi2O6'] and endmfact = [2.,1.] (meaning that the
                        actual endmembers are Mg2Si2O6 and CaMgSi2O6), the Gfunc should
                        give the Gibbs energy per mole of y*Mg2Si2O6 + (1-y)*CaMgSi2O6,
                        which is 4 times the Gibbs energy per mole of component CaO,
                        MgO and SiO2. 

        Optional:

          endmfact      Array of multiplication factors for the endmembers given in the
                        `endmembers` argument. If set to None, it is by default 1 for
                        all endmembers. For the above example of a mdb database containing
                        only MgSiO3, while your endmember formula unit is Mg2Si2O6, you
                        should then set the endfact multiplication factor for that
                        endmember to 2.0. So for a solid solution of endmembers
                        Mg2Si2O6 and CaMgSi2O6, with a mineral database containing
                        MgSiO3 and CaMgSi2O6, you should set endmfact = [2.,1.].

          factors      If list or array numbers of length len(components), then
                       the components are, in fact, components*factors. Useful
                       for when a component is, e.g., MnSi0.5O2 instead of
                       Mn2SiO4 (the component would then be Mn2SiO4 with a
                       factor of 0.5).

          kwforGfunc   A dictionary of possible keyword arguments for the Gfunc,
                       in case your function needs additional information to be
                       passed on.

          resetfunc    A function with arguments (T,P), i.e. temperature (in Kelvin) and
                       pressure (in bar) that reconfigures the Gfunc to compute
                       the Gibbs energy for the new T and P.

          ygrid        Your specified y-grid in the solid solution space.

          nres         If you do not specify ygrid, then nres is the number of grid points
                       along each axis of the solid solution space. A linear (triangular)
                       grid is then set up.
        """
        self.name       = name
        self.components = components
        self.factors    = factors
        self.endmembers = endmembers
        if endmfact is None:
            self.endmfact = np.ones(len(self.endmembers))
        else:
            assert len(endmfact)==len(self.endmembers), 'Error in endfact: If you specify endfact, it must be an array of numbers with the same length as the endmember array.'
            self.endmfact   = endmfact
        self.mdb        = mdb
        assert 'x' in self.mdb.columns, 'Error: the mineral database you provided does not have the x column yet.'
        self.ncomp      = len(components)
        self.nendm      = len(endmembers)
        assert self.nendm<=self.ncomp, 'Error: Cannot have more endmembers of this solid solution than components.'
        self.xendm      = np.zeros((self.nendm,self.ncomp))
        self.mfendm     = np.ones(self.nendm)
        self.mfGfact    = 1.0
        self.Gfunc      = Gfunc
        self.reset      = resetfunc
        self.kwforGfunc = kwforGfunc
        self.setup_base_grid(ygrid=ygrid,nres=nres)

    def setup_base_grid(self,ygrid=None,nres=30):
        if ygrid is None:
            ygrid       = make_x_grid(nres,self.nendm)
            # Maybe we need to remove the corner points? Or the pure crystals?
        self.ygrid      = ygrid
        for iend,name in enumerate(self.endmembers):
            if name in list(self.mdb['Abbrev']):
                self.xendm[iend,:] = self.mdb['x'][self.mdb['Abbrev']==name].iloc[0].copy()
                self.mfendm[iend]  = self.mdb['moles'][self.mdb['Abbrev']==name].iloc[0]/self.endmfact[iend]
            elif name in list(self.mdb['Formula']):
                self.xendm[iend,:] = self.mdb['x'][self.mdb['Formula']==name].iloc[0].copy()
                self.mfendm[iend]  = self.mdb['moles'][self.mdb['Formula']==name].iloc[0]/self.endmfact[iend]
            else:
                print(f'Error: Could not find endmember {name} in the mineral database you provided.')
        # For now we insist that mfendm is the same value for all endmembers, to avoid having
        # to make a non-linear transformation when computing G.
        self.mfGfact    = self.mfendm[0]
        for iend in range(1,len(self.mfendm)):
            assert self.mfendm[iend]==self.mfGfact, 'Error: For now the SolidSolution class cannot (yet) handle cases where the endmembers have different mfendm factors (an mfendm factor is e.g. 0.25 for endmember Mg2Si2O6 in a component system of CaO,MgO,SiO2). Will hopefully some day be built in.'
        self.xgrid      = (self.ygrid[:,:,None]*self.xendm[None,:,:]).sum(axis=1)

    def call_Gfunc(self):
        if self.kwforGfunc is not None:
            G   = self.Gfunc(self.ygrid,**self.kwforGfunc)
        else:
            G   = self.Gfunc(self.ygrid)
        return G * self.mfGfact

    def call_reset(self,T,P):
        if self.reset is not None:
            self.T = T
            self.P = P
            kwforGfunc = self.reset(T,P)
            if kwforGfunc is not None:
                self.kwforGfunc = kwforGfunc

class PhaseHull(object):
    """
    Class for the PhaseHull basic algorithm. This class performs the PhaseHull computations, and makes the
    basic classifications of the simplices. It does not contain any physical/chemical data. All that has
    to be provided as arguments to this class.
    """
    def __init__(self,components,crystals=None,liquids=None,solsols=None,T=None,P=None,nres0=30,nrefine=4,nfact=2,nspan=2, \
                 min_nr_tielines=2,nocompute=False,incl_ptnames=True,incl_xvals=True,incl_Gvals=True,        \
                 incl_Gcen=False,incl_xcen=False,incl_xtie=False,mrcrit=10.,refinepoints=None,xgrid=None,
                 factors=None,epsequ=1e-18,verbose=False):
        """
        Arguments:

          components   List of names of the components in the correct order
                       in which they are in the x fractional composition vector
                       to be used.

          crystals     A list of instances of the CrystalDataBase class. Usually just one.

          liquids      A list of instances of the Liquid class. Typically 1 (liquid) or 2 (liquid,solid).
                       Notice that liquid simply means: non-stoichiometric solution. It does not have
                       to be truly liquid. For instance, alloys are also continuous mixed phases, so they
                       would also be a 'liquid' in this sense.

          solsols      Some solid solutions may only exist in a lower-dimensional subspace of the full
                       system. Example: Ca(1-x)Mg(x)SiO4 in the CaO-MgO-SiO2 system. This cannot be
                       modelled using a Liquid class. For this the SolidSolution class is designed.
                       However, please note that this class is less powerful (e.g. it does not
                       participate in the automatic grid refinement). So if you can use the Liquid
                       class for your solid solution, then that is advisable.

          nres0        If liquids are present, nres0 determines the base grid resolution: it is the
                       number of grid spacings between 0 and 1.

          nrefine      If liquids are present, nrefine determines the nr of cycles of grid refinement.

          nfact        If liquids are present, nfact gives the refinement factor per refinement step.

          nspan        If liquids are present, nspan gives the width of the refinement zone around the
                       binodal curves.

          min_nr_tielines   Is a criterion for grouping of tie simplices.

          T            Temperature in Kelvin

          P            Pressure in bar

        There are a set of options for the kind of information to be stored for each simplex:
        
          incl_ptnames     Include the names of all the points of the simplex
          incl_xvals       Include the x-coordinates of all the points of the simplex
          incl_Gvals       Include the G values of all the points of the simplex
          incl_xcen        Include the x center of the simplex
          incl_Gcen        Include the G center of the simplex
          incl_xtie        For those simplices that are tie lines, include the tie line x values

        Other options:

          xgrid        If you want to have full control of the base grid of the liquids (before
                       refinement) you can set xgrid. It should be an array with indices [ipt,icomp]
                       where ipt is the index of the point, and icomp goes from 0 to M-1, which is
                       the index of the component. IMPORTANT NOTE: Make sure that nres0 is properly
                       set to 1/dx, where dx is the smallest grid spacing in the xgrid. This is
                       important for the grid refinement (if you set/keep nrefine>0).

          factors      If list or array numbers of length len(components), then
                       the components are, in fact, components*factors. Useful
                       for when a component is, e.g., MnSi0.5O2 instead of
                       Mn2SiO4 (the component would then be Mn2SiO4 with a
                       factor of 0.5).

          epsequ       The convex hull computation sometimes produces simplices that are
                       exactly "vertical", meaning their normal vector has the G-component
                       exactly 0. However, in numerics, this 'exact 0' sometimes has numerical
                       errors, and so the simplex may have an equations[:,-2] of the order of
                       -1e-22 or so. Since this is <0, it would be included as a simplex of
                       the phase diagram. This is not a major problem, but it is not really
                       desired. So as a default the threshold is equations[:,-2]<-epsequ, where
                       the default value of epsequ is set to 1e-18, which is usually much
                       smaller than any relevant values, and also substantially bigger than
                       the typical double predition errors. If you set it to 0.0, then this
                       slightly more selective threshold is not included.

        Calling PhaseHull will automatically start the computation of the phase diagram for the
        given temperature T and pressure P for which the crystals and liquid(s) are given. If a liquid
        or other non-stoichiometric material is included, then also a grid refinement is performed.
        This will create a set of every refined calculations. The corresponding results are all
        in the lists self.the<something> where <something> is e.g. pts, ids, simplices, hulls etc.
        The final (highest resulution) one is always the last one, which can be easiest obtained
        with the [-1] index, as in pts = self.thepoints[-1], and simplices = self.thesimplices[-1].

        Result:

          PhaseHull's product is a set of N simplices (= line elements in a binary system, triangles
          in a ternary system, tetrads in a quaternary system etc) that fill the entire phase space
          in the x-coordinates (= the region between 0 and 1 in a binary system, the triangle between
          [1,0,0], [0,1,0] and [0,0,1] for a ternary system etc). The number of simplices N can be
          any number >= 1, and tends to be larger, the more complex the phase diagram is. The
          information about this set of simplices is located in PhaseHull.thesimplices[-1]. So you
          have

            phull       = PhaseHull(....)
            simplices   = phull.thesimplices[-1]

          This is a dictionary containing a set of arrays/lists of length N (one value for each simplex).
          These arrays or lists are:

            simplices['x']          np.array((N,ncomp,ncomp)) where ncomp is the number of components, 
                                    such that simplices['x'].sum(axis=-1)==1. The last index is 
                                    therefore the index of the coordinates. The middle index counts
                                    the corners of the simplex, and the first index is, as said above,
                                    the index counting the simplices.
            simplices['stype']      The list of physical types of the simplices. E.g. if
                                    simplices['stype'][5]=='cryst_1_liq_1', then simplex 5 is of the
                                    type of a coexistence of one crystals with one liquid.
            simplices['id']         Just in case the order of the N simplices gets mixed up, this is
                                    the original integer index of the simplices.
            simplices['id_qhull']   The index of the simplex as it was in the Qhull results. Note that
                                    not all Qhull simplices are included here, because Qhull also
                                    returned the simplices at the top (rather than the bottom) of the
                                    convex hull.
            simplices['ipts']       np.array((N,ncomp)) of the indices of the points at the corners. 
                                    The coordinates of the points can be found in PhaseHull.thepoints[-1].
            simplices['G']          np.array((N,ncomp)) of the G values at the corners. Only if
                                    incl_Gvals is set.
            simplices['neighbors']  np.array((N,ncomp)) of the indices of the neighboring simplices.
                                    A neighbor shares a face with another simplex.

          Sometimes there may be fewer or more contents than these. With simplices.keys() you can
          get a list of all the contents of this dictionary.

        In cases with a large number of simplices (e.g., when you have one or more liquids), it might
        become a bit difficult to get an overview. Here are some methods to get a better overview:

          PhaseHull.select_simplices_of_a_given_kind(stype):
                   Returns the indices (for the above simplices arrays) of all simplices of the
                   kind stype, where stype is a string such as e.g., 'allcryst' or 'cryst-1-liq-1'.

          PhaseHull.get_tie_lines_simplices():
                   Returns the indices of all simplices that are tie lines.

          PhaseHull.get_tie_line_groups():
                   Returns a list of groups of tie lines (for ternary or larger systems). This is
                   useful because in ternary or larger systems, tie lines are usually in bundles.

          ...more to come...

        If you want to redo the computation for a different temperature or pressure, you can call:

          PhaseHull.reset(T,P)

        which then also automatically starts the computation, unless you set nocompute=True as keyword.

        IMPORTANT NOTE:

          So far (16 Sep 2025) only up to ternary phase diagrams have been thorougly
          tested. Not sure if all simplex types are complete for higher order
          systems. This may need some additional work.
        """
        self.verbose      = verbose
        self.epsequ       = epsequ
        self.mrcrit       = mrcrit
        self.components   = components
        self.factors      = factors
        self.ncomponents  = len(components)
        #self.ncomp        = 0
        self.ncryst       = 0
        self.nliq         = 0
        self.nsolsol      = 0
        if self.ncomponents>3:
            print('Note: At this moment the tie line identification has not been tested for nr of components>3.')

        # Add the crystals to PhaseHull
        if crystals is not None:
            self.ncryst = len(crystals.dbase)
            if type(crystals) is not list:
                # Note: The fact that also the crystals are a list is a relic. In hindsight not necessary
                crystals  = [crystals]
            # Do a self-check
            for i,cr in enumerate(crystals):
                if cr.components is not None:
                    assert np.all(np.array(components)==np.array(cr.components)), 'Error: The components of the crystal database are unequal to those of phasehull'
                    if cr.factors is not None and self.factors is not None:
                        assert factors==cr.factors, 'Error: The factors of the components of the crystal database are unequal to those of phasehull'
        else:
            crystals      = []
        self.crystals     = crystals

        # Add the liquid(s) to PhaseHull
        if liquids is not None:
            if type(liquids) is not list:
                # Liquids must be a list of liquids, because you can have multiple ones
                # (e.g. true liquid and a vapor phase, or true liquid and a solid solution).
                liquids   = [liquids]
            self.nliq     = len(liquids)
            self.liqnames = {}
            for i,lq in enumerate(liquids):
                self.liqnames[lq.name] = i
            # Do a self-check
            for i,lq in enumerate(liquids):
                if lq.components is not None:
                    assert np.all(np.array(components)==np.array(lq.components)), 'Error: The components of the liquid are unequal to those of phasehull'
                    if lq.factors is not None and self.factors is not None:
                        assert factors==lq.factors, 'Error: The factors of the components of the liquid are unequal to those of phasehull'
        else:
            liquids       = []
        self.liquids      = liquids

        # Add the solid solutions to PhaseHull
        if solsols is not None:
            if type(solsols) is not list:
                # Solsols must be a list of solsols, because you can have multiple ones.
                solsols   = [solsols]
            self.nsolsol  = len(solsols)
            self.solsolnames = {}
            for i,ss in enumerate(solsols):
                self.solsolnames[ss.name] = i
        else:
            solsols       = []
        self.solsols      = solsols

        # Update the temperature and pressure, and update all the
        # Gibbs free energies accordingly
        self.T            = T
        self.P            = P
        if T is not None or P is not None:
            for c in self.crystals:  # First the crystals as the may be used by liquids and solsols
                c.call_reset(T,P)
            for l in self.liquids:
                l.call_reset(T,P)
            for s in self.solsols:
                s.call_reset(T,P)

        # Set defaults
        self.nres0        = nres0
        self.nrefine      = nrefine
        self.nfact        = nfact
        self.nspan        = nspan
        self.min_nr_tielines = min_nr_tielines
        self.incl_ptnames = incl_ptnames
        self.incl_xvals   = incl_xvals
        self.incl_Gvals   = incl_Gvals
        self.incl_Gcen    = incl_Gcen
        self.incl_xcen    = incl_xcen
        self.incl_xtie    = incl_xtie
        self.refinepoints = refinepoints
        self.tieline_types= ['tieline_c0l2','tieline_c0l3','tieline_c1l2','tieline_c1l3',\
                             'tieline_c1l0s2','tieline_c0l0s3','tieline_c0l1s2','tieline_c0l2s1',\
                             'solsol_coexist','solsol_inmisc']
                            # Not possible should be 'tieline_c1l1s1', 'tieline_c2l0s1',

        # Wrap up
        if xgrid is not None:
            self.user_xgrid   = np.stack(xgrid)
        else:
            self.user_xgrid = None
        if not nocompute:
            self.compute()

    def reset(self,T,P=1,nocompute=False):
        """
        PhaseHull is designed to create a phase diagram for a given (fixed)
        temperature T [Kelvin] and pressure P [bar], in the space of x (the
        composition given by fractional abundances). To reset to a different
        temperature and pressure, use this function.

        Arguments:

          T       Temperature in Kelvin

          P       Pressure in bar
        
        """
        self.T = T
        self.P = P
        for c in self.crystals:  # First the crystals as the may be used by liquids and solsols
            c.call_reset(T,P)
        for l in self.liquids:
            l.call_reset(T,P)
        for s in self.solsols:
            s.call_reset(T,P)
        if not nocompute:
            self.compute()

    def compute(self):
        """
        This runs all the basic computations for the phase diagram.
        It is the core of PhaseHull.
        """
        self.setup_all_points_before_refinement()
        self.thehulls     = []
        self.thesimplices = []
        self.do_all_refinement_steps()
        self.get_tie_lines_simplices()
        if self.ncomponents==3:
            self.get_tie_line_groups()
        if len(self.crystals)>0:
            self.find_stable_crystals()

    def recompute_with_selected_phases(self,incl_liq=True,incl_solsol=False,incl_cryst=False):
        """
        It can be convenient to be able to compute the liquid G(x) surface at all (refined) grid
        points as if the other phases (solsol and crystals) are not there. Or compute the
        convex hull of all crystals, as if the liquid and solsol phases were not there. This is
        purely for analysis purposes, not for final science results. This function will
        allow you to do this. It works as if a new refinement layer is computed, but instead
        of refining the grid, it will keep the grid the same, just switch off/on the liquid and/or
        solsol and/or crystal phases.

        Arguments:
          incl_liq      If True, then plot the liquid
          incl_solsol   If True, then plot the solsols
          incl_cryst    If True, then plot the crystals
        """
        self.redo_computation_with_selected_phases(incl_liq=incl_liq,incl_solsol=incl_solsol,incl_cryst=incl_cryst)
        self.get_tie_lines_simplices()
        if self.ncomponents==3:
            self.get_tie_line_groups()
        if len(self.crystals)>0:
            self.find_stable_crystals()

    def create_index_for_the_simplices(self):
        """
        If you know the index of a point, or pair of point indices, or triple of points,
        which make up a corner, ribbon, wall etc of a simplex of the convex hull, you
        can find the simplex back using this index. For instance, to find all simplices
        that contain point 6:

          isims = self.index_for_simplices[-1][1][6]

        provided, of course, that point 6 is indeed on the convex hull. Let us start with
        a given simplex with index isim=10.

          import phasehull as ph
          phull = ph.PhaseHull(....fill in here....)
          simplices = phull.thesimplices[-1]      # Get the highest resolution grid
          phull.create_index_for_the_simplices()  # Create the index
          isim = 10
          ipts = list(simplices['ipts'][isim])
          ipts.sort()
          isims_points = phull.index_for_simplices[-1][1][ipts[0]]            # Only for point 0
          isims_walls  = phull.index_for_simplices[-1][2][(ipts[0],ipts[1])]  # Only for the pair points 0 and 1

        isims_points are the indices of all simplices that contain point 0.
        isims_walls are the indices of all simplices that contain points 0 and 1.
        """
        self.index_for_simplices = []
        for ilevel in range(len(self.thesimplices)):  # The refinement levels
            simplices = self.thesimplices[ilevel]
            indices   = {}
            for idim in range(1,self.ncomponents+1):    # The indices based on points, line ribbons, triangle walls, tetrads etc.
                indices[idim] = {}
                for isim in range(len(simplices['id'])):
                    ipts = simplices['ipts']
                    if idim<self.ncomponents:
                        walls = self.get_walls(ipts[isim],depth=self.ncomponents-idim)
                        for index in walls:
                            if index in indices[idim]:
                                indices[idim][index].append(isim)
                            else:
                                indices[idim][index] = [isim]
                    else:
                        index = list(ipts[isim])
                        if len(index)>1:
                            index.sort()
                            index = tuple(index)
                        else:
                            index = index[0]
                        indices[idim][index] = isim
        self.index_for_simplices.append(indices)

    def add_to_index(self,indices,index,isim):
        if type(index) is tuple:
            if len(index)==1:
                index = index[0]  # No tuples of one element
        if index in indices:
            indices[index].append(isim)
        else:
            indices[index] = [isim]

    def get_walls(self,ipts,depth=1):
        walls = []
        for i in range(len(ipts)):
            w = list(ipts).copy()
            w.remove(w[i])
            w.sort()
            if depth>1:
                ww = self.get_walls(w,depth=depth-1)
                for w in ww:
                    walls.append(w)
            else:
                w = tuple(w)
                if len(w)==1:
                    w = w[0]      # No tuples of one element
                walls.append(w)
        walls = list(set(walls))
        walls.sort()
        return walls

    def select_simplices_of_a_given_kind(self,stype):
        """
        The product of PhaseHull is a set of simplices (in a binary model = line elements,
        in a ternary model = triangles, in a quaternary model = tetrads) describing the
        bottom of the convex hull. PhaseHull will try to classify the physical meaning of
        each of these simplices. For instance, a tiny simplex is likely part of a continuous
        liquid; if the analytic G(x) lies below that simplex, then this simplex is clearly
        a discrete part of the liquid. A very elongated simplex, in particular when the
        analytic G(x) lies above it, is very likely a tie line. Currently implemented
        type (called 'stype') are:

           liquid            Fully liquid
           allcryst          A coexistence of 2 (for binary) or 3 (for 
                             ternary) (etc) crystals.
           cryst_1_liq_1     A coexistence of 1 crystal phase with
                             1 liquid phase (only for binary diagrams).
           cryst_2_liq_1     A coexistence of 2 crystal phases with
                             1 liquid phase (only for ternary diagrams).
           cryst_1_liq_2     A coexistence of 1 crystal phase with
                             2 liquid phases (only for ternary diagrams),
                             but not a tie line.
           tieline_c1l2      Like cryst_1_liq_2, but with the physical
                             meaning of a tie line between 1 crystal and
                             1 liquid point (only for ternary diagrams).
                             This is a very elongated simplex. Its
                             meaning is that of a binary phase line 
                             in a ternary diagram. The two liquid points
                             are actually a single liquid, but it becomes 
                             two very-nearby liquid points due to the
                             discretization.
           tieline_c0l3      A coexistence of two liquid phases that
                             are inmiscible  (only for ternary diagrams).
                             In reality 3 liquids are connected, two
                             being very close together.
           inmisc_liquids_3phase   Like 'inmisc_liquids' but for
                                   three phases. In a ternary diagram
                                   these occasionally show up as big
                                   single triangles.
    
        But in the near future more will likely be implemented. If you want to
        select all simplices of one of these stypes, you can use this method.

        Arguments:

          stype   The type you wish to select.

        Returns:

          isel    A list of indices of the simplices selected.

        Usage of isel, for example:

          simplices = self.thesimplices[-1]
          x         = simplices['x'][isel]
        
        """
        simplices  = self.thesimplices[-1]
        nsim       = len(simplices['id'])
        #ncomp      = simplices['x'][0].shape[-1]
        iselection = []
        simindices = set()
        simplices['neighbors_'+stype] = [None for _ in range(nsim)]
        for isim in range(nsim):
            if simplices['stype'][isim]==stype:
                iselection.append(isim)
                simindices.add(isim)
        for isel in iselection:
            idxnew = []
            for i in range(len(simplices['neighbors'][isel])):
                k = simplices['neighbors'][isel][i]
                if k in simindices:
                    idxnew.append(k)
            simplices['neighbors_'+stype][isel] = idxnew
        return iselection

    def check_if_two_neighboring_tie_lines_are_parallel(self,i,k):
        # If they are parallel, they should share mutuall contain each others peak vertex
        simplices  = self.thesimplices[-1]
        assert 'tiepeak' in simplices, 'Error: Cannot check parallel neighbors without tiepeak info'
        ipt_peak_i = simplices['ipts'][i][simplices['tiepeak'][i]]
        ipt_peak_k = simplices['ipts'][k][simplices['tiepeak'][k]]
        return ipt_peak_i in simplices['ipts'][k] and ipt_peak_k in simplices['ipts'][i]

    def find_stable_crystals(self):
        """
        Once the convex hull has been computed, we can figure out which of the original crystals
        is stable, i.e. lies on the bottom of the convex hull. This will add a column 'stable' to the
        crystal database with True (stable) or False (unstable).
        """
        self.find_crystals_among_points()
        cdb = self.crystals[-1].dbase
        cdb['stable'] = False
        for i,row in cdb.iterrows():
            if row['ipt'] in self.thehulls[-1].vertices:
                cdb.at[i,'stable'] = True

    def find_crystals_among_points(self):
        """
        Before starting the convex hull algorithm, all points, liquid or crystal or other types of
        phases, are joined into a single big point cloud, self.thepoints[-1]. It can be useful to
        know which of these points correspond to which crystals. This function adds a column to
        the crystal database with the point indices.
        """
        ids = np.array(self.theids[-1])
        cdb = self.crystals[-1].dbase
        cdb['ipt'] = -1
        for i,row in cdb.iterrows():
            w               = np.where(ids==row['Abbrev'])[0]
            if len(w)==1:
                if self.check_mineral_within_TP_range(row):
                    cdb.at[i,'ipt'] = w[0]
                else:
                    cdb.at[i,'ipt'] = -3   # Signalling: Temperature or pressure out of range of this mineral
            elif len(w)==0:
                cdb.at[i,'ipt'] = -1   # Signalling: This mineral is not in the point cloud
            else:
                cdb.at[i,'ipt'] = -2   # Signalling: This mineral has multiple points

    def compute_tie_lines_from_simplices(self,iselection):
        """
        In PhaseHull tie lines are, in fact, very narrow tie simplices. Their width is of
        the order of the finest grid resolution, while their length can be anything. In
        a ternary diagram (2D) they are needle-like triangles. The 'blunt' side (the 'tie base')
        consists (in 2D) of two points, while the 'sharp' side (the 'tie peak') has only one 
        point. The tie line it represents is best approximated by the line going from the
        tie peak to the middle of the tie base.

        The present function takes the indices of all simplices that have been identified
        as being tie lines. It will then, for each of them, compute the best approximated
        tie line simplices['xtieline'], and identify the 'tiepeak' and the 'tiebase'.
        """
        nsel      = len(iselection)
        simplices = self.thesimplices[-1]
        if 'xtieline' not in simplices:
            simplices['xtieline'] = [None for _ in range(len(simplices['id']))]
        if 'tiepeak' not in simplices:
            simplices['tiepeak'] = [None for _ in range(len(simplices['id']))]
        if 'tiebase' not in simplices:
            simplices['tiebase'] = [None for _ in range(len(simplices['id']))]
        for isel,isim in enumerate(iselection):
            lmin = []
            for i,x in enumerate(simplices['x'][isim]):
                l = ((x[None,:]-simplices['x'][isim])**2).sum(axis=-1)
                l[l==0] = 1e99
                lmin.append(l.min())
            lmin  = np.array(lmin)
            ipeak = lmin.argmax()
            xpeak = simplices['x'][isim][ipeak]
            xav   = np.zeros(len(x))
            n     = 0
            for i,x in enumerate(simplices['x'][isim]):
                if i!=ipeak:
                    xav += x
                    n += 1
            xav /= n
            ibase = []
            for i,ipt in enumerate(simplices['ipts'][isim]):
                if i!=ipeak:
                    ibase.append(ipt)
            simplices['xtieline'][isim] = np.stack([xpeak,xav])
            simplices['tiepeak'][isim] = ipeak
            simplices['tiebase'][isim] = np.stack(ibase)

    def remove_false_tie_line_neighbors(self,iselection,stype):
        """
        Sometimes Qhull creates tie line simplices that are joined by their short side (the
        two liquid points for a ternary system), instead by their long sides. This is rare,
        but it can happen when two tie line fans touch at one location. This leads to the
        tie line grouping algorithm to falsely group both tie line fans into the same group.
        So we should try to identify these joins, and remove them from the tie line neighbor
        list (not from the full neighbor list).
        """
        simplices = self.thesimplices[-1]
        for isel,isim in enumerate(iselection):
            for k in simplices['neighbors_'+stype][isim]:
                if(not self.check_if_two_neighboring_tie_lines_are_parallel(isim,k)):
                    print(f'Removing false tie line neighbors {isim} {k}')
                    simplices['neighbors_'+stype][isim].remove(k)
                    if isim in simplices['neighbors_'+stype][k]:
                        simplices['neighbors_'+stype][k].remove(isim)

    def get_tie_lines_simplices(self,stypes=None):
        """
        From the convex hull result, identify all simplices that are, physically, tie lines.
        These are then gathered in the self.tie_line_simplices dictionary, where the keys are
        the various types of tie lines. By self.tie_line_simplices.keys() you can get a list of
        all present types.
        """
        if stypes is None:
            stypes = self.tieline_types
        self.tie_line_simplices = {}
        for stype in stypes:
            dummy = self.select_simplices_of_a_given_kind(stype)
            if len(dummy)>0:
                self.tie_line_simplices[stype] = dummy
                self.compute_tie_lines_from_simplices(self.tie_line_simplices[stype])
                self.remove_false_tie_line_neighbors(self.tie_line_simplices[stype],stype)

    def get_tie_line_groups(self):
        """
        Tie lines, at least those that connect to a liquid, usually come in bundles of nearly parallel
        ones. They can form multiple distinct groups. This function tries to identify all groups
        and properly identify the group members. This can be done because, in PhaseHull, these
        tie lines are, in fact, tie simplices with a non-zero width. The neighboring tie simplices
        will thus be exact neighbors, which can be identified as such. This is done with the
        function self.sort_tie_lines().
        """
        assert self.ncomponents==3, 'Error: get_tie_line_groups() only works for ternary diagrams.'
        self.tie_line_groups       = {}
        for stype in self.tie_line_simplices:
            self.tie_line_groups[stype] = []
            groups = self.sort_tie_lines(self.tie_line_simplices[stype],stype)
            for g in groups:
                if len(g)>=self.min_nr_tielines:
                    self.tie_line_groups[stype].append(g)

    def get_x_values_of_tie_lines(self,stypes=None,stride=1):
        if stypes is None:
            stypes = self.tieline_types
        assert self.ncomponents==3, 'Error: get_x_values_of_tie_lines() only works for ternary diagrams.'
        xx = {}
        for stype in stypes:
            if stype in self.tie_line_groups:
                xx[stype] = []
                for i,g in enumerate(self.tie_line_groups[stype]):
                    x = self.get_tie_lines_x_values_for_a_group(g,stride=stride)
                    xx[stype].append(x)
        return xx

    def get_x_values_of_binodal_curves(self,stypes=None,stride=1,fullx=False):
        """
        Binodal curves in ternary plots are the curves created from the endpoints of tie lines in, e.g.,
        a liquid with an inmiscibility region. This function can also get the endpoints of tie lines
        starting at fixed-composition points. This is then called the liquidus.

        Arguments:

           stypes     The simplex types for which to compute these endpoints. Default is the
                      types listed in self.tieline_types.

           stride     To reduce the output data volume, you can set stride to a value >1, meaning
                      that only every other `stride` point will be returned.

           fullx      If True, it will return not just the 2D coordinates of these points, but also
                      the third (dependent) coordinate which is 1-minus-the other two coordinates.
        """
        if stypes is None:
            stypes = self.tieline_types
        assert self.ncomponents==3, 'Error: get_x_values_of_binodal_curves() only works for ternary diagrams.'
        xb = {}
        for stype in stypes:
            if stype in self.tie_line_groups:
                xb[stype] = []
                for g in self.tie_line_groups[stype]:
                    xx = self.get_tie_lines_x_values_for_a_group(g,stride=stride)
                    xl = xx[:,0,0]
                    xr = xx[:,1,0][::-1]
                    yl = xx[:,0,1]
                    yr = xx[:,1,1][::-1]
                    x  = np.hstack([xl,xr,[xl[0]]])
                    y  = np.hstack([yl,yr,[yl[0]]])
                    if fullx:
                        z = 1-x-y
                        xb[stype].append(np.stack([x,y,z]).T)
                    else:
                        xb[stype].append(np.stack([x,y]).T)
        return xb

    def get_tie_lines_x_values_for_a_group(self,group,stride=1):
        """
        A group is a group of tie line simplices that belong together. In 2D ternary diagrams
        these tie lines typically form "fans" of near-parallel tie lines, sometimes diverging
        from a point (a crystal) but not always (if they are e.g. binodals of an inmiscibility
        region). This function returns the endpoints of these tie lines (both sides).
        
        Arguments:

           group      The group of tie lines that belong together (are adjacent to each other
                      in the phase diagram). 
        
           stride     To reduce the output data volume, you can set stride to a value >1, meaning
                      that only every other `stride` point will be returned.

        Returns:

           x          Array of x coordinates of the endpoints of this set of tie lines.
        """
        simplices = self.thesimplices[-1]
        x = []
        icnt = 0
        for i in group:
            isel = i
            if icnt==0:
                xtl = simplices['xtieline'][isel]
                assert xtl is not None, f'Error: Tie line for simplex {i} is None...'
                x.append(xtl)
            icnt += 1
            if icnt>=stride:
                icnt = 0
        x = np.stack(x)
        return x

    def get_ipts_liquidus_for_one_group_2d(self,stype,igroup):
        """
        Get the point indices (in the convex hull point cloud) of the points that
        lie on the liquidus, i.e., the end point of tie lines.

        Arguments:

           stype      The simplex type for which to compute this. For the liquidus
                      in a ternary (2D) plot belonging to (i.e. starting from) a
                      certain crystal (fixed composition solid), stype should be
                      'tieline_c1l2'.

           igroup     The index of the tie line group for which to collect these.

        Returns:

           ipts       The indices of the end points of these tie lines along the
                      liquidus.
        """
        simplices = self.thesimplices[-1]
        isims     = self.tie_line_groups[stype][igroup]
        ibases    = []
        for isim in isims:
            ibases.append(tuple(simplices['tiebase'][isim]))
        chains    = self.get_chains(ibases)
        iptss     = []
        for ch in chains:
            c    = np.stack(ch)
            ipts = list([c[0,0]])+list(c[:,1])
            iptss.append(ipts)
        return iptss

    def get_x_liquidus_for_one_group_2d(self,stype,igroup,smooth=False):
        """
        Get the point x locations (in the convex hull point cloud) of the points that
        lie on the liquidus, i.e., the end point of tie lines.

        Arguments:

           stype      The simplex type for which to compute this. For the liquidus
                      in a ternary (2D) plot belonging to (i.e. starting from) a
                      certain crystal (fixed composition solid), stype should be
                      'tieline_c1l2'.

           igroup     The index of the tie line group for which to collect these.

           smooth     Since the liquidus points are the points of the short edge
                      of the "needle-like" simplices, these can be a bit jagged,
                      like a saw. By setting smooth, this alternating back-and-forth
                      is averaged out.

        Returns:

           xx         Array of x-coordinates of the end points of these tie lines 
                      along the liquidus.
        """
        xpts  = self.thepoints[-1]
        iptss = self.get_ipts_liquidus_for_one_group_2d(stype,igroup)
        xx    = []
        for ipts in iptss:
            if smooth:
                x        = np.zeros((len(ipts)+1,self.ncomponents))
                x[0,:-1] = xpts[ipts[0]][:-1]
                for i in range(len(ipts)-1):
                    x[i+1,:-1] = 0.5 * ( xpts[ipts[i]][:-1] + xpts[ipts[i+1]][:-1] )
                x[-1,:-1] = xpts[ipts[-1]][:-1]
                x[:,-1] = 1-x[:,:-1].sum(axis=-1)
            else:
                x     = np.zeros((len(ipts),self.ncomponents))
                for i in range(len(ipts)):
                    x[i,:-1] = xpts[ipts[i]][:-1]
                x[:,-1] = 1-x[:,:-1].sum(axis=-1)
            xx.append(x)
        return xx

    def complete_x(self,x):
        """
        If x is (or maybe is) only the first n-1 elements where n is the number of components,
        then this function will return a new x that contains the complete set of components
        that sums up to 1. Note that x can either be one vector of an array of vectors.

        Arguments:

          x      The x vector x[0:ncomponents-1] or array of vectors x[0:nx,0:ncomponents-1]

        Returns:

          x      The x vector x[0:ncomponents] or array of vectors x[0:nx,0:ncomponents], such
                 that x.sum(axis=-1)==1.
        """
        if type(x) is list: x=np.array(x)
        if x.shape[-1]==self.ncomponents-1:
            if len(x.shape)==2:
                nx   = x.shape[0]
            elif len(x.shape)==1:
                nx   = 1
            else:
                raise ValueError('Error: Dimensions of x are somehow wrong. Must be a 1D array of x vectors.')
            xnew = np.zeros((nx,self.ncomponents))
            xnew[:,:-1] = x[:,:]
            xnew[:,-1]  = 1-x.sum(axis=-1)
            x = xnew
        return x

    def polygon_from_simplex(self,isim):
        """
        For plotting of the simplices in a ternary diagram you often need the top, left, right x
        values for a full closed polygon. This is returned here.
        """
        simplices = self.thesimplices[-1]
        x = simplices['x'][isim]
        x = np.vstack((x,x[0,:]))
        t = x[:,0]
        l = x[:,1]
        r = x[:,2]
        return t,l,r

    def get_liquidus_ipts_and_x_2d(self):
        """
        In a ternary diagram (2D) the liquidus is the closed curve at the edge of
        the domain(s) of the purely liquid phase. The edge is then usually either
        the edge of the diagram or the start/end of a set of tie lines.

        This function returns two results:

            liquidus_ipts    A list of lists of indices of the points along the
                             liquidus curve. The last one is equal to the first
                             one, to show that the curve is closed.
            liquidus_x       A list of lists of x[0:ncomponent] vectors, also here
                             the last in the list equals the first, making it easier
                             to plot a closed curve or polygon.

        Note that these are lists-of-lists, because there can be multiple disjoint
        regions of liquid phase. 
        """
        simplices = self.thesimplices[-1]
        isims     = self.select_simplices_of_a_given_kind('liquid')
        nface     = self.ncomponents-1
        liquidus_walls  = []
        ## liquidus_edges  = [] # Something for ternary liquidi
        for isim in isims:
            walls = []
            for i,ineigh in enumerate(simplices['neighbors'][isim]):
                cpts = list(set(simplices['ipts'][isim]).intersection(set(simplices['ipts'][ineigh])))
                assert len(cpts)==nface, f'Face has wrong nr of points {cpts}'
                cpts.sort()
                tpts = tuple(cpts)
                walls.append(tpts)
                if simplices['stype'][ineigh]!='liquid':
                    # Neighbor with a non-liquid simplex == liquidus wall
                    liquidus_walls.append(tpts)
            if len(simplices['neighbors'][isim])<self.ncomponents:
                # Some walls are at the edge of the domain
                dummy = []
                for i in range(self.ncomponents):
                    l = list(simplices['ipts'][isim]).copy()
                    l.remove(l[i])
                    l.sort()
                    dummy.append(tuple(l))
                walls = list(set(dummy)-set(walls))
                for w in walls:
                    liquidus_walls.append(w)
        self.liquidus_walls = liquidus_walls.copy()
        chains = self.get_chains(liquidus_walls)
        liquidus_x    = []
        liquidus_ipts = []
        xpts  = self.thepoints[-1][:,:-1]
        for c in chains:
            cn   = np.stack(c)
            ipts = np.hstack((cn[0,0],cn[:,1],cn[0,0]))
            x    = np.zeros((len(ipts),self.ncomponents))
            for i in range(len(ipts)):
                x[i,:-1] = xpts[ipts[i]]
            x[:,-1] = 1-x[:,:-1].sum(axis=-1)
            liquidus_x.append(x)
            liquidus_ipts.append(ipts)
        self.liquidus_ipts  = liquidus_ipts
        self.liquidus_x     = liquidus_x
        return liquidus_ipts,liquidus_x

    def define_a_relevelling_plane(self,components=None,useliq=False,x=None,G=None):
        """
        The Gibbs free energy landscape is generally on a "steep slope" because the components
        typically have very different Delta_f G. The Gibbs energies of the intermediate composition
        phases or phase-combinations are, typically, only slightly below the surface spanned by the
        components. This makes it hard to analyze. The self.define_a_relevelling_plane() allows you 
        to pre-define the function self.get_G_plane(x) that returns the G values of a plane going
        through the components (or through any set of len(components) pairs of x and G) so that
        you can (for plotting, for instance) subtract those values from the G values obtained
        from the PhaseHull algorithm. In this way, these new Gibbs energies are 0 at the components
        (or the set of X,G pairs given), and (likely/mostly) <0 between them, giving a much clearer
        view of the Gibbs free energy landscape within the simplex spanned by the components (or
        set of x,G pairs). Only for convenience.

        Arguments (either/or):

          components      If set, the components to use to define the plane.
          useliq          If set, use the components of the liquids, not those of the crystals
                          to define the plane.

          x, G            If set, the set of len(components) to use to define the plane.

        Returns:

          nothing, but you can now use self.get_G_plane(x) to get the values of G at points x
          on that plane, which you can subtract from the G values you get from the PhaseHull()
          algorithm.
        """
        if components is not None:
            assert x is None and G is None, 'Error: Cannot set both components and x,G'
            ncomp = len(components)
            x = np.zeros(ncomp)
            Gzero = np.zeros(ncomp)
            if useliq:
                liq = self.liquids[-1]
                for i in range(ncomp):
                    x[:] = 0
                    x[i] = 1.
                    Gzero[i] = liq.call_Gfunc(components,x[:])
            else:
                cryst = self.crystals[-1]
                db    = cryst.dbase
                icomponents,Gzero = self.identify_component_minerals(db,components)
                Gzero = np.array(Gzero)
            self.relevel_G = Gzero
        else:
            assert x is not None and G is not None, 'Error: You must set either components or x,G'
            from scipy import linalg
            x     = np.array(x)
            G     = np.array(G)
            ncomp = x.shape[-1]
            evec  = np.zeros((ncomp-1,ncomp-1))
            for i in range(ncomp-1):
                evec[:,i] = x[i,:-1]-x[-1,:-1]
            einv     = linalg.inv(evec)
            xx       = np.zeros([ncomp,ncomp])
            for i in range(ncomp):
                xx[i,i] = 1.
            y        = np.zeros([ncomp,ncomp])
            y[:,:-1] = (einv[None,:,:]*(xx[:,None,:-1]-x[-1,None,:-1])).sum(axis=-1)
            y[:,-1]  = 1-y[:,:-1].sum(axis=-1)
            Gzero    = np.zeros(ncomp)
            for i in range(ncomp):
                Gzero += y[:,i]*G[i]
            self.relevel_G = Gzero

    def G_relevelling_plane(self,x):
        """
        After calling define_a_relevelling_plane() you can use G_relevelling_plane(x) to obtain
        the G value of this plane at any point x. By subtracting this from the G values you obtain
        from the PhaseHull() algorithm, you get "re-levelled G values", where the components have
        G=0. 
        """
        if x.shape[-1] == len(self.components)-1:
            xx = x
            sh = list(xx.shape)
            sh[-1] += 1
            x  = np.zeros(sh)
            x[...,:-1] = xx[...,:]
            x[...,-1]  = 1-xx[...,:].sum(axis=-1)
        assert x.shape[-1] == len(self.components), 'Error: Dimension of x incorrect'
        if hasattr(self,'relevel_G'):
            if len(x.shape)==1:
                return (self.relevel_G*x).sum()
            else:
                return (self.relevel_G[...,:]*x).sum(axis=-1)
        else:
            return np.zeros(x.shape[:-1])

    # ------ Inner working stuff for PhaseHull -----

    def call_Gfunctions_liquids(self,components,x,return_Gliq=False):
        """
        PhaseHull can handle multiple continua, for instance liquid and solid metal alloy. When these functions
        are called, the lowest of the continua is taken at any point x. Also the id-number of this continuum
        is then chosen for that point. Returned is G and the id number of the continuum, for every x.

        Argument:

          components   List of names of the components in the correct order
                       in which they are in x.

          x            The fractional abundances. x can be a 2D array x[N,M].

        Returns:

          G            The (lowest of) the G value(s) at each x of the liquids or alloys.

          idlm         The identification number of which liquid has lowest G at that x.
                       These id numbers start with -1 (for the first liquid in the self.liquids)
                       and -2 (for the second liquid), etc.
        """
        assert len(self.liquids)>0, 'Error: No continua (liquids, alloys or glasses) available.'
        G    = []
        idl  = []

        # Loop over all liquids (usually just one, and note that 'liquid' here means
        # a continuum, and can therefore also be a non-fixed composition solid, such
        # as an alloy).

        x    = np.stack(x)
        if len(x.shape)==2:
            nx   = x.shape[0]
        else:
            nx   = 1
        nliq = len(self.liquids)
        for iliq in range(nliq):
            gib = self.liquids[iliq].call_Gfunc(components,x)      # Call the function that computes G
            gib = np.reshape(gib,[nx])
            G.append(gib)
            idd = np.reshape(-iliq-1,[nx])
            idl.append(idd)                                        # Store the ID of this liquid
        if return_Gliq: Gliq = G
        G    = np.stack(G).T
        imin = G.argmin(axis=-1)                 # Determine which of the liquids has the lowest G
        Gmin = np.zeros(len(G))                  # Since x can be an array of points
        idlm = np.zeros(len(G),dtype=int)        # we find and identify the lowest G at each point
        for i in range(len(G)):
            im      = imin[i]
            Gmin[i] = G[i,im]
            idlm[i] = idl[im]
        if len(x.shape)==1:
            Gmin = Gmin[0]
            idlm = idlm[0]
        if return_Gliq:
            return Gmin,idlm,Gliq
        else:
            return Gmin,idlm

    def create_points_for_liquids_or_solid_solutions(self,xgrid,Ggrid,idgrid):
        """
        From a set of points xgrid with a G value at each point Ggrid,
        construct the points we shall use for the convex hull algorithm. This is
        used for setting up a grid of points to map continuous solutions (i.e., 
        liquids or solid solutions). These have no fixed stoichiometry, hence
        they have a Gibbs energy at all (or a continuous subspace) of the
        x phase space.
        """
        pts = np.zeros((len(Ggrid),self.ncomponents))
        ids = [None for _ in range(len(Ggrid))]
        for i in range(len(Ggrid)):
            pts[i,:-1] = xgrid[i][:-1].copy()    # For the convex hull we only need the independent x values, hence [:-1]
            pts[i,-1]  = Ggrid[i]                # The last value of the point coordinates is the G value
            ids[i]     = idgrid[i]               # The ids tell which liquid this point belongs to, if multiple continua are used.
        pts = np.stack(pts)
        return pts,ids

    def create_points_for_crystals(self):
        """
        Crystals (in PhaseHull) are fixed-composition points in x space.
        They are discrete points, each of which belongs to an entry in
        the mineral database of self.crystals. This function creates
        the corresponding points for the convex hull algorithm.

        Note that we use mfDfG, not DfG here. The reason is that from one
        mole of components, you do not always get 1 mole of crystal. Example
        from half a mole of MgO and half a mole of SiO2 (in total these
        comprise 1 mole of components) you get only half a mole of MgSiO3.
        So we should use 0.5 times the Gibbs energy of MgSiO3. This is
        what mfDfG is, in relation to DfG. 
        """
        pts   = []
        ids   = []
        ncomp = len(self.components)

        # Not always we have such fixed-composition minerals
        
        if len(self.crystals)>0:

            # Get the crystal database

            assert len(self.crystals)==1, 'Error: At the moment, only one crystal database is allowed.'
            mdb       = self.crystals[0].dbase

            assert(len(mdb)>0), 'Error: Crystal db empty'

            # Remove all minerals that are either out of their temperature or pressure range
            # or have clearly unphysical value of DfG (a way to signal PhaseHull not to include
            # this mineral).

            mdb       = self.remove_out_of_TP_range_minerals_from_db(mdb)
            mdb       = self.remove_invalid_G_minerals_from_db(mdb)

            # At the component locations only 1 crystal is allowed, so pick the lowest energy one

            mdb       = self.remove_non_lowest_components_from_db(mdb)

            # Loop over all remaining crystals
            
            for i,row in mdb.iterrows():
                pt    = np.zeros(ncomp)
                pt[:-1] = row['x'][:-1]   # For the convex hull we only need the independent x values, hence [:-1]
                pt[-1]  = row['mfDfG']    # The last value of the point coordinates is the G value, weighted by moles
                pts.append(pt)
                name    = row['Abbrev']   # Add the label of which crystal this is to this point, for later reference.
                ids.append(name)
        return pts,ids

    def remove_out_of_TP_range_minerals_from_db(self,mdb):
        """
        Some rows of the database may have a restricted validity range in
        temperature and/or pressure. These are given by the 'Tmin', 'Tmax',
        'Pmin' and 'Pmax' (in Kelvin resp bar) columns of the database.

        NOTE: This is done only for the fixed-composition crystals, not for
              the endmembers of liquid or solid solutions, because the
              Gibbs functions for those have to be provided as Python
              functions by the input model. So the input model has to
              take care of these ranges.
        """
        mdb   = mdb.reset_index(drop=True)
        T     = self.T
        P     = self.P
        if 'Tmin' in mdb.columns and T is not None: mdb = mdb[T>=mdb['Tmin']]
        if 'Tmax' in mdb.columns and T is not None: mdb = mdb[T< mdb['Tmax']]
        if 'Pmin' in mdb.columns and P is not None: mdb = mdb[P>=mdb['Pmin']]
        if 'Pmax' in mdb.columns and P is not None: mdb = mdb[P< mdb['Pmax']]
        return mdb

    def check_mineral_within_TP_range(self,dbrow):
        T     = self.T
        P     = self.P
        check = True
        if 'Tmin' in dbrow and T is not None:
            if T< dbrow['Tmin']: check = False
        if 'Tmax' in dbrow and T is not None:
            if T>=dbrow['Tmax']: check = False
        if 'Pmin' in dbrow and P is not None:
            if P< dbrow['Pmin']: check = False
        if 'Pmax' in dbrow and P is not None:
            if P>=dbrow['Pmax']: check = False
        return check

    def remove_invalid_G_minerals_from_db(self,mdb):
        """
        One way to signal to PhaseHull that a mineral is somehow invalid and should
        not be included is to put its value of 'DfG' to a value >=1e40.
        """
        mdb = mdb[mdb['DfG']<1e40]
        return mdb

    def remove_non_lowest_components_from_db(self,mdb):
        """
        Sometimes, if the crystal database contains multiple crystals at (1,0,0)
        or another corner point, then Qhull finds simplices connecting both
        (or more) points that are all at the same x. This function eliminates
        any multiplicity of crystals at the same corner by choosing only the
        one with the lowest energy.
        """
        mdb   = mdb.reset_index(drop=True)
        iincl = list(np.arange(len(mdb)))
        for k in range(self.ncomponents):
            ii = []
            GG = []
            for i,m in mdb.iterrows():
                if m['x'][k]==1:
                    ii.append(i)
                    GG.append(m['mfDfG'])
            #assert len(ii)>0, f'No Component found in direction {k}'
            if len(ii)>0:
                ikeep = ii[np.argmin(GG)]
                for i in ii:
                    if i!=ikeep:
                        iincl.remove(i)
        mdb = mdb.iloc[iincl].reset_index(drop=True)
        return mdb

    def setup_all_points_before_refinement(self):
        """
        For liquids or non-fixed-composition solids (alloys, glasses) we need a grid
        of x values. This grid can be refined if necessary, but first we set up a
        regular grid with self.nres0 grid points along each axis.

        Because we explicitly envision the use of recursive grid refinement, we
        allow for successive layers of refinement, while keeping the data of the
        coarser refinement levels in memory. The data for these grids are stored
        in a series of arrays, with names all starting with "the". The most important
        ones are:

            self.thepoints      The points used for the convex hull
            self.theids         The identifications of the points used for the convex hull

        The highest resolution grid is always the last. So the set of points of
        the base grid are self.thepoints[0], while the set of points of the
        most-refined grid are self.thepoints[-1] (where -1 stands for last).

        NOTE: Certain solid solutions cannot be modelled with the Liquid class, because
              they have a restricted subspace within the full system. Example: the
              Ca(1-x)Mg(x)SiO4 crystal binary family in the CaO-MgO-SiO2 ternary system.
              These have to be modelled using another class: The SolidSolution class.
              HOWEVER: If you can model your solid solution with the Liquid class, it is
              better. The solid solution must then exist for all x points in the
              system. 
        """
        if self.verbose: print('Starting Computation at base grid level...')

        # For each level of refinement, all data is stored in self.the<something>.
        # To get the highest-resolution data, always choose self.the<something>[-1]
        self.thepoints       = []  # The points used for the convex hull
        self.theids          = []  # The identifications of the points used for the convex hull
        self.thepoints_liq   = []  # Only for convenience
        self.theids_liq      = []  # Only for convenience
        self.thepoints_solsol= []  # Only for convenience
        self.theids_solsol   = []  # Only for convenience
        self.thepoints_cryst = []  # Only for convenience
        self.theids_cryst    = []  # Only for convenience
        if(len(self.solsols)>0):
            self.theyvals    = []

        # The liquids
        pts_liq = None
        ids_liq = None
        if(len(self.liquids)>0):
            nres              = self.nres0
            self.gridsnres    = [nres]
            self.igrids       = []
            self.xgrids       = []
            if self.user_xgrid is None:
                # Normal case: We make the base grid for the liquids (the solutions) here
                igrid             = make_integer_grid(nres,self.ncomponents)  # Call the actual grid builder
                self.igrids.append(igrid)
                xgrid             = igrid/nres
                self.xgrids.append(xgrid)
            else:
                # Special case: The user has his/her own base grid for the liquids (the solutions)
                xgrid         = self.user_xgrid
                igrid         = xgrid*nres
                # Check if this igrid is indeed integer
                ipgrid        = (igrid+1e-6).astype(int)
                imgrid        = (igrid+1-1e-6).astype(int)
                assert np.all(ipgrid==imgrid), f'ERROR: The user-specified xgrid is not based on an integer grid with nres = {nres}.'
                self.igrids.append(igrid)
                self.xgrids.append(xgrid)
            Ggrid             = []
            idgrid            = []
            for i,liq in enumerate(self.liquids):
                liq.xgrid     = xgrid
                liq.Ggrid     = np.zeros(len(xgrid))
            for ix,x in enumerate(xgrid):
                G,idd,Gl       = self.call_Gfunctions_liquids(self.components,x,return_Gliq=True)
                Ggrid.append(G)
                idgrid.append(idd)
                for i,liq in enumerate(self.liquids):
                    liq.Ggrid[ix] = Gl[i]
            pts_liq,ids_liq = self.create_points_for_liquids_or_solid_solutions(xgrid,Ggrid,idgrid)
            self.thepoints_liq.append(pts_liq)
            self.theids_liq.append(ids_liq)

        # The solid solution crystal families. Note: only recommended if this family has
        # a lower dimension than the nr of components. Otherwise: better use an instance
        # of the Liquid class and add to self.liquids, as that class has more features
        # such as grid refinement and more. The solid solution class is a rather primitive
        # class to add a grid of points independent of the liquid grid, and typically a
        # subspace of the full x-space. Example: We have CaO,MgO,SiO2 ternary space, but 
        # we want to add the Ca(1-y)Mg(y)MgSi2O6 binary solid solution (the Enstatite-
        # Diopside join) inside it.
        pts_solsol = None
        ids_solsol = None
        yvs_solsol = None
        if(len(self.solsols)>0):
            pts_solsol = []
            ids_solsol = []
            yvs_solsol = []
            for solsol in self.solsols:
                assert hasattr(solsol,'ygrid') and hasattr(solsol,'xgrid'), f'Error: The internal grid for the solid solution {solsol.name} has not yet been set up properly.'
                Ggrid  = solsol.call_Gfunc()
                solsol.Ggrid = Ggrid
                idgrid = [f'+{solsol.name}' for i in range(len(solsol.ygrid))]
                pts_ss,ids_ss = self.create_points_for_liquids_or_solid_solutions(solsol.xgrid,Ggrid,idgrid)
                pts_solsol += list(pts_ss)
                ids_solsol += list(ids_ss)
                yvs_solsol += list(solsol.ygrid)
            self.thepoints_solsol.append(pts_solsol)
            self.theids_solsol.append(ids_solsol)

        # The fixed stoichiometry crystals
        pts_cryst = None
        ids_cryst = None
        if(len(self.crystals)>0):
            pts_cryst,ids_cryst = self.create_points_for_crystals()
            self.thepoints_cryst.append(np.stack(pts_cryst))
            self.theids_cryst.append(ids_cryst)

        # Combine the points
        pts = []
        ids = []
        if(len(self.solsols)>0): yvs = []
        if pts_liq is not None:
            pts += list(pts_liq)
            ids += list(ids_liq)
            if(len(self.solsols)>0): yvs += [None for _ in range(len(pts_liq))]
        if pts_solsol is not None:
            pts += list(pts_solsol)
            ids += list(ids_solsol)
            yvs += list(yvs_solsol)
        if pts_cryst is not None:
            pts += list(pts_cryst)
            ids += list(ids_cryst)
            if(len(self.solsols)>0): yvs += [None for _ in range(len(pts_cryst))]
        pts = np.stack(pts)
        self.thepoints.append(pts)
        self.theids.append(ids)
        if(len(self.solsols)>0):
            self.theyvals.append(yvs)

    def do_convex_hull_algorithm(self,igrid):
        """
        The main routine of PhaseHull: this will call the convex hull algorithm
        for the computation of the phase diagram. The resulting set of simplices
        of the convex hull will then be analyzed: First, only the simplices at the
        bottom of the hull will be selected. Then each of the remaining simplices
        will be classified: is this simplex an element of a liquid, or is it a
        tie line, or is it a coexistence of crystals, or an inmiscibility simplex?

        The results are all listed in a dictionary called "simplices", which is then
        appended to the self.thesimplices list. This means in self.thesimplices[0]
        you have the results of the convex hull algorithm for the lowest-resolution
        (base grid) computation, while (if applicable) self.thesimplices[1] is the
        result of the next-higher refined grid, all the way to self.thesimplices[-1]
        which is the result of the highest refinement level.

        The simplices dictionary consists of a set of simple arrays, the index of
        which is the index of the simplex. So if the convex hull algorithm found
        230 simplices at the basegrid, then simplices=self.thesimplices[0] is a
        dictionary of arrays, each of which has length 230. By checkout out
        simplex.keys() you can see which arrays are available. The most important
        are:

          simplices['stype']    The physical meaning of the simplices. E.g. if
                                simplices['stype'][5]=='cryst_1_liq_1', then simplex
                                number 5 represents a coexistence of one crystals
                                with one liquid.
          simplices['ipts']     The indices of the points at its corners.
          simplices['x']        The coordinate of the corner points.
          simplices['G']        The G values at the corner points.
        """
        pts       = self.thepoints[igrid]
        idlist    = self.theids[igrid]
        answer    = self.find_lower_convex_hull_of_x_G_points(pts)
        hull      = answer['hull']
        self.thehulls.append(hull)
        simplices = self.classify_simplices(pts,idlist,hull)
        self.thesimplices.append(simplices)

    def do_one_refinement_step(self,refinepoints=None):
        """
        To correctly reproduce delicate details of the phase diagram it is
        usually important to have very high grid resolution close to certain
        special locations, e.g. close to the liquidus or the binodal curve.
        But to avoid wasting computing power on uninteresting regions, we
        employ adaptive grid refinement.

        This function takes the results of the previous refinement step,
        figures out where further refinement can be beneficial, adds gridpoints
        there, and then redoes the convex hull computation.

        Arguments:

          refinepoints    If None, then refine near the binodals and liquidi.
                          If a dict, then instead refine around those points
                          defined by refinepoints['x'], which should be a 2D
                          array of x[npoints,ncomp], which are the points
                          around which to refine, ['nfact'] is the factor of 
                          refinement, ['nspan'] is the span of the refinement,
                          ['niter'] is the nr of recursive refinements.
        """
        if self.verbose: print('Starting Computation at this refinement level...')
        assert len(self.liquids)>0, 'Error: No refinement necessary if no liquids/alloys/glasses available'
        assert len(self.thepoints)==len(self.thesimplices), f'First call do_convex_hull_algorithm({len(self.thepoints)-1})'
        igrid     = len(self.gridsnres)
        nres      = self.gridsnres[-1]

        # The liquids
        pts_liq   = self.thepoints_liq[igrid-1]
        ids_liq   = self.theids_liq[igrid-1]
        simplices = self.thesimplices[igrid-1]
        if refinepoints is None:
            nresnew   = nres*self.nfact
            print(f'Iteration {igrid} at dx = 1/{nresnew}')
            self.gridsnres.append(nresnew)
            ptsnew_liq,idsnew_liq = self.refine_near_binodals_2d(pts_liq,simplices,nres+1,nfact=self.nfact,nspan=self.nspan)
        else:
            nresnew   = nres*refinepoints['nfact']**refinepoints['niter']
            print(f'Refining grid at dx = 1/{nresnew}')
            self.gridsnres.append(nresnew)
            ptsnew_liq,idsnew_liq = self.refine_near_a_point_2d(pts_liq,simplices,nres+1,refinepoints['x'],
                                                                nfact=refinepoints['nfact'],
                                                                nspan=refinepoints['nspan'],
                                                                niter=refinepoints['niter'])
        self.thepoints_liq.append(ptsnew_liq)
        self.theids_liq.append(idsnew_liq)

        # The solid solutions (just copy)
        ptsnew_solsol = None
        if(len(self.solsols)>0):
            ptsnew_solsol = self.thepoints_solsol[igrid-1]
            idsnew_solsol = self.theids_solsol[igrid-1]
            self.thepoints_solsol.append(ptsnew_solsol)
            self.theids_solsol.append(idsnew_solsol)
        
        # The fixed stoichiometry crystals (just copy)
        ptsnew_cryst = None
        if(len(self.crystals)>0):
            ptsnew_cryst = self.thepoints_cryst[igrid-1]
            idsnew_cryst = self.theids_cryst[igrid-1]
            self.thepoints_cryst.append(np.stack(ptsnew_cryst))
            self.theids_cryst.append(idsnew_cryst)

        # Combine the points
        pts = []
        ids = []
        if ptsnew_liq is not None:
            pts += list(ptsnew_liq)
            ids += list(idsnew_liq)
        if ptsnew_solsol is not None:
            pts += list(ptsnew_solsol)
            ids += list(idsnew_solsol)
        if ptsnew_cryst is not None:
            pts += list(ptsnew_cryst)
            ids += list(idsnew_cryst)
        pts = np.stack(pts)
        self.thepoints.append(pts)
        self.theids.append(ids)

        # Call the convex hull algorithm
        self.do_convex_hull_algorithm(igrid)

    def do_all_refinement_steps(self):
        #print(f'First low resolution calculation at dx = 1/{self.nres0}')
        self.do_convex_hull_algorithm(0)
        if self.refinepoints is not None:
            self.do_one_refinement_step(refinepoints=self.refinepoints)
        if (len(self.liquids)>0) and len(self.components)>2:
            for iter in range(self.nrefine):
                self.do_one_refinement_step()

    def redo_computation_with_selected_phases(self,incl_liq=True,incl_solsol=False,incl_cryst=False):
        assert len(self.thepoints)==len(self.thesimplices), f'First call do_convex_hull_algorithm({len(self.thepoints)-1})'
        igrid     = len(self.gridsnres)
        nres      = self.gridsnres[-1]

        # The liquids
        ptsnew_liq   = None
        if(incl_liq):
            ptsnew_liq   = self.thepoints_liq[igrid-1]
            idsnew_liq   = self.theids_liq[igrid-1]
            self.thepoints_liq.append(ptsnew_liq)
            self.theids_liq.append(idsnew_liq)

        # The solid solutions (just copy)
        ptsnew_solsol = None
        if(incl_solsol):
            if(len(self.solsols)>0):
                ptsnew_solsol = self.thepoints_solsol[igrid-1]
                idsnew_solsol = self.theids_solsol[igrid-1]
                self.thepoints_solsol.append(ptsnew_solsol)
                self.theids_solsol.append(idsnew_solsol)
        
        # The fixed stoichiometry crystals (just copy)
        ptsnew_cryst = None
        if(incl_cryst):
            if(len(self.crystals)>0):
                ptsnew_cryst = self.thepoints_cryst[igrid-1]
                idsnew_cryst = self.theids_cryst[igrid-1]
                self.thepoints_cryst.append(np.stack(ptsnew_cryst))
                self.theids_cryst.append(idsnew_cryst)

        # Combine the points
        pts = []
        ids = []
        if ptsnew_liq is not None:
            pts += list(ptsnew_liq)
            ids += list(idsnew_liq)
        if ptsnew_solsol is not None:
            pts += list(ptsnew_solsol)
            ids += list(idsnew_solsol)
        if ptsnew_cryst is not None:
            pts += list(ptsnew_cryst)
            ids += list(idsnew_cryst)
        pts = np.stack(pts)
        self.thepoints.append(pts)
        self.theids.append(ids)

        # Call the convex hull algorithm
        self.do_convex_hull_algorithm(igrid)

    def find_lower_convex_hull_of_x_G_points(self,pts):
        """
        Find the convex hull of all the points in N-1 x-space dimensions +
        1 mfDfG space dimension. We are interested the lower facets (simplices),
        as they represent, for each x, the lowest Gibbs energy mfDfG. The
        convex hull, however, also has an upper facet (or multiple upper
        facets). They can be recognized by checking the last (=mfDfG) index
        of the facet normal vectors. We remove these, so that only the lowest
        energy facet for each x position is left: the bottom of the hull.
    
        Arguments:
    
          pts       The points in [x[:-1],mfDfG]-space. Pts is an array
                    with dimensions [npoints,N], where N is the number of
                    components. But of these N elements, only N-1 are x
                    values, because the last x can be found from the
                    sum to unity condition: x[-1]=1-x[:-1].sum(). This
                    frees up space for the Gibbs free energy as the
                    last value. So x = pts[:,:-1] while mfDfG = pts[:,-1].
    
        Returns:
    
          answer    A dictionary containing:
    
                     simplices    Like ConvexHull.simplices, but only
                                  the ones of the bottom of the hull.
                     equations    Like ConvexHull.equations, but only
                                  the ones of the bottom of the hull.
                     vertices     Like ConvexHull.vertices
                     hull         The original ConvexHull instance.
    
        Note: Since the simplices and equations are a subset of the
              hull.simplices and hull.equations, the indices of them
              are not to be used with hull.neighbors. The indices in
              hull.neighbors refer to hull.simplice instead.
        """
        points    = np.stack(pts)
        if self.verbose: print('Starting ConvexHull...')
        hull      = ConvexHull(points)
        if self.verbose: print('Done with ConvexHull...')
        include   = hull.equations[:,-2]<-self.epsequ
        simplices = hull.simplices[include,:]
        equations = hull.equations[include,:]
        vertices  = hull.vertices
        answer    = {}
        answer['simplices'] = simplices
        answer['equations'] = equations
        answer['vertices']  = vertices
        answer['hull']      = hull
        return answer

    def classify_simplices(self,pts,idlist,hull,mrcrit=None):
        """
        Interpret the output of the convex hull computation in terms
        of the meaning of each simplex. Each simplex will get a label
        (the 'stype' key of the dict) specifying the physical meaning.
        For now only physical meanings for up to ternary diagrams are
        included, but this is easily extendable to quaternary diagrams
        and higher. Types:
    
           liquid            Fully liquid
           allcryst          A coexistence of 2 (for binary) or 3 (for 
                             ternary) (etc) crystals.
           cryst_1_liq_1     A coexistence of 1 crystal phase with
                             1 liquid phase (only for binary diagrams).
           cryst_2_liq_1     A coexistence of 2 crystal phases with
                             1 liquid phase (only for ternary diagrams).
           cryst_1_liq_2     A coexistence of 1 crystal phase with
                             2 liquid phases (only for ternary diagrams),
                             but not a tie line.
           tieline_c1l2      Like cryst_1_liq_2, but with the physical
                             meaning of a tie line between 1 crystal and
                             1 liquid point (only for ternary diagrams).
                             This is a very elongated simplex. Its
                             meaning is that of a binary phase line 
                             in a ternary diagram. The two liquid points
                             are actually a single liquid, but it becomes 
                             two very-nearby liquid points due to the
                             discretization.
           tieline_c0l3      A coexistence of two liquid phases that
                             are inmiscible  (only for ternary diagrams).
                             In reality 3 liquids are connected, two
                             being very close together.
           inmisc_liquids_3phase   Like 'inmisc_liquids' but for
                                   three phases. In a ternary diagram
                                   these occasionally show up as big
                                   single triangles.
    
        Arguments:
    
          pts       The points in [x[:-1],mfDfG]-space. Pts is an array
                    with dimensions [npoints,N], where N is the number of
                    components. But of these N elements, only N-1 are x
                    values, because the last x can be found from the
                    sum to unity condition: x[-1]=1-x[:-1].sum(). This
                    frees up space for the Gibbs free energy as the
                    last value. So x = pts[:,:-1] while mfDfG = pts[:,-1].
    
          idlist    List of the same length as pt with the identity of
                    each point. For instance, if you have 3 solid
                    crystals (say 'SiO2','MgO', and 'MgSiO3'), and
                    200 liquid points, then the idlist would look like
                    this: ['SiO2','MgO', 'MgSiO3', -1, -1, ... , -1]
                    where -1 has the meaning of the liquid.
    
          hull      The object in the answer['hull'] from the function
                    find_lower_convex_hull_of_x_G_points(pts)
    
        Returns:
    
          simplices     A list of dicts with information for each simplex.

        IMPORTANT NOTE:

          So far (16 Sep 2025) only up to ternary phase diagrams have been thorougly
          tested. Not sure if all simplex types are complete for higher order
          systems. This may need some additional work.
        """
        if mrcrit is None: mrcrit=self.mrcrit
        if hasattr(self,'gridsnres'):
            nx = self.gridsnres[-1]
        pts   = np.stack(pts)
        npts  = len(pts)
        ncomp = pts.shape[-1]
        simplices = {'id':[],'id_qhull':[],'stype':[],'ipts':[],'neighbors_qhull':[]}
        if self.incl_ptnames:
            simplices['ptnames'] = []
        if self.incl_xvals:
            simplices['x'] = []
        if self.incl_Gvals:
            simplices['G'] = []
        if self.incl_xcen:
            simplices['xcen'] = []
        if self.incl_Gcen:
            simplices['Gcen'] = []
        if self.incl_xtie:
            simplices['xtie'] = []
        isimnew = 0
        simindices = {}

        # Loop over all simplices of the convex hull

        for isim,simplex in enumerate(hull.simplices):

            # But select only the simplices who's normal
            # vector point downward (bottom of the convex
            # hull)

            if(hull.equations[isim,-2]<-self.epsequ):
                names    = []
                x        = []
                G        = []
                allcryst = True
                allliq   = True
                allsol   = True

                # Loop over all corner points of this simplex
                # and store their x coordinates and G values,
                # as well as their IDs/names. If the name of a points
                # is an integer <0, then the point represents
                # a liquid (or continuous solid) point. If the
                # name is a string, then it represents a fixed
                # composition solid (which we call crystal in
                # PhaseHull). 

                for ipt in simplex:
                    name = idlist[ipt]
                    names.append(name)
                    x.append(pts[ipt,:-1])
                    G.append(pts[ipt,-1])
                    #if type(name) is int: allcryst = False
                    #if type(name) is str:
                    #    allcont  = False
                    #else:
                    #    allcryst = False
                    sname = str(name).strip()
                    if sname[0]=='-' or sname[0]=='+':
                        allcryst = False
                        if sname[0]=='+':
                            allliq   = False
                        if sname[0]=='-':
                            allsol   = False
                    else:
                        allliq   = False
                        allsol   = False
                assert not (allcryst and allliq), 'Error: Inconsistency in crystal/liquid.'
                assert not (allcryst and allsol), 'Error: Inconsistency in crystal/solidsolution.'
                assert not (allliq and allsol), 'Error: Inconsistency in liquid/solidsolution.'

                # Since the x values of these points are only
                # the independent x values (one fewer than the
                # number of components), we reconstruct the full
                # x values using x[last] = 1 - sum x[all but last].

                xx        = np.zeros((len(x),len(x[0])+1))
                xx[:,:-1] = x
                xx[:,-1]  = 1-xx.sum(axis=-1)

                # Store the current information in arrays 
                
                simplices['id'].append(isimnew)
                simplices['id_qhull'].append(isim)
                simplices['ipts'].append(simplex)
                if self.incl_ptnames: simplices['ptnames'].append(names)
                if self.incl_xvals:   simplices['x'].append(xx)
                if self.incl_Gvals:   simplices['G'].append(G)

                # Now classify the simplex: Is it a pure liquid (or continuous
                # solid/alloy, which is equivalent in PhaseHull) or is it a coexistence
                # of crystals (fixed composition points in the phase diagram)?
                # This is the stype of this simplex.
                
                if allcryst:
                    # All corners of this simplex are crystal solids
                    # A tie simplex
                    simplices['stype'].append('allcryst')
                elif allliq:
                    # All corners of this simplex are liquid/alloy
                    if nx is not None:
                        inmisc,x,l,mr = self._test_if_simplex_is_inmiscible_liquids(pts,isimnew,simplices['ipts'],nx=nx,return_all=True)
                    else:
                        inmisc        = self._test_if_simplex_is_inmiscible_liquids(pts,isimnew,simplices['ipts'],nx=nx,return_all=False)
                        mr            = None
                    if(inmisc):                                  # If we are in an inmiscibility part of the liquid, 
                        nliq  = self.ncomponents                 # then do some more work.
                        stype = f'tieline_c0l{nliq}'             # Normally if a liquid has inmiscibility, the simplex is a tie line
                        if mr is not None and self.ncomponents==3:
                            if mr>mrcrit:                        # In a ternary one can also have inmiscibility of three compositions.
                                stype = 'inmisc_liquids_3phase'  # This is a bit tricky to find: we use the shape of the simplex.
                        # Check if this simplex connects different 'liquids' (if you have more than 1 continuum)
                        if len(set(names))>1:
                            stype += '_crossliq'
                    else:
                        stype = 'liquid'                         # If not inmiscible, then this is a normal liquid part
                    simplices['stype'].append(stype)
                elif allsol:
                    # All corners of this simplex are a solid solution
                    # NOTE: We do not check for inmiscibility here. This would be more
                    #       complicated on the lower-dimensional subspaces spanned by
                    #       the SolidSolution class. For now, it is not worth the
                    #       trouble. If you want to study the solid solution in more
                    #       detail, then reduce the dimensionality of the problem to
                    #       only this subspace, and then replace the SolidSolution
                    #       object with a Liquid object, which has much more flexibility
                    #       and functionality.
                    if len(set(names))>1:
                        stype = 'solsol_coexist'
                    else:
                        stype = 'solsol'
                        inmis = self._test_if_simplex_is_inmiscible_solid_solutions(pts,isimnew,simplices['ipts'],names)
                        inmis = inmis[names[0][1:]]   # We expect only a single inmiscibility, because we have only 1 solution now
                        if inmis:
                            stype = 'solsol_inmisc'
                    simplices['stype'].append(stype)
                else:
                    # Some corners may be liquid, some may be crystal solids, some may be solid solutions
                    nliq      = 0   # Nr of liquid points
                    ncryst    = 0   # Nr of fixed-composition crystal points
                    nsolsol   = 0   # Nr of solid solution points
                    crystals  = []
                    liquids   = []
                    solsols   = []
                    for i in range(len(simplex)):
                        if type(names[i]) is str:
                            if names[i][0]=='+':
                                nsolsol   += 1   # Solid solution names start with a +
                                solsols.append(names[i][1:])
                            else:
                                assert names[i][0]!='-', 'Liquid point should be negative integer, not string with a - sign.'
                                ncryst += 1   # Fixed-composition names are all strings not starting with a +
                                crystals.append(names[i])
                        else:
                            nliq   += 1       # Liquid solution names are negative integers
                            liquids.append(names[i])
                    if self.ncomponents>2:
                        # Ternary or higher-dimensional
                        if self._test_if_simplex_is_tie_line(pts,isimnew,simplices['ipts'],crystals,liquids,solsols):
                            s = f'tieline_c{ncryst}l{nliq}'
                            if nsolsol>0:
                                s += f's{nsolsol}'
                        else:
                            s = f'cryst_{ncryst}_liq_{nliq}'
                            if nsolsol>0:
                                s += f'_sol_{nsolsol}'
                        simplices['stype'].append(s)
                    else:
                        # Binary
                        s = f'tieline_c{ncryst}l{nliq}'
                        if nsolsol>0:
                            s += f's{nsolsol}'
                        simplices['stype'].append(s)
                simplices['neighbors_qhull'].append(hull.neighbors[isim].copy())
                simindices[isim] = isimnew
                isimnew += 1

        simplices['neighbors'] = []
        for isimnew in range(len(simplices['stype'])):
            idxnew = []
            for i in range(len(simplices['neighbors_qhull'][isimnew])):
                k = simplices['neighbors_qhull'][isimnew][i]
                if k in simindices:
                    idxnew.append(simindices[k])
            simplices['neighbors'].append(idxnew)

        # Convert all lists to numpy arrays for easier handling
        simplices['id']              = np.array(simplices['id'])
        simplices['id_qhull']        = np.array(simplices['id_qhull'])
        simplices['ipts']            = np.array(simplices['ipts'])
        simplices['stype']           = np.array(simplices['stype'])
        #simplices['neighbors']       = np.array(simplices['neighbors'])
        #simplices['neighbors_qhull'] = np.array(simplices['neighbors_qhull'])
        if self.incl_ptnames:  simplices['ptnames'] = np.array(simplices['ptnames'])
        if self.incl_xvals:    simplices['x']       = np.array(simplices['x'])
        if self.incl_Gvals:    simplices['G']       = np.array(simplices['G'])
        if self.incl_xtie:     simplices['xtie']    = np.array(simplices['xtie'])
        if self.incl_xcen:     simplices['xcen']    = np.array(simplices['xcen'])
        if self.incl_Gcen:     simplices['Gcen']    = np.array(simplices['Gcen'])
        return simplices

    def _test_if_simplex_is_inmiscible_liquids(self,pts,isim,sim_ipts,nx=None,return_all=False):
        # Inmiscible liquids occur when the G-surface is not convex,
        # but instead has a "wiggle". If the present simplex lies (apart
        # from its corner points) below the true liquid G function, then
        # this simplex is a tie line or tie simplex of inmiscible liquids.
        # This is what is tested here.
        N   = len(pts[0])                                          # Nr of components 
        npt = len(pts)                                             # Nr of points available (not all are part of the convex hull)
        x   = np.zeros((N+1,N))                                    # Molar fractions of the corner points of this simplex
        for i in range(N):
            x[i,:-1] = pts[sim_ipts[isim][i]][:-1]
            x[i,-1]  = 1-x[i,:-1].sum(axis=-1)
        x[-1,:] = x[0,:]                                           # Copy the first x to the extra x, useful for the algorithm
        xcen = x[:-1,:].mean(axis=0)                               # The center of this simplex
        Gcont,idcont = self.call_Gfunctions_liquids(self.components,xcen)  # Compute the real G at this center
        Gcen = 0.                                                  # Now compute the mean of the G values at the corners of the simplex
        for i in range(N):
            Gcen += pts[sim_ipts[isim][i]][-1]
        Gcen /= N
        if return_all and nx is not None:
            l   = np.zeros(N)                                      # Compute the distance between successive points on the simplex
            for i in range(N):                                     # The method here is not the most correct one, but it works.
                l[i] = ((x[i,:]-x[i+1,:])**2).sum()**0.5
            dx = 1/nx
            mr = l.min()/dx                                        # Note: Here use min()
            return Gcen<Gcont,x,l,mr                               # In addition to inmiscibility, return also further information
        else:
            return Gcen<Gcont                                      # If Gcen<Gcont, then this simplex is a simplex of inmiscibility

    def _test_if_simplex_is_tie_line(self,pts,isim,sim_ipts,crystals,liquids,solsols,mrcrit=None,nx=None):
        if mrcrit is None: mrcrit=self.mrcrit
        assert self.ncomponents>2, 'Internal Error: Should not call _test_if_simplex_is_tie_line for binaries'

        # Do preliminary tests, by which we can already exclude this simplex being
        # a tie line.

        if self.ncomponents==3:

            # For ternaries the following exclusions (i.e. not being a tie line) apply:

            if len(crystals)>1: return False
            if len(set(liquids))==len(liquids) and len(set(solsols))==len(solsols): return False

        else:
            
            # For quaternaries and higher, we have not yet really formulated the rules for
            # what constitutes a tie line and what not. For now we declare all to be NOT tie lines.

            return False

        # Now that the current simplex might perhaps be a tie line, we check for its
        # geometry. If it is a tie line, it should be a needle like geometry. We
        # compare the length to its width. This is, unfortunately, a very inaccurate
        # method, which is resolution-dependent. Especially at low resolution this
        # might go wrong. So beware.
        
        if nx is None:
            if hasattr(self,'gridsnres'):
                nx = self.gridsnres[-1]
            else:
                raise ValueError('If nx is not set, you must have self.gridsnres set')
        N   = len(pts[0])
        npt = len(pts)
        x   = np.zeros((N+1,N))
        for i in range(N):
            x[i,:-1] = pts[sim_ipts[isim][i]][:-1]
            x[i,-1]  = 1-x[i,:-1].sum(axis=-1)
        x[-1,:] = x[0,:]
        l   = np.zeros(N)
        for i in range(N):
            l[i] = ((x[i,:]-x[i+1,:])**2).sum()**0.5
        dx = 1/nx
        mr = l.min()/dx   # Note: Here use min()
        return mr<=mrcrit

    def _test_if_simplex_is_inmiscible_solid_solutions(self,pts,isim,sim_ipts,names):
        # Check for inmiscibility in a solid solution. Similar to
        # _test_if_simplex_is_inmiscible_liquids(), but now only for
        # a solid solution, and without the nx and return_all options.
        # For now this is only used for simplices that are purely
        # solid solutions, but later one could also use this to check
        # if e.g. a single crystal (outside the plane of the solid
        # solution) connects to two points on the solid solution that
        # both form an inmiscibility gap.
        N   = len(pts[0])                                          # Nr of components 
        npt = len(pts)                                             # Nr of points available (not all are part of the convex hull)
        solsolnames = []
        for i in range(N):
            if names[i][0]=='+':
                solsolnames.append(names[i][1:])
        solsolnames = list(set(solsolnames))
        assert len(solsolnames)>0, 'Error in _test_if_simplex_is_inmiscible_solid_solutions(): No corner point is a solid solution.'
        answers = {}
        for ssname in solsolnames:
            isolsol  = self.solsolnames[ssname]
            solsol   = self.solsols[isolsol]
            M        = solsol.ygrid.shape[-1]   # Nr of endmembers of this solid solution
            assert M<=N, 'Error: Solid solution has more endmembers than the components of this phase diagram.'
            icorners = []
            for i in range(N):
                if names[i][1:]==ssname:
                    icorners.append(i)
            if len(icorners)>1:  # You can only have 'inmiscibility' for 2 or more points on the same solid solution
                y   = np.zeros((len(icorners),M))                  # Y molar fractions of the corner points of this simplex
                for i in range(len(icorners)):
                    y[i,:] = self.theyvals[-1][sim_ipts[isim][icorners[i]]]
                ycen  = y.mean(axis=0)                             # The center of this simplex in y-space
                ycen  = np.stack([ycen])
                Gcont = solsol.Gfunc(ycen)*solsol.mfGfact          # Compute the real G at this center
                Gcen  = 0.                                         # Now compute the mean of the G values at the corners of the simplex
                for i in range(len(icorners)):
                    Gcen += pts[sim_ipts[isim][i]][-1]
                Gcen /= len(icorners)
                answers[ssname] = Gcen<Gcont                       # If Gcen<Gcont, then this simplex is a simplex of inmiscibility
            else:
                answers[ssname] = False
        return answers

    def refine_near_binodals_2d(self,pts,simplices,nx,nfact=2,nspan=1,return_newnx=False):
        eps      = 1e-3
        ncomp    = len(pts[0])
        assert ncomp==3, 'Error: refine_near_binodals_2d() only works with 3 components.'
        tie_lines = []
        for isim in range(len(simplices['id'])):
            #
            # ADD MORE OF THESE TRIGGERS FOR TERNARY OR HIGHER
            #
            if simplices['stype'][isim]=='tieline_c0l3' or \
               simplices['stype'][isim]=='tieline_c1l2' or \
               simplices['stype'][isim]=='tieline_c0l2s1' or \
               simplices['stype'][isim]=='tieline_c0l1s2':
                tie_lines.append(isim)
        nxnew    = (nx-1)*nfact+1
        gridold  = (np.stack(pts)[:,:-1]*(nx-1)+eps).astype(int)
        check    = (np.stack(pts)[:,:-1]*(nx-1)+1-eps).astype(int)
        assert np.all(gridold==check), f'Error: Current grid is finer than 1/{nx-1}'
        gridnew  = gridold * nfact
        gridnset = set([tuple(g) for g in gridnew])
        gaddnew  = set()
        for ib in tie_lines:
            for x in simplices['x'][ib]:
                gr = (x[:ncomp-1]*(nx-1)+eps).astype(int) * nfact
                # Next lines only work for ncomp==3
                gadd  = set()
                for ix0 in range(gr[0]-nspan*nfact,gr[0]+nspan*nfact+1):
                    for ix1 in range(gr[1]-nspan*nfact,gr[1]+nspan*nfact+1):
                        if ix0>=0 and ix1>=0 and ix0+ix1<nxnew:
                            gadd.add((ix0,ix1))
                gaddnew = gaddnew.union(gadd)
        gridnset = gridnset.union(gaddnew)
        grnew    = np.stack(list(gridnset))
        xnew     = np.zeros((len(grnew),ncomp))
        xnew[:,:-1] = grnew.astype(float)/(nxnew-1)
        xnew[:,-1]  = 1-xnew[:,:-1].sum(axis=-1)
        Gnew     = np.zeros(len(grnew))
        idsnew   = np.zeros(len(grnew),dtype=int)
        for l,liq in enumerate(self.liquids):
            liq.xgrid     = xnew
            liq.Ggrid     = np.zeros(len(xnew))
        for ix in range(len(grnew)):
            G,id,Gl             = self.call_Gfunctions_liquids(self.components,xnew[ix],return_Gliq=True)
            Gnew[ix],idsnew[ix] = G,id
            for l,liq in enumerate(self.liquids):
                liq.Ggrid[ix] = Gl[l]
        ptsnew   = xnew.copy()
        ptsnew[:,-1] = Gnew[:]
        if return_newnx:
            return ptsnew,idsnew,nxnew
        else:
            return ptsnew,idsnew

    def refine_near_a_point_2d(self,pts,simplices,nx,xx,nfact=2,nspan=1,niter=1,return_newnx=False):
        if len(xx.shape)<2:
            xx   = np.stack([xx])
        eps      = 1e-3
        ncomp    = len(pts[0])
        assert ncomp==3, 'Error: refine_near_a_point_2d() only works with 3 components.'
        gridold  = (np.stack(pts)[:,:-1]*(nx-1)+eps).astype(int)
        check    = (np.stack(pts)[:,:-1]*(nx-1)+1-eps).astype(int)
        assert np.all(gridold==check), f'Error: Current grid is finer than 1/{nx-1}'
        for iiter in range(niter):
            nxnew    = (nx-1)*nfact+1
            gridnew  = gridold * nfact
            gridnset = set([tuple(g) for g in gridnew])
            gaddnew  = set()
            for x in xx:
                gr       = (x[:ncomp-1]*(nx-1)+eps).astype(int) * nfact
                # Next lines only work for ncomp==3
                gadd     = set()
                for ix0 in range(gr[0]-nspan*nfact,gr[0]+nspan*nfact+1):
                    for ix1 in range(gr[1]-nspan*nfact,gr[1]+nspan*nfact+1):
                        if ix0>=0 and ix1>=0 and ix0+ix1<nxnew:
                            gadd.add((ix0,ix1))
                gaddnew  = gaddnew.union(gadd)
            gridnset = gridnset.union(gaddnew)
            grnew    = np.stack(list(gridnset))
            # For the potential next iteration
            nx       = nxnew
            gridold  = grnew
        xnew     = np.zeros((len(grnew),ncomp))
        xnew[:,:-1] = grnew.astype(float)/(nxnew-1)
        xnew[:,-1]  = 1-xnew[:,:-1].sum(axis=-1)
        Gnew     = np.zeros(len(grnew))
        idsnew   = np.zeros(len(grnew),dtype=int)
        for l,liq in enumerate(self.liquids):
            liq.xgrid     = xnew
            liq.Ggrid     = np.zeros(len(xnew))
        for ix in range(len(grnew)):
            G,id,Gl             = self.call_Gfunctions_liquids(self.components,xnew[ix],return_Gliq=True)
            Gnew[ix],idsnew[ix] = G,id
            for l,liq in enumerate(self.liquids):
                liq.Ggrid[ix] = Gl[l]
        ptsnew   = xnew.copy()
        ptsnew[:,-1] = Gnew[:]
        if return_newnx:
            return ptsnew,idsnew,nxnew
        else:
            return ptsnew,idsnew

    def _follow_neighbors_till_end(self,iselection,istart,stype,maxstep=10000):
        icurr = istart
        simplices = self.thesimplices[-1]
        neigh = simplices['neighbors_'+stype][icurr]
        if(len(neigh)!=2):
            if(len(neigh)==1):
                return icurr,0
            else:
                return icurr,None
        iprev = neigh[0]  # Random choice of the two
        for istep in range(maxstep):
            assert iprev in neigh, 'Weird error: neighbors not bidirectional'
            if iprev==neigh[0]:
                inext = neigh[1]
                ii    = 0
            else:
                inext = neigh[0]
                ii    = 1
            iprev = icurr
            icurr = inext
            neigh = simplices['neighbors_'+stype][icurr]
            if(len(neigh)==1):
                assert iprev==neigh[0], 'Weird error: neighbors not bidirectional at endpoint'
                return icurr,0
            elif(len(neigh)>2):
                return iprev,ii
            if icurr==istart:
                return icurr,0
        raise ValueError('Did not find the start/end of the neighbor series')

    def _next_neighbor(self,iselection,iprev,icurr,stype):
        simplices = self.thesimplices[-1]
        neigh = simplices['neighbors_'+stype][icurr]
        if iprev<0:
            if(len(neigh))==1:
                return neigh[0]
            else:
                iprev = -(iprev+1)
                assert iprev in neigh, 'Weird error: iprev not in neigh'
                return iprev
        if len(neigh)!=2:
            return None  # Reached end
        assert iprev in neigh, 'Weird error: neighbors not bidirectional'
        if iprev==neigh[0]:
            return neigh[1]
        else:
            return neigh[0]
    
    def sort_tie_lines(self,iselection,stype,nrgroupmax=100,maxstep=10000):
        groups = []
        simplices = self.thesimplices[-1]
        left   = iselection.copy()
        for igroup in range(nrgroupmax):
            if len(left)==0:
                break
            icurr,ii = self._follow_neighbors_till_end(iselection,left[-1],stype,maxstep=10000)
            group = []
            if ii is None:
                left.remove(left[-1])
            else:
                assert icurr in left, f'Something went wrong in left. icurr={icurr}'
                iprev = -simplices['neighbors_'+stype][icurr][ii]-1
                for istep in range(maxstep):
                    group.append(icurr)
                    left.remove(icurr)
                    inext = self._next_neighbor(iselection,iprev,icurr,stype)
                    if inext is None:
                        break
                    if inext not in left:
                        break
                    iprev = icurr
                    icurr = inext
            if len(group)>0:
                groups.append(group)
        for group in groups:
            for i,isim in enumerate(group):
                xcurr = simplices['xtieline'][isim]
                if i>0:
                    dx0110 = ((xcurr[0,:]-xprev[1,:])**2).sum() + ((xcurr[1,:]-xprev[0,:])**2).sum()
                    dx0011 = ((xcurr[0,:]-xprev[0,:])**2).sum() + ((xcurr[1,:]-xprev[1,:])**2).sum()
                    if dx0110<dx0011:
                        xswap         = xcurr.copy()
                        xcurr[0,:]    = xswap[1,:]
                        xcurr[1,:]    = xswap[0,:]
                        simplices['xtieline'][isim] = xcurr
                xprev  = xcurr
        return groups

    def identify_component_minerals(self,mdb,components):
        """
        This function determines which of the crystals in the mineral database
        are the true components with the lowest DfG.

        Arguments:

          mdb              The mineral database (see read_minerals_and_liquids())
          components       List of the formulae of the components, e.g. ['SiO2','MgO','Al2O3'].

        Returns:

          icomponents      List of integer indices of the mdb database pointing to the true
                           components. If the mdb database has different versions of the components
                           (e.g. alpha-quartz or beta-quartz for SiO2), then the version is
                           picked that has (for the given T) the smallest value of DfG.
          DfGcomponents    List of DfG values of these components.
          
        """
        icomponents   = np.zeros(len(components),dtype=int)
        DfGcomponents = np.zeros(len(components))+1e90
        for k,e in enumerate(components):
            DfG  = 1e90
            iend = -1
            ms = mdb[mdb['Formula']==e]
            if len(ms)>0:
                for i,row in ms.iterrows():
                    if(row['DfG']<DfG):
                        iend = i
                        DfG  = row['DfG']
                assert i>-1, f'Error: Could not find component mineral {e} among minerals (stranger version)'
            else:
                x    = np.zeros(len(components))
                x[k] = 1.
                for i,row in mdb.iterrows():
                    xrow = row['x']
                    if(np.all(x==xrow)):
                        if(row['mfDfG']<DfG):
                            iend = i
                            DfG  = row['mfDfG']
                assert i>-1, f'Error: Could not find component mineral {e} among minerals (stranger version)'
            icomponents[k]   = iend
            DfGcomponents[k] = DfG
        return icomponents,DfGcomponents

    def convert_mole_fraction_into_mass_fraction(self,componmass,x,G=None):
        """
        If you have a mole fraction x (such that x.sum(axis=-1)==1), or an array
        of them (again such that x.sum(axis=-1)==1, so x[...,:]), then you can
        convert them into mass fractions xm (again such that xm.sum(axis=-1)==1)
        with this function.

        Arguments;

          componmass       List of the masses (in atomic units) of the components.
          x                Mole fraction x values. E.g. x = np.array([0.2,0.3,0.5]) or an
                           array of them, e.g. x = np.array([[0.2,0.3,0.5],[0.1,0.4,0.5]])

        Optional:

          G                If G is passed on, then also the adjusted value of G
                           (from J/mole to J/mass) is returned

        Returns:

          xm               Array of the same dimension as x, but this time with the
                           mass fractions instead of the mole fractions.

          mtot             The mass of 1 mole of this mineral.

          Gm               If G was passed on, this is the new version of G in
                           units of J/mass.
        """
        ncomp = len(componmass)
        if len(x.shape)==1:
            mtot = 0.
        else:
            mtot = np.zeros_like(x[...,0])
        for k in range(ncomp):
            mtot += x[...,k]*componmass[k]
        xm = np.zeros_like(x)
        for k in range(len(componmass)):
            xm[...,k] = x[...,k]*componmass[k]/mtot
        if G is not None:
            Gm = G/mtot
            return xm,mtot,Gm
        else:
            return xm,mtot

    def convert_mass_fraction_into_mole_fraction(self,componmass,xmass,Gmass=None):
        """
        The inverse of convert_mole_fraction_into_mass_fraction().

        If you have a mass fraction x (such that x.sum(axis=-1)==1), or an array
        of them (again such that x.sum(axis=-1)==1, so x[...,:]), then you can
        convert them into mole fractions xmol (again such that xmol.sum(axis=-1)==1)
        with this function.

        Arguments;

          componmass       List of the masses (in atomic units) of the components.
          xmass            Mass fraction x values. E.g. x = np.array([0.2,0.3,0.5]) or an
                           array of them, e.g. x = np.array([[0.2,0.3,0.5],[0.1,0.4,0.5]])

        Optional:

          Gmass            If Gmass is passed on, then also the adjusted value of G
                           (from J/mass to J/mole) is returned.

        Returns:

          xmol             Array of the same dimension as x, but this time with the
                           mole fractions instead of the mass fractions.

          moltot           (if return_also_moltot==True) the nr of moles of 1 g of this mineral.

          Gmol             If Gmass was passed on, this is the new version of G in
                           units of J/mol.
        """
        ncomp = len(componmass)
        componmol = 1/np.array(componmass)
        if len(xmass.shape)==1:
            moltot = 0.
        else:
            moltot = np.zeros_like(xmass[...,0])
        for k in range(len(icomponents)):
            moltot += xmass[...,k]*componmol[k]
        xmol = np.zeros_like(xmass)
        for k in range(ncomp):
            xmol[...,k] = xmass[...,k]*componmol[k]/moltot
        if Gmass is not None:
            Gmol = Gmass/moltot
            return xmol,moltot,Gmol
        else:
            return xmol,moltot

    def get_chain(self,tpairs,istart=0):
        assert type(tpairs[0]) is tuple, 'Error: tpairs must be a list of tuples.'
        npairs = np.stack(tpairs.copy())
        lpairs = [list(npairs[:,0]),list(npairs[:,1])]
        n      = len(tpairs)
        pstart = tuple(npairs[istart])
    
        tpairs.remove(pstart)
        lpairs[0].pop(istart)
        lpairs[1].pop(istart)
    
        # Forward
        p      = pstart
        m      = p[0]
        index  = istart
        chainf = [p]
        for k in range(n):
            j  = p[1]
            if j in lpairs[0]:
                index = lpairs[0].index(j)
                m     = lpairs[1][index]
                lpairs[0].pop(index)
                lpairs[1].pop(index)
                po    = tuple([j,m])
            elif j in lpairs[1]:
                index = lpairs[1].index(j)
                m     = lpairs[0][index]
                lpairs[0].pop(index)
                lpairs[1].pop(index)
                po    = tuple([m,j])
            else:
                break
            p     = tuple([j,m])
            chainf.append(p)
            tpairs.remove(po)
    
        # Backward
        p      = pstart
        m      = p[1]
        index  = istart
        chainb = []
        for k in range(n):
            j  = p[0]
            if j in lpairs[0]:
                index = lpairs[0].index(j)
                m     = lpairs[1][index]
                lpairs[0].pop(index)
                lpairs[1].pop(index)
                po    = tuple([j,m])
            elif j in lpairs[1]:
                index = lpairs[1].index(j)
                m     = lpairs[0][index]
                lpairs[0].pop(index)
                lpairs[1].pop(index)
                po    = tuple([m,j])
            else:
                break
            p     = tuple([m,j])
            chainb.append(p)
            tpairs.remove(po)
    
        chain = chainb[::-1] + chainf
        return chain,tpairs
    
    def get_chains(self,tpairs):
        chains = []
        while len(tpairs)>0:
            chain,tpairs = self.get_chain(tpairs)
            chains.append(chain)
        return chains

    def linear_interpolation_weights_in_simplex(self,x,xcorners,invert=True):
        """
        Given an x[:] vector and a simplex with corner points with
        coordinates xcorners[:,:] such that xcorners[0,:] are
        the coordinates of corner 0, xcorners[1,:] are
        the coordinates of corner 1, etc. If you now want to
        linearly interpolate something between the corners,
        you need the linear interpolation weights w[:] such
        that

          interpolated_value  = (w[:]*corner_values[:]).sum()

        This function computes these w[:]. But since x can be
        a vector of multiple points, w will be w[:,:] instead.

        Note that if one or more of these w coefficients is < 0,
        then the x point lies outside of the simplex.

        Options:

          invert    If True, then invert the matrix, then apply
                    this to x. Is better if x[:nx,:ncomp] has
                    a large number of points (i.e. nx>>1).

                    If False, then solve the matrix equation
                    for each x[k] vector separately. Is better
                    if nx is small but ncomp is large.
        """
        from scipy import linalg
        ncomp = len(x)
        assert ncomp==self.ncomponents, 'Nr of components not same as those of self'
        if len(x.shape)==1: x = x.reshape((1,len(x)))
        nx = len(x)
        assert len(xcorners.shape)==2, 'xcorners must have 2 indices'
        assert xcorners.shape[0]==ncomp, f'xcorners must be {ncomp}x{ncomp} in size'
        assert xcorners.shape[1]==ncomp, f'xcorners must be {ncomp}x{ncomp} in size'
        N = ncomp
        assert np.all(np.abs(x.sum(axis=-1)-1)<1e-6), 'Error in linear interpolation on a simplex: x does not sum to 1'
        for i in range(N):
            assert np.abs(xcorners[i].sum()-1)<1e-6, f'Error in linear interpolation on a simplex: xcorners[{i}][:] does not sum to 1'
        evec = np.zeros((N-1,N-1))
        for i in range(N-1):
            evec[:,i] = xcorners[i][:-1]-xcorners[-1][:-1]
        w        = np.zeros([nx,ncomp])
        if invert:
            einv     = linalg.inv(evec)
            w[:,:-1] = (einv[None,:,:]*(x[:,None,:-1]-xcorners[-1,None,:-1])).sum(axis=-1)
            w[:,-1]  = 1-w[:,:-1].sum(axis=-1)
        else:
            for k in range(nx):
                y = linalg.solve(evec,(x[k,:-1]-xcorners[-1][:-1]))
                y = np.hstack((y,1-y.sum()))
                w[k,:] = y
        return w

    def find_simplex_for_given_x(self,x,return_G=False,ilevel=-1):
        """
        Given a value, or a list of values, of x, this function
        returns the index (integer) of the simplex of the convex
        hull on which this point lies.

        Arguments:

          x         Location x[0:N] or array of locations x[0:nx,0:N]
                    where the interpolation should be done.

        Options:

          return_G  If True, then also return the interpolated G
                    value at the bottom of the convex hull

        Returns:

          isims     List of integers representing the indices of the
                    self.thesimplices[-1][:] simplices at the bottom
                    of the convex hull.

        Note: The algorithm is not at all optimal: For all x we try all
              simplices at the bottom of the hull, and select the highest
              G value, which is the correct one. Explanation: the highest 
              crossing with the planes through all simplices is, in fact, 
              the crossing with the simplex inside of which x is. since the
              simplices are the bottom of the hull, these highest points
              are, in fact, the lowest value, at x, of the volume enclosed
              by the convex hull. It works. But a much more efficient
              method would search for the right simplex by walking through
              the simplices using the hull.neighbors array. For arrays of
              contiguous x points, one can use the previous simplex as an
              initial guess for finding the current simplex. This is, however,
              not efficient in Python and would require an external C or
              Fortran library. However, for the current demonstration
              purposes, the present method is sufficient.
        """
        hull = self.thehulls[ilevel]
        if type(x) is list:
            x = np.stack(x)
        if len(x.shape)==1:
            x = x[None,:]
            
        # NOTE FOR IMPROVEMENT: The three lines below are a bit
        # dangerous, since they assume (presumably correctly, but
        # not 100% guaranteed if later changes are made to the code)
        # that the simplices in the self.thesimplices[ilevel] have
        # exactly the same order as the selected nvec and offset
        # from the original hull with the mask-selection. Better
        # would be to store the nvec and offset directly in the
        # two new columns of the self.thesimplices[ilevel], so that 
        # confusion cannot happen.
        # 2025-09-15
        
        mask    = hull.equations[:,-2]<-self.epsequ   # Select bottom of the hull
        nvec    = hull.equations[mask,:-1]   # Normal vectors of the simplices (at bottom)
        offset  = hull.equations[mask,-1]    # Offset of simplices from origin (at bottom)
        if type(x) is list:
            x = np.stack(x)
        N       = x.shape[-1]                # Nr of components
        nx      = x.shape[0]                 # Nr of x vectors to interpolate at
        ns      = len(offset)                # Nr of simplices at bottom of the hull
        heights = np.zeros((nx,ns))
        for i in range(N-1):
            heights -= x[:,i,None]*nvec[:,i][None,:]
        heights -= offset[None,:]
        heights /= nvec[:,-1][None,:]
        simid    = np.argmax(heights, axis=-1)   # Find which simplex the x hits at the bottom of the hull
        if return_G:
            G = np.zeros(nx)
            for ix in range(nx):
                G[ix] = heights[ix,simid[ix]] # Get the line-simplex crossing height at that simplex
            return simid,G
        else:
            return simid

    def get_simplex_for_given_x(self,x,ilevel=-1):
        """
        Given a value of x (no list of x points allowed!), this function
        returns a dict with all the properties of the simplex of the convex
        hull on which this point lies.

        Arguments:

          x         Location x[0:N] where to find the simplex

        Returns:

          simplex   Dict of all properties of the simplex. This is
                    essentially a single row of the 'dict of lists'
                    of self.thesimplices[-1]. 
        """
        from phasehull.phasehull_support import get_row_from_DL
        x = np.stack(x)
        assert len(x.shape)==1, 'Error in get_simplex_for_given_x(): x can only be a single point (must be an array of dimension 1).'
        isim = self.find_simplex_for_given_x(x,return_G=False,ilevel=ilevel)[0]
        return get_row_from_DL(self.thesimplices[ilevel],isim)

    def make_component_name_list_for_big_Y_vector(self,fullname=False):
        # Create the list of component names, so that the resulting bigY
        # vector is easier to interpret.
        if fullname:
            col = 'Mineral'
        else:
            col = 'Abbrev'
        self.bigY_component_names = []
        if self.ncryst>0:
            for isol in range(self.ncryst):
                self.bigY_component_names.append(self.crystals[0].dbase[col].iloc[isol])
        if self.nliq>0:
            for iliq,liq in enumerate(self.liquids):
                ilq0 = self.ncryst+iliq*self.ncomponents
                for iend in range(self.ncomponents):
                    self.bigY_component_names.append(liq.name+'_'+self.components[iend])
        return self.bigY_component_names

    def find_composition_for_given_x(self,x,simid=None,ilevel=-1,return_simid=False):
        """
        Once you have the phase diagram, you may want to determine the actual
        chemical composition of a material for a given component-composition x:
        The fraction of each possible mineral from the self.crystals database,
        and the amount of liquid (and the component-composition of the liquid) 
        for each liquid from the self.liquids list. This is done with this
        function.

        Arguments:

          x         Location x[0:N] or array of locations x[0:nx,0:N]
                    where the interpolation should be done.

        Returns:

          bigY      The big X vector with all the phases.

        If return_simid==True, then it also returns

          simid     The simplex id at each point

        Note: To know which phase is which component of bigY, you can look
              at self.bigY_component_names.
        """
        if type(x) is list: x=np.array(x)
        if len(x.shape)==1: x=x[None,:]
        if not hasattr(self,'bigY_component_names'):
            self.make_component_name_list_for_big_Y_vector()
        simplices = self.thesimplices[ilevel]
        if simid is None:
            simid = self.find_simplex_for_given_x(x,return_G=False,ilevel=ilevel)
        else:
            assert len(x)==len(simid), 'Error: Nr of simplex-indices unequal to nr of points'
        nx      = x.shape[0]                               # Nr of x vectors to interpolate at
        ncomp   = self.ncryst + self.nliq*self.ncomponents   # Size of the bigY vector
        bigY    = np.zeros((nx,ncomp))
        for ix in range(nx):
            xcorners = simplices['x'][simid[ix]]
            ipts     = simplices['ipts'][simid[ix]]
            #weight   = self.linear_interpolation_weights_in_simplex(x[ix],xcorners,invert=False)[0]
            weight   = self.linear_interpolation_weights_in_simplex(x[ix],xcorners,invert=True)[0]
            for iend in range(self.ncomponents):
                idd  = self.theids[ilevel][ipts[iend]]
                if type(idd) is str:
                    # This corner point is a crystal
                    k          = self.crystals[-1].index[idd]
                    bigY[ix,k] = weight[iend]
                else:
                    # This corner point is a liquid
                    for iliq,liq in enumerate(self.liquids):
                        ilq0 = self.ncryst+iliq*self.ncomponents
                        bigY[ix,ilq0:ilq0+self.ncomponents] += weight[iend]*xcorners[iend,:]
        if return_simid:
            return bigY,simid
        else:
            return bigY

    def cut_through_phase_diagram_1d(self,xstart,xend,nx=100,ilevel=-1):
        """
        This method will return a bigY vector (vector of abundances of all
        possible phases) as well as the id (index) of the simplex at a
        1D series of points between xstart and xend. It therefore acts
        as 1D 'probe' of the phase diagram, much like 'ice core drilling'
        on antarctica.

        Arguments:

          xstart     The starting position of the 1D 'drill'
          xend       The ending position of the 1D 'drill'

        Optional:
        
          nx         The nr of points of the 1D 'drill'. Default=100.

        Returns:

          s          The 1D coordinate along the path
          x          The array of x positions along the path
          bigY       The 2D array containing the abundances of all
                     the phases. NOTE: You can find the names of these
                     phases in self.bigY_component_names.
          simid      The array of simplex indices
        """
        ncomp      = self.ncomponents
        s          = np.linspace(0,1,nx)
        x          = (1-s[:,None])*xstart[None,:] + s[:,None]*xend[None,:]
        bigY,simid = self.find_composition_for_given_x(x,ilevel=-1,return_simid=True)
        return s,x,bigY,simid

    def map_phase_diagram(self,nres=100,ilevel=-1,colormap=None,xcorners=None):
        if xcorners is not None:
            ncorners    = len(xcorners)
            assert ncorners<=self.ncomponents, 'Error: Cannot have more corners than components'
        else:
            ncorners    = self.ncomponents
        xxgrid          = make_x_grid(nres,ncorners)
        if xcorners is not None:
            x           = np.zeros((len(xxgrid),self.ncomponents))
            for i in range(len(xcorners)):
                x      += xxgrid[:,i][:,None]*xcorners[i][None,:]
        else:
            x           = xxgrid
        simplices       = self.thesimplices[ilevel]
        answer          = {}
        answer['x']     = x
        answer['isim']  = self.find_simplex_for_given_x(answer['x'])
        answer['stype'] = []
        if colormap is not None:
            answer['color'] = []
        for isim in answer['isim']:
            stype = simplices['stype'][isim]
            answer['stype'].append(stype)
            if colormap is not None:
                if stype in colormap:
                    color = colormap[stype]
                else:
                    color = np.nan
                answer['color'].append(color)
        if xcorners is not None:
            return xxgrid,answer
        else:
            return answer

#---------------------------------------------------------------------------
# Some functions used in the above classes
#---------------------------------------------------------------------------
def make_integer_grid(nres,ncomponents):
    """
    The function that constructs the base grid in integer form. Note
    that for 3 enmembers the grid is triangular, for 4 components
    the grid is a tetrad, etc. The integer grid is constructed in
    a way that the full allowed space is uniformly mapped with
    nres points along each axis. 
    """
    ixgrid = set()
    if ncomponents==2:
        for ix0 in range(0,nres+1):
            if ix0>=0 and ix0<=nres:
                k = nres - ix0
                ixgrid.add((ix0,k))
    elif ncomponents==3:
        for ix0 in range(0,nres+1):
            for ix1 in range(0,nres+1):
                if ix0>=0 and ix1>=0 and ix0+ix1<=nres:
                    k = nres - ix0 - ix1
                    ixgrid.add((ix0,ix1,k))
    elif ncomponents==4:
        for ix0 in range(0,nres+1):
            for ix1 in range(0,nres+1):
                for ix2 in range(0,nres+1):
                    if ix0>=0 and ix1>=0 and ix2>=0 and ix0+ix1+ix2<=nres:
                        k = nres - ix0 - ix1 - ix2
                        ixgrid.add((ix0,ix1,ix2,k))
    elif ncomponents==5:
        for ix0 in range(0,nres+1):
            for ix1 in range(0,nres+1):
                for ix2 in range(0,nres+1):
                    for ix3 in range(0,nres+1):
                        if ix0>=0 and ix1>=0 and ix2>=0 and ix3>=0 and ix0+ix1+ix2+ix3<=nres:
                            k = nres - ix0 - ix1 - ix2 - ix3
                            ixgrid.add((ix0,ix1,ix2,ix3,k))
    elif ncomponents==6:
        for ix0 in range(0,nres+1):
            for ix1 in range(0,nres+1):
                for ix2 in range(0,nres+1):
                    for ix3 in range(0,nres+1):
                        for ix4 in range(0,nres+1):
                            if ix0>=0 and ix1>=0 and ix2>=0 and ix3>=0 and ix4>=0 and ix0+ix1+ix2+ix3+ix4<=nres:
                                k = nres - ix0 - ix1 - ix2 - ix3 - ix4
                                ixgrid.add((ix0,ix1,ix2,ix3,ix4,k))
    else:
        raise ValueError(f'Unfortunately at the moment we cannot handle nr of components = {ncomponents}')
    ixgrid = list(ixgrid)
    ixgrid.sort()
    ixgrid = np.array(ixgrid)
    return ixgrid

def make_x_grid(nres,ncomponents):
    """
    Wrapper around make_integer_grid(), where the integer values are rescaled back
    to values between 0 and 1.
    """
    ixgrid = make_integer_grid(nres,ncomponents)
    x      = ixgrid / nres
    return x

