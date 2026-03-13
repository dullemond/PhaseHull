#---------------------------------------------------------------------------
#                  Part of PhaseHull, a simple python package
#                    to compute equilibrium phase diagrams
#
#                           (C) C. P. Dullemond
#                      Heidelberg University, Germany
#                                Sept 2025
#
# This module is a classic Gibbs minimization tool, not related to
# the convex hull algorithm, but supplied as an alternative tool.
# It finds the coexisting phases and their mole fractions for a
# given (single) bulk composition. Advantages (compared to the convex
# hull algorithm): allows larger systems (more system components),
# allows multi-phase mixtures with many more subphases than the
# number of system components (most typical example: an ideal gas
# phase with numerous gaseous molecule species), and it is ideal
# for use in a simulation. Disadvantages (compared to the convex
# hull algorithm): Does not produce the entire phase diagram,
# does not give information about tie lines, tie simplices, etc,
# and it does not account for miscibility gaps. 
#---------------------------------------------------------------------------

import numpy as np
import pandas as pd
from phasehull.phasehull_support import *
from phasehull.phasehull_subsystem import *

class GibbsMinFinder(object):
    def __init__(self,components,T,P=1.,crystaldb=None,liquids=None,
                 multiphasesolutions=None,eps=1e-8,tol=1e-8,maxiter=100,
                 nrtrymax=4,fullname=False,factors=None):
        """
        This module is a classic Gibbs minimization tool, not related to
        the convex hull algorithm. It works in Y-space which has an Y_k for
        each candidate phase (e.g. Mg2SiO4, MgSiO3 etc). This list of phases
        is generally much larger than the set of system components
        (e.g. MgO, SiO2 etc or elements Mg, Si, O). The Y vector is in mole
        fraction, such that Y.sum()==1. 
        
        A note on the use of the letter Y instead of X: We use the letter Y for
        all phases that are not the basic system components, and x for the
        system components (usually the primitive oxides such as SiO2, MgO etc
        or elements Si, Mg, O). We capitalize Y to indicate that these phases
        are not meant as alternative system components (which would limit them
        to the number of independent system components), but they are all possible
        phases (typically many more than the number of independent system
        components), and we let the algorithm decide which ones are present
        (they have Y[i]>0) and which ones are not (they have Y[i]==0). So to
        summarize:

          ncomp             The nr of system components.
          x[0:ncomp]        The mole fractions of the ncomp system components
                            for a given phase.
          xbulk[0:ncomp]    The bulk x, i.e. the bulk composition (input).
          Y[0:nphases]      The mole fractions of the possible phases (output).
                            In all practical cases, nphases>ncomp.
          nphases           The number of phases. Note that each liquid
                            (being formally one phase) adds ncomp phases
                            to nphases. In that sense, each component of
                            a liquid is treated as ncomp phases, even though,
                            physically speaking, they are just one phase
                            per liquid.

        For the constraint equations (such that all elements are conserved)
        we use the same method as used in phase diagrams (i.e. as used in
        the PhaseHull algorithm): We assign each of the above phases an x-vector,
        where the components x[0:ncomp] give the mole fractions of the system 
        components from which that phase is constructed. These x vectors obey
        x.sum()==1. The bulk composition is then specified as a vector x, just
        as when using the convex hull algorithm for the phase diagrams. The
        constraints are then internally managed by requiring
        
          sum_k Y[k] * xphase[k,i] == xbulk[i]  for all i in [0,ncomp]
        
        The primary advantage of this way of implementing the constraints,
        as opposed to the classic method of counting elements, is that this
        way is compatible with the way the convex hull algorithm of PhaseHull
        sets its constraints. Mathematically, though, they are identical,
        just scaled versions of one another.

        The Y[k] value counts the mole fractions of phase k, where phase k
        is appropriately scaled to the formula unit consistent with one
        mole of constitute system component.
        
        For example: Two system components: SiO2 and MgO, and four phases:
        the crystalline solids SiO2, MgO, MgSiO3 and Mg2SiO4 (all in their
        given order). Then ncomp=2 and ncryst=4, meaning that len(x)==2
        and len(Y)==4. Y[0:ncryst] with ncryst=4 are the molar fractions of
        these crystal phases. Their x-vectors are [1,0], [0,1], [0.5,0.5]
        and [(1/3),(2/3)], respectively. With bulk composition xbulk=[0.5,0.5]
        one can have a 4-2=2-dimensional space of possible Y vectors. Some
        example points in this space are [0.5,0.5,0.,0.] (separate SiO2 and
        MgO phases in ratio 1:1), or [0.,0.,1.,0] (only MgSiO3), or
        [0.5,0.,0.,0.5] (separate SiO2 and Mg2SiO4).

        Important note about "scaling of phases":
        The amount of moles of all phases are counted with respect to
        their "scaled formula unit", which means that we use the formula
        unit that would be formed from one mole of system component.
        In the above 2-component example with four phases:
        The case of Y=[0.,0.,1.,0] (only MgSiO3) has "1." at the "MgSiO3"
        location, which means 1 mole of scaled (!) MgSiO3, where the
        scaled version of MgSiO3 is Mg(1/2)Si(1/2)O(3/2), because one
        mole of system components (in this case 0.5 mole of SiO2 and 0.5
        mole of MgO) creates 1 mole of Mg(1/2)Si(1/2)O(3/2). Likewise,
        the case of Y=[0.5,0.,0.,0.5] contains 0.5 moles of scaled Mg2SiO4,
        where the scaled version of Mg2SiO4 is Mg(2/3)Si(1/3)O(4/3),
        because 2/3 mole of MgO and 1/3 mole of SiO2 together make up
        1/3 of Mg2SiO4. The scaling factor of each phase relative to 
        the system components is listed in the mineral database in the 
        "moles" column, which is 2.0 for MgSiO3  and 3.0 for Mg2SiO4
        in the system of (SiO2,MgO).

        To use it you must prepare an instance of phasehull.CrystalDatabase
        and/or one or more instances of phasehull.Liquid and/or one or
        more instances of MultiPhaseSolution (see below). For the Liquids
        you must ensure to include a gammafunc() function, which should be
        a function that returns the activity coefficient vector gamma_i.
        This should be an exact function (do not use a numerical derivative
        to compute gamma_i, because it can compromise the computation of the
        Hessian).

        Once you have these, you set up the instance of GibbsMinFinder() with
        the following arguments:

        Arguments:

          components     The usual list of component names
          T              Tempeature in [K]
          P              Pressure in [bar]
          crystaldb      Instance of phasehull.CrystalDatabase
          liquids        List of instances of phasehull.Liquid
          multipsols     List of instances of MultiPhaseSolution

        Optional:

          eps            Step in Y used to compute numerical derivatives,
                         used in computing the Hessian.
          tol            Tolerance used for checking convergence
          maxiter        Maximum number of iterations for finding the
                         Y-location of the Gibbs minimum
          nrtrymax       Sometimes the solver returns an Y value with
                         (too) negative values. Another attempt is then
                         launched. At most nrtrymax attempts are done
                         before the minimization is aborted.
          fullname       If True, then in the self.Y_phase_name
                         (the list of component names in the Y vector)
                         will be the full names, not just the abbreviations.
          factors        If list or array numbers of length len(components), then
                         the components are, in fact, components*factors. Useful
                         for when a component is, e.g., MnSi0.5O2 instead of
                         Mn2SiO4 (the component would then be Mn2SiO4 with a
                         factor of 0.5).
        
        Once this is set up, e.g. using

          GMF = GibbsMinFinder(['CaO','SiO2'],2000.,1.,crystaldb=cr,liquids=[lq])

        where cr is an instance of phasehull.CrystalDatabase and lq an
        instance of phasehull.Liquid, you can now find the solution of the
        minimum:

          xbulk    = np.array([0.8,0.2])
          Yinit    = np.zeros(len(cr.dbase)+len(components))
          Yinit[0] = 1.
          Ymin     = GMF.find_minimum(xbulk,Yinit)
        
        NOTE: Currently, the 'solid solution' type (the solution along
              a restricted subspace of the full component space; see
              the phasehull.py main program) is not implemented. Only
              fixed composition crystals and full solutions (which we
              call liquids) are allowed.

        A note on miscibility gaps:
        If a miscibility gap occurs in the liquid (or in the future perhaps
        also in the MultiPhaseSolution case) due to strong interaction terms
        between the components (of the liquid) or subphases (of a
        MultiPhaseSolution), then miscibility gaps can occur. These would
        split the liquid phase (or MultiPhaseSolution case) into two or more
        coexisting phases. In the current setup of GibbsMinFinder() this is
        not possible, so instead the mean phase (of that solution) will be
        picked by the solver instead of the two or more split phases.
        """
        self.T          = T
        self.P          = P
        self.eps        = eps
        self.tol        = tol
        self.btol       = tol
        self.maxiter    = maxiter
        self.nrtrymax   = nrtrymax
        self.Rgas       = 8.314  # J/mol·K
        self.components = components
        if factors is None:
            factors     = np.ones(len(self.components))
        self.factors    = factors
        self.ncomp      = len(components)
        self.ncryst     = 0
        self.nliq       = 0
        self.nmps       = 0
        self.nmphases   = 0
        self.imps       = []

        # Add the crystals to GibbsMinFinder
        if crystaldb is not None:
            self.crystaldb = crystaldb
            self.mdb       = self.crystaldb.dbase
            assert 'x' in self.mdb.columns, 'Error: crystal database has no column called x'
            assert len(self.mdb['x'].iloc[0])==self.ncomp, 'Error: Nr of x components in crystal database not equal to number of components'
            if self.crystaldb.reset is not None: self.crystaldb.reset(self.T,self.P)
            self.ncryst   = len(self.mdb)
            # Do a self-check
            if crystaldb.components is not None:
                assert np.all(np.array(components)==np.array(drystaldb.components)), 'Error: The components of the crystal database are unequal to those of GibbsMinFinder'
                if drystaldb.factors is not None and self.factors is not None:
                    assert np.all(np.array(factors)==np.array(drystaldb.factors)), 'Error: The factors of the components of the crystal database are unequal to those of GibbsMinFinder'

        # Add the liquid(s) to GibbsMinFinder
        if liquids is not None:
            self.liquids = liquids
            if type(self.liquids) is not list:
                self.liquids = list(self.liquids)
            self.nliq = len(self.liquids)
            for liq in self.liquids:
                assert liq.gammafunc is not None, 'Error: The liquids must have a function gammafunc(x) for the activity coefficient, to be able to compute the Jacobian and Hessian of G.'
                if liq.reset is not None: liq.reset(self.T,self.P)
                self.compute_liquid_mu0(liq)
            # Do a self-check
            for i,lq in enumerate(liquids):
                if lq.components is not None:
                    assert np.all(np.array(components)==np.array(lq.components)), 'Error: The components of the liquid are unequal to those of GibbsMinFinder'
                    if lq.factors is not None and self.factors is not None:
                        assert np.all(np.array(factors)==np.array(lq.factors)), 'Error: The factors of the components of the liquid are unequal to those of GibbsMinFinder'

        # Add the multi-phase solutions to GibbsMinFinder
        if multiphasesolutions is not None:
            self.multips = multiphasesolutions
            self.nmps    = len(multiphasesolutions)
            imps         = self.ncryst + self.nliq*self.ncomp
            for iph,mps in enumerate(multiphasesolutions):
                self.imps.append(imps)
                nrphases = len(mps.dbase)
                self.nmphases += nrphases
                imps          += nrphases
            self.imps.append(imps)

        # Compute the total number of possible phases + phase components
        self.nphases = self.ncryst + self.nliq*self.ncomp + self.nmphases
        assert self.nphases>0, 'Error: Must have at least crystaldb or liquids or MultiPhaseSolutions'

        # Prepare information (such as name, formula, mass etc) for each element of Y
        self.make_phase_name_list_for_Y_vector(fullname=fullname)

    def find_minimum(self,xbulk,Yinit,T=None,P=None,return_res=False,return_unscaled=False,
                     options=None,nocrash=False,consY=True):
        """
        This is the actual Gibbs minimizer.

        Arguments:

          xbulk         The bulk composition, given as mole fraction of the system
                        components. Note that xbulk must have exactly len(components) elements.
                        If xbulk.sum() is not 1, then xbulk is interpreted automatically
                        as nbulk, which would be the amount of moles instead of mole fractions.
                        However: the resulting Y values are still normalized to Y.sum()==1.
                        The only way in which the nbulk interpretation finds its way into
                        the results is when you call self.get_interpretation_of_Y(): There
                        the mass values will now be scaled to the amount of moles.

          Yinit         The Y with which the search starts. This vector has the following
                        structure:
                        Array elements 0:ncryst are the molar fractions (per mole of component) of
                        the solid crystals
                        Elements ncryst:ncryst+ncomp are the molar fractions of the components of
                        the liquid (or if you have multiple liquids: the first one).
                        Elements ncryst+ncomp:ncryst+2*ncomp are the same, but for the second
                        liquid (if you have one).
                        etc

        Note that you do not have to have any liquids, nor any solids, but at least one of
        them. Note that a liquid can also be a vapor phase, in which case the gamma coefficient
        (the has to be always 1). 

        Options:

          T             Temperature in [K]. If you do not specify it, the current temperature
                        is used.
        
          P             Pressure in [bar]. If you do not specify it, the current pressure
                        is used.
        
          return_res    If True, then in addition to the solution, also the dict called res
                        obtained from minimize() is returned. Useful for debugging.

          return_unscaled   (default=False) If True, then instead of returning the scaled Y,
                            which are the moles of the scaled phases (e.g. Mg(2/3)Si(1/3)O(4/3)
                            for Mg2SiO4 in the system of SiO2,MgO, or H(2/3)O(1/3) in the
                            system of H and O), return instead the unscaled Y = Y / scale,
                            meaning the true moles of Mg2SiO4 and H2O in the above examples
                            (in both examples scale==3.0). In that case, though, Y.sum()!=1.
                            But the system remains normalized to 1 mole of constitute system
                            components.
        
          options       The options passed on to the minimizer algorithm. 
        
        Returns:

          Ymin          The result of the Gibbs minimizer: The mole fractions of all phases
                        in equilibrium. Note that, by default, the mole fractions of the
                        scaled phases are given (the formula units of the phases corresponding
                        to one mole of system component, e.g. Mg(2/3)Si(1/3)O(4/3) for Mg2SiO4
                        in the system of SiO2,MgO, or H(2/3)O(1/3) in the system of H and O.
                        If you set "return_unscaled" to True, then the moles of the phases
                        as they are listed in the self.Y_phase_name are returned, but they
                        then no longer obey Ymin.sum()==1. 
        """
        from scipy.optimize import minimize
        from functools import partial
        self.quantity = xbulk.sum()   # In case x.sum()!=1 we interpret x.sum() as the nr of moles in total
        self.xbulk    = xbulk/self.quantity
        self.Yinit    = Yinit/Yinit.sum()
        if options is None:
            options = {}
        if 'barrier_tol' not in options: options['barrier_tol'] = self.btol
        if 'maxiter' not in options: options['maxiter'] = self.maxiter
        self.do_reset_if_necessary(T=T,P=P)
        if self.ncryst>0:
            self.Gsol   = np.array(self.mdb['mfDfG'])
            self.xsol   = np.stack(self.mdb['x'])
            assert len(self.Gsol)==self.ncryst, 'Weird error: ncryst incorrect'
        assert len(Yinit) == self.nphases, 'Error: Yinit does not have the correct number of elements.'
        if consY:
            # One of the constraints will be on Y, the others on the composition
            cons = [{'type': 'eq', 'fun': lambda Y_offset: (Y_offset-1).sum()-1}]
            for k in range(self.ncomp-1):
                compos = partial(self.Composition,icomp=k)
                cons.append({'type': 'eq', 'fun': compos})
        else:
            # All constraints will be on the composition
            cons = []
            for k in range(self.ncomp):
                compos = partial(self.Composition,icomp=k)
                cons.append({'type': 'eq', 'fun': compos})
        bounds = []
        for i in range(self.nphases):
            bounds.append((1.,2.))
        bounds  = tuple(bounds)
        self.cons   = cons
        self.bounds = bounds
        method  = 'trust-constr'
        G       = lambda Yoff: self.GibbsEnergy(Yoff)
        J       = lambda Yoff: self.Jacobian(Yoff)
        H       = lambda Yoff: self.Hessian(Yoff)
        Yoff    = Yinit + 1
        res     = minimize(G,Yoff,method=method,bounds=bounds,constraints=cons,jac=J,hess=H,
                           tol=self.tol,options=options)
        Ymin    = res.x-1
        errbottom = -Ymin.min()
        if errbottom<0: errbottom=0
        if errbottom>1e-4 or not res['success']:
            #print(f'Warning: negative Y detected of magnitude {np.abs(errbottom)}')
            #print(f'The xbulk = {xbulk}. The Yinit was {Yinit}')
            success = False
            for itry in range(1,self.nrtrymax):
                Yinit = Ymin.copy()
                Yoff = Ymin.copy()
                Yoff[Yoff<0]=0.
                Yoff += 1
                res = minimize(G,Yoff,method=method,bounds=bounds,constraints=cons,jac=J,hess=H,
                               tol=self.tol,options=options)
                if res['success']:
                    Ymin    = res.x-1
                    errbottom = -Ymin.min()
                    if errbottom<0: errbottom=0
                    if errbottom>1e-4:
                        print(f'Try nr {itry} failed too with magnitude {np.abs(errbottom)}')
                    else:
                        success=True
                        break
                else:
                    print(f'Try nr {itry} failed too. Trying with random start.')
                    Ymin = np.random.random(Ymin.shape[-1])
                    Ymin /= Ymin.sum()
            if not success:
                print(f'Error: negative Y detected of magnitude {np.abs(errbottom)}')
                print(f'The xbulk = {xbulk}. The Yinit was {Yinit}')
                if nocrash:
                    return res
                else:
                    raise ValueError('Repeated retries have not helped. Aborting.')
        if return_unscaled:
            Ymin /= np.array(self.Y_phase_scale)
        if return_res:
            return Ymin,res
        else:
            return Ymin

    def get_interpretation_of_Y(self,Y,ythreshold=0.0):
        """
        The vector Y (the result of the Gibbs minimizer) is somewhat abstract.
        The function get_interpretation_of_Y() returns a more human-readable
        version of Y.

        Arguments:

          Y           The output of find_minimum(). Important: Do not use the
                      return_unscaled in find_minimum(); the unscaling will be
                      done in get_interpretation_of_Y().

          ythreshold  The minimum value of Y to include that phase or phase
                      component.

        Returns:

          A dict with lots of information ;-)
        """
        mu      = self.dGdY(Y)  # The chemical potential
        gamma   = self.gamma(Y) # The activity coefficient
        Ytot    = Y/np.array(self.Y_phase_scale)
        Ymfrac,mtot = convert_mole_fraction_into_mass_fraction(self.Y_phase_formula,Ytot,return_also_mtot=True,factors=self.Y_phase_factor)
        Ym      = Ymfrac*mtot
        phases  = []   # The list of physical phases
        if self.ncryst>0:
            ys    = Y[:self.ncryst]
            for i,y in enumerate(ys):
                if y>ythreshold:
                    mdb = self.crystaldb.dbase
                    phs = {'Phase':         self.Y_phase_phase[i],
                           'Name':          self.Y_phase_name[i],
                           'Abbrev':        self.Y_phase_abbrev[i],
                           'Formula':       self.Y_phase_formula[i],
                           'Moles':         Ytot[i]*self.quantity,
                           'MoleFrac':      Ytot[i],
                           'MoleFracSys':   Y[i],
                           'Mass':          Ym[i]*self.quantity,
                           'MassFrac':      Ymfrac[i],
                           'Xsys':          mdb[mdb['Abbrev']==self.Y_phase_abbrev[i]].iloc[0]['x'],
                           'ChemPotMol':    mu[i]*self.Y_phase_scale[i],
                           'ChemPotMass':   mu[i]*self.Y_phase_scale[i]/self.Y_phase_molmass[i],
                           'ChemPotMolSys': mu[i],
                           }
                    phases.append(phs)
        if self.nliq>0:
            for iliq,liq in enumerate(self.liquids):
                ilq0 = self.ncryst+iliq*self.ncomp
                yl   = Y[ilq0:ilq0+self.ncomp].copy()
                if yl.sum()>ythreshold:
                    mtotph = 0.0
                    ytotph = 0.0
                    for i in range(self.ncomp):
                        ytotph += Ytot[ilq0+i]
                        mtotph += Ym[ilq0+i]
                    phscs  = []
                    Xsys   = np.zeros(self.ncomp)
                    for i in range(self.ncomp):
                        Xsys[i] = Ytot[ilq0+i]/ytotph
                        if yl[i]>ythreshold:
                            phsc = {'Name':            self.Y_phase_name[ilq0+i],
                                    'Abbrev':          self.Y_phase_abbrev[ilq0+i],
                                    'Formula':         self.Y_phase_formula[ilq0+i],
                                    'Factor':          self.Y_phase_factor[ilq0+i],
                                    'Moles':           Ytot[ilq0+i]*self.quantity,
                                    'MoleFrac':        Ytot[ilq0+i],
                                    'MoleFracSys':     Y[ilq0+i],
                                    'MoleFracInPhase': Ytot[ilq0+i]/ytotph,
                                    'Mass':            Ym[ilq0+i]*self.quantity,
                                    'MassFrac':        Ymfrac[ilq0+i],
                                    'MassFracInPhase': Ym[ilq0+i]/mtotph,
                                    'ActivCoef':       gamma[ilq0+i],
                                    'Activity':        gamma[ilq0+i]*Y[ilq0+i],
                                    'ActivityInPhase': gamma[ilq0+i]*Y[ilq0+i]/ytotph,
                                    'ChemPotMol':      mu[ilq0+i]*self.Y_phase_scale[ilq0+i],
                                    'ChemPotMass':     mu[ilq0+i]*self.Y_phase_scale[ilq0+i]/self.Y_phase_molmass[ilq0+i],
                                    'ChemPotMolSys':   mu[ilq0+i]}
                            phscs.append(phsc)
                    phs = {'Phase':    liq.name,
                           'Mass':     mtotph*self.quantity,
                           'MassFrac': mtotph/mtot,
                           'Xsys':     Xsys,
                           'PhaseComponents':phscs}
                    phases.append(phs)
        if self.nmps>0:
            for imps,mps in enumerate(self.multips):
                iy0  = self.imps[imps]
                iy1  = self.imps[imps+1]
                ny   = iy1-iy0
                ym   = Y[iy0:iy1].copy()
                if ym.sum()>ythreshold:
                    mtotph = 0.0
                    ytotph = 0.0
                    ysumph = 0.0
                    for i in range(ny):
                        ytotph += Ytot[iy0+i]
                        ysumph += Y[iy0+i]
                        mtotph += Ym[iy0+i]
                    phscs  = []
                    Xsys   = np.zeros(self.ncomp)
                    for i in range(ny):
                        if 'x' in mps.dbase.columns:
                            xps   = mps.dbase[mps.dbase['Abbrev']==self.Y_phase_abbrev[iy0+i]].iloc[0]['x']
                            Xsys += xps * Y[iy0+i]/ysumph
                        if ym[i]>ythreshold:
                            phsc = {'Name':            self.Y_phase_name[iy0+i],
                                    'Abbrev':          self.Y_phase_abbrev[iy0+i],
                                    'Formula':         self.Y_phase_formula[iy0+i],
                                    'Factor':          self.Y_phase_factor[iy0+i],
                                    'Moles':           Ytot[iy0+i]*self.quantity,
                                    'MoleFrac':        Ytot[iy0+i],
                                    'MoleFracSys':     Y[iy0+i],
                                    'MoleFracInPhase': Ytot[iy0+i]/ytotph,
                                    'Mass':            Ym[iy0+i]*self.quantity,
                                    'MassFrac':        Ymfrac[iy0+i],
                                    'MassFracInPhase': Ym[iy0+i]/mtotph,
                                    'ActivCoef':       gamma[iy0+i],
                                    'Activity':        gamma[iy0+i]*Y[ilq0+i],
                                    'ActivityInPhase': gamma[iy0+i]*Y[iy0+i]/ytotph,
                                    'ChemPotMol':      mu[i]*self.Y_phase_scale[i],
                                    'ChemPotMass':     mu[i]*self.Y_phase_scale[i]/self.Y_phase_molmass[i],
                                    'ChemPotMolSys':   mu[i]
                                    }
                            phscs.append(phsc)
                    phs = {'Phase':    self.Y_phase_phase[i],
                           'Name':     self.Y_phase_name[i],
                           'Mass':     mtotph*self.quantity,
                           'MassFrac': mtotph/mtot,
                           'PhaseComponents':phscs}
                    if 'x' in mps.dbase.columns:
                        phs['Xsys'] = Xsys
                    phases.append(phs)
        return phases

    def LiquidGfunc(self,liq,x):
        """
        Wrapper around the liq.Gfunc() to allow x vectors with x.sum() != 1, which plays
        a role in the method here. What is done is to rescale x to x.sum()==1, then pass
        it on to liq.Gfunc(), then scale it back to the correct x.sum(). It also makes
        sure that the result is a scalar, not a vector.
        """
        xsum = x.sum()
        if xsum==0:
            G = 0.
        else:
            G = liq.Gfunc(x/xsum)*xsum
            if not np.isscalar(G): G=G[0]
        return G

    def Gfunc(self,Y):
        """
        The Gibbs energy of the full system, with all crystals and liquids included. This
        is the function that is minimized.
        """
        G = 0.
        if self.ncryst>0:
            ys    = Y[:self.ncryst]
            G     = (ys*self.Gsol).sum()
        if self.nliq>0:
            for iliq,liq in enumerate(self.liquids):
                ilq0 = self.ncryst+iliq*self.ncomp
                yl   = Y[ilq0:ilq0+self.ncomp].copy()
                yl[yl<0]=0
                #yl[yl>1]=1      # Do not limit <=1 to allow derivative outside ym.sum()==1
                G   += self.LiquidGfunc(liq,yl)
        if self.nmps>0:
            for imps,mps in enumerate(self.multips):
                iy0  = self.imps[imps]
                iy1  = self.imps[imps+1]
                ym   = Y[iy0:iy1].copy()
                ym[ym<0]=0
                #ym[ym>1]=1      # Do not limit <=1 to allow derivative outside ym.sum()==1
                ysum = ym.sum()
                ym  /= (ysum+1e-90)
                G   += mps.call_Gfunc(ym) * ysum
        if np.isnan(G): breakpoint()
        return G

    def dGdY(self,Y):
        """
        The gradient (Jacobian) of the Gibbs function with respect to the Y vector.
        Returns a vector. Note that this vector is simply the chemical potential, by
        definition.
        """
        RT          = self.Rgas*self.T
        dG          = np.zeros(len(Y))
        if self.ncryst>0:
            ys          = Y[:self.ncryst]
            dG[:self.ncryst] = self.Gsol
        if self.nliq>0:
            for iliq,liq in enumerate(self.liquids):
                ilq0 = self.ncryst+iliq*self.ncomp
                yl   = Y[ilq0:ilq0+self.ncomp].copy()
                yl[yl<0]=0
                yl[yl>1]=1   # ** CHECK: SHOULD WE COMMENT THIS OUT? **
                ylsum = yl.sum()
                if ylsum>1e-40:
                    ylrel = yl/(ylsum+1e-90)
                    dG[ilq0:ilq0+self.ncomp] = liq.mu0 + RT*np.log(ylrel+1e-90) + RT*np.log(liq.gammafunc(ylrel))
        if self.nmps>0:
            for imps,mps in enumerate(self.multips):
                iy0         = self.imps[imps]
                iy1         = self.imps[imps+1]
                ym          = Y[iy0:iy1].copy()
                dG[iy0:iy1] = mps.dGdY(ym)
        return dG

    def d2GdY2_num(self,Y):
        """
        The second derivative (Hessian) of the Gibbs function with respect to the
        Y vector. This is computed numerically from the numerical derivative of
        the Jacobian dGdY. 
        """
        eps   = self.eps
        d2G   = np.zeros((len(Y),len(Y)))
        dGdY0 = self.dGdY(Y)
        for i in range(len(Y)):
            Yp     = Y.copy()
            Yp[i] += eps
            dGdYp     = self.dGdY(Yp)
            d2G[i,:]  = (dGdYp-dGdY0)/eps
        return d2G

    def gamma(self,Y):
        """
        The activity coefficients of the phase components. For the fixed-composition
        crystals they are always 1, for the liquids they follow from the gammafunc()
        functions that have to be provided in the Liquid class. For the MultiPhaseSolution
        phases they are 1 if the solution is ideal, otherwise: see MultiPhaseSolution
        class.
        """
        gam        = np.zeros(len(Y))
        if self.ncryst>0:
            gam[:self.ncryst] = 1.0
        if self.nliq>0:
            for iliq,liq in enumerate(self.liquids):
                ilq0 = self.ncryst+iliq*self.ncomp
                yl   = Y[ilq0:ilq0+self.ncomp].copy()
                yl[yl<0]=0
                yl[yl>1]=1
                ylsum = yl.sum()
                if ylsum>1e-40:
                    ylrel = yl/(ylsum+1e-90)
                    gam[ilq0:ilq0+self.ncomp] = liq.gammafunc(ylrel)
        if self.nmps>0:
            for imps,mps in enumerate(self.multips):
                iy0         = self.imps[imps]
                iy1         = self.imps[imps+1]
                ym          = Y[iy0:iy1].copy()
                gam[iy0:iy1] = mps.gamma(ym)
        return gam

    def GibbsEnergy(self,Y_offset):
        Y = Y_offset - 1
        return self.Gfunc(Y)
    
    def Jacobian(self,Y_offset):
        Y = Y_offset - 1
        return self.dGdY(Y)

    def Hessian(self,Y_offset):
        Y = Y_offset - 1
        return self.d2GdY2_num(Y)

    def Composition(self,Y_offset,icomp=9999):
        """
        The condition function, for a given system component, that the
        total molar fraction of that system component over all the phases
        equals the bulk molar fraction (bulk composition).
        """
        if hasattr(self,'debug') or icomp==9999: breakpoint()
        Y  = Y_offset - 1
        xcomp = 0.
        if self.ncryst>0:
            xcomp += (Y[:self.ncryst]*self.xsol[:,icomp]).sum()
        if self.nliq>0:
            for iliq,liq in enumerate(self.liquids):
                ilq0   = self.ncryst+iliq*self.ncomp
                xcomp += Y[ilq0+icomp]
        if self.nmps>0:
            for imps,mps in enumerate(self.multips):
                iy0    = self.imps[imps]
                iy1    = self.imps[imps+1]
                xcomp += (Y[iy0:iy1]*np.stack(mps.dbase['x'])[:,icomp]).sum(axis=0)
        return xcomp - self.xbulk[icomp]

    def do_reset_if_necessary(self,T=None,P=None):
        reset = False
        if T is not None:
            if T!=self.T:
                self.T = T
                reset = True
        if P is not None:
            if P!=self.P:
                self.P = P
                reset = True
        if reset:
            if hasattr(self,'crystaldb'):
                self.crystaldb.reset(self.T,self.P)
            if hasattr(self,'liquids'):
                for liq in self.liquids:
                    liq.reset(self.T,self.P)
                    self.compute_liquid_mu0(liq)
            if hasattr(self,'multips'):
                for mps in self.multips:
                    mps.reset(self.T,self.P)

    def compute_liquid_mu0(self,liq):
        assert len(liq.components)==self.ncomp, 'Error: Nr of components of liquid incorrect'
        liq.mu0  = np.zeros(self.ncomp)
        for icomp in range(self.ncomp):
            xl             = np.zeros(self.ncomp)
            xl[icomp]      = 1.
            liq.mu0[icomp] = self.LiquidGfunc(liq,xl)

    def get_G_for_each_phase(self,Y,ythreshold=0):
        """
        The Gibbs energy of each phase separately. Useful for post-processing.
        Note: Scales linearly with Y.sum().

        Arguments:

          Y           The output of find_minimum(). Important: Do not use the
                      return_unscaled in find_minimum().

          ythreshold  The minimum value of Y to include that phase or phase
                      component.
        """
        Gi = {}
        if self.ncryst>0:
            for i in range(self.ncryst):
                ys    = Y[i]
                if ys>ythreshold:
                    Gi[self.Y_phase_abbrev[i]] = ys*self.Gsol[i]
        if self.nliq>0:
            for iliq,liq in enumerate(self.liquids):
                ilq0 = self.ncryst+iliq*self.ncomp
                yl   = Y[ilq0:ilq0+self.ncomp].copy()
                yl[yl<0]=0
                if yl.sum()>ythreshold:
                    Gi[liq.name] = self.LiquidGfunc(liq,yl)
        if self.nmps>0:
            for imps,mps in enumerate(self.multips):
                iy0  = self.imps[imps]
                iy1  = self.imps[imps+1]
                ym   = Y[iy0:iy1].copy()
                ym[ym<0]=0
                #ym[ym>1]=1      # Do not limit <=1 to allow derivative outside ym.sum()==1
                ysum = ym.sum()
                ym  /= (ysum+1e-90)
                Gi[mps.name] = mps.call_Gfunc(ym) * ysum
        return Gi

    def get_total_enthalpy(self,Y):
        """
        Get the total enthalpy H for the given Y. This can be useful for
        computing the amount of energy required to heat the material up.
        Note: Scales linearly with Y.sum().

        *** WORK IN PROGRESS AS OF 2026-01-11 ***
        
        """
        if self.ncryst>0:
            assert 'DfH' in self.crystaldb.dbase.columns, 'Error in get_total_enthalpy: DfH column not in crystal database.'
        if self.nliq>0:
            for iliq,liq in enumerate(self.liquids):
                assert liq.Hfunc is not None, 'Error in get_total_enthalpy: Liquid class does not have function Hfunc().'
        if self.nmps>0:
            for imps,mps in enumerate(self.multips):
                assert 'DfH' in mps.dbase.columns, 'Error in get_total_enthalpy: DfH column not in MultiPhaseSolution database.'
        H = 0.
        if self.ncryst>0:
            ys    = Y[:self.ncryst]
            Hsol  = np.array(self.mdb['DfH']/self.mdb['moles'])
            H     = (ys*Hsol).sum()
        if self.nliq>0:
            for iliq,liq in enumerate(self.liquids):
                ilq0 = self.ncryst+iliq*self.ncomp
                yl   = Y[ilq0:ilq0+self.ncomp].copy()
                yl[yl<0]=0
                H   += liq.Hfunc(np.stack([yl]))[0]
        if self.nmps>0:
            raise ValueError('For now, get_total_enthalpy does not work for MultiPhaseSolution class.')
        if np.isnan(H): breakpoint()
        return H
        
    def make_phase_name_list_for_Y_vector(self,fullname=False):
        # Create the list of component names, so that the resulting Y
        # vector is easier to interpret.
        if fullname:
            col = 'Name'
        else:
            col = 'Abbrev'
        self.Y_phase_phase   = []  # The physical phase (e.g. 'olivine', 'liquid' or 'gas')
        self.Y_phase_name    = []  # The name of the phase (or part of the phase) in Y
        self.Y_phase_abbrev  = []  # The abbreviation of the phases in Y
        self.Y_phase_formula = []  # The chemical formula
        self.Y_phase_scale   = []  # The scaling (>=1) of each phase such that phase_scaled = phase / Y_scale, so nmoles_scaled = nmoles * Y_scale
        self.Y_phase_factor  = []  # The factor to scale the nmoles of the phase before computing entropy of mixing
        self.Y_phase_molmass = []  # The mass (in units of gram) of 1 mole of this phase (factor*formula)
        if self.ncryst>0:
            for isol in range(self.ncryst):
                self.Y_phase_phase.append(self.crystaldb.dbase[col].iloc[isol])
                self.Y_phase_name.append(self.crystaldb.dbase['Name'].iloc[isol])
                self.Y_phase_abbrev.append(self.crystaldb.dbase['Abbrev'].iloc[isol])
                formula = self.crystaldb.dbase['Formula'].iloc[isol]
                self.Y_phase_formula.append(formula)
                self.Y_phase_scale.append(self.crystaldb.dbase['moles'].iloc[isol])
                if 'Factor' in self.crystaldb.dbase.columns:
                    # Note that this factor must already be included in moles above
                    factor = self.crystaldb.dbase['Factor'].iloc[isol]
                else:
                    factor = 1.0
                self.Y_phase_factor.append(factor)
                mol,mass,charge = dissect_molecule(formula)
                self.Y_phase_molmass.append(mass*factor)
        if self.nliq>0:
            for iliq,liq in enumerate(self.liquids):
                ilq0 = self.ncryst+iliq*self.ncomp
                for icomp in range(self.ncomp):
                    self.Y_phase_phase.append(liq.name)
                    self.Y_phase_name.append(liq.name+'_'+self.components[icomp])
                    self.Y_phase_abbrev.append(liq.name+'_'+self.components[icomp])
                    self.Y_phase_formula.append(self.components[icomp])
                    self.Y_phase_scale.append(1.0)
                    factor = self.factors[icomp]
                    self.Y_phase_factor.append(factor)
                    mol,mass,charge = dissect_molecule(self.components[icomp])
                    self.Y_phase_molmass.append(mass*factor)
        if self.nmps>0:
            for imps,mps in enumerate(self.multips):
                name = mps.name
                if len(name)>0 and name[-1]!='_': name=name+'_'
                species = np.array(mps.dbase['Formula'])
                for iphase in range(mps.nsubphases):
                    self.Y_phase_phase.append(name)
                    self.Y_phase_name.append(name+species[iphase])
                    self.Y_phase_abbrev.append(name+species[iphase])
                    self.Y_phase_formula.append(species[iphase])
                    self.Y_phase_scale.append(mps.dbase['moles'].iloc[iphase])
                    if 'Factor' in mps.dbase.columns:
                        # Note that this factor must already be included in moles above
                        factor = mps.dbase['Factor'].iloc[iphase]
                    else:
                        factor = 1.0
                    self.Y_phase_factor.append(factor)
                    mol,mass,charge = dissect_molecule(species[iphase])
                    self.Y_phase_molmass.append(mass*factor)


class MultiPhaseSolution(object):
    """
    As opposed to the convex hull algorithm, the Gibbs Minimizer algorithm 
    can involve an arbitrary set of phases that can be mixed (for now only
    ideal mixing). For instance, if you evaporate H2O, you have H2O vapor,
    but at very high temperatures, there can also be H2 gas, O2 gas, atomic H
    gas, etc. So even though you have formally only 1 system component (H2O)
    the gas phase can contain many different molecular species. Another
    example is Hastie & Bonnell's (1985) model of magma (as implemented in
    the code MAGMA by Fegley and Cameron (1987), as an ideal solution of
    not just the few system components, but an extensive list of pseudospecies
    (many more than the number of system components). These kinds of multi-
    phase (as in more than ncomp phases) solutions cannot be modelled by
    the convex hull algorithm (unless you go through the trouble of computing
    for each x, the most energetically favorable combination of phases for
    the "mixed-phase phase", but that almost defeats the purpose), but can
    be easily included in a Gibbs Minimizer algorithm.

    To facilitate this, the MultiPhaseSolution class allows you to specify
    a list of phases with their thermodynamic properties, very similar to
    the CrystalDatabase class of PhaseHull (see phasehull.py). But in
    contrast to the minerals of a CrystalDatabase object, the phases are
    now mixed at the molecular level, so that the Gibbs free energy gets
    an R*T*y_k*log(y_k) term of entropy for each phase k.

    At the moment only the entropy of mixing is included, not any potential
    interaction terms. At some later point, interaction terms could be
    added, for instance to allow the inclusion of non-ideal solid solutions.
    However, see the point about miscibility gaps.

    A note on the meaning of y:
    The capital letter Y is used in GibbsMinFinder as the mole fractions
    Y[0:nphases] of all the phases and "subphases" (in the case of liquids
    or MultiPhaseSolution instances), all scaled to the formula units
    equivalent to 1 mole of constitute system component. That means that
    at all times, if you decompose all phases into their system components,
    and add up all the moles of these system components, you get 1.0.
    An instance of the MultiPhaseSolution class is, physically, a single
    phase, but consists of a multitude of "subphases" that form an
    ideal solution. Within each instance of the MultiPhaseSolution class
    the mole fractions of these subphases (relative to the total amount of
    multiphasesolution) are called y (small letter), and sum up to
    y.sum()==1. The "subphases" of a multiphasesolution each have their
    place in the big Y vector, say, starting from Y-vector-index imultstart,
    we have Y[imultstart:imultstart+nsubphases] = Ysubphase*y[0:nsubphases],
    where Ysubphase is the total amount of moles of multiphasesolution. So
    both the total Y.sum()==1 and for each multiphasesolution y.sum()==1,
    and the conversion between the two requires Ysubphase.

    A note on ideal gas mixtures:
    The main application of the MultiPhaseSolution class is the gas phase.
    In some formulations (e.g. Timmermann et al. 2023) the mu0 are defined
    at pressure 1 bar, even if the system pressure p is not (necessarily)
    1 bar. The way this is corrected for is to add an extra term to the
    entropy of mixing Gibbs free energie that is proportional to log(p/1bar).
    Here, in MultiPhaseSolution (and in general in PhaseHull) the mu0
    values should be computed at the system pressure, in which case this
    extra term is not necessary (and not allowed).
    """
    def __init__(self,components,T,P,dbase,name='',resetfunc=None,debug=False):
        """
        Provide or read a database of phases to be mixed.

        Arguments:

          components   A list of system components, e.g. ['SiO2','MgO','CaO']. All
                       the phases in the dbase (see below) should be constructable
                       from these system components.

          T            Tempeature in [K]

          P            Pressure in [bar]

          dbase        Either a string containing the name of the .csv or fixed-width-format
                       file containing the database of phases to be mixed, or a
                       Pandas DataFrame of the database.
                       The database must have at least the columns:
                         "Abbrev"     The abbreviated name of the solid
                         "Formula"    The chemical formula, e.g. "Mg2SiO4"
                         "x"          The location in the phase diagram: An array x[0:ncomp]
                                      with ncomp the nr of components
                         "moles"      How many moles you get if you mix 1 mole of system
                                      components according to x to get this phase.
                                      Example: system components [SiO2,MgO], phase
                                      Mg2SiO4, then moles=3.0 and x=[(1/3),(2/3)].
                         "DfG"        The Delta_f G or Delta_a G Gibbs energy of formation
                         "mfDfG"      As DfG, but scaled to "per mole of system component"
                       Other columns can be added for the reset function (see resetfunc below).
                       Typically the DfG is computed by the reset function for
                       a given T and P.
                       A further column can be added voluntarily (default is 1.0):
                         "factor"     If 1.0, the formula unit to be used in the entropy
                                      of mixing (the R*y*ln(y) term), is the one in the
                                      "Formula" column. If it is, for instance, 2.0, then
                                      it would be the twice more massive one. Example:
                                      if you have MgSiO3 in the "Formula" column, but
                                      the unit used for the entropy is Mg2Si2O6, then you
                                      should set the number in the "factor" colum to 2.0.
                                      IMPORTANT: The "moles" column should be appropriately
                                                 adjusted. So with system components SiO2
                                                 and MgO, without factor, moles=2.0 (1
                                                 mole of SiO2 and 1 mole of MgO = 1 moles
                                                 of MgSiO3), but with factor = 2.0, the
                                                 moles=4.0 (2 moles of SiO2 and 2 mole of
                                                 MgO = 1 moles of Mg2Si2O6).

        Optional:

          name         The name of this mixed phase.

          resetfunc    A function with arguments (T,P), i.e. temperature (in Kelvin) and
                       pressure (in bar) that recomputes the Gibbs energy (in the mfDfG
                       column) of all the phases in their pure form.
        """
        self.name = name
        self.T    = T
        self.P    = P
        self.debug= debug
        if type(dbase) is str:
            if dbase[-4:]=='.csv':
                dbase = pd.read_csv(dbase)
            elif dbase[-4:]=='.fwf':
                dbase = pd.read_fwf(dbase)
            else:
                raise ValueError(f'Do not know how to read {dbase}')
        elif type(dbase)!=pd.DataFrame:
            raise ValueError(f'Error: dbase must be a pandas DataFrame')
        if "x" not in dbase.columns:
            from phasehull.phasehull_support import extract_from_mineral_database_based_on_components
            select  = extract_from_mineral_database_based_on_components(dbase,components)
            assert len(select)==len(dbase), 'Error: The database contains substances that cannot be made from the system components.'
            dbase   = select
        if 'Factor' not in dbase.columns:
            dbase['Factor'] = 1.0
        self.dbase      = dbase
        self.nsubphases = len(dbase)  # The number of phases to be mixed
        self.components = components
        self.ncomp      = len(components)
        self.reset  = resetfunc
        # For convenience: A dictionary from abbreviation to integer index
        self.index  = {index: value for value, index in enumerate(self.dbase['Abbrev'])}

    def call_reset(self,T,P):
        if self.reset is not None:
            self.T = T
            self.P = P
            self.reset(T,P)

    def call_Gfunc(self,y):
        Rgas   = 8.314  # J/mol·K
        RT     = Rgas*self.T
        y      = y/y.sum(axis=-1)    # ** IF WE USE DIFFERENT COMPOSITION CONSTRAINT: SHOULD WE COMMENT THIS OUT? **
        moles  = np.stack(self.dbase['moles'])
        factor = np.stack(self.dbase['Factor'])
        DfG    = np.stack(self.dbase['DfG'])
        ys     = y/moles                                                 # Moles of formula*factor
        ratio  = ys.sum(axis=-1)                                         # How many actual total moles of phases for 1 mole of system components?
        ys     = ys/ratio                                                # Normalize ys to ys.sum()==1
        G      = (ys * (DfG*factor)).sum(axis=-1)                        # The mu0 term. Here mfDfG is not needed, because 'moles' is already accounted for
        G     += RT * ( ys * np.log(np.abs(ys+1e-90)) ).sum(axis=-1)     # The entropy of mixing term
        G      = G * ratio                                               # Now scale back to Gibbs free energy per mole of constitute system component
        if self.debug:
            print(f'MPS: y = {y}, G = {G}, ratio = {ratio}')
        return G

    def dGdY(self,y):
        Rgas   = 8.314  # J/mol·K
        RT     = Rgas*self.T
        ym     = y/y.sum(axis=-1)   # ** IF WE USE DIFFERENT COMPOSITION CONSTRAINT: SHOULD WE COMMENT THIS OUT? **
        ym[ym<0]=0
        ym[ym>1]=1   # ** CHECK: SHOULD WE COMMENT THIS OUT? **
        moles  = np.stack(self.dbase['moles'])                           # How many moles of system component for 1 mole of this molecule*factor
        factor = np.stack(self.dbase['Factor'])                          # How many moles of database molecule for 1 mole of actual molecule (usually 1)
        DfG    = np.stack(self.dbase['DfG'])                             # Gibbs free energy per mole of system component
        ym    /= moles                                                   # Scale ym to moles of "actual molecule" to be mixed (factor*molecule of database)
        ysum   = ym.sum()                                                # Rescale to ym.sum()==1
        ym    /= (ysum+1e-90)
        if ysum>1e-40:
            mu0         = DfG * factor                                   # The mu0 of "actual molecule" to be mixed (factor*molecule of database)
            dG = ( mu0 + RT*np.log(ym+1e-90) ) / moles                   # The chemical potential of "actual molecule", backscaled such that it is again per mole of component
        else:
            dG = 0.0
        return dG

    def gamma(self,y):
        """
        For now the MultiPhaseSolution is an ideal solution, so the
        activity coefficients gamma_i must be 1.
        """
        return np.ones_like(y)
