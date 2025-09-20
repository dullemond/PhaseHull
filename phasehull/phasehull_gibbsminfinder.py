#---------------------------------------------------------------------------
#                  Part of PhaseHull, a simple python package
#                    to compute equilibrium phase diagrams
#
#                           (C) C. P. Dullemond
#                      Heidelberg University, Germany
#                                Sept 2025
#---------------------------------------------------------------------------

# This module is a classic Gibbs minimization tool, not related to
# the convex hull algorithm. It works in X-space (X with capital letter)
# which has an X_k for each candidate phase (while x small letter has an 
# x_i for each component). So X[0:nsol] is the molar fractions of the
# nsol solid crystals, where molar fraction is "moles of components".
# So X_{Mg2SiO4}==1 means 2/3 moles of MgO + 1/3 moles of SiO2 equals
# 1 moles of components equals 2/3 moles of Mg2SiO4.

import numpy as np

class GibbsMinFinder(object):
    def __init__(self,components,T,P=1.,crystaldb=None,liquids=None,eps=1e-8,
                 tol=1e-8,maxiter=100,nrtrymax=4,fullname=False):
        """
        This class is a classic Gibbs minimization tool, not related to
        the convex hull algorithm. It works in X-space (X with capital letter)
        which has an X_k for each candidate phase (while x small letter has an 
        x_i for each component). So X[0:nsol] is the molar fractions of the
        nsol solid crystals, where molar fraction is "moles of components".
        So X_{Mg2SiO4}==1 means 2/3 moles of MgO + 1/3 moles of SiO2 equals
        1 moles of components equals 2/3 moles of Mg2SiO4.

        To use it you must prepare an instance of phasehull.CrystalDatabase and/or
        one or more instances of phasehull.Liquid. For the Liquids you must
        include ensure to include a gammafunc() function, which should be
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

        Optional:

          eps            Step in X used to compute numerical derivatives,
                         used in computing the Hessian.
          tol            Tolerance used for checking convergence
          maxiter        Maximum number of iterations for finding the
                         X-location of the Gibbs minimum
          nrtrymax       Sometimes the solver returns an X value with
                         (too) negative values. Another attempt is then
                         launched. At most nrtrymax attempts are done
                         before the minimization is aborted.
          fullname       If True, then in the self.bigX_component_names
                         (the list of component names in the big X vector)
                         will be the full names, not just the abbreviations.
        
        Once this is set up, e.g. using

          GMF = GibbsMinFinder(['CaO','SiO2'],2000.,1.,crystaldb=cr,liquids=[lq])

        where cr is an instance of phasehull.CrystalDatabase and lq an
        instance of phasehull.Liquid, you can now find the solution of the
        minimum:

          xmean    = np.array([0.8,0.2])
          Xinit    = np.zeros(len(cr.dbase)+len(components))
          Xinit[0] = 1.
          Xmin     = GMF.find_minimum(xmean,Xinit)
        
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
        self.ncomp      = len(components)
        self.nsol       = 0
        self.nliq       = 0
        if crystaldb is not None:
            self.crystaldb = crystaldb
            self.mdb       = self.crystaldb.dbase
            assert 'x' in self.mdb.columns, 'Error: crystal database has no column called x'
            assert len(self.mdb['x'].iloc[0])==self.ncomp, 'Error: Nr of x components in crystal database not equal to number of components'
            self.crystaldb.reset(self.T,self.P)
            self.nsol   = len(self.mdb)
        if liquids is not None:
            self.liquids = liquids
            if type(self.liquids) is not list:
                self.liquids = list(self.liquids)
            self.nliq = len(self.liquids)
            for liq in self.liquids:
                assert liq.gammafunc is not None, 'Error: The liquids must have a function gammafunc(x) for the activity coefficient, to be able to compute the Jacobian and Hessian of G.'
                liq.reset(self.T,self.P)
                self.compute_liquid_mu0(liq)
        self.ncomp = self.nsol + self.nliq*self.ncomp
        assert self.ncomp>0, 'Error: Must have at least crystaldb or liquids'
        self.make_component_name_list_for_big_X_vector(fullname=fullname)

    def find_minimum(self,xmean,Xinit,T=None,P=None,return_res=False,options=None):
        """
        This is the actual Gibbs minimizer.

        Arguments:

          xmean         The (small) x mean molar fraction of the entire system in component
                        components. Note that xmean.sum() must be 1, and xmean must have
                        exactly len(components) elements.

          Xinit         The (big) X with which the search starts. This vector has the following
                        structure:
                        Elements 0:nsol are the molar fractions (per mole of component) of
                        the solid crystals
                        Elements nsol:nsol+ncomp are the molar fractions of the components of
                        the liquid (or if you have multiple liquids: the first one).
                        Elements nsol+ncomp:nsol+2*ncomp are the same, but for the second
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

          options       The options passed on to the minimizer algorithm. 
        
        """
        from scipy.optimize import minimize
        self.xmean = xmean
        self.Xinit = Xinit
        if options is None:
            options = {}
        if 'barrier_tol' not in options: options['barrier_tol'] = self.btol
        if 'maxiter' not in options: options['maxiter'] = self.maxiter
        self.do_reset_if_necessary(T=T,P=P)
        if self.nsol>0:
            self.Gsol   = np.array(self.mdb['mfDfG'])
            self.xsol   = np.stack(self.mdb['x'])
            assert len(self.Gsol)==self.nsol, 'Weird error: nsol incorrect'
        assert len(Xinit) == self.ncomp, 'Error: Xinit does not have the correct number of elements.'
        cons = [{'type': 'eq', 'fun': lambda xtot_offset: (xtot_offset-1).sum()-1}]
        for k in range(self.ncomp-1):
            cons.append({'type': 'eq', 'fun': lambda xtot_offset: self.Composition(xtot_offset,k)})
        bounds = []
        for i in range(self.ncomp):
            bounds.append((1.,2.))
        bounds  = tuple(bounds)
        method  = 'trust-constr'
        G       = lambda Xoff: self.GibbsEnergy(Xoff)
        J       = lambda Xoff: self.Jacobian(Xoff)
        H       = lambda Xoff: self.Hessian(Xoff)
        Xoff    = Xinit + 1
        res     = minimize(G,Xoff,method=method,bounds=bounds,constraints=cons,jac=J,hess=H,
                           tol=self.tol,options=options)
        Xmin    = res.x-1
        errbottom = -Xmin.min()
        if errbottom<0: errbottom=0
        if errbottom>1e-4:
            #print(f'Warning: negative X detected of magnitude {np.abs(errbottom)}')
            #print(f'The xmean = {xmean}. The Xinit was {Xinit}')
            success = False
            for itry in range(1,self.nrtrymax):
                Xinit = Xmin.copy()
                Xoff = Xmin.copy()
                Xoff[Xoff<0]=0.
                Xoff += 1
                res = minimize(G,Xoff,method=method,bounds=bounds,constraints=cons,jac=J,hess=H,
                               tol=self.tol,options=options)
                Xmin    = res.x-1
                errbottom = -Xmin.min()
                if errbottom<0: errbottom=0
                if errbottom>1e-4:
                    print(f'Try nr {itry} failed too with magnitude {np.abs(errbottom)}')
                else:
                    success=True
                    break
            if not success:
                print(f'Error: negative X detected of magnitude {np.abs(errbottom)}')
                print(f'The xmean = {xmean}. The Xinit was {Xinit}')
                raise ValueError('Repeated retries have not helped. Aborting.')
        if return_res:
            return Xmin,res
        else:
            return Xmin

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

    def Gfunc(self,xtot):
        """
        The Gibbs energy of the full system, with all crystals and liquids included. This
        is the function that is minimized.
        """
        G = 0.
        if self.nsol>0:
            xs    = xtot[:self.nsol]
            G     = (xs*self.Gsol).sum()
        if self.nliq>0:
            for iliq,liq in enumerate(self.liquids):
                ilq0 = self.nsol+iliq*self.ncomp
                xl   = xtot[ilq0:ilq0+self.ncomp].copy()
                xl[xl<0]=0
                xl[xl>1]=1
                G += self.LiquidGfunc(liq,xl)  # - (xl*Qliqb0).sum()
        if np.isnan(G): breakpoint()
        return G

    def dGdX(self,xtot):
        """
        The gradient (Jacobian) of the Gibbs function with respect to the big X vector.
        Returns a vector. Note that this vector is simply the chemical potential, by
        definition.
        """
        RT          = self.Rgas*self.T
        dG          = np.zeros(len(xtot))
        if self.nsol>0:
            xs          = xtot[:self.nsol]
            dG[:self.nsol] = self.Gsol
        if self.nliq>0:
            for iliq,liq in enumerate(self.liquids):
                ilq0 = self.nsol+iliq*self.ncomp
                xl   = xtot[ilq0:ilq0+self.ncomp].copy()
                xl[xl<0]=0
                xl[xl>1]=1
                xlsum = xl.sum()
                if xlsum>1e-40:
                    xlrel = xl/(xlsum+1e-90)
                    dG[ilq0:ilq0+self.ncomp] = liq.mu0 + RT*np.log(xlrel+1e-90) + RT*np.log(liq.gammafunc(xlrel))
        return dG

    def d2GdX2_num(self,xtot):
        """
        The second derivative (Hessian) of the Gibbs function with respect to the
        big X vector. This is computed numerically from the numerical derivative of
        the Jacobian dGdX. 
        """
        eps   = self.eps
        d2G   = np.zeros((len(xtot),len(xtot)))
        dGdX0 = self.dGdX(xtot)
        for i in range(len(xtot)):
            xtotp     = xtot.copy()
            xtotp[i] += eps
            dGdXp     = self.dGdX(xtotp)
            d2G[i,:]  = (dGdXp-dGdX0)/eps
        return d2G

    def GibbsEnergy(self,xtot_offset):
        xtot = xtot_offset - 1
        return self.Gfunc(xtot)
    
    def Jacobian(self,xtot_offset):
        xtot = xtot_offset - 1
        return self.dGdX(xtot)

    def Hessian(self,xtot_offset):
        xtot = xtot_offset - 1
        return self.d2GdX2_num(xtot)

    def Composition(self,xtot_offset,i):
        xtot  = xtot_offset - 1
        xcomp = 0.
        if self.nsol>0:
            xcomp += (xtot[:self.nsol]*self.xsol[:,i]).sum()
        if self.nliq>0:
            for iliq,liq in enumerate(self.liquids):
                ilq0   = self.nsol+iliq*self.ncomp
                xcomp += xtot[ilq0+i]
        return xcomp - self.xmean[i]

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

    def compute_liquid_mu0(self,liq):
        assert len(liq.components)==self.ncomp, 'Error: Nr of components of liquid incorrect'
        liq.mu0  = np.zeros(self.ncomp)
        for iend in range(self.ncomp):
            xl          = np.zeros(self.ncomp)
            xl[iend]    = 1.
            liq.mu0[iend] = self.LiquidGfunc(liq,xl)

    def make_component_name_list_for_big_X_vector(self,fullname=False):
        # Create the list of component names, so that the resulting Xbig
        # vector is easier to interpret.
        if fullname:
            col = 'Mineral'
        else:
            col = 'Abbrev'
        self.bigX_component_names = []
        if self.nsol>0:
            for isol in range(self.nsol):
                self.bigX_component_names.append(self.crystaldb.dbase[col].iloc[isol])
        if self.nliq>0:
            for iliq,liq in enumerate(self.liquids):
                ilq0 = self.nsol+iliq*self.ncomp
                for iend in range(self.ncomp):
                    self.bigX_component_names.append(liq.name+'_'+self.components[iend])
