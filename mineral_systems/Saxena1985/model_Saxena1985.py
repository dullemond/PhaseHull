#---------------------------------------------------------------------------
#                  Part of PhaseHull, a simple python package
#                    to compute equilibrium phase diagrams
#
#                           (C) C. P. Dullemond
#                      Heidelberg University, Germany
#                                May 2026
#
# The Saxena, Sykes & Eriksson 1985 model of the Si, Mg, Fe, Ca Pyroxene
# solid solution, consisting of the mixture of Diopside (CaMgSi2O6),
# Enstatite (Mg2Si2O6), Ferrosilite (Fe2Si2O6) and Hedenbergite (CaFeSi2O6).
# These four endmembers form a quadrilateral in the ternary spanned by
# Ca2Si2O6 - Mg2Si2O6 - Fe2Si2O6, hence the solid solution is only valid
# for x_Ca2Si2O6<0.5. The region of validity is therefore a trapezium shape.
# This means that this is a reciprocal solid solution, in which (for a
# given chemical composition) there is a reaction
#
#   CaMgSi2O6 + FeSiO3 <--> CaFeSi2O6 + MgSiO3
#
# which affects (only) the pairing of the elements on the M1 and M2 sites.
# The Saxena model contains two phases: orthopyroxene and clinopyroxene. 
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', 1000)
import os
import phasehull
from phasehull import dissect_molecule,dissect_oxide,identify_component_minerals,Margules
from pyroxene_math import *

class Saxena85(object):
    """
    The Saxena,  Sykes & Eriksson 1985 model of the Si, Mg, Fe, Ca Pyroxene
    solid solution. This does not have a liquid phase.
    You first set up this object (see arguments below). Then you can use the following
    functions and data (use ? to read their doc strings):

      reset(T,P)           Change the fixed temperature and pressure to new values.
                           Calls reset_crystals() and reset_liquids().

      reset_crystals(T,P)  Change the fixed temperature and pressure of the crystal
                           database mdb to new values, and stores T,P..
                           You can pass this on to PhaseHull.CrystalDatabase() for
                           allowing PhaseHull to automatically reset the T and P if
                           necessary.
    
      mdb                  The Pandas database of crystal phases

    """
    def __init__(self,T=298.15,P=1,path=None,ext=''):
        if path is None: path = os.path.dirname(__file__)
        self.Rgas       = 8.314  # J/mol·K
        self.components = ['Mg2Si2O6','Fe2Si2O6','Ca2Si2O6']
        self.T          = T
        self.P          = P
        self.mdb_orig   = pd.read_fwf(os.path.join(path,'minerals'+ext+'.fwf'))
        self.mdb        = phasehull.extract_from_mineral_database_based_on_components(self.mdb_orig,self.components)
        self.mdb['moles'] = 1.
        self.w_ortho    = pd.read_fwf(os.path.join(path,'orthopyroxene_w'+ext+'.fwf'))
        self.w_clino    = pd.read_fwf(os.path.join(path,'clinopyroxene_w'+ext+'.fwf'))
        self.compids    = {'En':0,'Fs':1,'Di':2,'Hd':3}
        self.compnms    = ['En','Fs','Di','Hd']
        self.reset(T,P)

    def reset(self,T,P=1):
        self.T          = T
        self.P          = P
        self.reset_crystals(T,P)

    def reset_crystals(self,T,P=1):
        self.T          = T
        self.P          = P
        self.compute_DfG_with_mole_fraction_weighting(self.mdb,T,P)

    def get_Cp(self,mdb,mineral,T):
        mn = mdb[mdb['Abbrev']==mineral].iloc[0]
        cp = mn['a'] + 1e-2*mn['bx1e2']*T + 1e7*mn['cx1e-7']/T**2 + mn['f']/np.sqrt(T)  # Joules/mole (mole of formula unit)
        return cp
    
    def get_int_Cp_dT(self,mdb,mineral,T):
        """
        The integral_{298.15}^T c_P(T) dT
        """
        mn    = mdb[mdb['Abbrev']==mineral].iloc[0]
        T1    = 298.15
        intcp = mn['a']*(T-T1) + 0.5*1e-2*mn['bx1e2']*(T**2-T1**2) - 1e7*mn['cx1e-7']*(1/T-1/T1) + 2*mn['f']*(np.sqrt(T)-np.sqrt(T1))  # Joules*K/mole (mole of formula unit)
        return intcp
        
    def get_int_CpdivT_dT(self,mdb,mineral,T):
        """
        The integral_{298.15}^T (c_P(T)/T) dT
        """
        mn    = mdb[mdb['Abbrev']==mineral].iloc[0]
        T1    = 298.15
        intcp = mn['a']*(np.log(T)-np.log(T1)) + 1e-2*mn['bx1e2']*(T-T1) - 0.5*1e7*mn['cx1e-7']*(1/T**2-1/T1**2) - 2*mn['f']*(1/np.sqrt(T)-1/np.sqrt(T1))  # Joules*K/mole (mole of formula unit)
        return intcp

    def get_mu0_at_T(self,mdb,mineral,T,P=1.,nolambda=False):
        return self.get_mu0_at_T_P(mdb,mineral,T,P,nolambda=nolambda)

    def get_mu0_at_T_P(self,mdb,mineral,T,P,nolambda=False):
        """
        The mu_0 for this mineral at temperature T. Eq. 28 of Berman & Brown 1984.
        Note that because this is for the pure (!) substance, the meaning of mu_0
        is the same as of Delta G_f_0 (which is the formation gibbs energy per mole).
        This is because adding moles of material makes the total Gibbs energy increase
        linearly:
    
          Delta G_f(N) = Delta G_f_0 * N
    
        so that
    
                     dDelta G_f(N)
          mu_0 =def= ------------- = Delta G_f_0
                         dN
    
        Arguments:
    
          mineral          The abbreviated name of the mineral (column Abbrev in mdb)
          T                Temperature in [K]
          P                Pressure in [bar]
    
        Returns:
    
          mu0              The mu_0 == Delta G_f_0 of the mineral [J/mole]
          
        """
        mn      = mdb[mdb['Abbrev']==mineral].iloc[0]
        intCp   = self.get_int_Cp_dT(mdb,mineral,T)
        intCpT  = self.get_int_CpdivT_dT(mdb,mineral,T)
        intVol  = 0.0
        DHf0    = mn['Enthalpy']*1e3  # Note: 1e3 because it is given as kiloJoule/mol
        S0      = mn['Entropy']
        mu0     = DHf0 + intCp - T * ( S0 + intCpT ) + intVol
        return mu0

    def extract_from_mineral_database_based_on_components(self,mdb,components):
        """
        Given a list of minerals in Pandas dataframe mdb (see read_minerals_and_liquids()), select only
        those minerals that are composed of the components given in the list components. Also add
        columns of x and moles.
    
        Arguments:
    
          mdb              The mineral database (see read_minerals_and_liquids())
          components       List of the formulae of the components, e.g. ['SiO2','MgO','Al2O3'].
    
        Returns:
    
          select           A version of mdb with only the minerals that can be created
                           from the components, and a column with the x and moles values.
                           The x are the mole fractions. The moles are the nr of moles
                           of that mineral that can be made from 1 mole of components.
                           Example: with 0.333 mole of SiO2 and 0.667 mole of MgO (in
                           total 1 mole worth of components) you can create 0.333 mole of
                           Mg2SiO4.
        """
        nm     = len(mdb)
        nem    = len(components)
        select = mdb.copy()
        select['ok']    = False
        select['x']     = np.zeros((nm,nem)).tolist()
        select['moles'] = 0.
        for i,mn in select.iterrows():
            d = dissect_oxide(mn['Formula'],components=components)
            if d['complete'] and d['positive']:
                select.at[i,'ok']     = True
                select.at[i,'x']      = d['x']
                select.at[i,'moles']  = d['moles']
        select = select[select['ok']].copy().reset_index(drop=True).drop('ok',axis=1)
        return select

    def compute_DfG_with_mole_fraction_weighting(self,mdb,T,P,no_mfDfG=False):
        """
        After having removed all minerals from the mdb database that are not part of the
        component system with extract_from_mineral_database_based_on_components(mdb,components),
        and (with the same function) computed the mole fractions x, we can now compute the
        Gibbs free energies for all remaining minerals.
    
        Each pure substance has a Delta_f G(T,p), which is the Gibbs free energy [J/mole] required
        to create the substance out of its standard state constituents (usually, but not necessarily,
        the elements in their atomic form) at the given temperature T [K] and pressure p [bar].
        For simplicity we write Delta_f G as DfG.
    
        Since we are concerned with processes happening in "solar nebular conditions" (meaning the
        densities and pressures in the protoplanetary disk), where the pressure << 1 bar, and
        given that for most minerals the difference in equilibrium is negligible between 0 and
        1 bar, we (for now) omit the p-dependency, and take p=1bar as standard value.
    
        The mass-fraction-weighted version of DfG means, e.g., that with 0.333 mole of SiO2 and
        0.667 mole of MgO (in total 1 mole worth of components) you can create 0.333 mole of
        Mg2SiO4. So mfDfG=0.333*DfG for Mg2SiO4 where DfG is the Delta_f G for 1 mole of Mg2SiO4.
    
        Note: mdb must have a column 'moles' (how many moles of that mineral can we create from
              1 mole total of components). It is easiest to use the function
              extract_from_mineral_database_based_on_components(mdb,components) to automatically
              add this column. If you do not want to compute mfDfG (the Delta_f G for mdb['mole']
              amounts of moles of mineral), you get set no_mfDfG=True
    
        Arguments:
    
          mdb              The mineral database (see read_minerals_and_liquids())
          T                The temperature in [K]
    
        Returns:
    
          modifies the mdb database in-place.
        
        """
        mdb['DfG']   = 1e90   # The DfG per mole of this substance
        if not no_mfDfG:
            mdb['mfDfG'] = 1e90   # The DfG per mole of the constituent components
        for i,row in mdb.iterrows():
            DfG                = self.get_mu0_at_T_P(mdb,row['Abbrev'],T,P)
            mdb.at[i,'DfG']    = DfG
            if not no_mfDfG and 'moles' in row:
                mdb.at[i,'mfDfG']  = DfG * row['moles']

    def Gfunc_ortho(self,x):
        if type(x) is list: x=np.array(x)
        if len(x.shape)==1:
            x = np.array([x,])
        xt_En     = x[...,0]
        xt_Fs     = x[...,1]
        xt_Wo     = x[...,2]
        zmin,zmax = z_valid_range(xt_En, xt_Fs, xt_Wo)
        nz        = 100
        z_shape   = (1,) * zmax.ndim + (nz,)
        z         = zmin[...,None] + (zmax-zmin)[...,None]*np.linspace(0,1,nz).reshape(z_shape)
        xq_En, xq_Fs, xq_Di, xq_Hd = ternary_to_quad(xt_En[...,None], xt_Fs[...,None], xt_Wo[...,None], z)
        assert np.all(xq_En>-1e-15)
        assert np.all(xq_Fs>-1e-15)
        assert np.all(xq_Di>-1e-15)
        assert np.all(xq_Hd>-1e-15)
        assert np.all(xq_En<1+1e-15)
        assert np.all(xq_Fs<1+1e-15)
        assert np.all(xq_Di<1+1e-15)
        assert np.all(xq_Hd<1+1e-15)
        xq_En[xq_En<1e-99]=1e-99
        xq_Fs[xq_Fs<1e-99]=1e-99
        xq_Di[xq_Di<1e-99]=1e-99
        xq_Hd[xq_Hd<1e-99]=1e-99
        xEnFsDiHd = np.stack((xq_En, xq_Fs, xq_Di, xq_Hd))
        Go        = self.compute_G_of_pyroxene(xEnFsDiHd,'o').min(axis=-1)
        return Go

    def Gfunc_clino(self,x):
        if type(x) is list: x=np.array(x)
        if len(x.shape)==1:
            x = np.array([x,])
        xt_En     = x[...,0]
        xt_Fs     = x[...,1]
        xt_Wo     = x[...,2]
        zmin,zmax = z_valid_range(xt_En, xt_Fs, xt_Wo)
        nz        = 100
        z_shape   = (1,) * zmax.ndim + (nz,)
        z         = zmin[...,None] + (zmax-zmin)[...,None]*np.linspace(0,1,nz).reshape(z_shape)
        xq_En, xq_Fs, xq_Di, xq_Hd = ternary_to_quad(xt_En[...,None], xt_Fs[...,None], xt_Wo[...,None], z)
        assert np.all(xq_En>-1e-15)
        assert np.all(xq_Fs>-1e-15)
        assert np.all(xq_Di>-1e-15)
        assert np.all(xq_Hd>-1e-15)
        assert np.all(xq_En<1+1e-15)
        assert np.all(xq_Fs<1+1e-15)
        assert np.all(xq_Di<1+1e-15)
        assert np.all(xq_Hd<1+1e-15)
        xq_En[xq_En<1e-99]=1e-99
        xq_Fs[xq_Fs<1e-99]=1e-99
        xq_Di[xq_Di<1e-99]=1e-99
        xq_Hd[xq_Hd<1e-99]=1e-99
        xEnFsDiHd = np.stack((xq_En, xq_Fs, xq_Di, xq_Hd))
        Gc        = self.compute_G_of_pyroxene(xEnFsDiHd,'c').min(axis=-1)
        return Gc

    def compute_G_of_pyroxene(self,xEnFsDiHd,version):
        """
        This is the heart of the model. Note that in the minerals we always use
        Mg2Si2O6, CaMgSi2O6 etc are units, while in the Saxena papers the to-be-mixed
        units are half that: MgSiO3, Ca(1/2)Mg(1/2)SiO3 etc. This makes a difference
        in the entropy of mixing. Hence the /2 factors below (rescaling everything
        to SiO3), and finally the *2 factor (rescaling back to our Si2O6 convention).
        """
        if version=='o':
            wdb = self.w_ortho
        elif version=='c':
            wdb = self.w_clino
        else:
            raise ValueError('Version must either be o for ortho or c for clino')
        xEnFsDiHd = np.array(xEnFsDiHd)
        T    = self.T
        P    = self.P
        RT   = self.Rgas*T
        mdb  = self.mdb.set_index('Abbrev')
        if len(xEnFsDiHd.shape)==1:
            G    = 0
        else:
            G    = np.zeros(xEnFsDiHd.shape[1:])
        for c in self.compnms:
            x    = xEnFsDiHd[self.compids[c]]
            G   += x * ( mdb.mfDfG[version+c]/2 + RT*np.log(x+1e-90) ) # Factor /2 is because in Saxena the to-be-mixed units are SiO3 not Si2O6
        for i,row in wdb.iterrows():
            x1   = xEnFsDiHd[self.compids[row['c1'][1:]]]
            x2   = xEnFsDiHd[self.compids[row['c2'][1:]]]
            xt   = x1+x2
            x01  = x1/(xt+1e-90)
            x02  = x2/(xt+1e-90)
            W12  = 1e3 * ( row['WH'] - row['WS']*T + row['WV']*(P/1e3-1) ) / 2 # Claude helped to identify P as being in kbars
            Gex  = x01  * x02**2 * W12   # The x2 is the one to be quadratic, see Saxena Contrib Mineral Petrol (1981) 78:345-351, Eq. 2
            G   += (x1+x2)**2 * Gex
        G *= 2  # Since we use Si2O6 as unit, while the mixing is for SiO3, we rescale back to Si2O6 units
        return G
    
if __name__=='__main__':

    #T   = 1600.
    #P   = 20e3
    
    T   = 1100.
    P   = 15e3
    s   = Saxena85(T,P)

    #test = 'EnDi'
    #test = 'TernZ'
    test = 'GsurfClino'

    if test=='EnDi':

        # This is the enstatite-diopside join. The two solutions have
        # a common tangent, which are, then, the two co-existing
        # phases for intermediate compositions. 
        
        nx  = 1000
        x   = np.linspace(0,1,nx)
        xx  = np.stack([1-x,0.*x,x,0.*x])       # x=0 == En, x=2 == Di
        Go  = s.compute_G_of_pyroxene(xx,'o')
        Gc  = s.compute_G_of_pyroxene(xx,'c')
        G0  = min(Go[0],Gc[0])
        G1  = min(Go[-1],Gc[-1])
        Gb  = (1-x)*G0 + x*G1
        dGo = Go-Gb
        dGc = Gc-Gb
        
        plt.figure()
        plt.plot(x,dGo,label='ortho')
        plt.plot(x,dGc,label='clino')
        plt.xlabel('x')
        plt.ylabel('G')
        plt.text(0,-4500,'En')
        plt.text(0.97,-4500,'Di')
        plt.legend()

    elif test=='TernZ':

        # Here we choose a position on the ternary (it must be on the
        # quadrilateral), and then compute the G for all values of
        # xq_Di == z that are at the same compositional location.
        # This should be a curve that has a minimum at some value
        # of z, and this is then the equilibrium. 

        xt_En  = 0.25+0.125
        xt_Fs  = 0.25+0.125
        xt_Wo  = 1-xt_En-xt_Fs
        assert xt_Wo<=0.5, 'Error: Outside of solid solution quadrilateral'
        zmin,zmax = z_valid_range(xt_En, xt_Fs, xt_Wo)
        zran      = z_random_mixing(xt_En, xt_Fs, xt_Wo)
        nz     = 100
        z      = np.linspace(zmin,zmax,nz)
        xq_En, xq_Fs, xq_Di, xq_Hd = ternary_to_quad(xt_En, xt_Fs, xt_Wo, z)

        xtr_En, xtr_Fs, xtr_Wo = quad_to_ternary(xq_En, xq_Fs, xq_Di, xq_Hd)
        
        xx     = np.stack((xq_En, xq_Fs, xq_Di, xq_Hd))
        Go     = s.compute_G_of_pyroxene(xx,'o')
        Gc     = s.compute_G_of_pyroxene(xx,'c')

        plt.figure()
        plt.plot(z,Go,label='ortho')
        plt.plot(z,Gc,label='clino')
        plt.xlabel('z')
        plt.ylabel('G')
        plt.text(0,-4500,'En')
        plt.text(0.97,-4500,'Di')
        plt.legend()

    elif test=='GsurfClino':

        # Make a rectangle out of the quadrilateral, where x
        # is the En-Fs direction (=Di-Hd direction) and y
        # is the En-Di direction (=Fs-Hd direction). 
        
        nx    = 20
        ny    = 20
        x     = np.linspace(0,1,nx)
        y     = np.linspace(0,1,ny)
        xt_Wo = y[None,:]*0.5
        xt_En = (1-x[:,None])*(1-xt_Wo)
        xt_Fs = x[:,None]*(1-xt_Wo)
        zmin,zmax = z_valid_range(xt_En, xt_Fs, xt_Wo)
        nz    = 100
        z     = zmin[:,:,None] + (zmax-zmin)[:,:,None]*np.linspace(0,1,nz)[None,None,:]
        xq_En, xq_Fs, xq_Di, xq_Hd = ternary_to_quad(xt_En[:,:,None], xt_Fs[:,:,None], xt_Wo[:,:,None], z)
        assert np.all(xq_En>-1e-15)
        assert np.all(xq_Fs>-1e-15)
        assert np.all(xq_Di>-1e-15)
        assert np.all(xq_Hd>-1e-15)
        assert np.all(xq_En<1+1e-15)
        assert np.all(xq_Fs<1+1e-15)
        assert np.all(xq_Di<1+1e-15)
        assert np.all(xq_Hd<1+1e-15)
        xq_En[xq_En<1e-99]=1e-99
        xq_Fs[xq_Fs<1e-99]=1e-99
        xq_Di[xq_Di<1e-99]=1e-99
        xq_Hd[xq_Hd<1e-99]=1e-99
        
        xEnFsDiHd = np.stack((xq_En, xq_Fs, xq_Di, xq_Hd))

        Gofull = s.compute_G_of_pyroxene(xEnFsDiHd,'o')
        Gcfull = s.compute_G_of_pyroxene(xEnFsDiHd,'c')
        Go     = Gofull.min(axis=-1)
        Gc     = Gcfull.min(axis=-1)

        # ix    = 0
        # #ix    = -1
        # #ix    = nx//2
        # Gblo  = min(Go[ix,0], Gc[ix,0] )
        # Gbup  = min(Go[ix,-1], Gc[ix,-1])
        # Gb    = (1-y[None,:])*Gblo + y[None,:]*Gbup
        # dGo   = Go-Gb
        # dGc   = Gc-Gb
        # plt.figure()
        # plt.plot(y,dGo[ix,:],label='ortho')
        # plt.plot(y,dGc[ix,:],label='clino')
        # plt.xlabel('x')
        # plt.ylabel('G')
        # plt.text(0,-4500,'En/Fs')
        # plt.text(0.97,-4500,'Di/Hd')
        # plt.legend()
        # plt.show()

        GbEn  = min(Go[0,0], Gc[0,0] )
        GbFs  = min(Go[-1,0],Gc[-1,0])
        #GbDi  = min(Go[0,-1], Gc[0,-1])
        #GbHd  = min(Go[-1,-1], Gc[-1,-1])
        Gbmdd = min(Go[nx//2,0], Gc[nx//2,0] )
        Gbmdu = min(Go[nx//2,-1], Gc[nx//2,-1] )
        GbWo  = 2*Gbmdu-Gbmdd
        Gb    = xt_En*GbEn + xt_Fs*GbFs + xt_Wo*GbWo

        # def surface(ff,x=None,y=None,rstride=None,cstride=None,stride=None,
        #         xlabel=None,ylabel=None,zlabel=None):
        #     f = np.squeeze(ff)
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     nx = f.shape[0]
        #     ny = f.shape[1]
        #     if x is None:
        #         x = np.linspace(0,nx-1,nx)
        #     if y is None:
        #         y = np.linspace(0,ny-1,ny)
        #     if stride is not None:
        #         rstride = stride
        #         cstride = stride
        #     if rstride is None:
        #         rstride = 1
        #     if cstride is None:
        #         cstride = 1
        #     xx, yy = np.meshgrid(x, y,indexing='ij')
        #     ax.plot_wireframe(xx, yy, f, rstride=rstride, cstride=cstride)
        #     if xlabel is not None: ax.set_xlabel(xlabel)
        #     if ylabel is not None: ax.set_ylabel(ylabel)
        #     if zlabel is not None: ax.set_zlabel(zlabel)
        #     return fig,ax
        # 
        # surface(Go-Gb)
        # surface(Gc-Gb)
        
        import mpltern
        xt_Wo = xt_Wo + 0*xt_En
        vmin  = min((Go-Gb).min(),(Gc-Gb).min()) / 1e3
        vmax  = max((Go-Gb).max(),(Gc-Gb).max()) / 1e3
        fig   = plt.figure()
        axo   = fig.add_subplot(projection="ternary")
        cso   = axo.tripcolor(xt_Wo.flatten(),xt_En.flatten(),xt_Fs.flatten(),(Go-Gb).flatten() / 1e3, shading='gouraud', vmin=vmin, vmax=vmax, rasterized=True,cmap='inferno')
        axo.set_tlabel('Wo')
        axo.set_llabel('En')
        axo.set_rlabel('Fs')
        axo.text(0.5,0.5,0.0,'Di ',ha='right')
        axo.text(0.5,0.0,0.5,' Hd',ha='left')
        axo.text(1,0.5,-0.5,f'T = {T:.0f} K, P = {P/1e3:.0f} kbar',ha='left')
        axo.text(1,-0.5,0.5,'$\hat G-\hat G_{\mathrm{base}}$ of ortho',ha='right')
        axo.plot([0.5,0.5],[0,0.5],[0.5,0],color='black',linewidth=0.5)
        cax = fig.add_axes([0.75, 0.55, 0.02, 0.30])
        cbar = fig.colorbar(cso, cax=cax)
        cbar.set_label("kJ/mol", fontsize=10)
        plt.savefig('fig_saxena85_quadri_ortho.pdf')

        fig   = plt.figure()
        axc   = fig.add_subplot(projection="ternary")
        csc   = axc.tripcolor(xt_Wo.flatten(),xt_En.flatten(),xt_Fs.flatten(),(Gc-Gb).flatten() / 1e3, shading='gouraud', vmin=vmin, vmax=vmax, rasterized=True,cmap='inferno')
        axc.set_tlabel('Wo')
        axc.set_llabel('En')
        axc.set_rlabel('Fs')
        axc.text(0.5,0.5,0.0,'Di ',ha='right')
        axc.text(0.5,0.0,0.5,' Hd',ha='left')
        axc.text(1,0.5,-0.5,f'T = {T:.0f} K, P = {P/1e3:.0f} kbar',ha='left')
        axc.text(1,-0.5,0.5,r'$\hat G-\hat G_{\mathrm{base}}$ of clino',ha='right')
        axc.plot([0.5,0.5],[0,0.5],[0.5,0],color='black',linewidth=0.5)
        cax = fig.add_axes([0.75, 0.55, 0.02, 0.30])
        cbar = fig.colorbar(csc, cax=cax)
        cbar.set_label("kJ/mol", fontsize=10)
        plt.savefig('fig_saxena85_quadri_clino.pdf')
        
        fig   = plt.figure()
        axs   = fig.add_subplot(projection="ternary")
        css   = axs.scatter(xt_Wo.flatten(),xt_En.flatten(),xt_Fs.flatten(),color='C1',marker='.',s=40)
        axs.set_tlabel('Wo')
        axs.set_llabel('En')
        axs.set_rlabel('Fs')
        axs.text(0.5,0.5,0.0,'Di ',ha='right')
        axs.text(0.5,0.0,0.5,' Hd',ha='left')
        ixiy = [[nx//3,2*ny//3],[2*nx//3,ny//3]]
        css   = axs.scatter(xt_Wo[tuple(ixiy)].flatten(),xt_En[tuple(ixiy)].flatten(),xt_Fs[tuple(ixiy)].flatten(),color='black')
        for i,ii in enumerate(ixiy):
            axs.text(xt_Wo[tuple(ii)],xt_En[tuple(ii)],xt_Fs[tuple(ii)],f' P{i+1}',color='black',size=15,ha='left')
        axs.text(1,0.5,-0.5,'Grid points',ha='left')
        plt.savefig('fig_saxena85_quadri_grid.pdf')

        for i,ii in enumerate(ixiy):
            plt.figure()
            plt.plot(z[ii[0],ii[1],:],Gofull[ii[0],ii[1],:],label='ortho',color='C1')
            plt.plot(z[ii[0],ii[1],:],Gcfull[ii[0],ii[1],:],label='clino',color='C0')
            izo = np.argmin(Gofull[ii[0],ii[1],:])
            izc = np.argmin(Gcfull[ii[0],ii[1],:])
            zo  = z[ii[0],ii[1],izo]
            zc  = z[ii[0],ii[1],izc]
            Gomin = Gofull[ii[0],ii[1],izo]
            Gcmin = Gcfull[ii[0],ii[1],izc]
            plt.plot([zo],[Gomin],'o',color='C1')
            plt.plot([zc],[Gcmin],'o',color='C0')
            plt.xlabel('z')
            plt.ylabel(r'$\hat G$ [J/mol]')
            plt.title(f'Point P{i+1}')
            plt.legend()
            plt.savefig(f'fig_saxena85_point_{ii[0]}_{ii[1]}.pdf')

        fig   = plt.figure()
        axc   = fig.add_subplot(projection="ternary")
        csc   = axc.tripcolor(xt_Wo.flatten(),xt_En.flatten(),xt_Fs.flatten(),(zmax-zmin).flatten(), shading='gouraud', vmin=0, vmax=(zmax-zmin).max(), rasterized=True)
        axc.set_tlabel('Wo')
        axc.set_llabel('En')
        axc.set_rlabel('Fs')
        axc.text(0.5,0.5,0.0,'Di ',ha='right')
        axc.text(0.5,0.0,0.5,' Hd',ha='left')
        axc.text(1,0.5,-0.5,r'$z_{\mathrm{max}}-z_{\mathrm{min}}$',ha='left')
        plt.savefig('fig_saxena85_quadri_zmaxzmin.pdf')
