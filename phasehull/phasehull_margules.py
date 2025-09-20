#---------------------------------------------------------------------------
#                  Part of PhaseHull, a simple python package
#                    to compute equilibrium phase diagrams
#
#                           (C) C. P. Dullemond
#                      Heidelberg University, Germany
#                                June 2025
#
# The Margules parameters are the expansion coefficients of the polynomial
# model of the interaction parameters for solid or liquid solutions of
# components. See Berman & Brown (1984) for a detailed description of
# how they work.
#
# This module provides the infrastructure for reading, symmetrizing and
# using these Margules parameters for computing the Gibbs energy excess of
# non-ideal mixing, and for the activity coefficients following from them.
#
# Note that most papers in the literature use simple regular parameters,
# i.e., only binary interactions WG[i,k], while Berman & Brown 1984 and
# Berman PhD thesis use quaternary interaction WG[i,j,k,l]. To compare
# them one should note that one can choose WG[i,j,k,l] in such a way
# that they effectively are WG[i,k]: To do this, one should choose
# for a pair of components i,k:
#
#    W[i,i,k,k] = 2*W[i,k,k,k] = 2*W[i,i,i,k] == 2*W[i,k]
#
# Conversely one could choose, for a pair of components i,k
# three independent parameters: WikM, WikL, WikR, where
#
#    WijM == W[i,k]
#
# and the L and R are zero-integral asymmetric components to the left
# and right. The W[i,j,k,l] are computed from these as:
# 
#    W[i,i,k,k]  = 2.*WikM - 1.5*(WikL+WikR)
#    W[i,k,k,k]  =    WikM + WikL
#    W[i,i,i,k]  =    WikM + WikR
#
# and conversely
#
#    W[i,k]      = ( W[i,i,k,k] + 1.5*W[i,k,k,k] + 1.5*W[i,i,i,k] ) / 5
#
# In this way one could, for instance, compare the Margules parameters
# of Berman & Brown 1984 to those of Ghiorso & Sack 1995. 
#
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

class Margules(object):
    def __init__(self,components):
        self.components = components
        self.ncomp      = len(components)
        self.Rgas       = 8.314  # J/mol·K

    def load_w(self,WH,WS=None,WV=None):
        """
        The Margules parameters for G are calculated as

          WG = WH - T*WS + P*WV

        with T in Kelvin and P in bar. 

        Each of these three contributions are tensors of
        rank equal to the order of the polynomial expansion
        of the excess Gibbs of mixing due to interaction
        between the mixed components. 
        """
        self.polyorder = len(WH.shape)
        if WS is None: WS = np.zeros_like(WH)
        if WV is None: WV = np.zeros_like(WH)
        self.WH = WH
        self.WS = WS
        self.WV = WV
        if(self.polyorder==2):
            for i in range(self.ncomp-1):
                for j in range(i,self.ncomp):
                    self.symmetrize_w(WH,[i,j])
                    self.symmetrize_w(WS,[i,j])
                    self.symmetrize_w(WV,[i,j])
        elif(self.polyorder==3):
            for i in range(self.ncomp-1):
                for j in range(i,self.ncomp):
                    for k in range(j,self.ncomp):
                        self.symmetrize_w(WH,[i,j,k])
                        self.symmetrize_w(WS,[i,j,k])
                        self.symmetrize_w(WV,[i,j,k])
        elif(self.polyorder==4):
            for i in range(self.ncomp-1):
                for j in range(i,self.ncomp):
                    for k in range(j,self.ncomp):
                        for l in range(k,self.ncomp):
                            self.symmetrize_w(WH,[i,j,k,l])
                            self.symmetrize_w(WS,[i,j,k,l])
                            self.symmetrize_w(WV,[i,j,k,l])
        else:
            raise ValueError(f'Cannot work with Margules parameters of polynomial order {self.polyorder}')

    def compute_w_gibbs(self,T,P=1):
        """
        Once the WH, WS and WV parameter tensors for the enthalpy (H), entropy (S) and volume (V)
        are read into this classe, we can compute the Margules parameter tensor for the Gibbs
        energy (WG)
        """
        return self.WH - T*self.WS + P*self.WV
        
    def reduce_margules_to_subset(self,components):
        """
        Sometimes you may wish to study a subset of the full ternary or
        quaternary (or higher-order) system, i.e., excluding one or more
        components. Or you may wish to reshuffle the order of the
        components (which comes first, second etc). Both can be done
        with this function.
    
        Arguments:
    
          components   A list of components that you wish to select
                       and you wish to reduce W to.

        """
        ncomp  = len(components)
        subset = np.zeros(ncomp,dtype=int)-1
        for k,e in enumerate(components):
            assert e in self.components, f'Error: Component {e} not in Margules.'
            for i in range(len(self.components)):
                if e==self.components[i]: subset[k]=i
        assert min(subset)>-1,'Error: Missing a component in the Margules'
        subset = list(subset)

        self.WH_orig = self.WH
        self.WS_orig = self.WS
        self.WV_orig = self.WV
        self.components_orig = self.components

        WH = self.WH
        WS = self.WS
        WV = self.WV

        if(self.polyorder==2):
            WHred = np.zeros((ncomp,ncomp))
            WSred = np.zeros((ncomp,ncomp))
            WVred = np.zeros((ncomp,ncomp))
            for i in range(ncomp):
                for j in range(ncomp):
                    s          = [subset[i],subset[j]]
                    WHred[i,j] = WH[s[0],s[1]]
                    WSred[i,j] = WS[s[0],s[1]]
                    WVred[i,j] = WV[s[0],s[1]]
        elif(self.polyorder==3):
            WHred = np.zeros((ncomp,ncomp,ncomp))
            WSred = np.zeros((ncomp,ncomp,ncomp))
            WVred = np.zeros((ncomp,ncomp,ncomp))
            for i in range(ncomp):
                for j in range(ncomp):
                    for k in range(ncomp):
                        s            = [subset[i],subset[j],subset[k]]
                        WHred[i,j,k] = WH[s[0],s[1],s[2]]
                        WSred[i,j,k] = WS[s[0],s[1],s[2]]
                        WVred[i,j,k] = WV[s[0],s[1],s[2]]
        elif(self.polyorder==4):
            WHred = np.zeros((ncomp,ncomp,ncomp,ncomp))
            WSred = np.zeros((ncomp,ncomp,ncomp,ncomp))
            WVred = np.zeros((ncomp,ncomp,ncomp,ncomp))
            for i in range(ncomp):
                for j in range(ncomp):
                    for k in range(ncomp):
                        for l in range(ncomp):
                            s              = [subset[i],subset[j],subset[k],subset[l]]
                            WHred[i,j,k,l] = WH[s[0],s[1],s[2],s[3]]
                            WSred[i,j,k,l] = WS[s[0],s[1],s[2],s[3]]
                            WVred[i,j,k,l] = WV[s[0],s[1],s[2],s[3]]
        else:
            raise ValueError(f'Cannot work with Margules parameters of polynomial order {self.polyorder}')

        self.WH         = WHred
        self.WS         = WSred
        self.WV         = WVred
        self.components = components        # Reduce the component list to the new one
        self.ncomp      = len(components)

    def compute_interaction_G(self,x,T,P=1,check=True):
        """
        Compute non-ideal contribution to the mixing Gibbs energy using
        Margules parameters. It is the Sum of the Margules parameters
        multiplied by the x values.
    
        Arguments:
    
          x          The array of molar(!) fractions. This can be a single
                     set of x values x[:] with x.sum()==1, or an array of
                     x values x[:,:] with x.sum(axis=-1)==1.
    
          T          Temperature in [Kelvin]

          P          Pressure in [bar]
    
        Returns:
    
          W*x*x...   The Gibbs excess energy according to the Margules
                     parameters multiplied by the x values. [J/mol]
    
        """
        self.ensure_x_2d(x,check=check)
        Gnonideal  = np.zeros_like(x[...,0])
        WG         = self.compute_w_gibbs(T,P=P)
        if self.polyorder==2:
            for i in range(self.ncomp-1):
                for j in range(i,self.ncomp):
                    Gnonideal += WG[i,j] * x[...,i]*x[...,j]
        elif self.polyorder==3:
            for i in range(self.ncomp-1):
                for j in range(i,self.ncomp):
                    for k in range(j,self.ncomp):
                        Gnonideal += WG[i,j,k] * x[...,i]*x[...,j]*x[...,k]
        elif self.polyorder==4:
            for i in range(self.ncomp-1):
                for j in range(i,self.ncomp):
                    for k in range(j,self.ncomp):
                        for l in range(k,self.ncomp):
                            Gnonideal += WG[i,j,k,l] * x[...,i]*x[...,j]*x[...,k]*x[...,l]
        else:
            raise ValueError(f'Cannot work with Margules parameters of dimension {self.polyorder}')
        return Gnonideal

    def compute_ideal_mixing_G(self,x,T,check=True):
        """
        Compute the ideal contribution to the mixing Gibbs energy.
    
        Arguments:
    
          x          The array of molar(!) fractions. This can be a single
                     set of x values x[:] with x.sum()==1, or an array of
                     x values x[:,:] with x.sum(axis=-1)==1.
    
          T          Temperature in [Kelvin]

        Returns:

          Gmixideal  The ideal part of the mixing G in [J/mol]

        """
        self.ensure_x_2d(x,check=check)
        Gideal = self.Rgas * T * (x*np.log(x+1e-90)).sum(axis=-1)
        return Gideal

    def get_activity_coefficients_of_components(self,x,T,P=1):
        """
        Using the Margules parameters WG and the molar abundances x (with x.sum()==1),
        compute the activities a of these liquid components. See equation 22
        of Berman & Brown (1984)

        Based on code by D. Ebel 2001.
        https://research.amnh.org/~debel/vapors1/codes/B83a/B83acts1.txt
    
        Arguments:
    
          x             The molar(!) fractions x
    
          T             Temperature in Kelvin

          P             Pressure in bar
    
        Returns:
    
          gamma         The activity coefficients such that the activities are
                        computed by a = gamma * x

        """
        assert len(x.shape)<=2, 'Error: This function can only accept a single x vector or a 1d set of x vectors.'
        x          = np.array(x)
        assert x.shape[-1]==self.ncomp, 'Error: Nr of x-dimensions unequal to nr of components'
        if len(x.shape)==2:
            nx         = x.shape[0]
            assert(np.all(np.abs(x.sum(axis=-1)-1)<1e-2)), 'Error: The X abundances do not add up to 1'
            x          = x.transpose()
            rtlngamma  = np.zeros((self.ncomp,nx))
        else:
            rtlngamma  = np.zeros(self.ncomp)
            assert(np.abs(x.sum()-1)<1e-2), 'Error: The X abundances do not add up to 1'
        pp         = 1-self.polyorder           # pp=(1-p) where p is polynomial degree
        WG         = self.compute_w_gibbs(T,P=P)
        for m in range(self.ncomp):
            for i in range(self.ncomp-1):
                for j in range(i,self.ncomp):
                    if(self.polyorder==2):
                        q = 0
                        if(m==i): q+=1
                        if(m==j): q+=1
                        rtlngamma[m] += WG[i,j] * ( (q*x[i]*x[j])/(x[m]+1e-99) + pp*x[i]*x[j] )
                    else:
                        for k in range(j,self.ncomp):
                            if(self.polyorder==3):
                                q = 0
                                if(m==i): q+=1
                                if(m==j): q+=1
                                if(m==k): q+=1
                                rtlngamma[m] += WG[i,j,k] * ( (q*x[i]*x[j]*x[k])/(x[m]+1e-99) + pp*x[i]*x[j]*x[k] )
                            else:
                                assert(self.polyorder==4), f'Cannot work with Margules parameters of dimension {self.polyorder}'
                                for l in range(k,self.ncomp):
                                    q = 0
                                    if(m==i): q+=1
                                    if(m==j): q+=1
                                    if(m==k): q+=1
                                    if(m==l): q+=1
                                    rtlngamma[m] += WG[i,j,k,l] * ( (q*x[i]*x[j]*x[k]*x[l])/(x[m]+1e-99) + pp*x[i]*x[j]*x[k]*x[l] )
        gamma = np.exp(rtlngamma / (self.Rgas*T))
        return gamma

    def symmetrize_w(self,W,indices):
        value = self.find_value_and_check_symmetry_w(W,indices)
        self.fill_w(W,indices,value)

    def find_value_and_check_symmetry_w(self,W,indices):
        perm  = list(permutations(indices))
        value = 0.0
        for p in perm:
            if W[tuple(p)]!=0.0:
                if value==0.0:
                    value = W[tuple(p)]
                else:
                    assert abs(W[tuple(p)]-value)<1e-3, f'Error: Margules tensor is asymmetric in indices {p}.'
        return value

    def fill_w(self,W,indices,value):
        perm  = list(permutations(indices))
        for p in perm:
            W[tuple(p)] = value

    def ensure_x_2d(self,x,check=True):
        if len(x.shape)==1:
            x = x.reshape([1,x.shape[-1]])
        if check:
            assert np.all(np.abs(x.sum(-1)-1)<1e-10), 'Error: x do not sum to 1'
        return x
