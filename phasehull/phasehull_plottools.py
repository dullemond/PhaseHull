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
import mpltern   # https://mpltern.readthedocs.io/en/latest/index.html
import pandas as pd
import scipy
import phasehull as ph
from copy import deepcopy
from matplotlib.backend_bases import MouseButton

def plot_binary_xG(phull,relevel=True,relevel_useliq=False,reverse=False,T=None,P=None,
                   labels=None,colors=None,ymin=None,ymax=None,ax=None,xbase=None,Gbase=None,
                   xlabel='x',coordtrans=None,Gtrans=None,compnames=None):
    """
    Standard plot of a binary system in G versus x. The liquid Gibbs is plotted
    as a dotted line where it is not stable, and solid line where it is stable. The
    fixed-composition crystals are ploted as red diamonds.

    Arguments:

      phull         Instance of the PhaseHull class, where the phase diagram has
                    been calculated.

    Optional:

      relevel       If True, then subtract a linear function G_base(x). If xbase and
                    Gbase are not set, then it will compute the G_base(0) and G_base(1)
                    from the mfDfG of the minerals at x=0 and x=1. If xbase and Gbase
                    are set, then it will use those to compute the linear G_base(x).

      xbase, Gbase  Both arrays (or lists) or 2 elements e.g. [xbase_left,xbase_right]
                    and likewise for Gbase.

      relevel_useliq   If True, then use the G_liquid(0) and G_liquid(1) to set up the
                       base G_base(x).

      reverse       Plot with opposite x direction

      T, P          Temperature and pressure (only for annotation)

      ymin, ymax    The vertical extent

      ax            An ax object to draw on.

      colors        Dict of color scheme. If not given, the standard color scheme
                    (as defined in phasehull_colors.py) is used.

      xlabel        The label on the x axis

      labels        Dict of labels to use for the different tie lines. If not set,
                    then the standard labels are used.

      coordtrans    A function to transform the x coordinate to another system.
                    Can be useful when comparing two incompatible systems with
                    each other, for example: the [SiO2,CaO] and [SiO2,SiCaO3]
                    system onto the same plot. It should be a function of the
                    SubSystem class in phasehull_subsystem.py, the function
                    convert_from_xcomp_to_xprim() or convert_from_xprim_to_xcomp().

      Gtrans        If you provide coordtrans(), you also must also provide Gtrans(),
                    the function to transform the G from the one system into the
                    other. It should be a function of the
                    SubSystem class in phasehull_subsystem.py, the function
                    convert_G_value_from_composite_to_primitive_system() or
                    convert_G_value_from_primitive_to_composite_system().
    """
    from phasehull.phasehull_colors import linecolors,fillcolors,linestyles
    assert phull.ncomponents==2, 'Error: This phase diagram is not a binary.'
    
    components = phull.components
    if compnames is None:
        compnames = components

    isel_allc  = phull.select_simplices_of_a_given_kind('allcryst')
    isel_liq   = phull.select_simplices_of_a_given_kind('liquid')
    isel_cltie = phull.select_simplices_of_a_given_kind('tieline_c1l1')
    isel_cstie = phull.select_simplices_of_a_given_kind('tieline_c1l0s1') + \
                 phull.select_simplices_of_a_given_kind('tieline_c0l1s1')
    isel_inmis = phull.select_simplices_of_a_given_kind('tieline_c0l2')
    isel_solsol= phull.select_simplices_of_a_given_kind('solsol')
    isel_sscoex= phull.select_simplices_of_a_given_kind('solsol_coexist')
    isel_ssinmi= phull.select_simplices_of_a_given_kind('solsol_inmisc')

    simplices  = deepcopy(phull.thesimplices[-1])

    if coordtrans is not None:
        assert Gtrans is not None, 'Error: If you provide coordtrans() function, you must also provide a Gtrans() function.'
        simplices['G'] = Gtrans(simplices['x'],simplices['G'])
        simplices['x'] = coordtrans(simplices['x'])

    x_liq      = []
    G_liq      = []
    for iliq,liq in enumerate(phull.liquids):
        xliq = liq.xgrid.copy()
        Gliq = liq.Ggrid.copy()
        if coordtrans is not None:
            assert Gtrans is not None, 'Error: If you provide coordtrans() function, you must also provide a Gtrans() function.'
            Gliq = Gtrans(xliq,Gliq)
            xliq = coordtrans(xliq)
        x_liq.append(xliq)
        G_liq.append(Gliq)
    x_sol      = []
    G_sol      = []
    for isol,sol in enumerate(phull.solsols):
        xsol = sol.xgrid.copy()
        Gsol = sol.Ggrid.copy()
        if coordtrans is not None:
            assert Gtrans is not None, 'Error: If you provide coordtrans() function, you must also provide a Gtrans() function.'
            Gsol = Gtrans(xsol,Gsol)
            xsol = coordtrans(xsol)
        x_sol.append(xsol)
        G_sol.append(Gsol)
    if len(phull.thepoints_cryst)>0:
        Gcryst    = phull.thepoints_cryst[-1][:,-1].copy()
        xcryst    = phull.complete_x(phull.thepoints_cryst[-1][:,:-1].copy())
        if coordtrans is not None:
            assert Gtrans is not None, 'Error: If you provide coordtrans() function, you must also provide a Gtrans() function.'
            Gcryst = Gtrans(xcryst,Gcryst)
            xcryst = coordtrans(xcryst)
        G_cryst   = Gcryst
        x_cryst   = xcryst
    ylabel     = 'G [kJ/mol]'

    if relevel:
        if Gbase is not None:
            assert xbase is not None
            phull.define_a_relevelling_plane(x=xbase,G=Gbase)
        else:
            phull.define_a_relevelling_plane(components=components,useliq=relevel_useliq)
        simplices['G'] -= phull.G_relevelling_plane(simplices['x'])
        for iliq,liq in enumerate(phull.liquids):
            G_liq[iliq] -= phull.G_relevelling_plane(x_liq[iliq][:,:-1])
        for isol,sol in enumerate(phull.solsols):
            G_sol[isol] -= phull.G_relevelling_plane(x_sol[isol][:,:-1])
        if len(phull.thepoints_cryst)>0:
            x           = phull.complete_x(phull.thepoints_cryst[-1][:,:-1])
            if coordtrans is not None:
                x = coordtrans(x)
            G_cryst    -= phull.G_relevelling_plane(x)
        ylabel          = r'$G-G_{\mathrm{base}}$ [kJ/mol]'

    if colors is None:
        colors     = linecolors
    if labels is None:
        labels     = {'allcryst':'cryst-cryst tie line','liquid':'liquid','tieline_c1l1':'cryst-liquid tie line','tieline_c0l2':'inmisc liquids','solsol':'solid solution', \
                      'tieline_c1l0s1':'tie line cryst solsol','tieline_c0l1s1':'tie line liq solsol','solsol_coexist':'coexisting solid solutions', \
                      'solsol_inmisc':'inmiscible solid solution'}
    labels     = deepcopy(labels)
    unit       = 1e3

    if not reverse:
        icmp = 0
    else:
        icmp = 1

    if ax is None:
        fig,ax = plt.subplots()
    for iliq,liq in enumerate(phull.liquids):
        plt.plot(x_liq[iliq][:,icmp],G_liq[iliq]/unit,linestyle=linestyles[iliq+1][1],color=colors['liquid'],label=liq.name)
    for isol,sol in enumerate(phull.solsols):
        plt.plot(x_sol[isol][:,icmp],G_sol[isol]/unit,linestyle=linestyles[isol+1][1],color=colors['solsol'],label=sol.name)
    if len(phull.thepoints_cryst)>0:
        plt.plot(x_cryst[:,icmp],G_cryst/unit,'D',color=colors['crystal'])
    for i in isel_allc:  plt.plot([simplices['x'][i,0,icmp],simplices['x'][i,1,icmp]],[simplices['G'][i,0]/unit,simplices['G'][i,1]/unit],'.-',color=colors['allcryst'],label=labels['allcryst']); labels['allcryst']=None
    for i in isel_liq:   plt.plot([simplices['x'][i,0,icmp],simplices['x'][i,1,icmp]],[simplices['G'][i,0]/unit,simplices['G'][i,1]/unit],color=colors['liquid'],label=labels['liquid']); labels['liquid']=None
    for i in isel_cltie: plt.plot([simplices['x'][i,0,icmp],simplices['x'][i,1,icmp]],[simplices['G'][i,0]/unit,simplices['G'][i,1]/unit],'.-',color=colors['tieline_c1l1'],label=labels['tieline_c1l1']); labels['tieline_c1l1']=None
    for i in isel_cstie: plt.plot([simplices['x'][i,0,icmp],simplices['x'][i,1,icmp]],[simplices['G'][i,0]/unit,simplices['G'][i,1]/unit],'.-',color=colors['tieline_c1l0s1'],label=labels['tieline_c1l0s1']); labels['tieline_c1l0s1']=None
    for i in isel_inmis: plt.plot([simplices['x'][i,0,icmp],simplices['x'][i,1,icmp]],[simplices['G'][i,0]/unit,simplices['G'][i,1]/unit],'.-',color=colors['tieline_c0l2'],label=labels['tieline_c0l2']); labels['tieline_c0l2']=None
    for i in isel_solsol:plt.plot([simplices['x'][i,0,icmp],simplices['x'][i,1,icmp]],[simplices['G'][i,0]/unit,simplices['G'][i,1]/unit],color=colors['solsol'],label=labels['solsol']); labels['solsol']=None
    for i in isel_sscoex:plt.plot([simplices['x'][i,0,icmp],simplices['x'][i,1,icmp]],[simplices['G'][i,0]/unit,simplices['G'][i,1]/unit],color=colors['solsol_coexist'],label=labels['solsol_coexist']); labels['solsol_coexist']=None
    for i in isel_ssinmi:plt.plot([simplices['x'][i,0,icmp],simplices['x'][i,1,icmp]],[simplices['G'][i,0]/unit,simplices['G'][i,1]/unit],color=colors['solsol_inmisc'],label=labels['solsol_inmisc']); labels['solsol_inmisc']=None
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    deltyrel = 0.03
    plt.ylim(ymin=ymin,ymax=ymax)
    xmin,xmax,ymin,ymax = plt.axis()
    delty = (ymax-ymin) * deltyrel
    #ytxt = ymin + 3*delty
    ytxt = ymin + 1.5*delty
    plt.text(0,ytxt,ph.latexify_chemical_formula(compnames[1-icmp]),ha='left')
    plt.text(1,ytxt,ph.latexify_chemical_formula(compnames[icmp]),ha='right')
    ytxt = plt.gca().get_ylim()[1] - 6
    if T is not None: plt.text(0.65,ytxt,f'T = {T:.0f} K')
    if P is not None: plt.text(0.65,ytxt-4*delty,f'P = 1 bar')
    if len(phull.thepoints_cryst)>0:
        for icr,row in phull.crystals[-1].dbase.iterrows():
            if row['stable']:
                x = row['x']
                G = row['mfDfG']
                if coordtrans is not None:
                    x = coordtrans(x)
                    G = Gtrans(x,G) - phull.G_relevelling_plane(x)
                plt.text(x[icmp],G/unit+2*delty,row['Abbrev'],ha='center',size=7)
    plt.xlim([-0.05,1.05])
    plt.legend()

    return ax

def ternary_plot_simplex(ax,x,order=[0,1,2],linecolor=None,fillcolor=None,fillalpha=1.0):
    if linecolor is None and fillcolor is None:
        print('Nothing to do. Must specify either linecolor and/or fillcolor')
        return None
    x = np.stack(x)
    assert len(x.shape)==2, 'Error in ternary_plot_simplex(): x must have 2 dimensions.'
    assert x.shape[1]==3, 'Error in ternary_plot_simplex(): x.shape[1] must be 3.'
    assert x.shape[0]==3 or x.shape[0]==4, 'Error in ternary_plot_simplex(): x.shape[0] must be 3 or 4.'
    if x.shape[0]==3:
        x = np.vstack((x,x[0,:]))
    result = []
    if linecolor is not None:
        a = ax.plot(x[:,order[0]],x[:,order[1]],x[:,order[2]],color=linecolor)
        result.append(a[0])
    if fillcolor is not None:
        a = ax.fill(x[:,order[0]],x[:,order[1]],x[:,order[2]],color=fillcolor,alpha=fillalpha)
        result.append(a[0])
    if len(result)>1:
        result = tuple(result)
    elif len(result)==0:
        result = None
    else:
        result = result[0]
    return result

def plot_ternary(phull,order=[0,1,2],stride=4,compnames=None,ax=None,limits=None,coordtrans=None):
    """
    Standard plot of a ternary system using the mpltern package. The different
    simplices have different colors, with liquid being blue, a three-crystal simplex
    being red, etc. Tie lines are shown as green. The fixed-composition crystals
    are ploted as red diamonds.

    Arguments:

      phull         Instance of the PhaseHull class, where the phase diagram has
                    been calculated.

    Optional:

      order         Which system component goes where on the triangle. Default [0,1,2].

      stride        Plot only every stride tie line, to avoid overcrowding.

      compnames     The names of the system components on the corners.

      limits        The limits of the ternary plot. Default {'t':[0,1],'l':[0,1],'r':[0,1]}.
                    See mpltern package for more details.
    
      ax            An ax object to draw on.

      coordtrans    A function to transform the x coordinate to another system.
                    Can be useful when comparing two incompatible systems with
                    each other, for example: the [SiO2,CaO] and [SiO2,SiCaO3]
                    system onto the same plot. It should be a function of the
                    SubSystem class in phasehull_subsystem.py, the function
                    convert_from_xcomp_to_xprim() or convert_from_xprim_to_xcomp().
    """
    from phasehull.phasehull_colors import linecolors,fillcolors
    T          = phull.T
    P          = phull.P
    if compnames is None:
        compnames = phull.components
    if limits is None:
        limits = {'t':[0,1],'l':[0,1],'r':[0,1]}

    # Request the binodal curve coordinates:

    xb = phull.get_x_values_of_binodal_curves(stride=1)

    # Request the tie line coordinates, with a stride

    xt = phull.get_x_values_of_tie_lines(stride=stride)

    # Request the liquidus

    #liquidus_ipts,liquidus_x = phull.get_liquidus_ipts_and_x_2d()
    #if coordtrans is not None: liquidus_x = coordtrans(liquidus_x)

    # Make a list of all simplex types that have to be plotted,
    # except the tie lines

    select  = list(set(phull.thesimplices[-1]['stype']) - set(phull.tieline_types) )
    iselect    = {}
    for sel in select:
        iselect[sel] = phull.select_simplices_of_a_given_kind(sel)

    # Plot them

    h = 0.07  # Vertical spacing of annotations
    if ax is None:
        fig = plt.figure()
        ax  = fig.add_subplot(projection="ternary")
        fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)
        ax.set_tlim(limits['t'][0],limits['t'][1])
        ax.set_llim(limits['l'][0],limits['l'][1])
        ax.set_rlim(limits['r'][0],limits['r'][1])
    # Fill the fully liquid parts
    #for x in liquidus_x:
    #    if coordtrans is not None: x = coordtrans(x)
    #    ax.fill(x[:,order[0]],x[:,order[1]],x[:,order[2]],color=fillcolors['liquid'])
    # Plot the binodal lines
    for stype in xb:
        for igroup in range(len(xb[stype])):
            x = phull.complete_x(xb[stype][igroup])
            if coordtrans is not None: x = coordtrans(x)
            ax.plot(x[:,order[0]],x[:,order[1]],x[:,order[2]],color=linecolors['binodal'])
    # Plot the triangles
    for sel in select:
        for isim in iselect[sel]:
            x = phull.thesimplices[-1]['x'][isim]
            if coordtrans is not None: x = coordtrans(x)
            x = np.vstack((x,x[0,:]))
            fillcolor = 'lightgray'
            linecolor = 'gray'
            if sel in fillcolors: fillcolor=fillcolors[sel]
            if sel in linecolors: linecolor=linecolors[sel]
            ax.fill(x[:,order[0]],x[:,order[1]],x[:,order[2]],color=fillcolor)
            ax.plot(x[:,order[0]],x[:,order[1]],x[:,order[2]],color=linecolor)
    # Plot the tie lines
    for stype in xt:
        for igroup in range(len(xt[stype])):
            for itie in range(len(xt[stype][igroup])):
                x = phull.complete_x(xt[stype][igroup][itie])
                if coordtrans is not None: x = coordtrans(x)
                ax.plot([x[0,order[0]],x[1,order[0]]],[x[0,order[1]],x[1,order[1]]],[x[0,order[2]],x[1,order[2]]],color=linecolors[stype],marker='o',ms=2,linewidth=0.5)
    # Plot the crystals
    db=phull.crystals[0].dbase
    db=db[db['stable']]
    x = np.stack(db['x'])
    if coordtrans is not None: x = coordtrans(x)
    ax.scatter(x[:,order[0]],x[:,order[1]],x[:,order[2]],s=64.0, c=linecolors['crystal'], edgecolors="k",zorder=100)
    size   = 8
    voff   = 0.06
    for i in range(len(x)):
        if x[i,order[0]]>0.7:
            off=-voff
        else:
            off=voff
        ax.text(x[i,order[0]]+off,x[i,order[1]]-off/2,x[i,order[2]]-off/2,db['Abbrev'].iloc[i],ha='center',va='center',size=size)
    ax.set_tlabel(compnames[order[0]])
    ax.set_llabel(compnames[order[1]])
    ax.set_rlabel(compnames[order[2]])
    if T is not None:
        ax.text(1,0.5,-0.5,f'T = {T:.0f} K ({T-273.15:.0f} C)',ha='left')

    return ax

def interactive_ternary(phull, Tmin=300.,Tmax=3000., nT=91, iTinit=30, Pmin=1., Pmax=None, nPr=None, iPinit=0,
                        fig=None, ax=None, stride=4, plotbutton=True, zoomboxes=None, fixedpar=None, block=False,
                        return_callback=False,compnames=None, coordtrans=None, **kwargs):
    """
    Same as plot_ternary() but now with a slider to change the temperature and/or pressure.
    Click the plotting button to recompute the ternary (this can take a bit of time).

    Arguments:

      phull         Instance of the PhaseHull class, where the phase diagram has
                    been calculated.

    Optional:

      Tmin, Tmax    The lowest and highest temperature to put on the slider.

      nT            The number of discrete temperature choices on the slider.

      iTinit        The starting temperature (as index of temperature grid)

      Pmin          The pressure or the lowest pressure of the pressure slider.

      Pmax          If set: The highest pressure on the pressure slider.
                    If not set: No pressure slider.

      nPr           The number of discrete pressure choices on the slider.

      iPinit        The starting pressure (as index of pressure grid)

      plotbutton    If False: recompute phase diagram always when a new T
                    or P is selected on the sliders. If True, recompute
                    only if button is clicked.

      zoomboxes     If set, add text boxes with the zoom-in parameters.

      block         If set, block python during widget activity. If not set,
                    detach widget from the python prompt.

      stride        Plot only every stride tie line, to avoid overcrowding.

      compnames     The names of the system components on the corners.

      fig           A figure object.
    
      ax            An ax object to draw on.

      coordtrans    A function to transform the x coordinate to another system.
                    Can be useful when comparing two incompatible systems with
                    each other, for example: the [SiO2,CaO] and [SiO2,SiCaO3]
                    system onto the same plot. It should be a function of the
                    SubSystem class in phasehull_subsystem.py, the function
                    convert_from_xcomp_to_xprim() or convert_from_xprim_to_xcomp().
    
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
    from phasehull import plot_ternary
    assert phull.ncomponents==3, f'Sorry, this is not a ternary. It has {phull.ncomponents} components, but should have 3.'
    # For the interactive ternary, we must assure that all databases have
    # a reset function.
    for i,c in enumerate(phull.crystals):
        assert c.reset is not None, f'ERROR: Crystal database {i} does not have a reset() function. So we cannot run an interactive ternary.'
    for i,l in enumerate(phull.liquids):
        assert l.reset is not None, f'ERROR: Liquid phase {i} does not have a reset() function. So we cannot run an interactive ternary.'
    for i,s in enumerate(phull.solsols):
        assert s.reset is not None, f'ERROR: Solid solution phase {i} does not have a reset() function. So we cannot run an interactive ternary.'
    # Compute spacing of plot, sliders and button
    hslider  = 0.03
    nslidrscl= 6

    # The temperature slider values
    Tgrid    = np.linspace(Tmin,Tmax,nT)
    params   = [Tgrid]
    parnames = ['T[K] = ']
    indexinit= [iTinit]

    # The pressure slider values
    if nPr is not None:
        Pgrid  = Pmin * (Pmax/Pmin)**np.linspace(0,1,nPr)
        params.append(Pgrid)
        parnames.append('P[bar] = ')
        indexinit.append(iPinit)

    indexinit = np.array(indexinit)
    
    # Compute location of sliders on the canvas
    if(len(params)>nslidrscl):
        hslider *= float(nslidrscl)/len(params)
    dyslider = hslider*(4./3.)
    xslider  = 0.3
    wslider  = 0.3
    hbutton  = 0.06
    wbutton  = 0.15
    xbutton  = 0.3
    xzoom    = 0.05
    hzoom    = 0.04
    wzoom    = 0.33
    dybutton = hbutton+0.01
    dyzoombx = hbutton+0.01
    panelbot = 0.0
    controlh = panelbot + len(params)*dyslider
    if plotbutton: controlh += dybutton
    if zoomboxes is not None: controlh += dybutton
    controltop = panelbot + controlh
    bmargin  = 0.15
    
    # select first image
    par = []
    for i in range(len(params)):
        par.append(params[i][indexinit[i]])
    
    # display phase diagram
    if ax is None:
        fig = plt.figure(figsize=(6.,7.))
        ax  = fig.add_subplot(projection="ternary")
        fig.subplots_adjust(left=0.1, right=0.9, wspace=0.3,bottom=0.3,top=0.9)
        
    #plot_ternary(phull,ax=ax,compnames=self.compnames,coordtrans=coordtrans)
    
    sliders = []
    for i in range(len(params)):
    
        # define slider
        axcolor = 'lightgoldenrodyellow'
        axs = fig.add_axes([xslider, controltop-i*dyslider, xslider+wslider, hslider], facecolor=axcolor)

        if parnames is not None:
            name = parnames[i]
        else:
            name = 'Parameter {0:d}'.format(i)
            
        slider = Slider(axs, name, 0, len(params[i]) - 1,
                    valinit=indexinit[i], valfmt='%i')
        sliders.append(slider)

    if plotbutton:
        axb = fig.add_axes([xbutton, panelbot+0.2*hbutton, xbutton+wbutton, hbutton])
        pbutton = Button(axb,'Plot')
    else:
        pbutton = None

    if zoomboxes is not None:
        tmin  = '0.0'
        lmin  = '0.0'
        rmin  = '0.0'
        if type(zoomboxes) is list:
            assert len(zoomboxes)==3, 'Error: If you set zoomboxes to a list, it must be of length 3'
            tmin = str(zoomboxes[0])
            lmin = str(zoomboxes[1])
            rmin = str(zoomboxes[2])
        axzt  = fig.add_axes([xzoom+0.0*wzoom, panelbot+1.6*hbutton, 0.8*wzoom, hzoom])
        zoomt = TextBox(axzt,'T ',tmin)
        axzl  = fig.add_axes([xzoom+1.0*wzoom, panelbot+1.6*hbutton, 0.8*wzoom, hzoom])
        zooml = TextBox(axzl,'L ',lmin)
        axzr  = fig.add_axes([xzoom+2.0*wzoom, panelbot+1.6*hbutton, 0.8*wzoom, hzoom])
        zoomr = TextBox(axzr,'R ',rmin)
    else:
        zoomt = None
        zooml = None
        zoomr = None

    class callback(object):
        def __init__(self,phull,params,sliders,ax,pbutton=None,fixedpar=None,ipar=None,stride=stride,
                     zoomt=None,zooml=None,zoomr=None,compnames=None,coordtrans=None):
            self.phull    = phull
            self.params   = params
            self.sliders  = sliders
            self.ax       = ax
            self.pbutton  = pbutton
            self.zoomt    = zoomt
            self.zooml    = zooml
            self.zoomr    = zoomr
            self.stride   = stride
            self.fixedpar = fixedpar
            self.parunits = None
            self.ax_tri   = None
            self.ax_pnt   = None
            self.ax_xst   = None
            self.ax_xen   = None
            self.ax_dmin  = 0.05
            self.ax_line  = None
            self.probe_fig= None
            self.probe_ax = None
            self.closed   = False
            self.zoom     = False
            self.compnames= compnames
            self.coordtrans=coordtrans
            self.limits   = {'t':[0,1],'l':[0,1],'r':[0,1]}
            if ipar is None:
                self.ipar = np.zeros(len(sliders),dtype=int)
            else:
                self.ipar = ipar
        def handle_close(self,event):
            self.closed   = True
        def myreadsliders(self):
            for isl in range(len(self.sliders)):
                ind = int(self.sliders[isl].val)
                self.ipar[isl]=ind
            par = []
            for i in range(len(self.ipar)):
                ip = self.ipar[i]
                value = self.params[i][ip]
                par.append(value)
                name = self.sliders[i].label.get_text()
                if '=' in name:
                    namebase = name.split('=')[0]
                    if self.parunits is not None:
                        valunit = self.parunits[i]
                    else:
                        valunit = 1.0
                    name = namebase + "= {0:10.3e}".format(value/valunit)
                    self.sliders[i].label.set_text(name)
            return par
        def myreadzoomboxes(self):
            try:
                tmin = float(self.zoomt.text)
            except:
                tmin = 0.0
            try:
                lmin = float(self.zooml.text)
            except:
                lmin = 0.0
            try:
                rmin = float(self.zoomr.text)
            except:
                rmin = 0.0
            print(tmin,lmin,rmin)
            return tmin,lmin,rmin
        def myreplot(self,par):
            T = par[0]
            self.phull.reset(T,nocompute=True)
            self.ax.clear()
            plt.draw()
            self.phull.compute()
            plot_ternary(phull,ax=ax,stride=self.stride,compnames=self.compnames,coordtrans=self.coordtrans)
            plt.draw()
        def mysupdate(self,event):
            par = self.myreadsliders()
            if self.pbutton is None: self.myreplot(par)
        def mybupdate(self,event):
            par = self.myreadsliders()
            if self.pbutton is not None: self.pbutton.label.set_text('Computing...')
            plt.pause(0.3)  # This pause is to force matplotlib to actually render the 'Computing...'
            self.myreplot(par)
            if self.zoomt is not None:
                self.myzupdate(None)
            if self.pbutton is not None: self.pbutton.label.set_text('Plot')
        def myzupdate(self,event):
            tmin,lmin,rmin = self.myreadzoomboxes()
            self.ax.set_ternary_min(tmin, lmin, rmin)
        def remove_ax_all_temporary(self):
            self.remove_ax_simplex()
            self.remove_ax_line()
            self.remove_probe_plot()
            plt.draw()
        def remove_ax_simplex(self):
            if self.ax_tri is not None:
                self.ax_tri.remove()
                self.ax_tri=None
            if self.ax_pnt is not None:
                self.ax_pnt.remove()
                self.ax_pnt=None
        def remove_ax_line(self):
            if self.ax_line is not None:
                self.ax_line.remove()
                self.ax_line=None
            self.ax_xst = None
            self.ax_xen = None
        def remove_probe_plot(self):
            pass
            #if self.probe_ax is not None:
            #    self.probe_ax.remove()
            #    self.probe_ax = None
            #if self.probe_fig is not None:
            #    plt.close(self.probe_fig)
            #    self.probe_fig = None
        def get_x_from_xy_canvas(self,xdata,ydata):
            x    = np.array([0.,0.,0.])
            x[0] = ydata
            if x[0]<1.0:
                x[1] = (1-x[0])*(1-xdata/np.sqrt(1/3)/(1-x[0]))/2
                x[2] = (1-x[0])*(1+xdata/np.sqrt(1/3)/(1-x[0]))/2
            return x
        def click_on_simplex(self,event):
            print('--------------------------------------------------------------')
            x    = self.get_x_from_xy_canvas(event.xdata,event.ydata)
            print(f'Ternary coordinates {x[0]} {x[1]} {x[2]}')
            simplex = self.phull.get_simplex_for_given_x(x)
            for col in simplex:
                print(f'{col}: {simplex[col]}')
            print(f'For the point x = [{x[0]}, {x[1]}, {x[2]}] on this simplex')
            from phasehull import lever_rule_on_simplex
            y = lever_rule_on_simplex(x,simplex['x'])
            print(f'we have the following coexisting phases:')
            for i in range(3):
                name = simplex['ptnames'][i]
                print(f'  {name:13s}: {y[i]}')
            G = (y*simplex['G']).sum()
            print(f'with mean Gibbs: {G/1e3:.3f} kJ/mol')
            # Now plot the triangle
            self.remove_ax_simplex()
            a = ternary_plot_simplex(self.ax,simplex['x'],linecolor='black')
            self.ax_tri = a
            b = self.ax.plot([x[0]],[x[1]],[x[2]],'o',color='black')[0]
            self.ax_pnt = b
            plt.draw()
        def plot_or_update_line(self):
            if self.ax_xst is not None and self.ax_xen is not None:
                if self.ax_line is None:
                    xstart = self.get_x_from_xy_canvas(self.ax_xst[0],self.ax_xst[1])
                    xend   = self.get_x_from_xy_canvas(self.ax_xen[0],self.ax_xen[1])
                    x      = np.vstack([xstart,xend])
                    self.ax_line = self.ax.plot(x[:,0],x[:,1],x[:,2],color='black')[0]
                else:
                    self.ax_line.set_data(np.array([self.ax_xst[0],self.ax_xen[0]]),np.array([self.ax_xst[1],self.ax_xen[1]]))
                plt.draw()
        def on_move(self,event):
            if self.ax_xst is not None:
                if(hasattr(event.inaxes,'taxis')):
                    self.ax_xen = (event.xdata,event.ydata)
                    self.remove_ax_simplex()
                    self.plot_or_update_line()
                    plt.draw()
        def on_click(self,event):
            if event.button is MouseButton.LEFT:
                # Check if this is a mpltern axis
                if(hasattr(event.inaxes,'taxis')):
                    self.ax_xst = (event.xdata,event.ydata)
                else:
                    self.remove_ax_all_temporary()
        def on_release(self,event):
            if event.button is MouseButton.LEFT:
                if self.ax_xst is not None and self.ax_xen is not None:
                    dist = np.sqrt((self.ax_xen[0]-self.ax_xst[0])**2+(self.ax_xen[1]-self.ax_xst[1])**2)
                else:
                    dist = 0.
                if dist<self.ax_dmin:
                    # This is a 'show simplex' event
                    # Check if this is a mpltern axis
                    if(hasattr(event.inaxes,'taxis')):
                        self.click_on_simplex(event)
                        self.remove_ax_line()                        
                    else:
                        self.remove_ax_all_temporary()
                else:
                    # This is a '1d cut' event
                    xstart = self.get_x_from_xy_canvas(self.ax_xst[0],self.ax_xst[1])
                    xend   = self.get_x_from_xy_canvas(self.ax_xen[0],self.ax_xen[1])
                    self.ax_xst = None
                    self.ax_xen = None
                    self.remove_ax_simplex()
                    self.remove_probe_plot()
                    fig,ax = plot_1d_probe(self.phull,xstart,xend,nx=200,log=False,minlog=-5,onlynonzero=True,threshold=1e-4)
                    plt.show(block=False)
                    self.probe_fig = fig
                    self.probe_ax  = ax
        #def on_keyboard(self,event):
        #    print('press', event.key)
        #    sys.stdout.flush()
        #    if event.key == 'x':
        #        visible = xl.get_visible()
        #        xl.set_visible(not visible)
        #        fig.canvas.draw()

    mcb = callback(phull,params,sliders,ax,pbutton=pbutton,zoomt=zoomt,zooml=zooml,zoomr=zoomr,
                   fixedpar=fixedpar,ipar=indexinit,compnames=compnames,coordtrans=coordtrans)
            
    mcb.mybupdate(0)
        
    if plotbutton:
        pbutton.on_clicked(mcb.mybupdate)
    if zoomboxes is not None:
        zoomt.on_submit(mcb.myzupdate)
        zooml.on_submit(mcb.myzupdate)
        zoomr.on_submit(mcb.myzupdate)
    for s in sliders:
        s.on_changed(mcb.mysupdate)
    plt.connect('button_press_event', mcb.on_click)
    plt.connect('button_release_event', mcb.on_release)
    plt.connect('motion_notify_event', mcb.on_move)

    fig._mycallback    = mcb
    
    if block:
        plt.show(block=True)

    if not return_callback:
        return fig,ax
    else:
        return fig,ax,mcb

def plot_1d_probe(phull,xstart,xend,nx=200,log=False,minlog=-5,onlynonzero=True,threshold=1e-4):
    """
    Make a 1D cut through the phase diagram and plot the minerals found at each grid
    point on the 1D cut.

    Arguments:

      phull         Instance of the PhaseHull class, where the phase diagram has
                    been calculated.

      xstart        The x-coordinates (e.g. [0.,0.5,0.5]) of the starting point
                    of the 1D cut in the phase diagram.
    
      xend          The x-coordinates (e.g. [1.,0.25,0.25]) of the ending point
                    of the 1D cut in the phase diagram.

    Options:
    
      nx            The number of grid points along the 1D cut.

      log           If True, make the y-axis (the Y-values) log

      minlog        The minimal value for the log axis. Default -5.

      onlynonzero   If True, then only plot the phases that have at least one point
                    where they are stable, otherwise ignore. If False: always plot
                    all phases.

      threshold     If onlynonzero is True, then decide if a phase is present if
                    Y for that phase > threshold.
    """
    xstart = np.array(xstart)
    xend   = np.array(xend)
    s,x,Xbig,simid = phull.cut_through_phase_diagram_1d(xstart,xend,nx=nx,ilevel=-1)
    fig,ax = plt.subplots()
    for iph in range(Xbig.shape[-1]):
        doplot = True
        if onlynonzero:
            if(Xbig[:,iph].max()<threshold): doplot = False
        if doplot:
            plt.plot(s,Xbig[:,iph],label=phull.bigX_component_names[iph])
    if log:
        plt.yscale('log')
        plt.ylim(ymin=10**minlog)
    plt.legend()
    plt.xlabel('s')
    return fig,ax
