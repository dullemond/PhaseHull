linecolors                          = {}
fillcolors                          = {}

# General
linecolors['crystal']               = 'C3'
linecolors['allcryst']              = 'C3';            fillcolors['allcryst']              = '#F0D4D5'
linecolors['liquid']                = '#ACC7DE';       fillcolors['liquid']                = '#ACC7DE'
linecolors['solsol']                = 'C1';            fillcolors['solsol']                = '#FAE6D1'
linecolors['solsol_coexist']        = 'C0';            fillcolors['solsol_coexist']        = '#ACC7DE'
linecolors['solsol_inmisc']         = 'C2';            fillcolors['solsol_inmisc']         = '#ACC7DE'

# Binaries
linecolors['tieline_c1l1']          = 'C2';            fillcolors['tieline_c1l1']          = '#B0F0B0'
linecolors['tieline_c0l2']          = 'C9';            fillcolors['tieline_c0l2']          = 'paleturquoise'
linecolors['tieline_c1l0s1']        = 'C2';            fillcolors['tieline_c1l0s1']        = '#B0F0B0'
linecolors['tieline_c0l1s1']        = 'C2';            fillcolors['tieline_c0l1s1']        = '#B0F0B0'

# Ternaries
linecolors['binodal']               = 'C0';            fillcolors['binodal']               = '#ACC7DE'
linecolors['tieline_c1l2']          = 'C2';            fillcolors['tieline_c1l2']          = '#B0F0B0'
linecolors['tieline_c0l3']          = 'C2';            fillcolors['tieline_c0l3']          = '#B0F0B0'
linecolors['cryst_2_liq_1']         = 'C4';            fillcolors['cryst_2_liq_1']         = '#E9E0F3'
linecolors['cryst_1_liq_2']         = 'C1';            fillcolors['cryst_1_liq_2']         = '#FAE6D1'
linecolors['inmisc_liquids_3phase'] = 'C2';            fillcolors['inmisc_liquids_3phase'] = '#D5D7B1'

# An extended set of line styles (useful for e.g. different solid solutions)
# Copied from https://matplotlib.org/3.3.1/gallery/lines_bars_and_markers/linestyles.html

linestyles = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot'),  # Same as '-.'
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),
     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
