import numpy as np
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u
from auger_tools import generate_RandomCatalogue, get_xibs, apply_deflection_mask
import configparser
import treecorr
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import integrate, stats

def read_config(config_file):
    """Read and parse the configuration file."""
    config = configparser.ConfigParser()
    config.read(config_file)
    params = {
        'minsep': config.getfloat('Parameters', 'minsep'),
        'maxsep': config.getfloat('Parameters', 'maxsep'),
        'nbins': config.getint('Parameters', 'nbins'),
        'nmult': config.getint('Parameters', 'nmult'),
        'nbootstrap': config.getint('Parameters', 'nbootstrap'),
        'brute': config.getboolean('Parameters', 'brute'),
        'nquant': config.getint('Parameters', 'nquant'),
        'sample': config.get('Parameters', 'sample'),
        'write': config.getboolean('Parameters', 'write'),
        'corrplot': config.getboolean('Parameters', 'corrplot'),
        'gclass': config.getint('Parameters', 'gclass'),
        'bptagn': config.getint('Parameters', 'bptagn'),
        'bin_K': config.get('Parameters', 'bin_K'),
        'def_thresh': config.getfloat('Parameters', 'def_thresh'),
        'def': config.get('Parameters', 'def'),
        'bin_type': config.get('Parameters', 'bin_type'),
        'skyplot': config.getboolean('Parameters', 'skyplot'),
        'cz_min': config.getfloat('Parameters', 'cz_min'),
        'cz_max': config.getint('Parameters', 'cz_max')
    }

    # Optional parameter: cz_max
    cz_max_raw = config['Parameters'].get('cz_max', None)
    params['cz_max'] = None if cz_max_raw in [None, '', 'None'] else float(cz_max_raw)

    return params

def load_data(sample, cz_min=1200, cz_max=None, gclass=None):
    """Load data based on the sample type."""
    if sample == '700control':
        filename_g = '../data/VLS_ang5_cz_700control_def.txt'
    elif sample == 'nocontrol':
        filename_g = '../data/2MRSxWISE_VLS_d1d5_sinAGNWISEniBPT_cz1000.txt'
    elif sample == 'all2MRS_noAGN':
        filename_g = '../data/2MRSxWISE_sinBPTAGNs.txt'
    elif sample == 'agn':
        filename_g = '../data/VLS_WISEorBPT_AGNs.txt'
    else:
        raise ValueError(f"Unknown sample type: {sample}")
    print(f'Sample file: {filename_g}')

    # Read galaxy data
    data = ascii.read(filename_g)

    # Cuts for 2MRS sample:
    if sample == 'all2MRS_noAGN':
        #data = data[data['_DEJ2000'] < 45.] # Cut declination
        data = data[data['class']!= 1] # Exclude WISE AGNs

    # Cut distance
    if cz_min < 1200: raise ValueError(f'cz_min must be at least 1200 km/s, got {cz_min}')
    if cz_max is not None and cz_max > 9400.: raise ValueError(f'cz_max must be at most 9400 km/s for completeness, got {cz_max}')
    data = data[data['cz'] > cz_min]  
    if cz_max is not None:
        data = data[data['cz'] < cz_max]

    # Read class 
    if gclass == 2:
        data = data[data['class'] == 2]
    elif gclass == 3:
        data = data[data['class'] == 3]

    return data

def skyplot(skyplotname, params, data, events_a8, ra_random, dec_random, quantiles):

    def format_axes(ax):
        """Format axes with RA in hours and Dec in degrees."""
        xticks_deg = [-120, -60, 0, 60, 120]
        xticks_rad = np.radians(xticks_deg)
        ax.set_xticks(xticks_rad)
        ax.set_xticklabels([f'{int(deg)}°' for deg in xticks_deg])
        yticks_deg = [-60, -30, 0, 30, 60]
        yticks_rad = np.radians(yticks_deg)
        ax.set_yticks(yticks_rad)
        ax.set_yticklabels([f'{deg}°' for deg in yticks_deg])
        ax.tick_params(axis='both', which='major', labelsize=12)
        #ax.tick_params(axis='both', which='minor', labelsize=8)
        ax.grid(True)
    print('Plotting sky coordinates')
    if params['nquant'] == 1:
        nrows = 1
        ncols = 1
    else:
        nrows = int(params['nquant'] / 2)
        ncols = 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, \
                            figsize=(12, 8), subplot_kw={'projection': 'aitoff'})        

    # Ensure axs is always iterable
    axs = np.array(axs).reshape(-1)

    for q, ax in zip(range(params['nquant']),axs):
        gxs_sc = SkyCoord(data[q]['_RAJ2000'],data[q]['_DEJ2000'],frame='icrs',unit='degree')

        eve_sc = SkyCoord(events_a8['RA'],events_a8['dec'],frame='icrs',unit='degree')
        ran_sc = SkyCoord(ra_random[q], dec_random[q],frame='icrs',unit='degree')

        #ax = fig.add_subplot(111, projection="aitoff")
        ax.scatter(ran_sc.ra.wrap_at(180*u.degree).to(u.rad),ran_sc.dec.to(u.rad),s=3,c='k',label='Random Data')
        ax.scatter(eve_sc.ra.wrap_at(180*u.degree).to(u.rad),eve_sc.dec.to(u.rad),s=.1,label='UHECRs')
        ax.scatter(gxs_sc.ra.wrap_at(180*u.degree).to(u.rad),gxs_sc.dec.to(u.rad),s=5,c='C03',label='Galaxies')

        ax.legend(loc=1,fontsize=10)
        ax.set_title(f'{quantiles[q]:.1f} '+r'< K_{abs} < '+f'{quantiles[q+1]:.1f}')
        ax.grid(True)
        format_axes(ax)

    plt.tight_layout()
    plt.savefig(skyplotname)

def correlation_plot(corrplotname, params, th, xi_true, xi_bs, varxi_bs, quantiles):
    print('Plotting correlations')
    fig, ax = plt.subplots()

    alpha, capsize = .2, 2
    labels = [f'{quantiles[q]:.1f}<K_abs<{quantiles[q + 1]:.1f}' for q in range(params['nquant'])]
    colors = ['C00', 'C01', 'C02', 'C03', 'C04', 'C05']

    for q in range(params['nquant']):
        var_bs = np.var(xi_bs[q], axis=0)
        ax.fill_between(th, y1=xi_true[q]+np.sqrt(var_bs), y2=xi_true[q]-np.sqrt(var_bs), color=colors[q], alpha=alpha)
        ax.plot(th, xi_true[q], c=colors[q], label=labels[q])

    ax.set_xlabel(r'$\theta$ (degrees)')
    ax.set_ylabel(r'$\omega(\theta)$')
    ax.set_xlim([params['minsep'], params['maxsep']])
    if params['bin_type']=='Log':
        ax.set_xscale('log')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.savefig(corrplotname)

def get_corrplotname(params):
    corrplotname = f'../plots/cross_treecorr_nq{params["nquant"]}_nmult{params["nmult"]}_nbs{params["nbootstrap"]}_{params["sample"]}'
    # Add class
    if params['gclass'] == 2: corrplotname+=f'class{params['gclass']}'
    elif params['gclass'] == 3: corrplotname+=f'class{params['gclass']}'
    # Add deflection
    if params['def'] == 'low': corrplotname+=f'_def{params['def']}{int(params['def_thresh'])}'
    elif params['def'] == 'high': corrplotname+=f'_def{params['def']}{int(params['def_thresh'])}'
    # Add czmax
    if params['cz_max'] is not None:
        corrplotname += f'_cz{int(params["cz_min"])}-{int(params["cz_max"])}'
    # Add format
    corrplotname += '.png'
    print(f'Save correlation plots to: {corrplotname}')

    return corrplotname

def get_skyplotname(params):
    skyplotname = f'../plots/skyplot_nq{params["nquant"]}_nmult{params["nmult"]}_nbs{params["nbootstrap"]}_{params["sample"]}'
    # Add class
    if params['gclass'] == 2: skyplotname+=f'class{params['gclass']}'
    elif params['gclass'] == 3: skyplotname+=f'class{params['gclass']}'
    # Add deflection
    if params['def'] == 'low': skyplotname+=f'_def{params['def']}{int(params['def_thresh'])}'
    elif params['def'] == 'high': skyplotname+=f'_def{params['def']}{int(params['def_thresh'])}'
    # Add czmax
    if params['cz_max'] is not None:
        skyplotname += f'_cz{int(params["cz_min"])}-{int(params["cz_max"])}'
    # Add format
    skyplotname += '.png'
    print(f'Save sky coverage plots to: {skyplotname}')

    return skyplotname

def get_filename(params):
    filename = f'../data/int{str(int(params["maxsep"]))}_K_nq{params["nquant"]}_nbs{params["nbootstrap"]}_{params["sample"]}'
    # Add class
    if params['gclass'] == 2: filename+=f'class{params['gclass']}'
    elif params['gclass'] == 3: filename+=f'class{params['gclass']}'
    # Add deflection
    if params['def'] == 'low': filename+=f'_def{params['def']}{int(params['def_thresh'])}'
    elif params['def'] == 'high': filename+=f'_def{params['def']}{int(params['def_thresh'])}'
    # Add czmax
    if params['cz_max'] is not None:
        filename += f'_cz{int(params["cz_min"])}-{int(params["cz_max"])}'
    # Add format
    filename += '.npz'  

    return filename

def get_filecorrname(params):
    filecorrname = f'../data/cross_treecorr_nq{params["nquant"]}_nmult{params["nmult"]}_nbs{params["nbootstrap"]}_{params["sample"]}'
    # Add class
    if params['gclass'] == 2: filecorrname+=f'class{params["gclass"]}'
    elif params['gclass'] == 3: filecorrname+=f'class{params["gclass"]}'
    # Add deflection
    if params['def'] == 'low': filecorrname+=f'_def{params["def"]}{int(params["def_thresh"])}'
    elif params['def'] == 'high': filecorrname+=f'_def{params["def"]}{int(params["def_thresh"])}'
    # Add czmax
    if params['cz_max'] is not None:
        filecorrname += f'_cz{int(params["cz_min"])}-{int(params["cz_max"])}'
    # Add format
    filecorrname += '.npz'

    return filecorrname

def crosscorrelations(data, events_a8, params, treecorr_config, write_corr=True):
    # Read UHECR data
    ecat = treecorr.Catalog(ra=events_a8['RA'], dec=events_a8['dec'], ra_units='deg', dec_units='deg')

    # Generate random catalogue
    #seeds = np.linspace(1000,1+params['nquant']-1,params['nquant'],dtype=int)
    ra_random = []
    dec_random = []
    for q in range(params['nquant']):
        randoms = generate_RandomCatalogue(len(data[q]['_RAJ2000']), params['nmult'], \
                                            seed=999, milkyway_mask=True, deflection=params['def'])
        ra_random.append(randoms[0])
        dec_random.append(randoms[1])

    rcat = [treecorr.Catalog(ra=ra_random[q], dec=dec_random[q],
            ra_units='deg', dec_units='deg') for q in range(params['nquant'])]

    # Calculate cross-correlations
    xi_bs, varxi_bs = [], []
    xi_true = np.zeros((params['nquant'], params['nbins']))
    for q in range(params['nquant']):
        print(f'{q + 1}/{params["nquant"]}')
        results = get_xibs(data[q], params['nbootstrap'], params['nbins'], rcat[q], ecat, treecorr_config)
        xi_bs.append(results[1])
        varxi_bs.append(results[2])
        xi_true[q] = results[0]
    th = results[3]

    if write_corr:
        filecorr = get_filecorrname(params)
        print(f'Writing cross-correlations to: {filecorr}')
        np.savez(filecorr, xi_true=xi_true, xi_bs=xi_bs, varxi_bs=varxi_bs, th=th)

    return xi_true, xi_bs, varxi_bs, th, ra_random, dec_random

def integration(xi_true, xi_bs, th, params):
    int_results = [np.zeros(params['nbootstrap']) for _ in range(params['nquant'])]
    int_mean = np.zeros(params['nquant'])
    for q in range(params['nquant']):
        int_mean[q] = integrate.trapezoid(xi_true[q] * th, x=th)
        for i in range(params['nbootstrap']):
            int_results[q][i] = integrate.trapezoid(th * xi_bs[q][i], x=th)

    #int_mean = np.array([np.mean(int_results[q]) for q in range(params['nquant'])])
    int_std = np.array([np.std(int_results[q],ddof=1) for q in range(params['nquant'])])

    return int_mean, int_std

def write_results(filename, int_mean, int_std, quantiles):
    print(f'Writing results in: {filename}')

    mean_mag = np.array([(quantiles[i]+quantiles[i+1])/2. for i in range(len(quantiles)-1)])
    ascii.write(np.column_stack([int_mean, mean_mag, int_std]), filename,
                names=['int_mean', 'meanMag', 'int_std'], overwrite=True)

def get_quantiles_K(params, gxs):

    if params['nquant'] == 1:
        # If only one quantile, return the min and max of K_abs
        quantiles = np.array([np.min(gxs['K_abs']), np.max(gxs['K_abs'])])

    # Define quantiles
    if params['bin_K']=='quantiles':
        quantiles = np.quantile(gxs['K_abs'], np.linspace(0, 1, params['nquant'] + 1))

    # Define bins
    elif params['bin_K'] == 'adhoc':
        if params['nquant']==4: quantiles = np.array([-26,-24,-23.2,-22.5,-22.])
        elif params['nquant']==3: quantiles = np.array([-26,-23.2,-22.8,-22.])
    else:
        raise ValueError(f"Unknown binning method: {params['bin_K']}")
    
    print(f'K Quantiles: {quantiles}')

    return quantiles
    
def main():
    print('Reading files')
    params = read_config('cross+int.ini')
    treecorr_config = {
        "min_sep": params['minsep'],
        "max_sep": params['maxsep'],
        "nbins": params['nbins'],
        "sep_units": 'degree',
        "bin_type": params['bin_type'],
        "brute": params['brute'],
        "metric": 'Arc'
    }

    # Read UHECR data
    events_a8 = ascii.read('../data/Auger/events_a8_lb.dat')
    eve = SkyCoord(events_a8['RA'], events_a8['dec'], frame='icrs', unit='degree')
    mask_eve = np.where(abs(eve.galactic.b) > 5. * u.degree)[0]
    events_a8 = events_a8[mask_eve]

    # Read galaxy data
    gxs = load_data(params['sample'], cz_min=params['cz_min'], cz_max=params['cz_max'], gclass=params['gclass'])

    # If deflection region is specified, select accordingly
    deflection_file = '../data/JF12_GMFdeflection_Z1_E10EeV.csv'
    if params['def'] == 'high' or params['def'] == 'low':
        gxs = gxs[apply_deflection_mask(deflection_file, gxs['_RAJ2000'], gxs['_DEJ2000'], params['def'])]
        events_a8 = events_a8[apply_deflection_mask(deflection_file, events_a8['RA'], events_a8['dec'], params['def'])]

    # Read AGN type
    if params['sample'] == 'agn':
        if params['bptagn'] == 0: gxs = gxs[gxs['BPTAGN'] == 1]
        elif params['bptagn'] == 1: gxs = gxs[gxs['BPTAGN'] == 1]

    # Define quantiles
    quantiles = get_quantiles_K(params, gxs)

    # Split sample into quantiles
    data = [gxs[(gxs['K_abs'] > quantiles[q]) & (gxs['K_abs'] < quantiles[q + 1])] for q in range(params['nquant'])]

    # Filenames
    if params['corrplot']:
        corrplotname = get_corrplotname(params)
    if params['skyplot']:
        skyplotname = get_skyplotname(params)
    if params['write']:
        filename = get_filename(params)

    # Calculations
    print('Calculating crosscorrelations')
    xi_true, xi_bs, varxi_bs, th, ra_random, dec_random = crosscorrelations(data, events_a8, params, treecorr_config)

    # Correlation plot
    if params['corrplot']:
        correlation_plot(corrplotname, params, th, xi_true, xi_bs, varxi_bs, quantiles)

    # Sky Plot
    if params['skyplot']:
        skyplot(skyplotname, params, data, events_a8, ra_random, dec_random, quantiles)

    # Integration
    print('Integration')
    int_mean, int_std = integration(xi_true, xi_bs, th, params)

    # Write results
    if params['write']:
        write_results(filename, int_mean, int_std, quantiles)

if __name__ == "__main__":
    main()