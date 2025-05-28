import numpy as np
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u
from auger_tools import generate_RandomCatalogue, get_xibs
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
        'bin_type': config.get('Parameters', 'bin_type')
    }
    return params

def load_data(sample):
    """Load data based on the sample type."""
    if sample == '700control':
        filename_g = '../data/VLS_ang5_cz_700control_def.txt'
    elif sample == 'nocontrol':
        filename_g = '../data/2MRSxWISE_VLS_d1d5_sinAGNWISEniBPT_cz1000.txt'
    elif sample == 'agn':
        filename_g = '../data/VLS_WISEorBPT_AGNs_def.txt'
    else:
        raise ValueError(f"Unknown sample type: {sample}")
    print(f'Sample file: {filename_g}')
    return ascii.read(filename_g)

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
    gxs = load_data(params['sample'])
    gxs = gxs[gxs['cz'] > 1000.]

    # Read class 
    if params['gclass'] == 2:
        gxs = gxs[gxs['class'] == 2]
    elif params['gclass'] == 3:
        gxs = gxs[gxs['class'] == 3]

    if params['sample'] == 'agn':
        if params['bptagn'] == 0: gxs = gxs[gxs['BPTAGN'] == 1]
        elif params['bptagn'] == 1: gxs = gxs[gxs['BPTAGN'] == 1]

    # Read def
    if params['def'] == 'low':
        gxs = gxs[gxs['deflection'] <= params['def_thresh']]
    elif params['def'] == 'high':   
        gxs = gxs[gxs['deflection'] > params['def_thresh']]

    # Define quantiles
    if params['bin_K']=='quantiles':
        quantiles = np.quantile(gxs['K_abs'], np.linspace(0, 1, params['nquant'] + 1))
    # Define bins
    elif params['bin_K'] == 'adhoc':
        if params['nquant']==4: quantiles = np.array([-26,-24,-23.2,-22.5,-22.])
        elif params['nquant']==3: quantiles = np.array([-26,-23.2,-22.8,-22.])
    else:
        raise ValueError(f"Unknown binning method: {params['bin_K']}")

    # Split sample into quantiles
    data = [gxs[(gxs['K_abs'] > quantiles[q]) & (gxs['K_abs'] < quantiles[q + 1])] for q in range(params['nquant'])]

    # Determine filename for Correlations plot
    if params['corrplot']:
        corrplotname = f'../plots/cross_treecorr_nq{params["nquant"]}_nmult{params["nmult"]}_nbs{params["nbootstrap"]}_{params["sample"]}'
        # Add class
        if params['gclass'] == 2: corrplotname+=f'class{params['gclass']}'
        elif params['gclass'] == 3: corrplotname+=f'class{params['gclass']}'
        # Add deflection
        if params['def'] == 'low': corrplotname+=f'_def{params['def']}{int(params['def_thresh'])}'
        elif params['def'] == 'high': corrplotname+=f'_def{params['def']}{int(params['def_thresh'])}'
        # Add format
        corrplotname += '.png'
        print(f'Save correlation plots to: {corrplotname}')

    # Determine filename for writing results
    if params['write']:
        filename = f'../data/int{str(int(params["maxsep"]))}_K_nq{params["nquant"]}_nbs{params["nbootstrap"]}_{params["sample"]}'
        # Add class
        if params['gclass'] == 2: filename+=f'class{params['gclass']}'
        elif params['gclass'] == 3: filename+=f'class{params['gclass']}'
        # Add deflection
        if params['def'] == 'low': filename+=f'_def{params['def']}{int(params['def_thresh'])}'
        elif params['def'] == 'high': filename+=f'_def{params['def']}{int(params['def_thresh'])}'
        # Add format
        filename += '.npz'
        print(f'Save results to: {filename}')

    # Calculations
    print('Calculating crosscorrelations')
    ecat = treecorr.Catalog(ra=events_a8['RA'], dec=events_a8['dec'], ra_units='deg', dec_units='deg')
    #seeds = np.linspace(1000,1+params['nquant']-1,params['nquant'],dtype=int)
    ra_random = []
    dec_random = []
    for q in range(params['nquant']):
        #radec_random.append( np.zeros((2,params['nmult']*len(data[q]))) )
        #radec_random[q] = generate_RandomCatalogue(data[q]['_RAJ2000'], data[q]['_DEJ2000'], params['nmult'], seed=None,mask=True)
        randoms = generate_RandomCatalogue(data[q]['_RAJ2000'], data[q]['_DEJ2000'], params['nmult'], seed=None, mask=True)
        ra_random.append(randoms[0])
        dec_random.append(randoms[1])
    #print(radec_random[0][0])
    rcat = [treecorr.Catalog(ra=ra_random[q], dec=dec_random[q],
            ra_units='deg', dec_units='deg') for q in range(params['nquant'])]

    xi_bs, varxi_bs = [], []
    for q in range(params['nquant']):
        print(f'{q + 1}/{params["nquant"]}')
        results = get_xibs(data[q], params['nbootstrap'], params['nbins'], rcat[q], ecat, treecorr_config)
        xi_bs.append(results[0])
        varxi_bs.append(results[1])
    th = results[2]

    # Correlation plot
    if params['corrplot']:
        print('Plotting correlations')
        fig, ax = plt.subplots()

        alpha, capsize = .2, 2
        labels = [f'{quantiles[q]:.1f}<K_abs<{quantiles[q + 1]:.1f}' for q in range(params['nquant'])]
        colors = ['C00', 'C01', 'C02', 'C03', 'C04', 'C05']

        for q in range(params['nquant']):
            xi_bs_mean = np.mean(np.reshape(xi_bs[q],(params['nbootstrap'],len(th))),axis=0)
            xi_bs_var = np.mean(np.reshape(varxi_bs[q],(params['nbootstrap'],len(th))),axis=0)  
            ax.fill_between(th, y1=xi_bs_mean+np.sqrt(xi_bs_var), y2=xi_bs_mean-np.sqrt(xi_bs_var), color=colors[q],
                           alpha=alpha)
            ax.plot(th, xi_bs_mean, c=colors[q], label=labels[q])

        ax.set_xlabel(r'$\theta$ (degrees)')
        ax.set_ylabel(r'$\omega(\theta)$')
        ax.set_xlim([params['minsep'], params['maxsep']])
        if params['bin_type']=='Log':
            ax.set_xscale('log')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.savefig(corrplotname)

    # Integration
    print('Integration')
    int_results = [np.zeros(params['nbootstrap']) for _ in range(params['nquant'])]
    for q in range(params['nquant']):
        for i in range(params['nbootstrap']):
            int_results[q][i] = integrate.trapezoid(th * xi_bs[q][i], x=th)

    int_mean = np.array([np.mean(int_results[q]) for q in range(params['nquant'])])
    int_std = np.array([np.std(int_results[q],ddof=1) for q in range(params['nquant'])])

    # Write results
    if params['write']:
        #mean_mag = np.array([np.mean(data[q]['K_abs']) for q in range(params['nquant'])])
        mean_mag = np.array([(quantiles[i]+quantiles[i+1])/2. for i in range(len(quantiles)-1)])
        print(f'Writing results in: {filename}')
        ascii.write(np.column_stack([int_mean, mean_mag, int_std]), filename,
                    names=['int_mean', 'meanMag', 'int_std'], overwrite=True)

if __name__ == "__main__":
    main()