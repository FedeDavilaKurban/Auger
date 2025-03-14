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
        'cutoff': config.getint('Parameters', 'cutoff'),
        'sample': config.get('Parameters', 'sample'),
        'write': config.getboolean('Parameters', 'write'),
        'corrplot': config.getboolean('Parameters', 'corrplot'),
        #'ratioplot': config.getboolean('Parameters', 'ratioplot'),
        'gclass': config.getint('Parameters', 'gclass')
    }
    return params

def load_data(sample):
    """Load data based on the sample type."""
    if sample == 'passivecrop':
        filename_g = '../data/VLS/2MRSxWISE_VLS_passivecrop.txt'
    elif sample == 'sinAGNWISE':
        filename_g = '../data/2MRSxWISE_VLS_d1d5_sinAGNWISE.txt'
    elif sample == 'sinAGNWISEniBPT':
        filename_g = '../data/2MRSxWISE_VLS_d1d5_sinAGNWISEniBPT.txt'
    elif sample == 'control':
        filename_g = '../data/2MRSxWISE_VLS_d1d5_sinAGNWISEniBPT_control_SF_passive_cz_Kabs_ang5_cz1000.txt'
    else:
        filename_g = '../data/VLS/2MRSxWISE_VLS.txt'
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
        "bin_type": 'Linear',
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



    # Read class for control sample
    if params['sample']=='control':
        if params['gclass'] != 0:
            if params['gclass']==2: gxs = gxs[gxs['class'] == 2]
            elif params['gclass']==3: gxs = gxs[gxs['class'] == 3]

    # Define quantiles
    quantiles = np.quantile(gxs['K_abs'], np.linspace(0, 1, params['nquant'] + 1))
    print(quantiles)
    
    # Split sample into quantiles
    data = [gxs[(gxs['K_abs'] > quantiles[q]) & (gxs['K_abs'] < quantiles[q + 1])] for q in range(params['nquant'])]

        
    if params['corrplot']:
        corrplotname = f'../plots/cross_treecorr_nq{params["nquant"]}_nmult{params["nmult"]}_nbs{params["nbootstrap"]}_{params["sample"]}'
        if params['gclass'] != 0:
            corrplotname+=f'class{params['gclass']}'
        corrplotname += '.png'
        print(f'Save correlation plots to: {corrplotname}')
#    if params['ratioplot']:
#        ratioplotname = f'int_L_nquant{params["nquant"]}_nbs{params["nbootstrap"]}_{params["sample"]}_noRatio.png'
#        print(f'Save ratio plot to: {ratioplotname}')
    if params['write']:
        filename = f'../data/int_K_nq{params["nquant"]}_nbs{params["nbootstrap"]}_{params["sample"]}'
        if params['gclass'] != 0:
            filename+=f'class{params['gclass']}'
        filename += '.npz'
        print(f'Save results to: {filename}')

    print('Calculating crosscorrelations')
    ecat = treecorr.Catalog(ra=events_a8['RA'], dec=events_a8['dec'], ra_units='deg', dec_units='deg')
    #seeds = np.linspace(1000,1+params['nquant']-1,params['nquant'],dtype=int)
    rcat = [treecorr.Catalog(ra=generate_RandomCatalogue(data[q]['_RAJ2000'], data[q]['_DEJ2000'], params['nmult'], seed=None,mask=True)[0],
            dec=generate_RandomCatalogue(data[q]['_RAJ2000'], data[q]['_DEJ2000'], params['nmult'], seed=None, mask=True)[1],
            ra_units='deg', dec_units='deg') for q in range(params['nquant'])]

    xi_bs, varxi_bs = [], []
    for q in range(params['nquant']):
        results = get_xibs(data[q], params['nbootstrap'], params['nbins'], rcat[q], ecat, treecorr_config)
        xi_bs.append(results[0])
        varxi_bs.append(results[1])
        print(f'{q + 1}/{params["nquant"]}')
    th = results[2]

    if params['corrplot']:
        print('Plotting correlations')
        fig, ax = plt.subplots()
        ax.hlines(0., 0., 90., ls=':', color='k', alpha=.7)
        fillalpha = .2
        xi1_max = [np.max(xi_bs[0][:, i]) for i in range(params['nbins'])][:-params['cutoff']]
        ax.fill_between(th[:-params['cutoff']], y1=np.max(xi1_max), color='k', alpha=fillalpha)

        alpha, capsize = .2, 2
        labels = [f'{quantiles[q]:.1f}<K_abs<{quantiles[q + 1]:.1f}' for q in range(params['nquant'])]
        colors = ['C00', 'C01', 'C02', 'C03', 'C04', 'C05']
        for q in range(params['nquant']):
            for i in range(params['nbootstrap']):
                ax.errorbar(th, xi_bs[q][i], yerr=np.sqrt(varxi_bs[q][i]), color=colors[q], label=labels[q],
                           alpha=alpha, capsize=capsize)

        handles = [plt.errorbar([], [], yerr=1, color=colors[i]) for i in range(params['nquant'])]
        handles.append(plt.fill_between([], [], color='k', alpha=fillalpha))
        labels_ = labels + ['Integration range']
        ax.legend(handles, labels_, loc=1, fancybox=True, framealpha=0.5, ncol=1)

        ax.set_xlabel(r'$\theta$ (degrees)')
        ax.set_ylabel(r'$\omega(\theta)$')
        ax.set_xlim([params['minsep'], params['maxsep']])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.savefig(corrplotname)

    print('Integration')
    int_results = [np.zeros(params['nbootstrap']) for _ in range(params['nquant'])]
    for q in range(params['nquant']):
        for i in range(params['nbootstrap']):
            int_results[q][i] = integrate.trapezoid(th[:-params['cutoff']] * xi_bs[q][i][:-params['cutoff']], x=th[:-params['cutoff']])

    int_mean = np.array([np.mean(int_results[q]) for q in range(params['nquant'])])
    int_std = np.array([np.std(int_results[q]) for q in range(params['nquant'])])
    #ratio_mean = int_mean / int_mean[0]
    #std_mean = int_std / int_mean[0]

    # if params['ratioplot']:
    #     print('Plotting ratios')
    #     mean_mag = np.array([np.mean(data[q]['K_abs']) for q in range(params['nquant'])])
    #     res = stats.linregress(-mean_mag, int_mean)
    #     print(f"R-squared: {res.rvalue**2:.3f}")

    #     fig, ax = plt.subplots()
    #     ax.plot(-mean_mag, res.intercept - res.slope * mean_mag, 'r:',
    #             label=f'Linear regression; $R^2={res.rvalue**2:.2f}$')
    #     ax.errorbar(-mean_mag, int_mean, yerr=int_std, c='C00', fmt='o')
    #     ax.set_ylabel(r'$<\int\omega_Nd\theta>$')
    #     ax.set_xlabel(r'$-K_{\mathrm{abs}}$')
    #     ax.legend(loc=4)
    #     plt.savefig(f'../plots/{ratioplotname}')

    if params['write']:
        mean_mag = np.array([np.mean(data[q]['K_abs']) for q in range(params['nquant'])])
        print(f'Writing results in: {filename}')
        ascii.write(np.column_stack([int_mean, mean_mag, int_std]), filename,
                    names=['int_mean', 'meanMag', 'int_std'], overwrite=True)

if __name__ == "__main__":
    main()