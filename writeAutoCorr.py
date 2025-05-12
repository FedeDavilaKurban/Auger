import numpy as np
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u
from auger_tools import generate_RandomCatalogue, get_xibs_auto
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
    elif sample == '700control':
        filename_g = '../data/VLS_ang5_cz_700control.txt'
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
    if params['sample'] == 'Auger':
        data = ascii.read('../data/Auger/events_a8_lb.dat')
        data_ = SkyCoord(data['RA'], data['dec'], frame='icrs', unit='degree')
        mask_data = np.where(abs(data_.galactic.b) > 5. * u.degree)[0]
        data = data[mask_data]

    else:
    # Read galaxy data
        data = load_data(params['sample'])
        data = data[data['cz'] > 1000.]


    # Read class for control sample
    if params['sample']=='control':
        data = data[data['cz'] > 1000.]
        if params['gclass'] != 0:
            if params['gclass']==2: data = data[data['class'] == 2]
            elif params['gclass']==3: data = data[data['class'] == 3]


    if params['write']:
        filename = f'../data/autoCorr_nbs{params["nbootstrap"]}_{params["sample"]}'
        if params['gclass'] != 0:
            filename+=f"class{params['gclass']}"
        filename += '.npz'
        print(f'Save results to: {filename}')

    print('Calculating Autocorrelation')
    if params['sample']=='Auger':
        RAcol = 'RA'
        DECcol = 'dec'
    else:
        RAcol = '_RAJ2000'
        DECcol = '_DEJ2000'
    
    random_ra, random_dec = generate_RandomCatalogue(data[RAcol], data[DECcol], params['nmult'], seed=None, mask=True)

    rcat = treecorr.Catalog(ra=random_ra,
            dec=random_dec,
            ra_units='deg', dec_units='deg')

    results = get_xibs_auto(data, RAcol, DECcol, params['nbootstrap'], params['nbins'], rcat, treecorr_config)
    xi_mean = results[0]
    varxi = results[1]
    theta = results[2]


    if params['write']:
        print(f'Writing results in: {filename}')
        ascii.write(np.column_stack([xi_mean, varxi, theta]), filename,
                    names=['xi_mean', 'varxi', 'theta'], overwrite=True)

if __name__ == "__main__":
    main()