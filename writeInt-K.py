import numpy as np
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u
#from auger_tools import generate_RandomCatalogue, get_xibs, apply_deflection_mask
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
        #'cz_max': config.getint('Parameters', 'cz_max')
        'dec_max': config.getfloat('Parameters', 'dec_max'),
        'cluster_mask': config.getboolean('Parameters', 'cluster_mask')
    }

    # Optional parameter: cz_max
    cz_max_raw = config['Parameters'].get('cz_max', None)
    params['cz_max'] = None if cz_max_raw in [None, '', 'None'] else float(cz_max_raw)

    return params

def load_data(sample, dec_max, cz_min=1200, cz_max=None, gclass=None):
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
    data = data[data['_DEJ2000'] < dec_max] # Cut declination

    # Cuts for 2MRS sample:
    if sample == 'all2MRS_noAGN':
        #data = data[data['_DEJ2000'] < dec_max] # Cut declination
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

def generate_RandomCatalogue(N,nmult,dec_max,seed=None, milkyway_mask=True, deflection=None, cluster_mask=False, clusters=None, \
                             deflection_file='../data/JF12_GMFdeflection_Z1_E10EeV.csv'):
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    if cluster_mask and clusters is None:
        raise ValueError("clusters must be provided if cluster_mask is True")
    
    if seed!=None: np.random.seed(seed)

    ra_min = 0.
    ra_max = 360
    dec_min = -90.
    dec_max = dec_max

    rand_ra = np.random.uniform(ra_min, ra_max, N*nmult*100)
    rand_sindec = np.random.uniform(np.sin(dec_min*np.pi/180.), np.sin(dec_max*np.pi/180.), \
                                    N*nmult*100)
    rand_dec = np.arcsin(rand_sindec)*180./np.pi

    #Eliminates points within 5° in galactic latitude
    if milkyway_mask==True:
        ran = SkyCoord(rand_ra,rand_dec,frame='icrs',unit='degree')
        mask_ran = np.where([abs(ran.galactic.b)>5.*(u.degree)])[1]
        rand_ra = rand_ra[mask_ran]
        rand_dec = rand_dec[mask_ran]

    # If deflection region is specified, select accordingly
    if deflection == 'high' or deflection == 'low':
        randoms = np.column_stack((rand_ra, rand_dec))
        deflection_mask = apply_deflection_mask(deflection_file, randoms[:, 0], randoms[:, 1], deflection)
        randoms = randoms[deflection_mask]
        rand_ra = randoms[:, 0]
        rand_dec = randoms[:, 1]

    # If cluster mask is specified, apply it
    if cluster_mask:
        mask_ran = mask_around_clusters(rand_ra, rand_dec, clusters)
        rand_ra = rand_ra[mask_ran]
        rand_dec = rand_dec[mask_ran]

    rand_ra_cut = rand_ra[:N*nmult]
    rand_dec_cut = rand_dec[:N*nmult]

    # Check if the size of the random catalogue matches the expected size
    if rand_ra_cut.size != N*nmult:
        raise ValueError(f"Random catalogue size mismatch: expected {N*nmult}, got {rand_ra_cut.size}")

    return rand_ra_cut, rand_dec_cut 

def generate_CR_like_randoms(N, cr_events, milkyway_mask=True, deflection=None, cluster_mask=False, clusters=None,\
                             deflection_file='../data/JF12_GMFdeflection_Z1_E10EeV.csv'):
    
    from scipy.interpolate import interp1d
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    if cluster_mask and clusters is None:
        raise ValueError("clusters must be provided if cluster_mask is True")

    """Generate random RA and Dec matching CR declination distribution."""
    dec_vals = cr_events['dec']
    #ra_vals = cr_events['RA']

    # Empirical PDF of Dec
    hist, bin_edges = np.histogram(dec_vals, bins=50, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    cdf = np.cumsum(hist)
    cdf /= cdf[-1]

    # Inverse CDF
    inv_cdf = interp1d(cdf, bin_centers, bounds_error=False, fill_value=(bin_centers[0], bin_centers[-1]))

    # Sample
    ra_rand = np.random.uniform(0, 360, N*10)
    dec_rand = inv_cdf(np.random.uniform(0, 1, N*10))

    #Eliminates points within 5° in galactic latitude
    if milkyway_mask==True:
        ran = SkyCoord(ra_rand,dec_rand,frame='icrs',unit='degree')
        mask_ran = np.where([abs(ran.galactic.b)>5.*(u.degree)])[1]
        ra_rand = ra_rand[mask_ran]
        dec_rand = dec_rand[mask_ran]

    # If deflection region is specified, select accordingly
    if deflection == 'high' or deflection == 'low':
        randoms = np.column_stack((ra_rand, dec_rand))
        deflection_mask = apply_deflection_mask(deflection_file, randoms[:, 0], randoms[:, 1], deflection)
        randoms = randoms[deflection_mask]
        ra_rand = randoms[:, 0]
        dec_rand = randoms[:, 1]

    # If cluster mask is specified, apply it
    if cluster_mask:
        mask_ran = mask_around_clusters(ra_rand, dec_rand, clusters)
        ra_rand = ra_rand[mask_ran]
        dec_rand = dec_rand[mask_ran]

    rand_ra_cut = ra_rand[:N]
    rand_dec_cut = dec_rand[:N]

    if len(rand_ra_cut) < N:
        raise ValueError(f"Random catalogue size mismatch: expected {N}, got {len(rand_ra_cut)}")

    return rand_ra_cut, rand_dec_cut

def get_xibs(data,nbootstrap,nbins,rcat,rcat_auger,ecat,config,seed=None):
    import numpy as np
    import treecorr

    xi_bs = np.zeros((nbootstrap,nbins))
    varxi_bs = np.zeros((nbootstrap,nbins))

    dd = treecorr.NNCorrelation(config)
    dr = treecorr.NNCorrelation(config)
    rr = treecorr.NNCorrelation(config)
    rd = treecorr.NNCorrelation(config)

    rr.process(rcat,rcat_auger)
    rd.process(ecat,rcat)
    for n in range(nbootstrap):
        if seed!=None: np.random.seed(seed)
        elif seed==None: np.random.seed()
        databs = np.random.choice(data,size=len(data))
        gcat = treecorr.Catalog(ra=databs['_RAJ2000'], dec=databs['_DEJ2000'],\
                                ra_units='deg', dec_units='deg')

        dd.process(gcat,ecat)
        dr.process(gcat,rcat_auger)
        #rr.process(gcat,rcat_auger) #Esto emularía el estimador DD/DR-1

        xi_bs[n], varxi_bs[n] = dd.calculateXi(rr=rr)#,dr=dr,rd=rd)

    # Calculate the true correlation function
    gcat = treecorr.Catalog(ra=data['_RAJ2000'], dec=data['_DEJ2000'],\
                                ra_units='deg', dec_units='deg')
    dd.process(gcat,ecat)
    #dr.process(gcat,rcat)
    xi_true = dd.calculateXi(rr=rr)[0]#, dr=dr, rd=rd)[0]
    return xi_true, xi_bs, varxi_bs, dd.meanr

def get_xibs_auto(data,RAcol,DECcol,nbootstrap,nbins,rcat,config):
    import numpy as np
    import treecorr

    xi_bs = np.zeros((nbootstrap,nbins))
    varxi_bs = np.zeros((nbootstrap,nbins))
    theta_ = np.zeros((nbootstrap,nbins))

    dd = treecorr.NNCorrelation(config)
    dr = treecorr.NNCorrelation(config)
    rr = treecorr.NNCorrelation(config)

    for n in range(nbootstrap):
        databs = np.random.choice(data,size=len(data))
        gcat = treecorr.Catalog(ra=databs[RAcol], dec=databs[DECcol],\
                                ra_units='deg', dec_units='deg')

        rr.process(rcat)
        dd.process(gcat)
        dr.process(gcat,rcat)

        xi_bs[n], varxi_bs[n] = dd.calculateXi(rr=rr,dr=dr)
        theta_[n] = dd.meanr

    xi_mean = xi_bs.mean(axis=0)
    varxi = varxi_bs.mean(axis=0)
    theta = theta_.mean(axis=0)
    return xi_mean, varxi, theta

def apply_deflection_mask(defl_file, ra_deg, dec_deg, deflection):
    import numpy as np
    import healpy as hp

    # === Load/prepare deflection map ===
    data = np.loadtxt(defl_file, delimiter=',', skiprows=1)
    pixel_ids = data[:, 0].astype(int)
    deflection_data = data[:, 3]
    npix = int(np.max(pixel_ids)) + 1
    nside = hp.npix2nside(npix)
    nside = 64
    deflection_map = np.full(npix, hp.UNSEEN)
    deflection_map[pixel_ids] = deflection_data

    # === Create binary masks ===
    valid = deflection_map != hp.UNSEEN
    threshold = np.median(deflection_map[valid])
    if deflection=='high':
        deflection_mask = np.zeros_like(deflection_map, dtype=bool)
        deflection_mask[valid] = deflection_map[valid] > threshold
    elif deflection=='low':
        deflection_mask = np.zeros_like(deflection_map, dtype=bool)
        deflection_mask[valid] = deflection_map[valid] <= threshold

    theta = np.radians(90 - dec_deg)
    phi = np.radians(ra_deg)
    pix = hp.ang2pix(nside, theta, phi)
    return deflection_mask[pix]


def skyplot(skyplotname, params, data, events_a8, ra_random, dec_random, quantiles, clusters=None):

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
    ran_sc = SkyCoord(ra_random, dec_random,frame='icrs',unit='degree')
    
    for q, ax in zip(range(params['nquant']),axs):

        gxs_sc = SkyCoord(data[q]['_RAJ2000'],data[q]['_DEJ2000'],frame='icrs',unit='degree')
        eve_sc = SkyCoord(events_a8['RA'],events_a8['dec'],frame='icrs',unit='degree')
        #ran_sc = SkyCoord(ra_random[q], dec_random[q],frame='icrs',unit='degree')

        #ax = fig.add_subplot(111, projection="aitoff")
        ax.scatter(ran_sc.ra.wrap_at(180*u.degree).to(u.rad),ran_sc.dec.to(u.rad),s=1,c='k',label='Random Data')
        ax.scatter(eve_sc.ra.wrap_at(180*u.degree).to(u.rad),eve_sc.dec.to(u.rad),s=2,label='UHECRs')
        ax.scatter(gxs_sc.ra.wrap_at(180*u.degree).to(u.rad),gxs_sc.dec.to(u.rad),s=5,c='C03',label='Galaxies')

        if clusters is not None:
            clusters_sc = SkyCoord(clusters['RAJ2000'], clusters['DEJ2000'], frame='icrs', unit='degree')
            ax.scatter(clusters_sc.ra.wrap_at(180*u.degree).to(u.rad), clusters_sc.dec.to(u.rad), marker='x', s=30, c='C02', label='Clusters')

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
    corrplotname += f'_dec{int(params['dec_max'])}'
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
    skyplotname += f'_dec{int(params['dec_max'])}'
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
    filename += f'_dec{int(params['dec_max'])}'
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
    filecorrname += f'_dec{int(params['dec_max'])}'
    # Add format
    filecorrname += '.npz'

    return filecorrname

def crosscorrelations(data, events_a8, params, treecorr_config, write_corr=True, clusters=None):
    # Read UHECR data
    ecat = treecorr.Catalog(ra=events_a8['RA'], dec=events_a8['dec'], ra_units='deg', dec_units='deg')

    # Generate random catalogue
    #seeds = np.linspace(1000,1+params['nquant']-1,params['nquant'],dtype=int)
    ra_random = []
    dec_random = []
    for q in range(params['nquant']):
        randoms = generate_RandomCatalogue(len(data[q]['_RAJ2000']), params['nmult'], params['dec_max'],\
                                            seed=9999, milkyway_mask=True, deflection=params['def'], \
                                                cluster_mask=params['cluster_mask'], clusters=clusters if params['cluster_mask'] else None)
        
        ra_random.append(randoms[0])
        dec_random.append(randoms[1])
    rcat = [treecorr.Catalog(ra=ra_random[q], dec=dec_random[q],
            ra_units='deg', dec_units='deg') for q in range(params['nquant'])]
    
    # Generate randoms for Auger events
    # randoms_auger = generate_CR_like_randoms(len(events_a8)*50, events_a8, 
    #                                          milkyway_mask=True, deflection=params['def'], \
    #                                             cluster_mask=params['cluster_mask'], clusters=clusters if params['cluster_mask'] else None)
    randoms_auger = generate_RandomCatalogue(len(events_a8), 20, params['dec_max'],\
                                              seed=9999, milkyway_mask=True, deflection=params['def'], \
                                                cluster_mask=params['cluster_mask'], clusters=clusters if params['cluster_mask'] else None)
    rcat_auger = treecorr.Catalog(ra=randoms_auger[0], dec=randoms_auger[1],
            ra_units='deg', dec_units='deg')
    
    
    # Calculate cross-correlations
    xi_bs, varxi_bs = [], []
    xi_true = np.zeros((params['nquant'], params['nbins']))
    for q in range(params['nquant']):
        print(f'{q + 1}/{params["nquant"]}')
        results = get_xibs(data[q], params['nbootstrap'], params['nbins'], rcat[q], rcat_auger, ecat, treecorr_config)
        xi_bs.append(results[1])
        varxi_bs.append(results[2])
        xi_true[q] = results[0]
    th = results[3]

    if write_corr:
        filecorr = get_filecorrname(params)
        print(f'Writing cross-correlations to: {filecorr}')
        np.savez(filecorr, xi_true=xi_true, xi_bs=xi_bs, varxi_bs=varxi_bs, th=th)

    return xi_true, xi_bs, varxi_bs, th, randoms_auger[0], randoms_auger[1], ra_random, dec_random

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

def mask_around_clusters(cat_ra, cat_dec, clusters, factor=4.0):
    """
    Mask out sources within factor × R500 angular radius of each cluster.
    
    Parameters:
    - cat_ra, cat_dec: arrays of RA/Dec in degrees
    - clusters: astropy table with RAJ2000, DEJ2000 (degrees), z (unitless), R500 (in Mpc)
    - factor: multiplier for the angular exclusion radius (e.g., 2 × R500)
    
    Returns:
    - mask: boolean array (True = keep, False = exclude)
    """
    from astropy.cosmology import Planck15 as cosmo
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    coords = SkyCoord(ra=cat_ra*u.deg, dec=cat_dec*u.deg)
    mask = np.ones(len(cat_ra), dtype=bool)

    for cluster in clusters:
        z = cluster['z']
        r500_mpc = cluster['R500']
        ang_rad = np.arctan((factor * r500_mpc * u.Mpc / cosmo.angular_diameter_distance(z)).decompose())
        c_coord = SkyCoord(ra=cluster['RAJ2000']*u.deg, dec=cluster['DEJ2000']*u.deg)
        sep = coords.separation(c_coord)
        mask &= sep.radian > ang_rad.value 

    return mask

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

    # Check parameters
    if params['sample'] == 'all2MRS_noAGN' and params['nquant'] != 1:
        raise ValueError('For sample "all2MRS_noAGN", nquant must be 1.')
    
    # Read UHECR data
    events_a8 = ascii.read('../data/Auger/events_a8_lb.dat')
    eve = SkyCoord(events_a8['RA'], events_a8['dec'], frame='icrs', unit='degree')
    mask_eve = np.where(abs(eve.galactic.b) > 5. * u.degree)[0]
    events_a8 = events_a8[mask_eve]
    mask_eve = np.where(events_a8['dec']< params['dec_max'])[0]
    events_a8 = events_a8[mask_eve]

    # Read galaxy data
    gxs = load_data(params['sample'], params['dec_max'], cz_min=params['cz_min'], cz_max=params['cz_max'], gclass=params['gclass'])

    # Read Local Clusters file
    if params['cluster_mask']:
        local_clusters_file = '../data/Local_Clusters_z0.055.txt'
        print('Reading Local Clusters file:', local_clusters_file)
        clusters = ascii.read(local_clusters_file)
        clusters = clusters[clusters['DEJ2000'] < params['dec_max']]  # Cut declination
        clusters = clusters[(clusters['z']*3e5 > params['cz_min']) & (clusters['z']*3e5 < params['cz_max'])]

        # Apply mask to galaxies
        gxs_mask = mask_around_clusters(gxs['_RAJ2000'], gxs['_DEJ2000'], clusters)
        gxs = gxs[gxs_mask]

        # Apply mask to UHECRs
        eve_mask = mask_around_clusters(events_a8['RA'], events_a8['dec'], clusters)
        events_a8 = events_a8[eve_mask]



    # If deflection region is specified, select accordingly
    deflection_file = '../data/JF12_GMFdeflection_Z1_E10EeV.csv'
    if params['def'] == 'high' or params['def'] == 'low':
        gxs = gxs[apply_deflection_mask(deflection_file, gxs['_RAJ2000'], gxs['_DEJ2000'], params['def'])]
        events_a8 = events_a8[apply_deflection_mask(deflection_file, events_a8['RA'], events_a8['dec'], params['def'])]
        if params['cluster_mask']:
            clusters = clusters[apply_deflection_mask(deflection_file, clusters['RAJ2000'], clusters['DEJ2000'], params['def'])]

    # Read AGN type
    if params['sample'] == 'agn':
        if params['bptagn'] == 0: gxs = gxs[gxs['BPTAGN'] == 0]
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
    xi_true, xi_bs, varxi_bs, th, ra_rand_auger, dec_rand_auger, ra_rand_gxs, dec_rand_gxs = \
        crosscorrelations(data, events_a8, params, treecorr_config, \
                          clusters=clusters if params['cluster_mask'] else None)

    # Correlation plot
    if params['corrplot']:
        correlation_plot(corrplotname, params, th, xi_true, xi_bs, varxi_bs, quantiles)

    # Sky Plot
    if params['skyplot']:
        skyplot(skyplotname, params, data, events_a8, ra_rand_auger, dec_rand_auger, quantiles, clusters=clusters if params['cluster_mask'] else None)

    # Integration
    print('Integration')
    int_mean, int_std = integration(xi_true, xi_bs, th, params)

    # Write results
    if params['write']:
        write_results(filename, int_mean, int_std, quantiles)

if __name__ == "__main__":
    main()