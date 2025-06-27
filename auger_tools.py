def generate_RandomCatalogue(N,nmult,seed=None, milkyway_mask=True, deflection=None, deflection_file='../data/JF12_GMFdeflection_Z1_E10EeV.csv'):
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    if seed!=None: np.random.seed(seed)

    ra_min = 0.
    ra_max = 360
    dec_min = -90.
    dec_max = 45.

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

    rand_ra_cut = rand_ra[:N*nmult]
    rand_dec_cut = rand_dec[:N*nmult]

    # Check if the size of the random catalogue matches the expected size
    if rand_ra_cut.size != N*nmult:
        raise ValueError(f"Random catalogue size mismatch: expected {N*nmult}, got {rand_ra_cut.size}")

    return rand_ra_cut, rand_dec_cut 

def get_xibs(data,nbootstrap,nbins,rcat,ecat,config,seed=None):
    import numpy as np
    import treecorr

    xi_bs = np.zeros((nbootstrap,nbins))
    varxi_bs = np.zeros((nbootstrap,nbins))

    dd = treecorr.NNCorrelation(config)
    dr = treecorr.NNCorrelation(config)
    rr = treecorr.NNCorrelation(config)
    rd = treecorr.NNCorrelation(config)

    rr.process(rcat)
    rd.process(ecat,rcat)
    for n in range(nbootstrap):
        if seed!=None: np.random.seed(seed)
        elif seed==None: np.random.seed()
        databs = np.random.choice(data,size=len(data))
        gcat = treecorr.Catalog(ra=databs['_RAJ2000'], dec=databs['_DEJ2000'],\
                                ra_units='deg', dec_units='deg')

        dd.process(gcat,ecat)
        dr.process(gcat,rcat)

        xi_bs[n], varxi_bs[n] = dd.calculateXi(rr=rr,dr=dr,rd=rd)

    # Calculate the true correlation function
    gcat = treecorr.Catalog(ra=data['_RAJ2000'], dec=data['_DEJ2000'],\
                                ra_units='deg', dec_units='deg')
    dd.process(gcat,ecat)
    dr.process(gcat,rcat)
    xi_true = dd.calculateXi(rr=rr, dr=dr, rd=rd)[0]
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
