def generate_RandomCatalogue(ra,dec,nmult,seed=None,mask=True, deflection=None):
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    if seed!=None: np.random.seed(seed)

    ra_min = np.min(ra)
    ra_max = np.max(ra)
    dec_min = np.min(dec)
    dec_max = np.max(dec)

    rand_ra = np.random.uniform(ra_min, ra_max, len(ra)*nmult*100)
    rand_sindec = np.random.uniform(np.sin(dec_min*np.pi/180.), np.sin(dec_max*np.pi/180.), \
                                    len(ra)*nmult*100)
    rand_dec = np.arcsin(rand_sindec)*180./np.pi

    # If deflection region is specified, select accordingly
    if deflection != None:
        if deflection=='high':
           rand_dec = rand_dec[(rand_ra > 200.)|(rand_ra < 90.)]
           rand_ra = rand_ra[(rand_ra > 200.)|(rand_ra < 90.)]

        elif deflection=='low': 
           rand_dec = rand_dec[(rand_ra < 200.)&(rand_ra > 90.)]
           rand_ra = rand_ra[(rand_ra < 200.)&(rand_ra > 90.)]

    #Eliminates points within 5° in galactic latitude
    if mask==True:
        ran = SkyCoord(rand_ra,rand_dec,frame='icrs',unit='degree')
        mask_ran = np.where([abs(ran.galactic.b)>5.*(u.degree)])[1]
        rand_ra = rand_ra[mask_ran]
        rand_dec = rand_dec[mask_ran]
    
    rand_ra_cut = rand_ra[:len(ra)*nmult]
    rand_dec_cut = rand_dec[:len(ra)*nmult]

    if rand_ra_cut.size != len(ra)*nmult:
        raise ValueError(f"Random catalogue size mismatch: expected {len(ra)*nmult}, got {rand_ra_cut.size}")

    return rand_ra_cut, rand_dec_cut 

def get_xibs(data,nbootstrap,nbins,rcat,ecat,config):
    import numpy as np
    import treecorr

    xi_bs = np.zeros((nbootstrap,nbins))
    varxi_bs = np.zeros((nbootstrap,nbins))

    dd = treecorr.NNCorrelation(config)
    dr = treecorr.NNCorrelation(config)
    rr = treecorr.NNCorrelation(config)
    rd = treecorr.NNCorrelation(config)

    for n in range(nbootstrap):
        databs = np.random.choice(data,size=len(data))
        gcat = treecorr.Catalog(ra=databs['_RAJ2000'], dec=databs['_DEJ2000'],\
                                ra_units='deg', dec_units='deg')

        rr.process(rcat)
        dd.process(gcat,ecat)
        dr.process(gcat,rcat)
        rd.process(ecat,rcat)

        xi_bs[n], varxi_bs[n] = dd.calculateXi(rr=rr,dr=dr,rd=rd)
    return xi_bs, varxi_bs, dd.meanr

def get_xibs_auto(data,RAcol,DECcol,nbootstrap,nbins,rcat,config):
    import numpy as np
    import treecorr

    xi_bs = np.zeros((nbootstrap,nbins))
    varxi_bs = np.zeros((nbootstrap,nbins))
    theta_ = np.zeros((nbootstrap,nbins))

    dd = treecorr.NNCorrelation(config)
    dr = treecorr.NNCorrelation(config)
    rr = treecorr.NNCorrelation(config)

    # Calculate xi_true
    gcat = treecorr.Catalog(ra=data[RAcol], dec=data[DECcol],\
                            ra_units='deg', dec_units='deg')
    rr.process(rcat)
    dd.process(gcat)
    dr.process(gcat,rcat)
    xi_true, varxi_true = dd.calculateXi(rr=rr,dr=dr)

    # Bootstrap resampling for variance estimation
    for n in range(nbootstrap):
        databs = np.random.choice(data,size=len(data))
        gcat = treecorr.Catalog(ra=databs[RAcol], dec=databs[DECcol],\
                                ra_units='deg', dec_units='deg')

        rr.process(rcat)
        dd.process(gcat)
        dr.process(gcat,rcat)

        xi_bs[n], varxi_bs[n] = dd.calculateXi(rr=rr,dr=dr)
        theta_[n] = dd.meanr

    #xi_mean = xi_bs.mean(axis=0)
    varxi = varxi_bs.mean(axis=0)
    theta = theta_.mean(axis=0)
    return xi_true, varxi, theta