def generate_RandomCatalogue(ra,dec,nmult,seed=None,mask=True):
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    if seed!=None: np.random.seed(seed)

    ra_min = np.min(ra)
    ra_max = np.max(ra)
    dec_min = np.min(dec)
    dec_max = np.max(dec)

    rand_ra = np.random.uniform(ra_min, ra_max, len(ra)*nmult)
    rand_sindec = np.random.uniform(np.sin(dec_min*np.pi/180.), np.sin(dec_max*np.pi/180.), \
                                    len(ra)*nmult)
    rand_dec = np.arcsin(rand_sindec)*180./np.pi

    #Eliminates points within 5° in galactic latitude
    if mask==True:
        ran = SkyCoord(rand_ra,rand_dec,frame='icrs',unit='degree')
        mask_ran = np.where([abs(ran.galactic.b)>5.*(u.degree)])[1]
        rand_ra = rand_ra[mask_ran]
        rand_dec = rand_dec[mask_ran]

    return rand_ra, rand_dec