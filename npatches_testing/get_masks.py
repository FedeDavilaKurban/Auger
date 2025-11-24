def get_milkyway_mask(ra, dec):
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    """Apply a mask to eliminate points within 5° in galactic latitude."""
    ran = SkyCoord(ra, dec, frame='icrs',unit='degree')
    mask = np.where([abs(ran.galactic.b)>5.*(u.degree)])[1]
    return mask

# def get_deflection_mask(defl_file, ra_deg, dec_deg, deflection):
#     import numpy as np
#     import healpy as hp
#     import os

#     # === Load/prepare deflection map ===
#     data = np.loadtxt(defl_file, delimiter=',', skiprows=1)
#     pixel_ids = data[:, 0].astype(int)
#     deflection_data = data[:, 3]
#     npix = int(np.max(pixel_ids)) + 1
#     nside = hp.npix2nside(npix)
#     nside = 64
#     deflection_map = np.full(npix, hp.UNSEEN)
#     deflection_map[pixel_ids] = deflection_data

#     # === Create binary masks ===
#     valid = deflection_map != hp.UNSEEN
#     threshold = np.median(deflection_map[valid])
#     if deflection=='high':
#         deflection_mask = np.zeros_like(deflection_map, dtype=bool)
#         deflection_mask[valid] = deflection_map[valid] > threshold
#     elif deflection=='low':
#         deflection_mask = np.zeros_like(deflection_map, dtype=bool)
#         deflection_mask[valid] = deflection_map[valid] <= threshold

#     theta = np.radians(90 - dec_deg)
#     phi = np.radians(ra_deg)
#     pix = hp.ang2pix(nside, theta, phi)
#     return deflection_mask[pix]


# Includes possibility of deflection as a tuple of quantiles
def get_deflection_mask(defl_file, ra_deg, dec_deg, deflection, defl_column='deflection'):
    import numpy as np
    import healpy as hp
    from astropy.io import ascii

    # === Load/prepare deflection map ===
    data = ascii.read(defl_file)
    if defl_column not in data.colnames:
        raise ValueError(f"Column '{defl_column}' not found in deflection file.")
    pixel_ids = data['pixid']
    deflection_data = data[defl_column]
    npix = int(np.max(pixel_ids)) + 1
    nside = hp.npix2nside(npix)
    nside = 64  # override to ensure consistent nside
    deflection_map = np.full(npix, hp.UNSEEN)
    deflection_map[pixel_ids] = deflection_data

    # === Define valid region ===
    valid = deflection_map != hp.UNSEEN
    valid_defl = deflection_map[valid]

    # === Build mask based on mode ===
    deflection_mask = np.zeros_like(deflection_map, dtype=bool)

    if deflection == 'high':
        threshold = np.median(valid_defl)
        deflection_mask[valid] = deflection_map[valid] > threshold
    elif deflection == 'low':
        threshold = np.median(valid_defl)
        deflection_mask[valid] = deflection_map[valid] <= threshold
    elif isinstance(deflection, tuple) and len(deflection) == 2:
        # Mask values within the specified quantile range
        qmin, qmax = deflection
        lo = np.quantile(valid_defl, qmin)
        hi = np.quantile(valid_defl, qmax)
        deflection_mask[valid] = (deflection_map[valid] >= lo) & (deflection_map[valid] <= hi)
    else:
        raise ValueError("deflection must be 'high', 'low', or a (qmin, qmax) tuple")

    # === Query input coordinates ===
    theta = np.radians(90 - dec_deg)
    phi = np.radians(ra_deg)
    pix = hp.ang2pix(nside, theta, phi)
    
    return deflection_mask[pix]

def get_deflection_smoothing_mask(defl_file, ra_deg, dec_deg, deflection, defl_column='deflection', fwhm_deg=10.0):
    import numpy as np
    import healpy as hp
    from astropy.io import ascii

    # === Load/prepare deflection map ===
    data = ascii.read(defl_file)
    if defl_column not in data.colnames:
        raise ValueError(f"Column '{defl_column}' not found in deflection file.")
    pixel_ids = data['pixid']
    deflection_data = data[defl_column]
    npix = int(np.max(pixel_ids)) + 1
    nside = hp.npix2nside(npix)
    nside = 64  # override to ensure consistent nside
    deflection_map = np.full(npix, hp.UNSEEN)
    deflection_map[pixel_ids] = deflection_data

    # === Smooth the map ===
    if fwhm_deg and fwhm_deg > 0:
        fwhm_rad = np.radians(fwhm_deg)
        deflection_map = hp.smoothing(deflection_map, fwhm=fwhm_rad, verbose=False)

    # === Define valid region ===
    valid = deflection_map != hp.UNSEEN
    valid_defl = deflection_map[valid]

    # === Build mask based on mode ===
    deflection_mask = np.zeros_like(deflection_map, dtype=bool)

    if deflection == 'high':
        threshold = np.median(valid_defl)
        deflection_mask[valid] = deflection_map[valid] > threshold
    elif deflection == 'low':
        threshold = np.median(valid_defl)
        deflection_mask[valid] = deflection_map[valid] <= threshold
    elif isinstance(deflection, tuple) and len(deflection) == 2:
        # Mask values within the specified quantile range
        qmin, qmax = deflection
        lo = np.quantile(valid_defl, qmin)
        hi = np.quantile(valid_defl, qmax)
        deflection_mask[valid] = (deflection_map[valid] >= lo) & (deflection_map[valid] <= hi)
    else:
        raise ValueError("deflection must be 'high', 'low', or a (qmin, qmax) tuple")

    # === Query input coordinates ===
    theta = np.radians(90 - dec_deg)
    phi = np.radians(ra_deg)
    pix = hp.ang2pix(nside, theta, phi)

    return deflection_mask[pix]

def get_cluster_mask(cat_ra, cat_dec, clusters, factor=4.0):
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
