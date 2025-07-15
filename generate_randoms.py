def generate_RandomCatalogue(N, params, dec=None, ra=None, seed=None, nmult=None):
    """
    Generate random RA and Dec coordinates based on input parameters.
    If no RA nor Dec iareprovided, RA/Dec are uniformly distributed.
    If only Dec is provided, random declination is sampled from a fitted parabola.
    If both RA and Dec are provided, a 2D KDE is used to sample coordinates

    """
    import numpy as np
    from scipy.optimize import curve_fit

    if seed is not None:
        np.random.seed(seed)

    if nmult is None:
        nmult = params['nmult']
    
    dec_min = params['dec_min']
    dec_max = params['dec_max']
    N_total = N * nmult

    if dec is None and ra is None:
        # Default to uniform sin(dec) distribution
        rand_ra = np.random.uniform(0, 360, N_total)
        rand_sindec = np.random.uniform(
            np.sin(np.radians(dec_min)), np.sin(np.radians(dec_max)), N_total
        )
        rand_dec = np.degrees(np.arcsin(rand_sindec))
        return rand_ra, rand_dec

    if ra is None and dec is not None:
        # --- Fit a parabola to the declination histogram ---
        hist, bin_edges = np.histogram(dec, bins=20, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        def parabola(x, a, b, c):
            return a * x**2 + b * x + c

        popt, _ = curve_fit(parabola, bin_centers, hist)

        # Normalize parabola to form a proper PDF
        x_vals = np.linspace(dec_min, dec_max, 1000)
        pdf_vals = parabola(x_vals, *popt)
        pdf_vals = np.clip(pdf_vals, 0, None)  # avoid negatives
        norm = np.trapz(pdf_vals, x_vals)
        pdf_vals /= norm

        # Compute CDF
        cdf_vals = np.cumsum(pdf_vals)
        cdf_vals /= cdf_vals[-1]
        
        # Inverse CDF interpolation
        from scipy.interpolate import interp1d
        inv_cdf = interp1d(cdf_vals, x_vals, bounds_error=False, fill_value=(x_vals[0], x_vals[-1]))

        # Sample declination using inverse CDF
        u = np.random.uniform(0, 1, N_total)
        rand_dec = inv_cdf(u)

        # Sample RA uniformly
        rand_ra = np.random.uniform(0, 360, N_total)

        return rand_ra, rand_dec
    
    if ra is not None and dec is not None:

        # 2D KDE
        from scipy.stats import gaussian_kde   
        
        # Fit KDE in 2D
        values = np.vstack([ra, dec])
        kde = gaussian_kde(values, bw_method=0.3)  # tweak smoothing here

        # Sampling via rejection
        ra_min, ra_max = 0, 360
        dec_min, dec_max = params['dec_min'], params['dec_max']

        ra_rand = []
        dec_rand = []

        n_trials = 0
        while len(ra_rand) < N_total:
            # Batch trial
            ra_try = np.random.uniform(ra_min, ra_max, 1000)
            dec_try = np.random.uniform(dec_min, dec_max, 1000)
            samples = np.vstack([ra_try, dec_try])
            probs = kde(samples)
            probs /= np.max(probs)  # normalize for rejection sampling

            keep = np.random.rand(1000) < probs
            ra_rand.extend(ra_try[keep])
            dec_rand.extend(dec_try[keep])

            n_trials += 1
            if n_trials > 1000:
                raise RuntimeError("KDE rejection sampling too inefficient.")

        ra_rand = np.array(ra_rand[:N_total])
        dec_rand = np.array(dec_rand[:N_total])

        return ra_rand, dec_rand


def generate_CR_like_randoms(N, nmult, cr_events):
    """Generate random RA and Dec matching CR declination distribution."""

    from scipy.interpolate import interp1d
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    dec_vals = cr_events['dec']

    # Empirical PDF of Dec
    hist, bin_edges = np.histogram(dec_vals, bins=50, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    cdf = np.cumsum(hist)
    cdf /= cdf[-1]

    # Inverse CDF
    inv_cdf = interp1d(cdf, bin_centers, bounds_error=False, fill_value=(bin_centers[0], bin_centers[-1]))

    # Sample
    ra_rand = np.random.uniform(0, 360, N*nmult)
    dec_rand = inv_cdf(np.random.uniform(0, 1, N*nmult))

    return ra_rand, dec_rand

def auger_exposure(delta_deg, lat_deg=-35.2, theta_max_deg=80):
    """Compute relative exposure ω(δ) for the Pierre Auger Observatory."""

    import numpy as np

    delta = np.radians(delta_deg)
    phi = np.radians(lat_deg)
    theta_max = np.radians(theta_max_deg)

    cos_theta_max = np.cos(theta_max)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)

    arg = (cos_theta_max - sin_phi * sin_delta) / (cos_phi * cos_delta)

    alpha_m = np.arccos(np.clip(arg, -1, 1))
    alpha_m[arg > 1] = 0
    alpha_m[arg < -1] = np.pi

    omega = cos_phi * cos_delta * np.sin(alpha_m) + alpha_m * sin_phi * sin_delta
    return omega


def generate_exposure_filtered_randoms(n_samples, params, \
                                       oversample_factor=2, plot=False):
    """Generate RA/Dec randoms filtered by Auger exposure."""
    import numpy as np

    dec_min = params['dec_min']
    dec_max = params['dec_max']
    theta_max_deg = params['azimuth_max']

    n_attempts = int(n_samples * oversample_factor)

    # Generate uniform sky in solid angle
    ra_try = np.random.uniform(0, 360, n_attempts)
    sin_dec_min = np.sin(np.radians(dec_min))
    sin_dec_max = np.sin(np.radians(dec_max))
    sin_dec = np.random.uniform(sin_dec_min, sin_dec_max, n_attempts)
    dec_try = np.degrees(np.arcsin(sin_dec))

    # Apply exposure filter
    omega = auger_exposure(dec_try, theta_max_deg=theta_max_deg)
    accept_prob = omega / np.max(omega)
    mask = np.random.uniform(0, 1, size=dec_try.shape) < accept_prob

    ra_filtered = ra_try[mask]
    dec_filtered = dec_try[mask]

    if len(ra_filtered) > n_samples:
        idx = np.random.choice(len(ra_filtered), n_samples, replace=False)
        ra_filtered = ra_filtered[idx]
        dec_filtered = dec_filtered[idx]

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.hist(dec_try, bins=100, color='k', density=True, alpha=0.5, label='Raw Randoms')
        plt.hist(dec_filtered, bins=100, density=True, alpha=0.6, label='Filtered Randoms')
        
        # if include_auger:
        #     cr = get_auger_data()
        #     plt.hist(cr['dec'], bins=100, density=True, alpha=0.5, label='Auger Data')

        # Add theoretical exposure curve
        dec_vals = np.linspace(dec_min, dec_max, 1000)
        pdf = auger_exposure(dec_vals, theta_max_deg=theta_max_deg)
        pdf /= np.trapz(pdf, dec_vals)
        plt.plot(dec_vals, pdf, 'k--', lw=2, label='Exposure PDF')

        plt.xlabel('Declination [deg]')
        plt.ylabel('Probability Density (per deg)')
        plt.title('Exposure-Filtered Declination Distribution')
        plt.legend(framealpha=.2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('../plots/exposure_filtered_distribution.png')

    return ra_filtered, dec_filtered