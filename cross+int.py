import numpy as np
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u
from auger_tools import generate_RandomCatalogue, get_xibs
import configparser

"""
Read config file
"""
print('Reading files')

config = configparser.ConfigParser()
config.read('cross+int.ini')

minsep = config.getfloat('Parameters','minsep')      # Min theta
maxsep = config.getfloat('Parameters','maxsep')      # Max theta
nbins = config.getint('Parameters','nbins')          # Bins in theta
nmult = config.getint('Parameters','nmult')          # nmult := Nr/Nd
nbootstrap = config.getint('Parameters','nbootstrap')# No. of bootstrap resampling
brute = config.getboolean('Parameters','brute')      # Brute force for TreeCorr
nquant = config.getint('Parameters','nquant')        # No. of quantiles to split sample in Mag_K
cutoff = config.getint('Parameters','cutoff')        # Ignore the 'cutoff' no. of last bins for integration

#seed1 = 12936
#seed2 = 19284

sample = config.get('Parameters','sample')
write = config.getboolean('Parameters','write')
corrplot = config.getboolean('Parameters','corrplot')
ratioplot = config.getboolean('Parameters','ratioplot')
tc_config = {"min_sep": minsep, \
          "max_sep": maxsep, \
            "nbins": nbins, \
            "sep_units": 'degree', \
            "bin_type": 'Linear', \
            "brute": brute, \
            "metric": 'Arc', \
            }

# Read UHECR
filename_e = '../data/Auger/events_a8_lb.dat'
events_a8 = ascii.read(filename_e)
# Galaxy Mask
eve = SkyCoord(events_a8['RA'],events_a8['dec'],frame='icrs',unit='degree')
mask_eve = np.where([abs(eve.galactic.b)>5.*(u.degree)])[1]
events_a8 = events_a8[mask_eve]

# Read Gxs
if sample=='passivecrop': filename_g = '../data/VLS/2MRSxWISE_VLS_passivecrop.txt'
elif sample=='sinAGNWISE': filename_g = '../data/2MRSxWISE_VLS_d1d5_sinAGNWISE.txt'
elif sample=='sinAGNWISEniBPT': filename_g = '../data/2MRSxWISE_VLS_d1d5_sinAGNWISEniBPT.txt'
else: filename_g = '../data/VLS/2MRSxWISE_VLS.txt'
print('Sample file:',filename_g)
gxs = ascii.read(filename_g)

# Bright/Faint
quantiles = np.quantile(gxs['K_abs'],np.linspace(0,1,nquant+1))

data = []

for q in range(nquant):
  data.append(

    gxs[(gxs['K_abs']>quantiles[q])&(gxs['K_abs']<quantiles[q+1])]

  )

if corrplot==True:
    corrplotname = f'../data/cross_treecorr_nq{nquant}_nmult{nmult}_nbs{nbootstrap}'
    corrplotname += f'_{sample}'
    corrplotname += '.png'
    print('Save correlation plots to:',corrplotname)
if ratioplot==True:    
    ratioplotname = f'int_L_nquant{nquant}_nbs{nbootstrap}_{sample}.png'
    print('Save ratio plot to:',ratioplotname)
if write==True:
    filename = f'../data/int_L_nq{nquant}_nbs{nbootstrap}_{sample}.npz'
    print('Save results to:',filename)

"""
CALCULATIONS
"""
print('Calculating crosscorrelations')
import treecorr

# TreeCorr Catalogues
ecat = treecorr.Catalog(ra=events_a8['RA'], dec=events_a8['dec'], \
                        ra_units='deg', dec_units='deg')

rcat = []
for q in range(nquant):
    rand_ra, rand_dec = generate_RandomCatalogue(data[q]['_RAJ2000'],data[q]['_DEJ2000'],\
                                               nmult, mask=True)
    rcat.append(treecorr.Catalog(ra=rand_ra, dec=rand_dec, \
                        ra_units='deg', dec_units='deg'))

xi_bs = []
varxi_bs = []
for q in range(nquant):
    results =  get_xibs(data[q],nbootstrap,nbins,rcat[q],ecat,tc_config) 
    xi_bs.append(results[0])
    varxi_bs.append(results[1])
    print(q+1,'/',nquant)
th = results[2]

"""
Corr PLOT
"""
if corrplot==True:

    print('Plotting correlations')

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    fig, ax = plt.subplots()

    ax.hlines(0.,0.,90.,ls=':',color='k',alpha=.7)
    fillalpha=.2
    xi1_max = [np.max(xi_bs[0][:,i]) for i in range(nbins)][:-cutoff]
    ax.fill_between(th[:-cutoff], y1=np.max(xi1_max), color='k', alpha=fillalpha)

    alpha=.2
    capsize = 2
    labels = []
    for q in range(nquant):
        labels.append(
            f'{quantiles[q]:.1f}'+r'$<K_{abs}<$'+f'{quantiles[q+1]:.1f}'
        )

    c = ['C00','C01','C02','C03','C04','C05']
    for q in range(nquant):
        for i in range(nbootstrap):
            line = ax.errorbar(th, xi_bs[q][i], yerr=np.sqrt(varxi_bs[q][i]), \
                        color=c[q], label=labels[q],\
                        alpha=alpha, capsize=capsize)

    #Legend
    handles = [plt.errorbar([],[],yerr=1,color=c[i]) for i in range(nquant)]#,\
    handles.append(plt.fill_between([],[],color='k',alpha=fillalpha))
    labels_ = [labels[i] for i in range(nquant)]
    labels_.append('Integration range')
    plt.legend(handles, labels_, loc=1, fancybox=True, framealpha=0.5, ncol=2)


    ax.set_xlabel(r'$\theta$ (degrees)')
    ax.set_ylabel(r'$\omega(\theta)$')

    ax.set_xlim([minsep,maxsep])

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.savefig(corrplotname)

    #plt.show()
    #plt.close()

"""
INTEGRATION
"""
print('Integration')


from scipy import integrate

int = []
for q in range(nquant):
    int.append(np.zeros(nbootstrap))

for q in range(nquant):
    for i in range(nbootstrap):
        int[q][i] = integrate.trapezoid(xi_bs[q][i][:-cutoff],x=th[:-cutoff])

ratio_mean = np.zeros(nquant)
std_mean = np.zeros(nquant)

for q in range(nquant):
    ratio_mean[q] = np.mean(int[q])/np.mean(int[0])
    std_mean[q] = np.std(int[q])/np.mean(int[0])

"""
RATIO PLOTS
"""
if ratioplot==True:
    print('Plotting ratios')

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    meanMag = np.zeros(nquant)
    for q in range(nquant):
        meanMag[q] = np.mean(data[q]['K_abs'])


    L_ratio = np.zeros(nquant)
    for q in range(nquant):
        L_ratio[q] = 10**(-.4*(meanMag[q]-meanMag[0]))

    print(L_ratio,ratio_mean)

    ax.scatter(1,1,c='C00')
    for i in range(len(ratio_mean)):
        ax.errorbar(L_ratio[i],ratio_mean[i],yerr=std_mean[i],c='C00',fmt='o')
    ax.axline((1,1),slope=1,c='k',ls=':')
    ax.set_ylabel(r'$<\int\omega_Nd\theta>/<\int\omega_1d\theta>$')
    ax.set_xlabel(r'$L_N/L_1$')

    plt.savefig('../plots/'+ratioplotname)

    #plt.show()
    #plt.close()


if write==True:

    names = ['int_ratio','L_ratio','int_std']

    print('Writing results in:', filename)
    ascii.write(np.column_stack([ratio_mean,L_ratio,std_mean]),filename,names=names,overwrite=True)
