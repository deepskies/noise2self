#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
import fitsio
import numpy as np

# galsim.Sersic documentation
# http://galsim-developers.github.io/GalSim/classgalsim_1_1sersic_1_1_sersic.html

# galsim.Shear documentation
# http://galsim-developers.github.io/GalSim/classgalsim_1_1shear_1_1_shear.html#details

ZP = 30.0 # coadd zeropoint
BANDS = ['g','r','i','z']
DTYPE = [
    ('id',int),                          # object id
    ('ra',float),                        # for filling position
    ('dec',float),                       # 
    ('flags',int),                       # flags
    ('mag',(float,len(BANDS))),          # magnitude in each band [griz]
    ('flux',(float,len(BANDS))),         # flux (ph/s) in each band [griz]
    ('sersic_n',float),                  # sersic index (galsim expects float)
    ('sersic_reff',float),               # half-light radius (arcsec) (galsim expects float)
    ('sersic_ell',float),                # ellipticity 
    ('sersic_beta',float),               # position angle (degrees)
    ('sersic_mueff',(float,len(BANDS))), # effective surface brightness (mag/arcsec2)
    ('sersic_mu0',(float,len(BANDS))),   # central surface brightness (mag/arcsec2)
    ]
# Polynomial fits to Bressan 2012 SSP
GI_POLY=np.poly1d([ 1.47190088, -0.05785033]) 
GZ_POLY=np.poly1d([ 1.65698477, -0.04642952]) 

def fill_general(catalog):
    """
    Fill general catalog properties.
    """
    size = len(catalog)

    catalog['id'][:] = np.arange(size)
    # Location doesn't matter
    catalog['ra'] = 0.
    catalog['dec'] = 0.
    # Everything good
    catalog['flags'] = 0

def fill_magnitude(catalog):
    """ 
    Fill the mag and flux values in the catalog. Assumes 4 bands
    [griz] for both mag and flux. Colors are determined from a fit to
    a Bressan 2012 isochrone.
    """
    size = len(catalog) 

    g = np.random.uniform(15,25,size=size)
    color_spread = np.random.normal(scale=0.2,size=size)

    gr_color = np.random.uniform(-0.25,1.25,size=size)
    gi_color = GI_POLY(gr_color) + color_spread
    gz_color = GZ_POLY(gr_color) + color_spread

    catalog['mag'][:,0] = g
    catalog['mag'][:,1] = g - gr_color
    catalog['mag'][:,2] = g - gi_color
    catalog['mag'][:,3] = g - gz_color

    # Convert from magnitude to flux using coadd zeropoint
    catalog['flux'][:] = 10**((catalog['mag'][:] - ZP)/-2.5)

def fill_sersic(catalog):
    """ 
    Fill the Sersic profile parameters.
    """
    # Some notes on Sersic profiles:
    # https://ned.ipac.caltech.edu/level5/March05/Graham/Graham2.html

    size = len(catalog)
    catalog['sersic_n'] = np.random.uniform(0.3,4.5,size=size)
    catalog['sersic_reff'] = np.random.uniform(2,20,size=size)
    catalog['sersic_ell'] = np.random.uniform(0.,0.8,size=size)
    catalog['sersic_beta'] = np.random.uniform(0.,180.,size=size)
    # Surface brightness within the effective (half-light) radius
    catalog['sersic_mueff'][:] = catalog['mag'][:] + 2.5*np.log10(2.) \
        + 2.5*np.log10(np.pi * catalog['sersic_reff']**2)[:,np.newaxis]
    # Central surface brightness
    catalog['sersic_mu0'][:] = sersic_mu0(catalog['sersic_mueff'],catalog['sersic_n'])

def sersic_mu0(mueff,n):
    """
    Approximate Sersic central surface brightness.

    From Graham's notes:
    https://ned.ipac.caltech.edu/level5/March05/Graham/Graham2_2.html

      mu0 = mueff - 2.5 bn / ln(10)
    """
    return mueff - 2.5 / np.log(10) * sersic_bn(n)[:,np.newaxis]
    
def sersic_bn(n):
    """
    An approximate value for bn is provided in Equation A1 of
    (really a reference to Ciotti & Bertin 1999)

      bn ~ 2n - 1/3 + 4/405n + 46/25515n**2 + 131/1148175n**3 - 2194697/30690717750n**4

    For n < 0.36 MacArthur, Courteau & Holtzman 2003 provide the following polynomial fit
    http://adsabs.harvard.edu/abs/2003ApJ...582..689

      a0=0.01945; a1=-0.8902; a2=10.95; a3=-19.67; a4=13.43;

    Also see a numerical table from:
    http://www.astr.tohoku.ac.jp/~akhlaghi/Sersic_C.html
    """
    def big_n(n):
        return 2*n - 1./3 + 4/405. * n**-1 + 46/25515. * n**-2 + 131/1148175 * n**-3 \
            - 2194697/30690717750. * n**-4

    def small_n(n):
        return 0.01945 - 0.8902*n + 10.95*n**2 - 19.67*n**3 + 13.43*n**4

    if np.isscalar(n):
        return np.asscalar(np.where(n > 0.36, big_n(n), small_n(n)))
    else:
        return np.where(n > 0.36, big_n(n), small_n(n))

def generate_catalog(size=1e5):
    size = int(size)

    catalog = np.ones(size,dtype=DTYPE)

    fill_general(catalog)
    fill_magnitude(catalog)
    fill_sersic(catalog)

    return catalog

def cut_catalog(catalog):
    """ Apply cuts to the catalog """
    #sel  = np.all(catalog['sersic_mueff'] > 23, axis=1)
    #sel &= np.all(catalog['sersic_mueff'] < 30, axis=1)
    
    sel  = np.all(catalog['sersic_mueff'] > 24, axis=1)
    sel &= np.all(catalog['sersic_mueff'] < 28, axis=1)

    return catalog[sel]

def simulate_catalog(size=1e5, chunk=None):
    """ Generate and cut catalogs """
    size = int(size)
    if chunk is None: 
        chunk = 10*size

    catalog = []
    while len(catalog) < size:
        cat = generate_catalog(chunk)
        cat = cut_catalog(cat)
        if not catalog:
            catalog = cat
        else:
            catalog = np.concatenate([catalog,cat])
    
    return catalog[:size]

def plot_diagnostics(catalog):
    """
    Make catalog diagnostic plots.
    """
    import pylab as plt

    gr_color = catalog['mag'][:,0] - catalog['mag'][:,1]
    gi_color = catalog['mag'][:,0] - catalog['mag'][:,2]
    gz_color = catalog['mag'][:,0] - catalog['mag'][:,3]

    params = dict(s=3,c='k',alpha=0.1)
    # color-color figure
    x = np.linspace(gr_color.min(),gr_color.max())
    fig,[ax0,ax1] = plt.subplots(1,2,figsize=(14,5))

    ax0.scatter(gi_color,gr_color,**params)
    ax0.plot(GI_POLY(x),x,'--r')
    ax0.set_xlabel(r'$g-i$');ax0.set_ylabel(r'$g-r$')
    ax0.set_xlim(-0.25,1.5);ax0.set_ylim(-0.5,1.25)

    ax1.scatter(gz_color,gr_color,**params)
    ax1.plot(GZ_POLY(x),x,'--r')
    ax1.set_xlabel(r'$g-z$');ax1.set_ylabel(r'$g-r$')
    ax1.set_xlim(-0.25,1.5);ax1.set_ylim(-0.5,1.25)

    plt.savefig('udg_color_color.png')

    # surface brightness figure
    fig,[ax0,ax1] = plt.subplots(1,2,figsize=(14,5))
    ax0.scatter(catalog['sersic_mueff'][:,0],catalog['sersic_reff'],**params)
    ax0.axvline(24.3,ls='--',color='r')
    ax0.set_xlim(30,20);ax0.set_ylim(2,20)
    ax0.set_xlabel(r'$\mu_{\rm eff}(g)\, ({\rm mag\, arcsec^{-2}})$');
    ax0.set_ylabel(r'$r_{\rm eff}\, ({\rm arcsec})$')

    ax1.scatter(catalog['sersic_mu0'][:,0],catalog['sersic_n'],**params)
    n = np.linspace(0.05,4.7,1000)
    ax1.plot(sersic_mu0(np.array([24.3]),n),n,'--r')
    ax1.set_xlim(30,16);ax1.set_ylim(0,4.5)
    ax1.set_xlabel(r'$\mu_0(g)\, ({\rm mag\, arcsec^{-2}})$');
    ax1.set_ylabel(r'$n$')

    plt.savefig('udg_mueff.png')

    # Sersic profile
    fig,[ax0,ax1] = plt.subplots(1,2,figsize=(14,5))
    ax0.hist(catalog['sersic_ell'],bins=np.linspace(0,1,21),histtype='stepfilled',color='k',
             normed=True)
    ax0.set_xlabel('Ellipticity');
    ax0.set_ylabel('Normalized Counts')
    
    for i,(b,c) in enumerate(zip(BANDS,['#008060','#ff4000','#850000','#6600cc'])):
        ax1.hist(catalog['mag'][:,i],
                 bins=np.linspace(10,25,31),histtype='stepfilled',color=c,
                 normed=True,alpha=0.3,label=b)
    ax1.set_xlabel('Magnitude')
    ax1.set_ylabel('Normalized Counts')
    ax1.legend(loc='upper left')
    
    plt.savefig('udg_ell_mag.png')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename',nargs='?',default='udg_test_catalog.fits',
                        help="output catalog filename")
    parser.add_argument('-s','--size',default=int(1e5),type=int,
                        help="simulated catalog size")
    parser.add_argument('-p','--plot',action='store_true',
                        help="plot diagnostics")
    parser.add_argument('--seed',default=0,type=int,
                        help="random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    catalog = simulate_catalog(args.size)
    fitsio.write(args.filename,catalog,clobber=True)

    if args.plot:
        plot_diagnostics(fitsio.read(args.filename))
