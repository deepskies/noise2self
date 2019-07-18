#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
import os
import yaml
import fitsio
import numpy as np
import glob
from collections import defaultdict as ddict, OrderedDict as odict
import numpy.lib.recfunctions as recfn

from esutil import wcsutil

SIZE = 128
BANDS = ['deg','g','r','i','z']

XMIN,XMAX = 0, 2048  # CCD x-dimensions
YMIN,YMAX = 0, 4096  # CCD y-dimensions
CMIN,CMAX = 0, 10000 # coadd tile dimensions
NMIN = 15            # minimum number of exposures

filename = 'configs/bal_config_0_DES0324-3040.yaml'

with open(filename) as f:
    list_doc = yaml.safe_load_all(f)
    configs = list(list_doc)
info = configs.pop(0)

#truth = fitsio.read('../output_v2/y3v02/balrog_images/0/y3v02/DES0324-3040/DES0324-3040_0_balrog_truth_cat_gals.fits')
#truth = fitsio.read('y3_balrog_test_udg_mof_v3.fits')
truth = fitsio.read('udg_test_catalog_v4.fits')
catalog = fitsio.read('DES0324-3040_0_balrog_truth_cat_gals.fits')

# Exclude duplicate objects (make life difficult)
ids,cts = np.unique(catalog['id'],return_counts=True)
exclude_id = ids[cts>1]
exclude_idx = np.where(np.in1d(truth['id'],exclude_id))[0]

print("Excluding the following ids: %s"%exclude_id)
print("Excluding the following idxs: %s"%exclude_idx)

# Reformat config file
temp = dict(id=None,x=None,y=None,filename=None)
objects = ddict(list)

print("Re-organizing config...")
for j,c in enumerate(configs):
    if j%10==0: print("Reading config %i..."%j)
    for i,index in enumerate(c['gal']['index']['items']):
        if index in exclude_idx: 
            #print("Skipping duplicate idx: %i"%index)
            continue
        x = c['image']['image_pos']['x']['items'][i]
        y = c['image']['image_pos']['y']['items'][i]
        band = c['input']['udg_catalog']['bands']
        f = c['output']['file_name']
        objects[index] += [[index,x,y,band,f]]

# Build the dictionary of coadd images
print("Opening coadd images...")
bands = ['det','g','r','i','z']
filename = 'DES0324-3040/coadd/DES0324-3040_%s.fits'
images = odict()
for b in bands:
    fits = fitsio.FITS(filename%b)
    wcs = wcsutil.WCS(fits[1].read_header())
    images[b] = (fits,wcs)

#idx = 8146
#idx = 90134

size = SIZE
names = ['idx','x','y','band','filename']

outdir = 'cutouts'
basename = os.path.join(outdir,'balrog_cutouts_%05d.fits.gz')

if not os.path.exists(outdir): os.makedirs(outdir)

for idx in sorted(objects.keys()):
    
    # Minimum number of exposures
    if len(objects[idx]) < NMIN: continue
    print('Generating cutouts for object: %i'%idx)

    # Create the metadata
    data = np.rec.fromrecords(objects[idx],names=names)
    data = recfn.rec_append_fields(truth[len(data)*[idx]], names, [data[n] for n in names])
    # Extra columns
    data = recfn.rec_append_fields(data,['flag'],[np.zeros(len(data),dtype=int)])

    # Get the ra,dec
    sel = (catalog['id'] == data['id'][0])
    ra = np.asscalar(catalog[sel]['ra'])
    dec = np.asscalar(catalog[sel]['dec'])
    data['ra'] = ra
    data['dec'] = dec

    # Create the single exposure stack
    stack = np.nan*np.ones((len(data),SIZE,SIZE))
    for i,d in enumerate(data):
        fits = fitsio.FITS(d['filename'])
        x,y = int(np.round(d['x'])), int(np.round(d['y']))

        ixmin, xmin = 0, x-size//2
        if xmin < XMIN: ixmin,xmin = XMIN - xmin, XMIN
        ixmax, xmax = size, x+size//2
        if xmax > XMAX: ixmax,xmax = XMAX - xmax, XMAX
        iymin, ymin = 0, y-size//2
        if ymin < YMIN: iymin, ymin = YMIN - ymin, YMIN
        iymax, ymax = size, y+size//2
        if ymax > YMAX: iymax,ymax = YMAX - ymax, YMAX
            

        img = fits['SCI'][ymin:ymax, xmin:xmax]
        stack[i,iymin:iymax,ixmin:ixmax] = img
    data['flag'] |= np.isnan(stack).any(axis=-1).any(axis=-1)

    # Create the coadd cutouts
    coadd = np.nan*np.ones((len(BANDS),SIZE,SIZE))
    for i,(fits,wcs) in enumerate(images.values()):
        x,y = wcs.sky2image(ra,dec)
        x,y = int(np.round(x)), int(np.round(y))

        ixmin, xmin = 0, x-size//2
        if xmin < CMIN: ixmin,xmin = CMIN - xmin, CMIN
        ixmax, xmax = size, x+size//2
        if xmax > CMAX: ixmax,xmax = CMAX - xmax, CMAX
        iymin, ymin = 0, y-size//2
        if ymin < CMIN: iymin, ymin = CMIN - ymin, CMIN
        iymax, ymax = size, y+size//2
        if ymax > CMAX: iymax,ymax = CMAX - ymax, CMAX

        img = fits['SCI'][ymin:ymax, xmin:xmax]
        coadd[i,iymin:iymax,ixmin:ixmax] = img
        
    #if data['flag'].sum(): print('  Flagging...')
    if np.isnan(coadd).any(): 
        print('  Coadd tile edge; skipping...')

    outfile = basename%idx
    print("  Writing %s..."%outfile)
    if os.path.exists(outfile): os.remove(outfile)
    out = fitsio.FITS(outfile,'rw')
    out.write(stack,extname='IMAGE')
    out.write(data,extname='DATA')
    out.write(coadd,header={'BANDS':BANDS},extname='COADD')
    out.close()
