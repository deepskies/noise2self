from glob import glob
import subprocess as sp
import sep
import fitsio
import numpy as np
import math
import time


def Sep_Gal(input_path , num_files , bkg_box_size, filter_size , det_thresh, txt_name):
    files = glob(input_path) # 'data/*.fits'
    start = time.time()
    for u in range(num_files):
        begin = time.time()
        fits = fitsio.FITS(files[u])
        metadata = fits[1].read()
        Magnitude_zeropoint = metadata['mag'][0,0] + 2.5*math.log10(metadata['flux'][0,0])
        mag = metadata['mag'][0,0]

        dataa = fitsio.read(files[u])
        data = dataa[0]
        m, s = np.mean(data), np.std(data)
        bkg = sep.Background(data , bw = bkg_box_size, bh = bkg_box_size, fw = filter_size, fh = filter_size, fthresh = 0.0)
        #bkg_box_size = 38, filter_size = 3
        bkg_image = bkg.back()
        bkg_rms = bkg.rms()
        data_sub = data - bkg
        objects = sep.extract(data_sub, det_thresh , err=bkg.globalrms) # det_thresh = 1.2
        flux, fluxerr, flag = sep.sum_circle(data_sub, objects['x'], objects['y'],
                                         3.0, err=bkg.globalrms, gain=1.0)
        center = []
        for v in range(len(objects['x'])):
            if objects['x'][v] < 68 and objects['x'][v] > 60 and objects['y'][v] > 60 and objects['y'][v] < 68:
                center.append(objects['x'][v])   
        
    
        Object_1 = '''
 0) sky                    #  object type
 1) 1.3920      1          #  sky background at center of fitting region [ADUs]
 2) 0.0000      0          #  dsky/dx (sky gradient in x)
 3) 0.0000      0          #  dsky/dy (sky gradient in y)
 Z) 0                      #  output option (0 = resid., 1 = Don't subtract) 
'''
    
                 
        if len(center)== 0 :
            Object_2 = '''
 0) sersic                 #  object type
 1) 64  64  1 1  #  position x, y
 3) 20     1          #  Integrated magnitude	
 4) 5.1160      1          #  R_e (half-light radius)   [pix]
 5) 4.2490      1          #  Sersic index n (de Vaucouleurs n=4) 
 6) 0.0000      0          #     ----- 
 7) 0.0000      0          #     ----- 
 8) 0.0000      0          #     ----- 
 9) 0.7570      1          #  axis ratio (b/a)  
10) -60.3690    1          #  position angle (PA) [deg: Up=0, Left=90]
 Z) 0                      #  output option (0 = resid., 1 = Don't subtract) 
'''
        else:
            Object_2 = " "
        
                   
                
        Object_add = ''
        for i in range(len(objects['x'])):
            Object_add += '''
 0) sersic                 #  object type
 1) ''' + str(int(objects['x'][i])) + ' ' + str(int(objects['y'][i])) + '''  1 1  #  position x, y
 3) ''' + str(Magnitude_zeropoint - 2.5*math.log10(objects['flux'][i])) + '''     1          #  Integrated magnitude	
 4) 5.1160      1          #  R_e (half-light radius)   [pix]
 5) 4.2490      1          #  Sersic index n (de Vaucouleurs n=4) 
 6) 0.0000      0          #     ----- 
 7) 0.0000      0          #     ----- 
 8) 0.0000      0          #     ----- 
 9) 0.7570      1          #  axis ratio (b/a)  
 10) -60.3690    1          #  position angle (PA) [deg: Up=0, Left=90]
 Z) 0                      #  output option (0 = resid., 1 = Don't subtract) 
'''
            

        N_In = str(files[u])
        N_out = str(files[u].replace('/balrog_cutouts_','imgblock_')) 

        front = '''===============================================================================
# IMAGE and GALFIT CONTROL PARAMETERS
A) ''' + N_In + '''      # Input data image (FITS file)
B) ''' + N_out + '''       # Output data image block
C) none                # Sigma image name (made from data if blank or "none") 
D) psf.fits   #        # Input PSF image and (optional) diffusion kernel
E) 1                   # PSF fine sampling factor relative to data 
F) none                # Bad pixel mask (FITS image or ASCII coord list)
G) none                # File with parameter constraints (ASCII file) 
H) 1    128   1    128   # Image region to fit (xmin xmax ymin ymax)
I) 100    100          # Size of the convolution box (x y)
J) ''' + str(Magnitude_zeropoint) + '''              # Magnitude photometric zeropoint 
K) 0.038  0.038        # Plate scale (dx dy)    [arcsec per pixel]
O) regular             # Display type (regular, curses, both)
P) 0                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps

''' 
        end = '''
================================================================================'''
        All =  front + Object_1 + Object_2 + Object_add + end

    
        outfile = open(txt_name, "w")
        outfile.write(All)
        outfile.close()
        
        Befehl = str("./galfit " + txt_name)
        sp.check_call(Befehl , shell = True)
        ending = time.time()
        print('file' + str(u)+ ':  '+ str(ending - begin))

    
    end = time.time()
    print('total: '+ str(end - start))