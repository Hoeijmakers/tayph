
import pkg_resources
import os
import pdb
from astropy.io import fits
import astropy.constants as const
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

import sys
import tayph.util as ut
from tayph.vartests import typetest,dimtest
import tayph.tellurics as mol
import tayph.system_parameters as sp
import tayph.functions as fun
import tayph.operations as ops
from tayph.ccf import xcor
import copy
import scipy.interpolate as interp
import pickle
from pathlib import Path
import warnings
import glob
from scipy import interpolate
import tayph.masking as masking
import subprocess
import textwrap
from astropy.utils.data import download_file

from .phoenix import get_phoenix_wavelengths, get_phoenix_model_spectrum


def read_e2ds(inpath,outname,read_s1d=True,mode='HARPS',measure_RV=True,star='solar'):
    """This is the workhorse for reading in a time-series of archival 2D echelle
    spectra from a couple of instrument pipelines that produce a standard output,
    and formatting these into the order-wise FITS format that Tayph uses. These

    The user should point this script to a folder (located at inpath) that contains
    their pipeline-reduced echelle spectra. The script expects a certain data
    format, depending on the instrument in question. It is designed to accept
    pipeline products of the HARPS, HARPS-N, ESPRESSO, CARMENES and UVES instruments. In the
    case of HARPS, HARPS-N, CARMENES and ESPRESSO these may be downloaded from the archive.
    UVES is a bit special, because the 2D echelle spectra are not a standard
    pipeline output. Typical use cases are explained further below.

    This script formats the time series of 2D echelle spectra into 2D FITS images,
    where each FITS file is the time-series of a single echelle order. If the
    spectrograph has N orders, an order spans npx pixels, and M exposures
    were taken during the time-series, there will be N FITS files, each measuring
    M rows by npx columns. This script will read the headers of the pipeline
    reduced data files to determine the date/time of each, the exposure time, the
    barycentric correction (without applying it) and the airmass, and writes
    these to an ASCII table along with the FITS files containing the spectral
    orders.

    By setting the measure_RV keyword (True by default), the code will run a preliminary
    cross-correlation with a stellar (PHOENIX) and a telluric template at both vaccuum
    and air wavelengths to provide the user with information about the nature of the
    adopted wavelength solution and possible wavelength shifts. the star keyword
    allows the user to switch between 3 stellar PHOENIX templates. A solar type (6000K),
    a hot (9000K) or a cool (4000K) template are available, and accessed by setting the
    star keyword to 'solar', 'hot' or 'cool' respectively.

    A crucial functionality of Tayph is that it also acts as a wrapper
    for the Molecfit telluric correction software. If installed properly, the
    user can call this script with the read_s1d keyword to extract 1D spectra from
    the time-series. The tayph.tellurics.molecfit function can then be used to
    let Molecfit loop ver the entire timeseries. To enable this functionality,
    the current script needs to read the full-width, 1D spectra that are output
    by the instrument pipelines. These are saved along with the 2D data.
    Tayph.tellurics.molecfit can then be called separately on the output data-folder
    to apply molecfit to this time-series of 1D spectra, creating a
    time-series of models of the telluric absorption spectrum that is saved along
    with the 2D fits files. Tayph later interpolates these models onto the 2D
    spectra to telluric-correct them, as part of the main dataflow.
    Molecfit is called once in GUI-mode, allowing the user to select the
    relevant fitting regions and parameters, after which it is repeated
    automatically for the entire time series.

    The read_s1d keyword is ignored when reading UVES pipeline data, because the s1d and
    e2ds files are saved in a complex way that I see no point in disentangling.

    The processing of HARPS, HARPS-N and ESPRESSO data is executed in an almost
    identical manner, because the pipeline-reduced products are almost identical.
    To run on either of these instruments, the user simply downloads all pipeline
    products of a given time-series, and extracts these in the same folder, producing
    a sequence of ccfs, e2ds/s2d, s1d, blaze, wave files for each exposure.

    For UVES, the functionality is much more constricted because the pipeline
    reduced data in the ESO archive is generally not of sufficient stability to
    enable precise time-resolved spectroscopy. I designed this function therefore
    to run on intermediary pipeline-products produced by the Reflex (GUI) software. For this,
    a user should download the raw UVES data of their time series, letting ESO's
    calselector tool find the associated calibration files. This can easily be
    many GBs worth of data for a given observing program. The user should then
    reduce these data with the Reflex software. Reflex creates resampled, stitched
    1D spectra as its primary output. However, we will use the intermediate
    pipeline products, which include the 2D extracted orders, located in Reflex's
    working directory after the reduction process is completed.

    A further complication of UVES data is that it can be used with different
    dichroics and 'arms', leading to spectral coverage on the blue, and/or redu and redl
    chips. The user should take care that their time series contains only one blue or
    red types at any time. If they are mixed, this script will throw an exception.
    The blue and red arms are regarded as two different spectrographs (they are), but
    the two red chips (redu and redl) are combined when reading in the data.

    For HARPS, HARPS-N and ESPRESSO, information about the BERV correction is obtained
    from the FITS headers. For UVES, the BERV-correction is calculated using astropy.
    For ESPRESSO, the S2D wavelengths are BERV-corrected, leading to a different wavelength
    solution for each of the exposures. This is not desired, so the BERV-correction is
    undone at the time of reading. ISSUE: THIS COSTS US AN INTERPOLATION.

    Set the nowave keyword to True if the dataset is HARPS or HARPSN, but it has
    no wave files associated with it. This may happen if you downloaded ESO
    Advanced Data Products, which include reduced science e2ds's but not reduced
    wave e2ds's. The wavelength solution is still encoded in the fits header however,
    so we take it from there, instead. This keyword is ignored when not dealing with
    HARPS data.

    Set the ignore_exp keyword to a list of exposures (start counting at 0) that
    need to be ignored when reading, e.g. because they are bad for some reason.
    If you have set molecfit to True, this becomes an expensive parameter to
    play with in terms of computing time, so its better to figure out which
    exposures you'd wish to ignore first (by doing most of your analysis),
    before actually running Molecfit, which is icing on the cake in many use-
    cases in the optical.

    The config parameter points to a configuration file (usually your generic
    run definition file) that is only used to point the Molecfit wrapper to the
    Molecfit installation on your system. If you are not using molecfit, you may
    pass an empty string here.

    """

    print('\n \n \n')
    print('\n \n \n')
    print('\n \n \n')
    print('\n \n \n')
    print(' = = = = = = = = = = = = = = = = =')
    print(' = = = WELCOME TO READ_E2DS! = = =')
    print(' = = = = = = = = = = = = = = = = =')
    print('')
    print(f'    Reading {mode} data')
    print('')
    print(' = = = = = = = = = = = = = = = = =')
    print('\n \n \n')


    #First check the input:
    inpath=ut.check_path(inpath,exists=True)
    typetest(outname,str,'outname in read_e2ds()')
    typetest(read_s1d,bool,'read_s1d switch in read_e2ds()')
    typetest(mode,str,'mode in read_e2ds()')

    if mode not in ['HARPS','HARPSN','HARPS-N','ESPRESSO','UVES-red','UVES-blue']:
        raise ValueError("in read_e2ds: mode needs to be set to HARPS, HARPSN, UVES-red, UVES-blue or ESPRESSO.")
    if measure_RV:
        #Define the paths to the stellar and telluric templates if RV's need to be measured.

        if star.lower()=='solar' or star.lower() =='medium':
            T_eff = 6000
        elif star.lower() =='hot' or star.lower() =='warm':
            T_eff = 9000
        elif star.lower() == 'cool' or star.lower() == 'cold':
            T_eff = 4000
        else:
            warnings.warn(f"in read_e2ds: The star keyword was set to f{star} but only solar, hot or cold are allowed. Assuming a solar template.",RuntimeWarning)
            T_eff = 6000

        print(f'---Downloading / reading diagnostic telluric and {star} PHOENIX models.')

        #Load the PHOENIX spectra from the Goettingen server:
        fxm = get_phoenix_model_spectrum(T_eff=T_eff)
        wlm = get_phoenix_wavelengths()/10.0#Angstrom to nm.

        #Load the telluric spectrum from my Google Drive:
        telluric_link = 'https://drive.google.com/uc?export=download&id=1yAtoLwI3h9nvZK0IpuhLvIpNc_1kjxha'
        telpath = download_file(telluric_link)
        ttt=fits.getdata(telpath)
        os.remove(telpath)#Free up the drive again.
        fxt=ttt[1]
        wlt=ttt[0]

        #Continuum-subtract the telluric model
        wlte,fxte=ops.envelope(wlt,fxt,2.0,selfrac=0.05,threshold=0.8)
        e_t = interpolate.interp1d(wlte,fxte,fill_value='extrapolate')(wlt)
        fxtn = fxt/e_t
        fxtn[fxtn>1]=1.0

        wlt=np.concatenate((np.array([100,200,300,400]),wlt))
        wlt=ops.vactoair(wlt)
        fxtn=np.concatenate((np.array([1.0,1.0,1.0,1.0]),fxtn))
        fxtn[wlt<500.1]=1.0

        #Continuum-subtract the PHOENIX model
        wlme,fxme=ops.envelope(wlm,fxm,5.0,selfrac=0.005)
        e_s = interpolate.interp1d(wlme,fxme,fill_value='extrapolate')(wlm)
        fxmn=fxm/e_s
        fxmn[fxmn>1.0]=1.0
        fxmn[fxmn<0.0]=0.0

        fxmn=fxmn[wlm>300.0]
        wlm=wlm[wlm>300.0]
        wlm=ops.vactoair(wlm)#Everything in the world is in air.
        #The stellar and telluric templates are now ready for application in cross-correlation at the very end of this script.


    outpath = Path('data/'+outname)
    if os.path.exists(outpath) != True:
        os.makedirs(outpath)



    #Start reading data files.
    filelist=os.listdir(inpath)#If mode == UVES, these are folders. Else, they are fits files.
    if len(filelist) == 0:
        raise FileNotFoundError(f" in read_e2ds: input folder {str(inpath)} is empty.")


    #MODE SWITCHING AND READING:
    #ADD READ_S1D OPTION. RIGHT NOW S1Ds ARE REQUIRED BY THE BELOW READING FUNCTIONS,
    #EVEN THOUGH THE USER HAS A SWITCH TO TURN THIS OFF.
    if mode=='HARPS-N': mode='HARPSN'#Guard against various ways of referring to HARPS-N.
    print(f'---Read_e2ds is attempting to read a {mode} datafolder at {str(inpath)}.')
    print('---These are the files encountered:')
    if mode in ['HARPS','HARPSN']:
        DATA = read_harpslike(inpath,filelist,mode,read_s1d=read_s1d)#This is a dictionary containing all the e2ds files, wavelengths, s1d spectra, etc.
    elif mode in ['UVES-red','UVES-blue']:
        DATA = read_uves(inpath,filelist,mode)
    elif mode == 'ESPRESSO':
        DATA = read_espresso(inpath,filelist,read_s1d=read_s1d)
    elif mode == 'CARMENES':
        DATA = read_carmenes(inpath,filelist,construct_s1d=read_s1d)
    else:
        raise ValueError(f'Error in read_e2ds: {mode} is not a valid instrument mode.')


    #There is a problem in this way of dealing with airmass and that is that the
    #derivative of the airmass may not be constant for higher airmass, meaning that
    #the average should be weighted in some way. This can probably be achieved by using
    #information about the start end and time duration between the two, but its a bit
    #more difficult. All the more reason to keep exposures short...


    wave    = DATA['wave']#List of 2D wavelength frames for each echelle order.
    e2ds    = DATA['e2ds']#List of 2D extracted echelle orders.
    wave1d  = DATA['wave1d']#List of 1D stitched wavelengths.
    s1d     = DATA['s1d']#List of 1D stiched spectra.
    norders     = DATA['norders']#Vertical size of each e2ds file.
    npx         = DATA['npx']#Horizontal size of each e2ds file.
    framename   = DATA['framename']#Name of file read.
    mjd     = DATA['mjd']#MJD of observation.
    s1dmjd  = DATA['s1dmjd']#MJD of 1D stitched spectrum (should be the same as MJD).
    header     = DATA['header']#List of header objects.
    s1dhdr  = DATA['s1dhdr']#List of S1D headers. Used for passing weather information to Molecfit.
    berv    = DATA['berv']#Radial velocity of the Earth w.r.t. the solar system barycenter.
    airmass = DATA['airmass']#The average airmass of the observation ((end-start)/2)
    texp = DATA['texp']#The exposure time of each exposure in seconds.
    date = DATA['date']#The date of observation in yyyymmddThhmmss.s format.
    obstype = DATA['obstype']#Observation type, should be SCIENCE.
    del DATA#Free memory.

    #Typetest all of these:
    typetest(wave,list,'wave in read_e2ds()')
    typetest(e2ds,list,'e2ds in read_e2ds()')
    typetest(wave1d,list,'wave1d in read_e2ds()')
    typetest(s1d,list,'s1d in read_e2ds()')
    typetest(date,list,'date in read_e2ds()')
    typetest(norders,np.ndarray,'norders in read_e2ds()')
    typetest(npx,np.ndarray,'npx in read_e2ds()')
    typetest(mjd,np.ndarray,'mjd in read_e2ds()')
    typetest(s1dmjd,np.ndarray,'s1dmjd in read_e2ds()')
    typetest(berv,np.ndarray,'berv in read_e2ds()')
    typetest(airmass,np.ndarray,'airmass in read_e2ds()')
    typetest(texp,np.ndarray,'texp in read_e2ds()')
    e2ds_count  = len(e2ds)



    #Now we catch some things that could be wrong with the data read:
    #1-The above should have read a certain number of e2ds files that are classified as SCIENCE frames.
    #2-There should be equal numbers of wave and e2ds frames and wave1d as s1d frames.
    #3-The number of s1d files should be the same as the number of e2ds files.
    #4-All exposures should have the same number of spectral orders.
    #5-All orders should have the same number of pixels.
    #6-The wave frames should have the same dimensions as the e2ds frames.

    #Test 1
    if e2ds_count == 0:
        raise FileNotFoundError(f"in read_e2ds: The input folder {str(inpath)} does not contain files recognised as e2ds-format FITS files with SCIENCE keywords.")
    #Test 2,3
    if len(wave1d) != len(s1d) and read_s1d == True:
        raise ValueError(f"in read_e2ds: The number of 1D wavelength axes and S1D files does not match ({len(wave1d)},{len(s1d)}). Between reading and this test, one of them has become corrupted. This should not realisticly happen.")
    if len(wave) != len(e2ds):
        raise ValueError(f"in read_e2ds: The number of 2D wavelength frames and e2ds files does not match ({len(wave)},{len(e2ds)}). Between reading and this test, one of them has become corrupted. This should not realisticly happen.")
    if len(e2ds) != len(s1d) and read_s1d == True:
        raise ValueError(f"in read_e2ds: The number of e2ds files and s1d files does not match ({len(e2ds)},{len(s1d)}). Make sure that s1d spectra matching the e2ds spectra are provided.")
    #Test 4 - and set norders to its constant value.
    if np.max(np.abs(norders-norders[0])) == 0:
        norders=int(norders[0])
    else:
        print('\n \n \n')
        print("These are the files and their numbers of orders:")
        for i in range(e2ds_count):
            print('   '+framename[i]+'  %s' % norders[i])
        raise ValueError("in read_e2ds: Not all e2ds files have the same number of orders. The list of frames is printed above.")
    #Test 5 - and set npx to its constant value.
    if np.max(np.abs(npx-npx[0])) == 0:
        npx=int(npx[0])
    else:
        print('\n \n \n')
        print("These are the files and their numbers of pixels:")
        for i in range(len(obstype)):
            print('   '+framename[i]+'  %s' % npx[i])
        raise ValueError("in read_e2ds: Not all e2ds files have the same number of pixels. The list of frames is printed above.")
    #Test 6
    for i in range(e2ds_count):
        if np.shape(wave[i])[0] != np.shape(e2ds[i])[0] or np.shape(wave[i])[1] != np.shape(e2ds[i])[1]:
            raise ValueError(f"in read_e2ds: wave[{i}] does not have the same dimensions as e2ds[{i}] ({np.shape(wave)},{np.shape(e2ds)}).")
        if len(wave1d[i]) != len(s1d[i]) and read_s1d == True:
            raise ValueError(f"in read_e2ds: wave1d[{i}] does not have the same length as s1d[{i}] ({len(wave)},{len(e2ds)}).")
        if read_s1d == True:
            dimtest(wave1d[i],np.shape(s1d[i]),f'wave1d[{i}] and s1d[{i}] in read_e2ds()')#This error should never be triggered, but just in case.

    #Ok, so now we should have ended up with a number of lists that contain all
    #the relevant science data and associated information.
    #We determine how to sort the resulting lists in time:
    sorting = np.argsort(mjd)

    #If s1ds were read, we now sort them and save them to the output folder:
    if read_s1d == True:
        print('',end='\r')
        print('')
        print(f'---Saving S1D files to {str(outpath/"s1ds.pkl")}')
        s1dsorting = np.argsort(s1dmjd)
        if len(sorting) != len(s1dsorting):
            raise ValueError("in read_e2ds: Sorted science frames and sorted s1d frames are not of the same length. Telluric correction can't proceed. Make sure that the number of files of each type is correct.")
        s1dhdr_sorted=[]
        s1d_sorted=[]
        wave1d_sorted=[]
        for i in range(len(s1dsorting)):
            s1dhdr_sorted.append(s1dhdr[s1dsorting[i]])
            s1d_sorted.append(s1d[s1dsorting[i]])
            wave1d_sorted.append(wave1d[sorting[i]])
            #Sort the s1d files for application of molecfit.
        #Store the S1Ds in a pickle file. I choose not to use a FITS image because the dimensions of the s1ds can be different from one exposure to another.
        with open(outpath/'s1ds.pkl', 'wb') as f: pickle.dump([s1dhdr_sorted,s1d_sorted,wave1d_sorted],f)

    #Provide diagnostic output.
    print('---These are the filetypes and observing date-times recognised.')
    print('---These should all be SCIENCE and in chronological order.')
    for i in range(len(sorting)): print(f'------{obstype[sorting[i]]}  {date[sorting[i]]}  {mjd[sorting[i]]}')

    #CONTINUE HERE! SPLIT OFF MOLECFIT STRAIGHT AND SAVE 2D WAVE FILES!

    #If we run diagnostic cross-correlations, we prepare to store output:
    if measure_RV:
        list_of_orders=[]
        list_of_waves=[]
    #Now we loop over all exposures and collect the i-th order from each exposure,
    #put these into a new matrix and save them to FITS images:
    f=open(outpath/'obs_times','w',newline='\n')
    headerline = 'MJD'+'\t'+'DATE'+'\t'+'EXPTIME'+'\t'+'MEAN AIRMASS'+'\t'+'BERV (km/s)'+'\t'+'FILE NAME'
    # pdb.set_trace()
    for i in range(norders):
        order = np.zeros((len(sorting),npx))
        wave_order = copy.deepcopy(order)
        # print('CONSTRUCTING ORDER %s' % i)
        print(f"---Constructing order {i}", end="\r")
        for j in range(len(sorting)):#Loop over exposures
            exposure = e2ds[sorting[j]]
            wave_exposure = wave[sorting[j]]
            order[j,:] = exposure[i,:]
            wave_order[j,:] = wave_exposure[i,:]
            #Now I also need to write it to file.
            if i ==0:#Only do it the first time, not for every order.
                line = str(mjd[sorting[j]])+'\t'+date[sorting[j]]+'\t'+str(texp[sorting[j]])+'\t'+str(np.round(airmass[sorting[j]],3))+'\t'+str(np.round(berv[sorting[j]],5))+'\t'+framename[sorting[j]]+'\n'
                f.write(line)


        if mode in ['UVES-red','UVES-blue','ESPRESSO']:#UVES and ESPRESSO pipelines pad with zeroes. We remove these.
            # order[np.abs(order)<1e-10*np.nanmedian(order)]=0.0#First set very small values to zero.
            npx_order = np.shape(order)[1]
            sum=np.nansum(np.abs(order),axis=0)#Anything that is zero in this sum (i.e. the edges) will be clipped.
            leftsize=npx_order-len(np.trim_zeros(sum,'f'))#Size of zero-padding on the left
            rightsize=npx_order-len(np.trim_zeros(sum,'b'))#Size of zero-padding on the right
            order = order[:,leftsize:npx_order-rightsize-1]
            wave_order=wave_order[:,leftsize:npx_order-rightsize-1]

        fits.writeto(outpath/('order_'+str(i)+'.fits'),order,overwrite=True)
        fits.writeto(outpath/('wave_'+str(i)+'.fits'),wave_order,overwrite=True)
        if measure_RV:
            list_of_orders.append(order)
            list_of_waves.append(wave_order)
    f.close()
    print('\n \n \n')
    print(f'---Time-table written to {outpath/"obs_times"}.')















    #The rest is for if diagnostic radial velocity measurements are requested.
    #We take and plot all the 1D and 2D spectra, do a rudimentary cleaning (de-colour, outlier correction)
    #and perform cross-correlations with solar and telluric templates.
    if measure_RV:
        #From this moment on, I will start reusing some variable that I named above, i.e. overwriting them.
        drv=1.0
        RVrange=160.0
        print(f'---Preparing for diagnostic correlation with a telluric template and a {star} PHOENIX model.')
        print(f'---The wavelength axes of the spectra will be shown here as they were saved by read_e2ds.')
        print(f'---Cleaning 1D spectra for cross-correlation.')

        if mode in ['HARPS','HARPSN','ESPRESSO']:
            #gamma = (1.0-(berv[0]*u.km/u.s/const.c).decompose().value)
            wave_1d = wave1d[0]/10.0#*gamma#Universal berv-un-corrected wavelength axis in nm in air.
            s1d_block=np.zeros((len(s1d),len(wave_1d)))
            for i in range(0,len(s1d)):
                # wave = wave1d[i]#(s1dhdr[i]['CDELT1']*fun.findgen(len(s1d[i]))+s1dhdr[i]['CRVAL1'])
                s1d_block[i]=interp.interp1d(wave1d[i]/10.0,s1d[i],bounds_error=False,fill_value='extrapolate')(wave_1d)
        #
        # plt.plot(wave_1d,s1d[0]/np.mean(s1d[0]),linewidth=0.5,label='Berv-un-corrected in nm.')
        # plt.plot(wave1d[1]/10.0,s1d[1]/np.mean(s1d[1]),linewidth=0.5,label='Original (bervcorrected) from A to nm.')
        # plt.plot(wlt,fxtn,label='Telluric')
        # plt.legend()
        # plt.show()
        # pdb.set_trace()
        wave_1d,s1d_block,r1,r2=ops.clean_block(wave_1d,s1d_block,deg=4,verbose=True,renorm=False)#Slow. Needs parallelising.

        if mode in ['UVES-red','UVES-blue']:
            pdb.set_trace()

        # ut.writefits('test.fits',s1d_block)
        # plt.plot(s1d_mjd,s1d_berv,'.')
        # plt.show()
        # for i in range(len(s1d)): plt.plot(wave_1d,s1d_block[i],color='blue',alpha=0.2)
        # for i in range(len(s1d)): plt.plot(wave_1d_berv_corrected,s1d_block_berv_corrected[i],color='red',alpha=0.2)
        # plt.show()

        print(f'---Cleaning 2D orders for cross-correlation.')
        list_of_orders_trimmed=[]
        list_of_waves_trimmed=[]
        for i,o in enumerate(list_of_orders):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w_i = list_of_waves[i][0]
                o_i = np.zeros((len(list_of_orders[i]),len(w_i)))
                for j in range(len(list_of_orders[i])):
                    o_i[j]=interp.interp1d(list_of_waves[i][j],list_of_orders[i][j],bounds_error=False,fill_value='extrapolate')(w_i)
                w_trimmed,o_trimmed,r1,r2=ops.clean_block(w_i,o_i,w=100,deg=4,renorm=True)#Slow. Needs parallelising.
                list_of_waves_trimmed.append(w_trimmed)
                list_of_orders_trimmed.append(o_trimmed)
                ut.statusbar(i,len(list_of_orders))


        print(f"---Interpolating over NaN's")
        list_of_orders_trimmed = masking.interpolate_over_NaNs(list_of_orders_trimmed)#Slow, also needs parallelising.
        s1d_block=masking.interpolate_over_NaNs([s1d_block])[0]
        print(f'---Performing cross-correlation and plotting output.')
        rv,ccf,Tsums=xcor(list_of_waves_trimmed,list_of_orders_trimmed,np.flipud(np.flipud(wlm)),fxmn-1.0,drv,RVrange)
        rv1d,ccf1d,Tsums1d=xcor([wave_1d],[s1d_block],np.flipud(np.flipud(wlm)),fxmn-1.0,drv,RVrange)
        rvT,ccfT,TsusmT=xcor([wave_1d],[s1d_block],np.flipud(np.flipud(wlt)),fxtn-1.0,drv,RVrange)
        rvT2D,ccfT2D,TsusmT2D=xcor(list_of_waves_trimmed,list_of_orders_trimmed,np.flipud(np.flipud(wlt)),fxtn-1.0,drv,RVrange)
        #The rest is for plotting.
        plt.figure(figsize=(13,5))
        minwl=np.inf#For determining in the for-loop what the minimum and maximum wl range of the data is.
        maxwl=0.0

        mean_of_orders = np.nanmean(np.hstack(list_of_orders_trimmed))
        for i in range(len(list_of_orders)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s_avg = np.nanmean(list_of_orders_trimmed[i],axis=0)
            minwl=np.min([np.min(list_of_waves[i]),minwl])
            maxwl=np.max([np.max(list_of_waves[i]),maxwl])

            if i == 0:
                plt.plot(list_of_waves_trimmed[i],s_avg/mean_of_orders,color='red',linewidth=0.9,alpha=0.5,label='2D echelle orders to be cross-correlated')
            else:
                plt.plot(list_of_waves_trimmed[i],s_avg/mean_of_orders,color='red',linewidth=0.7,alpha=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s1d_avg=np.nanmean(s1d_block,axis=0)
        plt.plot(wave_1d,s1d_avg/np.nanmean(s1d_avg),color='orange',label='1D spectrum to be cross-correlated',linewidth=0.9,alpha=0.5)

        plt.title(f'Time-averaged spectral orders and {star} PHOENIX model')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Flux (normalised to order average)')
        plt.plot(wlm,fxmn,color='green',linewidth=0.7,label='PHOENIX template (air, used in ccf)',alpha=0.5)
        plt.plot(wlt,fxtn,color='blue',linewidth=0.7,label='Skycalc telluric model',alpha=0.6)
        plt.xlim(minwl-5,maxwl+5)#Always nm.
        plt.legend(loc='upper right',fontsize=8)
        plt.show()

        centroids2d=[]
        centroids1d=[]
        centroidsT2d=[]
        centroidsT1d=[]



        plt.figure(figsize=(13,5))
        for n,i in enumerate(ccf):
            if n == 0:
                plt.plot(rv,i/np.nanmean(i),linewidth=0.7,alpha=0.3,color='black',label='2D orders-PHOENIX')
            else:
                plt.plot(rv,i/np.nanmean(i),linewidth=0.7,alpha=0.3,color='black')
            centroids2d.append(rv[np.argmin(i)])
        for n,i in enumerate(ccf1d):
            if n == 0:
                plt.plot(rv1d,i/np.nanmean(i),linewidth=0.7,alpha=0.3,color='red',label='1D spectra-PHOENIX')
            else:
                plt.plot(rv1d,i/np.nanmean(i),linewidth=0.7,alpha=0.3,color='red')
            centroids1d.append(rv1d[np.argmin(i)])
        for n,i in enumerate(ccfT):
            if n == 0:
                plt.plot(rvT,i/np.nanmean(i),linewidth=0.7,alpha=0.2,color='blue',label='1D spectra-TELLURIC')
            else:
                plt.plot(rvT,i/np.nanmean(i),linewidth=0.7,alpha=0.2,color='blue')
            centroidsT1d.append(rvT[np.argmin(i)])
        for n,i in enumerate(ccfT2D):
            if n == 0:
                plt.plot(rvT2D,i/np.nanmean(i),linewidth=0.7,alpha=0.2,color='navy',label='2D orders-TELLURIC')
            else:
                plt.plot(rvT2D,i/np.nanmean(i),linewidth=0.7,alpha=0.2,color='navy')
            centroidsT2d.append(rvT2D[np.argmin(i)])
        plt.axvline(np.nanmean(centroids2d),color='black',alpha=0.5)
        plt.axvline(np.nanmean(centroids1d),color='red',alpha=0.5)
        plt.axvline(np.nanmean(centroidsT1d),color='blue',alpha=0.5)
        plt.axvline(np.nanmean(centroidsT2d),color='navy',alpha=1.0)
        plt.title(f'CCF between {mode} data and {star} PHOENIX and telluric models. See commentary in terminal for details',fontsize=9)
        plt.xlabel('Radial velocity (km/s)')
        plt.ylabel('Mean flux')
        plt.legend()
        print('\n \n \n')

        print('The derived line positions are as follows:')
        print(f'1D spectra with PHOENIX:  Line center near RV = {np.round(np.nanmean(centroids1d),1)} km/s.')
        print(f'1D spectra with tellurics:  Line center near RV = {np.round(np.nanmean(centroidsT1d),1)} km/s.')
        print(f'2D orders with PHOENIX:  Line center near RV ={np.round(np.nanmean(centroids2d),1)} km/s.')
        print(f'2D orders with tellurics:  Line center near RV ={np.round(np.nanmean(centroidsT2d),1)} km/s.')
        print('\n \n \n')


        terminal_height,terminal_width = subprocess.check_output(['stty', 'size']).split()#Get the window size of the terminal, from https://stackoverflow.com/questions/566746/how-to-get-linux-console-window-width-in-python
        if mode == 'ESPRESSO':
            explanation=[f'For ESPRESSO, the S1D and S2D spectra are typically \
provided in the barycentric frame, in air. Because the S1D spectra are used for \
telluric correction, this code undoes this barycentric correction so that the \
S1Ds are saved in the telluric rest-frame while the S2D spectra remain in the \
barycentric frame.','','Therefore, you should see the following values for the \
the measured line centers:' ,\
f'- The 1D spectra correlated with PHOENIX should peak at the systemic velocity minus \
the BERV correction (equal to {np.round(np.nanmean(berv),2)} km/s on average).', \
'- The 1D spectra correlated with the telluric model should peak at 0 km/s.', \
'- The 2D spectra correlated with PHOENIX should peak the systemic velocity of this star.', \
'- The 2D spectra correlated with the tellurics should peak at the barycentric velocity.','', \
'If this is correct, in the Tayph runfile of this dataset, do_berv_correction should be set to False and \
air in the config file of this dataset should be set to True.']


        if mode in ['HARPS','HARPSN']:
            explanation=[f'For HARPS, the s1d spectra are typically \
provided in the barycentric restframe, while the e2ds spectra are left in the observatory \
frame, both in air. Because the s1d spectra are used for telluric correction, this code undoes \
this barycentric correction so that the s1ds are saved in the telluric rest-frame so that they \
end up in the same frame as the e2ds spectra.','', \
'Therefore, you should see the following values for the the measured line centers:', \
f'- The 1D spectra correlated with PHOENIX should peak at the systemic velocity minus \
the BERV correction (equal to {np.round(np.nanmean(berv),2)} km/s on average).', \
'- The 1D spectra correlated with the telluric model should peak at 0 km/s.', \
f'- The 2D spectra correlated with PHOENIX should peak the systemic velocity minus the \
 BERV correction (equal to {np.round(np.nanmean(berv),2)} km/s on average).', \
'- The 2D spectra correlated with the tellurics should peak at 0 km/s.','', \
'If this is correct, in the Tayph runfile of this dataset, do_berv_correction should be set to True and air \
in the config file of this dataset should be set to True.']


        for s in explanation: print(textwrap.fill(s, width=int(terminal_width)-5))
        print('\n \n \n')
        final_notes = "Large deviations can occur if the wavelength solution from the pipeline is \
incorretly assumed to be in air, or if for whatever reason, the wavelength solution is \
provide in the reference frame of the star. In the former case, you need to specify in \
the CONIG file of the data that the wavelength solution written by this file is in vaccuum. \
This wil be read by Tayph and Molecfit. \
In the atter case, barycentric correction (and possibly Keplerian corrections) are likely \
implicily taken into account in the wavelength solution, meaning that these corrections need \
to be sitched off when running the Tayph's cascade. Molecfit can't be run in the default way \
in thiscase, because the systemic velocity is offsetting the telluric spectrum. \
If no peak is visible at all, the spectral orders are suffering from unknown velocity shifts \
or contnuum fluctuations that this code was not able to take out. In this case, please inspect \
your daa carefully to determine whether you can locate the source of the problem. Continuing to \
run Tayph from this point on would probably make it very difficult to obtain meaningful cross-correlations."
        print(textwrap.fill(final_notes, width=int(terminal_width)-5))
        plt.show()

    print('\n \n \n')
    print('Read_e2ds completed successfully.')
    print('')










def read_harpslike(inpath,filelist,mode,read_s1d=True):
    """This reads a folder of HARPS or HARPSN data. Input is a list of filepaths and the mode (HARPS or HARPSN)."""

    if mode=='HARPS':
        catkeyword = 'HIERARCH ESO DPR CATG'
        bervkeyword = 'HIERARCH ESO DRS BERV'
        thfilekeyword = 'HIERARCH ESO DRS CAL TH FILE'
        Zstartkeyword = 'HIERARCH ESO TEL AIRM START'
        Zendkeyword = 'HIERARCH ESO TEL AIRM END'
    elif mode=='HARPSN':
        catkeyword = 'OBS-TYPE'
        bervkeyword = 'HIERARCH TNG DRS BERV'
        thfilekeyword = 'HIERARCH TNG DRS CAL TH FILE'
        Zstartkeyword = 'AIRMASS'
        Zendkeyword = 'AIRMASS'#These are the same because HARPSN doesnt have start and end keywords.
        #Down there, the airmass is averaged, so there is no problem in taking the average of the same number.
    else:
        raise ValueError(f"Error in read_harpslike: mode should be set to HARPS or HARPSN ({mode})")

    #The following variables define lists in which all the necessary data will be stored.
    framename=[]
    header=[]
    s1dhdr=[]
    obstype=[]
    texp=np.array([])
    date=[]
    mjd=np.array([])
    s1dmjd=np.array([])
    npx=np.array([])
    norders=np.array([])
    e2ds=[]
    s1d=[]
    wave1d=[]
    airmass=np.array([])
    berv=np.array([])
    wave=[]
    # wavefile_used = []
    for i in range(len(filelist)):
        if filelist[i].endswith('e2ds_A.fits'):
            print(f'------{filelist[i]}', end="\r")
            hdul = fits.open(inpath/filelist[i])
            data = copy.deepcopy(hdul[0].data)
            hdr = hdul[0].header
            hdul.close()
            del hdul[0].data
            if hdr[catkeyword] == 'SCIENCE':
                framename.append(filelist[i])
                header.append(hdr)
                obstype.append(hdr[catkeyword])
                texp=np.append(texp,hdr['EXPTIME'])
                date.append(hdr['DATE-OBS'])
                mjd=np.append(mjd,hdr['MJD-OBS'])
                npx=np.append(npx,hdr['NAXIS1'])
                norders=np.append(norders,hdr['NAXIS2'])
                e2ds.append(data)
                berv=np.append(berv,hdr[bervkeyword])
                airmass=np.append(airmass,0.5*(hdr[Zstartkeyword]+hdr[Zendkeyword]))#This is an approximation where we take the mean airmass.
                # if nowave == True:
                # wavefile_used.append(hdr[thfilekeyword])
                #Record which wavefile was used by the pipeline to
                #create the wavelength solution.
                wavedata=ut.read_wave_from_e2ds_header(hdr,mode=mode)/10.0#convert to nm.
                wave.append(wavedata)
                # if filelist[i].endswith('wave_A.fits'):
                #     print(filelist[i]+' (wave)')
                #     if nowave == True:
                #         warnings.warn(" in read_e2ds: nowave was set to True but a wave_A file was detected. This wave file is now ignored in favor of the header.",RuntimeWarning)
                #     else:
                #         wavedata=fits.getdata(inpath/filelist[i])
                #         wave.append(wavedata)

                if read_s1d:
                    s1d_path=inpath/Path(str(filelist[i]).replace('e2ds_A.fits','s1d_A.fits'))
                    ut.check_path(s1d_path,exists=True)#Crash if the S1D doesn't exist.
        # if filelist[i].endswith('s1d_A.fits'):
                    hdul = fits.open(s1d_path)
                    data_1d = copy.deepcopy(hdul[0].data)
                    hdr1d = hdul[0].header
                    hdul.close()
                    del hdul
            # if hdr[catkeyword] == 'SCIENCE':
                    s1d.append(data_1d)
                # npx1d.append(len(data_1d))
                    s1dhdr.append(hdr1d)
                    s1dmjd=np.append(s1dmjd,hdr1d['MJD-OBS'])
                    berv1d = hdr1d[bervkeyword]
                    if berv1d != hdr[bervkeyword]:
                        wrn_msg = ('WARNING in read_harpslike(): BERV correction of s1d file is not'
                        f'equal to that of the e2ds file. {berv1d} vs {hdr[bervkeyword]}')
                        ut.tprint(wrn_msg)
                    gamma = (1.0-(berv1d*u.km/u.s/const.c).decompose().value)#Doppler factor BERV.
                    wave1d.append((hdr1d['CDELT1']*fun.findgen(len(data_1d))+hdr1d['CRVAL1'])*gamma)
    if mode == 'HARPSN': #In the case of HARPS-N we need to convert the units of the elevation and provide a UTC keyword.
        for i in range(len(header)):
            s1dhdr[i]['TELALT'] = np.degrees(float(s1dhdr[i]['EL']))
            s1dhdr[i]['UTC'] = (float(s1dhdr[i]['MJD-OBS'])%1.0)*86400.0
    #Check that all exposures have the same number of pixels, and clip s1ds if needed.
    # min_npx1d = int(np.min(np.array(npx1d)))
    # if np.sum(np.abs(np.array(npx1d)-npx1d[0])) != 0:
    #     warnings.warn("in read_e2ds when reading HARPS data: Not all s1d files have the same number of pixels. This could have happened if the pipeline has extracted one or two extra pixels in some exposures but not others. The s1d files will be clipped to the smallest length.",RuntimeWarning)
    #     for i in range(len(s1d)):
    #         wave1d[i]=wave1d[i][0:min_npx1d]
    #         s1d[i]=s1d[i][0:min_npx1d]
    #         npx1d[i]=min_npx1d
    output = {'wave':wave,'e2ds':e2ds,'header':header,'wave1d':wave1d,'s1d':s1d,'s1dhdr':s1dhdr,'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,'norders':norders,'berv':berv,'airmass':airmass,'s1dmjd':s1dmjd}
    return(output)




def read_uves(inpath,filelist,mode):
    """This reads a folder of UVES-blue or UVES-red intermediate pipeline products. Input is a list of filepaths and the mode."""
    #The following variables define lists in which all the necessary data will be stored.
    framename=[]
    header=[]
    s1dhdr=[]
    obstype=[]
    texp=np.array([])
    date=[]
    mjd=np.array([])
    s1dmjd=np.array([])
    npx=np.array([])
    npx1d=np.array([])
    norders=np.array([])
    e2ds=[]
    s1d=[]
    wave1d=[]
    airmass=np.array([])
    berv=np.array([])
    wave=[]
    catkeyword = 'HIERARCH ESO DPR CATG'
    bervkeyword = 'HIERARCH ESO DRS BERV'
    thfilekeyword = 'HIERARCH ESO DRS CAL TH FILE'
    Zstartkeyword = 'HIERARCH ESO TEL AIRM START'
    Zendkeyword = 'HIERARCH ESO TEL AIRM END'
    for i in range(len(filelist)):
        print(f'------{filelist[i]}', end="\r")
        if (inpath/Path(filelist[i])).is_dir():
            tmp_products = [i for i in (inpath/Path(filelist[i])).glob('resampled_science_*.fits')]
            tmp_products1d = [i for i in (inpath/Path(filelist[i])).glob('red_science_*.fits')]
            if mode == 'UVES-red' and len(tmp_products) != 2:
                raise ValueError(f"in read_e2ds: When mode=UVES-red there should be 2 resampled_science files (redl and redu), but {len(tmp_products)} were detected in {str(inpath/Path(filelist[i]))}.")
            if mode == 'UVES-blue' and len(tmp_products) != 1:
                raise ValueError(f"in read_e2ds: When mode=UVES-rblue there should be 1 resampled_science files (blue), but {len(tmp_products)} were detected in {str(inpath/Path(filelist[i]))}.")
            if mode == 'UVES-red' and len(tmp_products1d) != 2:
                raise ValueError(f"in read_e2ds: When mode=UVES-red there should be 2 red_science files (redl and redu), but {len(tmp_products1d)} were detected in {str(inpath/Path(filelist[i]))}.")
            if mode == 'UVES-blue' and len(tmp_products1d) != 1:
                raise ValueError(f"in read_e2ds: When mode=UVES-rblue there should be 1 red_science files (blue), but {len(tmp_products1d)} were detected in {str(inpath/Path(filelist[i]))}.")

            data_combined = []#This will store the two chips (redu and redl) in case of UVES_red, or simply the blue chip if otherwise.
            wave_combined = []
            wave1d_combined=[]
            data1d_combined=[]
            norders_tmp = 0
            for tmp_product in tmp_products:
                hdul = fits.open(tmp_product)
                data = copy.deepcopy(hdul[0].data)
                hdr = hdul[0].header
                hdul.close()
                del hdul[0].data
                if not hdr['HIERARCH ESO PRO SCIENCE']:#Only add if it's actually a science product:#I force the user to supply only science exposures in the input  folder. No BS allowed... UVES is hard enough as it is.
                    raise ValueError(f' in read_e2ds: UVES file {tmp_product} is not classified as a SCIENCE file, but should be. Remove it from the folder?')
                wavedata=ut.read_wave_from_e2ds_header(hdr,mode='UVES')/10.0#Convert to nm.
                data_combined.append(data)
                wave_combined.append(wavedata)
                norders_tmp+=np.shape(data)[0]

            for tmp_product in tmp_products1d:
                hdul = fits.open(tmp_product)
                data_1d = copy.deepcopy(hdul[0].data)
                hdr1d = hdul[0].header
                hdul.close()
                del hdul[0].data
                if not hdr1d['HIERARCH ESO PRO SCIENCE']:#Only add if it's actually a science product:#I force the user to supply only science exposures in the input  folder. No BS allowed... UVES is hard enough as it is.
                    raise ValueError(f' in read_e2ds: UVES file {tmp_product} is not classified as a SCIENCE file, but should be. Remove it from the folder?')
                npx_1d = hdr1d['NAXIS1']
                wavedata = fun.findgen(npx_1d)*hdr1d['CDELT1']+hdr1d['CRVAL1']
                data1d_combined.append(data_1d)
                wave1d_combined.append(wavedata)

            if len(data_combined) < 1 or len(data_combined) > 2:#Double-checking that length here...
                raise ValueError(f'in read_e2ds(): Expected 1 or 2 chips, but {len(data_combined)} files were somehow read.')
            #The chips generally don't give the same size. Therefore I will pad the smaller one with NaNs to make it fit:
            if len(data_combined) != len(data1d_combined):
                raise ValueError(f'in read_e2ds(): The number of chips in the 1d and 2d spectra is not the same {len(data1d_combined)} vs {len(data_combined)}.')

            if len(data_combined) == 2:
                chip1 = data_combined[0]
                chip2 = data_combined[1]
                wave1 = wave_combined[0]
                wave2 = wave_combined[1]
                npx_1 = np.shape(chip1)[1]
                npx_2 = np.shape(chip2)[1]
                no_1 = np.shape(chip1)[0]
                no_2 = np.shape(chip2)[0]
                npx_max = np.max([npx_1,npx_2])
                npx_min = np.min([npx_1,npx_2])
                diff = npx_max-npx_min
                #Pad the smaller one with NaNs to match the wider one:
                if npx_1 < npx_2:
                    chip1=np.hstack([chip1,np.zeros((no_1,diff))*np.nan])
                    wave1=np.hstack([wave1,np.zeros((no_1,diff))*np.nan])
                else:
                    chip2=np.hstack([chip2,np.zeros((no_2,diff))*np.nan])
                    wave2=np.hstack([wave2,np.zeros((no_2,diff))*np.nan])
                #So now they can be stacked:
                e2ds_stacked = np.vstack((chip1,chip2))
                wave_stacked = np.vstack((wave1,wave2))
                if np.shape(e2ds_stacked)[1] != np.shape(wave_stacked)[1]:
                    raise ValueError("Width of stacked e2ds and stacked wave frame are not the same. Is the wavelength solution in the header of this file correct?")
                npx=np.append(npx,np.shape(e2ds_stacked)[1])

                e2ds.append(e2ds_stacked)
                wave.append(wave_stacked)
                chip1_1d = data1d_combined[0]
                chip2_1d = data1d_combined[1]
                wave1_1d = wave1d_combined[0]
                wave2_1d = wave1d_combined[1]
                if np.nanmean(wave1_1d) < np.nanmean(wave2_1d):
                    combined_data_1d = np.concatenate((chip1_1d,chip2_1d))
                    combined_wave_1d = np.concatenate((wave1_1d,wave2_1d))
                else:
                    combined_data_1d = np.concatenate((chip2_1d,chip1_1d))
                    combined_wave_1d = np.concatenate((wave2_1d,wave1_1d))
                wave1d.append(combined_wave_1d)
                s1d.append(combined_data_1d)
                npx1d=np.append(npx1d,len(combined_wave_1d))
            else:
                e2ds.append(data_combined[0])
                wave.append(wave_combined[0])
                npx=np.append(npx,np.shape(data_combined[0])[1])
                wave1d.append(wave1d_combined[0])
                s1d.append(data1d_combined[0])
                npx1d=np.append(npx1d,len(combined_wave_1d))
            #Only using the keyword from the second header in case of redl,redu.
            s1dmjd=np.append(s1dmjd,hdr1d['MJD-OBS'])
            framename.append(hdr['ARCFILE'])
            header.append(hdr)
            obstype.append('SCIENCE')
            texp=np.append(texp,hdr['EXPTIME'])
            date.append(hdr['DATE-OBS'])
            mjd=np.append(mjd,hdr['MJD-OBS'])
            norders=np.append(norders,norders_tmp)
            airmass=np.append(airmass,0.5*(hdr[Zstartkeyword]+hdr[Zendkeyword]))#This is an approximation where we take the mean airmass.
            berv_i=sp.calculateberv(hdr['MJD-OBS'],hdr['HIERARCH ESO TEL GEOLAT'],hdr['HIERARCH ESO TEL GEOLON'],hdr['HIERARCH ESO TEL GEOELEV'],hdr['RA'],hdr['DEC'])
            berv = np.append(berv,berv_i)
            hdr1d['HIERARCH ESO QC BERV']=berv_i#Append the berv here using the ESPRESSO berv keyword, so that it can be used in molecfit later.
            s1dhdr.append(hdr1d)

    #Check that all exposures have the same number of pixels, and clip orders if needed.
    min_npx = int(np.min(np.array(npx)))
    min_npx1d = int(np.min(np.array(npx1d)))
    if np.sum(np.abs(np.array(npx)-npx[0])) != 0:
        warnings.warn("in read_e2ds when reading UVES data: Not all e2ds files have the same number of pixels. This could have happened if the pipeline has extracted one or two extra pixels in some exposures but not others. The e2ds files will be clipped to the smallest width.",RuntimeWarning)
        for i in range(len(e2ds)):
            wave[i]=wave[i][:,0:min_npx]
            e2ds[i]=e2ds[i][:,0:min_npx]
            npx[i]=min_npx
    if np.sum(np.abs(np.array(npx1d)-npx1d[0])) != 0:
        warnings.warn("in read_e2ds when reading UVES data: Not all s1d files have the same number of pixels. This could have happened if the pipeline has extracted one or two extra pixels in some exposures but not others. The s1d files will be clipped to the smallest width.",RuntimeWarning)
        for i in range(len(s1d)):
            wave1d[i]=wave1d[i][0:min_npx1d]
            s1d[i]=s1d[i][0:min_npx1d]
            npx1d[i]=min_npx1d
    output = {'wave':wave,'e2ds':e2ds,'header':header,'wave1d':wave1d,'s1d':s1d,'s1dhdr':s1dhdr,'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,'npx1d':npx1d,'norders':norders,'berv':berv,'airmass':airmass,'s1dmjd':s1dmjd}
    return(output)



def read_espresso(inpath,filelist,read_s1d=True):
    #The following variables define lists in which all the necessary data will be stored.
    framename=[]
    header=[]
    s1dhdr=[]
    obstype=[]
    texp=np.array([])
    date=[]
    mjd=np.array([])
    s1dmjd=np.array([])
    npx=np.array([])
    norders=np.array([])
    e2ds=[]
    s1d=[]
    wave1d=[]
    airmass=np.array([])
    berv=np.array([])
    wave=[]
    catkeyword = 'EXTNAME'
    bervkeyword = 'HIERARCH ESO QC BERV'
    airmass_keyword1 = 'HIERARCH ESO TEL'
    airmass_keyword2 = ' AIRM '
    airmass_keyword3_start = 'START'
    airmass_keyword3_end = 'END'
    for i in range(len(filelist)):
        if filelist[i].endswith('S2D_BLAZE_A.fits'):
            hdul = fits.open(inpath/filelist[i])
            data = copy.deepcopy(hdul[1].data)
            hdr = hdul[0].header
            hdr2 = hdul[1].header
            wavedata=copy.deepcopy(hdul[5].data)
            hdul.close()
            del hdul

            if hdr2[catkeyword] == 'SCIDATA':
                # print('science keyword found')
                print(f'------{filelist[i]}', end="\r")
                framename.append(filelist[i])
                header.append(hdr)
                obstype.append('SCIENCE')
                texp=np.append(texp,hdr['EXPTIME'])
                date.append(hdr['DATE-OBS'])
                mjd=np.append(mjd,hdr['MJD-OBS'])
                npx=np.append(npx,hdr2['NAXIS1'])
                norders=np.append(norders,hdr2['NAXIS2'])
                e2ds.append(data)
                berv=np.append(berv,hdr[bervkeyword])#in km.s.
                telescope = hdr['TELESCOP'][-1]
                airmass = np.append(airmass,0.5*(hdr[airmass_keyword1+telescope+' AIRM START']+hdr[airmass_keyword1+telescope+' AIRM END']))
                wave.append(wavedata/10.0)#*(1.0-(hdr[bervkeyword]*u.km/u.s/const.c).decompose().value))
                #Ok.! So unlike HARPS, ESPRESSO wavelengths are actually BERV corrected in the S2Ds.
                #WHY!!!?. WELL SO BE IT. IN ORDER TO HAVE E2DSes THAT ARE ON THE SAME GRID, AS REQUIRED, WE UNDO THE BERV CORRECTION HERE.
                #WHEN COMPARING WAVE[0] WITH WAVE[1], YOU SHOULD SEE THAT THE DIFFERENCE IS NILL.
                #THATS WHY LATER WE CAN JUST USE WAVE[0] AS THE REPRESENTATIVE GRID FOR ALL.
                #BUT THAT IS SILLY. JUST SAVE THE WAVELENGTHS!

                if read_s1d:
                    s1d_path=inpath/Path(str(filelist[i]).replace('_S2D_BLAZE_A.fits','_S1D_A.fits'))
                    ut.check_path(s1d_path,exists=True)#Crash if the S1D doesn't exist.
                    hdul = fits.open(s1d_path)
                    data_table = copy.deepcopy(hdul[1].data)
                    hdr1d = hdul[0].header
                    hdul.close()
                    del hdul
                    s1d.append(data_table.field(2))

                    berv1d = hdr1d[bervkeyword]
                    if berv1d != hdr[bervkeyword]:
                        wrn_msg = ('WARNING in read_espresso(): BERV correction of S1D file is not'
                        f'equal to that of the S2D file. {berv1d} vs {hdr[bervkeyword]}')
                        ut.tprint(wrn_msg)
                    gamma = (1.0-(berv1d*u.km/u.s/const.c).decompose().value)
                    wave1d.append(data_table.field(1)*gamma)#This is in angstroms.
                    #We need to check to which UT ESPRESSO was connected, so that we can read
                    #the weather information (which is UT-specific) and parse them into the
                    #header using UT-agnostic keywords that are in the ESPRESSO.par file.
                    TELESCOP = hdr1d['TELESCOP'].split('U')[1]#This is the number of the UT, either 1, 2, 3 or 4.
                    if TELESCOP not in ['1','2','3','4']:
                        raise ValueError(f"in read_e2ds when reading ESPRESSO data. The UT telescope is not recognised. (TELESCOP={hdr['TELESCOP']})")
                    else:
                        hdr1d['TELALT']     = hdr1d[f'ESO TEL{TELESCOP} ALT']
                        hdr1d['RHUM']       = hdr1d[f'ESO TEL{TELESCOP} AMBI RHUM']
                        hdr1d['PRESSURE']   = hdr1d[f'ESO INS ADC{TELESCOP} SENS1']
                        hdr1d['AMBITEMP']   = hdr1d[f'ESO TEL{TELESCOP} AMBI TEMP']
                        hdr1d['M1TEMP']     = hdr1d[f'ESO TEL{TELESCOP} TH M1 TEMP']
                    s1dhdr.append(hdr1d)
                    s1dmjd=np.append(s1dmjd,hdr1d['MJD-OBS'])
    output = {'wave':wave,'e2ds':e2ds,'header':header,'wave1d':wave1d,'s1d':s1d,'s1dhdr':s1dhdr,'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,'norders':norders,'berv':berv,'airmass':airmass,'s1dmjd':s1dmjd}
    return(output)
