
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


def read_e2ds(inpath,outname,config,nowave=False,molecfit=False,mode='HARPS',measure_RV=True,star='solar',ignore_exp=[]):
    """This is the workhorse for reading in a time-series of archival 2D echelle
    spectra and formatting these into the order-wise FITS format that Tayph uses.

    The user should point this script to a folder (located at inpath) that contains
    their pipeline-reduced echelle spectra. The script expects a certain data
    format, depending on the instrument in question. It is designed to accept
    pipeline products of the HARPS, HARPS-N, ESPRESSO and UVES instruments. In the
    case of HARPS, HARPS-N and ESPRESSO these may be downloaded from the archive.
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

    A crucial functionality of this script is that it also acts as a wrapper
    for the Molecfit telluric correction software. If installed properly, the
    user can call this script with the molecfit keyword to let Molecfit loop
    over the entire timeseries. To enable this functionality, the script
    reads the full-width, 1D spectra that are output by the instrument pipelines
    as well. Molecfit is applied to this time-series of 1D spectra, creating a
    time-series of models of the telluric absorption spectrum that is saved along
    with the 2D fits files. Tayph later interpolates these models onto the 2D
    spectra. Molecfit is called once in GUI-mode, allowing the user to select the
    relevant fitting regions and parameters, after which it is repeated
    automatically for the entire time series.

    Without Molecfit, this script finishes in a matter of seconds. However with
    molecfit enabled, it can take many hours (so if I wish to telluric-correct
    my data, I run this script overnight).

    The processing of HARPS, HARPS-N and ESPRESSO data is executed in an almost
    identical manner, because the pipeline-reduced products are almost identical.
    To run on either of these instruments, the user simply downloads all pipeline
    products of a given time-series, and extracts these in the same folder (meaning
    ccfs, e2ds/s2d, s1d, blaze, wave files, etc.) This happens to be the standard
    format when downloading pipeline-reduced data from the archive.

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
    dichroics and 'arms', leading to spectral coverage on the blue, redu and/or redl
    chips. The user should take care that their time series contains only one blue or
    red types at any time. If they are mixed, this script will throw an exception.
    The blue and red arms are regarded as two different spectrographs (they are), but
    the two red chils (redu and redl) are combined when reading in the data.

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


    # molecfit = False
    #First check the input:
    inpath=ut.check_path(inpath,exists=True)
    typetest(outname,str,'outname in read_e2ds()')
    typetest(nowave,bool,'nowave switch in read_e2ds()')
    typetest(molecfit,bool,'molecfit switch in read_e2ds()')
    typetest(ignore_exp,list,'ignore_exp in read_e2ds()')
    typetest(mode,str,'mode in read_e2ds()')
    if molecfit:
        config = ut.check_path(config,exists=True)
    if mode not in ['HARPS','HARPSN','HARPS-N','ESPRESSO','UVES-red','UVES-blue']:
        raise ValueError("in read_e2ds: mode needs to be set to HARPS, HARPSN, UVES-red, UVES-blue or ESPRESSO.")
    if measure_RV:
        #Define the paths to the stellar and telluric templates if RV's need to be measured.
        starwavepath=Path(pkg_resources.resource_filename('tayph',str(Path('data')/Path('WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'))))
        telpath = Path(pkg_resources.resource_filename('tayph',str(Path('data')/Path('skycalc_vac.fits'))))
        if star.lower()=='solar' or star.lower() =='medium':
            starpath=Path(pkg_resources.resource_filename('tayph',str(Path('data')/Path('lte06000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'))))
        elif star.lower() =='hot' or star.lower() =='warm':
            starpath=Path(pkg_resources.resource_filename('tayph',str(Path('data')/Path('lte09000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'))))
        elif star.lower() == 'cool' or star.lower() == 'cold':
            starpath=Path(pkg_resources.resource_filename('tayph',str(Path('data')/Path('lte04000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'))))
        else:
            warnings.warn(f"in read_e2ds: The star keyword was set to f{star} but only solar, hot or cold are allowed. Assuming a solar template.",RuntimeWarning)
            starpath=Path(pkg_resources.resource_filename('tayph',str(Path('data')/Path('lte06000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'))))
        ut.check_path(starpath,exists=True)
        ut.check_path(starwavepath,exists=True)
        ut.check_path(telpath,exists=True)

        #Load the spectra from FITS files provided along with Tayph
        fxm=fits.getdata(starpath)
        wlm=fits.getdata(starwavepath)/10.0#Angstrom to nm.
        ttt=fits.getdata(telpath)
        fxt=ttt[1]
        wlt=ttt[0]


        #Continuum-subtract the telluric model
        wlte,fxte=ops.envelope(wlt,fxt,2.0,selfrac=0.05,threshold=0.8)
        e_t = interpolate.interp1d(wlte,fxte,fill_value='extrapolate')(wlt)
        fxtn = fxt/e_t
        fxtn[fxtn>1]=1.0

        wlt=np.concatenate((np.array([100,200,300,400]),wlt))
        fxtn=np.concatenate((np.array([1.0,1.0,1.0,.10]),fxtn))
        fxtn[wlt<500.1]=1.0

        #Continuum-subtract the PHOENIX model
        wlme,fxme=ops.envelope(wlm,fxm,5.0,selfrac=0.005)
        e_s = interpolate.interp1d(wlme,fxme,fill_value='extrapolate')(wlm)
        fxmn=fxm/e_s
        fxmn[fxmn>1.0]=1.0
        fxmn[fxmn<0.0]=0.0

        fxmn=fxmn[wlm>300.0]
        wlm=wlm[wlm>300.0]
        #The stellar and telluric templates are now ready for application in cross-correlation at the very end of this script.



    #Start reading data files.
    filelist=os.listdir(inpath)#If mode == UVES, these are folders. Else, they are fits files.
    N=len(filelist)

    if len(filelist) == 0:
        raise FileNotFoundError(f" in read_e2ds: input folder {str(inpath)} is empty.")


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
    npx1d=np.array([])#Only used for UVES
    norders=np.array([])
    e2ds=[]
    s1d=[]
    wave1d=[]
    airmass=np.array([])
    berv=np.array([])
    wave=[]
    blaze=[]
    wavefile_used = []
    outpath = Path('data/'+outname)
    if os.path.exists(outpath) != True:
        os.makedirs(outpath)


    e2ds_count = 0
    sci_count = 0
    wave_count = 0
    blaze_count = 0
    s1d_count = 0



    if mode=='HARPS-N': mode='HARPSN'#Guard against various ways of referring to HARPS-N.

    #MODE SWITCHING HERE:
    if mode in ['HARPS','UVES-red','UVES-blue']:
        catkeyword = 'HIERARCH ESO DPR CATG'
        bervkeyword = 'HIERARCH ESO DRS BERV'
        thfilekeyword = 'HIERARCH ESO DRS CAL TH FILE'
        Zstartkeyword = 'HIERARCH ESO TEL AIRM START'
        Zendkeyword = 'HIERARCH ESO TEL AIRM END'
    if mode == 'HARPSN':
        catkeyword = 'OBS-TYPE'
        bervkeyword = 'HIERARCH TNG DRS BERV'
        thfilekeyword = 'HIERARCH TNG DRS CAL TH FILE'
        Zstartkeyword = 'AIRMASS'
        Zendkeyword = 'AIRMASS'#These are the same because HARPSN doesnt have start and end keywords.
        #Down there, the airmass is averaged, so there is no problem in taking the average of the same number.
    if mode == 'ESPRESSO':
        catkeyword = 'EXTNAME'
        bervkeyword = 'HIERARCH ESO QC BERV'
        airmass_keyword1 = 'HIERARCH ESO TEL'
        airmass_keyword2 = ' AIRM '
        airmass_keyword3_start = 'START'
        airmass_keyword3_end = 'END'
        #However there is a problem in this way of dealing with airmass and that is that
        #the derivative of the airmass may not be constant for higher airmass, meaning that
        #the average should be weighted in some way. This can probably be achieved by using
        #information about the start end and time duration between the two, but its a bit
        #more difficult. All the more reason to keep exposures short...



    #Here is the actual parsing of the list of files that were read above. The
    #behaviour is different depending on whether this is HARPS, UVES or ESPRESSO
    #data, so it switches with a big if-statement in which there is a forloop
    #over the filelist in each case. The result is lists or np.arrays containing
    #the 2D spectra, the 1D spectra, their 2D and 1D wavelength solutions, the
    #headers, the MJDs, the BERVs and the airmasses, as well as (optionally) CCFs
    #and blaze files, though these are not really used.


    print(f'Read_e2ds is attempting to read a {mode} datafolder at {str(inpath)}.')
    print('These are the files encountered:')
    if mode == 'UVES-red' or mode == 'UVES-blue':#IF we are UVES-like
        for i in range(N):
            print(filelist[i])
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
                    wavedata=ut.read_wave_from_e2ds_header(hdr,mode='UVES')
                    data_combined.append(data)
                    wave_combined.append(wavedata)
                    norders_tmp+=np.shape(data)[0]

                    # pdb.set_trace()


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
                sci_count += 1
                s1d_count += 1
                e2ds_count += 1


    elif mode == 'ESPRESSO':
        for i in range(N):
            if filelist[i].endswith('S2D_BLAZE_A.fits'):
                e2ds_count += 1
                print(filelist[i])
                hdul = fits.open(inpath/filelist[i])
                data = copy.deepcopy(hdul[1].data)
                hdr = hdul[0].header
                hdr2 = hdul[1].header
                wavedata=copy.deepcopy(hdul[5].data)
                hdul.close()
                del hdul
                #We also need to read the blaze that comes with each exposure, because
                #The science frame is blaze-corrected.
                # blaze_path=inpath/Path(str(filelist[i]).replace('_A.fits','_BLAZE_A.fits'))
                # ut.check_path(blaze_path,exists=True)#Crash if the blaze doesn't exist.
                # blaze = fits.open(blaze_path)
                # pdb.set_trace()
                # data/=blaze#We ALWAYS reintroduce the blaze to the data.
                # pdb.set_trace()

                if hdr2[catkeyword] == 'SCIDATA':
                    # print('science keyword found')
                    framename.append(filelist[i])
                    header.append(hdr)
                    obstype.append('SCIENCE')
                    texp=np.append(texp,hdr['EXPTIME'])
                    date.append(hdr['DATE-OBS'])
                    mjd=np.append(mjd,hdr['MJD-OBS'])
                    npx=np.append(npx,hdr2['NAXIS1'])
                    norders=np.append(norders,hdr2['NAXIS2'])
                    e2ds.append(data)
                    sci_count += 1
                    berv=np.append(berv,hdr[bervkeyword]*1000.0)
                    telescope = hdr['TELESCOP'][-1]
                    airmass = np.append(airmass,0.5*(hdr[airmass_keyword1+telescope+' AIRM START']+hdr[airmass_keyword1+telescope+' AIRM END']))
                    wave.append(wavedata)#*(1.0-(hdr[bervkeyword]*u.km/u.s/const.c).decompose().value))
                    #Ok.! So unlike HARPS, ESPRESSO wavelengths are actually BERV corrected in the S2Ds.
                    #WHY!!!?. WELL SO BE IT. IN ORDER TO HAVE E2DSes THAT ARE ON THE SAME GRID, AS REQUIRED, WE UNDO THE BERV CORRECTION HERE.
                    #WHEN COMPARING WAVE[0] WITH WAVE[1], YOU SHOULD SEE THAT THE DIFFERENCE IS NILL.
                    #THATS WHY LATER WE CAN JUST USE WAVE[0] AS THE REPRESENTATIVE GRID FOR ALL.
                    #BUT THAT IS SILLY. JUST SAVE THE WAVELENGTHS!

            # if filelist[i].endswith('CCF_A.fits'):
            #     #ccf,hdr=fits.getdata(inpath+filelist[i],header=True)
            #     hdul = fits.open(inpath/filelist[i])
            #     ccf = copy.deepcopy(hdul[1].data)
            #     hdr = hdul[0].header
            #     hdr2 = hdul[1].header
            #     hdul.close()
            #     del hdul[1].data
            #
            #     if hdr2[catkeyword] == 'SCIDATA':
            #         print('CCF ADDED')
            #         #ccftotal+=ccf
            #         ccfs.append(ccf)
            #         ccfmjd=np.append(ccfmjd,hdr['MJD-OBS'])
            #         nrv=np.append(nrv,hdr2['NAXIS1'])
            #         ccf_count += 1

            if filelist[i].endswith('S1D_A.fits'):
                hdul = fits.open(inpath/filelist[i])
                data_table = copy.deepcopy(hdul[1].data)
                hdr = hdul[0].header
                hdr2 = hdul[1].header
                hdul.close()
                # pdb.set_trace()
                del hdul[1].data
                if hdr['HIERARCH ESO PRO SCIENCE'] == True:
                    s1d.append(data_table.field(2))
                    wave1d.append(data_table.field(1))


                    #We need to check to which UT ESPRESSO was connected, so that we can read
                    #the weather information (which is UT-specific) and parse them into the
                    #header using UT-agnostic keywords that are in the ESPRESSO.par file.
                    TELESCOP = hdr['TELESCOP'].split('U')[1]#This is the number of the UT, either 1, 2, 3 or 4.
                    if TELESCOP not in ['1','2','3','4']:
                        raise ValueError(f"in read_e2ds when reading ESPRESSO data. The UT telescope is not recognised. (TELESCOP={hdr['TELESCOP']})")
                    else:
                        hdr['TELALT'] = hdr[f'ESO TEL{TELESCOP} ALT']
                        hdr['RHUM'] = hdr[f'ESO TEL{TELESCOP} AMBI RHUM']
                        hdr['PRESSURE'] = hdr[f'ESO INS ADC{TELESCOP} SENS1']
                        hdr['AMBITEMP'] = hdr[f'ESO TEL{TELESCOP} AMBI TEMP']
                        hdr['M1TEMP'] = hdr[f'ESO TEL{TELESCOP} TH M1 TEMP']

                    s1dhdr.append(hdr)

                    s1dmjd=np.append(s1dmjd,hdr['MJD-OBS'])
                    s1d_count += 1






    else:#IF we are HARPS-like:
        for i in range(N):
            if filelist[i].endswith('e2ds_A.fits'):
                e2ds_count += 1
                print(filelist[i])


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
                    sci_count += 1
                    berv=np.append(berv,hdr[bervkeyword])
                    airmass=np.append(airmass,0.5*(hdr[Zstartkeyword]+hdr[Zendkeyword]))#This is an approximation where we take the mean airmass.
                    if nowave == True:
                    #Record which wavefile was used by the pipeline to
                        #create the wavelength solution.
                        wavefile_used.append(hdr[thfilekeyword])
                        wavedata=ut.read_wave_from_e2ds_header(hdr,mode=mode)
                        wave.append(wavedata)
            # else:
                # berv=np.append(berv,np.nan)
                # airmass=np.append(airmass,np.nan)
            if filelist[i].endswith('wave_A.fits'):
                print(filelist[i]+' (wave)')
                if nowave == True:
                    warnings.warn(" in read_e2ds: nowave was set to True but a wave_A file was detected. This wave file is now ignored in favor of the header.",RuntimeWarning)
                else:
                    wavedata=fits.getdata(inpath/filelist[i])
                    wave.append(wavedata)
                    wave_count += 1
            if filelist[i].endswith('blaze_A.fits'):
                    print(filelist[i]+' (blaze)')
                    blazedata=fits.getdata(inpath/filelist[i])
                    blaze.append(blazedata)
                    blaze_count += 1
            if filelist[i].endswith('s1d_A.fits'):
                hdul = fits.open(inpath/filelist[i])
                data_1d = copy.deepcopy(hdul[0].data)
                hdr = hdul[0].header
                hdul.close()
                del hdul[0].data
                if hdr[catkeyword] == 'SCIENCE':
                    s1d.append(data_1d)
                    s1dhdr.append(hdr)
                    s1dmjd=np.append(s1dmjd,hdr['MJD-OBS'])
                    s1d_count += 1
    #Now we catch some errors:
    #-The above should have read a certain number of e2ds files.
    #-A certain number of these should be SCIENCE frames.
    #-There should be at least one WAVE file.
    #-All exposures should have the same number of spectral orders.
    #-All orders should have the same number of pixels (this is true for HARPS).
    #-The wave frame should have the same dimensions as the order frames.
    #-If nowave is set, test that all frames used the same wave_A calibrator.
    #-The blaze file needs to have the same shape as the e2ds files.
    #-The number of s1d files should be the same as the number of e2ds files.


    if e2ds_count == 0:
        raise FileNotFoundError(f"in read_e2ds: The input folder {str(inpath)} does not contain files recognised as e2ds-format FITS files.")
    if sci_count == 0:
        print('')
        print('')
        print('')
        print("These are the files and their types:")
        for i in range(len(obstype)):
            print('   '+framename[i]+'  %s' % obstype[i])
        raise ValueError("in read_e2ds: The input folder (%2) contains e2ds files, but none of them are classified as SCIENCE frames with the HIERARCH ESO DPR CATG, OBS-TYPE or HIERARCH ESO PRO SCIENCE keyword. The list of frames is printed above.")
    if np.max(np.abs(norders-norders[0])) == 0:
        norders=int(norders[0])
    else:
        print('')
        print('')
        print('')
        print("These are the files and their numbers of orders:")
        for i in range(len(obstype)):
            print('   '+framename[i]+'  %s' % norders[i])
        raise ValueError("in read_e2ds: Not all e2ds files have the same number of orders. The list of frames is printed above.")


    if mode in ['UVES-red','UVES-blue']:
        min_npx = int(np.min(np.array(npx)))
        min_npx1d = int(np.min(np.array(npx1d)))
        if np.sum(np.abs(np.array(npx)-npx[0])) != 0:
            warnings.warn("in read_e2ds when reading UVES data: Not all e2ds files have the same number of pixels. This could have happened if the pipeline has extracted one or two extra pixels in some exposures but not others. The e2ds files will be clipped to the smallest width.",RuntimeWarning)
            for i in range(len(e2ds)):
                wave[i]=wave[i][:,0:min_npx]
                e2ds[i]=e2ds[i][:,0:min_npx]
        if np.sum(np.abs(np.array(npx1d)-npx1d[0])) != 0:
            warnings.warn("in read_e2ds when reading UVES data: Not all s1d files have the same number of pixels. This could have happened if the pipeline has extracted one or two extra pixels in some exposures but not others. The e2ds files will be clipped to the smallest width.",RuntimeWarning)
            for i in range(len(s1d)):
                wave1d[i]=wave1d[i][0:min_npx1d]
                s1d[i]=s1d[i][0:min_npx1d]
        npx=min_npx
        npx1d=min_npx1d
    else:
        if np.max(np.abs(npx-npx[0])) == 0:
            npx=int(npx[0])
        else:
            print('')
            print('')
            print('')
            print("These are the files and their numbers of pixels:")
            for i in range(len(obstype)):
                print('   '+framename[i]+'  %s' % npx[i])
            raise ValueError("in read_e2ds: Not all e2ds files have the same number of pixels. The list of frames is printed above.")



    if wave_count >= 1:#Is only made nonzero in case of harps so other modes are not tested for.
        wave=wave[0]#SELECT ONLY THE FIRST WAVE FRAME. The rest is ignored.
    else:
        if nowave == False and mode not in ['UVES-red','UVES-blue','ESPRESSO']:
            print('')
            print('')
            print('')
            print("ERROR in read_e2ds: No wave_A.fits file was detected.")
            print("These are the files in the folder:")
            for i in range(N):
                print(filelist[i])
            print('')
            print("This may have happened if you downloaded the HARPS data from the")
            print("ADP query form, in which case wave_A files are not included (as far as I")
            print("have seen). Set the /nowave keyword in your call to read_e2ds")
            print("if you indeed do not expect a wave_A file to be present with your HARPS data.")
            raise FileNotFoundError("No wave_A.fits file was detected. More details are printed above.")




    if nowave == True and mode not in ['UVES-red','UVES-blue','ESPRESSO']:#This here is peculiar to HARPS/HARPSN.
        if all(x == wavefile_used[0] for x in wavefile_used):
            print("Nowave is set, and simple extraction of the wavelength calibration information")
            print("works because all files in the dataset were originally reduced using the same wave_A file.")
            wave=wave[0]
        else:
            print('')
            print('')
            print('')
            print("These are the filenames and their wave_A file used:")
            for i in range(N-1):
                print('   '+framename[i]+'  %s' % wavefile_used[0])
            warnings.warn("in read_e2ds: Nowave is set, but not all files in the dataset used the same wave_A file when the pipeline was run. This script will continue using only the first wavelength solution. Theoretically, this may affect the quality of the data if the solution is wrong (in which case interpolation would be needed), but due to the stability of HARPS/HARPS-N this is probably not an issue worth interpolating for.",RuntimeWarning)
            wave=wave[0]



    #Here we do a lot of messing with ESPRESSO's exquisite wavelengths. I think that I want to avoid this!
    if mode == 'ESPRESSO':
        dimtest(wave1d,np.shape(s1d),'wave and s1d in read_e2ds()')
        dimtest(wave,np.shape(e2ds),'wave and e2ds in read_e2ds()')
        diff = wave-wave[0]
        if np.abs(np.nanmax(diff))>(np.nanmin(wave[0])/1e6):
            warnings.warn(" in read_e2ds: The wavelength solution over the time series is not constant. I continue by interpolating all the data onto the wavelength solution of the first frame. This will break if the wavelength solutions of your time-series are vastly different, in which case the output will be garbage.",RuntimeWarning)
            for ii in range(len(e2ds)):
                if ii>0:
                    for jj in range(len(e2ds[ii])):
                        e2ds[ii][jj] = interp.interp1d(wave[ii][jj],e2ds[ii][jj],fill_value='extrapolate')(wave[0][jj])
        wave=wave[0]
        diff1d = wave1d-wave1d[0]
        if np.abs(np.nanmax(diff1d))>(np.nanmin(wave1d)/1e6):#if this is true, all the wavelength solutions are the same. Great!
            warnings.warn(" in read_e2ds: The wavelength solution over the time series of the 1D spectra is not constant. I continue by interpolating all the 1D spectra onto the wavelength solution of the first frame. This will break if the wavelength solutions of your time-series are vastly different, in which case the output will be garbage.",RuntimeWarning)
            for ii in range(len(s1d)):
                if ii>0:
                        s1d[ii] = interp.interp1d(wave1d[ii],s1d[ii],fill_value='extrapolate')(wave1d[0])
        wave1d=wave1d[0]



    #We are going to test whether the wavelength solution is the same for the
    #entire time series in the case of UVES. In normal use cases this should be mostly true, but it's possible
    #that the pipeline has extracted a different number of pixels in each, in which case we may need to
    #interpolate.

    if mode in ['UVES-red','UVES-blue']:

        dimtest(wave1d,np.shape(s1d),'wave and s1d in read_e2ds()')
        typetest(wave,list,'wave in read_e2ds()')
        dimtest(wave[i],np.shape(e2ds[i]),f'wave[{i}] and e2ds[{i}] in read_e2ds()')
        diff = wave-wave[0]
        if np.abs(np.nanmax(diff))==0:#if this is true, all the wavelength solutions are the same. Great!
            wave=wave[0]
        else:
            warnings.warn(" in read_e2ds: The wavelength solution over the time series is not constant. I continue by interpolating all the data onto the wavelength solution of the first frame. This will break if the wavelength solutions of your time-series are vastly different, in which case the output will be garbage.",RuntimeWarning)
            for ii in range(len(e2ds)):
                if ii>0:
                    for jj in range(len(e2ds[ii])):
                        e2ds[ii][jj] = interp.interp1d(wave[ii][jj],e2ds[ii][jj],fill_value='extrapolate')(wave[0][jj])
            wave=wave[0]
        diff1d = np.array(np.abs(np.array(wave1d)-wave1d[0]))#Crazy error here forces me to write np.array everywhere??
        if np.nansum(np.abs(diff1d))==0:#if this is true, all the wavelength solutions are the same. Great!
            wave1d=wave1d[0]
        else:
            warnings.warn(" in read_e2ds: The wavelength solution over the time series of the 1D spectra is not constant. I continue by interpolating all the 1D spectra onto the wavelength solution of the first frame. This will break if the wavelength solutions of your time-series are vastly different, in which case the output will be garbage.",RuntimeWarning)
            for ii in range(len(s1d)):
                if ii>0:
                        s1d[ii] = interp.interp1d(wave1d[ii],s1d[ii],fill_value='extrapolate')(wave1d[0])
            wave1d=wave1d[0]






    if np.shape(wave) != np.shape(e2ds[0]):#If UVES/ESPRESSO, this was already checked implicitly.
        raise ValueError(f"in read_e2ds: A wave file was detected but its shape ({np.shape(wave)[0]},{np.shape(wave)[1]}) does not match that of the orders ({np.shape(e2ds[0])[0]},{np.shape(e2ds[0])[1]})")
    # if np.shape(blaze) != np.shape(e2ds[0]) and blaze_count > 0:
    #     raise ValueError(f"in read_e2ds: A blaze file was detected but its shape ({np.shape(blaze)[0]},{np.shape(wave)[1]}) does not match that of the orders ({np.shape(e2ds[0])[0]},{np.shape(e2ds[0])[1]})")
    if len(s1dhdr) != len(e2ds) and molecfit == True:#Only a problem if we actually run Molecfit.
        raise ValueError(f'in read_e2ds: The number of s1d SCIENCE files and e2ds SCIENCE files is not the same. ({len(s1dhdr)} vs {len(e2ds)})')


    #Ok, so now we should have ended up with a number of lists that contain all
    #the relevant science data and associated information.
    #We determine how to sort the resulting lists in time:
    sorting = np.argsort(mjd)
    s1dsorting = np.argsort(s1dmjd)

    if len(ignore_exp) > 0:
        sorting = [x for i,x in enumerate(sorting) if i not in ignore_exp]
        s1dsorting = [x for i,x in enumerate(s1dsorting) if i not in ignore_exp]


    if mode == 'HARPSN': #In the case of HARPS-N we need to convert the units of the elevation and provide a UTC keyword.
        for i in range(len(header)):
            s1dhdr[i]['TELALT'] = np.degrees(float(s1dhdr[i]['EL']))
            s1dhdr[i]['UTC'] = (float(s1dhdr[i]['MJD-OBS'])%1.0)*86400.0

    s1dhdr_sorted=[]
    s1d_sorted=[]
    for i in range(len(s1dsorting)):
        s1dhdr_sorted.append(s1dhdr[s1dsorting[i]])
        s1d_sorted.append(s1d[s1dsorting[i]])

    #Sort the s1d files for application of molecfit.
    if molecfit == True:
        if len(sorting) != len(s1dsorting):
            raise ValueError("in read_e2ds: Sorted science frames and sorted s1d frames are not of the same length. Telluric correction can't proceed. Make sure that the number of files of each type is correct.")

        print("")
        print("")
        print("")
        print('Molecfit will be executed onto the files with dates in this order:')
        for x in s1dhdr_sorted:
            print(x['DATE-OBS'])
        print("")
        print("If these are not chronologically ordered that indicates a problem with the way dates are formatted in the header and you are advised to abort this program.")
        print("")
        print("")
        if mode in ['ESPRESSO','UVES-red','UVES-blue']:
            list_of_wls,list_of_trans = mol.do_molecfit(s1dhdr_sorted,s1d_sorted,config,load_previous=False,mode=mode,wave=wave1d)
        else:
            list_of_wls,list_of_trans = mol.do_molecfit(s1dhdr_sorted,s1d_sorted,config,load_previous=False,mode=mode)

        if len(list_of_trans) != len(sorting):
            raise ValueError("in read_e2ds(): Molecfit did not produce the same number of spectra as there are in the e2ds spectra.")
        mol.write_telluric_transmission_to_file(list_of_wls,list_of_trans,outpath/'telluric_transmission_spectra.pkl')

    if measure_RV:
        list_of_orders=[]
        list_of_waves=[]
    #Now we loop over all exposures and collect the i-th order from each exposure,
    #put these into a new matrix and save them to FITS images:
    f=open(outpath/'obs_times','w',newline='\n')
    headerline = 'MJD'+'\t'+'DATE'+'\t'+'EXPTIME'+'\t'+'MEAN AIRMASS'+'\t'+'BERV (km/s)'+'\t'+'FILE NAME'
    for i in range(norders):
        order = np.zeros((len(sorting),npx))
        wave_axis = wave[i,:]/10.0#Convert to nm.
        # print('CONSTRUCTING ORDER %s' % i)
        print(f"CONSTRUCTING ORDER {i}", end="\r")
        c = 0#To count the number of science frames that have passed. The counter
        # c is not equal to j because the list of files contains not only SCIENCE
        # frames.


        for j in range(len(sorting)):#Loop over exposures
            if i ==0:
                # print('These are the file types (should all be SCIENCE) and observing date-times recognised:')
                print(' - '+obstype[sorting[j]]+'  '+date[sorting[j]])
            if obstype[sorting[j]] == 'SCIENCE':#This check may be redundant.
                exposure = e2ds[sorting[j]]
                order[c,:] = exposure[i,:]
                #T_i = interp.interp1d(list_of_wls[j],list_of_trans[j])#This should be time-sorted, just as the e2ds files.
                #Do a manual check here that the MJDs are identical.
                #Also, determiine what to do with airtovac.
                #tel_order[c,:] = T_i[wave_axis]
                #Now I also need to write it to file.
                if i ==0:#Only do it the first time, not for every order.
                    line = str(mjd[sorting[j]])+'\t'+date[sorting[j]]+'\t'+str(texp[sorting[j]])+'\t'+str(np.round(airmass[sorting[j]],3))+'\t'+str(np.round(berv[sorting[j]],5))+'\t'+framename[sorting[j]]+'\n'
                    f.write(line)
                c+=1

        if mode in ['UVES-red','UVES-blue','ESPRESSO']:#UVES and ESPRESSO pipelines pad with zeroes. We remove these.
            # order[np.abs(order)<1e-10*np.nanmedian(order)]=0.0#First set very small values to zero.
            npx_order = np.shape(order)[1]
            sum=np.nansum(np.abs(order),axis=0)#Anything that is zero in this sum (i.e. the edges) will be clipped.
            leftsize=npx_order-len(np.trim_zeros(sum,'f'))#Size of zero-padding on the left
            rightsize=npx_order-len(np.trim_zeros(sum,'b'))#Size of zero-padding on the right
            wave_axis=wave_axis[leftsize:npx_order-rightsize-1]
            order = order[:,leftsize:npx_order-rightsize-1]

        fits.writeto(outpath/('order_'+str(i)+'.fits'),order,overwrite=True)
        fits.writeto(outpath/('wave_'+str(i)+'.fits'),wave_axis,overwrite=True)
        if measure_RV:
            list_of_orders.append(order)
            list_of_waves.append(wave_axis)
    f.close()
    print(f'Time-table written to {outpath/"obs_times"}.')






















    #The rest is for if diagnostic radial velocity measurements are requested.
    #We take and plot all the 1D and 2D spectra, do a rudimentary cleaning (de-colour, outlier correction)
    #and perform cross-correlations with solar and telluric templates.
    if measure_RV:
        #From this moment on, I will start reusing some variable that I named above, i.e. overwriting them.
        drv=1.0
        RVrange=150.0
        dw=50.0
        print(f'Performing diagnostic correlation with a telluric template and a {star} PHOENIX model.')
        print(f'   Cleaning 1D spectra for cross-correlation.')


        if mode in ['HARPS','HARPSN']:
            wave_1d = (s1dhdr[0]['CDELT1']*fun.findgen(len(s1d[0]))+s1dhdr[0]['CRVAL1'])#Master 1D S1D array.
            wave_1d_berv_corrected = wave_1d*(1.0-(s1dhdr[0][bervkeyword]*u.km/u.s/const.c).decompose().value)
            s1d_block=np.zeros((len(s1d),len(wave_1d)))
            s1d_block_berv_corrected=np.zeros((len(s1d),len(wave_1d)))
            s1d_mjd=[]
            s1d_berv=[]
            for i in range(0,len(s1d)):
                wave = (s1dhdr[i]['CDELT1']*fun.findgen(len(s1d[i]))+s1dhdr[i]['CRVAL1'])
                s1d_block[i]=interp.interp1d(wave,s1d[i],bounds_error=False,fill_value='extrapolate')(wave_1d)
                s1d_block_berv_corrected[i]=interp.interp1d(wave*(1.0-(s1dhdr[i][bervkeyword]*u.km/u.s/const.c).decompose().value),s1d[i],bounds_error=False,fill_value='extrapolate')(wave_1d_berv_corrected)

        if mode == 'ESPRESSO':
            wave_1d=wave1d/10.0#Convert to nm.
            wave_1d_berv_corrected=wave_1d*(1.0-(s1dhdr[0][bervkeyword]*u.km/u.s/const.c).decompose().value)
            s1d_block=np.zeros((len(s1d),len(wave_1d)))
            s1d_block_berv_corrected=np.zeros((len(s1d),len(wave_1d)))
            s1d_mjd=[]
            s1d_berv=[]
            for i in range(0,len(s1d)):
                s1d_mjd.append(s1dhdr[i]['MJD-OBS'])
                s1d_berv.append(s1dhdr[i][bervkeyword])
                s1d_block[i]=s1d[i]
                s1d_block_berv_corrected[i]=interp.interp1d(wave_1d*(1.0-(s1dhdr[i][bervkeyword]*u.km/u.s/const.c).decompose().value),s1d[i],bounds_error=False,fill_value='extrapolate')(wave_1d_berv_corrected)
            # wave_1d = wave_1d_berv_corrected
            # s1d_block = s1d_block_berv_corrected

        wave_1d,s1d_block,r1,r2=ops.clean_block(wave_1d,s1d_block,deg=4,verbose=True)#Slow. Needs parallelising.

        if mode in ['UVES-red','UVES-blue']:
            pdb.set_trace()

        # ut.writefits('test.fits',s1d_block)
        # plt.plot(s1d_mjd,s1d_berv,'.')
        # plt.show()
        # for i in range(len(s1d)): plt.plot(wave_1d,s1d_block[i],color='blue',alpha=0.2)
        # for i in range(len(s1d)): plt.plot(wave_1d_berv_corrected,s1d_block_berv_corrected[i],color='red',alpha=0.2)
        # plt.show()

        print(f'   Cleaning 2D orders for cross-correlation.')
        list_of_orders_trimmed=[]
        list_of_waves_trimmed=[]
        for i,o in enumerate(list_of_orders):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w_trimmed,o_trimmed,r1,r2=ops.clean_block(list_of_waves[i],list_of_orders[i],w=100,deg=4)#Slow. Needs parallelising.
                list_of_waves_trimmed.append(w_trimmed)
                list_of_orders_trimmed.append(o_trimmed)
                ut.statusbar(i,len(list_of_orders))



        print(f"   Interpolating over NaN's")
        list_of_orders_trimmed = masking.interpolate_over_NaNs(list_of_orders_trimmed)#Slow, also needs parallelising.
        s1d_block=masking.interpolate_over_NaNs([s1d_block])[0]
        print(f'   Performing cross-correlation and plotting output.')
        rv,ccf,Tsums=xcor(list_of_waves_trimmed,list_of_orders_trimmed,np.flipud(np.flipud(ops.vactoair(wlm))),fxmn-1.0,drv,RVrange)
        rv1d,ccf1d,Tsums1d=xcor([wave_1d],[s1d_block],np.flipud(np.flipud(ops.vactoair(wlm))),fxmn-1.0,drv,RVrange)



        #The rest is for plotting.
        plt.figure(figsize=(8,8))
        minwl=np.inf#For determining in the for-loop what the minimum and maximum wl range of the data is.
        maxwl=0.0
        for i in range(len(list_of_orders)):
            s_avg = np.nanmean(list_of_orders[i],axis=0)
            s_avg[s_avg<=0]=np.nan
            # print(minwl,np.min(list_of_waves[i]),np.min(s_avg/np.nanmean(s_avg)))
            minwl=np.min([np.min(list_of_waves[i]),minwl])
            maxwl=np.max([np.max(list_of_waves[i]),maxwl])

            if i == 0:
                plt.plot(list_of_waves[i],s_avg/np.nanmean(s_avg),color='red',linewidth=0.9,alpha=0.5,label='Spectra orders to be cross-correlated')
            else:
                plt.plot(list_of_waves[i],s_avg/np.nanmean(s_avg),color='red',linewidth=0.7,alpha=0.5)
        s1d_avg=np.nanmean(s1d_block,axis=0)
        plt.plot(wave_1d,s1d_avg/np.nanmean(s1d_avg)+1,color='orange',label='1D spectrum (offset +1)',linewidth=0.9,alpha=0.5)
            # plt.plot(list_of_waves[i],list_of_orders[i][0],color='red',linewidth=0.7)
            # plt.plot(list_of_waves[i],list_of_orders[i][1],color='blue',linewidth=0.7)
            # plt.plot(list_of_waves[i],list_of_orders[i][3],color='green',linewidth=0.7)
        plt.title(f'Time-averaged spectral orders and {star} PHOENIX model')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Flux (normalised to order average)')
        plt.plot(ops.vactoair(wlm),fxmn,color='green',linewidth=0.7,label='PHOENIX template (air, used in ccf)',alpha=0.5)
        plt.plot(wlm,fxmn,color='olive',linewidth=0.7,label='PHOENIX template (vaccuum)',alpha=0.3)
        plt.xlim(minwl-5,maxwl+5)#Always nm.
        plt.legend(loc='upper right',fontsize=8)
        plt.show()

        centroids=[]
        centroids1d=[]
        for n,i in enumerate(ccf):
            if n == 0:
                plt.plot(rv,i/np.nanmean(i),linewidth=0.7,alpha=0.2,color='black',label='2D orders')
            else:
                plt.plot(rv,i/np.nanmean(i),linewidth=0.7,alpha=0.2,color='black')
            centroids.append(rv[np.argmin(i)])
        for n,i in enumerate(ccf1d):
            if n == 0:
                plt.plot(rv1d,i/np.nanmean(i),linewidth=0.7,alpha=0.1,color='red',label='1D spectra')
            else:
                plt.plot(rv1d,i/np.nanmean(i),linewidth=0.7,alpha=0.1,color='red')
            centroids1d.append(rv1d[np.argmin(i)])
        plt.axvline(np.nanmean(centroids),color='black',alpha=0.5)
        plt.axvline(np.nanmean(centroids1d),color='red',alpha=0.5)
        plt.title(f'CCF with {star} PHOENIX spectrum in air ({np.nanmean(centroids)} km/s). See commentary in terminal for details',fontsize=8)
        plt.xlabel('Radial velocity (km/s)')
        plt.ylabel('Mean flux')
        plt.legend()
        print('')
        print('')
        print('')
        print(f'1D data:  Line center near radial velocity of {np.nanmean(centroids1d)} km/s.')
        print(f'2D data:  Line center near radial velocity of {np.nanmean(centroids)} km/s.')
        print('These values should be equal to the expected radial velocity of the system (i.e. the systemic velocity, to be found in e.g. SIMBAD or the discovery paper), perhaps minus the barycentric velocity of the Earth towards the target at the time of observation (i.e. the barycentric correction), depending on what spectrograph you are using.')
        print('Large deviations can occur if the wavelength solution from the pipeline is incorrectly assumed to be in air, or if for whatever reason, the wavelength solution is provided in the reference frame of the star.')
        print("In the former case, you need to specify in the CONFIG file of the data that the wavelength solution written by this file is in vaccuum. In the latter case, barycentric correction (and possibly Keplerian corrections) are likely implicitly taken into account in the wavelength solution, meaning that these corrections need to be switched off when running the Tayph's cascade.")
        print("")
        print("If no peak is visible at all, the spectral orders are suffering from unknown velocity shifts or continuum fluctuations that this code was not able to take out.")
        print("In this case, please inspect your data carefully to determine whether you can locate the source of the problem.")
        print("If this is 'just the way the data is' you should probably expect significant problems when using Tayph to obtain cross-correlations.")
        # plt.plot(rv,ccf[0]/np.mean(ccf[0]),color='red')
        # plt.plot(rv,ccf[10]/np.mean(ccf[10]),color='blue')
        # plt.plot(rv,ccf[20]/np.mean(ccf[20]),color='green')
        print('')
        print('')
        print('')
        plt.show()
        print('Read_e2ds successfully completed')
        print('')


        pdb.set_trace()


def test_function():
    print('ohai')
