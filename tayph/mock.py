__all__ = [
	'get_stellar_spectrum',
    'translate_phi_spectra',
    'load_mock_spectra',
    'interpolate_phases',
    'create_obs_times',
    'radial_velocity',
    'combine_spectra',
    'make_orders',
    'create_mock_obs'
]


def get_stellar_spectrum(T_eff, logg, vacuum=False):
    """
    Function:		Queries phoenix spectrum using tayph and returns wavelength and spectrum array
    Input: 			T_eff 		effective temperature of the star in Kelvin
                    logg		log(g) of the star in cgi units
                    vacuum		True if spectrograph in vacuum (optical), False if in air (IR)
    Output:			returns wavelength in micron and stellar spectrum in erg / s / cm2 / micron
    """
    import tayph.phoenix as pho
    import astropy.units as u
    import numpy as np
    import tayph.operations as ops

    stellar_spectrum = pho.get_phoenix_model_spectrum(T_eff, logg) * u.erg / u.s / u.cm**3
    stellar_wavelengths = pho.get_phoenix_wavelengths(vacuum=vacuum) * u.Angstrom

    return stellar_wavelengths.to(u.nm).value, stellar_spectrum.to(u.erg / u.s / u.cm**2 / u.nm).value


def translate_phi_spectra(dp, mock_dp, savename, phi0=0.):
    
    
    import numpy as np
    from astropy.io import fits
    import tayph.operations as ops

    phases = np.array([phi0])

    data = fits.open(dp)[0].data
    wave = ops.vactoair(data[0,:])
    transit_depth = data[1,:]

    hdul = fits.HDUList()
    
    hdul.append(fits.ImageHDU(data=phases))
    hdul.append(fits.ImageHDU(data=wave))
    hdul.append(fits.ImageHDU(data=transit_depth))

    hdul.writeto(mock_dp+savename+'.fits', overwrite=True)

    return 0


def load_mock_spectra(dp):
    """
    Reading in the fits file with the phases, wls and flux
    """
    from astropy.io import fits
    file = fits.open(dp)
    phases = file[0].data
    wl = file[1].data
    flux = file[2].data

    return phases, wl, flux

def interpolate_phases(phases, wl_planet, spectrum_planet, obs_start=0, obs_end=0, step=10, transit=True):
    """
    This interpolates the data onto a finer phase grid
    """

    import numpy as np
    import astropy.units as u
    import scipy as sc
    from scipy.interpolate import interp1d

    print('phases:', phases)
    if transit:
        phase_int = np.linspace(obs_start, obs_end, np.abs(step))
        
    else: 
        phase_int = np.linspace(obs_start, obs_end, np.abs(step))


    Fp_int = np.zeros((len(phase_int), np.shape(spectrum_planet)[1]))
    wl_int = np.zeros((len(phase_int), np.shape(spectrum_planet)[1]))
    for i in range(np.shape(spectrum_planet)[1]):
        Fp_int[:,i] = sc.interpolate.interp1d(phases, spectrum_planet[:,i], fill_value='extrapolate', kind='nearest')(phase_int) 
        wl_int[:,i] = sc.interpolate.interp1d(phases, wl_planet[:,i],  fill_value='extrapolate', kind='nearest')(phase_int) 

    return phase_int, Fp_int, wl_int


def create_obs_times(phases, texp, mock_dp, dp, berv=False, eccentric_orbit=False):
    import tayph.system_parameters as sp
    import os
    from astropy.time import Time, TimeDelta
    import astropy.units as u
    import shutil
    import numpy as np
    from PyAstronomy.pyasl import KeplerEllipse
    import radvel
	# first we need to remove the old obs_times file if it exists!
    if berv:
        baryvel = np.round(sp.astropyberv(mock_dp),8)

    if os.path.exists(f'{mock_dp}/obs_times'):
        os.remove(f'{mock_dp}/obs_times') 


    #shutil.copyfile(f'{dp}/config', f'{mock_dp}/config')
    P = ((sp.paramget('P', dp)*u.d).to(u.s)).value
    Tc = sp.paramget('Tc',dp)
    orbit = TimeDelta(P, format='sec').to_value('jd')

    ## we have the transit centre time and also the period, for a circular orbit, this down here is fine
    
    if eccentric_orbit:
        # In this case, the observation times are a bit more difficult. 
        # We need to generate datapoints around the mid-transit point without making use of the phases
        # because the phase is not really 'the same' anymore...... damn ecc.
        # so we assume that the spacing is given by readout + expt,
        # but at which phase is Tc? 
        
        # let's calculate the phase of Tc
        bjds = np.linspace(Tc-sp.paramget('t', dp)/24, Tc+sp.paramget('t', dp)/24, 2000)
        
        T_per = radvel.orbit.timetrans_to_timeperi(Tc, sp.paramget('P', dp), sp.paramget('ecc', dp), np.radians(sp.paramget('omega', dp)))
        
        ke = KeplerEllipse(
        
            a=sp.paramget('aRstar', dp),
            per=sp.paramget('P', dp),
            e=sp.paramget('ecc', dp),
            i=sp.paramget('inclination', dp), #in degrees
            Omega=0., #We don't care about the orientation in the sky
            w=sp.paramget('omega', dp), #in degrees,
            tau=T_per
        )
    
        ecc_anomaly = ke.eccentricAnomaly(bjds)
        true_anomaly = 2.*np.arctan(np.sqrt((1.+sp.paramget('ecc', dp))/(1.-sp.paramget('ecc', dp)))*np.tan(ecc_anomaly/2.))

        observation_times = []
        observation_phases = []
        
        for phi in phases:
            observation_phases.append(true_anomaly[np.argmin(np.abs(true_anomaly - phi))])
            observation_times.append(bjds[np.argmin(np.abs(true_anomaly - phi))])
        
        date = (Time(observation_times,format='jd',scale='tdb')).to_value('datetime64')
        obstimes = (Time(observation_times,format='jd',scale='tdb')).to_value('mjd')
    
    else:
        diff = TimeDelta((phases*P), format='sec').to_value('jd')
        diff[diff > orbit] += orbit
        date = (diff + Time(Tc,format='jd',scale='tdb')).to_value('datetime64')
        obstimes = (diff + Time(Tc,format='jd',scale='tdb')).to_value('mjd')

    for i in range(len(obstimes)):

    
        if berv:
            write_string = f"{str(obstimes[i])}\t{str(date[i])}\t{texp}\t{texp}\t{baryvel[i]}\n"
        else:
            write_string = f"{str(obstimes[i])}\t{str(date[i])}\t{texp}\n"

        f = open(f'{mock_dp}/obs_times', 'a')
        f.write(write_string)
        
    f.close()

    return 0


def radial_velocity(dp, phases, vorb=None):
    import numpy as np
    import tayph.system_parameters as sp

    i=sp.paramget('inclination',dp)
    vsys=sp.paramget('vsys',dp)
    if vorb==None:
        vorb=sp.v_orb(dp)
    
    rv=vorb*np.sin(2.0*np.pi*phases)*np.sin(np.radians(i)) #- vsys
    return rv

# now we need to fix the wavelengths for each of these exposures by introducing the RV shift. I am absolutely positive that this will go wrong, but let's just try...

def make_orders(blaze, wls, wlsp, wlss, Fsp, Fxs, phases, transit, mock_dp, spec, snr, scaling, eccentric_orbit):
    import tayph.util as ut
    import numpy as np
    import scipy as sc
    from scipy.interpolate import interp1d
    import tayph.system_parameters as sp
    from tqdm import tqdm 
    from joblib import Parallel,delayed
    import pdb

    n_ord = np.shape(wls)[0]
    
    def parallel_order(i):
        twoD_order = []
        twoD_wave = []
        wl = wls[i] 

        for k in range(len(phases)):

            # interpolate star and planet onto the same grid
            wlsp_i = wlsp[k][(wlsp[k] <= wl[-1]) * (wlsp[k] >= wl[0])]
            Fp_i = Fsp[k][(wlsp[k] <= wl[-1]) * (wlsp[k] >= wl[0])]
        
            wlss_i = wlss[k][(wlss[k] <= wl[-1]) * (wlss[k] >= wl[0])]
            Fs_i = Fxs[k][(wlss[k] <= wl[-1]) * (wlss[k] >= wl[0])]
            
            if len(Fs_i) == 0 or len(Fp_i)==0:
                Fp_order = np.nan * np.ones(len(wl))
                Fs_order = np.nan * np.ones(len(wl))
            else:
                Fp_order = sc.interpolate.interp1d(wlsp_i, Fp_i, fill_value="extrapolate", kind='linear')(wl)
                Fs_order = sc.interpolate.interp1d(wlss_i, Fs_i, fill_value="extrapolate", kind='linear')(wl)

            # combine star and planet!
            F_order = make_flux(Fp_order, Fs_order, transit, mock_dp, k, spec, eccentric_orbit)

            #print(F_order)
            F_comb = F_order 
            
            # import pdb
            # import matplotlib.pyplot as plt
            # pdb.set_trace()

            if type(scaling) == float:
                F_comb *= scaling
            else:
                F_comb *= scaling[i][k]
            F_comb *= blaze[i]

            if type(snr) == float:
                noise = np.random.normal(0, 1, len(F_comb))  / snr * F_comb
            
            elif len(snr[i]) == len(F_comb):
                noise = np.random.normal(0, 1, len(F_comb))  / (snr[i]) * F_comb
            else:
                noise = np.random.normal(0, 1, len(F_comb))  / (snr[i][k]) * F_comb
            F_comb += noise
        
            twoD_order.append(F_comb)
            twoD_wave.append(wl)


        twoD_order = np.asarray(twoD_order)
        twoD_wave = np.asarray(twoD_wave)
        
        ut.writefits(f'{mock_dp}/order_{i}.fits', twoD_order)
        ut.writefits(f'{mock_dp}/wave_{i}.fits', twoD_wave)

        return 0

    
    Parallel(n_jobs=1)(delayed(parallel_order)(i) for i in tqdm(range(n_ord)))

    return 0

def compute_SNR(dp, mode):
    import numpy as np
    import tayph.functions as fun
    from astropy.io import fits
    
    if mode == 'ESPRESSO': nrorders = 170
    elif mode == 'HARPS': nrorders = 72
    elif mode == 'HARPSN': nrorders = 69
    #elif mode == 'ANDES':
    elif mode == 'IGRINS': nrorders = 72
    
    else: 
        print(f"Mode {mode} not recognised, falling back to default case")
        skip = True
    
    SNR = []
    scaling = []
    
    if skip:
        SNR = 100
        scaling = 1
        
    else:
        for i in range(nrorders):
            order = fits.open(dp+f'order_{i}.fits')[0].data
            order_norm = (order.T / np.nanmean(order, axis=1)).T        # This calculates the normalised order over wavelength 
            order_res = order_norm / np.nanmean(order_norm, axis=0)     # and this normalises it over time
            Ft = np.nanmean(order, axis=1) / np.nanmean(np.nanmean(order, axis=1)) # This is a function over time that stores how the flux changes between the different exposures
            Ft_scale = np.sqrt(Ft) # SNR change over time
            order_res = (1.+ ((order_res.T-1.) / Ft_scale)).T # This scales the residual to take out the SNR change over time
            snr_norm = 1./ fun.running_MAD_2D(order_res, w=20) 
            
            SNR.append(snr_norm[np.newaxis, :] * Ft_scale[:, np.newaxis])
            scaling.append(Ft)
            #fits.writeto(outpath/('scaling_'+str(i)+'.fits'),Ft,overwrite=True)
            #fits.writeto(outpath/('snr_'+str(i)+'.fits'),snr,overwrite=True)
    
    return SNR, scaling
    

def read_SNR(dp, mode='ESPRESSO', real_data=True):
    from astropy.io import fits
    import numpy as np
    import tayph.system_parameters as sp
    import pdb
    
    if real_data:
        SNR, scaling = compute_SNR(dp, mode=mode)

    elif mode == 'ANDES':
        SNR = np.load(dp+'SNR_ANDES.npy')
        scaling = 1.
    else:
        
        try: 
            SNR = sp.paramget('SNR', dp)
            scaling = 1.
        except:
            print('No SNR specifications provided, I will just assume SNR = 100')
            SNR, scaling =  100., 1.

    return SNR, scaling


def transit_ecc(dp, return_phases=False, start=False, end=False):
    """
    Calculates the orbital phase for an eccentric orbit
    """
    import radvel
    from astropy.io import ascii 
    from astropy.time import Time
    from PyAstronomy import pyasl
    import tayph.system_parameters as sp
    import numpy as np
    
    eccentricity = sp.paramget('ecc', dp) 
    omega = sp.paramget('omega', dp)
    period = sp.paramget('P', dp)
    T0 = sp.paramget('Tc', dp)
    aRs = sp.paramget('aRstar', dp)
    inclination = sp.paramget('inclination', dp)
    RpRs = sp.paramget('RpRstar', dp)
    lampoo = sp.paramget('lampoo', dp)
    T_per = radvel.orbit.timetrans_to_timeperi(T0, period, eccentricity, np.radians(omega))
    
    # converts the mjds in the obs_times files to bjds
    d=ascii.read(f'{dp}/obs_times',comment="#")
    mjds = Time(d['col1'], format='mjd')
    
    bjds = mjds.jd
    
    ke = pyasl.KeplerEllipse(
        
            a=aRs,
            per=period,
            e=eccentricity,
            i=inclination, #in degrees
            Omega=0., #We don't care about the orientation in the sky
            w=omega, #in degrees,
            tau=T_per
        )
    
    ecc_anomaly = ke.eccentricAnomaly(bjds)
    true_anomaly = 2.*np.arctan(np.sqrt((1.+eccentricity)/(1.-eccentricity))*np.tan(ecc_anomaly/2.))
    
    radius_vec = aRs * (1.-eccentricity**2)/(1.+eccentricity*np.cos(true_anomaly))
    
    node = np.radians(lampoo) 
    true_lat = np.radians(omega) + ke.trueAnomaly(bjds)

    x_ = radius_vec*(-np.cos(node)*np.cos(true_lat)+np.sin(node)*np.sin(true_lat)*np.cos(np.radians(inclination)))
    y_ = radius_vec*(-np.sin(node)*np.cos(true_lat)-np.cos(node)*np.sin(true_lat)*np.cos(np.radians(inclination)))
    z_ = radius_vec*np.sin(true_lat)*np.sin(np.radians(inclination))
    
    rho   = np.sqrt(x_**2+y_**2) 
    dmin  = rho-RpRs #minimal distance between the planet ellipse and the origin
    dmax  = rho+RpRs #maximal distance between the planet ellipse and the origin

    transit = np.zeros_like(bjds)
    
    transit[z_ < 0.] = 1. # planet is out of transit
    transit[dmin >= 1.] = 1 # planet is not overlapping with the stellar disk. 
    
    #print(transit)
    if return_phases: return true_anomaly, transit
    else: return transit  
    

def make_flux(fxp, fxs, transit, mock_dp, exp_nr, spec='depth', eccentric_orbit=False):
    import scipy as sc
    from scipy.interpolate import interp1d
    import astropy.units as u
    import numpy as np
    from tqdm import tqdm
    import pdb
    import tayph.system_parameters as sp
    from joblib import delayed, Parallel
    import astropy.units as u
    from bokeh.plotting import figure, show
    import tayph.util as ut
    import sys
    sys.path.insert(0, '/data/bibi/Papers/narrowBandCalcium/NUMPYRO-RUNS/')

    if eccentric_orbit:
        intransit = transit_ecc(mock_dp)
    else:
        intransit=sp.transit(mock_dp)

    if transit:
        mask=(intransit-1.0)/(np.min(intransit-1.0))
    else: #Emission
        mask=intransit*0.0+1.0
        #print(mask)
        
    if spec=='depth':
        
        fxp_flux = fxs * (1. + mask[exp_nr] * (fxp - 1.0))

    if spec=='flux':
        #Rs = sp.paramget('Rs', mock_dp) # reads the stellar radius
        #print(mask)
        fxp_flux = fxs * ( 1. + mask[exp_nr]*fxp*(sp.paramget('RpRstar', mock_dp))**2)
        #*(Rs * u.Rsun)**2
        
    return fxp_flux

def RV_planet_ecc(dp):
    
    import astropy.units as u
    import tayph.system_parameters as sp
    
    RVstar = RV_star_ecc(dp)
    
    #RVstar = RV_star_ecc(dp)
    vsys = sp.paramget('vsys', dp)
    Ms = sp.paramget('Ms', dp) * u.Msun
    Mp = sp.paramget('Mp', dp) * u.Mjup
    

    RVplanet = -1. * (RVstar) * (Ms / Mp).decompose()
    return RVplanet

def RV_star_ecc(dp):
    import radvel
    from PyAstronomy import modelSuite as ms
    from astropy.io import ascii 
    from astropy.time import Time
    import numpy as np
    import tayph.system_parameters as sp
    
    eccentricity = sp.paramget('ecc', dp) 
    K = sp.paramget('K', dp)
    omega = sp.paramget('omega', dp)
    period = sp.paramget('P', dp)
    T0 = sp.paramget('Tc', dp)
    Mstar = sp.paramget('Ms', dp)
    vsys =  sp.paramget('vsys', dp)
    incl = sp.paramget('inclination', dp)
    a = sp.paramget('a', dp)
    Mp = sp.paramget('Mp', dp)

    T_per = radvel.orbit.timetrans_to_timeperi(T0, period, eccentricity, np.radians(omega))

    # converts the mjds in the obs_times files to bjds
    d=ascii.read(f'{dp}/obs_times',comment="#")
    mjds = Time(d['col1'], format='mjd')
    bjds = mjds.jd
    
    bjds = mjds.jd
    

    rv = ms.KeplerRVModel()
    rv.assignValue({"per1":period,
                    'K1':K,
                    'e1':eccentricity,
                    'tau1':T_per,
                    'w1':omega,
                    'mstar':Mstar,
                    'c0':vsys,
                    'a1':a,
                    'msini1':Mp*np.sin(np.radians(incl))})
    
    RVs = rv.evaluate(bjds)
    return RVs - vsys

def create_mock_obs(configfile, create_obs_file=True, mode='HARPS', real_data=False, spec='flux', rot=True, eccentric_orbit=False, phase_range=[]):
    import sys
    import scipy as sc
    from scipy.interpolate import interp1d
    import os
    import os.path
    from pathlib import Path
    from glob import glob
    import tayph.system_parameters as sp
    import astropy.units as u
    import numpy as np
    from astropy.io import fits
    import pandas as pd
    import tayph.models as models
    import astropy.constants as const
    import tayph.operations as ops
    import scipy as sc
    import tayph.functions as fun
    import pdb
    from tqdm import tqdm
    from PyAstronomy import pyasl
    import bokeh.plotting as bkplt
    from joblib import delayed, Parallel
    import sys
    sys.path.insert(0, '/data/bibi/Papers/narrowBandCalcium/NUMPYRO-RUNS/')
    
    cf = configfile

    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print(' = = = = = = = = = = = = = = = = = = = = = = = = ')
    print(" = = = = WELCOME TO TAYPH'S MOCK MODULE! = = = = ")
    print(' = = = = = = = = = = = = = = = = = = = = = = = = ')
    print('')
    print(f'           Running {cf}                  ')
    print('')
    print(' = = = = = = = = = = = = = = = = = = = = = = = = ')
    print('')
    print('')
    print('')

    if mode == 'HARPS-N':
        mode = 'HARPSN'
    
    # Reading in all parameters in the configuration file 
    print(f'--- Load parameters from {cf}')
    
    transit = sp.paramget('transit', cf, full_path=True)
    dp = sp.paramget('datapath',cf,full_path=True)
    mock_dp = sp.paramget('mockdatapath', cf, full_path=True)
    mock_data = sp.paramget('mockdata', cf, full_path=True)
    
    if eccentric_orbit: 
        
        ecc = sp.paramget('ecc', dp)
        omega = sp.paramget('omega', dp)
        lda = sp.paramget('lampoo', dp)
        print(f'--- Assuming an eccentric orbit with e={ecc}, omega={omega} and lambda={lda}.')
        #transit = sp.paramget('transit', cf, full_path=True)
        
    else:
        print(f'--- Assuming a circular orbit')
        
    Rp = sp.paramget('Rp', dp) 
    Rs = (((sp.paramget('RpRstar', dp))**(-1)*Rp)*u.Rjup).value
    resolution = sp.paramget('resolution', dp)
    period = sp.paramget('P', dp)
    inclination = sp.paramget('inclination', dp)
    T_eff = sp.paramget('Teff', dp)
    logg = sp.paramget('logg', dp)
    
    print(f"--- Your instrument of choice is {mode} with a resolution of {int(resolution)}.")
    print(f"--- Using planetary model {mock_data}.")
    c=const.c.to('km/s').value

    if real_data:
        snr, scaling = read_SNR(dp, mode=mode, real_data=real_data)
        exp_time = pd.read_csv(dp+'/obs_times', header=None, sep='\t').iloc[0,4] # takes the exposure time from the first exposure in the obs file
        
    else:
        #snr = sp.paramget('SNR', dp) # if no data given, we need to specify this from the ETC
        snr, scaling = read_SNR(dp, mode=mode, real_data=real_data)
        exp_time = sp.paramget('exp_time', dp) # if no data given, we need to specify the exposure time
        t = sp.paramget('t', dp) # DURATION OF OBSERVATION
        read_out = sp.paramget('readout', dp)
   
    phases, wl_planet, spectrum_planet = load_mock_spectra(mock_dp+mock_data)  # load in the data

    if len(phases)==1:
        print('--- The same spectrum of the planet will be used for all the phases. I guess you are working with one simple model spectrum.')
        
        if real_data:
            
            if eccentric_orbit:
                phases_int, transit_int = transit_ecc(dp, return_phases=True)
            else:
                phases_int = sp.phase(dp)
            
            print(phases_int)

            if len(wl_planet) == 1:
                wl_planet = np.reshape(wl_planet, (np.shape(wl_planet)[1],))
            spectrum_planet_int = np.zeros((len(phases_int), len(wl_planet)))
            wl_planet_int = np.zeros((len(phases_int), len(wl_planet)))

            for i in range(len(phases_int)):
                spectrum_planet_int[i] = spectrum_planet
                wl_planet_int[i] = wl_planet                

        else: 
            
            if len(phase_range) == 0:
                delta_t = (0.5*t * u.h) 
                phi0 = phases[0]
                phi_start = phi0 -(delta_t / (period * u.d)).to(1).value
                phi_end = phi0 + (delta_t / (period * u.d)).to(1).value
                delta_phi = (30 * u.min / (period * u.d)).to(1).value

                phi_start -= delta_phi
                phi_end +=delta_phi 
                
            else:
                phi_start = phase_range[0]
                phi_end = phase_range[-1]
                delta_phi = (30 * u.min / (period * u.d)).to(1).value

            tt = t + 1. #total transit time plus an hour for baseline as it is default, readout = 70 s
            expnr = int((tt * u.h / ((exp_time + read_out)*u.s)).to(1).value)
            print(f'----- Per default, I will add 30 mins baseline on both sides')
            print(f'----- I created {expnr} exposures for you with an exp time of {exp_time} s, a readout time of {read_out} s, spanning phases {np.round(phi_start, 3)} to {np.round(phi_end, 3)}.')
            #print(f'----- The range starts at phase {np.round(phi_start+delta_phi, 3)} and ends at {np.round(phi_end-delta_phi, 3)}')

            phases_int = np.linspace(phi_start, phi_end, expnr)
            if len(wl_planet) == 1:
                wl_planet = np.reshape(wl_planet, (np.shape(wl_planet)[1],))
            spectrum_planet_int = np.zeros((len(phases_int), len(wl_planet)))
            wl_planet_int = np.zeros((len(phases_int), len(wl_planet)))

            print(f'----- The phases are:', phases_int)
            for i in range(len(phases_int)):
                spectrum_planet_int[i] = spectrum_planet
                wl_planet_int[i] = wl_planet
        
            scaling = 1. #np.ones_like(phases_int)
            #snr = snr*np.ones_like(phases_int)
            #import pdb
            #pdb.set_trace()
    else: 

        if real_data:
            obs_start, obs_end = sp.phase(dp)[0], sp.phase(dp)[-1]
            nexp = len(sp.phase(dp))
            
        else: 
            delta_t = (0.5*t * u.h)
            phi_start = -(delta_t / (period * u.d)).to(1).value
            phi_end = (delta_t / (period * u.d)).to(1).value
            delta_phi = (30 * u.min / (period * u.d)).to(1).value

            phi_start -= delta_phi
            phi_end +=delta_phi 

            tt = t + 1. #total transit time plus an hour for baseline as it is default, readout = 70 s
            expnr = int((tt * u.h / ((exp_time + read_out)*u.s)).to(1).value)

            obs_start, obs_end = phi_start, phi_end
            nexp = expnr
        
        
        phases_int, spectrum_planet_int, wl_planet_int = interpolate_phases(phases, wl_planet, spectrum_planet, obs_start=obs_start, obs_end=obs_end, step=nexp, transit=transit) 
        print(phases_int)
        scaling= 1. 


    if rot:
        # Daniel's models are crazy high res, so I am going to interpolate it onto a less fine grid.
        print('--- Blurring planetary spectrum')

        # wlt_cv = np.linspace(wl_planet[0], wl_planet[-1], int(len(wl_planet) / 5))
        # T_cv = sc.interpolate.interp1d(wl_planet, spectrum_planet)(wlt_cv)
        # wl_planet = wlt_cv * 1.0
        # spectrum_planet = T_cv * 1.0

        def parallel_blur_rotate(i):
        #for i in tqdm(range(len(phases_int))):
            fxp_b= ops.blur_rotate(wl_planet_int[i],spectrum_planet_int[i],c/resolution,Rp,period,inclination)
            oversampling = 1.
            wl_planet_int_new,spectrum_planet_int_new,vstep=ops.constant_velocity_wl_grid(wl_planet_int[i],fxp_b,oversampling=oversampling)
            

            return wl_planet_int_new, spectrum_planet_int_new

        wl_new, spec_new = zip(*Parallel(n_jobs=len(phases_int))(delayed(parallel_blur_rotate)(i) for i in range(len(phases_int))))

        wl_new = np.asarray(wl_new)
        spec_new = np.asarray(spec_new)

        wl_planet_int = wl_new
        spectrum_planet_int = spec_new

    if create_obs_file:
        print('--- Writing obs times')
        create_obs_times(phases_int, exp_time, mock_dp, dp, berv=False, eccentric_orbit=eccentric_orbit)
    else: 
        print('No obs_times created. I guess you already have one?')


    print('--- Extracting stellar spectrum')
    wl_star_i, spectrum_star_i = get_stellar_spectrum(T_eff, logg, vacuum=False) # load the stellar spectrum in air, sadly, it is super big, so we need to crop it
    
    if len(phases)==1:
        wl_star = wl_star_i[(wl_star_i >= (wl_planet[0])-50) * (wl_star_i <= (wl_planet[-1] + 50))]
        spectrum_star = spectrum_star_i[(wl_star_i >= (wl_planet[0])-50) * (wl_star_i <= (wl_planet[-1] + 50))]
    else:
        wl_star = wl_star_i[(wl_star_i >= (wl_planet[0][0])-50) * (wl_star_i <= (wl_planet[0][-1] + 50))]
        spectrum_star = spectrum_star_i[(wl_star_i >= (wl_planet[0][0])-50) * (wl_star_i <= (wl_planet[0][-1] + 50))]
    
    print('----- Rotation broadening of the star')
    wl_star_evenly = np.linspace(wl_star[0] * 10, wl_star[-1] * 10, 2*len(wl_star))
    flux_evenly = sc.interpolate.interp1d(wl_star*10, spectrum_star, fill_value="extrapolate", kind='nearest')(wl_star_evenly)

    #print('hello')sp.paramget('vsini',mock_dp)
    fxs_cv = pyasl.fastRotBroad(wl_star_evenly, flux_evenly, 0.0, sp.paramget('vsini',mock_dp))

    wl_star = wl_star_evenly / 10.
    spectrum_star = fxs_cv

    wlts = np.zeros((len(phases_int), len(wl_star)))
    fxts = np.zeros((len(phases_int), len(wl_star)))

    for i in range(len(phases_int)):
        wlts[i] = wl_star
        fxts[i] = spectrum_star

    print('--- Shift planetary spectrum into system restframe')
    # This is positive!
    if eccentric_orbit:
        v_planet = RV_planet_ecc(mock_dp) + sp.paramget('vsys',mock_dp)  * np.ones(len(phases_int))
    else:
        v_planet = radial_velocity(mock_dp, phases_int) + sp.paramget('vsys',mock_dp)  * np.ones(len(phases_int))
    wlp = [(1+v_planet[i]/c)*wl_planet_int[i] for i in range(len(phases_int))]
    fxp = spectrum_planet_int


    print('--- Shift stellar spectrum into system restframe')
    # this is positive, because the code already accounts for the different moving direction of the star and the planet
    
    if eccentric_orbit:
        v_star = RV_star_ecc(mock_dp) + sp.paramget('vsys',mock_dp)  * np.ones(len(phases_int))
    else:
        v_star = sp.RV_star(mock_dp) + sp.paramget('vsys',mock_dp)  * np.ones(len(phases_int))
    wls_sys = [(1+v_star[i]/c)*wlts[i] for i in range(len(phases_int))]
    fxs_sys = fxts
    
    #print('--- Merging star and planet...')
    #wlsp, fxsp = make_flux(wlp, fxp, wls_sys, fxs_sys, transit, mock_dp, spec)
    #import pdb
    #pdb.set_trace()
            
    ## we need to write the berv values into the obs file
    #print('--- Adding berv to obsfile')
    
    if not real_data:
       create_obs_times(phases_int, exp_time, mock_dp, dp, berv=True, eccentric_orbit=eccentric_orbit)

    # import pdb
    # pdb.set_trace()
    print(f'--- Creating {mode} orders')

    if mode=='HARPS':
        wls_orders = np.load(mock_dp+'order_def_HARPS.npy')
        blaze = np.load(mock_dp+'blaze_HARPS.npy')
        
        make_orders(blaze, wls_orders, wlp, wls_sys, fxp, fxs_sys, phases_int, transit, mock_dp, spec, snr, scaling, eccentric_orbit)
    
    elif mode=='HARPSN':
        wls_orders = np.load(mock_dp+'order_def_HARPSN.npy')
        blaze = np.load(mock_dp+'blaze_HARPSN.npy')
        
        make_orders(blaze, wls_orders, wlp, wls_sys, fxp, fxs_sys, phases_int, transit, mock_dp, spec, snr, scaling, eccentric_orbit)

    elif mode=='ESPRESSO':
        wls_orders = np.load(mock_dp+'order_def_ESPRESSO.npy', allow_pickle=True)
        blaze = np.load(mock_dp+'blaze_ESPRESSO.npy', allow_pickle=True)

        make_orders(blaze, wls_orders, wlp, wls_sys, fxp, fxs_sys, phases_int, transit, mock_dp, spec, snr, scaling, eccentric_orbit)
    
    elif mode=='IGRINS':
        wls_orders = np.load(mock_dp+'order_def_IGRINS_HK.npy')
        blaze = np.ones(len(wls_orders))        
        make_orders(blaze, wls_orders, wlp, wls_sys, fxp, fxs_sys, phases_int, transit, mock_dp, spec, snr, scaling, eccentric_orbit)
    
    elif mode=='SPIROU':
        wls_orders = np.load(mock_dp+'order_def_SPIROU.npy')
        blaze = np.load(mock_dp+'blaze_SPIROU.npy')  

        make_orders(blaze, wls_orders, wlp, wls_sys, fxp, fxs_sys, phases_int, transit, mock_dp, spec, snr, scaling, eccentric_orbit)
    
    elif mode=='ANDES':
        wls_orders = np.load(mock_dp+'order_def_ANDES.npy')
        blaze = np.ones(len(wls_orders))
        make_orders(blaze, wls_orders, wlp, wls_sys, fxp, fxs_sys, phases_int, transit, mock_dp, spec, snr, scaling, eccentric_orbit)

    return 0




# def translate_GCM_spectra(dp, mock_dp, savename, mode='emission', tm_scaling=1.):
#     import glob
#     import tayph.system_parameters as sp
#     import astropy.units as u
#     import numpy as np
#     from pathlib import Path
#     import tayph.util as ut
#     import tayph.operations as ops
#     import os
#     from astropy.io import fits
#     import astropy.units as u

#     files = glob.glob(mock_dp+'/*.txt')
#     files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
#     Rp = (sp.paramget('Rp', dp) * u.Rjup).to(u.cm).value

#     phases = []
#     Fps = []
#     wls = []

#     for i in range(len(files)):
#         print(files[i])
#         data = np.loadtxt(files[i], skiprows=1) #skipping the first few points
#         head = np.loadtxt(files[i], max_rows=1)
#         if mode=='emission':
#             wl = (data[:,0] * u.micron).to(u.nm).value
#             frac = data[:,1]
#             Ltot = (data[:,2]) 
#             Rp = head[1] # head[1] will increase the flux (2 or 3 times stronger), head[2] Rp + atm
#             phase = 1.- (head[3] - 180.)/360.
#             if phase >=1:
#                 phase -=1.
#             # Formula for spectra in erg s-1 cm-2 um-1
#             Fps.append((((frac * Ltot) / (Rp**2))* u.erg / u.s / u.cm**2 / u.cm).to(u.erg / u.s / u.cm**2 / u.nm).value)
#             phases.append(phase)
#             wls.append(ops.vactoair(wl))

#         if mode=='transmission':
#             wl = (data[:,0] * u.micron).to(u.nm).value
#             Rp = head[1]
#             sumf = data[:,1]
#             Fps.append(1.- tm_scaling*sp.paramget('RpRstar', dp)**2*(1. + 2.0 * sumf[:]/Rp**2))
#             phase = (-(head[3] - 180.)/360.)
#             phases.append(phase)
#             wls.append(ops.vactoair(wl))

#     if mode =='transmission':
#         phases = np.asarray(phases[::-1])
#         wls = np.asarray(wls[::-1])
#         Fps = np.asarray(Fps[::-1])
#     else:

#         phases = np.asarray(phases)
#         wls = np.asarray(wls)
#         Fps = np.asarray(Fps)

#     print(phases)
    
#     hdul = fits.HDUList()
#     hdul.append(fits.ImageHDU(data=phases))
#     hdul.append(fits.ImageHDU(data=wls))
#     hdul.append(fits.ImageHDU(data=Fps))

#     hdul.writeto(mock_dp+savename+'.fits', overwrite=True)
    

#     return 0

# def translate_depth_spectra(dp, mock_dp, savename):
#     """
#     The spectra come in the planetary restframe in vacuum. Need to do vac to air correction
#     Additionally, they are in transit depths, so we need to make sure the code knows that!
#     This can actually be extended to starting from scratch?
#     """
#     import numpy as np
#     from astropy.io import fits
#     import tayph.operations as ops

#     phases = np.array([0])

#     data = fits.open(dp)[0].data
#     wave = ops.vactoair(data[0,:])
#     transit_depth = data[1,:]

#     hdul = fits.HDUList()
    
#     hdul.append(fits.ImageHDU(data=phases))
#     hdul.append(fits.ImageHDU(data=wave))
#     hdul.append(fits.ImageHDU(data=transit_depth))

#     hdul.writeto(mock_dp+savename+'.fits', overwrite=True)

#     return 0