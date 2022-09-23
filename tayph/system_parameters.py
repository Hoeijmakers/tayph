__all__ = [
    "check_dp",
    "t_eff",
    "paramget",
    "berv",
    "airmass",
    "v_orb",
    "astropyberv",
    "calculateberv",
    "phase",
    "RV_star",
    "transit",
    "RV",
    "dRV"
]

def check_dp(dp):
    """
    This is a helper function that checks for the presence of a config file
    and an obs_times file at path dp. It also checks that dp is a Path object and returns
    it as such.
    """
    from tayph.util import check_path
    from pathlib import Path
    check_path(dp)
    if isinstance(dp,str):
        dp = Path(dp)
    p1=dp/'obs_times'
    p2=dp/'config'
    check_path(p1,exists=True)
    check_path(p2,exists=True)
    return(dp)


def paramget(keyword,dp,full_path=False,force_string = False):
    """This code queries a planet system parameter from a config file located in the folder
    specified by the path dp; or run configuration parameters from a file speciefied by the full
    path dp, if full_path is set to True.

    Parameters
    ----------
    keyword : str
        A keyword present in the cofig file.

    dp : str, Path
        Output filename/path.

    full_path: bool
        If set, dp refers to the actual file, not the location of a folder with a config.dat;
        but the actual file itself.

    Returns
    -------
    value : int, float, bool, str
        The value corresponding to the requested keyword.

    """
    from tayph.vartests import typetest
    from tayph.util import check_path
    import pathlib
    import distutils.util
    import pdb


    #This is probably the only case where I don't need obs_times and config to exist together...
    dp=check_path(dp)
    typetest(keyword,str,'keyword in paramget()')

    if isinstance(dp,str) == True:
        dp=pathlib.Path(dp)
    try:
        if full_path == False:
            dp = dp/'config'
        f = open(dp, 'r')
    except FileNotFoundError:
        raise FileNotFoundError('parameter file does not exist at %s' % str(dp)) from None
    x = f.read().splitlines()
    f.close()
    n_lines=len(x)
    keywords={}
    for i in range(0,n_lines):
        line=x[i].split()
        if len(line) > 1:
            if force_string:
                value=(line[1])
            else:
                try:
                    value=float(line[1])
                except ValueError:
                    try:
                        value=bool(distutils.util.strtobool(line[1]))
                    except ValueError:
                        value=(line[1])
            keywords[line[0]] = value
    try:
        return(keywords[keyword])
    except KeyError:
        # print(keywords)
        raise Exception('Keyword %s is not present in parameter file at %s' % (keyword,dp)) from None

def t_eff(M,R):
    """This function computes the mass and radius of a star given its mass and radius relative to solar."""
    #WHY IS THIS HERE? IS IT USED IN TAYPH?
    from tayph.vartests import typetest
    import numpy as np
    import astropy.constants as const

    typetest(M,[int,float],'M in t_eff()')
    typetest(R,[int,float],'R in t_eff()')
    M=float(M)
    R=float(R)

    Ms = const.M_sun
    Rs = const.R_sun
    Ls = const.L_sun
    sb = const.sigma_sb

    if M < 0.43:
        a = 0.23
        b = 2.3
    elif M < 2:
        a = 1.0
        b = 4.0
    elif M < 55:
        a = 1.4
        b = 3.5
    else:
        a = 32000.0
        b = 1.0

    T4 = a*M**b * Ls / (4*np.pi*R**2*Rs**2*sb)
    return(T4**0.25)


def berv(dp):
    """This retrieves the BERV corrcetion tabulated in the obs_times table.
    Example: brv=berv('data/Kelt-9/night1/')
    The output is an array with length N, corresponding to N exposures. These values
    are / should be taken from the FITS header.
    """
    from astropy.io import ascii
    from pathlib import Path
    import tayph.util as ut
    dp=ut.check_path(dp,exists=True)#Path object

    d=ascii.read(ut.check_path(dp/'obs_times',exists=True),comment="#")
    try:
        berv = d['col5']#Needs to be in col 5.
    except:
        raise Exception(f'Runtime error in sp.berv(): col5 could not be indexed. Check the integrity of your obst_times file located at {dp}.')
    return berv.data

def airmass(dp):
    """This retrieves the airmass tabulated in the obs_times table.
    Example: brv=airmass('data/Kelt-9/night1/')
    The output is an array with length N, corresponding to N exposures. These values
    are / should be taken from the FITS header.
    """
    from astropy.io import ascii
    from pathlib import Path
    import tayph.util as ut
    dp=ut.check_path(dp,exists=True)#Path object

    d=ascii.read(ut.check_path(dp/'obs_times',exists=True),comment="#")
    try:
        airm = d['col5']#Needs to be in col 5.
    except:
        raise Exception(f'Runtime error in sp.airmass(): col6 could not be indexed. '
        f'Check the integrity of your obst_times file located at {dp}.')
    return airm.data


def SNR(dp):
    """This retrieves the SNR tabulated in the obs_times table.
    Example: snr=sp.SNR('data/Kelt-9/night1/')
    The output is an array with length N, corresponding to N exposures. These values
    are / should be taken from the FITS header. Currently only active for ESPRESSO, at 550 nm.
    """
    from astropy.io import ascii
    from pathlib import Path
    import tayph.util as ut
    dp=ut.check_path(dp,exists=True)#Path object

    d=ascii.read(ut.check_path(dp/'obs_times',exists=True),comment="#")
    try:
        snr = d['col7']#Needs to be in col 5.
    except:
        raise Exception(f'Runtime error in sp.SNR(): col8 could not be indexed. '
        f'Check the integrity of your obst_times file located at {dp}.')
    return snr.data


def v_orb(dp):
    """
    This program calculates the orbital velocity in km/s for the planet in the
    data sequence provided in dp, the data-path. dp starts in the root folder,
    i.e. it starts with data/projectname/. This assumes a circular orbit.

    The output is a number in km/s.

    Parameters
    ----------
    dp : str, path like
        The path to the dataset containing the config file.

    Returns
    -------
    v_orb : float
        The planet's orbital velocity.

    list_of_sigmas_corrected : list
        List of 2D error matrices, telluric corrected.

    """
    import numpy as np
    import pdb
    import astropy.units as u
    from tayph.vartests import typetest,postest

    dp=check_dp(dp)#Path object
    P=paramget('P',dp)
    r=paramget('a',dp)
    typetest(P,float,'P in sp.v_orb()')
    typetest(r,float,'r in sp.v_orb()')
    postest(P,'P in sp.v_orb()')
    postest(r,'r in sp.v_orb()')

    return (2.0*np.pi*r*u.AU/(P*u.d)).to('km/s').value





def astropyberv(dp):
    """
    This does the same as berv(dp), but uses astropy to compute the BERV for the
    dates of observation given a data parameter file.
    Useful if the BERV keyword was somehow wrong or missing, or if you wish to
    cross-validate. Requires latitude, longitude, ra, dec and elevation to be provided in
    the config file as lat, long, RA, DEC and elev in units of degrees and meters.
    Date should be provided as mjd.
    """
    from tayph.vartests import typetest
    from pathlib import Path
    import numpy as np
    from astropy.io import ascii
    from astropy.time import Time
    from astropy import units as u
    from astropy.coordinates import SkyCoord, EarthLocation
    dp=check_dp(dp)#Path object
    d=ascii.read(dp/'obs_times',comment="#")#,names=['mjd','time','exptime','airmass'])
    #Not using named columns because I may not know for sure how many columns
    #there are, and read-ascii breaks if only some columns are named.
    #The second column has to be an MJD date array though.
    dates = d['col1']
    RA=paramget('RA',dp)
    DEC=paramget('DEC',dp)
    typetest(RA,str,'RA in sp.astropyberv()')
    typetest(DEC,str,'DEC in sp.astropyberv()')
    berv = []
    observatory = EarthLocation.from_geodetic(lat=paramget('lat',dp)*u.deg, lon=paramget('long',dp)*u.deg, height=paramget('elev',dp)*u.m)
    sc = SkyCoord(RA+' '+DEC, unit=(u.hourangle, u.deg))
    for date in dates:
        barycorr = sc.radial_velocity_correction(obstime=Time(date,format='mjd'), location=observatory).to(u.km/u.s)
        berv.append(barycorr.value)
    return berv


def calculateberv(date,earth_coordinates,ra,dec,mode=False):
    """This is a copy of the astropyberv above, but as a function for a single
    date in mjd. lat, long, RA, DEC and elev are in units of degrees and meters.


    Parameters
    ----------
    mode : str
        Specify the spectrograph mode you wish to use

    Returns
    -------
    barycorr : float
        The correction for the barycentric velocity

    """
    from astropy.time import Time
    from astropy import units as u
    from astropy.coordinates import SkyCoord, EarthLocation, Angle


    if mode == False:
        raise ValueError(f"No mode specified for the function system_parameters")

    elif mode == "FIES":
        observatory = EarthLocation.from_geocentric(x=earth_coordinates[0],
                                                    y=earth_coordinates[1],
                                                    z=earth_coordinates[2],
                                                    unit=u.m)
        sc = SkyCoord(ra=ra * u.deg,
                      dec=dec * u.deg)

    elif mode in ['UVES-red', 'UVES-blue',"FOCES"]:
        observatory = EarthLocation.from_geodetic(lat=earth_coordinates[0]*u.deg,
                                                  lon=earth_coordinates[1]*u.deg,
                                                  height=earth_coordinates[2]*u.m)
        sc = SkyCoord(f'{ra} {dec}', unit=(u.hourangle, u.deg))

    barycorr = sc.radial_velocity_correction(obstime=Time(date,format='mjd'), location=observatory).to(u.km/u.s)
    return(barycorr.value)



def phase(dp,start=False,end=False):
    """
    Calculates the orbital phase of the planet in the data
    sequence provided using the parameters in dp/config and the timings in
    dp/obstimes.

    The output is an array with length N, corresponding to N exposures.

    Be CAREFUL: This program provides a time difference of ~1 minute compared
    to IDL/calctimes. This likely has to do with the difference between HJD
    and BJD, and the TDB timescale. In the future you should have a thorough
    look at the time-issue, because you should be able to get this right to the
    second. At the very least, make sure that the time conventions are ok.

    More importantly: The transit center time needs to be provided in config
    in BJD.

    As of 20-08-2022, this function returns the phase at the *middle* of the observation as it is
    supposed to, by adding half of the exposure time to each timestamp. This is relied on by
    functions such as inject_model, transit and construct_kpvsys, as per isssue #113.

    Set the start keyword to return back to the old functionality to get the phase at the start of
    each exposure.
    """
    from tayph.vartests import typetest
    import numpy as np
    from astropy.io import ascii
    from astropy.time import Time
    from astropy import units as u, coordinates as coord
    import tayph.util as ut
    import pdb
    dp=check_dp(dp)#Path object
    d=ascii.read(dp/'obs_times',comment="#")#,names=['mjd','time','exptime','airmass'])
    #Not using the named columns because I may not know for sure how many columns
    #there are, and read-ascii breaks if only some columns are named.
    #The second column has to be a date array though.

    # t = Time(d['col2'],scale='utc', location=coord.EarthLocation.of_site('paranal'))# I determined that the difference between this and geodetic 0,0,0 is zero.
    t = Time(d['col2'],scale='utc', location=coord.EarthLocation.from_geodetic(0,0,0))

    jd = t.jd
    P=paramget('P',dp)
    RA=paramget('RA',dp)
    DEC=paramget('DEC',dp)
    Tc=paramget('Tc',dp)#Needs to be given in BJD!

    typetest(P,float,'P in sp.phase()')
    typetest(Tc,float,'Tc in sp.phase()')
    typetest(RA,str,'RA in sp.phase()')
    typetest(DEC,str,'DEC in sp.phase()')
    typetest(start,bool,'start in sp.phase()')
    typetest(end,bool,'end in sp.phase()')

    if start == True and end == True:
        raise Exception("Error in sp.phase(): start and end can't both be true.")

    ip_peg = coord.SkyCoord(RA,DEC,unit=(u.hourangle, u.deg), frame='icrs')
    ltt_bary = t.light_travel_time(ip_peg)

    n=0.0
    Tc_n=Time(Tc,format='jd',scale='tdb')
    while Tc_n.jd >= min(jd):
        Tc_n=Time(Tc-100.0*n*P,format='jd',scale='tdb')#This is to make sure that the Transit central time PRECEDES the observations (by tens or hundreds or thousands of years). Otherwise, the phase could pick up a minus sign somewhere and be flipped. I wish to avoid that.
        n+=1
    BJD = t.tdb + ltt_bary

    if start == True:
        diff = BJD-Tc_n
    elif end == True:
        diff = BJD+texp(dp)/3600.0/24.0/2-Tc_n
    else:
        diff = BJD+0.5*texp(dp)/3600.0/24.0/2-Tc_n#This adds half the exposure time to the timestamp, because MJD_OBS is measured at the start of the frame.


    phase=((diff.jd) % P)/P
    return phase

def transit(dp,p=[]):
    """This code uses Ians astro python routines for the approximate Mandel &
    Agol transit lightcurve to produce the predicted transit lightcurve for the
    planet described by the configfile located at dp/config.
    This all assumes a circular orbit.
    ===========
    Derivation:
    ===========
    occultnonlin_small(z,p, cn) is the algorithm of the Mandel&Agol derivation.
    z = d/R_star, where d is the distance of the planet center to the LOS to the
    center of the star.
    sin(alpha) = d/a, with a the orbital distance (semi-major axis).
    so sin(alpha)*a/Rstar = d/a*a/Rstar = d/Rstar = z.
    a/Rstar happens to be a quantity that is well known from the transit light-
    curve. So z = sin(2pi phase)*a/Rstar. But this is in the limit of i = 90.

    From Cegla 2016 it follows that z = sqrt(xp^2 + yp^2). These are given
    as xp = a/Rstar sin(2pi phase) and yp = -a/Rstar * cos(2pi phase) * cos(i).

    The second quantity, p, is Rp/Rstar, also well known from the transit light-
    curve.

    cn is a four-element vector with the nonlinear limb darkening coefficients.
    If a shorter sequence is entered, the later values will be set to zero.
    By default I made it zero; i.e. the injected model does not take into
    account limb-darkening.
    """
    from tayph.vartests import typetest
    import tayph.util as ut
    import tayph.iansastropy as iap
    import numpy as np
    import pdb
    dp=ut.check_path(dp)
    if len(p)==0:
        p=phase(dp)
    else:
        p=np.array(p)
    a_Rstar=paramget('aRstar',dp)
    Rp_Rstar=paramget('RpRstar',dp)
    i=paramget('inclination',dp)
    typetest(a_Rstar,float,'Rp_Rstar')
    typetest(a_Rstar,float,'a_Rstar')
    typetest(i,float,'i')

    xp=np.sin(p*2.0*np.pi)*a_Rstar
    yp=np.cos(p*2.0*np.pi)*np.cos(np.radians(i))*a_Rstar
    z=np.sqrt(xp**2.0 + yp**2.0)
    transit=iap.occultnonlin_small(z,Rp_Rstar,[0.0,0.0])
    return transit



def RV_star(dp):
    """
    This calculates the radial velocity in km/s for the star in the
    data sequence provided in dp. The output is an array with length N,
    corresponding to N exposures. The radial velocity is provided in km/s.
    This is meant to be used to correct (align) the stellar spectra to the same
    reference frame. It requires K (the RV-semi amplitude to be provided in the
    config file, in km/s as well. Often this value is given in discovery papers.
    Like all my routines, this assumes a circular orbit.
    """
    from tayph.vartests import typetest
    import numpy as np
    dp=check_dp(dp)
    p=phase(dp)
    K=paramget('K',dp)
    typetest(K,float,'K in sp.RV_star()')
    rv=K*np.sin(2.0*np.pi*p) * (-1.0)
    return(rv)

def RV(dp,vorb=None,vsys=False):
    """This program calculates the radial velocity in km/s for the planet in the
    data sequence provided in dp, the data-path. dp starts in the root folder,
    i.e. it starts with data/projectname/, and it ends with a slash.

    Example: v=RV('data/Kelt-9/night1/')
    The output is an array with length N, corresponding to N exposures.
    The radial velocity is provided in km/s."""
    import tayph.util as ut
    import numpy as np
    from tayph.vartests import typetest
    dp=ut.check_path(dp)
    p=phase(dp)
    i=paramget('inclination',dp)
    typetest(i,float,'i')
    if vorb == None:
        vorb=v_orb(dp)
    typetest(vorb,float,'vorb in sp.RV')
    rv=vorb*np.sin(2.0*np.pi*p)*np.sin(np.radians(i))

    if vsys == True:
        vs=paramget('vsys',dp)
        rv+=vs
    return rv#In km/s.


def dRV(dp):
    """This program calculates the change in radial velocity in km/s for the
    planet in the data sequence provided in dp, the data-path. dp starts in the
    root folder,i.e. it starts with data/projectname/, and it ends with a slash.

    Example: dv=dRV('data/Kelt-9/night1/')
    The output is an array with length N, corresponding to N exposures.
    The change in radial velocity is calculated using the first derivative of the
    formula for RV, multiplied by the exposure time provided in obs_times.
    The answer is provided in units of km/s change within each exposure."""
    from tayph.vartests import typetest
    import numpy as np
    import astropy.units as u
    from astropy.io import ascii
    import pdb
    import tayph.util as ut
    dp=ut.check_path(dp,exists=True)
    obsp=ut.check_path(dp/'obs_times',exists=True)

    d=ascii.read(obsp,comment="#")
    #Texp=d['exptime'].astype('float')
    Texp=d['col3'].data#astype('float')
    vorb=v_orb(dp)
    p=phase(dp)
    P=paramget('P',dp)
    i=paramget('inclination',dp)
    typetest(P,float,'P in dRV()')
    typetest(i,float,'i in dRV()')

    dRV=vorb*np.cos(2.0*np.pi*p)*2.0*np.pi/((P*u.d).to('s').value)*np.sin(np.radians(i))
    return abs(dRV*Texp)


def texp(dp):
    """Returns the exposure time of the time-series. Useful for calculating the total amount
    of time spent on source."""
    from tayph.vartests import typetest
    import numpy as np
    import astropy.units as u
    from astropy.io import ascii
    import pdb
    import tayph.util as ut
    dp=ut.check_path(dp,exists=True)
    obsp=ut.check_path(dp/'obs_times',exists=True)

    d=ascii.read(obsp,comment="#")
    #Texp=d['exptime'].astype('float')
    Texp=d['col3'].data#astype('float')
    return(Texp)
