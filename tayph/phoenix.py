import numpy as np
from astropy.utils.data import download_file
from astropy.io import fits

__all__ = ['get_phoenix_model_spectrum', 'phoenix_model_temps',
           'get_phoenix_wavelengths']


phoenix_model_temps = np.array(
    [3200, 6400, 14500, 4100, 12500, 5000, 6700,
     10400, 2700, 11600, 3600, 7000, 7600, 4500,
     10600, 5400, 5700, 3100, 7400, 10000, 4000,
     6000, 4900, 8400, 7200, 2600, 6300, 15000,
     3500, 4400, 6600, 9600, 13500, 5300, 7800,
     3000, 6900, 9400, 3900, 12000, 4800, 5600,
     2500, 9200, 8800, 3400, 5900, 4300, 8000,
     9000, 5200, 6200, 9800, 2900, 13000, 3800,
     6500, 11400, 14000, 4700, 2400, 6800, 8600,
     3300, 4200, 5500, 11800, 5100, 11000, 10200,
     2800, 5800, 3700, 8200, 10800, 4600, 6100,
     2300, 11200]
)


def get_url(T_eff, log_g):
    closest_grid_temperature = phoenix_model_temps[np.argmin(np.abs(phoenix_model_temps - T_eff))]

    url = ('ftp://phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/'
           'PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte{T_eff:05d}-{log_g:1.2f}-0.0.PHOENIX-'
           'ACES-AGSS-COND-2011-HiRes.fits').format(T_eff=closest_grid_temperature,
                                                    log_g=log_g)
    return url


def get_phoenix_model_spectrum(T_eff, log_g=4.5, cache=True):
    """
    Download a PHOENIX model atmosphere spectrum for a star with given
    properties.

    Parameters
    ----------
    T_eff : float
        Effective temperature. The nearest grid-temperature will be selected.
    log_g : float
        This must be a log g included in the grid for the effective temperature
        nearest ``T_eff``.
    cache : bool
        Cache the result to the local astropy cache. Default is `True`.

    Returns
    -------
    spectrum : `~specutils.Spectrum1D`
        Model spectrum
    """
    url = get_url(T_eff=T_eff, log_g=log_g)
    fluxes_path = download_file(url, cache=cache, timeout=30)
    fluxes = fits.getdata(fluxes_path)

    return fluxes


def get_phoenix_wavelengths(cache=True, vacuum=True):
    """
    Download a PHOENIX model atmosphere's wavelength grid

    Parameters
    ----------
    cache : bool
        Cache the result to the local astropy cache. Default is `True`.
    vacuum : bool (optional)
        Return vacuum wavelengths, otherwise air.

    Returns
    -------
    wavelengths : `~np.ndarray`
        Wavelength array grid in vacuum wavelengths
    """
    wavelength_url = ('ftp://phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/'
                      'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
    wavelength_path = download_file(wavelength_url, cache=cache, timeout=30)
    wavelengths_vacuum = fits.getdata(wavelength_path)

    # Wavelengths are provided at vacuum wavelengths. For ground-based
    # observations convert this to wavelengths in air, as described in
    # Husser 2013, Eqns. 8-10:
    sigma_2 = (10**4 / wavelengths_vacuum)**2
    f = (1.0 + 0.05792105/(238.0185 - sigma_2) + 0.00167917 /
         (57.362 - sigma_2))
    wavelengths_air = wavelengths_vacuum / f

    if vacuum:
        return wavelengths_vacuum

    return wavelengths_air
