.. _workflow:

***************
Workflow
***************

Tayph assumes a specific (perhaps non-standard) format to represent high-resolution echelle spectra obtained during a time
series, grouped by the spectral orders of the Echelle spectrograph: For all N exposures in a time-series, the section
belonging to order i (out of M orders) is selected and stacked into a single 2D matrix, which is saved as a fits file
This format therefore yields M fits files each measuring N exposures by Npx, the number of spectral pixels in each order,
saved in a named subdirectory of the data/ directory.
These files contrast with the e2ds files or s2d files created by the HARPS and ESPRESSO pipelines, which group all the
orders of each exposure into a 2D matrix instead of grouping all exposures in each order into a 2D matrix. Besides the
orders, their corresponding wavelength axes are also extracted and saved to separate files. This format makes it easy to
treat the data as a time-series, perform time-dependent operations and visualise the time-series easily with nothing but a
FITS viewer.

Pipeline-reduced downloaded from various archives (including ESO and TNG-HARPS-N) can be reformatted automatically by Tayph. During this conversion process,
Tayph also extracts necessary header information that describes the times of observation, airmass, barycentric velocity and the
exposure time, which are saved in an ASCII table named obs_times along with the above FITS files. Optionally, the user may pass these spectra
through a wavelength correction module, and through Molecfit to correct for telluric absorption lines in each exposure. Molecfit is applied to the 1D resampled
spectra that the HARPS and EPSRESSO pipelines normally produce as well, and the fit telluric models are interpolated onto the
individual 2D spectral orders. As Molecfit is a relatively complex fitting tool, this process can take on the order of hours.
This is without doubt the slowest operation of the entire workflow, but for many applications a precise telluric correction
is important, and it only needs to be done for a dataset once. The fitted telluric models are saved in the same place as the
FITS files of the data.

To run Tayph, the user needs to provide two files that further describe the dataset, as well as the particulars of the run.
These are a config.dat file that is saved in the data/dataset_name/ folder along with the obs_times file, the telluric models
and the FITS files. A similar parameter file called a run-file (which may be located anywhere), is used to switch between
datasets, templates, models and other run options.

After starting a run, Tayph proceeds to read the parameters in the run file and the configuration file associated with the chosen
dataset. It will read the spectral orders, optionally convert air wavelengths to vaccuum wavelengths (models and tempaltes
are  at all times assumed to be in vaccuum), and optionally apply the telluric correction. The spectral orders, corresponding
errors and wavelength axes are each saved in lists. The list of orders is therefore a list of length M of 2D blocks each measuring
N x Npx_i. The code checks that all orders have the same number of exposures (N), though the number of pixels (Npx) is allowed
to vary between orders. The error on each pixel is computed from the flux values by default, as the square root of the flux.
This is standard practise for HARPS data (the HARPS DRS does not provide error estimates), but other instruments (including
ESPRESSO) may do error computation on their own. In this case, the errors are provided to Tayph using sigma_i FITS files, of
the same shape, numbering and in the same location as the FITS files containing the orders.

The rest-frames of the spectra are corrected for the motion of Earth around the barycenter (this is purposefully delayed until after
telluric correction), and the orbit of the star around the center of mass between it and the planet (which can be over 100
m/s for hot Jupiters), putting the spectra in the rest frame of the target star, up to a constant shift which is the systemic
velocity.

Next, outlying spectral values are vetted in a process termed masking. The mask is composed of a list with the same
shape as the list of orders. This list is populated with NaN values to indicate outliers and bad regions, in two steps:
First, a running-MAD algorithm identifies values that are more than 5 sigma away from the mean in a block of 40 px wide
(i.e. a block of 40 x N_exp). Secondly, a matplotlib GUI interface allows the user to visually inspect the spectral orders
and manually identify columns that are considered 'bad', for example regions at the edges of orders where the SNR is low,
or at the location of strong telluric lines that were not perfectly corrected. Here, the user can also mask out entire orders,
if needed (instead of having to delete the FITS files of the unwanted orders).

The resulting masks (the ones made manually and automatically) are saved to pickle files in the folder containing the dataset.
This is done so that they can be reloaded instantly in subsequent runs, without the need to recompute or manually redefine.
Forcing the masks to be remade is done using the make_mask keyword in the run configuration file.

Then, if the apply_mask keyword is set in the run configuration file, Tayph continues to apply the manually and automatically
constructed masks to the data. Spectral columns that are entirely masked (as those identified by the user in manual mode) are
set to NaN entirely. The cross-correlation function later will treat those regions as if those spectral channels were never part
of the spectrographs wavelength coverage. Isolated masked pixels (i.e. those that occur independently from one exposure to the
other) are interpolated over. This is done because a NaN value affects the evaluation of the cross-correlation (through the
integral of the template in the denominator), and if this changes from one exposure to the next, a time-dependence would be
artificially introduced to the cross-correlation function. In regions where there are many masked values close together, Tayph
sets any spectral column with too many masked values (set to 20% of N_exp) to NaN entirely.

After masking Tayph optionally performs what is termed 'colour correction'. High-resolution spectrographs may be very stable
in terms of their line-spread function and therefore the relative line depth; but broad-band variations of the contiuum may
occur as the result of atmospheric dispersion, weather or instrumental effects. This causes the shape of the envelope of the
spectrum to change, and because the cross-correlation is implemented as weighing by the flux of the host star, this althers
the weights assigned to different spectral channels from one exposure to the next, again making the cross-correlation
time-dependent. This is mitigated by calculating the mean spectrum (i.e. the mean broadband shape), and dividing each of the
spectra by this mean. If the line-profiles are exactly constant, this creates residuals that describe the colour variation
from one spectrum to the next. A polynomial with degree 3 is fit to these (being careful of outliers) and divided out of each
spectrum in the time-series.

After this step, the data is ready for cross-correlation and Tayph starts to load the desired templates and models. The user may
supply a list of templates and models in the configuration file to execute in sequence; which is often the case if many atoms
or molecules are sought for, and/or multiple models need to be compared to. The templates and models are to be provided as
2-row FITS images, with wavelength on the first row and corresponding flux on the second row. These are identified by Tayph via
model library files in the models/ subdirectory; and paired with short, easy to remember labels. For example, a user may wish to
use a model spectrum of neutral iron, computed in chemical equilibrium at solar metallicity and 2,000 K as a template. This
template is saved in a FITS file somewhere in the models/ directory, or elsewhere on the system; and is indexed in a 2-column ascii
table in the models/ directory that lists the model's label and its filepath, which in this case could be:
FeI_2000   /User/some/filepath/model_spectra/FeI_2000_1_chem_eq.fits. The library files to use for the templates and the models
are provided to Tayph via the configuration file of the run.

The templates can be provided as spectra, meaning that there may be a spectral continuum. Upon reading the desired templates,
Tayph performs a continuum subtraction by measuring the top-envelope of the model spectrum, and subtracting this off.
Subtraction is warranted because the unit of the spectrum is not preserved, as the cross-correlation is normalised by the integral
of the template. To avoid slight inaccuracies and numerical errors, any values smaller (in an absolute sense) than 1/10,000th of
the deepest spectral line are set to zero. In practise, model spectra of transiting planets will have lines with depths up to
a percent of the stellar flux, or less; meaning that all lines deeper than 1e-6 to 1e-7 are preserved, with a completely flat
continuum in-between.

For the purpose of model comparison, models are injected into the data via multiplication. The injected and non-injected
data result in injected and non-injected cross-correlation functions, which can be directly compared to see the effect of a
known transmission spectrum on the cross-correlation signal. Model spectra are read in the same way as templates, and are
multiplied into the data without any pre-treatment. This means that the model is assumed to already be a transit transmission
spectrum, with values that are typically slightly below 1.0. This multiplication step is done before the colour correction
step, so that the continuum of the injected model is not preserved, as is the case in reality.

Cross-correlation is then performed using all the model-injected and non-injected lists of spectral orders. The velocity-range
and step-size of the cross-correlation can be set in the configuration file of the run. Typically, the user will wish to
correlate over a range of hundreds of km/s (as to include a large range of baseline at velocities away from the planetary
signal), with steps of 1 or a few km/s. Although the computation time of the cross-correlation increases linearly with the
number of velocity steps (Nrv), the computation time of the cross-correlation function is typically smaller than the time taken
in the pre-processing steps, so large velocity ranges and small step-sizes are feasible.

The cross-correlation routine results in a single cross-correlation for each template, plus each combination of template
and model-injection. Each cross-correlation measures N rows by Nrv columns, and is saved in the output/ folder. This
cross-correlation function will supposedly contain the planetary signal, but in the vast majority of cases, this signal is
not strong enough to be seen by eye, so additional processing and analysis is needed to isolate it; and to quantify a
physical interpretation. Tayph having brought the user from pipeline-reduced data to cross-correlation signals, it is the
astronomer's job to take care of this interpretation. Tayph does contain a number of tools to aide in the handling and
interpretation of cross-correlation functions, but users are encouraged to use their own analysis routines appropriate for
their individual needs and analysis strategies.
