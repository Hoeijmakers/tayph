.. _getting_started:

**********************
Demonstration tutorial
**********************


Along with the core package of Tayph , a package with demo data is made available containing a part of the HARPS data
that was first used to find iron in the transmission spectrum of the UHJ KELT-9 b. This package also contains the necessary
configuration files and templates to obtain cross-correlations reminiscent of Hoeijmakers et al. (2018), but without the
application of telluric correction.

To get started, download Tayph, open a terminal in the root folder and run::

    python3 setup.py install.

This installs Tayph as a python package on your system.

Create an empty directory somewhere on your system. This will be the working directory that contains your dataset(s), models,
templates, and Tayph output. For the purpose of this walk-through, we will assume that folder is located at
/Users/tayph/xcor_project/

Open a python interpreter and call::

    import tayph.run
    tayph.run.make_project_folder(p)

where you have set p to a string describing the filepath of the empty directory you just created. This creates the necessary folder structure Tayph uses.

Download the dummy data, located here [URL]. You may also download any other pipeline-reduced HARPS dataset from the ESO archive.
A pipeline-reduced dataset will consist of a number of files for each exposure, i.e. e2ds_A/B, s1d_A/B, blaze_A/B files, etc.
Tayph does *not* work on raw echelle data. For the purpose of this walk-through, we will assume that this folder is located at
/Users/tayph/downloads/HARPS_data/. Take care to download and process only *one* transit at a time. Do not put the observations
of multiple transits in the same download folder, because Tayph will treat them as a single time-series. Although certain
use-cases may exist where this is desirable, it is non-standard from the point of view of present literature.

Open a python interpreter in your working directory (i.e. /Users/tayph/xcor_project/) and call::

    import Tayph.read
    tayph.read.read_e2ds('/Users/tayph/downloads/HARPS_data/','KELT-9/night1','',nowave=True,molecfit=False,mode='HARPS',ignore_exp=[])

to convert the pipeline-reduced data to the format used by Tayph, and place it in the data folder in your working directory.
The input parameters mean the following:
The first parameter is the location of your downloaded data.
The second is the name of your dataset, as a folder name. Typically, this takes the form of system_name_b, or system_name_b/night_n
if multiple transits of the same system are available.
The third is the location of your configuration file (see below). This is only needed in case Molecfit is used (because the
configuration file points Tayph to where Molecfit is installed). This is left blank for now because we are not planning to
use Molecfit at the moment.
The nowave keyword indicates whether a wavelength file is present in the downloaded data. This would be the case if you had
run the HARPS pipeline yourself, but the wavelength file is typically not present for data downloaded from the archive.
We therefore set this keyword to True, telling Tayph to take the wavelength solution from the FITS headers instead.
Molecfit is set to False because we ignore it for the time being. For most metals in the optical, telluric correction is not
crucially important, at least in first instance.
The mode keyword can be used to switch between HARPS, ESPRESSO, UVES-red and UVES-blue modes. In this case, we are dealing
with HARPS data.
ignore_exp can be set to a list of integers, which allows you to ignore exposures in the time series. This can be useful if you
find out, after reading in the data that some exposures are bad for whatever reason; and you want to ignore these from the
analysis without going around deleting files. Of course you'd want to do this inspection before running Molecfit (if you were
going to), because Molecfit will need to be run again after ignore_exp is changed, and typically takes a long time.

This has produced a new folder /Users/tayph/xcor_project/data/KELT-9/night1/, in which the various files are located. This has
not created a configuration file, which we will typically need to make ourselves, but one is provided in the prepackaged template.
The configuration file is a 2-column tab-separated table with keywords in the first column and corresponding values in the second
column. The configuration file for this dataset may look like this::

    P			1.4811235
    a			0.03462
    aRstar			3.153
    Rp			1.891
    Mp			2.48
    K       0.275
    RpRstar			0.08228
    vsys    -18
    RA			20:31:26.4
    DEC			+39:56:20
    Tc			2457095.68572
    duration		235.0
    resolution		110000.0
    inclination		86.79
    vsini       111.0
    air         True

which describe the orbital period in days, the semi-major axis in AU, the mass/radius of the planet relative to Jupiter, the radial
velocity semi-amplitude of the star in km/s, the radius-ratio of the planet and star, the systemic velocity in km/s, the RA and DEC
coordinates, the transit center time, the spectral resolution of the instrument, the transit duration in minutes, the orbital
inclination in degrees (close to 90 if the planet is transiting), the projected equatorial rotation velocity of the stellar disk,
whether or not the wavelength solution is in air.

After the data is reformatted and a configuration file is created, we need to proceed to create a run-file that specifies the
working settings of our cross-correlation run. This file is again a 2-column ASCII table with keywords in the first column
and values in the second. This may look like below. The entries in the second column may be followed by commentary that
explains keywords or choices that are not self-descriptive or that you wish to remember.

    molecfit_input_folder     /Users/username/Molecfit/share/molecfit/spectra/cross_cor/
    molecfit_prog_folder      /Users/username/Molecfit/bin/
    model                     FeI
    template                  FeI,FeII,TiI,TiII
    c_subtract                True    Set to True if your templates are not already continuum-subtracted.
    do_telluric_correction    True
    do_colour_correction      True
    do_xcor                   True    Set this to True if you want the CCF to be recomputed. Set to False if you have already computed the CCF in a previous run, and now you just want to alter some plotting, cleaning or doppler shadow parameters. CCFs need to be rerun when masking, orbital parameters, velocity corrections, injected models or telluric corrections are altered.
    inject_model              False
    plot_xcor                 True
    make_mask                 False   Don't be enthusiastic in making a mask. Once you change things like BERVs and airtovac corrections, the mask wont be valid anymore. Make 100% sure that these are correct first.
    apply_mask                True
    do_berv_correction        True
    do_keplerian_correction   True
    make_doppler_model        True     Make a new doppler model (True) / use the previously generated one (False). If multiple templates are provided, the GUI to make a model will only be called on the first template. Make sure that is a template with strong stellar lines, i.e. FeI or FeII.
    skip_doppler_model        False    This is skipping the application of the doppler model altogether.
    RVrange                   1000.0    Extent of the CCF velocity excursion. Linearly increases computation time.
    drv                       2.0       Cross-correlation step size in km/s.
    f_w                       60.0      Cross-correlation filter width in km/s. Set to zero to disable hipass filter.
    shadowname                shadow_FeII     This is the name of the file containing the doppler model shadow that is to be made or loaded. This file is located in the data folder, along with the spectral orders, telluric correction files, etc.
    maskname                  generic_mask    Same, for the mask.



This file is typically saved in the working directory (i.e. /Users/tayph/xcor_project/), and is the primer for initializing
a cross-correlation run.

The required model/template library file, as well as the models themselves are prepackaged along with
the dummy data. Place these in the models subfolder of the working directory.
