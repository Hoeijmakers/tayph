.. _getting_started:

**************************
Getting started with Tayph
**************************

This page provides a tutorial for how to get started with Tayph from scratch, covering installation
and the application of Tayph's standard workflow to a set of demo data. When finishing this tutorial,
you should have enough information to use Tayph for the analysis of standard high-resolution datasets.

Requirements
############

Tayph is written in python 3, and requires multiple packages to function. All of these can be installed
using standard package managers (e.g. pip or Anaconda):

- Python version 3.x
- Matplotlib
- Numpy
- Scipy
- Astropy
- lmfit
- joblib

Installation
############


To get started, download or clone Tayph from the `GitHub page <https://github.com/Hoeijmakers/tayph>`_,
open a terminal in the root folder and run::

    python3 setup.py install

This installs Tayph as a python package on your system. Verify that Tayph can be imported by opening a
python interpreter and importing Tayph::

  import tayph

If no errors are raised, Tayph has been successfully installed on your system.


A note on terms used
####################

High resolution spectroscopy is often made more challenging by the fact that this small community uses
their own lingo. This is true for Tayph as well. The following terms are used frequently within Tayph,
and it helps to have these clear as you go through the rest of this tutorial:

- Data: A time-series of high-resolution spectra, pipeline reduced, obtained by e.g. HARPS or ESPRESSO.
- Template: A spectrum, often numerically modelled, with which the data is cross-correlated, acting as a weight function for spectral lines in the data.
- Model: A numerical model for the emergent spectrum of the star (e.g. from PHOENIX), the exoplanet (transmission or emission) or the Earths atmosphere (telluric). A model is used to compare the data to. A cross-correlation template may therefore be a "model spectrum", but a model is not necessarily used as a cross-correlation template.
- e2ds/s2d: Naming of the primary pipeline dataproduct used by Tayph, containing 2-dimensional extracted echelle orders that Tayph reads in via the read_e2ds() function.
- Configuration file (data): A file that holds fundamental parameters regarding the dataset Tayph is reading, and the exoplanet system in question.
- Configuration file (molecfit): A file that tells Tayph where your Molecfit installation and the desired scratch folder are located.
- Runfile: A file that holds parameter settings for the execution of Tayph, instructing which dataset to analyse and in what way.
- Library: An ASCII table that contains filepaths and shorthands for model spectra, to be used as cross correlation templates or models for comparison.
- Mask/masking: A collection of wavelength channels that are to be ignored during the data analysis, due to artefacts or systematic noise.
- Doppler shadow: Residual signature obtained by dividing in-transit spectra by out-of-transit spectra, caused by the passage of the planet in front of the rotating host star. This effect is modelled and removed by Tayph after cross-correlation.
- Binary mask: An ASCII table forming a list of line positions and relative strengths, to be used as a cross-correlation templates but with lines that are as narrow as the spectral sampling rate of the data.
- BERV correction: Shifting of spectra from the rest-frame of the observatory on Earth to the rest-frame of the center of the solar system. The main component is the radial velocity of the Earth towards the target star, caused by the Earth's orbital motion.
- Keplerian correction: Shifting of stellar spectra to the rest-frame of the star by correcting for the small velocity excursion caused by the orbiting planet (on the order of 10's to 100's of m/s).
- Telluric correction: Removal of the Earth's transmission spectrum from the high-resolution data.






Setting up Tayph with demo data
###############################


Tayph works within a folder structure in which your data, models, templates and Tayph output results are
organised. We call this the 'project folder' or 'working directory'. Create an empty directory somewhere on your system.
For the purpose of this walk-through, we will assume that this folder is called
:code:`/Users/tayph/xcor_project/`. Open a python 3 interpreter and call::

    import tayph.run
    p = '/Users/tayph/xcor_project/'
    tayph.run.make_project_folder(p)
    exit()

where you have set :code:`p` to a string describing the filepath of the empty directory you just created.
This creates the necessary folder structure Tayph uses.

Along with the core package of Tayph, a package with demo data is made available containing the
HARPS-N spectra that were first used to find iron in the transmission spectrum of the exoplanet
KELT-9 b. This package also contains the necessary configuration files and templates to obtain
cross-correlations reminiscent of Hoeijmakers et al. (2018), but without the application of
telluric correction (for those see section below).

Download the dummy data, located `here <https://drive.google.com/file/d/1BxO6I7gyJPqnzDrvz--rlxs72wXtoE_4/view?usp=sharing>`__.
You may also download any other pipeline-reduced HARPS
or HARPS-N dataset from the ESO or TNG archives (Tayph does *not* work on
raw echelle data). A pipeline-reduced dataset will consist of a number of files for each exposure,
i.e. e2ds_A/B, s1d_A/B, blaze_A/B files, etc. For the purpose of this walk-through, we will assume
that this folder is located at :code:`/Users/tayph/downloads/demo_data/`. If downloading your own
data, take care to download and process only *one* transit at a time. Do not put the observations
of multiple transits in the same download folder, because Tayph will treat them as a single
time-series.  Although certain use-cases may exist where this is desirable, it is non-standard from
the points of view of present literature and of this demonstration.

To continue with the demo data, move to your working directory in the terminal
(i.e. :code:`cd /Users/tayph/xcor_project/`), open a python 3 interpreter and call::

    import tayph.run as run
    run.read_e2ds('/Users/tayph/downloads/demo_data/kelt-9-spectra','KELT-9/night1',instrument='HARPSN',config=True)

This converts the pipeline-reduced data to the format used by Tayph, places it in the data
folder in your working directory, and executes a preliminary cross-correlation to measure the
radial velocities of the stitched, wide-band (1D) spectra and the indivual echelle orders (2D).
This preliminary cross-correlation is quite time consuming, but it helps to judge whether the
data is read properly and how to set Tayph to correctly deal with velocity corrections and the
wavelength solution.

Read_e2ds is meant to provide you with a quick gateway to handling pipeline reduced echelle spectra,
and is designed to work for multiple different spectrographs out of the box; explicitly to try to
protect you against confusion regarding wavelength solutions and velocity corrections
(where different pipelines have different conventions). The input parameters of read_e2ds are
structured in the following way::

    tayph.run.read_e2ds('input_folder','output_name',instrument='HARPSN',config=True)

- :code:`'input_folder'`: The first parameter is the location of your downloaded data. This is typically a dedicated folder in your project or even your downloads folder.
- :code:`'output_name'`: The second is the name of your dataset, as a folder name. Typically, this takes the form of system_name_b, or system_name_b/night_n if multiple transits of the same system are available.
- :code:`mode='HARPS'`:The mode keyword can be used to switch between HARPS, HARPSN (or HAPRS-N), ESPRESSO, UVES-red,UVES-blue, CARMENES-VIS and CARMENES-NIR modes. In this case, we are dealing with HARPS-N data.
- :code:`config=True`: If set, Tayph will create an empty configuration file with some values filled in, depending on the instrument mode.


Read_e2ds has produced a new folder :code:`/Users/tayph/xcor_project/data/KELT-9/night1/` in
which the various files are located, including a dummy configuration file called
:code:`config_empty`. The user would now need to proceed by filling in this configuration
and renaming it from :code:`config_empty`: to :code:`config`:. However, a finished configuration
file has been provided along with the prepackaged demo data (in
:code:`/Users/tayph/downloads/demo_data/configuration_files/config`), so for the purpose of this
tutorial, you should proceed by copying this file to the data folder instead.



After this, you can run a function called Measure_RV to check on the output of your data, notably the
radial velocities. This function will do a quick cleaning and cross-correlation of your data with
a PHOENIX stellar atmosphere model and a telluric model, and show you how the stellar radial velocity
varies during your transit. Under normal circumstances, if you are working with a stabilised fiber-fed
spectrograph (as is HARPS-N), these drifts will correspond (at most) to the changing barycentric velocity
correction and the Keplerian motion induced by the orbiting planet. Both these effects are taken care
of by Tayph in a later stage. Additional drifts may occur if you are working with a non-stabilised or a
slit-spectrograph. In these cases, measure_RV will tell you whether to perform corrections of the
wavelength solution. Finally, measure_RV also provides you with plots of the airmass and an estimate
of the peak SNR per spectral order. To run this function, call::

    run.measure_RV('input_folder',star='hot',ignore_s1d=False,parallel=False)

- :code:`'input_folder'`: The path to the data just created by read_e2ds, e.g. 'data/KELT-9/night1'.
- :code:`star='hot'`: A PHOENIX model will used will either match that of the sun (code:`star='solar'`), that of a 9000K A-star (code:`star='hot'`) or a cool 4000K K-dwarf (code:`star='cool'`).
- :code:`ignore_s1d`: By default, meausure_RV will read any s1d spectra created by read_e2ds. If this is set to True, any 1D spectra will be ignored and only results for 2D spectra will be computed and shown.
- :code:`parallel`: Tayph employs an experimental implementation of parallellisation of for-loops (see more examples later). Setting this to True will speed up the computation, but may not work on all systems.

The output of this code is a multi-panel plot showing you the spectra, the correlation functions and fitted centroid velocities.
It depends on your spectrograph what drifts are expected to occur.
However in general, read_e2ds is set up such that most 1D and 2D spectra are returned in the telluric rest-frame, meaning that tellurics are expected to occur at 0km/s, and the star is expected to drift according to the barycentric velocity correction.


The configuration file
**********************

The configuration file is a 2-column tab-separated table with keywords in the first column and
corresponding values in the second column. The configuration file for this dataset may look like
this, describing HARPS-N observations of KELT-9 b::


      P	          1.4811235
      a	          0.03462
      aRstar	    3.153
      Mp	        2.48
      Rp          1.891
      K           0.275
      RpRstar     0.08228
      vsys        -18
      RA          20:31:26.4
      DEC         +39:56:20
      Tc          2457095.68572
      resolution  110000.0
      inclination	86.79
      vsini	      111.0
      long	      -17.8850
      lat         28.7573
      elev        2396.0
      air         True

which describe the orbital period in days, the semi-major axis in AU, the mass/radius of the planet
relative to Jupiter, the radial velocity semi-amplitude of the star in km/s, the radius-ratio of
the planet and star, the systemic velocity in km/s, the RA and DEC coordinates, the transit centre
time, the spectral resolution of the instrument, the orbital inclination in degrees (close to 90 if
the planet is transiting), the projected equatorial rotation velocity of the stellar disc, the
geographical location of the observatory and whether or not the wavelength solution is in air.
When running supported instruments, instrument-specific information will have been filled in
automatically.

.. note::
  When setting the configuration file, the transit duration is derived from the combination of
  transit parameters (a/Rstar, period and the inclination). This duration is used to inject models
  into the data, but also to select which spectra are to be co-added in the rest-frame of the
  planet. The accuracy of these parameters therefore has an effect on how the spectra are treated.



Model and template library files
********************************

After the data is reformatted and a configuration file is created, we need to point Tayph to a set
of model spectra that are going to be used as cross-correlation templates and (optionally) for model
injection-comparison. Models may be located in the :code:`/Users/tayph/xcor_project/models/`
directory, with optional subdirectories for different sets of models. In most use-cases, the user
will have multiple sets of models to choose from, which may or may not be similar in their naming
or content. To be able to access different sets of similar models, Tayph assumes that models are
organised in so-called libraries, which are ASCII tables that act as dictionaries through which the
user can refer to model files saved in subfolders using short-hand names (i.e. labels).

The library files are structured as 2-column ASCII tables in the models/ directory. A library file
called :code:`kelt-9-model-library.dat` is provided along with the demo data, and is as follows::

    FeI_4k     KELT-9/4000K_1_Fe.fits
    FeII_4k    KELT-9/4000K_1_Fe_p.fits
    MgI_4k     KELT-9/4000K_1_Mg.fits
    NaI_4k     KELT-9/4000K_1_Na.fits
    ScII_4k    KELT-9/4000K_1_Sc_p.fits
    CrII_4k    KELT-9/4000K_1_Cr_p.fits
    TiII_4k    KELT-9/4000K_1_Ti_p.fits
    YII_4k     KELT-9/4000K_1_Y_p.fits

Individual models are to be saved as FITS files, which are assumed to be located in subdirectories
starting in the :code:`/Users/tayph/xcor_project/models/` directory. Absolute paths (e.g.
:code:`/Users/tayph/xcor_project/models/KELT-9/4000K_1_Fe.fits`) may also be provided.
Each FITS file is a 2-row FITS image, with wavelength (in nm) on the first row, and flux on the
second row. In the case of transit spectra, this flux will typically be the expected transit radius
of the planet as a function of wavelength. To convert models into cross-correlation templates,
Tayph (optionally) performs a continuum subtraction (controlled by the c_subtract switch below).

.. note::
  For Tayph to correctly work, the template needs to have a continuum of zero, either a priori or
  after application of the continuum subtraction option included in Tayph. In addition, absorption
  lines need to be in the negative direction. Otherwise, built-in routines that deal with the cross-
  correlation functions may mis-interpret the results.

In this example, the FITS files of the cross-correlation templates are to be located in the
:code:`/Users/tayph/xcor_project/models/KELT-9` directory, and an example of a library file and
associated model files are prepackaged along with the dummy data. Create a subfolder
:code:`KELT-9` in the :code:`/Users/tayph/xcor_project/models/` directory, place the
template FITS files from the demo package inside (located in
:code:`/Users/tayph/downloads/demo_data/templates`), and finally place the pre-packaged library
file (:code:`/Users/tayph/downloads/demo_data/configuration_files/KELT-9-model-library`) in the
the :code:`/Users/tayph/xcor_project/models/` directory. The library file and template name/label
are going to be passed to Tayph at runtime, allowing Tayph to find the model template files.

Later, when the user wishes to analyse a dataset of a different planet, a second library file
located at :code:`/Users/tayph/xcor_project/models/WASP-123456-models` may be placed in the
:code:`models/` directory as well, pointing to different (but perhaps similar) models, e.g. as
follows::

  FeI_2k      WASP-123456/2000K_FeI.fits
  FeI_3k      WASP-123456/3000K_FeI.fits
  FeII_3k     WASP-123456/3000K_FeII.fits
  FeI_2k      WASP-123456/2000K_TiI.fits
  FeI_3k      WASP-123456/3000K_TiI.fits
  FeII_3k     WASP-123456/3000K_TiII.fits
  TiO         WASP-123456/2000K_TiO.fits
  H2O         WASP-123456/2000K_H2O.fits

For each run of Tayph, only one model library or template library may be specified, so the user
should organise their library files according to what models and templates they wish to run in
batches.



The run file
************

The final step is to create a run-file that controls the working parameters of our
cross-correlation run. This file is again a 2-column ASCII table with keywords in the first column
and values in the second. This may look like below. The entries in the second column may be
followed by commentary that explains keywords or choices that are not self-descriptive or that you
wish to remember for yourself.::

    datapath                  data/KELT-9/night1  #The path to your test data.
    template_library          models/KELT-9-model-library.dat   #The path to your library of models to be used as templates.
    model_library             models/KELT-9-model-library.dat   #The path to your library of models to be used as injection models.
    model                     FeI_4k                 #A comma-separated list of templates as defined in your library file.
    template                  FeII_4k,FeI_4k  #A comma-separated list of templates as defined in your library file.
    c_subtract                True    #Set to True if your templates are not already continuum-subtracted. True for demo data.
    do_telluric_correction    False   #Molecfit has not been run for the demo data.
    do_colour_correction      True
    do_xcor                   True    #Set this to True if you want the CCF to be recomputed. Set to False if you have already computed the CCF in a previous run, and now you just want to alter some plotting, cleaning or doppler shadow parameters. CCFs need to be rerun when masking, orbital parameters, velocity corrections, injected models or telluric corrections are altered.
    inject_model              False
    plot_xcor                 True
    make_mask                 False   #Don't be enthusiastic in making a mask. Once you change things like BERVs and airtovac corrections, the mask wont be valid anymore. Make 100% sure that these are correct first.
    apply_mask                False
    do_berv_correction        True
    do_keplerian_correction   True
    transit                   True    #Differentiate between in-and-out-of-transit exposures when removing time-average spectra.
    make_doppler_model        False   #Make a new doppler model (True) / use the previously generated one (False). If multiple templates are provided, the GUI to make a model will only be called on the first template. Make sure that is a template with strong stellar lines, i.e. FeI or FeII.
    skip_doppler_model        True    #This is skipping the application of the doppler model altogether.
    RVrange                   300.0   #Extent of the CCF velocity excursion. Linearly increases computation time.
    drv                       1.0     #Cross-correlation step size in km/s.
    f_w                       0.0     #Cross-correlation filter width in km/s. Set to zero to disable hipass filter.
    shadowname                shadow_FeII     #This is the name of the file containing the doppler model shadow that is to be made or loaded. This file is located in the data folder, along with the spectral orders, telluric correction files, etc.
    maskname                  generic_mask    #Same, for the mask.



This file is typically saved in the working directory, although it can be placed anywhere in your
system (make sure to adjust the paths correspondingly!). The demo package contains a pre-made run file for the KELT-9 dummy data, located at
:code:`/Users/tayph/downloads/demo_data/configuration_files/demorun.dat`). Place it into your
working directory, and from the working directory, initialise a cross-correlation run by calling::

    import tayph.run
    tayph.run.start_run('demorun.dat')

This initialises the processing cascade of Tayph. Cross-correlation output is saved in the
output directory :code:`/Users/tayph/xcor_project/output/KELT-9/night1/`, with a subfolder
for each template library (a dataset can be cross-correlated with templates of different
libraries), in which there are subfolders for each template. The CCF data is stored in separate
FITS files, with the main output being :code:`ccf_cleaned.fits`. If this file was successfully
generated, you should see a slanted dark streak in the CCFs of Fe I and Fe II, which is the
signature of these atoms in the atmosphere of KELT-9 b.

Congratulations! You have now successfully installed and executed Tayph!

.. note::
    By default, various repetitive routines are processed in parallel using the `joblib` package.
    If your architecture does not support parallel execution, you can switch off the importing and
    usage of this package by running tayph via  :code:`tayph.run.start_run('demorun.dat',parallel=False)`.
    In addition, if you are running Tayph on a machine with sufficient RAM, you can run
    multiple templates in parallel by calling Tayph as
    :code:`tayph.run.start_run('demorun.dat',xcor_parallel=True)`, to gain in
    execution time during cross-correlation. This is only beneficial if the size of the data axis of
    your data times the number of templates times the number of radial velocity steps is (safely) smaller
    than your RAM, typically in the order of dozens to a hundred GB.


Interactive processing
**********************

The functionality of Tayph includes two GUI interfaces. The first allows users to interactively
specify bad regions in their spectral orders. This is activated by setting the make_mask and
apply_mask. parameters in the run file to True. After cross-correlation, a second GUI can be
opened to allow the user to fit the Doppler shadow feature with a single or double-gaussian model.
This is activated by setting make_doppler_model to True and skip_doppler_model to False.
After having been run once, the mask files and doppler model files are saved in the data folder
with names as specified by the shadowname and maskname parameters in the run file.


Using Molecfit for telluric corrections
#######################################

So far we have not used Molecfit in order to correct for telluric lines.
If you wish to integrate Molecfit into Tayph for telluric corrections, these are the necessary
steps that you need to take:

- Install standalone version 1.5.9 of Molecfit on your system.
- Replace some files within Molecfit to make it exectutable, python 3.0 compatible and to fix a line-list error.
- Create a parameter file for your instrument. Parameter files for the supported instruments packaged in the demo data package, but you need to modify these slightly to make Molecfit work on your system.
- Use Tayph create a configuration file for Molecfit, which establishes the interface between the two.


Where to download Molecfit
**************************
Molecfit is developed by ESO and hosted `on the ESO webpages <https://www.eso.org/sci/software/pipelines/skytools/molecfit>`_.
However in 2020, ESO moved to integrate Molecfit into its data reduction environment, deprecating
the standalone execution of Molecfit that is needed for use with non-ESO data, and that Tayph uses.
As of 2021, previous standalone versions are still hosted `on ESO's FTP server <ftp://ftp.eso.org/pub/dfs/pipelines/skytools/molecfit/>`_,
but these may be removed in the future. We have therefore host a copy of Molecfit version 1.5.9
along with the demo data. Importantly, version 1.5.9 is not pyhon 3.0 compatible and it contains an
error in the line-list of water, and so we have updated the relevant files in our repackaged version.


You can find a compressed package of our version of Molecfit `here <https://drive.google.com/file/d/1I-fG2nxx78qDdMNyvEAXg-odpOu4mKKQ/view?usp=sharing>`__.
For the rest of this tutorial, we assume that the package contents have been extracted to a folder
called :code:`molecfit_package`, somewhere on your system.


Installing Molecfit
*******************

The Molecfit package comes with installation instructions written by ESO
(:code:`molecfit_package/install.txt` and the User Manual
:code:`molecfit_package/1.5.9/VLT-MAN-ESO-19550-5772_Molecfit_User_Manual.pdf`).

We highly recommend following the instructions to use the Binary installation (section 3.2 of the
User Manual), which automatically installs local versions of crucial (and sometimes old) third-party
dependencies. For standard Linux distributions, these instructions will suffice.

Installation on OSX (and in particular OSX Catalina) can be slightly more complicated and likely
requires a downgrade of XCode to version 11.7. We have therefore modified ESO's installation
instructions for OSX users as follows.

The installation of the basic molecfit binary package
requires:

- C99 compatible compiler (e.g. gcc or clang).
- glibc 2.11 or newer on Linux or OSX 10.7 or newer.
- common unix utilities (bash, tar, sed, grep).
- XCode Version 11.7. It is likely that you have a higher version of XCode, which means that you will need to download 11.7 from Apple's `download pages <https://developer.apple.com/download/>`__.

The GUI interface is described by ESO as optional, but we highly recommend its usage and Tayph
requires it. ESO's version of Molecfit requires python 2.6 or 2.7 and uses MacPorts to install
dependencies, but we have modified the GUI source code to be 3.x compatible, eliminating the need
for MacPorts. Our requirements for the GUI are therefore as follows:

- wxPython v2.8 or newer.
- Python matplotlib v1.0 or newer.
- Astropy (tested to work on version 4, but older versions may work).

These can be installed with e.g. pip as :code:`pip3 install wxpython matplotlib astropy`.

The command line client also has optional display features which require gnuplot v4.2 patchlevel 3
or newer, but these are not used by Tayph.

We proceed to execute the binary installer. First the downloaded installer needs to be made
executable. To do this, change into the directory :code:`cd molecfit_package/1.5.9/` and run::

  chmod u+x ./molecfit_installer_macosx_x86_64.run

Now the installer can be executed from the same folder with::

  ./molecfit_installer_macosx_x86_64.run

This will ask for an installation directory where it will extract its contents to.
It is recommended to choose an empty directory to avoid overwriting existing files. For the purpose
of this tutorial, we assume that you install Molecfit in the following directory::

  /usr/local/src/Molecfit


After the installer has successfully finished, the Molecfit and Molecfit GUI executables are
installed into :code:`/usr/local/src/Molecfit/bin`. They can be executed by specifying their full or
relative paths, which is what Tayph will do when fitting telluric models to your spectra.


Python 3.x compatibility and fixing an error in the water line-list
*******************************************************************

In order to correct for an error in one line of a GUI python file, as well as making the GUI
executable with python version 3.x, some files within your Molecfit installation have to be changed
manually. The necessary files are provided in the Molecfit package, located at
:code:`molecfit_package/1.5.9/molecfit_replacement.zip`. Extracting this folder reveals the folder
structure of Molecfit located in :code:`/usr/local/src/Molecfit/`.

Backup and replace the python files in :code:`/usr/local/src/Molecfit/share/molecfit/gui/`, and
backup and replace the line-list file in :code:`/usr/local/src/Molecfit/share/molecfit/data/hitran/`
with the files provided in the package.

When replacing the file :code:`SM02GUI_Main.py`, its alias located in
:code:`/usr/local/src/Molecfit/bin/` becomes invalid. Make a new alias to
:code:`/usr/local/src/Molecfit/share/molecfit/gui/SM02GUI_Main.py` (right-click, Make Alias),
rename it to :code:`molecfit_gui` and place it back in :code:`/usr/local/src/Molecfit/bin/`.

Molecfit should now be in working order.


The parameter files
*******************

Molecfit runs are configured using a parameter file, which specifies the input and output spectra,
the characteristics of the observatory and importantly, the FITS header keywords that describe
certain environmental information. When set incorrectly, Molecfit crashes with poorly intelligible
error logging, making it difficult to spot errors in these parameter files. Therefore, we have
pre-packaged Molecfit parameter files for the instruments currently supported by Tayph. These are
the .par files that can be found in your downloaded dummy data package at
:code:`/Users/tayph/downloads/demo_data/configuration_files/`.

Make a new folder called :code:`molecfit` in Tayph's working directory that you created earlier
(located at :code:`/Users/tayph/xcor_project/`). Place the .par files here.

In each of the .par files (e.g. HARPSN.par), the following three lines have to be changed to match
the situation on your system:

- :code:`user_workdir`: A directory used by the GUI to save fitting regions and other settings. We use :code:`user_workdir: /Users/tayph/xcor_project/molecfit/`.
- :code:`filename`: The filename of the fits file that is created during the molecfit run has to be set. This file shall be named after your parameter file for each instrument, and is used by Tayph to write your spectra to. We set it to: :code:`filename: /Users/tayph/xcor_project/molecfit/HARPSN.fits`.
- :code:`output_dir`: The output directory for intermediate molecfit output (located somewhere further down in the .par files). We define it to be the same folder as the input directory: :code:`output_dir: /Users/tayph/xcor_project/molecfit/`.

Repeat these steps for the other instrument parameter files.


Optional: Tayph's Molecfit configuration file
*********************************************

Now we are almost there. For Tayph to be able to find your Molecfit installation, a configuration
file has to be made. This does not have to be done now, because it will be done implicitly during
the first time you run Molecfit using Tayph, and it will be placed somewhere in Tayph's internals.
However, if you are working on a system with multiple users (e.g. a server
environment), each user will have to set their own Molecfit configuration file or they will end up
having to use each other's Molecfit input and output folders, parameter files, etc.

We will create a Molecfit configuration file in the project folder. To do
so, navigate to your project folder (i.e. :code:`cd /Users/tayph/xcor_project/`), open a python 3
interpreter and call::

    import tayph.tellurics as tellurics
    tellurics.set_molecfit_config('/Users/tayph/xcor_project/molecfit/molecfit_config.dat')

You will be asked to enter the following information:

-   **In what folder are parameter files defined and should (intermediate) molecfit output be written to?**
    This is to be the location of your parameter files, i.e. :code:`'/Users/tayph/xcor_project/molecfit/'`.

-   **In what folder is the molecfit binary located?**
    This is the location of your Molecfit installation, i.e. :code:`'/usr/local/src/Molecfit/bin'`

-   **What is your python 3.x alias?**
    The alias with which you open python. This could be :code:`'python'` or :code:`'python3'`.

.. note::
    Users of Catalina and above may not be able to access the GUI environment using their standard
    python 3 alias. Provide :code:`pythonw` in the Molecfit configuration file instead.


You can test that your configuration file is set correctly by calling::

    import tayph.tellurics as tellurics
    tellurics.test_molecfit_config('/Users/tayph/xcor_project/molecfit/molecfit_config.dat')


Calling Molecfit
****************

We are now ready to apply Molecfit to our demo data of KELT-9 b. Continuing the above example
of reading in the data followed by telluric correction, call::

    import tayph.run as run
    run.read_e2ds('/Users/tayph/downloads/demo_data/kelt-9-spectra','KELT-9/night1',instrument='HARPSN',measure_RV=False)
    run.molecfit('KELT-9/night1',instrument='HARPSN',configfile='/Users/tayph/xcor_project/molecfit/molecfit_config.dat')

This will read in the data (which is not necessary if you did so before, but it is shown here for
clarity), and start the the molecfit GUI using the configuration file that you just made.
The spectrum shown is the middle spectrum of your time series, and you will use this spectrum to
choose your fitting regions and parameters. These are then saved to the output directory that was
indicated in the parameter file, and applied to the rest of the time-series. This can take hours or
even a day, depending on how many spectra you have and how fast your system is. Don't worry,
ideally you'll only need to do this once per dataset.


Starting Molecfit in GUI mode requires access to an X-window, while the hours-long fitting
process does not. This may be very inconvenient if you are running Molecfit on a server in the
background. Therefore, calls to Molecfit can be split into a GUI mode and a batch mode. To do
this, call::

    run.molecfit('KELT-9/night1',instrument='HARPSN',mode='GUI')
    run.molecfit('KELT-9/night1',instrument='HARPSN',mode='batch')


In general, the call to Molecfit takes the form::

    run.molecfit(dataname, instrument, mode)

where:

- :code:`dataname` is the name of the dataset that you read in using e2ds, that contains the s1d files that Molecfit needs and that were read in by read_e2ds(). This may also be set to a relative or absolute path (starting with ".", ".." or "/" ).
- :code:`instrument` indicates the instrument you are working with, i.e. :code:`'HARPSN'`, :code:`'HARPS'` or :code:`'ESPRESSO'`.
- :code:`mode` indicates the mode in which Molecfit should be called. The options are :code:`both` (default), :code:`GUI` or :code:`batch`.


.. note::
    The GUI requires screen access, so remember to add -X when logging into an external server.
    Users of Mac OS may need XQuartz to be installed for this to work. The batch process runs
    through without interaction. So if you want to run Tayph on a server, it is recommended to call
    those two tasks separately and execute the batch process in the background, for example
    overnight.



Calling Tayph after Molecfit
****************************

When Molecfit has run through the entire spectral time series in batch mode successfully, the telluric
spectra will have been saved in a pickle file along with the spectral orders in the subdirectory that
contains the associated data, i.e. :code:`data/KELT-9/night1/telluric_transmission_spectra.pkl`. To apply these
models to the 2D spectral orders when running Tayph, simply run Tayph as above, but with the `do_telluric_correction`
keyword in the run file set to `True`. This will interpolate and divide out the telluric transmission model
when the spectral orders are being read in by Tayph, removing the vast majority of telluric absorption features.


.. note::
    You can use the pickle file containing the telluric model spectra to investigate the robustness of
    the working of Molecfit. The file, located at e.g. :code:`data/KELT-9/night1/telluric_transmission_spectra.pkl`
    contains a simple tuple of three arrays representing the model telluric spectra. This file can be read in using the
    pickle module, or with a wrapper in :code:`tayph.tellurics`, called
    as :code:`T=tayph.tellurics.read_telluric_transmission_from_file('data/KELT-9/night1/telluric_transmission_spectra.pkl')`.
    The first element of :code:`T` contains the wavelength axis of the model spectra, the second the model spectra, and the third
    the 1D spectra to which the model was fit to.
