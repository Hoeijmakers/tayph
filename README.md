High Resolution Spectroscopy of Exoplanet Atmospheres
-----------------------------------------------------
THIS IS A WORK IN PROGRESS, BUT MOST FUNCTIONS ARE OPERATIONAL. REFER TO THE DOCUMENTATION FOR INSTALLATION AND DEMO INSTRUCTIONS.
----------------------------------------------------------------------------------------------------------------------------------
AS WE REGULARLY PUSH (SMALL) UPDATES AND FIXES, PLEASE PULL THE LATEST VERSION IF YOU RUN INTO AN ERROR, AND FEEL FREE TO REACH OUT
-----------------------------------------------------------------------------------------------------------------------------------
AND INTERACT BY OPENING ISSUES OR BY SUBMITTING PULL REQUESTS.
--------------------------------------------------------------

[![Astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org)
[![ReadTheDocs](https://readthedocs.org/projects/tayph/badge/?version=latest)](https://tayph.readthedocs.io/en/latest/?badge=latest)
![CI Tests](https://github.com/Hoeijmakers/tayph/workflows/CI%20Tests/badge.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)
![ViewCount](https://views.whatilearened.today/views/github/Hoeijmakers/tayph.svg)


Tayph is envisioned as a package to help you analyse high-resolution spectral time-series of exoplanets using the Cross-correlation technique introduced by [Snellen et al. 2010](https://www.nature.com/articles/nature09111) in a *robust*, *transparent*, *flexible* and *user friendly* way.

- **Robust**: The analysis of high-resolution spectral time series involves multiple steps, that may strongly depend on the type of data, the planetary system, and circumstances. The steps applied in the analysis cascade have been developed and tested over the course of a number of published analyses. The code itself is thoroughly tested at the build and execution levels.
- **Transparent**: Tayph aims to be transparent in its capabilities, assumptions and limitations. To this end, the entire package is open-source, and the analysis steps have been developed and described in published literature ([Hoeijmakers et al 2018](https://www.nature.com/articles/s41586-018-0401-y), [2019](https://www.aanda.org/articles/aa/pdf/2019/07/aa35089-19.pdf), [2020](https://www.aanda.org/articles/aa/pdf/2020/09/aa38365-20.pdf)). The code is documented following the Astropy style, and thoroughly commented in-line. Ongoing issues, bugs and improvements are maintained and documented in public on [Tayphs GitHub page](https://github.com/Hoeijmakers/tayph).
- **Flexible**: Tayph is designed to work on spectral time series observed with commonly used High-resolution Echelle spectrographs that observe at optical wavelengths and that have publicly available pipelines or for which pipeline-reduced data is made available in public archives. When applied to pipeline-reduced data of HARPS, HARPS-North, CARMENES and ESPRESSO, Tayph will run out of the box with minimal effort required. For other instruments, Tayph requires the user to format their spectral time-series into a specific format that is human-readable and easy to control. The routines and analysis steps employed by Tayph can be adjusted to allow for the analysis of other wavelength ranges (e.g. the NIR/MIR). The modular nature provided by the python package ecosystem and the transparency of the data structure will allow the user to insert their own modules whenever needed. We are working continuously to add other types of data to the workflow.
- **User-friendly**: Tayph provides meaningful status reports and errors, and important functions test for the integrity of their input variables. This aims to make it easy for the user to solve problems, and to raise issues on [Tayphs GitHub page](https://github.com/Hoeijmakers/tayph) in the case of bugs. In addition, because high-resolution spectrographs can produce large, unwieldy spectra; considerable effort has been directed to providing meaningful plots and GUIs for the user to inspect, understand and interact with their data.

Despite the aims set out above, both the methods used by the community to analyse high-resolution observations of exoplanet atmospheres as well as Tayph are likely never perfect, and subject to approximations, mistakes or bugs. Secondly, the code is focused on transparency and flexibility, and we only recently started to focus on making efficient use of computer resources by vectorising operations and simple CPU-based parallellisation where possible. The entire package is written in Python, and there are probably many instances where factors can be gained in terms of computation and memory efficiency. Making Tayph more nimble is hence an ongoing effort. We hope that you will benefit from this package, and share your questions, feedback, ideas or pull requests. We will strive to use your input to better Tayph.

A paper describing the functionality of Tayph is currently in preparation. If you make use of Tayph in your publication, please consider citing the analysis presented in [Hoeijmakers et al. 2020](https://www.aanda.org/articles/aa/pdf/2020/09/aa38365-20.pdf), which most closely matches the current implementation of Tayph.


Documentation
-------------

For instructions and demo's on how to install and use Tayph, please refer to the official [documentation pages](https://tayph.readthedocs.io/en/latest/).

Main contributors
-----------------

- Jens Hoeijmakers
- Bibiana Prinoth
- Nicholas Borsato
- Brett Morris
- Brian Thorsbro



License
-------

This project is Copyright (c) Jens Hoeijmakers and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the Astropy package template <https://github.com/astropy/package-template>
which is licensed under the BSD 3-clause license. See the licenses folder for
more information.
