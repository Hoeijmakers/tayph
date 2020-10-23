.. include:: references.txt



********************************************************************
Cross-correlation analysis of high resolution spectroscopy (`Tayph`)
********************************************************************

Welcome to Tayph
==================

The objective of **Tayph** is to provide a flexible, transparent tool for the analysis of
high-resolution spectroscopic time-series observations of close-in exoplanets, using a
cross-correlation technique similar to what was first implemented by `Snellen et al. (2010) <https://www.nature.com/articles/nature09111>`_, but expanded upon over the course
of multiple papers; Hoeijmakers et al. (`2018 <https://www.nature.com/articles/s41586-018-0401-y>`_, `2019 <https://www.aanda.org/articles/aa/abs/2019/07/aa35089-19/aa35089-19.html>`_, `2020 <https://www.aanda.org/articles/aa/abs/2020/09/aa38365-20/aa38365-20.html>`_).



The software is designed to be applied to transit observations of hot Jupiters made with echelle
spectrographs at optical wavelengths. In particular, it can be applied to pipeline-reduced observations by HARPS,
HARPS-N, ESPRESSO, CARMENES and to a certain extent UVES, with minimal interaction required. The software can
also work on observations made with other instruments, provided that the user provides these according
to a specific format (described below). Another feature is that the software can be used in conjunction with
`Molecfit`_ (`Smette et al., 2015 <https://www.aanda.org/articles/aa/abs/2015/04/aa23932-14/aa23932-14.html>`_) provided that the latter is properly installed.

The software is presented as a Python 3 package, following the `Astropy`_ template. It is named Tayph,
which is Arabic for Spectrum, and carries additional meanings of phantom, shining and glint or glimmer.
This documentation is made using the `Astroplan`_ documentation as a template.


Links
=====

* `Source code <https://github.com/Hoeijmakers/tayph>`_
* `Docs <https://tayph.readthedocs.io/>`_
* `Issues <https://github.com/Hoeijmakers/tayph/issues>`_

License: BSD-3

.. _tayph_docs:

General Documentation
=====================

.. toctree::
   :maxdepth: 2

   installation
   introduction
   workflow
   getting_started
   api

.. _tayph_authors:

Authors
=======

Maintainers
-----------
* Jens Hoeijmakers
* Bibiana Prinoth

Contributors
------------
* Jens Hoeijmakers
* Bibiana Prinoth
* Brett Morris
