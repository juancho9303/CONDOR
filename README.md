# CONDOR (COsmic Noon Dynamics with Optimised Resolutions)

![logo](logo.jpeg)

Introduction
------------

CONDOR is a kinematic fitting code that finds the best kinemmatic model of a rotating disk galaxy by combining its high- and low-resolution velocity fields. The code is optimized for data acquired with facilities that use adaptive optics (AO) suchs as OSIRIS (Keck) and SINFONI (VLT) and their natural seeing counterparts (NS) with KMOS (VLT). 

The code uses a "2.5D" strategy where model datacubes are created. These datacubes match the pixelscale of the input data and are convolved using the right shape for the PSF. Model velocity fields are extracted and rthen compared to the input data through a MCMC resampling method with emcee. The best kinematic model is thus the model that best represents the input data at both resolutions:

![image](https://user-images.githubusercontent.com/14315715/149685240-cfedff11-254e-4f5e-8bd3-71a6df31d28f.png)


Finally, the code calculates the angular momentum content of the galaxy assuming an exponential disk for the surtface brightness profile:

![image](https://user-images.githubusercontent.com/14315715/149685218-671363ff-1dff-4399-b25e-b689519ced35.png)


For more details on what CONDOR does, see: https://academic.oup.com/mnras/article/509/2/2318/6375429?guestAccessKey=a381c4ed-ab7b-4e7a-ba09-f837542541f7


Usage
-----

Some basic settings need to be done in order to run the pipeline.

The basic workflow is as follows:

- Set up a ".txt" or ".dat" file specifying the input parameterts for the geometrical deprojection of the disk (inclination, position angle, size), as well as the kinematic parameters such as the "rflat" and "vflat" parameters which characterize the velocity field of the galaxy following a simple exponential disk:

![image](https://user-images.githubusercontent.com/14315715/149685181-ef4fb779-8143-40e5-a000-60c0b26b9095.png)


The input parameters are merely a first guess that the code will use as priors for the mcmc resampling.

.. _defaults:

Defaults
--------

The default values:

- Thce::

  variable = value
  variable2 = value2
  ...

Th

Examples
--------
Do this and that with ``data_``::

    import matplotlib.pyplot as plt

    peg.save_to_file(peg.name + '.peg')

.. image:: images/example1.png

Plotting a co

.. image:: images/example.png

Further work
------------

Future versions will include:

- Multiple kinematic models.

Acknowledgement
---------------
CONDOR has been written as part of my PhD work at the Centre for Astrophysics and Supercomputing at
Swinburne University of Technology.
