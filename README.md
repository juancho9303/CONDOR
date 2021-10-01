# CONDOR (COsmic Noon Dynamics with Optimised Resolutions)

![logo](logo.jpeg)

Introduction
------------

CONDOR does this and that.

For more on CONDOR see: https://arxiv.org/abs/2109.10614


Usage
-----

Us.

The basic workflow is as follows:

- Creme.

Allry.

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

    from pypegase import *
    import matplotlib.pyplot as plt

    PEGASE.pegase_dir = '/home/me/PEGASE.2/' # unless PEGASE_HOME is set

    peg = PEGASE('mydata') # default IMF, Scenario, etc
    peg.generate() # some minutes may be required, console will show progress
    colors = peg.colors()
    colors['B-V'][-1] # B-V color at last timestep (20 Gyr) = .922
    plt.plot(colors['time'], colors['B-V']) # plot times versus B-V color
    plt.legend(loc = 'lower right', numpoints=1)

    peg.save_to_file(peg.name + '.peg')

.. image:: images/example1.png

Plotting a co

.. image:: images/example.png

Further work
------------

Future versions will include:

- PaI

Acknowledgement
---------------
CONDOR has been written as part of my PhD work at the Centre for Astrophysics and Supercomputing at
Swinburne University of Technology.
