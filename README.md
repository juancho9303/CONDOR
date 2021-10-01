# CONDOR (COsmic Noon Dynamics with Optimised Resolutions)

![logo](https://user-images.githubusercontent.com/14315715/135545635-6c22fbbe-3ee5-4201-a8bc-9c0d4e562978.jpeg)

Introduction
------------

CONDOR provides a convenient way to generate galaxy models using the `PEGASE <http://www2.iap.fr/users/fioc/PEGASE.html>`_ version 2 code. I.

Ped ``SSPs``,
``scenarios``, ``spectra``, ``calib`` and ``colors``.

For more on CONDOR see: https://arxiv.org/abs/2109.10614

Installation and configuration (not available)
------------------------------


Usage
-----

Us.

The basic workflow is as follows:

- Creme.

All of this can be done in one line, see the examples below.

Note that all the outputs created by PEGASE end up in the PEGASE home directory.

.. _defaults:

Defaults
--------

The default values are all as per the PEGASE documentation and code, with the following exceptions:

- The default IMF is Salpeter 1955
- The default SFR is exponentially decreasing instead of instantaneous burst, with the default values for p1 and p2 as per the code (1000 and 1 respectively).

For convenience, defaults can be overridden by creating a file ``.pypegase`` in your home directory. It
should have the format::

  variable = value
  variable2 = value2
  ...

To see the built-in defaults, execute ``PEGASE.list_defaults()``. You can use this string as a template. To get started, try::

  python -c "import pypegase as pp; pp.PEGASE.list_defaults()" > ~/.pypegase

This will create an file for you that you can edit should you wish to override some of the defaults. (Note: If pypegase.py is not in your system's library dir, you will need it in the current directory or you can add the location of pypegase.py to to ``PYTHONPATH`` environment variable.)

Examples
--------
Generate a set of data using the defaults, in files prefixed with ``mydata_``::

    from pypegase import *
    import matplotlib.pyplot as plt

    PEGASE.pegase_dir = '/home/me/PEGASE.2/' # unless PEGASE_HOME is set

    peg = PEGASE('mydata') # default IMF, Scenario, etc
    peg.generate() # some minutes may be required, console will show progress
    colors = peg.colors()
    colors['B-V'][-1] # B-V color at last timestep (20 Gyr) = .922
    plt.plot(colors['time'], colors['B-V']) # plot times versus B-V color
    plt.plot(colors['time'], colors['B-V'], "b-", label="B-V") # plot times versus B-V color
    plt.plot(colors['time'], colors['g-r'], "g-", label="g-r")
    plt.legend(loc = 'lower right', numpoints=1)

    peg.save_to_file(peg.name + '.peg')

    peg2 = PEGASE.from_file(peg.name + '.peg')
    peg2.colors() # Same as above

.. image:: images/example1.png

Plotting a continuum spectrum at t=13000 Myr::

  peg = ... # one I made earlier with defaults
  spectra = peg.spectra(time_lower=13000, time_upper=13000)
  lambdas = []
  vals = []
  filters = (4010, 7010) # lower, upper wavelengths (roughly V)
  for col in spectra.colnames:
      try:
          l = float(col)
          if l > filters[0] and l < filters[1]:
              lambdas.append(l)
              vals.append(spectra[col][0])
      except ValueError:
          pass # Column is not a number (i.e. wavelength)
  plot(lambdas[:149], vals[:149], "b-") # Removed the lines for this example
  xlabel("wavelength (Angstroms)")
  ylabel("flux")

.. image:: images/example_spectra.png

Specifying parameters explicitly (these are all the default values and any can be omitted)::

    peg = PEGASE("custom", ssps=SSP(
        IMF(IMF.IMF_Salpeter), ejecta=SNII_ejecta.MODEL_B, winds=True
    ), scenario=Scenario(
        binaries_fraction=0.04, metallicity_ism_0=0, infall=False, sfr=SFR(SFR.EXPONENTIAL_DECREASE, p1=1e3, p2=1),
        metallicity_evolution=True, substellar_fraction=0, winds=False, neb_emission=True,
        extinction=Extinction.NO_EXTINCTION
    ))
    peg.generate()
    spec = peg.spectra(cols=['time', 'l_bol', '7135.00'])
    spec['l_bol'][20] # == 2.499E34

Experimenting with IMFs::

    # Built-in
    imf = IMF(IMF.IMF_Scalo86)
    imf = IMF(IMF.IMF_MillerScalo) # and so on for built-in IMFs

    # Custom
    imf = IMF(IMF.CUSTOM, lower_mass=0.1, upper_mass=120, gamma=-1.35) # A custom IMF equivalent to Salpeter with
                                                                       # default cutoffs
    imf = IMF(IMF.CUSTOM, lower_mass=0.1, upper_mass=120, powers=[
        (0.1, -0.4),
        (1., -1.5),
        (10., -2.3)
    ]) # A custom IMF equivalent to Miller-Scalo (see IMF_MillerScalo.dat)

    peg = PEGASE("custom_imf", ssps=SSP(imf))

Generating a series of models with varying parameters::

    pegase = PEGASE('test')
    for gamma in np.arange(-1.7, -1.0, 0.05):
        # Reuse the same instance each iteration
        pegase.name = "imftest_" + str(gamma)
        pegase.ssps.imf = IMF(IMF.CUSTOM, gamma=gamma)
        pegase.generate()
        pegase.save_to_file(pegase.name + ".peg")

Plotting the results::

    pegs = PEGASE.from_dir(".")
    # Now we have a list of PEGASE instances

    for i, peg in enumerate(pegs):
        colors = peg.colors(cols=['B-V']) # Note 'time' included by default
        plt.plot(colors['time'], colors['B-V'])

    plt.suptitle(r'Color vs time for varying IMF gradient')
    plt.xlabel('time (Myr)')
    plt.ylabel('color (B-V)')
    plt.show()

.. image:: images/example.png

Further work
------------

Future versions will include:

- Passing a redshift value into spectra and colors
- Custom filters
- Calculating colors directly from spectra information
- Better ability to handle customised installations of PEGASE, in particular an altered IMF list/timesteps
- Ability to instantiate from a dictionary, JSON and other formats
- Ability to access color and spectra data by time as well as row number
- More intuitive use of multiple scenarios
- Ability to specify your own defaults
- Simplified process for wrapping an existing run (will examine the file system and reverse-engineer the parameters)
- Human-readable pickled (saved) files
- Unit tests for a greater variety of scenarios
- A more robust ``copy()`` implementation
- Console GUI

Acknowledgement
---------------
PyPegase was written with the support of the Centre for Astrophysics and Supercomputing at
Swinburne University of Technology.
