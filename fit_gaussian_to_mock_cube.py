##!/usr/bin/env python2
## -*- coding: utf-8 -*-
"""
Created on Mon May 27 08:31:53 2019
@author: atiley, modified by Juan Espejo to use with mock datacubes
"""

from astropy.io import fits
import os
import sys
import scipy
from scipy.stats import sigmaclip
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import pyspeckit
from scipy import ndimage as ndi
import mpfit
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
from pathlib import Path

home = str(Path.home())
import warnings
warnings.filterwarnings("ignore")

def getwl3(cubefile, ext=0, kw1="CRVAL3", kw2="CRPIX3", kw3="CDELT3"):
    '''
    Function to get wavelength array from FITS headers
    '''

    hdu = fits.open(cubefile)

    crval1 = hdu[ext].header[kw1]
    crpix1 = hdu[ext].header[kw2]
    cdelt1 = hdu[ext].header[kw3]

    sz = np.shape(hdu[ext].data)
    wl = crval1 * 10 + cdelt1 * 10 * np.arange(sz[0]) - (crpix1 - 1) * cdelt1 * 10

    return wl


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def convolve_spectrum(wl, spec, FWHM_old=2.3548200450309493, FWHM_new=None, l_K=11845., R_K=3582., regrid=True):
    '''
   Function to convolve a spectrum to a desired spectral FWHM
   '''

    # Get the KROSS FWHM if none is given
    if FWHM_new == None:
        FWHM_new = 4

    # Get the old and new sigmas
    sig_new = FWHM_new / (2. * (2 * np.log(2)) ** 0.5)
    #print('sig_new is=', sig_new)
    sig_old = FWHM_old / (2. * (2 * np.log(2)) ** 0.5)
    #print('sig_old is=', sig_old)

    # Calculate the change required
    dsig = (sig_new ** 2. - sig_old ** 2.) ** 0.5

    # Convert units to pixels
    dsig = dsig / (wl[1] - wl[0])

    # Now convolve the spectrum
    newspec = gaussian_filter(spec, dsig)

    return newspec


def get_prelim_maps(cube, wav, redshift, clip=True):
    '''
    Function to get a median continuum map and summed narrow band Halpha map
    '''

    # Tidy cube
    cube[np.isnan(cube)] = 0.
    if clip == True:
        c, low, high = sigmaclip(cube, low=10, high=30)
        cube[np.logical_or(cube < low, cube > high)] = 0.

    # Find slice with central Halpha
    idx = np.where(wav == find_nearest(wav, 6562.8 * (1 + redshift)))[0][0]

    # Get mask for signal region
    mask = np.zeros(len(cube))
    mask[idx - 7:idx + 8] = 1.

    # Get median continuum map
    cont = np.nanmedian(cube[mask == 0], axis=0)

    # Get narrow band image
    inten = np.nansum(cube[mask == 1], axis=0)

    return cont, inten


def mask_array(array, xcen, ycen, radius):
    '''
    Function that returns a (circular) mask the same size as a 2D array
    '''

    # Make an empty array for mask
    mask = np.zeros_like(array, dtype=bool)

    # Mask pixels within the chosen circle
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            if (x - xcen) ** 2 + (y - ycen) ** 2 <= radius ** 2:
                mask[y][x] = True

    return mask


def get_weights(ksky, power=5.):
    '''
    Function to make a weighting array from a sky spectrum
    '''

    ksky[np.isnan(ksky)] = 0.
    # ksky[ksky<0.]=0.
    ksky = ksky / np.nanmax(ksky)
    ksky = ksky + 1.
    w = 1 / (ksky)

    w = w ** power

    return w


def extract_spectrum(wav, cube, zguess, mask=None, microns=True, sumspec=False):
    '''
    Extracts a summed global spectrum from centre of cube and measures redshift.
    RETURNS: spectrum, basic_redshift
    '''

    # Get sky spectrum (and keep it in Angstroms)
    skywav, skyflux = np.genfromtxt(
        home + '/Dropbox/PhD/Data/OSIRIS/osiris_data_for_processing/skylines/rousselot2000.dat', unpack=True)

    # Identify channels with lots of sky in them
    badsky = np.where(skyflux > 50.)  # 50 an arbitrary choice but in line with Stott
    badskywav = skywav[badsky]

    # Get sky regions
    badwav = np.zeros(np.shape(wav))
    for i in range(len(badwav)):
        if np.sum(np.where(np.abs(wav[i] - badskywav) < np.abs(wav[0] - wav[2]))) > 0:
            badwav[i] = 1

    if mask.any() == None:  # If there's no mask then
        # Get region to extract spectrum from
        ymin = xmin = cube.shape[1] / 3
        ymax = xmax = (2 * cube.shape[1]) / 3
        # Extract median spectrum from centre of cube
        spectra = []  # First get array of spectra in chosen region
        for i in range(ymin, ymax + 1):
            for j in range(xmin, xmax + 1):
                spectra.append(cube[:, i, j])

        if sumspec == True:
            spectra = np.nansum(spectra)
        else:
            spectra = np.mean(spectra, axis=0)

        spectra[np.isnan(spectra)] = 0.
        totspec = spectra.copy()

    else:  # Else get the region from the mask
        coords = np.array(np.where(mask == 1)[:])
        spectra = np.zeros(len(wav))  # First make array for spectra in chosen region
        for i in range(coords.shape[1]):
            spectra = spectra + cube[:, coords[:, i][0], coords[:, i][1]]

        spectra[np.isnan(spectra)] = 0.
        if sumspec == False:
            spectra = spectra / len(mask[mask == 1])
        totspec = spectra.copy()

    # look for Halpha or OIII?

    if zguess < np.max(wav) / 6562.8 - 1.:
        zobs = zguess

    else:
        zobs = zguess

    return totspec, zobs


# 1D gaussian
def gauss(pars, array):
    x = np.array(array)
    a, b, c = pars
    return a * np.exp(-0.5 * (((x - b) ** 2) / (c ** 2)))


def singlegaussfit(xarr, yarr, guess, dy=None, window=None, thin=False, quiet=0):
    # Define figure of merit
    def myfunct(p, fjac=None, x=None, y=None, dy=None):
        # Function to return the weighted deviates
        model = gauss(p, x)
        status = 0
        error = dy
        return ([status, (y - model) / error])

    # Set up priors array
    parinfo = [{'value': 0., 'fixed': 0, 'limited': [0, 0], 'limits': [0., 0.], 'tied': ''} for i in range(3)]

    # Set prior variable limits for fitting
    parinfo[0]['limited'][0] = 1
    parinfo[0]['limited'][1] = 0
    parinfo[0]['limits'][0] = 0
    parinfo[0]['limits'][1] = 0
    parinfo[1]['limited'][0] = 1
    parinfo[1]['limited'][1] = 1
    if window != None:
        parinfo[1]['limits'][0] = window[0]
        parinfo[1]['limits'][1] = window[1]
    else:
        parinfo[1]['limits'][0] = xarr[0]
        parinfo[1]['limits'][1] = xarr[-1]
    if thin == False:
        parinfo[2]['limited'][0] = 1
        parinfo[2]['limited'][1] = 0
        parinfo[2]['limits'][0] = 0
        parinfo[2]['limits'][1] = 0
    else:
        parinfo[2]['limited'][0] = 1
        parinfo[2]['limited'][1] = 1
        parinfo[2]['limits'][0] = 0
        parinfo[2]['limits'][1] = 0.3

    # if quiet == 0:
    #     print('fitting gaussian...')

    # Rest of set up for mpfit
    if dy.any() == None:
        fa = {'x': xarr, 'y': yarr, 'dy': np.std(yarr[np.logical_or(xarr < window[0], xarr > window[1])])}
    else:
        fa = {'x': xarr, 'y': yarr, 'dy': dy}

    # Make first guess for fit
    p0 = guess
    # Do the fit
    m = mpfit.mpfit(myfunct, p0, functkw=fa, parinfo=parinfo, quiet=quiet)

    # Get the normalised errors
    DOF = len(yarr) - len(p0)
    try:
        error = np.sqrt(m.fnorm / DOF) * m.perror
    except:
        error = m.perror

    # if quiet == 0:
    #     print('Single Gaussian fitting complete!')
    return [m.params, error]


def singlekmos(par, x):
    '''
    Single Gaussian function
    '''

    x0, I, sig = par

    Hafit = (I / sig / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (x - x0) ** 2 / sig ** 2)

    return Hafit


def makefit(xarr, yarr, rms, guess, window=None, w=None, limits=None):
    # Define figure of merit
    def myfunct(p, fjac=None, x=None, y=None, dy=None, weights=None):
        # Function to return the weighted deviates
        model = singlekmos(p, x)
        status = 0
        return ([status, weights * (y - model) / dy])

    # Set up priors array
    parinfo = [{'value': 0., 'fixed': 0, 'limited': [0, 0], 'limits': [0., 0.], 'tied': ''} for i in range(3)]

    if limits == None:
        # Set prior variable limits for fitting
        parinfo[0]['limited'][0] = 1
        parinfo[0]['limited'][1] = 1
        parinfo[0]['limits'][0] = window[0]
        parinfo[0]['limits'][1] = window[1]
        parinfo[1]['limited'][0] = 1
        parinfo[1]['limited'][1] = 0
        parinfo[1]['limits'][0] = 0
        parinfo[1]['limits'][1] = 0
        parinfo[2]['limited'][0] = 1
        parinfo[2]['limited'][1] = 0
        parinfo[2]['limits'][0] = 0
        parinfo[2]['limits'][1] = 0
    else:
        # Set prior variable limits for fitting
        parinfo[0]['limited'][0] = limits[0][0][0]
        parinfo[0]['limited'][1] = limits[0][0][1]
        parinfo[0]['limits'][0] = limits[1][0][0]
        parinfo[0]['limits'][1] = limits[1][0][1]
        parinfo[1]['limited'][0] = limits[0][1][0]
        parinfo[1]['limited'][1] = limits[0][1][1]
        parinfo[1]['limits'][0] = limits[1][1][0]
        parinfo[1]['limits'][1] = limits[1][1][1]
        parinfo[2]['limited'][0] = limits[0][2][0]
        parinfo[2]['limited'][1] = limits[0][2][1]
        parinfo[2]['limits'][0] = limits[1][2][0]
        parinfo[2]['limits'][1] = limits[1][2][1]

    # Rest of set up for mpfit
    fa = {'x': xarr, 'y': yarr, 'dy': rms, 'weights': w}

    # Make first guess for fit
    p0 = guess
    # Do the fit
    m = mpfit.mpfit(myfunct, p0, functkw=fa, parinfo=parinfo, quiet=1)
    # Get the normalised errors
    DOF = len(yarr) - len(p0)
    try:
        error = np.sqrt(m.fnorm / DOF) * m.perror
    except:
        error = m.perror

    return [m.params, error]


def fitvelkmos_Ha(cubefile, wav, zmeas, linewl, linewid, area, exten=0, justcont=False, filter='Hn3', norm=1, ksky=None,
                  rmsmap=False, minbin=0, maxbin=1, minsig=False, s2ncut=3., power=5., clipnoise=False, modchi=False,
                  smooth=False, clipval=10, clip=True):
    '''
    Function that uses Python to fit a kmos data cube and produce a hypercube (acube) of all the fit parameters
    '''

    # Get dx wav positions
    halpha = 6562.8
    dx = (linewid * 2.355) * 6  # 2.5 (a shift of 10 Angstroms is equivalent to +/- 500 km/s)
    #print('dx:', dx)

    # first remove any NaN values in the cube
    #print('removing NaN values from cube...')
    cube = cubefile#fits.open(cubefile)[np.int(exten)].data  # load cube
    cube[np.isnan(cube)] = 0.  # replace NaNs with zero

    # make empty acube
    if rmsmap == False:
        acube = np.zeros((9, cube.shape[1], cube.shape[2]))
    else:
        acube = np.zeros((10, cube.shape[1], cube.shape[2]))

    # define signal region
    windowwav = wav[np.logical_and(wav >= ((6548 * (1 + zmeas)) - 30), \
                                   wav <= ((6583 * (1 + zmeas)) + 30))]
    # get edges of window region
    window = [np.min(windowwav), np.max(windowwav)]

    # approx instrumental resolutions in Angstroms  I NEED TO MAKE SURE THIS RESOLUTION IS CORRECT

    dlam_min = (0.5 * (wav[0] + wav[-1]) / 3238.01) / 2.35
    #print('dlam_min is=', dlam_min)

    # normalise the sky
    w = get_weights(ksky.copy(), power=power)
    badsky = np.where(1. / (w ** (1. / power)) > 1.1)
    badwav = np.zeros(len(wav))
    badwav[badsky] = 1
    dlamKROSSmicron = (wav[-1] - wav[0]) / (len(wav) - 1.) * 2.0  # just to check the value that we get.

    # ==========================================================================
    # for every spaxel now find best fit to spectrum
    # ==========================================================================

    #print('fitting cube...')
    for i in range(cube.shape[1]):
    #for i in tqdm(range(cube.shape[1])):
        # print('fitting row ',i)
        for j in range(cube.shape[2]):

            keepgoing = True
            for bini in range(minbin,maxbin):#

                if keepgoing:
                    spec = cube[:, i, j].copy()
                    ##get the spectrum (according to the required bin size)
                    # if bini == 0:
                    #     spec = cube[:, i, j].copy()
                    # else:
                    #     spec = np.zeros(cube.shape[0])
                    #     lowy = np.int(np.max([0, i - bini]))
                    #     highy = np.int(np.min([cube.shape[1], i + bini]))
                    #     lowx = np.int(np.max([0, j - bini]))
                    #     highx = np.int(np.min([cube.shape[2], j + bini]))
                    #     for p in range(lowy, highy):
                    #         for q in range(lowx, highx):
                    #             spec = spec + cube[:, p, q] / (2. * bini + 1.) ** 2

                    spec2 = spec.copy()
                    c, low, high = sigmaclip(spec2[~np.isnan(spec2)], 1.5)
                    spec2[np.logical_or(spec2 < low, spec2 > high)] = 0.
                    spec2[np.isnan(spec2)] = 0.
                    basepars = np.polyfit(wav, spec2, 1)
                    #print(basepars)
                    spec = spec - (basepars[0] * wav + basepars[1])

                    if np.nanmax(spec) == 0.:
                        continue

                    # get rms of spectrum away from sky and signal
                    spec4noise = spec[np.logical_and(badwav == 0, np.logical_or(
                        np.logical_and(wav > window[0] - 5 * dx, wav < window[0]),
                        np.logical_and(wav > window[1], wav < window[1] + 5 * dx)))]
                    # spec4noise=spec[np.logical_and(badwav==0, np.logical_or(np.logical_and(wav>window[0]-5*dx,wav<window[0]),np.logical_and(wav>window[1],wav<window[1]+5*dx)))]
                    spec4noise = spec4noise[~np.isnan(spec4noise)]
                    c, low, high = sigmaclip(spec4noise, 5.)
                    if clipnoise == True:
                        rms = np.std(c)
                    else:
                        rms = np.std(spec4noise)

                    # truncate spectrum to within the signal region for fitting
                    widerange = np.where(np.logical_and(wav > window[0], wav < window[1]))
                    fitrange = np.where(np.logical_and(wav > linewl - (dx), wav < linewl + (dx)))

                    # get straightline chi2
                    av = np.nanmedian(spec4noise)  # baseline level
                    if modchi == True:
                        chi0 = len(w[widerange]) * np.sum(
                            w[widerange] * ((spec[widerange] - av) ** 2) / rms ** 2) / np.sum(w[widerange])
                    else:
                        chi0 = np.sum(((spec[widerange] - av) ** 2) / rms ** 2)

                    # get straightline chi2
                    av = np.nanmedian(spec4noise)  # baseline level

                    # make first guess for fit
                    specnorm = max(0, 1. / np.nanmax(spec[fitrange]))

                    if specnorm == 0:
                        specnorm = 1.
                    p0 = [linewl, area * 10 * specnorm, linewid]# * 0.2]  # These initial guesses are either from the
                    # integrated one or the individual lines
                    # do the fit
                    # pars,error=makefit(wav,spec.copy()*specnorm,rms*specnorm,p0,[wav[fitrange][0],wav[fitrange][-1]],w)
                    pars, error = makefit(wav[widerange], (spec[widerange].copy() - av) * specnorm, rms * specnorm, p0,
                                          [wav[fitrange][0], wav[fitrange][-1]], w[widerange])
                    # print(pars)
                    try:
                        len(error)
                    except:
                        error = np.zeros(len(pars))
                    # reverse normalisation of areas
                    pars[1] = pars[1] / specnorm
                    error[1] = error[1] / specnorm

                    # get s2n

                    # first calculate chi2 for fit
                    if modchi == True:
                        chi2 = len(w[widerange]) * np.sum(w[widerange] * (((spec[widerange] - av) - singlekmos(pars[:3],
                                                                                                               wav[
                                                                                                                   widerange])) ** 2) / rms ** 2) / np.sum(
                            w[widerange])
                    else:
                        chi2 = np.sum(
                            (((spec[widerange] - av) - singlekmos(pars[0:3], wav[widerange])) ** 2) / rms ** 2)

                    # then sqrt of difference of fit and straight line
                    if chi2 <= chi0:
                        s2n = np.sqrt(chi0 - chi2)
                    else:
                        s2n = 0.

                    # save continuum & rms in each case
                    acube[5, i, j] = av
                    if rmsmap != False:
                        acube[9, i, j] = rms

                    #s2n = 2*s2ncut

                    # print(pars)
                    # if the s2n is sufficient then
                    if s2n >= s2ncut and np.abs(3e5*(linewl-pars[0])/linewl) < 500.:# and pars[2] > 1.15*dlam_min:# and s2n < 250.:

                        fitted_line = singlekmos(pars, wav)

                        # fig, ax = plt.subplots()
                        # plot = ax.plot(wav[fitrange],fitted_line[fitrange], color='red', label='Best fit')
                        # ax.step(wav[fitrange],spec[fitrange], label='Spectrum')
                        # ax.axvline(x=16540, linewidth=4, ls='--',color='grey', alpha=0.5, label=r'Reference $H_\alpha$')
                        # ax.set_xlabel(r'$\lambda$')
                        # ax.set_ylabel('Intensity')
                        # ax.legend()
                        # ax.set_title('Pixel=%d, %d' % (i,j))
                        # plt.savefig('./results/Pixel=%d_%d' % (i,j))
                        # plt.close('all')

                        if rmsmap == False:
                            acube[:, i, j] = [pars[0], pars[1], pars[2], bini, s2n, av, error[0], error[1], error[2]]
                        else:
                            acube[:, i, j] = [pars[0], pars[1], pars[2], bini, s2n, av, error[0], error[1], error[2],
                                              rms]
                        keepgoing = False

    return acube


def unwrap_acube(acubefile, zmeas, justcont=False, filt='Hn3', clip=False):
    '''
    Function to get products from the acube of a galaxy. NOTE: zmeas should be same zmeas given to fitting routine
    '''

    acube = acubefile#fits.open(acubefile)[0].data

    #print(acube[0])

    if justcont == True:
        base = acube[5].copy()
    else:
        # extract info from hypercube (acube)
        #print('extracting maps...')
        inten = acube[1].copy()  # Halpha intensity
        intenerr = acube[7].copy()  # Halpha intensity error
        sigmalam = acube[2].copy()  # Halpha sigma
        sigmalamerr = acube[8].copy()  # Halpha sigma error
        lc = acube[0].copy()  # Halpha position (wavelength)
        lcerr = acube[6].copy()  # Halpha position error (wavelength)
        bins = acube[3].copy()  # map of binsize for fitting
        S2N = acube[4].copy()  # S/N map
        base = acube[5].copy()  # continuum level map
        noise = acube[-1].copy()  # rms boise map

        # ================================================================
        # Still need to make velocity maps and sigma maps (in that order)
        # ================================================================

        # get restframe wavelength of Halpha
        l0 = 6562.77 * (1 + zmeas)

        # get velocity map
        # first check there's something there
        if np.nanmax(lc) == 0.:
            print('velocity map extraction failed!')

        # get velocity map from lc
        velmap = (lc - l0) / l0 * 3e5
        # get velocity error map from lcerr
        velmaperr = np.zeros(lcerr.shape)
        velneg = np.where(velmap < 0.)
        velpos = np.where(velmap > 0.)
        velmaperr[velneg] = velmap[velneg] - (np.abs(lc[velneg] - lcerr[velneg]) - l0) / l0 * 3e5
        velmaperr[velpos] = (np.abs(lc[velpos] + lcerr[velpos]) - l0) / l0 * 3e5 - velmap[velpos]

        # clean the velmap
        velmap[velmap == -3e5] = np.nan
        # get rid of rogue pixels that will affect the fitting process (use iterative 3sigma clip)
        c, low, high = sigmaclip(np.abs(velmap[~np.isnan(velmap)]), 3., 3.)
        # c is clipped sample, low is final lower clip limit & high is final upper clip limit

        # remove clipped spaxels
        if clip == True:
            velmap[np.abs(velmap) > high] = np.nan

        if filt == 'Hn3':
            dlamKROSSmicron = 2.  # (wav[-1]-wav[0])/(len(wav)-1.)*2.0
            # print('The resolution for sigma is:', dlamKROSSmicron)#2. #spectral resolution in Angstroms
            # (calculated as (wav[-1]-wav[0])/(len(wav)-1.)*2.0)
        elif filt == 'YJ':
            dlamKROSSmicron = 3.5058590583503246
        elif filt == 'IZ':
            dlamKROSSmicron = 3.02734435535968

        # get KROSS resolution in velocity units
        resolution = l0 / dlamKROSSmicron
        vfwhm = (3e5) / resolution
        instsig = vfwhm / 2.35
        # print('instsig=',instsig)

        sigma = sigmalam / l0 * 3e5  # apply corrections
        sigmaerr = (sigmalamerr) / l0 * 3e5
        # transfer observed sigma map to python
        sigmap = sigma.copy()
        sigmaperr = sigmaerr.copy()

        return inten, velmap, velmaperr


def process_cube(cubefile, wav, redshift, prod_dir, filt='Hn3', maxbin=1, s2ncut=5., power=5, modchi=False,
                 smooth=False, exten=0, pixscale=0.1, clipval=100):
    '''
    function to fit cube Halpha emission and extract analysis products e.g. velocity map, flux map, etc.

    cubefile: path to cube
    wav: wavelength array
    redshift: best guess redshift of object in cube
    prod_dir: where to put the analysis products
    '''

    # load the cube
    cube = cubefile
    c, low, high = sigmaclip(cube[~np.isnan(cube)], low=10, high=30)
    #cube[np.logical_or(cube < low, cube > high)] = 0.

    # get continuum and channel maps
    cont, narrow = get_prelim_maps(cube.copy(), wav.copy(), redshift)

    # get a sky spectrum
    skywav, skyflux = np.genfromtxt(
        home + '/Dropbox/PhD/Data/OSIRIS/osiris_data_for_processing/skylines/rousselot-oh-binned1A.list', unpack=True)
    # interpolate
    f2 = scipy.interpolate.interp1d(x=skywav, y=skyflux)
    # resample on a linear grid
    skywav = np.arange(skywav[0], skywav[-1], skywav[1] - skywav[0])
    skyflux = f2(skywav)
    # convolve model sky to match KROSS
    skyflux = convolve_spectrum(skywav, skyflux, FWHM_old=2.3548200450309493)  # Why is it zero?
    # interpolate again
    f2 = scipy.interpolate.interp1d(x=skywav, y=skyflux)
    # resample at kmos wavelengths
    ksky = f2(wav)
    # get weights
    ksky = ksky / np.max(ksky)
    sky = ksky.copy()

    # get mask
    mask = mask_array(cont.copy(), cont.shape[1] / 2., cont.shape[0] / 2., 1.0 / pixscale)

    # weighting
    w = get_weights(sky.copy(), power=1.)
    # print(w)

    # normalisation
    norm = 1. / np.nanmax(cube[:, cube.shape[1] // 2, cube.shape[2] // 2])  # I included an aditional / to make it work

    # window
    l0 = 6562.77 * (1 + redshift)
    window = [(-1500 * l0 / 3e5) + l0, (1500 * l0 / 3e5) + l0]
    # print('the window is=',window)

    # ====================================================================
    # get a measure of redshift from global spectrum
    # ====================================================================

    # extract global spectrum and basic redshift measurement
    spectrum, zobs = extract_spectrum(wav, cube, redshift, mask=mask.copy())

    # baseline subtract
    pyspec = pyspeckit.Spectrum(data=spectrum.copy(), error=1. / w, xarr=wav)
    pyspec.baseline(xmin=wav[100], xmax=wav[-100], exclude=[window[0], window[1]], subtract=True, reset_selection=True,
                    order=3)
    spectrum = pyspec.flux

    spectrum = ndi.median_filter(spectrum, 3)

    # feed spectrum and redshift in to fitting procedure
    # to get redshift from line fit and parameters of that line fit
    specpars, specerrs = singlegaussfit(wav, spectrum.copy() * norm, guess=[.01, 6562.77 * (1. + redshift), 2.5],
                                        dy=1. / w)
    # print(specpars)
    specpars[0] = specpars[0] / norm
    #specerrs[0] = specerrs[0] / norm
    zmeas = (specpars[1] / 6562.77) - 1.
    zmeas = redshift#1.5199
    #print('zmeas=',zmeas)
    l0 = 6562.77 * (1 + zmeas)
    window = [(-1500 * l0 / 3e5) + l0, (1500 * l0 / 3e5) + l0]
    # print('the window is=',window)

    # ====================================================================
    # extract the acube from the cube
    # ====================================================================

    # guesses needed for John's extraction routine (already have zmeas above)...
    # observed central wavelength of Halpha emission
    linewl = 6562.77 * (1 + redshift)#16537.524#specpars[1]  # 16572.68#
    #print('linewl=', linewl)
    # Halpha linewidth i.e. sigma of Ha gaussian
    linewid = 3.6#specpars[2]  # 3.6           #specpars[2]
    #print('linewid=', linewid)
    # guess of area under Ha line
    area = 0.8# scipy.integrate.trapz(x=wav, y=gauss(specpars[0:3], wav))  # from integrated spectrum
    #print('area=', area)

    # get the acube
    # acubefile=fitvelkmos_Ha(cubefile,wav,zmeas,linewl,linewid,area)
    # acubefile=kmos_cube_fit(cubefile,zmeas,linewl,linewid,area)
    acubefile = fitvelkmos_Ha(cubefile, wav, zmeas, linewl, linewid, area, ksky=sky.copy(), norm=norm, rmsmap=True,
                              maxbin=maxbin, minsig=True, s2ncut=s2ncut, power=power, modchi=modchi, smooth=False,
                              exten=exten, clipval=clipval)
    acubefile = acubefile

    # get the maps from the acube
    if filt == 'Hn3':
        res = unwrap_acube(acubefile, zmeas)
    elif filt == 'YJ':
        res = unwrap_acube(acubefile, zmeas, filt='YJ')

    return res[0], res[1], res[2]

    # move the products to the analysis directory
    os.system("mv *" + cubefile.split('/')[-1].split('.fit')[0] + "* " + prod_dir)

    #print('products extracted from cube!')
