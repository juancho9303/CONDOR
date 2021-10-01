"""
Created on 2020
@author: Juan Espejo @ jespejosalcedo@swin.edu.au

This script does the full kinematic modelling of the disk galaxies by doing:

1. Creating model datacubes with their right convolutions
2. Extracting the velocity fields from those datacubes
3. Running emcee to obtain the best model (maximum likelihood) for both the individual and combined cases
4. Return the best angular momentum measurements and plot all the results using the script "plot_and_save_results"

The code is run using: "python full_modelling.py ID data walkers steps type"

ID = 0...38 is the galaxy counter
data = ["NS", "AO", "combined"]
walkers = number of walkers (integer)
steps = number of steps (integer)
type = ["plot","run"]

For an explanation of the code and method see Espejo Salcedo et al. 2021 (https://arxiv.org/abs/2109.10614)
"""
from __future__ import print_function
import gc, sys, os
import argparse
import numpy as np
from termcolor import colored
import warnings
from pathlib import Path
import pandas as pd
from scipy import ndimage, misc

from astropy.io import fits, ascii
import matplotlib.pyplot as plt

from copy import copy

from cosmology_calc import angulardistance
import plot_and_save_results as make_plots
from modules import fitting_functions as my_funcs
from get_j import get_j
from galaxy import Galaxy

warnings.filterwarnings("ignore")
home = str(Path.home())

path = home + "/Dropbox/PhD/paper_2/"
par = pd.read_csv(f'{path}/Data/data_parameters.csv')
results = pd.read_csv(f'{path}/results/results_table.csv')
types = ["inten", "vel", "velerr"]  # H alpha intensity, vel map and vel error


# Load all galaxy parameters
def load_parameters(i):
    gal = Galaxy(par.name[i],
                 par.z_ns[i],
                 par.halpha_ns[i],
                 par.velmax_ns[i],
                 par.reff_kpc[i],
                 par.psf_fwhm_NS_paper[i],
                 par.maxbin_ns[i],
                 par.x0_ns[i],
                 par.y0_ns[i],
                 par.pafit_ns[i],
                 par.pixscale_ns[i],
                 par.SFR[i],
                 par.M_s[i],
                 par.type_ns[i],
                 par.sigma_kernel_pix_ns[i],
                 par.phot_file_name[i],
                 par.H_band[i],
                 par.J_band[i],
                 par.psf_fwhm_h[i],
                 par.psf_fwhm_j[i],
                 par.pixscale_hst[i],
                 par.sigma_x_ns[i],
                 par.sigma_y_ns[i],
                 par.q_ns[i],
                 par.theta_ns[i],
                 par.psf_fwhm_mine_ns[i],
                 par.z_ao[i],
                 par.halpha_ao[i],
                 par.pa_ao[i],
                 par.velmax_ao[i],
                 par.reff_kpc_ao[i],
                 par.psf_fwhm_paper_ao[i],
                 par.maxbin_ao[i],
                 par.x0_ao[i],
                 par.y0_ao[i],
                 par.pafit_ao[i],
                 par.pixscale_ao[i],
                 par.extent_to_show_ao[i],
                 par.type_ao[i],
                 par.alpha[i],
                 par.beta[i],
                 par.q[i],
                 par.theta[i],
                 par.airy[i],
                 par.peak[i],
                 par.strehl[i],
                 par.FWHM_airy[i],
                 par.FWHM_moffat[i],
                 par.FWHM_total[i],
                 par.pa_guess[i],
                 par.inc_guess[i],
                 par.rflat_guess[i],
                 par.vflat_guess[i],
                 par.b_a_tachella_guillman[i],
                 par.r_d_exp_disk[i])
    return gal

# Load all the maps and convolution kernels
def load_maps(i):
    maps = np.empty([3, 39, 4], dtype=object)  # Create empty object for all  maps to be stored
    for k in range(len(types)):
        maps[0][i][k] = fits.open(f"{path}/../My_AM_code/Data_disks/{load_parameters(i).name}_NS_{types[k]}.fits")[0].data
        maps[1][i][k] = fits.open(f"{path}/../My_AM_code/Data_disks/{load_parameters(i).name}_AO_{types[k]}.fits")[0].data

    AO_kernel = fits.open(f"{path}/psf_model/AO/{load_parameters(i).name}_ao_model.fits")[0].data
    AO_kernel = AO_kernel / AO_kernel.sum()
    NS_kernel = fits.open(f"{path}/psf_model/NS/{load_parameters(i).name}_NS_model.fits")[0].data
    NS_kernel = NS_kernel / NS_kernel.sum()

    if load_parameters(i).pa_ao != 0:
        AO_kernel = ndimage.rotate(AO_kernel, -load_parameters(i).pa_ao, reshape=False)

    if load_parameters(i).H_band != 'no_data':
        den_data = fits.open(
            f'{path}/../Data/gals_photometry/' + load_parameters(i).phot_file_name + '/' +
            load_parameters(i).phot_file_name + '_' + load_parameters(i).H_band + '_drz.fits')[0].data
    elif load_parameters(i).H_band == 'no_data' and load_parameters(i).J_band != 'no_data':
        den_data = fits.open(
            f'{path}/../Data/gals_photometry/' + load_parameters(i).phot_file_name + '/' +
            load_parameters(i).phot_file_name + '_' + load_parameters(i).J_band + '_drz.fits')[0].data
    else:
        den_data = copy(maps[1][i][0])

    return maps[0][i][1], maps[0][i][0], maps[0][i][2], maps[1][i][1], maps[1][i][0], maps[1][i][2], AO_kernel, NS_kernel, den_data

# Load the previous runs for plotting only
def load_previous_runs(i, type_of_data):

    if type_of_data=='NS':
        dat = 0
    elif type_of_data=='AO':
        dat = 1
    elif type_of_data=='combined':
        dat = 2

    pa = results.pa[i * 3 + dat]
    inc = results.inc[i * 3 + dat]
    rflat = results.rflat[i * 3 + dat]
    vflat = results.vflat[i * 3 + dat]
    x0_ns = results.x0_ns[i * 3 + dat]
    y0_ns = results.y0_ns[i * 3 + dat]
    x0_ao = results.x0_ao[i * 3 + dat]
    y0_ao = results.y0_ao[i * 3 + dat]
    #r_d_pix = results.r_d[i * 3 + dat]

    return pa, inc, rflat, vflat, x0_ns, y0_ns, x0_ao, y0_ao#, r_d_pix


if __name__ == "__main__":

    RESOLUTION_OPTIONS = ['NS', 'AO', 'combined']

    parser = argparse.ArgumentParser()
    parser.add_argument("galaxy", type=int, help="Galaxy ID? (From 0 to 9)")
    parser.add_argument("resolution", type=str, help="Galaxy resolution: NS, AO or combined")
    parser.add_argument("nwalkers", type=int, help="Number of walkers for the MCMC run")
    parser.add_argument("steps", type=int, help="Number of steps in the MCMC run")
    parser.add_argument("type_run", type=str, help="Type of analysis: plot or run")

    args = parser.parse_args()
    nwalkers, steps  = args.nwalkers, args.steps
    type_run = args.type_run

    # Tell the user if there are errors:
    if args.resolution not in RESOLUTION_OPTIONS:
        raise ValueError("Invalid resolution option: must be one of {}"
                         .format(RESOLUTION_OPTIONS))

    if args.resolution == "NS":
        print(colored("Calculating Angular Momentum of", "green"), colored(par.name[args.galaxy], "green"),
              colored("at the NS resolution", "green"))
        # import yappi
        # yappi.set_clock_type("cpu")  # Use set_clock_type("wall") for wall time
        # yappi.start()
        get_j(args.galaxy, load_parameters(args.galaxy), load_maps(args.galaxy),
                 load_previous_runs(args.galaxy, 'NS'), res='NS', nwalkers = args.nwalkers,
                 steps =args.steps, type_run=args.type_run)
        # yappi.get_func_stats().print_all()
        # yappi.get_thread_stats().print_all()

    if args.resolution == "AO":
        print(colored("Calculating Angular Momentum of", "green"), colored(par.name[args.galaxy], "green"),
              colored("at the AO resolution", "green"))
        get_j(args.galaxy, load_parameters(args.galaxy), load_maps(args.galaxy),
                 load_previous_runs(args.galaxy, 'AO'), res='AO', nwalkers = args.nwalkers,
                 steps =args.steps, type_run=args.type_run)

    if args.resolution == "combined":
        print(colored("Calculating Angular Momentum of", "green"), colored(par.name[args.galaxy], "green"),
              colored("combining the resolutions", "green"))
        get_j(args.galaxy, load_parameters(args.galaxy), load_maps(args.galaxy),
                 load_previous_runs(args.galaxy, 'combined'), res='combined', nwalkers = args.nwalkers,
                 steps =args.steps, type_run=args.type_run)
