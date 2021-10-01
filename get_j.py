from __future__ import print_function
import gc
import sys, os
import argparse
import numpy as np
import warnings
from copy import copy
import zeus
import emcee
from pathlib import Path
from scipy import ndimage, misc

from astropy.io import fits, ascii

from multiprocessing import Pool
from copy import copy
from cosmology_calc import angulardistance
import plot_and_save_results as make_plots
from modules import fitting_functions as my_funcs
from galaxy import Galaxy

warnings.filterwarnings("ignore")

#@profile
def get_j(i, gal, maps, previous_results, res, nwalkers, steps, type_run):

    pixscale = [gal.pixscale_ns, gal.pixscale_ao]
    sigma_ns = gal.psf_fwhm_mine_ns / 2.35 / gal.pixscale_ns
    sigma_ao = gal.psf_fwhm_ao / 2.35 / gal.pixscale_ao
    resolution_ratio = gal.pixscale_ns / gal.pixscale_ao
    Da = angulardistance(gal.z_ao)

    l0 = 6562.77 * (1 + gal.z_ao)
    wav = np.arange(l0 - 101, l0 + 101, 2)

    vel_data_ns = maps[0]
    inten_data_ns = maps[1]
    vmap_er_ns = maps[2]
    vel_data_ao = maps[3]
    inten_data_ao = maps[4]
    vmap_er_ao = maps[5]
    AO_kernel = maps[6]
    NS_kernel = maps[7]
    den_data = maps[8]

    den_data = den_data / np.max(den_data)
    x0_phot = den_data.shape[0] / 2
    y0_phot = den_data.shape[1] / 2
    hst_pixscale = 0.05
    #inc_paper = my_funcs.inclination_from_axis_ratio(gal.b_a, 0.2) # q_0 = 0.2 for thick disks
    inc_axis_ratio = gal.inc_axis_ratio()

    if res == 'NS':

        kpc_per_pix = gal.pixscale_ns / 3600 / 180 * np.pi * Da * 1000
        rflat_lim = 9 / kpc_per_pix
        r_d_pix = gal.r_d / kpc_per_pix
        vflat_max = np.nanmax(vel_data_ns)
        y, x = np.indices([vel_data_ns.shape[0], vel_data_ns.shape[1]])
        vmap_er_ns[vmap_er_ns == 0] = np.nan
        ndim = 6
        constant_inputs = [vel_data_ns, pixscale, Da, r_d_pix, l0, wav, NS_kernel, AO_kernel, gal.z_ao, gal.x0_ns,
                           gal.y0_ns, inc_axis_ratio, rflat_lim, vflat_max]

        if type_run=="run":

            with Pool() as pool:
                #sampler = zeus.EnsembleSampler(nwalkers, ndim, my_funcs.lnprob_cube,
                sampler = emcee.EnsembleSampler(nwalkers, ndim, my_funcs.lnprob_cube,
                                                args=(vel_data_ns, x, y, my_funcs.ns_maps, constant_inputs, i,
                                                      vmap_er_ns ** 2), pool=pool)

                print("Running MCMC resampling...")
                import time
                start = time.time()

                sampler.run_mcmc(gal.set_priors_ns(ndim,nwalkers), steps, progress=True)

                end = time.time()
                multi_time = end - start
                print("Multiprocessing took {0:.1f} seconds".format(multi_time))

            samples = sampler.chain[:, -20:, :].reshape((-1, ndim))

            best_param_median = np.median(samples, axis=0)
            pa, inc, rflat, vflat, x0, y0 = best_param_median

            my_funcs.print_in_terminal(pa=pa, inc=inc, rflat=rflat, vflat=vflat, x0=x0, y0=y0)

        elif type_run=="plot":
            print("just plotting...")
            pa, inc, rflat, vflat, x0, y0, x0_unused, y0_unused, r_d_pix = previous_results
            rflat /= kpc_per_pix
            r_d_pix /= kpc_per_pix
            sampler   = None

        save_all = make_plots.individual(i, 0, gal.name, pa, inc, rflat, vflat, x0, y0,
                                         x0_phot, y0_phot, x, y, gal.z_ao, r_d_pix, NS_kernel, AO_kernel, pixscale,
                                         resolution_ratio, kpc_per_pix, vel_data_ns, vmap_er_ns, inten_data_ns, den_data,
                                         sigma_ns, gal.sigma_x_ns, gal.sigma_y_ns, gal.theta_ns, sigma_ao, gal.q_ao,
                                         gal.theta_ao, sampler, steps, hst_pixscale, type_run, copy(AO_kernel))

    if res == 'AO':

        kpc_per_pix = gal.pixscale_ao / 3600 / 180 * np.pi * Da * 1000  # Calculate kpc per pixel
        rflat_lim = 9 / kpc_per_pix
        r_d_pix = gal.r_d / kpc_per_pix
        vflat_max = np.nanmax(vel_data_ao)
        y, x = np.indices([vel_data_ao.shape[0], vel_data_ao.shape[1]])
        vmap_er_ao[vmap_er_ao == 0] = np.nan
        ndim = 6
        constant_inputs = [vel_data_ao, pixscale, Da, r_d_pix, l0, wav, NS_kernel, AO_kernel, gal.z_ao, gal.x0_ao,
                           gal.y0_ao, inc_axis_ratio, rflat_lim, vflat_max]

        if type_run=="run":

            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, my_funcs.lnprob_cube,
                                                args=(
                                                vel_data_ao, x, y, my_funcs.ao_maps, constant_inputs, i, vmap_er_ao ** 2),
                                                pool=pool)
                print("Running MCMC resampling...")
                sampler.run_mcmc(gal.set_priors_ao(ndim,nwalkers), steps, progress=True)

            samples = sampler.chain[:, -10:, :].reshape((-1, ndim))

            best_param_median = np.median(samples, axis=0)
            pa, inc, rflat, vflat, x0, y0 = best_param_median

            my_funcs.print_in_terminal(pa=pa, inc=inc, rflat=rflat, vflat=vflat, x0=x0, y0=y0)

        elif type_run=="plot":
            print("just plotting...")
            pa, inc, rflat, vflat, x0_unused, y0_unused, x0, y0, r_d_pix = previous_results
            rflat /= kpc_per_pix
            r_d_pix /= kpc_per_pix
            sampler  = None

        save_all = make_plots.individual(i, 1, gal.name, pa, inc, rflat, vflat, x0, y0, x0_phot,
                                         y0_phot, x, y, gal.z_ao, r_d_pix, NS_kernel, AO_kernel, pixscale, resolution_ratio,
                                         kpc_per_pix, vel_data_ao, vmap_er_ao, inten_data_ao, den_data, sigma_ns,
                                         gal.sigma_x_ns, gal.sigma_y_ns, gal.theta_ns, sigma_ao, gal.q_ao,
                                         gal.theta_ao, sampler, steps, hst_pixscale, type_run, copy(AO_kernel))

    if res == 'combined':

        vflat_max_ns = np.nanmax(vel_data_ns)
        y_ns, x_ns = np.indices([vel_data_ns.shape[0], vel_data_ns.shape[1]])
        vmap_er_ns[vmap_er_ns == 0] = np.nan

        vflat_max_ao = np.nanmax(vel_data_ao)
        y_ao, x_ao = np.indices([vel_data_ao.shape[0], vel_data_ao.shape[1]])
        vmap_er_ao[vmap_er_ao == 0] = np.nan
        ndim = 8

        kpc_per_pix = pixscale[1] / 3600 / 180 * np.pi * Da * 1000
        rflat_lim = 9 / kpc_per_pix
        r_d_pix = gal.r_d / kpc_per_pix

        constant_inputs_ns = [vel_data_ns, pixscale, Da, r_d_pix, l0, wav, NS_kernel, AO_kernel, gal.z_ao, gal.x0_ns,
                              gal.y0_ns, inc_axis_ratio, rflat_lim, vflat_max_ns]
        constant_inputs_ao = [vel_data_ao, pixscale, Da, r_d_pix, l0, wav, NS_kernel, AO_kernel, gal.z_ao, gal.x0_ao,
                              gal.y0_ao, inc_axis_ratio, rflat_lim, vflat_max_ao]


        if type_run == "run":

            with Pool() as pool:
                #sampler = zeus.EnsembleSampler(nwalkers, ndim, my_funcs.lnprob_combine_cube,
                sampler = emcee.EnsembleSampler(nwalkers, ndim, my_funcs.lnprob_combine_cube,
                                                args=(vel_data_ns, x_ns, y_ns, vmap_er_ns, my_funcs.ns_maps,
                                                      constant_inputs_ns, vel_data_ao, x_ao, y_ao, vmap_er_ao,
                                                      my_funcs.ao_maps, constant_inputs_ao, i, resolution_ratio),
                                                pool=pool)

                print("Running MCMC resampling...")
                sampler.run_mcmc(gal.set_priors_comb(ndim,nwalkers), steps, progress=True)

            samples = sampler.chain[:, -10:, :].reshape((-1, ndim))

            best_param_median = np.median(samples, axis=0)
            pa, inc, rflat, vflat, x0_ns, y0_ns, x0_ao, y0_ao = best_param_median

            my_funcs.print_in_terminal(pa=pa, inc=inc, rflat=rflat, vflat=vflat,
                                       x0_ns=x0_ns, y0_ns=y0_ns, x0_ao=x0_ao, y0_ao=y0_ao)


        elif type_run == "plot":
            print("just plotting...")
            pa, inc, rflat, vflat, x0_ns, y0_ns, x0_ao, y0_ao, r_d_pix = previous_results
            rflat /= kpc_per_pix
            r_d_pix /= kpc_per_pix
            sampler = None

        save_all = make_plots.combined(i, gal.name, pa, inc, rflat, vflat, x0_ns, y0_ns,
                                    x0_ao, y0_ao, x0_phot, y0_phot, x_ns, y_ns, x_ao, y_ao, gal.z_ao, r_d_pix,
                                    NS_kernel, AO_kernel, pixscale, resolution_ratio, kpc_per_pix, vel_data_ns,
                                    vmap_er_ns, vel_data_ao, vmap_er_ao, inten_data_ns, inten_data_ao, den_data,
                                    sigma_ns, gal.sigma_x_ns, gal.sigma_y_ns, gal.theta_ns, sigma_ao, gal.q_ao,
                                    gal.theta_ao, sampler, steps, hst_pixscale, type_run, copy(AO_kernel))


