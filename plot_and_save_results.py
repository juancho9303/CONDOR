# Import relevant libraries/modules
from __future__ import print_function
import gc, sys
import numpy as np
from termcolor import colored
import warnings
from regions import PixCoord, CirclePixelRegion
from copy import copy
from scipy.interpolate import interp1d
from pathlib import Path

from astropy.io import fits, ascii
from astropy.coordinates import Angle
#import pypdfplot.backend
import matplotlib as mpl
import matplotlib.pyplot as plt
import corner
import cmasher as cmr

from modules import fitting_functions as my_funcs
from cosmology_calc import angulardistance

from modules.fitting_functions import my_vel_model
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage import gaussian_filter
from astropy.convolution import convolve, Gaussian2DKernel, AiryDisk2DKernel
from scipy import ndimage, misc

warnings.filterwarnings("ignore")
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

home = str(Path.home())
path = home + "/Dropbox/PhD/paper_2/"

instrument = ["NS", "AO", "Merged"]
types = ["inten", "vel", "velerr"]  # H alpha intensity, vel map, HST continuum and vel error
rmax_NS = 4.5# approximation of the extent of the data based on the 2D maps
rmax_AO = 3.1# approximation of the extent of the data based on the 2D maps
sigma_hst_data = 1.5

import matplotlib.font_manager as fm
fontprops = fm.FontProperties(size=14)


def individual(i, j, galaxy, pa_fit, inc_fit, rflat_fit, vflat_fit, x0_fit, y0_fit, x0_phot, y0_phot, x, y, z,
               r_d, NS_kernel, AO_kernel, pixscale, resolution_ratio, kpc_per_pix, vel_data, vmap_er, inten_data,
               den_data, sigma_ns, sigma_x_ns, sigma_y_ns, theta_ns, sigma_ao, q_ao, theta_ao, sampler, steps,
               hst_pixscale, type, kernel_HST):

    size = [~np.isnan(vel_data)]
    vflat_max = np.nanmax(vel_data)

    # Use the optimized parameters to calculate AM
    vel_fit = my_funcs.my_vel_model_for_plot(x, y, pa_fit, inc_fit, rflat_fit, vflat_fit, x0_fit, y0_fit, None, None, i, vel_data)

    r_deprojected = copy(vel_fit[6])

    # Get the deprojected velocity map
    dep_vmap_fit = vel_fit[0] / (vel_fit[1] + 1e-5)
    dep_vmap_fit[(vel_fit[1] < 0.1) & (vel_fit[1] > -0.1)] = 0 # Filter some

    rad_int = my_funcs.radial_profile(vel_data, x0_fit, y0_fit, vmap_er)[0]  # radius as integers

    r_d_guess = copy(r_d)
    r_d_min = copy(r_d)
    l0 = 6562.77 * (1 + z)
    wav = np.arange(l0 - 120, l0 + 120, 2)

    print(colored("r_d (pixels) input =", "blue"), "%.2f" % r_d_min)

    # if type=="run":
    #
    #     # Estimation of the density profile
    #     nwalkers_phot, steps_phot = 150, 300
    #     ndim_den = 5
    #     y_den, x_den = np.indices(den_data.shape)
    #
    #     p0_den = np.random.rand(ndim_den * nwalkers_phot).reshape((nwalkers_phot, ndim_den))
    #
    #     for ii in range(len(p0_den)):
    #         p0_den[ii, :] = [
    #             np.random.uniform(r_d_guess - 0.2, r_d_guess + 0.2),
    #             np.random.uniform(x0_phot - 1.5, x0_phot + 1.5),
    #             np.random.uniform(y0_phot - 1.5, y0_phot + 1.5),
    #             np.random.uniform(inc_fit - 15, inc_fit + 15),
    #             np.random.uniform(pa_fit - 15, pa_fit + 15),
    #         ]
    #
    #
    #     with Pool() as pool:
    #         sampler_den = emcee.EnsembleSampler(nwalkers_phot, ndim_den, my_funcs.lnprob_den,
    #                                             args=(den_data, x_den, y_den,
    #                                                   (6.5e-4 * np.random.normal(size=den_data.shape)) ** 2, kernel_HST,
    #                                                   r_d_guess, x0_phot, y0_phot, inc_fit, pa_fit), pool=pool)
    #         print(colored("Finding the best density model:", "green"))
    #         sampler_den.run_mcmc(p0_den, steps_phot, progress=True)
    #
    #     fit_params_den = np.empty([ndim_den])
    #     fit_params_den_errors = np.empty([ndim_den, 2])
    #     den_parameter_error = np.empty([ndim_den])
    #
    #     for t in range(len(fit_params_den)):
    #         mcmc_den = np.percentile(sampler_den.chain[:, -1, t],
    #                                  [16, 50, 84])  # 16, 50 and 84 percentiles from the chains
    #         q_den = np.diff(mcmc_den)  # obtain the errors for each parameter
    #         fit_params_den_errors[t] = [q_den[0], q_den[1]]
    #         den_parameter_error[t] = np.mean(fit_params_den_errors[t, :])
    #         fit_params_den[t] = mcmc_den[1]
    #
    #     r_d_min_HST, x0_phot, y0_phot, inc_phot, pa_phot = fit_params_den
    #     r_d_min_error, x0_phot_error, y0_phot_error, inc_phot_error, pa_phot_error = den_parameter_error
    #
    #     fig, ((ax1, ax2, ax3, ax4, ax5)) = plt.subplots(figsize=(17, 3), ncols=5, nrows=1)
    #
    #     for count, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    #         ax.set_xlim(0, steps_phot)
    #         ax.set_xlabel(r"$\mathrm{Steps}$")
    #         ax.plot(sampler_den.chain[:, :, count].T, c='cornflowerblue', alpha=0.4)
    #
    #     ax1.set_ylabel(r"$r_d$", fontsize=14)
    #     ax2.set_ylabel(r'$x_0$', fontsize=14)
    #     ax3.set_ylabel(r'$y_0)$', fontsize=14)
    #     ax4.set_ylabel(r"$i(^\circ)$", fontsize=14)
    #     ax5.set_ylabel(r"$\theta(^\circ)$", fontsize=14)
    #     plt.savefig(f"{path}/results/{galaxy}/{galaxy}_{instrument[j]}_exp_disk.pdf", overwrite=True, bbox_inches='tight')
    #     plt.close()
    #
    #     print(colored("r_d fit (HST pixels) from MCMC =", "blue"), "%.2f" % r_d_min_HST)
    #     r_d_min_ao = r_d_min_HST * hst_pixscale / pixscale[1]
    #     r_d_min = r_d_min_ao * (pixscale[1] / pixscale[j])
    #     print(colored("r_d (pixels) at the right res =", "blue"), "%.2f" % r_d_min)
    #     print(colored("r_eff (kpc) at the right res =", "blue"), "%.2f" % (1.68* r_d_min * kpc_per_pix))
    #     print(colored("inc kin =", "blue"), "%.2f" % inc_fit, colored("inc phot =", "blue"), "%.2f" % inc_phot)
    #     print(colored("pa kin =", "blue"), "%.2f" % pa_fit, colored("pa phot =", "blue"), "%.2f" % pa_phot)
    #
    #     fig, (ax1,ax2,ax3) = plt.subplots(ncols=3)
    #     den_model, den_model_faceon = my_funcs.density_profile_convolved(x_den, y_den, 1, r_d_guess, x0_phot, y0_phot,
    #                                                                      inc_phot, pa_phot, kernel_HST)
    #     ax1.imshow(den_model)
    #     den_model, den_model_faceon = my_funcs.density_profile_convolved(x_den, y_den, 1, r_d_min_HST, x0_phot, y0_phot,
    #                                                                      inc_phot, pa_phot, kernel_HST)
    #     ax2.imshow(den_model)
    #     ax3.imshow(den_data)
    #     plt.savefig(f"{path}/results/{galaxy}/{galaxy}_{instrument[j]}_experiment.pdf", overwrite=True,
    #                 bbox_inches='tight')
    #     plt.close()

    # Create the density profile model from the best fit parameters
    den_model, den_model_faceon = my_funcs.density_profile_mcmc(x, y, 1, r_d_min, x0_fit, y0_fit, inc_fit, pa_fit)

    # Get the deprojected velocity map by dividing by the velocity factor
    dep_vmap = vel_data / (vel_fit[1] + 1e-5)
    dep_vmap[(vel_fit[1] < 0.2) & (vel_fit[1] > -0.2)] = 0  # Filter values from the deprojection factor
    dep_vmap[dep_vmap==0]=0
    dep_vmap_er = copy(vmap_er)
    weight = dep_vmap_er ** (-2)
    weight[np.isinf(weight)] = 0

    dep_vmap_ = copy(dep_vmap)
    dep_vmap_[np.isnan(dep_vmap_)] = 0

    print(colored("Calculating angular momentum:", "green"))

    # Calculate j model for extrapolation
    J_model, M_model, j_model = my_funcs.calculate_j(r_deprojected, vel_fit[2], den_model_faceon, 0, kpc_per_pix)
    #my_funcs.print_in_terminal(J_model=J_model, M_model=M_model, j_model=j_model)

    vel_fit_only = copy(vel_fit[2])
    vel_fit_only[np.isnan(vel_data)] = 0
    J_model_data, M_model_data, j_model_data = my_funcs.calculate_j(r_deprojected, vel_fit_only, den_model_faceon, 0,
                                                           kpc_per_pix)

    # Get the circular velocity:
    binned_vmap = my_funcs.radial_profile(dep_vmap_, x0_fit, y0_fit, data_error=dep_vmap_er) # Do the annular binning
    dep_rad_long = np.arange(0, len(binned_vmap[4])+5, 1)  # Full deprojected radius
    v_model = my_funcs.v_circular_model(dep_rad_long * kpc_per_pix, vflat_fit, rflat_fit * kpc_per_pix)  # rotation curve model

    vel_fit_only = copy(vel_fit[2])
    vel_fit_only[np.isnan(vel_data)] = 0

    # Calculate j up to a limiting radius (1rh)
    r_1rh = 1.68 * r_d_min
    mask_1rh = my_funcs.create_circular_mask((vel_data.shape[0]), (vel_data.shape[1]), center=(x0_fit, y0_fit),radius=r_1rh)
    vel_fit_1rh = copy(vel_fit[2])
    vel_fit_1rh[~mask_1rh] = 0
    den_1rh = copy(den_model_faceon)
    den_1rh[~mask_1rh] = 0

    J_1rh, M_1rh, j_1rh = my_funcs.calculate_j(r_deprojected, vel_fit_1rh, den_1rh, 0, kpc_per_pix)

    # Calculate j up to a limiting radius (2rh)
    r_2rh = 3.4 * r_d_min
    mask_2rh = my_funcs.create_circular_mask((vel_data.shape[0]),(vel_data.shape[1]),center=(x0_fit,y0_fit),radius=r_2rh)
    vel_fit_2rh = copy(vel_fit[2])
    vel_fit_2rh[~mask_2rh] = 0
    den_2rh = copy(den_model_faceon)
    den_2rh[~mask_2rh] = 0

    J_2rh, M_2rh, j_2rh = my_funcs.calculate_j(r_deprojected, vel_fit_2rh, den_2rh, 0, kpc_per_pix)

    # Calculate j up to a limiting radius (3rh)
    r_3rh = 5.04 * r_d_min
    mask_3rh = my_funcs.create_circular_mask((vel_data.shape[0]), (vel_data.shape[1]), center=(x0_fit, y0_fit), radius=r_3rh)
    vel_fit_3rh = copy(vel_fit[2])
    vel_fit_3rh[~mask_3rh] = 0
    den_3rh = copy(den_model_faceon)
    den_3rh[~mask_3rh] = 0

    J_3rh, M_3rh, j_3rh = my_funcs.calculate_j(r_deprojected, vel_fit_3rh, den_3rh, 0, kpc_per_pix)
    #my_funcs.print_in_terminal(j_1rh=j_1rh, j_2rh=j_2rh, j_3rh=j_3rh)

    j_1rh = my_funcs.calc_j_analitically(r_d_min, rflat_fit, vflat_fit, r_1rh, kpc_per_pix)
    j_2rh = my_funcs.calc_j_analitically(r_d_min, rflat_fit, vflat_fit, r_2rh, kpc_per_pix)
    j_3rh = my_funcs.calc_j_analitically(r_d_min, rflat_fit, vflat_fit, r_3rh, kpc_per_pix)
    j_model = my_funcs.calc_j_analitically(r_d_min, rflat_fit, vflat_fit, 5*r_3rh, kpc_per_pix)
    print(colored("j analitic =", "blue"), "%.2f" % j_model)

    # Calculate j model observed
    J_observed, M_observed, j_observed = my_funcs.calculate_j_weighted(r_deprojected, dep_vmap_fit, den_model_faceon, weight,
                                                              size, kpc_per_pix)

    # Calculate j model observed WITH HST MAP
    #deprojected_HST_map = den_data *  (vel_fit[1])
    # p1, p2, p3, dep_fac = my_funcs.deproject_HST_map(x, y, 1, r_d_min, x0_phot, y0_phot, inc_fit, pa_fit, den_data)
    # den_data = den_data / np.max(den_data)
    # deprojected_HST = den_data / dep_fac
    # deprojected_HST[(dep_fac < 0.1) & (dep_fac > -0.1)] = 0  # Filter some
    # J_observed, M_observed, j_observed = my_funcs.calculate_j_weighted(r_deprojected, dep_vmap_fit, den_model_faceon,
    #                                                                    weight, size, kpc_per_pix)
    # J_observed_HST, M_observed_HST, j_observed_HST = my_funcs.calculate_j_weighted(r_deprojected, dep_vmap_fit,
    #                                                                                deprojected_HST, weight,
    #                                                                                size, kpc_per_pix)

    rad_long = np.arange(0, len(binned_vmap[4]) + 5, 0.01)
    v_mod = my_funcs.v_circular_model(rad_long * kpc_per_pix, vflat_fit, rflat_fit * kpc_per_pix)
    v_2rh = v_mod[np.abs(rad_long - 2 * r_d_min).argmin()]
    j_approx = (1.19 * v_2rh * r_1rh * kpc_per_pix)
    print(colored("j_approx =", "blue"), "%.2f" % j_approx)

    if j == 0:
        extended_radius = np.arange(0, len(binned_vmap[4]) + 3, 1)
    elif j == 1:
        extended_radius = np.arange(0, len(binned_vmap[4]) + 6, 1)

    J_cum_model, M_cum_model, j_cum_model = my_funcs.calculate_cumulative_j(r_deprojected, vel_fit[2], den_model_faceon,
                                                                            extended_radius, rad_int, kpc_per_pix)

    extended_radius = np.arange(0, len(binned_vmap[4]) + 8, 1)

    J_cum_model, M_cum_model, j_cum_model = my_funcs.calculate_cumulative_j(r_deprojected, vel_fit[2], den_model_faceon,
                                                                            extended_radius, rad_int, kpc_per_pix)

    #-------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ Make all plots --------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------

    vel_fit[0][np.isnan(vel_data)] = np.nan
    rad = len(vel_data[0])
    ang_fit = [0, np.pi] + np.radians(pa_fit) - np.pi
    data = my_funcs.radial_profile(vel_data, x0_fit, y0_fit, vmap_er)

    if sampler is not None:
        samples = sampler.chain[:, -10:, :].reshape((-1, 6))
        best_param_std = np.std(samples, axis=0)
        pa_fit_error, inc_fit_error, rflat_fit_error, vflat_fit_error, x0_fit_error, y0_fit_error = best_param_std

    def make_plots():

        plt.rc('text', usetex=True)

        if sampler is not None:
            # Plot the walkers for all parameters
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(figsize=(8, 3), ncols=3, nrows=2)

            for count, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
                ax.set_xlim(0, steps)
                ax.plot(sampler.chain[:, :, count].T, c='cornflowerblue', alpha=0.4)

            for ax in [ax1,ax2,ax3]:
                ax.set_xticks([])

            ax1.axhline(y=pa_fit, ls='--', c="darkorange")
            ax1.set_ylabel(r"$\theta(^\circ)$", fontsize=14)

            ax2.axhline(y=inc_fit, ls='--', c="darkorange")
            ax2.set_ylabel(r'$i(^\circ)$', fontsize=14)

            ax3.axhline(y=rflat_fit, ls='--', c="darkorange")
            ax3.set_ylabel(r'$r_\mathrm{flat}$', fontsize=14)
            ax3.set_title('rflat=%.1f kpc' % (rflat_fit * kpc_per_pix))

            ax4.axhline(y=vflat_fit, ls='--', c="darkorange")
            ax4.set_ylabel(r'$v_\mathrm{flat}(\mathrm{km/s})$', fontsize=14)
            ax4.set_xlabel(r"$\mathrm{Steps}$")

            ax5.axhline(y=x0_fit, ls='--', c="darkorange")
            ax5.set_ylabel(r'$x_{0}(\mathrm{pixel})$', fontsize=14)
            ax5.set_xlabel(r"$\mathrm{Steps}$")

            ax6.axhline(y=y0_fit, ls='--', c="darkorange")
            ax6.set_ylabel(r'$y_{0}(\mathrm{pixel})$', fontsize=14)
            ax6.set_xlabel(r"$\mathrm{Steps}$")
            plt.subplots_adjust(wspace=0.43, hspace=0.0)
            plt.savefig(f"{path}/results/{galaxy}/{galaxy}_{instrument[j]}_walkers.png", overwrite=True,
                        bbox_inches='tight')

            # Corner plot
            cornerfig = corner.corner(samples,
                                      labels=[r"$\theta_\mathrm{PA}$", r"$i$", r"$r_\mathrm{flat}$",
                                              r"$v_\mathrm{flat}$",r"$x_0$", r"$y_0$"],
                                      quantiles=[0.16, 0.5, 0.84], show_titles=True)

            ax = cornerfig.add_axes([0.6, .65, .2, .2])
            ax.imshow(vel_data, cmap=mpl.cm.RdYlBu_r, origin='lower')
            ax.set_facecolor('black')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(r"$\mathrm{Data\quad steps=%d}$" % (steps))
            ax.plot(rad * np.cos(ang_fit) + x0_fit, rad * np.sin(ang_fit) + y0_fit, ls='--', c="red", lw=3)  # Zero-vel
            ax.plot(-rad * np.sin(ang_fit) + x0_fit, rad * np.cos(ang_fit) + y0_fit, ls='--', c="lime",
                     lw=3)  # Major ax
            cornerfig.show()
            plt.savefig(f"{path}/results/{galaxy}/{galaxy}_{instrument[j]}_corner.pdf", overwrite=True,
                        bbox_inches='tight')

        # Plot with the HST map
        fig, (ax0, ax4, ax1, ax2, ax3) = plt.subplots(figsize=(14, 4), ncols=5, nrows=1)

        fontprops = fm.FontProperties(size=15)
        ax0.imshow(den_data, cmap=cmr.sunburst, origin='lower', interpolation='nearest')
        Da = angulardistance(z)
        kpc_per_pix_HST = hst_pixscale / 3600 / 180 * np.pi * Da * 1000
        scale = AnchoredSizeBar(ax0.transData, 5 * kpc_per_pix_HST, r"$5\,\, \mathrm{kpc}$", 'lower right',
                                pad=0.1, color='white', frameon=False, size_vertical=0.4, fontproperties=fontprops)
        arcsec = AnchoredSizeBar(ax0.transData, 0.5 / hst_pixscale, r'$0.5"$', 'lower left',
                                 pad=0.1, color='white', frameon=False, size_vertical=0.4, fontproperties=fontprops)
        ax0.set_xlim(-0.5, den_data.shape[1] - 0.5)
        ax0.set_ylim(-0.5, den_data.shape[0] - 0.5)

        ax0.add_artist(scale)
        ax0.add_artist(arcsec)
        ax0.text(0.03, 0.97, r"$\mathrm{%s}$" %galaxy, c="w", size=14,rotation=0., ha="left",va="top", transform=ax0.transAxes)
        ax0.text(0.03, 0.88, r"$\textit{I-}\mathrm{band},\, z=1.52$", c="w", size=14, rotation=0., ha="left",
                 va="top", transform=ax0.transAxes)
        ax0.set_title(r"$\mathit{HST\,\,}\mathrm{image}$", size=20)


        ax4.imshow(inten_data, cmap=cmr.sunburst, origin='lower', interpolation='nearest')
        ax4.set_xlim(-0.5, inten_data.shape[1] - 0.5)
        ax4.set_ylim(-0.5, inten_data.shape[0] - 0.5)
        if j == 0:
            ax4.add_patch(my_funcs.draw_elliptical_psf(sigma_x_ns, sigma_y_ns, theta=Angle(theta_ns - 45, 'deg')))
        elif j == 1:
            ax4.add_patch(my_funcs.draw_elliptical_psf(sigma_ao, sigma_ao * q_ao, theta=Angle(theta_ao - 45, 'deg')))

        ax4.set_title(r"$\mathrm{H\alpha\,\, intensity}$", size=20)

        for ax in [ax1, ax2, ax3, ax0, ax4]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('black')

        # 3" halo to check the scale
        center_halo = PixCoord(x0_fit, y0_fit)
        halo = CirclePixelRegion(center=center_halo, radius=(r_3rh))
        patch_halo = halo.as_artist(facecolor='w', edgecolor='w', color='w', fill=False, lw=3)
        ax1.add_patch(patch_halo)
        ax1.text(0.02, 0.98, r'$\mathrm{r}=3r_\mathrm{eff}$', c="w", size=20, rotation=0., ha="left", va="top",
                 transform=ax1.transAxes)

        vel_fit_for_plot = np.nanmax(np.abs(vel_fit[0]))

        ax1.imshow(vel_data, cmap=mpl.cm.RdYlBu_r, vmin=vel_fit_for_plot * (-1),
                   vmax=vel_fit_for_plot, origin='lower', interpolation='nearest')
        ax1.set_xlim(-0.5, vel_data.shape[1] - 0.5)
        ax1.set_ylim(-0.5, vel_data.shape[0] - 0.5)

        ax1.plot(rad * np.cos(ang_fit) + x0_fit, rad * np.sin(ang_fit) + y0_fit, ls='--', c="red", lw=3)  # Zero-vel
        ax1.plot(-rad * np.sin(ang_fit) + x0_fit, rad * np.cos(ang_fit) + y0_fit, ls='--', c="lime", lw=3)  # Major ax
        if j == 0:
            ax1.add_patch(my_funcs.draw_elliptical_psf(sigma_x_ns, sigma_y_ns, theta=Angle(theta_ns - 45, 'deg')))
        elif j == 1:
            ax1.add_patch(my_funcs.draw_elliptical_psf(sigma_ao, sigma_ao * q_ao, theta=Angle(theta_ao - 45, 'deg')))
        ax1.set_title(r"$v_\mathrm{rot}\,\,\mathrm{data}$", size=20)

        input_params = [pa_fit, inc_fit, rflat_fit, vflat_fit, x0_fit, y0_fit, x0_fit, y0_fit]
        constant_inputs = [vel_data, pixscale, angulardistance(z), r_d_guess, l0, wav, NS_kernel, AO_kernel, z, x0_fit,
                           y0_fit, inc_fit, rflat_fit, vflat_max]

        if j == 0:
            vel_model = my_funcs.ns_maps(i, input_params, constant_inputs)

        if j == 1:
            vel_model = my_funcs.ao_maps(i, input_params, constant_inputs)

        ax2.imshow(vel_model, cmap=mpl.cm.RdYlBu_r, vmin=vel_fit_for_plot * (-1),
                   vmax=vel_fit_for_plot, origin='lower', interpolation='nearest')
        ax2.plot(rad * np.cos(ang_fit) + x0_fit, rad * np.sin(ang_fit) + y0_fit, ls='--', c="red", lw=3)  # Zero-vel
        ax2.plot(-rad * np.sin(ang_fit) + x0_fit, rad * np.cos(ang_fit) + y0_fit, ls='--', c="lime",
                 lw=3)  # Major ax PA
        cs = ax2.contour(x, y, vel_fit[0], cmap=mpl.cm.RdYlBu_r, interpolation='none')
        ax2.contour(cs, colors='k')
        ax2.set_xlim(-0.5, vel_data.shape[1] - 0.5)
        ax2.set_ylim(-0.5, vel_data.shape[0] - 0.5)
        ax2.set_title(r"$v_\mathrm{rot}\,\,\mathrm{model}$", size=20)

        plot = ax3.imshow(vel_data - vel_model, cmap=mpl.cm.RdYlBu_r, vmin=np.max(vel_fit_for_plot) * (-1),
                          vmax=np.max(vel_fit_for_plot), origin='lower', interpolation='nearest')
        ax3.set_xlim(-0.5, vel_data.shape[1] - 0.5)
        ax3.set_ylim(-0.5, vel_data.shape[0] - 0.5)
        ax3.text(1.4, 0.5, r'$v\mathrm{\, [km/s]}$', size=20, ha='right',
                 va='center', rotation='vertical', transform=ax3.transAxes)
        ax3.set_title(r"$v_\mathrm{rot}\,\,\mathrm{residuals}$", size=20)
        cb = plt.colorbar(plot, cax=fig.add_axes([0.9, 0.235, 0.02, 0.52]),
                          ticks=[-vel_fit_for_plot, 0, vel_fit_for_plot], ax=ax1,
                          orientation='vertical')
        cb.ax.set_yticklabels([r"$%d$" % int(-vel_fit_for_plot), r"$0$",
                               r"$%d$" % int(vel_fit_for_plot)])  # horizontal colorbar
        cb.ax.yaxis.set_tick_params(color='black', pad=1, size=4)
        cb.outline.set_visible(True)
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="black", size=18)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        plt.savefig(f"{path}/results/{galaxy}/{galaxy}_{instrument[j]}_with_HST.pdf", overwrite=True, bbox_inches='tight')
        plt.close()

        # Plots for paper
        fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 4), ncols=3, nrows=1)

        for ax in [ax1, ax2, ax3]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('black')

        # 3" halo to check the scale
        center_halo = PixCoord(x0_fit, y0_fit)
        halo = CirclePixelRegion(center=center_halo, radius=(r_3rh))
        patch_halo = halo.as_artist(facecolor='w', edgecolor='w', color='w', fill=False, lw=3)
        ax1.add_patch(patch_halo)
        ax1.text(0.02, 0.98, r'$\mathrm{r}=3r_\mathrm{eff}$', c="w", size=25, rotation=0., ha="left", va="top",
                 transform=ax1.transAxes)
        if j == 0:
            ax1.add_patch(my_funcs.draw_elliptical_psf(sigma_x_ns, sigma_y_ns, theta=Angle(theta_ns - 45, 'deg')))
        elif j == 1:
            ax1.add_patch(my_funcs.draw_elliptical_psf(sigma_ao, sigma_ao * q_ao, theta=Angle(theta_ao - 45, 'deg')))

        vel_fit_for_plot = np.nanmax(np.abs(vel_fit[0]))

        ax1.imshow(vel_data, cmap=mpl.cm.RdYlBu_r, vmin=vel_fit_for_plot * (-1),
                   vmax=vel_fit_for_plot, origin='lower', interpolation='nearest')
        ax1.set_xlim(-0.5, vel_data.shape[1] - 0.5)
        ax1.set_ylim(-0.5, vel_data.shape[0] - 0.5)

        ax1.plot(rad * np.cos(ang_fit) + x0_fit, rad * np.sin(ang_fit) + y0_fit, ls='--', c="red",lw=3)  # Zero-vel
        ax1.plot(-rad * np.sin(ang_fit) + x0_fit, rad * np.cos(ang_fit) + y0_fit, ls='--', c="lime",lw=3) # Major ax
        ax1.set_title(r"$\mathrm{Data}$", size=28)

        input_params = [pa_fit, inc_fit, rflat_fit, vflat_fit, x0_fit, y0_fit, x0_fit, y0_fit]
        constant_inputs = [vel_data, pixscale, angulardistance(z), r_d_guess, l0, wav, NS_kernel, AO_kernel, z, x0_fit,
                           y0_fit, inc_fit, rflat_fit, vflat_max]

        if j==0:
            vel_model = my_funcs.ns_maps(i, input_params, constant_inputs)

        if j==1:
            vel_model = my_funcs.ao_maps(i, input_params, constant_inputs)

        ax2.imshow(vel_model, cmap=mpl.cm.RdYlBu_r, vmin=vel_fit_for_plot * (-1),
                   vmax=vel_fit_for_plot, origin='lower', interpolation='nearest')
        ax2.plot(rad * np.cos(ang_fit) + x0_fit, rad * np.sin(ang_fit) + y0_fit, ls='--', c="red", lw=3)  # Zero-vel
        ax2.plot(-rad * np.sin(ang_fit) + x0_fit, rad * np.cos(ang_fit) + y0_fit, ls='--', c="lime",lw=3)  # Major ax PA
        cs = ax2.contour(x, y, vel_fit[0], cmap=mpl.cm.RdYlBu_r, interpolation='none')
        ax2.contour(cs, colors='k')
        ax2.set_xlim(-0.5, vel_data.shape[1] - 0.5)
        ax2.set_ylim(-0.5, vel_data.shape[0] - 0.5)
        ax2.set_title(r"$\mathrm{Model}$", size=28)

        plot = ax3.imshow(vel_data - vel_model, cmap=mpl.cm.RdYlBu_r, vmin=np.max(vel_fit_for_plot) * (-1),
                          vmax=np.max(vel_fit_for_plot), origin='lower', interpolation='nearest')
        ax3.set_xlim(-0.5, vel_data.shape[1] - 0.5)
        ax3.set_ylim(-0.5, vel_data.shape[0] - 0.5)
        ax3.text(1.35, 0.5, r'$v\mathrm{\, [km/s]}$', size=25, ha='right',
                 va='center', rotation='vertical', transform=ax3.transAxes)
        ax3.set_title(r"$\mathrm{Residuals}$", size=28)
        cb = plt.colorbar(plot, cax=fig.add_axes([0.9, 0.118, 0.02, 0.75]),
                          ticks=[-vel_fit_for_plot, 0, vel_fit_for_plot], ax=ax1, orientation='vertical')
        cb.ax.set_yticklabels([r"$%d$"%int(-vel_fit_for_plot), r"$0$", r"$%d$"%int(vel_fit_for_plot)])  # horizontal colorbar
        cb.ax.yaxis.set_tick_params(color='black', pad=5, size=5)
        cb.outline.set_visible(True)
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="black", size=25)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        plt.savefig(f"{path}/results/{galaxy}/{galaxy}_{instrument[j]}.pdf", overwrite=True, bbox_inches='tight')
        plt.close()

        # Circular velocity combined with cumulatice j plot
        fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2,sharex=True)
        dep_vmap[dep_vmap == 0] = np.nan
        dep_rad_unbinned = np.arange(0, len(binned_vmap[4])+8, 0.01)
        v_model_unbinned = my_funcs.v_circular_model(dep_rad_unbinned * kpc_per_pix, vflat_fit, rflat_fit * kpc_per_pix)

        copy_data = copy(data[1])
        edge = copy(inten_data)
        edge[edge != 0] = 1
        copy_data[edge < 1] = np.nan
        dep_vmap[dep_vmap<10] = np.nan

        ax1.errorbar(data[1].ravel() / r_d_min, dep_vmap.ravel(), yerr=dep_vmap_er.ravel(), c='darkgreen',
                     fmt='o', ms=7, capsize=3, ecolor='dimgrey',markerfacecolor='white',
                     markeredgewidth=1.0, markeredgecolor="dimgrey", alpha=1,elinewidth=0.9)
        ax1.plot(dep_rad_unbinned / r_d_min, v_model_unbinned, '--', lw=4, label=r'$\mathrm{Model \,\, fit}$',
                        alpha=1,zorder=10)
        ax1.axvline(x=rflat_fit / r_d_min, ls='--', lw=2.5, c='red', alpha=0.9, label=r"$r_{\mathrm{flat}}=%.1f \mathrm{\,kpc}$" % (rflat_fit * kpc_per_pix))
        data[1][np.isnan(dep_vmap)] = np.nan
        colors_instrument=["darkgreen","darkorange"]
        type_name = ["NS","AO"]
        if np.nanmax(copy_data.ravel() / r_d_min) > np.min([np.max(dep_rad_long / r_d_min), 8]):
            ax1.annotate(r'$\mathrm{%s} \,\, r_\mathrm{max}$' % type_name[j], xy=(1, 0.23), xycoords='axes fraction',size=14, xytext=(0.7, 0.2),
                         arrowprops={'arrowstyle': '->','ls': '--', 'lw': 2.5, 'color': colors_instrument[j]}, alpha=0.9)
            ax1.axvline(x=np.nanmax(copy_data.ravel() / r_d_min), ls='--', lw=2.5, c=colors_instrument[j], alpha=0.9,
                        label=r'$\mathrm{%s} \,\, r_\mathrm{max}$' % type_name[j])
        else:
            ax1.axvline(x=np.nanmax(copy_data.ravel() / r_d_min), ls='--', lw=2.5, c=colors_instrument[j], alpha=0.9, label=r'$\mathrm{%s} \,\, r_\mathrm{max}$' % type_name[j])
        ax1.axhline(y=vflat_fit, ls='--', lw=2.5, c='grey', alpha=0.9)
        if np.min([np.max(extended_radius / r_d_min), 8]) > 6.3:
            ax1.axvline(x=5.04, ls='--', c='grey', alpha=0.9)
        ax1.axvline(x=3.4, ls='--', c='grey', alpha=0.9)
        ax1.set_ylabel(r'$v\mathrm{[km/s]}$', fontsize=20)
        ax1.tick_params(axis='both', which='major', direction='in', labelsize=18, pad=3)
        plt.xticks(fontsize=20, rotation=0)
        plt.yticks(fontsize=20, rotation=0)
        ax1.set_xlim(0, np.min([np.max(dep_rad_long / r_d_min), 8]))
        ax1.set_ylim(0, vflat_fit + 100)
        ax1.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, 1.3), columnspacing=0.85, handletextpad=0.53,
                   fancybox=True, shadow=True, ncol=3)

        # Cumulative j plot:
        f = interp1d(extended_radius / r_d_min, j_cum_model / np.max(j_cum_model), kind='cubic', fill_value="extrapolate")
        ax2.plot(dep_rad_unbinned, f(dep_rad_unbinned), "--", label=r"$\mathrm{Best \, model}$", lw=3.5, alpha=0.9)
        if np.nanmax(copy_data.ravel() / r_d_min) > np.min([np.max(dep_rad_long / r_d_min), 8]):
            ax2.axvline(x=np.nanmax(copy_data.ravel() / r_d_min), ls='--', lw=2.5, c=colors_instrument[j], alpha=0.9,
                        label=r'$\mathrm{%s} \,\, r_\mathrm{max}$')
        else:
            ax2.axvline(x=np.nanmax(copy_data.ravel() / r_d_min), ls='--', lw=2.5, c=colors_instrument[j], alpha=0.9, label=r'$\mathrm{%s} \,\, r_\mathrm{max}$')
        ax2.set_xlim(0, )
        ax2.set_xlabel(r"$\mathrm{deprojected} \quad r/r_d$", fontsize=20)
        ax2.set_ylabel(r"$j_*/j_\mathrm{*,tot}$", fontsize=20)
        ax2.axhline(y=1.0, ls='--', lw=2.5, c='grey', alpha=0.9)
        ax2.axvline(x=rflat_fit / r_d_min, ls='--', lw=2.5,color='red')
        ax2.axvline(x=3.4, ls='--', c='grey', alpha=0.9)
        if np.min([np.max(extended_radius / r_d_min), 8]) > 6.3:
            ax2.axvline(x=5.04, ls='--', c='grey', alpha=0.9)
            ax2.text((5.15), 0.45, r"$r{=}3r_\mathrm{eff}$", c="k", size=16)
        if np.min([np.max(extended_radius / r_d_min), 8]) > 4.5:
            ax2.text((3.5), 0.45, r"$r{=}2r_\mathrm{eff}$", c="k", size=16)
        ax2.set_xlim(0, np.min([np.max(extended_radius / r_d_min), 8]))
        ax2.set_ylim(0, 1.23)
        ax2.tick_params(axis='both', which='major', direction='in', labelsize=18, pad=3)
        plt.subplots_adjust(wspace=0.03, hspace=0.00)
        plt.savefig(f"{path}/results/{galaxy}/{galaxy}_{instrument[j]}_cumu_j_circ_vel.pdf", overwrite=True, bbox_inches='tight')
        plt.close()

        # Circular velocity combined with cumulative j plot and photometry fit
        fig, (ax1, ax2, ax3) = plt.subplots(figsize=(5,6),ncols=1, nrows=3, sharex=True)
        dep_vmap[dep_vmap == 0] = np.nan
        dep_rad_unbinned = np.arange(0, len(binned_vmap[4]) + 8, 0.01)
        v_model_unbinned = my_funcs.v_circular_model(dep_rad_unbinned * kpc_per_pix, vflat_fit, rflat_fit * kpc_per_pix)

        copy_data = copy(data[1])
        edge = copy(inten_data)
        edge[edge != 0] = 1
        copy_data[edge < 1] = np.nan
        dep_vmap[dep_vmap < 10] = np.nan

        ax1.errorbar(data[1].ravel() / r_d_min, dep_vmap.ravel(), yerr=dep_vmap_er.ravel(), c='darkgreen',
                     fmt='o', ms=7, capsize=3, ecolor='dimgrey', markerfacecolor='white',
                     markeredgewidth=1.0, markeredgecolor="dimgrey", alpha=1, elinewidth=0.9)
        ax1.plot(dep_rad_unbinned / r_d_min, v_model_unbinned, '--', lw=4, label=r'$\mathrm{Model \,\, fit}$',
                 alpha=1, zorder=10)
        ax1.axvline(x=rflat_fit / r_d_min, ls='--', lw=2.5, c='red', alpha=0.9,
                    label=r"$r_{\mathrm{flat}}=%.1f \mathrm{\,kpc}$" % (rflat_fit * kpc_per_pix))
        data[1][np.isnan(dep_vmap)] = np.nan
        colors_instrument = ["darkgreen", "darkorange"]
        type_name = ["NS", "AO"]
        if np.nanmax(copy_data.ravel() / r_d_min) > np.min([np.max(dep_rad_long / r_d_min), 8]):
            ax1.annotate(r'$\mathrm{%s} \,\, r_\mathrm{max}$' % type_name[j], xy=(1, 0.23), xycoords='axes fraction',
                         size=14, xytext=(0.7, 0.2),
                         arrowprops={'arrowstyle': '->', 'ls': '--', 'lw': 2.5, 'color': colors_instrument[j]},
                         alpha=0.9)
            ax1.axvline(x=np.nanmax(copy_data.ravel() / r_d_min), ls='--', lw=2.5, c=colors_instrument[j], alpha=0.9,
                        label=r'$\mathrm{%s} \,\, r_\mathrm{max}$' % type_name[j])
        else:
            ax1.axvline(x=np.nanmax(copy_data.ravel() / r_d_min), ls='--', lw=2.5, c=colors_instrument[j], alpha=0.9,
                        label=r'$\mathrm{%s} \,\, r_\mathrm{max}$' % type_name[j])
        ax1.axhline(y=vflat_fit, ls='--', lw=2.5, c='grey', alpha=0.9)
        if np.min([np.max(extended_radius / r_d_min), 8]) > 6.3:
            ax1.axvline(x=5.04, ls='--', c='grey', alpha=0.9)
        ax1.axvline(x=3.4, ls='--', c='grey', alpha=0.9)
        ax1.set_ylabel(r'$v\mathrm{[km/s]}$', fontsize=20)
        ax1.tick_params(axis='both', which='major', direction='in', labelsize=18, pad=3)
        plt.xticks(fontsize=20, rotation=0)
        plt.yticks(fontsize=20, rotation=0)
        ax1.set_xlim(0, np.min([np.max(dep_rad_long / r_d_min), 8]))
        ax1.set_ylim(0, vflat_fit + 100)
        ax1.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, 1.3), columnspacing=0.85, handletextpad=0.53,
                   fancybox=True, shadow=True, ncol=3)

        # Cumulative j plot:
        f = interp1d(extended_radius / r_d_min, j_cum_model / np.max(j_cum_model), kind='cubic',
                     fill_value="extrapolate")
        ax2.plot(dep_rad_unbinned, f(dep_rad_unbinned), "--", label=r"$\mathrm{Best \, model}$", lw=3.5, alpha=0.9)
        if np.nanmax(copy_data.ravel() / r_d_min) > np.min([np.max(dep_rad_long / r_d_min), 8]):
            ax2.axvline(x=np.nanmax(copy_data.ravel() / r_d_min), ls='--', lw=2.5, c=colors_instrument[j], alpha=0.9,
                        label=r'$\mathrm{%s} \,\, r_\mathrm{max}$')
        else:
            ax2.axvline(x=np.nanmax(copy_data.ravel() / r_d_min), ls='--', lw=2.5, c=colors_instrument[j], alpha=0.9,
                        label=r'$\mathrm{%s} \,\, r_\mathrm{max}$')
        ax2.set_xlim(0, )
        ax2.set_xlabel(r"$\mathrm{deprojected} \quad r/r_d$", fontsize=20)
        ax2.set_ylabel(r"$j_*/j_\mathrm{*,tot}$", fontsize=20)
        ax2.axhline(y=1.0, ls='--', lw=2.5, c='grey', alpha=0.9)
        ax2.axvline(x=rflat_fit / r_d_min, ls='--', lw=2.5, color='red')
        ax2.axvline(x=3.4, ls='--', c='grey', alpha=0.9)
        if np.min([np.max(extended_radius / r_d_min), 8]) > 6.3:
            ax2.axvline(x=5.04, ls='--', c='grey', alpha=0.9)
            ax2.text((5.15), 0.45, r"$r{=}3r_\mathrm{eff}$", c="k", size=16)
        if np.min([np.max(extended_radius / r_d_min), 8]) > 4.5:
            ax2.text((3.5), 0.45, r"$r{=}2r_\mathrm{eff}$", c="k", size=16)
        ax2.set_xlim(0, np.min([np.max(extended_radius / r_d_min), 8]))
        ax2.set_ylim(0, 1.23)
        ax2.tick_params(axis='both', which='major', direction='in', labelsize=18, pad=3)

        r_d = r_d_min*resolution_ratio
        den_dataz = den_data / np.nanmax(den_data)

        radial_profile = my_funcs.radial_data(den_dataz, annulus_width=1, working_mask=None, x=None, y=None, rmax=None)
        dep_rad = (radial_profile.r - 0.5) / r_d#_min

        dataz = my_funcs.radial_profile(den_dataz, x0_phot, y0_phot, np.ones_like(den_data) / 10)
        dep_rad = np.arange(0, len(dataz[4]), 1)
        error = 0.005 * dep_rad + 0.05

        model = my_funcs.surf_mass_den_profile(dep_rad, 1, r_d)
        model = gaussian_filter(model, sigma = sigma_hst_data)
        ax3.plot(dep_rad/r_d, model, color='red')
        ax3.plot(dep_rad/r_d, dataz[4], "o", color="dodgerblue")
        ax3.plot(dep_rad/r_d, dataz[4], color="dodgerblue")
        ax3.set_xlabel(r"$\mathrm{deprojected} \quad r/r_d$", fontsize=20)
        ax3.set_ylabel(r"$I(r)/I_\mathrm{max}$", fontsize=20)
        plt.subplots_adjust(wspace=0.03, hspace=0.00)
        plt.savefig(f"{path}/results/{galaxy}/{galaxy}_{instrument[j]}_cumu_j_circ_vel_WITH_HST.pdf", overwrite=True,
                    bbox_inches='tight')
        plt.close()


        # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(figsize=(15, 3), ncols=5, nrows=1)
        #
        # if j == 0:
        #     center_psf = PixCoord(sigma_ns + 0.5, sigma_ns + 0.5)
        #     reg = CirclePixelRegion(center=center_psf, radius=1.175 * sigma_ns)
        # elif j == 1:
        #     center_psf = PixCoord(sigma_ao + 1, sigma_ao + 1)
        #     reg = CirclePixelRegion(center=center_psf, radius=1.175 * sigma_ao)
        #
        # patch = reg.as_artist(facecolor='w', edgecolor='k', color='w', fill=True, lw=1)
        #
        # for ax in [ax1, ax2, ax3]:
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_facecolor('black')
        #
        # # 3" halo to check the scale
        # center_halo = PixCoord(x0_fit, y0_fit)
        # halo = CirclePixelRegion(center=center_halo, radius=(r_3rh))
        # patch_halo = halo.as_artist(facecolor='w', edgecolor='w', color='w', fill=False, lw=3)
        # ax1.add_patch(patch_halo)
        #
        # vel_fit_for_plot = np.nanmax(np.abs(vel_fit[0]))
        #
        # ax1.imshow(vel_data, cmap=mpl.cm.RdYlBu_r, vmin=vel_fit_for_plot * (-1),
        #            vmax=vel_fit_for_plot, origin='lower', interpolation='nearest')
        # ax1.set_xlim(-0.5, vel_data.shape[1] - 0.5)
        # ax1.set_ylim(-0.5, vel_data.shape[0] - 0.5)
        #
        # ax1.plot(rad * np.cos(ang_fit) + x0_fit, rad * np.sin(ang_fit) + y0_fit, ls='--', c="red", lw=3)  # Zero-vel
        # ax1.plot(-rad * np.sin(ang_fit) + x0_fit, rad * np.cos(ang_fit) + y0_fit, ls='--', c="lime", lw=3)  # Major ax
        # ax1.add_patch(patch)
        # if j==0:
        #     ax1.set_ylabel(r"$\mathrm{NS}$", size=28, rotation=0, ha="right", va="center", labelpad=15)
        # if j==1:
        #     ax1.set_ylabel(r"$\mathrm{AO}$", size=28, rotation=0, ha="right", va="center", labelpad=15)
        #
        # input_params = [pa_fit, inc_fit, rflat_fit, vflat_fit, x0_fit, y0_fit, x0_fit, y0_fit]
        # constant_inputs = [vel_data, pixscale, angulardistance(z), r_d_guess, l0, wav, NS_kernel, AO_kernel, z, x0_fit, y0_fit]
        #
        # if j == 0:
        #     vel_model = my_funcs.ns_maps(i, input_params, constant_inputs)
        #
        # if j == 1:
        #     vel_model = my_funcs.ao_maps(i, input_params, constant_inputs)
        #
        # ax2.imshow(vel_model, cmap=mpl.cm.RdYlBu_r, vmin=vel_fit_for_plot * (-1),
        #            vmax=vel_fit_for_plot, origin='lower', interpolation='nearest')
        # ax2.plot(rad * np.cos(ang_fit) + x0_fit, rad * np.sin(ang_fit) + y0_fit, ls='--', c="red", lw=3)  # Zero-vel
        # ax2.plot(-rad * np.sin(ang_fit) + x0_fit, rad * np.cos(ang_fit) + y0_fit, ls='--', c="lime", lw=3)
        # cs = ax2.contour(x, y, vel_fit[0], cmap=mpl.cm.RdYlBu_r, interpolation='none')
        # ax2.contour(cs, colors='k')
        # ax2.set_xlim(-0.5, vel_data.shape[1] - 0.5)
        # ax2.set_ylim(-0.5, vel_data.shape[0] - 0.5)
        #
        # plot = ax3.imshow(vel_data - vel_model, cmap=mpl.cm.RdYlBu_r, vmin=np.max(vel_fit_for_plot) * (-1),
        #                   vmax=np.max(vel_fit_for_plot), origin='lower', interpolation='nearest')
        # ax3.set_xlim(-0.5, vel_data.shape[1] - 0.5)
        # ax3.set_ylim(-0.5, vel_data.shape[0] - 0.5)
        # cb = plt.colorbar(plot, cax=fig.add_axes([0.36, 0.24, 0.04, 0.04]),
        #                   ticks=[-vel_fit_for_plot, 0, vel_fit_for_plot], ax=ax1, orientation='horizontal')
        # cb.ax.set_xticklabels([r"$%d$" % int(-vel_fit_for_plot), r"$0$", r"$%d$" % int(vel_fit_for_plot)])
        # cb.ax.yaxis.set_tick_params(color='white', pad=0, size=1)
        # cb.ax.xaxis.set_tick_params(color='None', pad=0.5, size=1)
        # cb.outline.set_visible(False)
        # plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color="white", size=12)
        #
        # vel_data[np.isnan(vel_data)] = 0
        # vmap_er[np.isnan(vmap_er)] = 0
        #
        # if j == 0:
        #     vel_model = my_vel_model_for_plot(x, y, pa_fit, inc_fit, rflat_fit, vflat_fit, x0_fit, y0_fit,
        #                              NS_kernel, None, i=i, data=vel_data)[0]
        # if j == 1:
        #     vel_model = my_vel_model_for_plot(x, y, pa_fit, inc_fit, rflat_fit, vflat_fit, x0_fit, y0_fit,
        #                              None, AO_kernel, i=i, data=vel_data)[0]
        #
        # # -- Extract the line...
        # radd = np.min([vel_data.shape[0],vel_data.shape[1]]) / 2.5
        # radd_offset = np.min([vel_data.shape[0],vel_data.shape[1]]) / 2.5 + 0.5
        # x0, y0 = -radd * np.sin(np.radians(pa_fit-90) - np.pi) + x0_fit, radd * np.cos(np.radians(pa_fit-90) - np.pi) + y0_fit  # These are in _pixel_ coordinates!!
        # x1, y1 = -radd * np.sin(np.radians(pa_fit-90)) + x0_fit, radd * np.cos(np.radians(pa_fit-90)) + y0_fit
        # length = int(np.hypot(x1 - x0, y1 - y0))
        # xe, ye = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
        # #vel_data = copy(vel_data[0:length,0:length])
        #
        # # Extract the values along the line
        # ma_values = vel_data[ye.astype(np.int), xe.astype(np.int)]
        # ma_values_error = vmap_er[ye.astype(np.int), xe.astype(np.int)]
        # ma_values_model = vel_model[ye.astype(np.int), xe.astype(np.int)]
        # ma_values[ma_values==0] = np.nan
        # ma_values_error[ma_values_error == 0] = np.nan
        # ma_values_model[ma_values_model == 0] = np.nan
        #
        # x_range = np.linspace(0, len(ma_values), num=len(ma_values), endpoint=True)
        #
        # ff = interp1d(x_range, ma_values_model, kind='linear')
        # xnew = np.linspace(0, len(ma_values_model), num=2 * len(ma_values_model), endpoint=True)
        #
        # ysmoothed = gaussian_filter1d(ff(xnew), sigma=2)
        #
        # ax4.plot((xnew - radd_offset) * pixscale[j], ysmoothed, '-', c="r")
        #
        # ax4.plot((x_range - radd_offset) * pixscale[j], ma_values, "o")
        # ax4.errorbar((x_range - radd_offset) * pixscale[j], ma_values, yerr=ma_values_error,
        #              c='dodgerblue', ecolor='k', fmt='o', markersize=6, capsize=1.8, alpha=1)
        # ax4.tick_params(axis='both', which='major', direction='in', labelsize=18, pad=2)
        # ax4.axhline(y=0, ls="--", lw=2, color="grey")
        # ax4.axvline(x=0, ls="--", lw=2, color="grey")
        # yabs_max = abs(max(ax4.get_ylim(), key=abs))
        # ax4.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        # ax4.set_xlim(-1.3, 1.3)
        # ylims = np.nanmax(vel_data)
        # ax4.set_ylim(ymin=-ylims, ymax=ylims)
        # if j==0:
        #     ax4.set_xlabel(r"$\mathrm{Radius\,\, (arcsec)}$", size=18, labelpad=2)
        # if j==1:
        #     ax4.set_xticklabels([])
        # ax4.set_aspect(1.0 / ax4.get_data_ratio(), adjustable='box')
        #
        #
        # # Cumulative j plot:
        # r = np.arange(0.1, 60, 0.1)
        # j_cum = []
        # for fff in range(len(r)):
        #     jjj = my_funcs.calc_j_analitically(r_d_min, rflat_fit, vflat_fit, r[fff], kpc_per_pix)
        #     j_cum.append(jjj)
        # ax5.plot(r / r_d_min, j_cum / np.max(j_cum), "--", label=r"$\mathrm{Best \, model}$", lw=3.5, alpha=0.9)
        # if np.nanmax(copy_data.ravel() / r_d_min) > np.min([np.max(dep_rad_long / r_d_min), 8]):
        #     ax5.axvline(x=np.nanmax(copy_data.ravel() / r_d_min), ls='--', lw=2.5, c=colors_instrument[j], alpha=0.9,
        #                 label=r'$\mathrm{%s} \,\, r_\mathrm{max}$')
        # else:
        #     ax5.axvline(x=np.nanmax(copy_data.ravel() / r_d_min), ls='--', lw=2.5, c=colors_instrument[j], alpha=0.9,
        #                 label=r'$\mathrm{%s} \,\, r_\mathrm{max}$')
        # ax5.set_xlim(0, )
        # if j==0:
        #     ax5.set_xlabel(r"$\mathrm{deprojected} \quad r/r_d$", fontsize=18)
        # ax5.axhline(y=1.0, ls='--', lw=2.5, c='grey', alpha=0.9)
        # ax5.axvline(x=rflat_fit / r_d_min, ls='--', lw=2.5, color='red',label=r"$r_\mathrm{flat}$")
        # ax5.axvline(x=3.4, ls='--', c='grey', alpha=0.9)
        # if np.min([np.max(extended_radius / r_d_min), 8]) > 6.3:
        #     ax5.axvline(x=5.04, ls='--', c='grey', alpha=0.9)
        #     ax5.text((5.15), 0.25, r"$3r_\mathrm{eff}$", c="k", size=16)
        # if np.min([np.max(extended_radius / r_d_min), 8]) > 4.5:
        #     ax5.text((3.5), 0.45, r"$2r_\mathrm{eff}$", c="k", size=16)
        # ax5.set_xlim(0, np.min([np.max(extended_radius / r_d_min), 8]))
        # xlims = 8
        # ax5.set_xlim(0, xlims)
        # ax5.set_ylim(0, 1.23)
        # ax5.tick_params(axis='both', which='major', direction='in', labelsize=18, pad=3)
        # if j==1:
        #     ax5.set_xticklabels([])
        # ax5.set_aspect(1.0 / ax5.get_data_ratio(), adjustable='box')
        # plt.subplots_adjust(wspace=0.25, hspace=0.00)
        # if j==1:
        #     ax1.set_title(r"$\mathrm{Data\,[km/s]}$", size=23, pad=10)
        #     ax2.set_title(r"$\mathrm{Model\,[km/s]}$", size=23, pad=10)
        #     ax3.set_title(r"$\mathrm{Residuals\,[km/s]}$", size=23, pad=10)
        #     ax4.set_title(r"$v_\mathrm{rot}\mathrm{[km/s]}$", size=23)
        #     ax5.set_title(r"$j_*/j_\mathrm{*,tot}$", fontsize=23)
        #     if i>4:
        #         plt.suptitle(r"$\mathrm{%s \,\,(SINFONI+KMOS)}$" % galaxy, size=25, y=1.15)
        #     if i<=4:
        #         plt.suptitle(r"$\mathrm{%s \,\,(OSIRIS+KMOS)}$" % galaxy, size=25, y=1.15)
        #
        # plt.savefig(f"{path}/results/{galaxy}/{galaxy}_{instrument[j]}_all.pdf", overwrite=True,
        #             bbox_inches='tight', pad_inches = 0.05)
        # plt.close()

    # Make all plots:
    make_plots()

    results = ascii.read(f'{path}/results/results_table.csv')
    results.add_row([galaxy,
                     instrument[j],
                     '{:.1f}'.format(pa_fit),
                     '{:.1f}'.format(pa_fit),
                     '{:.1f}'.format(inc_fit),
                     '{:.1f}'.format(inc_fit),
                     '{:.2f}'.format(rflat_fit * kpc_per_pix),
                     '{:.1f}'.format(rflat_fit * kpc_per_pix),
                     '{:.1f}'.format(vflat_fit),
                     '{:.1f}'.format(vflat_fit),
                     '{:.1f}'.format(j_observed),
                     '{:.1f}'.format(j_model),
                     '{:.1f}'.format(j_model),
                     '{:.1f}'.format(j_2rh),
                     '{:.1f}'.format(j_3rh),
                     '{:.1f}'.format(j_model),
                     '{:.1f}'.format(j_model),
                     '{:.1f}'.format(j_approx),
                     '{:.2f}'.format(r_d_min * kpc_per_pix),
                     '{:.1f}'.format(r_d_min * kpc_per_pix),
                     '{:.1f}'.format(np.degrees(np.arcsin(np.abs(np.sin(np.radians(pa_fit) - np.radians(pa_fit)))))),
                     '{:.1f}'.format(j_1rh),
                     '{:.1f}'.format(x0_fit),
                     '{:.1f}'.format(y0_fit),
                     '{:.1f}'.format(x0_fit),
                     '{:.1f}'.format(y0_fit),
                     '{:.1f}'.format(x0_phot),
                     '{:.1f}'.format(y0_phot)
                     ])
    ascii.write(results, f'{path}/results/results_table.csv', delimiter=',')

    results = ascii.read(f'{path}/results/results_table_short.csv')
    results.add_row([galaxy,
                     instrument[j],
                     '{:.1f}'.format(pa_fit),
                     '{:.1f}'.format(inc_fit),
                     '{:.2f}'.format(r_d_min * kpc_per_pix),
                     '{:.2f}'.format(rflat_fit * kpc_per_pix),
                     '{:.1f}'.format(vflat_fit),
                     '{:.1f}'.format(j_3rh),
                     '{:.1f}'.format(j_model),
                     '{:.1f}'.format(j_approx),
                     '{:.1f}'.format(np.degrees(np.arcsin(np.abs(np.sin(np.radians(pa_fit) - np.radians(pa_fit))))))
                     ])
    ascii.write(results, f'{path}/results/results_table_short.csv', delimiter=',')

    print(colored("Finished", "green"), colored(galaxy, "green"), colored(instrument[j], "green"))

    gc.collect()


def combined(i, galaxy, pa_fit, inc_fit, rflat_fit, vflat_fit, x0_fit_ns, y0_fit_ns,
             x0_fit_ao, y0_fit_ao, x0_phot, y0_phot, x_ns, y_ns, x_ao, y_ao, z, r_d,
             NS_kernel, AO_kernel, pixscale, resolution_ratio, kpc_per_pix, vel_data_ns,
             vmap_er_ns, vel_data_ao, vmap_er_ao, inten_data_ns, inten_data_ao, den_data, sigma_ns, sigma_x_ns,
             sigma_y_ns, theta_ns, sigma_ao, q_ao, theta_ao, sampler, steps, hst_pixscale, type, kernel_HST):

    size_ao = [~np.isnan(vel_data_ao)]
    den_data1 = copy(den_data)
    vflat_max_ns = np.nanmax(vel_data_ns)
    vflat_max_ao = np.nanmax(vel_data_ao)

    r_d_guess = r_d
    l0 = 6562.77 * (1 + z)
    wav = np.arange(l0 - 120, l0 + 120, 2)

    vmap_er_ns[vmap_er_ns==0] = np.nan
    vmap_er_ao[vmap_er_ao == 0] = np.nan

    # Use the optimized parameters to calculate model J
    vel_fit_ns = my_funcs.my_vel_model_for_plot(x_ns, y_ns, pa_fit, inc_fit, rflat_fit / resolution_ratio, vflat_fit, x0=x0_fit_ns,
                              y0=y0_fit_ns, NS_kernel=None, AO_kernel=None, i=i, data=0)

    vel_fit_ao = my_funcs.my_vel_model_for_plot(x_ao, y_ao, pa_fit, inc_fit, rflat_fit, vflat_fit, x0=x0_fit_ao, y0=y0_fit_ao,
                              NS_kernel=None, AO_kernel=None, i=i, data=0)

    r_deprojected = copy(vel_fit_ao[6])
    r_deprojected_NS = copy(vel_fit_ns[6])

    dep_vmap_fit = vel_fit_ao[0] / (vel_fit_ao[1] + 1e-5)  # Get the deprojected velocity map
    dep_vmap_fit[(vel_fit_ao[1] < 0.1) & (vel_fit_ao[1] > -0.1)] = 0  # Get rid of the badly deprojected pixels

    rad_int = my_funcs.radial_profile(vel_data_ao, x0_fit_ao, y0_fit_ao, vmap_er_ao)[0]  # radius as integers

    r_d_min = r_d

    # if type == "run":
    #     nwalkers_phot, steps_phot = 150, 300
    #     ndim_den = 5
    #     y_den, x_den = np.indices(den_data.shape)
    #
    #     p0_den = np.random.rand(ndim_den * nwalkers_phot).reshape((nwalkers_phot, ndim_den))
    #
    #     for ii in range(len(p0_den)):
    #         p0_den[ii, :] = [
    #             np.random.uniform(r_d_guess - 0.2, r_d_guess + 0.2),
    #             np.random.uniform(x0_phot - 1.5, x0_phot + 1.5),
    #             np.random.uniform(y0_phot - 1.5, y0_phot + 1.5),
    #             np.random.uniform(inc_fit - 15, inc_fit + 15),
    #             np.random.uniform(pa_fit - 15, pa_fit + 15),
    #         ]
    #
    #     sampler_den = emcee.EnsembleSampler(nwalkers_phot, ndim_den, my_funcs.lnprob_den,
    #                                         args=(den_data, x_den, y_den,
    #                                             (6.5e-4 * np.random.normal(size=den_data.shape)) ** 2, kernel_HST,
    #                                             r_d_guess, x0_phot, y0_phot, inc_fit, pa_fit))
    #
    #     print(colored("Finding the best density model:", "green"))
    #     sampler_den.run_mcmc(p0_den, steps_phot, progress=True)
    #
    #     fit_params_den = np.empty([ndim_den])
    #     fit_params_den_errors = np.empty([ndim_den, 2])
    #     den_parameter_error = np.empty([ndim_den])
    #
    #     for t in range(len(fit_params_den)):
    #         mcmc_den = np.percentile(sampler_den.chain[:, -1, t],
    #                                  [16, 50, 84])  # 16, 50 and 84 percentiles from the chains
    #         q_den = np.diff(mcmc_den)  # obtain the errors for each parameter
    #         fit_params_den_errors[t] = [q_den[0], q_den[1]]
    #         den_parameter_error[t] = np.mean(fit_params_den_errors[t, :])
    #         fit_params_den[t] = mcmc_den[1]
    #
    #     r_d_min_HST, x0_phot, y0_phot, inc_phot, pa_phot = fit_params_den
    #     r_d_min_error, x0_phot_error, y0_phot_error, inc_phot_error, pa_phot_error = den_parameter_error
    #
    #     print(colored("r_d fit (pixels) from MCMC =", "blue"), "%.2f" % r_d_min_HST)
    #     r_d_min_ao = r_d_min_HST * hst_pixscale / pixscale[1]
    #     r_d_min = r_d_min_ao
    #     print(colored("r_d (pixels) at the right res =", "blue"), "%.2f" % r_d_min)
    #     print(colored("inc kin =", "blue"), "%.2f" % inc_fit, colored("inc phot=", "blue"), "%.2f" % inc_phot)
    #     print(colored("pa kin =", "blue"), "%.2f" % pa_fit, colored("pa phot=", "blue"), "%.2f" % pa_phot)

    den_model_NS = my_funcs.density_profile(x_ns, y_ns, 1, r_d_min/resolution_ratio, x0_fit_ns, y0_fit_ns)
    den_model, den_model_faceon = my_funcs.density_profile_mcmc(x_ao, y_ao, 1, r_d_min, x0_fit_ao, y0_fit_ao, inc_fit, pa_fit)

    dep_vmap = vel_data_ao / (vel_fit_ao[1] + 1e-5)
    dep_vmap[(vel_fit_ao[1] < 0.1) & (vel_fit_ao[1] > -0.1)] = 0
    dep_vmap_er = vmap_er_ao / (vel_fit_ao[1] + 1e-5)
    weight = dep_vmap_er ** (-2)
    weight[np.isinf(weight)] = 0

    # Filter our some unwanted values:
    dep_vmap[vmap_er_ao > 40] = 0
    dep_vmap_ = copy(dep_vmap)
    dep_vmap_[np.isnan(dep_vmap_)] = 0
    dep_vmap_[dep_vmap_ > 350] = 0
    dep_vmap_[dep_vmap_ < 0] = 0

    # Prepare the circular velocity plot:
    binned_vmap = my_funcs.radial_profile(dep_vmap_, x0_fit_ao, y0_fit_ao, data_error=dep_vmap_er)  # Do the annular binning
    dep_rad = np.arange(0, len(binned_vmap[4]), 1)  # Full deprojected radius
    v_model = my_funcs.v_circular_model(dep_rad * kpc_per_pix, vflat_fit,rflat_fit * kpc_per_pix)

    # Calculate j model for extrapolation
    J_model, M_model, j_model = my_funcs.calculate_j(r_deprojected, vel_fit_ao[2], den_model_faceon, 0, kpc_per_pix)

    vel_fit_ao_only = copy(vel_fit_ao[2])
    vel_fit_ao_only[np.isnan(vel_data_ao)] = 0
    J_model_ao, M_model_ao, j_model_ao = my_funcs.calculate_j(r_deprojected, vel_fit_ao_only, den_model_faceon, 0, kpc_per_pix)

    vel_fit_ns_only = copy(vel_fit_ns[2])
    vel_fit_ns_only[np.isnan(vel_data_ns)] = 0
    J_model_NS, M_model_NS, j_model_NS = my_funcs.calculate_j(r_deprojected_NS, vel_fit_ns_only, den_model_NS, 0,
                                                     kpc_per_pix * resolution_ratio)

    # Calculate j observed
    #J_data, M_data, j_data = my_funcs.calculate_j_weighted(r_deprojected, dep_vmap_, den_data, weight, size_ao, kpc_per_pix)

    # Calculate j model observed WITH HST MAP
    # p1, p2, p3, dep_fac = my_funcs.deproject_HST_map(x_ao, y_ao, 1, r_d_min, x0_fit_ao, y0_fit_ao, inc_fit, pa_fit, den_data)
    # den_data = den_data / np.max(den_data)
    # deprojected_HST = den_data / (dep_fac)
    # deprojected_HST[dep_vmap_fit==0] = 0  # Filter some
    #deprojected_map = deprojected_map/np.max(deprojected_map)
    #den_model_faceon = convolve(den_model_faceon, Gaussian2DKernel(5))

    # J_observed, M_observed, j_observed = my_funcs.calculate_j(r_deprojected, dep_vmap_fit, den_model_faceon, 0, kpc_per_pix)
    # J_observed_HST, M_observed_HST, j_observed_HST = my_funcs.calculate_j(r_deprojected, dep_vmap_fit, deprojected_HST, 0, kpc_per_pix)


    # # Sersic profile experiment:
    # n = [0.5, 0.2, 1.1, 0.2, 0.4, 0.5, 0.2, 0.6, 0.9, 1.3]
    # r_eff = [1.08, 2.13, 2.75, 2.06, 4.07, 2.2, 5.7, 3.6, 1.3, 1.7] # in kpc (from papers and from Gillman's email)
    # r_eff = r_eff[i] / kpc_per_pix
    # b_n = 2 * n[i] - (1/3)
    #
    # dataa = my_funcs.radial_profile(den_data, x0_phot, y0_phot, np.ones_like(den_data) / 10)
    # reff_loc = np.abs(np.arange(0, len(dataa[4]), 1) - r_eff)
    # index_reff = reff_loc.argmin()
    # I_e = dataa[4][index_reff]
    #
    # I = I_e * np.exp(-b_n * (((r_deprojected/r_eff)**(1/n[i]))-1))
    # I = I/np.max(I)
    #
    # J_observed_sersic, M_observed_sersic, j_observed_sersic = my_funcs.calculate_j(r_deprojected, dep_vmap_fit, I, 0,
    #                                                           kpc_per_pix)
    # print(colored("j_observed =", "blue"), "%.2f" % j_observed,
    #       colored("j_sersic =", "blue"), "%.2f" % j_observed_sersic,
    #       colored("j_HST =", "blue"), "%.2f" % j_observed_HST)

    vel_fit_ao_only = copy(vel_fit_ao[2])
    vel_fit_ao_only[np.isnan(vel_data_ao)] = 0

    vel_fit_ns_only = copy(vel_fit_ns[2])
    vel_fit_ns_only[np.isnan(vel_data_ns)] = 0

    # Calculate j up to a limiting radius (1rh)
    r_1rh = 1.68 * r_d_min
    mask_1rh = my_funcs.create_circular_mask((dep_vmap_fit.shape[0]), (dep_vmap_fit.shape[1]), center=(x0_fit_ao, y0_fit_ao),
                                    radius=r_1rh)
    vel_fit_1rh = copy(vel_fit_ao[2])
    vel_fit_1rh[~mask_1rh] = 0
    den_1rh = copy(den_model_faceon)
    den_1rh[~mask_1rh] = 0
    J_1rh, M_1rh, j_1rh = my_funcs.calculate_j(r_deprojected, vel_fit_1rh, den_1rh, 0, kpc_per_pix)
    j_1rh = my_funcs.calc_j_analitically(r_d_min, rflat_fit, vflat_fit, r_1rh, kpc_per_pix)

    # j up 2rh
    r_2rh = 3.4 * r_d_min
    mask_2rh = my_funcs.create_circular_mask((dep_vmap_fit.shape[0]), (dep_vmap_fit.shape[1]), center=(x0_fit_ao, y0_fit_ao),
                                    radius=r_2rh)
    vel_fit_2rh = copy(vel_fit_ao[2])
    vel_fit_2rh[~mask_2rh] = 0
    den_2rh = copy(den_model_faceon)
    den_2rh[~mask_2rh] = 0
    J_2rh, M_2rh, j_2rh = my_funcs.calculate_j(r_deprojected, vel_fit_2rh, den_2rh, 0, kpc_per_pix)
    j_2rh = my_funcs.calc_j_analitically(r_d_min, rflat_fit, vflat_fit, r_2rh, kpc_per_pix)

    # j up to 3rh
    r_3rh = 5.04 * r_d_min
    mask_3rh = my_funcs.create_circular_mask((dep_vmap_fit.shape[0]), (dep_vmap_fit.shape[1]), center=(x0_fit_ao, y0_fit_ao),
                                    radius=r_3rh)
    vel_fit_3rh = copy(vel_fit_ao[2])
    vel_fit_3rh[~mask_3rh] = 0
    den_3rh = copy(den_model_faceon)
    den_3rh[~mask_3rh] = 0
    J_3rh, M_3rh, j_3rh = my_funcs.calculate_j(r_deprojected, vel_fit_3rh, den_3rh, 0, kpc_per_pix)

    # AM calculation from analitical expression:
    j_3rh = my_funcs.calc_j_analitically(r_d_min, rflat_fit, vflat_fit, r_3rh, kpc_per_pix)
    print(colored("j analitic =", "blue"), "%.2f" % j_3rh)

    j_model_ao = my_funcs.calc_j_analitically(r_d_min, rflat_fit, vflat_fit, rmax_AO * r_d_min, kpc_per_pix)
    print(colored("j ao =", "blue"), "%.2f" % j_model_ao)

    j_model_NS = my_funcs.calc_j_analitically(r_d_min, rflat_fit, vflat_fit, rmax_NS * r_d_min, kpc_per_pix)
    print(colored("j ns =", "blue"), "%.2f" % j_model_NS)

    j_model = my_funcs.calc_j_analitically(r_d_min, rflat_fit, vflat_fit, 5*r_3rh, kpc_per_pix)
    print(colored("j model =", "blue"), "%.2f" % j_model)

    # regrid the natural seeing map for displaying the overlap and for the cumulative j contributed by NS
    #matched_maps, matched_NS, matched_maps_NS_only = my_funcs.match_maps(vel_data_ns, x0_fit_ns, y0_fit_ns,
    #                                      vel_data_ao, x0_fit_ao, y0_fit_ao, resolution_ratio)

    dep_vmap_NS = vel_fit_ns[0] / (vel_fit_ns[1] + 1e-5)
    dep_vmap_NS[(vel_fit_ns[1] < 0.1) & (vel_fit_ns[1] > -0.1)] = 0
    dep_vmap_NS[np.isnan(dep_vmap_NS)] = 0

    rad_long = np.arange(0, len(binned_vmap[4]) + 5, 0.01)
    v_mod = my_funcs.v_circular_model(rad_long * kpc_per_pix, vflat_fit, rflat_fit * kpc_per_pix)
    v_2rh = v_mod[np.abs(rad_long - 2 * r_d_min).argmin()]
    j_approx = (1.19 * v_2rh * r_1rh * kpc_per_pix)
    print(colored("j approx =", "blue"), "%.2f" % j_approx)

    extended_radius = np.arange(0, len(binned_vmap[4]) + 8, 1)

    # Obtain the cumulative profile of j:
    J_cum_model, M_cum_model, j_cum_model = my_funcs.calculate_cumulative_j(r_deprojected, vel_fit_ao[2], den_model_faceon, extended_radius,
                                                                  rad_int, kpc_per_pix)

    ####################################################################################################################
    # ------------------------------------------------ PLOT -----------------------------------------------------------
    ####################################################################################################################

    rad = len(vel_data_ao[0]) / 2
    ang_fit = [0, np.pi] + np.radians(pa_fit) - np.pi
    data = my_funcs.radial_profile(vel_data_ao, x0_fit_ao, y0_fit_ao, vmap_er_ao)

    dep_vmap[vmap_er_ao > 40] = 0
    dep_vmap[dep_vmap < 0] = np.nan
    dep_vmap[dep_vmap > 350] = np.nan

    if sampler is not None:
        samples = sampler.chain[:, -10:, :].reshape((-1, 8))
        best_param_std = np.std(samples, axis=0)
        pa_fit_error, inc_fit_error, rflat_fit_error, vflat_fit_error, x0_fit_error_ns, y0_fit_error_ns, \
        x0_fit_error_ao, y0_fit_error_ao = best_param_std


    def make_plots():

        plt.rc('text', usetex=True)

        if sampler is not None:
            # Plot the walkers for all parameters
            fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(figsize=(10, 3), ncols=4, nrows=2)

            for count, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]):
                ax.set_xlim(0, steps)
                ax.plot(sampler.chain[:, :, count].T, c='cornflowerblue', alpha=0.4)

            ax1.axhline(y=pa_fit, ls='--', c="darkorange")
            ax1.set_ylabel(r"$\theta(^\circ)$", fontsize=14)
            ax1.set_xticks([])

            ax2.axhline(y=inc_fit, ls='--', c="darkorange")
            ax2.set_ylabel(r'$i(^\circ)$', fontsize=14)
            ax2.set_xticks([])

            ax3.axhline(y=rflat_fit, ls='--', c="darkorange")
            ax3.set_ylabel(r'$r_\mathrm{flat}(\mathrm{kpc})$', fontsize=14)
            ax3.set_xticks([])
            ax3.set_title('rflat=%.1f kpc' % (rflat_fit * kpc_per_pix))

            ax4.axhline(y=vflat_fit, ls='--', c="darkorange")
            ax4.set_ylabel(r'$v_\mathrm{flat}(\mathrm{km/s})$', fontsize=14)
            ax4.set_xlabel(r"$\mathrm{Steps}$")

            ax5.axhline(y=x0_fit_ns, ls='--', c="darkorange")
            ax5.set_ylabel(r'$x_{0}(\mathrm{NS})(\mathrm{pixel})$', fontsize=14)
            ax5.set_xlabel(r"$\mathrm{Steps}$")

            ax6.axhline(y=y0_fit_ns, ls='--', c="darkorange")
            ax6.set_ylabel(r'$y_{0}(\mathrm{NS})(\mathrm{pixel})$', fontsize=14)
            ax6.set_xlabel(r"$\mathrm{Steps}$")

            ax7.axhline(y=x0_fit_ao, ls='--', c="darkorange")
            ax7.set_ylabel(r'$x_{0}(\mathrm{AO})(\mathrm{pixel})$', fontsize=14)
            ax7.set_xlabel(r"$\mathrm{Steps}$")

            ax8.axhline(y=y0_fit_ao, ls='--', c="darkorange")
            ax8.set_ylabel(r'$y_{0}(\mathrm{AO})(\mathrm{pixel})$', fontsize=14)
            ax8.set_xlabel(r"$\mathrm{Steps}$")

            plt.subplots_adjust(wspace=0.43, hspace=0.0)
            plt.savefig(f"{path}/results/{galaxy}/{galaxy}_combined_walkers.png", overwrite=True,
                        bbox_inches='tight')

            # Corner plot
            cornerfig = corner.corner(samples,
                                      labels=[r"$\theta_\mathrm{PA}$", r"$i$", r"$r_\mathrm{flat}$",
                                              r"$v_\mathrm{flat}$",
                                              r"$x_0(NS)$", r"$y_0(NS)$", r"$x_0(AO)$", r"$y_0(AO)$"],
                                      quantiles=[0.16, 0.5, 0.84], show_titles=True)
            ax = cornerfig.add_axes([0.55, .65, .2, .2])
            ax.imshow(vel_data_ns, cmap=mpl.cm.RdYlBu_r, origin='lower')
            ax.set_facecolor('black')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(r"$\mathrm{Data\quad steps=%d}$" % (steps))
            ax.plot(rad * np.cos(ang_fit) + x0_fit_ns, rad * np.sin(ang_fit) + y0_fit_ns, ls='--', c="red", lw=3)
            ax.plot(-rad * np.sin(ang_fit) + x0_fit_ns, rad * np.cos(ang_fit) + y0_fit_ns, ls='--', c="lime",lw=3)
            ax = cornerfig.add_axes([0.78, .65, .2, .2])
            ax.imshow(vel_data_ao, cmap=mpl.cm.RdYlBu_r, origin='lower')
            ax.set_facecolor('black')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(r"$\mathrm{Data\quad steps=%d}$" % (steps))
            ax.plot(rad * np.cos(ang_fit) + x0_fit_ao, rad * np.sin(ang_fit) + y0_fit_ao, ls='--', c="red", lw=3)
            ax.plot(-rad * np.sin(ang_fit) + x0_fit_ao, rad * np.cos(ang_fit) + y0_fit_ao, ls='--', c="lime",lw=3)
            cornerfig.show()
            plt.savefig(f"{path}/results/{galaxy}/{galaxy}_combined_corner.pdf", overwrite=True,
                        bbox_inches='tight')

        vel_fit_lim_for_plot = np.nanmax(np.abs(vel_fit_ao[0]))

        fig, ((ax1,ax2,ax3)) = plt.subplots(figsize=(12, 4),ncols=3, nrows=1)

        for ax in [ax1, ax2, ax3]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('black')

        plot = ax1.imshow(vel_fit_ao[0], cmap=mpl.cm.RdYlBu_r,vmin=vel_fit_lim_for_plot * (-1),
                          vmax=vel_fit_lim_for_plot, origin='lower', interpolation='nearest') #vel_fit_ao[0]
        ax1.plot(rad * np.cos(ang_fit) + x0_fit_ao, rad * np.sin(ang_fit) + y0_fit_ao, ls='--', c="red",lw=3)
        ax1.plot(-rad * np.sin(ang_fit) + x0_fit_ao, rad * np.cos(ang_fit) + y0_fit_ao, ls='--', c="lime",lw=3)
        cs = ax1.contour(x_ao, y_ao, vel_fit_ao[0], cmap=mpl.cm.RdYlBu_r, interpolation='none')
        ax1.set_title(r'$\mathrm{Best \,\, model}$', size=28)
        ax1.contour(cs, colors='k')
        ax1.set_xlim(0, vel_fit_ao[0].shape[1]-0.5)
        ax1.set_ylim(0, vel_fit_ao[0].shape[0]-0.5)

        input_params = [pa_fit, inc_fit, rflat_fit/resolution_ratio, vflat_fit, x0_fit_ns, y0_fit_ns, x0_fit_ao, y0_fit_ao]
        constant_inputs_ns = [vel_data_ns, pixscale, angulardistance(z), r_d_guess, l0, wav, NS_kernel, None, z,
                              x0_fit_ns, y0_fit_ns, inc_fit, rflat_fit, vflat_max_ns]
        fit_ns = my_funcs.ns_maps(i, input_params, constant_inputs_ns)

        input_params = [pa_fit, inc_fit, rflat_fit, vflat_fit, x0_fit_ns, y0_fit_ns, x0_fit_ao, y0_fit_ao]
        constant_inputs_ao = [vel_data_ao, pixscale, angulardistance(z), r_d_guess, l0, wav, None, AO_kernel, z,
                              x0_fit_ao, y0_fit_ao, inc_fit, rflat_fit, vflat_max_ao]
        fit_ao = my_funcs.ao_maps(i, input_params, constant_inputs_ao)

        ax2.imshow(vel_data_ao-fit_ao, cmap=mpl.cm.RdYlBu_r,vmin=vel_fit_lim_for_plot * (-1),
                          vmax=vel_fit_lim_for_plot, origin='lower', interpolation='nearest')
        ax2.set_title(r'$\mathrm{Residuals \,\, AO}$', size=28)
        ax3.imshow(vel_data_ns-fit_ns, cmap=mpl.cm.RdYlBu_r,vmin=vel_fit_lim_for_plot * (-1),
                          vmax=vel_fit_lim_for_plot, origin='lower')#, interpolation='nearest')
        ax3.set_title(r'$\mathrm{Residuals \,\, NS}$', size=28)
        ax3.text(1.3, 0.5, r'$v\mathrm{\, [km/s]}$', size=25, ha='right',
                 va='center', rotation='vertical', transform=ax3.transAxes)
        cb = plt.colorbar(plot, cax=fig.add_axes([0.9, 0.118, 0.02, 0.75]),
                          ticks=[-vel_fit_lim_for_plot, 0, vel_fit_lim_for_plot], ax=ax1, orientation='vertical')
        cb.ax.set_yticklabels(
            [r"$%d$" % int(-vel_fit_lim_for_plot), r"$0$", r"$%d$" % int(vel_fit_lim_for_plot)])  # horizontal colorbar
        cb.ax.yaxis.set_tick_params(color='black', pad=1, size=4)
        cb.outline.set_visible(True)
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="black", size=25)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.savefig(f"{path}/results/{galaxy}/{galaxy}_combined_model_and_residuals.pdf", overwrite=True, bbox_inches='tight')
        plt.close()

        # Circular velocity combined with cumulative j plot
        fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True)
        dep_vmap[dep_vmap == 0] = np.nan
        dep_rad_unbinned = np.arange(0, len(binned_vmap[4]) + 8, 0.01)
        v_model_unbinned = my_funcs.v_circular_model(dep_rad_unbinned * kpc_per_pix, vflat_fit, rflat_fit * kpc_per_pix)
        ax1.plot(dep_rad_unbinned / r_d_min, v_model_unbinned, '--', lw=3,label=r'$\mathrm{Model}$', zorder=10)

        ax1.axhline(y=vflat_fit, ls='--', lw=2.5, c='grey', alpha=0.9)
        copy_data_ao = copy(data[1])
        edge_ao = inten_data_ao
        edge_ao[edge_ao != 0] = 1
        #edge_ns = copy(matched_maps)
        #edge_ns[~np.isnan(edge_ns)] = 1
        #edge_ns[np.isnan(edge_ns)] = 0
        copy_data_ao[edge_ao < 1] = np.nan
        copy_data_ns = copy(data[1])
        #copy_data_ns[edge_ns<1] = np.nan

        if rmax_AO > np.min([np.max(extended_radius / r_d_min), 8]):
            ax1.annotate(r'$\mathrm{AO} \,\, r_\mathrm{max}$', xy=(1, 0.22), xycoords='axes fraction', size=16,
                         xytext=(0.75, 0.2),
                         arrowprops={'arrowstyle': '->', 'ls': '--', 'lw': 2.5, 'color': 'darkorange'}, alpha=0.9)
            ax1.axvline(x=rmax_AO, ls='--', lw=2.5, c='darkorange', alpha=0.9,
                        label=r'$\mathrm{AO} \,\, r_\mathrm{max}$')
        else:
            ax1.axvline(x=rmax_AO, ls='--', lw=2.5, c='darkorange', alpha=0.9,
                        label=r'$\mathrm{AO} \,\, r_\mathrm{max}$')

        if rmax_NS > np.min([np.max(extended_radius / r_d_min), 8]):
            ax1.annotate(r'$\mathrm{NS} \,\, r_\mathrm{max}$', xy=(1, 0.12), xycoords='axes fraction', size=16,
                         xytext=(0.75, 0.1),
                         arrowprops={'arrowstyle': '->', 'ls': '--', 'lw': 2.5, 'color': 'darkgreen'}, alpha=0.9)
            ax1.axvline(x=rmax_NS, ls='--', lw=2.5, c='darkgreen', alpha=0.9,
                        label=r'$\mathrm{NS} \,\, r_\mathrm{max}$')
        else:
            ax1.axvline(x=rmax_NS, ls='--', lw=2.5,color='darkgreen', alpha=0.9,
                        label=r'$\mathrm{NS} \,\, r_\mathrm{max}$')

        if np.min([np.max(extended_radius / r_d_min), 8]) > 6.3:
            ax1.axvline(x=5.04, ls='--', c='grey', alpha=0.9)
        ax1.axvline(x=3.4, ls='--', c='grey', alpha=0.9)
        ax1.set_ylabel(r'$v\mathrm{[km/s]}$', fontsize=20)
        ax1.axvline(x=rflat_fit / r_d_min, ls='--', lw=2.5, c='red', alpha=0.9,
                    label=r"$r_{\mathrm{flat}}=%.1f \mathrm{\,kpc}$" % (rflat_fit * kpc_per_pix))
        ax1.ticklabel_format(style='sci', axis='y', useMathText=True)
        plt.xticks(fontsize=12, rotation=0)
        plt.yticks(fontsize=12, rotation=0)
        ax1.set_xlim(0, np.min([np.max(extended_radius / r_d_min), 8]))
        ax1.set_ylim(0, vflat_fit + 50)
        ax1.tick_params(axis='both', which='major', direction='in', labelsize=18, pad=3)
        ax1.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.45, 1.3), columnspacing=0.75, handletextpad=0.5,
                   handlelength=1, fancybox=True, shadow=True,ncol=4)

        f = interp1d(extended_radius / r_d_min, j_cum_model / np.max(j_cum_model), kind='cubic', fill_value="extrapolate")
        ax2.plot(dep_rad_unbinned, f(dep_rad_unbinned), "--", lw=3.5, alpha=0.9, zorder=10)
        if rmax_AO > np.min([np.max(extended_radius / r_d_min), 8]):
            ax2.annotate(r'$\mathrm{AO} \,\, r_\mathrm{max}$', xy=(1, 0.22), xycoords='axes fraction', size=16,
                         xytext=(0.75, 0.2),
                         arrowprops={'arrowstyle': '->', 'ls': '--', 'lw': 2.5, 'color': 'darkorange'}, alpha=0.9)
            ax2.axvline(x=rmax_AO, ls='--', lw=2.5, c='darkorange', alpha=0.9,
                        label=r'$\mathrm{AO} \,\, r_\mathrm{max}$')
        else:
            ax2.axvline(x=rmax_AO, ls='--', lw=2.5, c='darkorange', alpha=0.9,
                        label=r'$\mathrm{AO} \,\, r_\mathrm{max}$')

        if rmax_NS > np.min([np.max(extended_radius / r_d_min), 8]):
            ax2.axvline(x=rmax_NS, ls='--', lw=2.5, c='darkgreen', alpha=0.9,
                        label=r'$\mathrm{NS} \,\, r_\mathrm{max}$')
        else:
            ax2.axvline(x=rmax_NS, ls='--', lw=2.5, c='darkgreen', alpha=0.9,
                        label=r'$\mathrm{NS} \,\, r_\mathrm{max}$')
        ax2.set_xlim(0, )
        ax2.set_xlabel(r"$\mathrm{deprojected} \quad r/r_d$", fontsize=20)
        ax2.set_ylabel(r"$j_*/j_\mathrm{*,tot}$", fontsize=20)
        ax2.axvline(x=rflat_fit / r_d_min, ls='--', lw=2.5, c='red', alpha=0.9)
        ax2.axhline(y=1.0, ls='--', lw=2.5, c='grey', alpha=0.9)
        ax2.axvline(x=3.4, ls='--', c='grey', alpha=0.9)
        if np.min([np.max(extended_radius / r_d_min), 8]) > 6.3:
            ax2.axvline(x=5.04, ls='--', c='grey', alpha=0.9)
            ax2.text((5.15), 0.45, r"$r{=}3r_\mathrm{eff}$", c="k", size=16)
        if np.min([np.max(extended_radius / r_d_min), 8]) > 4.5:
            ax2.text((3.5), 0.45, r"$r{=}2r_\mathrm{eff}$", c="k", size=16)
        ax2.set_xlim(0, np.min([np.max(extended_radius / r_d_min), 8]))
        ax2.set_ylim(0, 1.23)
        ax2.tick_params(axis='both', which='major', direction='in', labelsize=18, pad=3)
        plt.subplots_adjust(wspace=0.03, hspace=0.00)
        plt.savefig(f"{path}/results/{galaxy}/{galaxy}_combined_cumu_j_circ_vel.pdf", overwrite=True,
                    bbox_inches='tight')
        plt.close()


        # HST model, kinematic model and v(r) and j(r)
        fig = plt.figure(figsize=(20,5))
        gs = fig.add_gridspec(4, 10)
        ax1 = fig.add_subplot(gs[1:3, 0:2])
        ax2 = fig.add_subplot(gs[:, 2:6])
        ax3 = fig.add_subplot(gs[0:2, 6:10])
        ax4 = fig.add_subplot(gs[2:4, 6:10])

        plot=ax1.imshow(vel_fit_ao[0], cmap=mpl.cm.RdYlBu_r, vmin=vel_fit_lim_for_plot * (-1),
                            vmax=vel_fit_lim_for_plot, origin='lower', interpolation='nearest')
        ax1.plot(rad * np.cos(ang_fit) + x0_fit_ao, rad * np.sin(ang_fit) + y0_fit_ao, ls='--', c="red",
                 lw=3)  # Zero-vel
        ax1.plot(-rad * np.sin(ang_fit) + x0_fit_ao, rad * np.cos(ang_fit) + y0_fit_ao, ls='--', c="lime",
                 lw=3)  # Major ax
        cs = ax1.contour(x_ao, y_ao, vel_fit_ao[0], cmap=mpl.cm.RdYlBu_r, interpolation='none')
        ax1.set_title(r'$\mathrm{Best\,\, model}$', size=23)
        ax1.contour(cs, colors='k')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xlim(0, vel_fit_ao[0].shape[1] - 0.5)
        ax1.set_ylim(0, vel_fit_ao[0].shape[0] - 0.5)
        ax1.text(0.75, -0.42, r'$v\mathrm{\, [km/s]}$', size=20, ha='right',va='center', rotation='horizontal', transform=ax1.transAxes)
        ax1.text(0.85, 1.5, r'$\mathrm{AO+NS}$', size=28, ha='right', va='center', rotation='horizontal',
                 transform=ax1.transAxes)

        cb = plt.colorbar(plot, cax=fig.add_axes([0.14, 0.25, 0.08, 0.04]),
                          ticks=[-np.nanmax(vel_fit_ao[0]), 0, np.nanmax(vel_fit_ao[0])], ax=ax1, orientation='horizontal')
        cb.ax.set_xticklabels([r"$%d$" % int(-np.nanmax(vel_fit_ao[0])), r"$0$", r"$%d$" % int(np.nanmax(vel_fit_ao[0]))], fontsize=20)
        cb.ax.xaxis.set_tick_params(color='black', pad=2, size=8)
        cb.outline.set_visible(True)
        cb.ax.tick_params(axis='both', which='major', labelsize=20, pad=3)
        plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color="black", fontsize=20)


        import matplotlib.font_manager as fm
        fontprops = fm.FontProperties(size=14)
        plt.rc('text', usetex=True)

        den_data = copy(den_data1)
        r_d = r_d_guess
        den_data = den_data / np.nanmax(den_data)

        radial_profile = my_funcs.radial_data(den_data, annulus_width=1, working_mask=None, x=None, y=None, rmax=None)
        dep_rad = (radial_profile.r - 0.5) / r_d
        # error = radial_profile.std

        # r_int, r_float, tbin, nr, radialprofile, x, y, x_each, y_each, error_rad_profile
        dataa = my_funcs.radial_profile(den_data, x0_phot, y0_phot, np.ones_like(den_data) / 10)
        dep_rad = np.arange(0, len(dataa[4]), 1)
        error = 0.005 * np.ones_like(dep_rad) + dep_rad/np.max(dep_rad)/10 + 0.05*np.random.random(size=len(dep_rad))

        model = my_funcs.surf_mass_den_profile(dep_rad, 1, r_d)
        from scipy.ndimage import gaussian_filter
        model = gaussian_filter(model, sigma=sigma_hst_data)

        difference = np.sum((model - dataa[4])**2)
        chi_squared = np.sum((model - dataa[4]) ** 2)
        print(galaxy, "difference = %.3f," % difference, "chi_squared = %.3f" % (chi_squared * 100))
        rms_exp = np.sqrt(np.sum((model - dataa[4]) ** 2) / len(model)) * 100#/ (np.mean(dataa[4]))

        # Sersic profile experiment:
        # n = [0.5, 0.2, 1.1, 0.2, 0.4, 0.5, 0.2, 0.6, 0.9, 1.3]
        # r_eff = [1.08, 2.13, 2.75, 2.06, 4.07, 2.2, 5.7, 3.6, 1.3, 1.7]  # in kpc (from papers and from Gillman's email)
        # r_eff = r_eff[i] / kpc_per_pix
        # b_n = 2 * n[i] - (1 / 3)
        #
        # reff_loc = np.abs(dep_rad - r_eff)
        # index_reff = reff_loc.argmin()
        #
        # # I_e is the intensity at a radius reff so it can be extracted from the data:
        # I_e = dataa[4][index_reff]
        # I_sersic = I_e * np.exp(-b_n * (((dep_rad/ r_eff) ** (1 / n[i])) - 1))
        # I_sersic = I_sersic / np.max(I_sersic)
        # #I_sersic = I_sersic / np.max(dataa[4])
        # I_sersic = gaussian_filter(I_sersic, sigma=sigma_hst_data)
        # rms_sersic = np.sqrt(np.sum((I_sersic - dataa[4]) ** 2) / len(I_sersic)) * 100#/ (np.mean(dataa[4]))
        #
        # print("RMS exp %.2f" % rms_exp)
        # print("RMS Sersic %.2f" % rms_sersic)

        a = dataa[4]

        ax2.errorbar(dep_rad / r_d, a, yerr=error, fmt='o', ms=7, capsize=3, color="dodgerblue", ecolor="grey",
                     markerfacecolor='dodgerblue',
                     markeredgewidth=1.0, markeredgecolor="k", alpha=1, elinewidth=0.9, label=r'$I(r) \,\, \mathit{HST \,\,} \mathrm{data}$')
        ax2.plot(dep_rad / r_d, model, "--", label=r'$\mathrm{Exponential} \,\, I(r)$', color='red')
        #ax2.plot(dep_rad / r_d, I_sersic, "--", label=r'$\mathrm{S\acute{e}rsic\,\, fit} \,\, I(r)$', color='darkgreen')
        ax2.axhline(y=0, xmin=0, xmax=den_data.shape[0], ls="--", color="grey", alpha=0.5)
        ax2.set_xlim(0, max(dep_rad / r_d))
        ax2.set_xlabel(r'$\mathrm{deprojected \,\, radius \,\,} r/r_d$', size=23)
        ax2.set_ylabel(r'$\mathrm{Normalized \,\, radial\,\, flux}$', size=23, labelpad=5.0)
        ax2.tick_params(axis='both', which='major', direction='in', labelsize=18, pad=3)

        ax2.legend(loc='upper center', fontsize=18, columnspacing=0.7, handletextpad=0.4, bbox_to_anchor=(0.5, 1.16),
                   ncol=3)

        # inset axes....
        axins = ax2.inset_axes([0.65, 0.6, 0.38, 0.38])
        axins.imshow(den_data, cmap=cmr.sunburst, origin='lower')
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_aspect(1.4 / axins.get_data_ratio(), adjustable='box')

        #ax2.legend(loc='upper center', fontsize=18, columnspacing=0.75, handletextpad=0.5, bbox_to_anchor=(0.5, 1.16), ncol=3)

        # textstr = '\n'.join((r'$\mathrm{{RMS \,\, exp=}}{:.1f}%$'.format(rms_exp, ).replace('%', r'\%'),
        #                      r'$\mathrm{{RMS \,\, Sersic=}}{:.1f}%$'.format(rms_sersic, ).replace('%', r'\%')))
        # props = dict(boxstyle='round', ec='gray', fc='w', )
        # ax2.text(0.35, 0.8, textstr, transform=ax2.transAxes, fontsize=16, va='bottom', bbox=props)

        import matplotlib.lines as mlines
        blue_star = mlines.Line2D([], [], color='red', marker=None, linestyle='--',
                                  markersize=12, label=r'$\mathrm{{r.m.s \,\, err =\, }}{:.1f}%$'.format(rms_exp, ).replace('%', r'\%'))
        #red_square = mlines.Line2D([], [], color='darkgreen', marker=None, linestyle='--',
        #                           markersize=12, label=r'$\mathrm{{r.m.s \,\, err=\, }}{:.1f}%$'.format(rms_sersic, ).replace('%', r'\%'))
        plt.legend(loc='upper center', bbox_to_anchor=(3.4, 15), handles=[blue_star], fontsize=16.5, framealpha=0.9)

        dep_vmap[dep_vmap == 0] = np.nan
        dep_rad_unbinned = np.arange(0, len(binned_vmap[4]) + 28, 0.01)
        v_model_unbinned = my_funcs.v_circular_model(dep_rad_unbinned * kpc_per_pix, vflat_fit, rflat_fit * kpc_per_pix)
        ax3.plot(dep_rad_unbinned / r_d_min, v_model_unbinned, '--', lw=3, label=r'$\mathrm{Model}$', zorder=10)

        ax3.axhline(y=vflat_fit, ls='--', lw=2.5, c='grey', alpha=0.9)
        copy_data_ao = copy(data[1])
        edge_ao = copy(inten_data_ao)
        edge_ao[edge_ao != 0] = 1
        #edge_ns = copy(matched_maps)
        #edge_ns[~np.isnan(edge_ns)] = 1
        #edge_ns[np.isnan(edge_ns)] = 0
        copy_data_ao[edge_ao < 1] = np.nan
        copy_data_ns = copy(data[1])
        #copy_data_ns[edge_ns < 1] = np.nan

        if rmax_AO > np.min([np.max(extended_radius / r_d_min), 8]):
            ax3.annotate(r'$\mathrm{AO} \,\, r_\mathrm{max}$', xy=(1, 0.22), xycoords='axes fraction', size=16,
                         xytext=(0.75, 0.2),
                         arrowprops={'arrowstyle': '->', 'ls': '--', 'lw': 2.5, 'color': 'darkorange'}, alpha=0.9)
            ax3.axvline(x=rmax_AO, ls='--', lw=2.5, c='darkorange', alpha=0.9,
                        label=r'$\mathrm{AO} \,\, r_\mathrm{max}$')
        else:
            ax3.axvline(x=rmax_AO, ls='--', lw=2.5, c='darkorange', alpha=0.9,
                        label=r'$\mathrm{AO} \,\, r_\mathrm{max}$')

        if rmax_NS > np.min([np.max(extended_radius / r_d_min), 8]):
            ax3.axvline(x=rmax_NS, ls='--', lw=2.5, c='darkgreen', alpha=0.9,
                        label=r'$\mathrm{NS} \,\, r_\mathrm{max}$')
        else:
            ax3.axvline(x=rmax_NS, ls='--', lw=2.5, color='darkgreen', alpha=0.9,
                        label=r'$\mathrm{NS} \,\, r_\mathrm{max}$')

        if np.min([np.max(extended_radius / r_d_min), 8]) > 6.3:
            ax3.axvline(x=5.04, ls='--', c='grey', alpha=0.9)
        ax3.axvline(x=3.4, ls='--', c='grey', alpha=0.9)
        ax3.set_ylabel(r'$v\mathrm{[km/s]}$', fontsize=23)
        ax3.set_xticks([])
        ax3.axvline(x=rflat_fit / r_d_min, ls='--', lw=2.5, c='red', alpha=0.9,
                    label=r"$r_{\mathrm{flat}}$")
        ax3.ticklabel_format(style='sci', axis='y', useMathText=True)

        xlims = 8
        ax3.set_xlim(0, xlims)
        ax3.set_ylim(0, vflat_fit + 50)
        ax3.tick_params(axis='both', which='major', direction='in', labelsize=20, pad=3)
        ax3.legend(loc='upper center', fontsize=20, bbox_to_anchor=(0.5, 1.34), columnspacing=0.75, handletextpad=0.5,
                   handlelength=1, fancybox=True, shadow=False, ncol=4)  # 0.5, 1.3bbox_to_anchor=(0.5, 1.3)

        r = np.arange(0.1, 60, 0.1)
        j_cum = []
        #j_cum_NS = []
        #j_cum_AO = []
        for fff in range(len(r)):
            jjj = my_funcs.calc_j_analitically(r_d_min, rflat_fit, vflat_fit, r[fff], kpc_per_pix)
            j_cum.append(jjj)
            #jjj_NS = my_funcs.calc_j_analitically(r_d_min, rflat_fit_NS/resolution_ratio, vflat_fit_NS, r[fff], kpc_per_pix)
            #j_cum_NS.append(jjj_NS)
            #jjj_AO = my_funcs.calc_j_analitically(r_d_min, rflat_fit_AO, vflat_fit_AO, r[fff], kpc_per_pix)
            #j_cum_AO.append(jjj_AO)
        ax4.plot(r / r_d_min, j_cum / np.max(j_cum), "--", label=r"$\mathrm{Best \, model}$", lw=3.5, alpha=0.9)
        #ax4.plot(r / r_d_min, j_cum_AO / np.max(j_cum_AO), "--", label=r"$\mathrm{Best \, model\,\,AO}$", lw=2.5, alpha=0.9, color="darkorange")
        #ax4.plot(r / r_d_min, j_cum_NS / np.max(j_cum_NS), "--", label=r"$\mathrm{Best \, model\,\, NS}$", lw=2.5, alpha=0.9, color="darkgreen")
        if rmax_AO > np.min([np.max(extended_radius / r_d_min), 8]):
            ax4.annotate(r'$\mathrm{AO} \,\, r_\mathrm{max}$', xy=(1, 0.22), xycoords='axes fraction', size=16,
                         xytext=(0.75, 0.2),
                         arrowprops={'arrowstyle': '->', 'ls': '--', 'lw': 2.5, 'color': 'darkorange'}, alpha=0.9)
            ax4.axvline(x=rmax_AO, ls='--', lw=2.5, c='darkorange', alpha=0.9,
                        label=r'$\mathrm{AO} \,\, r_\mathrm{max}$')
        else:
            ax4.axvline(x=rmax_AO, ls='--', lw=2.5, c='darkorange', alpha=0.9,
                        label=r'$\mathrm{AO} \,\, r_\mathrm{max}$')

        if rmax_NS > np.min([np.max(extended_radius / r_d_min), 8]):
            ax4.axvline(x=rmax_NS, ls='--', lw=2.5, c='darkgreen', alpha=0.9,
                        label=r'$\mathrm{NS} \,\, r_\mathrm{max}$')
        else:
            ax4.axvline(x=rmax_NS, ls='--', lw=2.5, c='darkgreen', alpha=0.9,
                        label=r'$\mathrm{NS} \,\, r_\mathrm{max}$')
        ax4.set_xlim(0, )
        ax4.set_xlabel(r"$\mathrm{deprojected} \quad r/r_d$", fontsize=23)
        ax4.set_ylabel(r"$j_*/j_\mathrm{*,tot}$", fontsize=23)
        ax4.axvline(x=rflat_fit / r_d_min, ls='--', lw=2.5, c='red', alpha=0.9)
        ax4.axhline(y=1.0, ls='--', lw=2.5, c='grey', alpha=0.9)
        ax4.axvline(x=3.4, ls='--', c='grey', alpha=0.9)
        if np.min([np.max(extended_radius / r_d_min), 8]) > 6.3:
            ax4.axvline(x=5.04, ls='--', c='grey', alpha=0.9)
            ax4.text((5.15), 0.45, r"$r{=}3r_\mathrm{eff}$", c="k", size=16)
        if np.min([np.max(extended_radius / r_d_min), 8]) > 4.5:
            ax4.text((3.5), 0.45, r"$r{=}2r_\mathrm{eff}$", c="k", size=16)
        ax4.set_xlim(0, xlims)
        ax4.set_ylim(0, 1.23)
        ax4.tick_params(axis='both', which='major', direction='in', labelsize=20, pad=3)
        plt.subplots_adjust(wspace=2.5, hspace=0.00)

        plt.savefig(f"{path}/results/{galaxy}/{galaxy}_HST_model.pdf", overwrite=True,
                    bbox_inches='tight', pad_inches = 0.05)
        plt.close()

    make_plots()

    # results = ascii.read(f'{path}/results/results_table.csv')
    # results.add_row([galaxy,
    #                  "comb",
    #                  '{:.1f}'.format(pa_fit),
    #                  '{:.1f}'.format(pa_fit),
    #                  '{:.1f}'.format(inc_fit),
    #                  '{:.1f}'.format(inc_fit),
    #                  '{:.2f}'.format(rflat_fit * kpc_per_pix),
    #                  '{:.1f}'.format(rflat_fit * kpc_per_pix),
    #                  '{:.1f}'.format(vflat_fit),
    #                  '{:.1f}'.format(vflat_fit),
    #                  '{:.1f}'.format(j_model),
    #                  '{:.1f}'.format(j_model),
    #                  '{:.1f}'.format(j_model),
    #                  '{:.1f}'.format(j_2rh),
    #                  '{:.1f}'.format(j_3rh),
    #                  '{:.1f}'.format(j_model_ao),
    #                  '{:.1f}'.format(j_model_NS),
    #                  '{:.1f}'.format(j_approx),
    #                  '{:.2f}'.format(r_d_min * kpc_per_pix),
    #                  '{:.1f}'.format(r_d_min * kpc_per_pix),
    #                  '{:.1f}'.format(np.degrees(np.arcsin(np.abs(np.sin(np.radians(pa_fit) - np.radians(pa_fit)))))),
    #                  '{:.1f}'.format(j_1rh),
    #                  '{:.1f}'.format(x0_fit_ao),
    #                  '{:.1f}'.format(y0_fit_ao),
    #                  '{:.1f}'.format(x0_fit_ns),
    #                  '{:.1f}'.format(y0_fit_ns),
    #                  '{:.1f}'.format(x0_phot),
    #                  '{:.1f}'.format(y0_phot)
    #                  ])
    # ascii.write(results, f'{path}/results/results_table.csv', delimiter=',')
    #
    # results = ascii.read(f'{path}/results/results_table_short.csv')
    # results.add_row([galaxy,
    #                  "comb",
    #                  '{:.1f}'.format(pa_fit),
    #                  '{:.1f}'.format(inc_fit),
    #                  '{:.2f}'.format(r_d_min * kpc_per_pix),
    #                  '{:.2f}'.format(rflat_fit * kpc_per_pix),
    #                  '{:.1f}'.format(vflat_fit),
    #                  '{:.1f}'.format(j_3rh),
    #                  '{:.1f}'.format(j_model),
    #                  '{:.1f}'.format(j_approx),
    #                  '{:.1f}'.format(np.degrees(np.arcsin(np.abs(np.sin(np.radians(pa_fit) - np.radians(pa_fit))))))
    #                  ])
    # ascii.write(results, f'{path}/results/results_table_short.csv', delimiter=',')

    print(colored("Finished", "green"), colored(galaxy, "green"), colored("(combined)", "green"))

    gc.collect()


