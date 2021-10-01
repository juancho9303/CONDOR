import numpy as np
from modules import fitting_functions as funcs

class Galaxy():

    def __init__(self, name, z_ns, halpha_ns, velmax_ns, reff_kpc, psf_fwhm_NS_paper, maxbin_ns, x0_ns, y0_ns,
                 pafit_ns, pixscale_ns, SFR, M_s, type_ns, sigma_kernel_pix_ns, phot_file_name, H_band,
                 J_band, psf_fwhm_h, psf_fwhm_j, pixscale_hst, sigma_x_ns, sigma_y_ns, q_ns, theta_ns,
                 psf_fwhm_mine_ns, z_ao, halpha_ao, pa_ao, velmax_ao, reff_kpc_ao, psf_fwhm_paper_ao,
                 maxbin_ao, x0_ao, y0_ao, pafit_ao, pixscale_ao, extent_to_show_ao, type_ao, alpha,
                 beta, q_ao, theta_ao, airy, peak, strehl, FWHM_airy, FWHM_moffat, psf_fwhm_ao, pa, inc, rflat,
                 vflat, b_a, r_d):
        self.name = name
        self.z_ns = z_ns
        self.halpha_ns = halpha_ns
        self.velmax_ns = velmax_ns
        self.reff_kpc = reff_kpc
        self.psf_fwhm_NS_paper = psf_fwhm_NS_paper
        self.maxbin_ns = maxbin_ns
        self.x0_ns = x0_ns
        self.y0_ns = y0_ns
        self.pafit_ns = pafit_ns
        self.pixscale_ns = pixscale_ns
        self.SFR = SFR
        self.M_s = M_s
        self.type_ns = type_ns
        self.sigma_kernel_pix_ns = sigma_kernel_pix_ns
        self.phot_file_name = phot_file_name
        self.H_band = H_band
        self.J_band = J_band
        self.psf_fwhm_h = psf_fwhm_h
        self.psf_fwhm_j = psf_fwhm_j
        self.pixscale_hst = pixscale_hst
        self.sigma_x_ns = sigma_x_ns
        self.sigma_y_ns = sigma_y_ns
        self.q_ns = q_ns
        self.theta_ns = theta_ns
        self.psf_fwhm_mine_ns = psf_fwhm_mine_ns
        self.z_ao = z_ao
        self.halpha_ao = halpha_ao
        self.pa_ao = pa_ao
        self.velmax_ao = velmax_ao
        self.reff_kpc_ao = reff_kpc_ao
        self.psf_fwhm_paper_ao = psf_fwhm_paper_ao
        self.maxbin_ao = maxbin_ao
        self.x0_ao = x0_ao
        self.y0_ao = y0_ao
        self.pafit_ao = pafit_ao
        self.pixscale_ao = pixscale_ao
        self.extent_to_show_ao = extent_to_show_ao
        self.type_ao = type_ao
        self.alpha = alpha
        self.beta = beta
        self.q_ao = q_ao
        self.theta_ao = theta_ao
        self.airy = airy
        self.peak = peak
        self.strehl = strehl
        self.FWHM_airy = FWHM_airy
        self.FWHM_moffat = FWHM_moffat
        self.psf_fwhm_ao = psf_fwhm_ao
        self.pa = pa
        self.inc = inc
        self.rflat = rflat
        self.vflat = vflat
        self.b_a = b_a
        self.r_d = r_d

    # Methods of the class
    def inc_axis_ratio(self):
        return funcs.inclination_from_axis_ratio(self.b_a, 0.2)

    def set_priors_ns(self, ndim, nwalkers):

        p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
        for ii in range(len(p0)):
            p0[ii, :] = [
                np.random.uniform(self.pa - 35, self.pa + 35),
                np.random.uniform(self.inc_axis_ratio() - 3, self.inc_axis_ratio() + 3),
                np.random.uniform(self.rflat - 1.5, self.rflat + 1.5),
                np.random.uniform(self.vflat - 40, self.vflat + 40),
                np.random.uniform(self.x0_ns - 1.5, self.x0_ns + 1.5),
                np.random.uniform(self.y0_ns - 1.5, self.y0_ns + 1.5),
            ]

        return p0

    def set_priors_ao(self, ndim, nwalkers):

        p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
        for ii in range(len(p0)):
            p0[ii, :] = [
                np.random.uniform(self.pa - 35, self.pa + 35),
                np.random.uniform(self.inc_axis_ratio() - 3, self.inc_axis_ratio() + 3),
                np.random.uniform(self.rflat - 1.5, self.rflat + 1.5),
                np.random.uniform(self.vflat - 40, self.vflat + 40),
                np.random.uniform(self.x0_ao - 1.5, self.x0_ao + 1.5),
                np.random.uniform(self.y0_ao - 1.5, self.y0_ao + 1.5),
            ]

        return p0

    def set_priors_comb(self, ndim, nwalkers):

        p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))

        for ii in range(len(p0)):
            p0[ii, :] = [
                np.random.uniform(self.pa - 35, self.pa + 35),
                np.random.uniform(self.inc_axis_ratio() - 3, self.inc_axis_ratio() + 3),
                np.random.uniform(self.rflat - 1.5, self.rflat + 1.5),
                np.random.uniform(self.vflat - 40, self.vflat + 40),
                np.random.uniform(self.x0_ns - 1.5, self.x0_ns + 1.5),
                np.random.uniform(self.y0_ns - 1.5, self.y0_ns + 1.5),
                np.random.uniform(self.x0_ao - 1.5, self.x0_ao + 1.5),
                np.random.uniform(self.y0_ao - 1.5, self.y0_ao + 1.5),
            ]

        return p0

if __name__ == "__main__":
    Galaxy()