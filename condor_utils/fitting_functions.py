# Import relevant libraries/modules
from __future__ import print_function
import numpy as np
from plotbin.symmetrize_velfield import symmetrize_velfield
from plotbin.plot_velfield import plot_velfield
from regions import PixCoord, CirclePixelRegion, EllipsePixelRegion
from scipy.interpolate import interp1d
from scipy import interpolate
import scipy
from copy import copy
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
import pyfftw
import itertools
from lmfit import Parameters,minimize, fit_report

import numba
from astropy.io import fits, ascii
from scipy.optimize import curve_fit as cf

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.offsetbox
from matplotlib.lines import Line2D


def merge(AO_object, NaturalSeeing_object):

    # Function that merges the AO and Natural Seeing objects (prioritizing AO)

    merged_object = np.empty([AO_object.shape[0],AO_object.shape[1]])
    for i in range(AO_object.shape[0]):
        for j in range(AO_object.shape[1]):
            if np.abs(AO_object[i,j])>0:
                merged_object[i,j] = AO_object[i,j]
            else:
                merged_object[i,j] = NaturalSeeing_object[i,j]

    return merged_object[:,:]

#-----------------------------------------------------------------------------------------------------------------------

def surf_mass_den_profile(r, s_d, r_d):

    # Function that defines the surface mass density profile and depends on s_d which is SMD normalization
    # and r_d which is the disk scalelength

    profile = s_d*(np.exp(-r/r_d))

    return profile

#-----------------------------------------------------------------------------------------------------------------------
@numba.njit
def v_circular_model(r, vflat, rflat):

    # Function that defines the used circular model, which is an exponential rflat profile

    circular_model = vflat*(1-np.exp(-r/rflat))

    return circular_model

#-----------------------------------------------------------------------------------------------------------------------

def v_circular_model_arctan(r, vtan, rtan):

    # Function that defines the used circular model, whith an arctan profile

    circular_model_arctan = 2/np.pi * vtan * np.arctan(r/rtan)

    return circular_model_arctan

#-----------------------------------------------------------------------------------------------------------------------

def radial_profile(data, x0, y0, data_error):

    # Function that calculates a radial profile from the data and the coordinates

    y, x   = np.indices((data.shape))                   # Get the X and Y coordinates indices
    x_each = x-x0                                       # Distance in x from the center for each bin
    y_each = y-y0                                       # Distance in y from the center for each bin
    r_float = np.sqrt((x - x0)**2 + (y - y0)**2)        # First determine radii of all pixels from the centre
    r_int   = r_float.astype(np.int)                    # Transform them into integers
    tbin    = np.bincount(r_int[data!=0].ravel(), data[data!=0].ravel())
    tbin_error = np.bincount(r_int[data!=0].ravel(), data_error[data!=0].ravel())
    nr      = np.bincount(r_int[data!=0].ravel())       # Number of data points per bin
    radialprofile = tbin / (nr+1e-7)                    # radial profile from the binned data
    error_rad_profile = (tbin_error/nr)# / nr#np.sqrt(nr)     # / standard error of the mean

    return r_int, r_float, tbin, nr, radialprofile, x, y, x_each, y_each, error_rad_profile

#-----------------------------------------------------------------------------------------------------------------------

def convert_1D_to_2D_profile(distance_matrix,profile,length_axis):

    # Function that creates a 2D array according to a 1D model ny interpolating

    r = np.arange(length_axis)
    f = interp1d(r, profile)

    return f(distance_matrix.flat).reshape(distance_matrix.shape)

#-----------------------------------------------------------------------------------------------------------------------

def azimuthalAverage(image, center=None):

    # Function that calculates an azimuthal average radial profile from a 2D map
    # image - The 2D image
    # center - The [x,y] pix coords. Default is None, which then uses the center of the image (including fracitonal pixels).

    y, x = np.indices(image.shape)  # Calculate the indices from the image

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind      = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]    # Assumes all radii represented
    rind   = np.where(deltar)[0]       # location of changed radius
    nr     = rind[1:] - rind[:-1]      # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    #csim = np.mean(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    error_rad_profile = radial_prof / np.sqrt(nr)

    return radial_prof, ind, r_sorted, i_sorted, nr, rind, r_int, error_rad_profile, tbin

#-----------------------------------------------------------------------------------------------------------------------

def fit_kinematic_pa(x, y, vel, debug=False, nsteps=361,quiet=False, plot=False, dvel=10):

    # Michele Capellari's code to get the best PA

    xsize = vel.shape[0]
    ysize = vel.shape[1]

    x, y, vel = map(np.ravel, [x, y, vel])

    vel[np.isnan(vel)]=0

    assert x.size == y.size == vel.size, 'Input vectors (x, y, vel) must have the same size'

    nbins = x.size
    n = nsteps
    angles = np.linspace(0, 180, n) # 0.5 degrees steps by default
    chi2 = np.empty_like(angles)
    for j, ang in enumerate(angles):
        velSym = symmetrize_velfield(x, y, vel, sym=1, pa=ang)
        chi2[j] = np.sum(((vel - velSym)/dvel)**2)
        if debug:
            print('Ang: %5.1f, chi2/DOF: %.4g' % (ang, chi2[j]/nbins))
            plt.cla()
            plot_velfield(x, y, velSym,cmap=mpl.cm.RdYlBu_r)
            plt.pause(0.01)
    k = np.argmin(chi2)
    angBest = angles[k]

    # Compute fit at the best position
    velSym = symmetrize_velfield(x, y, vel, sym=1, pa=angBest)  # Saving the best model
    if angBest < 0:
        angBest += 180

    # 3sigma confidence limit, including error on chi^2
    f = chi2 - chi2[k] <= 9 + 3*np.sqrt(2*nbins)
    minErr = max(0.5, (angles[1] - angles[0])/2.0)
    if f.sum() > 1:
        angErr = (np.max(angles[f]) - np.min(angles[f]))/2.0
        if angErr >= 45:
            good = np.degrees(np.arctan(np.tan(np.radians(angles[f]))))
            angErr = (np.max(good) - np.min(good))/2.0
        angErr = max(angErr, minErr)
    else:
        angErr = minErr

    vSyst = np.median(vel - velSym)

    model_2D = velSym.reshape(xsize,ysize)    # This is to convert the array to a 2D matrix for plotting

    if not quiet:
        print('  Kin PA: %5.1f' % angBest, ' +/- %5.1f' % angErr, ' (3*sigma error)')
        print('Velocity Offset: %.2f' % vSyst)

    # Plot results
    if plot:

        mn, mx = np.percentile(velSym, [2.5, 97.5])
        mx = min(mx, -mn)
        #velSym[velSym==0]=np.nan
        plt.subplot(121)
        plot_velfield(x, y, velSym, cmap=mpl.cm.RdYlBu_r,vmin=-mx, vmax=mx)
        plt.title('Symmetrized')

        #vel[vel==0]=np.nan
        plt.subplot(122)
        plot_velfield(x, y, vel - vSyst, cmap=mpl.cm.RdYlBu_r,vmin=-mx, vmax=mx)
        plt.title('Data and best PA')
        rad = np.sqrt(np.max(x**2 + y**2))
        ang = [0,np.pi] + np.radians(angBest)
        plt.plot(rad*np.cos(ang), rad*np.sin(ang), linestyle='--', color="red", linewidth=3) # Zero-velocity line
        plt.plot(-rad*np.sin(ang), rad*np.cos(ang), linestyle='--', color="black", linewidth=3) # Major axis PA

    return angBest, angErr, vSyst, velSym, model_2D, x, y, vel

#-----------------------------------------------------------------------------------------------------------------------
#     Michele cappellari, Paranal, 10 November 2013

def _rotate_points(x, y, ang):
    """
    Rotates points counter-clockwise by an angle ANG-90 in degrees.
    """
    theta = np.radians(ang - 90.)
    xNew = x*np.cos(theta) - y*np.sin(theta)
    yNew = x*np.sin(theta) + y*np.cos(theta)

    return xNew, yNew

#-----------------------------------------------------------------------------------------------------------------------

def symmetrize_velfield(xbin, ybin, vel_bin, sym=2, pa=90.):
    """
    This routine generates a bi-symmetric ('axisymmetric') of point-symmetric
    version of a given set of kinematical measurements.
    PA: is the angle in degrees, measured counter-clockwise,
      from the vertical axis (Y axis) to the galaxy major axis.
    SYM: by-simmetry: is 1 for (V, h3, h5) and 2 for (sigma, h4, h6)
      point-simmetry: is 3 for (V, h3, h5) and 4 for (sigma, h4, h6)

    """
    xbin, ybin, vel_bin = map(np.asarray, [xbin, ybin, vel_bin])

    assert xbin.size == ybin.size == vel_bin.size, \
        "The vectors (xbin, ybin, velBin) must have the same size"
    assert isinstance(sym, int), "sym must be integer"
    assert 1 <= sym <= 4, "must be 1 <= sym <= 4"

    if sym < 3:
        x, y = _rotate_points(xbin, ybin, -pa)  # Negative PA for counter-clockwise  #xbin,ybin
        xout = np.hstack([x,-x, x,-x])
        yout = np.hstack([y, y,-y,-y])
        vel_out = interpolate.griddata((x, y), vel_bin, (xout, yout))
        #print(vel_out.shape)
        vel_out = vel_out.reshape(4, xbin.size)
        vel_out[0, :] = vel_bin  # see V3.0.1
        if sym == 1:
            vel_out[[1, 3], :] *= -1.
        #print(vel_out.shape)
    else:
        vel_out = interpolate.griddata((xbin, ybin), vel_bin, (-xbin, -ybin))
        if sym == 3:
            vel_out = -vel_out
        vel_out = np.row_stack([vel_bin, vel_out])

    vel_sym = np.nanmean(vel_out, axis=0)

    return vel_sym

#---------------------------------------------------------------------------------------------------------------------

def my_vel_model(x ,y, pa, inc, rflat, vflat, x0, y0, NS_kernel, AO_kernel, i, data):

    r_float = np.sqrt((x - x0)**2 + (y - y0)**2) # First determine radii of all pixels
    r_int = r_float.astype(np.int)               # Transform them into integers

    xfo = ((x-x0)*np.cos(np.radians(pa))+(y-y0)*np.sin(np.radians(pa)))+1e-7 # face-on x-coordinate
    yfo = ((y-y0)*np.cos(np.radians(pa))-(x-x0)*np.sin(np.radians(pa)))/np.cos(np.radians(inc))+1e-17  # face-on y-coordinate
    r   = np.sqrt(xfo**2+yfo**2)                                   # [pixel] face-on radius from origin

    vel_fac = np.sin(np.radians(inc))/np.sqrt(1+(yfo/xfo)**2)*np.sign(xfo) # [-] velocity projection factor: v_radial = vel_fac*v_circular
    #vel_fac = vel_fac/np.nanmax(vel_fac)
    #vel_fac[(vel_fac < 0.15) & (vel_fac > -0.15)] = np.nan
    #vel_mod = v_circular_model(r_float,vflat,rflat) # rflat exponential profile
    vel_mod = v_circular_model(r, vflat, rflat)  # rflat exponential profile
    #vel_mod[(vel_fac < 0.15) & (vel_fac > -0.15)] = np.nan

    # No convolution
    if (NS_kernel is None) and (AO_kernel is None):
        model = vel_fac * vel_mod
        #model[np.isnan(data)] = np.nan
        full_model = copy(model)

    # NS convolution
    elif (NS_kernel is not None) and (AO_kernel is None):
        model = vel_fac * vel_mod
        full_model = copy(model)
        model[np.isnan(data)] = np.nan
        model = convolve(model, NS_kernel, boundary='extend', preserve_nan=True, normalize_kernel=True)
        #print("stuff")

    # # AO convolution for the smoothed maps (i=1, i=2, pretty slow)
    # elif AO_kernel != 0 and NS_kernel==0 and 0<i<3:
    #     model = vel_fac * vel_mod
    #     full_model = copy(model)
    #     model[np.isnan(data)] = np.nan
    #     model = convolve(model, AO_kernel, boundary='extend', preserve_nan=True, normalize_kernel=True)
    #     model = ndimage.filters.generic_filter(model, np.nanmedian, size=2)

    # AO convolution
    elif (AO_kernel is not None) and (NS_kernel is None):
        model = vel_fac * vel_mod
        full_model = copy(model)
        model[np.isnan(data)] = np.nan
        model = convolve(model, AO_kernel, boundary='extend', preserve_nan=True, normalize_kernel=True)
        #print("stuff")

    # fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    # ax1.imshow(model, origin="lower")
    # ax1.set_title("model")
    # ax2.imshow(vel_mod, origin="lower")
    # ax2.set_title("face-on model")
    # ax3.imshow(vel_fac, origin="lower")
    # ax3.set_title("vel fac")
    # plt.show()
    # sys.exit()

    return full_model, vel_fac, vel_mod, r, xfo, yfo, r_float, model


#---------------------------------------------------------------------------------------------------------------------

def my_vel_model_for_plot(x ,y, pa, inc, rflat, vflat, x0, y0, NS_kernel, AO_kernel, i, data):

    r_float = np.sqrt((x - x0)**2 + (y - y0)**2) # First determine radii of all pixels
    r_int = r_float.astype(np.int)               # Transform them into integers

    xfo = ((x-x0)*np.cos(np.radians(pa))+(y-y0)*np.sin(np.radians(pa)))+1e-7 # face-on x-coordinate
    yfo = ((y-y0)*np.cos(np.radians(pa))-(x-x0)*np.sin(np.radians(pa)))/np.cos(np.radians(inc))+1e-17  # face-on y-coordinate
    r   = np.sqrt(xfo**2+yfo**2)                                   # [pixel] face-on radius from origin

    vel_fac = np.sin(np.radians(inc))/np.sqrt(1+(yfo/xfo)**2)*np.sign(xfo) # [-] velocity projection factor: v_radial = vel_fac*v_circular
    #vel_fac = vel_fac / np.nanmax(vel_fac)
    #vel_fac[(vel_fac < 0.15) & (vel_fac > -0.15)] = 0
    vel_mod = v_circular_model(r,vflat,rflat) # rflat exponential profile


    # No convolution
    if (NS_kernel is None) and (AO_kernel is None):
        model = vel_fac * vel_mod
        full_model = copy(model)

    # NS convolution
    elif (NS_kernel is not None) and (AO_kernel is None):
        model = vel_fac * vel_mod
        full_model = copy(model)
        model[np.isnan(data)] = np.nan
        model = convolve(model, NS_kernel, boundary='extend', preserve_nan=True, normalize_kernel=True)
        #print("stuff")

    # AO convolution
    elif (AO_kernel is not None) and (NS_kernel is None):
        model = vel_fac * vel_mod
        full_model = copy(model)
        model[np.isnan(data)] = np.nan
        model = convolve(model, AO_kernel, boundary='extend', preserve_nan=True, normalize_kernel=True)
        #print("stuff")

    return full_model, vel_fac, vel_mod, r, xfo, yfo, r_float, model


#-----------------------------------------------------------------------------------------------------------------------

def my_vel_differential_convolution(x ,y, pa, inc, rflat, vflat, x0, y0, sigma_ns, sigma_peak, sigma_broad, size_ao):
    # Perform the deprojection

    r_float = np.sqrt((x - x0)**2 + (y - y0)**2)      # First determine radii of all pixels
    r_int = r_float.astype(np.int)                    # Transform them into integers

    xfo = ((x-x0)*np.cos(np.radians(pa))+(y-y0)*np.sin(np.radians(pa)))+1e-7               # face-on x-coordinate
    yfo = ((y-y0)*np.cos(np.radians(pa))-(x-x0)*np.sin(np.radians(pa)))/np.cos(np.radians(inc))+1e-17  # face-on y-coordinate
    r   = np.sqrt(xfo**2+yfo**2)                                   # [pixel] face-on radius from origin

    vel_fac = np.sin(np.radians(inc))/np.sqrt(1+(yfo/xfo)**2)*np.sign(xfo)     # [-] velocity projection factor: v_radial = vel_fac*v_circular
    vel_mod = v_circular_model(r_float,vflat,rflat)

    if sigma_ns == 0 and sigma_peak == 0 and sigma_broad == 0:
        model_full = vel_fac * vel_mod# + voffset  # Fix for deprojection by multiplying by the velocity factor

    elif sigma_broad == 0:
        model = vel_fac * vel_mod
        model_ao = convolve(model, Gaussian2DKernel(sigma_peak), boundary='extend')
        model_ns = convolve(model, Gaussian2DKernel(sigma_ns), boundary='extend')
        model_full = copy(model_ns)
        model_full[size_ao] = model_ao[size_ao]

    else:
        model = vel_fac * vel_mod
        model_ao = convolve(model, Gaussian2DKernel(sigma_peak) + Gaussian2DKernel(sigma_broad), boundary='extend')
        model_ns = convolve(model, Gaussian2DKernel(sigma_ns), boundary='extend')
        model_full = copy(model_ns)
        model_full[size_ao] = model_ao[size_ao]

    return model_full, vel_fac, vel_mod, r, xfo, yfo


# ----------------------------------------------------------------------------------------------------------------------

def calculate_j(positions, deprojected_vmap, photometry_map, size, kpc_per_pix):

    # For all pixels
    if size == 0:
        J = np.nansum( (positions.ravel()) * deprojected_vmap.ravel() * photometry_map.ravel()) # Calculate J spaxel by spaxel
        M = np.nansum(photometry_map.ravel()[deprojected_vmap.ravel()!=0])   # Get the total mass
        j = J / M  * kpc_per_pix

    # For the observed pixels
    else:
        J = np.nansum((positions[size].ravel()) * deprojected_vmap[size].ravel() * photometry_map[size].ravel())  #
        M = np.nansum(photometry_map[size].ravel()[deprojected_vmap.ravel()!=0])                                                              # Get the total mass
        j = J / M  * kpc_per_pix


    return J, M, j

#-----------------------------------------------------------------------------------------------------------------------

def calculate_j_weighted(positions, deprojected_vmap, photometry_map, weight, size, kpc_per_pix):

    # For all pixels
    if size == 0:
        J = np.nansum( positions.ravel() * deprojected_vmap.ravel() * photometry_map.ravel() * weight.ravel()) # Calculate J spaxel by spaxel
        M = np.nansum(photometry_map.ravel()[deprojected_vmap.ravel()!=0] * weight.ravel()[deprojected_vmap.ravel()!=0])                                                 # Get the total mass
        j = J / M * kpc_per_pix

    # For the observed pixels
    else:
        J = np.nansum(positions[size].ravel() * deprojected_vmap[size].ravel() * photometry_map[size].ravel() * weight[size].ravel())  #
        M = np.nansum(photometry_map[size].ravel()[deprojected_vmap[size].ravel()!=0] * weight[size].ravel()[deprojected_vmap[size].ravel()!=0])                                                              # Get the total mass
        j = J / M * kpc_per_pix


    return J, M, j

#-----------------------------------------------------------------------------------------------------------------------

def calculate_cumulative_j(r, vel, den, dep_rad, rad_int, kpc_per_pix):

    J_array = []
    M_array = []
    for r_bin in range(len(dep_rad)):
        filt_vel = vel[rad_int == r_bin]
        filt_den = den[rad_int == r_bin]
        filt_rad = r[rad_int == r_bin]
        M_array.append( np.nansum(filt_den))
        J_array.append( np.nansum((filt_rad * kpc_per_pix) * filt_vel * filt_den) )

    J_cum = np.cumsum(J_array)
    M_cum = np.nancumsum(M_array)
    j_cum = J_cum/M_cum
    j_cum[0] = 0
    M_cum[0] = 0

    return J_cum, M_cum ,j_cum

#-----------------------------------------------------------------------------------------------------------------------

class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """ size: length of bar in data units
    extent : height of bar ends in axes units """
    def __init__(self, size=1, extent = 0.03, label="",loc=2, bbox_to_anchor= None, ax=None,
                 pad=0.4, borderpad=0.5, ppad = 0, sep=2, prop=None,
                 frameon=True, **kwargs):
        if not ax:
            ax = plt.gca()
            trans = ax.get_xaxis_transform()
            size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
            line = Line2D([0,size],[0,0], linewidth=2.5,**kwargs)
            vline1 = Line2D([0,0],[-extent/2.,extent/2.], linewidth=2.5, **kwargs)
            vline2 = Line2D([size,size],[-extent/2.,extent/2.], linewidth=2.5, **kwargs)
            size_bar.add_artist(line)
            size_bar.add_artist(vline1)
            size_bar.add_artist(vline2)
            txt = matplotlib.offsetbox.TextArea(label, minimumdescent=False, textprops=dict(color="white", size=18))
            self.vpac = matplotlib.offsetbox.VPacker(children=[size_bar,txt],align="center", pad=ppad, sep=sep)
            matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                                            bbox_to_anchor=bbox_to_anchor, child=self.vpac, prop=prop, frameon=frameon)


#-----------------------------------------------------------------------------------------------------------------------

def halfbin_regrid(array):

    # unction to regrid an array to half the bin size with no interpolation at all

    outKsz1=array.shape[0]*2
    outKsz2=array.shape[1]*2
    newarray=np.zeros((outKsz1,outKsz2))
    for i in range(newarray.shape[0]):
        for j in range(newarray.shape[1]):
            newarray[i,j]=array[np.int(np.round((i-0.5)/2.)),np.int(np.round((j-0.5)/2.))]

    return newarray

#------------------------------------------------------------------------------------

def match_maps(ns_map, x0_ns, y0_ns, ao_map, x0_ao, y0_ao, resolution_ratio):

    if resolution_ratio == 2:
        matched_map = halfbin_regrid(ns_map)

    if resolution_ratio == 4:
        prematched_map = halfbin_regrid(ns_map)
        matched_map = halfbin_regrid(prematched_map)

    if resolution_ratio ==1:
        matched_map = ao_map

    aspect_x = int(np.min([int(x0_ns)*resolution_ratio, ns_map.shape[0]*resolution_ratio-int(x0_ns)*resolution_ratio,
                           int(x0_ao), ao_map.shape[0]-int(x0_ao)]))
    aspect_y = int(np.min([int(y0_ns)*resolution_ratio, ns_map.shape[1]*resolution_ratio-int(y0_ns)*resolution_ratio,
                           int(y0_ao), ao_map.shape[1]-int(y0_ao)]))

    copy_ao = copy(ao_map)

    copy_ao[int(int(x0_ao) - aspect_x):int(int(x0_ao)  + aspect_x),
    int(int(y0_ao) - aspect_y):int(int(y0_ao) + aspect_y)] = \
        matched_map[int(int(x0_ns) * resolution_ratio - aspect_x):int(int(x0_ns) * resolution_ratio + aspect_x),
        int(int(y0_ns) * resolution_ratio - aspect_y):int(int(y0_ns) * resolution_ratio + aspect_y)]

    matched_maps = copy(copy_ao)
    matched_maps_NS_only = copy(copy_ao)
    matched_maps[~np.isnan(ao_map)] = ao_map[~np.isnan(ao_map)]

    return matched_maps, matched_map, matched_maps_NS_only

#-----------------------------------------------------------------------------------------------------------------------

def halfbin_regrid_cube(cube,flux=False):
    '''
    function to regrid a cube to half the bin size with no interpolation at all
    '''
    cube2=cube.tolist()
    for i in range(len(cube)):

        cube2[i]=halfbin_regrid(cube[i].copy())
        if flux==True:
            cube2[i]=cube2[i]*np.sum(cube[i])/np.sum(cube2[i])

    cube2=np.array(cube2)

    return cube2


#-----------------------------------------------------------------------------------------------------------------------
@numba.njit
def density_profile(x, y, s_d, r_d_min, x0, y0):

    r = np.sqrt((x - x0)**2 + (y - y0)**2)      # First determine radii of all pixels
    profile = s_d * (np.exp(-r / r_d_min))
    profile = profile/np.max(profile)

    return profile

#-----------------------------------------------------------------------------------------------------------------------
@numba.njit
def density_profile_mcmc(x, y, s_d, r_d_min, x0, y0, inc_phot, pa_phot):

    x_phot_dep = ((x - x0) * np.cos(np.radians(pa_phot)) + (y - y0) * np.sin(np.radians(pa_phot)))  # face-on x-coordinate
    y_phot_dep = ((y - y0) * np.cos(np.radians(pa_phot)) - (x - x0) * np.sin(np.radians(pa_phot))) / np.cos(np.radians(inc_phot)) + 1e-17  # face-on y-coordinate
    r_phot_dep = np.sqrt(x_phot_dep ** 2 + y_phot_dep ** 2)

    profile = s_d * (np.exp(-r_phot_dep / r_d_min))

    faceon_profile = density_profile(x, y, s_d, r_d_min, x0, y0)

    return profile, faceon_profile

#-----------------------------------------------------------------------------------------------------------------------

def density_profile_linear(x, y, s_d, r_d_min, x0, y0, inc, pa):

    x_phot_dep = ((x - x0) * np.cos(np.radians(pa)) + (y - y0) * np.sin(np.radians(pa)))  # face-on x-coordinate
    y_phot_dep = ((y - y0) * np.cos(np.radians(pa)) - (x - x0) * np.sin(np.radians(pa))) / np.cos(np.radians(inc)) + 1e-17  # face-on y-coordinate
    r_phot_dep = np.sqrt(x_phot_dep ** 2 + y_phot_dep ** 2)

    profile = r_phot_dep**(-0.5)
    #profile = s_d * (r_d_min/r_phot_dep)
    #profile = s_d * (np.exp(-r_phot_dep / r_d_min))
    profile = profile/np.max(profile)

    faceon_profile = density_profile(x, y, s_d, r_d_min, x0, y0)

    return profile, faceon_profile


#-----------------------------------------------------------------------------------------------------------------------

def density_profile_convolved(x, y, s_d, r_d_min, x0, y0, inc, pa, conv_kernel):

    x_phot_dep = ((x - x0) * np.cos(np.radians(pa)) + (y - y0) * np.sin(np.radians(pa)))  # face-on x-coordinate.
    y_phot_dep = ((y - y0) * np.cos(np.radians(pa)) - (x - x0) * np.sin(np.radians(pa))) / np.cos(np.radians(inc)) + 1e-17  # face-on y-coordinate
    r_phot_dep = np.sqrt(x_phot_dep ** 2 + y_phot_dep ** 2)

    profile = s_d * (np.exp(-r_phot_dep / r_d_min))

    faceon_profile = density_profile(x, y, s_d, r_d_min, x0, y0)

    profile = convolve(profile, conv_kernel, boundary='extend', preserve_nan=True)

    profile = profile/np.max(profile)

    return profile, faceon_profile


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- MCMC Density -----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

def lnprior_den(param, r_d_guess, x0_guess, y0_guess, inc_fit, pa_fit, inc_paper, pa_bound):

    r_d_min, x0, y0, inc_phot, pa_phot = param

    if (0.1 < r_d_min < r_d_guess+6
        and x0_guess - 5 < x0 < x0_guess + 5
        and y0_guess - 5 < y0 < y0_guess + 5
        and (inc_paper-3) < inc_phot < (inc_paper+3)
        and (pa_bound-15) < pa_phot < (pa_bound+15)):
        return 0.0

    return -np.inf

#-----------------------------------------------------------------------------------------------------------------------

def Lg_den(param, den, x, y, var, conv_kernel):
    r_d_min, x0, y0, inc_phot, pa_phot = param

    prob_den = np.nansum(- (den - density_profile_convolved(x, y, 1.0, r_d_min, x0, y0, inc_phot, pa_phot, conv_kernel)[0]) ** 2 / (2 * var))

    return prob_den

#-----------------------------------------------------------------------------------------------------------------------

def lnprob_den(param, den, x, y, var, conv_kernel, r_d_guess, x0_guess, y0_guess, inc_fit, pa_fit, inc_paper, pa_bound):
    lp = lnprior_den(param, r_d_guess, x0_guess, y0_guess, inc_fit, pa_fit, inc_paper, pa_bound)

    if not np.isfinite(lp):
        return -np.inf
    return lp + Lg_den(param, den, x, y, var, conv_kernel)

#-----------------------------------------------------------------------------------------------------------------------

def create_circular_mask(h, w, center=None, radius=None):

    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask

#---------------------------------------------------------------------------------------------

def rebin(a, shape):
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)

#-------------------------------------------------------------------------------------------
@numba.jit(nopython=True)
def singlegaussian(par,x):
    '''
    Single Gaussian function
    '''

    x0,I,sig=par

    Hafit= (I/sig/np.sqrt(2.0*np.pi)) * np.exp(-0.5*(x-x0)**2/sig**2)

    return Hafit


#________________________________________________________________________________________

def gaussian(x0, I, sig, x):
    Hafit = (I / sig / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (x - x0) ** 2 / sig ** 2)
    return Hafit

#---------------------------------------------------------------------------------------

def getwl3(cubefile,ext=0,kw1="CRVAL3",kw2="CRPIX3",kw3="CDELT3"):
    '''
    Function to get wavelength array from FITS headers
    '''

    hdu=fits.open(cubefile)

    crval1 = hdu[ext].header[kw1]
    crpix1 = hdu[ext].header[kw2]
    cdelt1 = hdu[ext].header[kw3]

    sz = np.shape(hdu[ext].data)
    wl = crval1*10 + cdelt1*10 * np.arange(sz[0]) - (crpix1-1)*cdelt1*10

    return wl

#-------------------------------------------------------------------------------------------------------

# LIKELIHOOD FUNCTIONS FOR CUBE ANALYSIS:

# Log prior
@numba.njit
def lnprior_cube(param, x0_fixed, y0_fixed, inc_fixed, rflat_lim, vflat_max):
    pa, inc, rflat, vflat, x0, y0 = param

    if (0 < pa < 360
        and (inc_fixed-3) < inc < (inc_fixed+3)
        and 0.1 < rflat < rflat_lim
        and 50 < vflat < (vflat_max/np.sin(np.radians(inc_fixed))+50)
        and (x0_fixed - 1.5) < x0 < (x0_fixed + 1.5)
        and (y0_fixed - 1.5) < y0 < (y0_fixed + 1.5)):

        #return np.log(np.sin(np.radians(inc)))#0.0
        return np.log(np.sin(np.radians(inc))) - np.log(vflat) - np.log(rflat)

    return -np.inf
#-----------------------------------------------------------------------------------------------------------------------

# The natural logarithm of the joint likelihood
#@numba.njit
def Lg_cube(param, vel, x, y, model, constant_inputs, i, var):
    pa, inc, rflat, vflat, x0, y0 = param
    input_params = [pa, inc, rflat, vflat, x0, y0, x0, y0]
    prob = np.nansum(- (vel - model(i, input_params, constant_inputs)) ** 2 / (2 * var))

    return prob# * (1 + punishment)
    
#-----------------------------------------------------------------------------------------------------------------------

# log-probability function (log of the joint posterior)
def lnprob_cube(param, vel, x, y, model, constant_inputs, i, var):

    x0_fixed, y0_fixed, inc_fixed = constant_inputs[-5], constant_inputs[-4], constant_inputs[-3]
    rflat_lim = constant_inputs[-2]
    vflat_max = constant_inputs[-1]

    lp = lnprior_cube(param, x0_fixed, y0_fixed, inc_fixed, rflat_lim, vflat_max)
    if not np.isfinite(lp):
        return -np.inf
    #print(lp, Lg_cube(param, vel, x, y, model, constant_inputs, i, var), lp + Lg_cube(param, vel, x, y, model, constant_inputs, i, var))
    return lp + Lg_cube(param, vel, x, y, model, constant_inputs, i, var)


#------------------------------------------#### CUBE combine ####-------------------------------------------------------

def lnprior_combine_cube(param, x0_ns_fixed, y0_ns_fixed, x0_ao_fixed, y0_ao_fixed, inc_fixed, rflat_lim, vflat_max):

    pa, inc, rflat, vflat, x0_ns, y0_ns, x0_ao, y0_ao = param

    if (0 < pa < 360
        and (inc_fixed-3) < inc < (inc_fixed+3)
        and 0.1 < rflat < rflat_lim
        and 50 < vflat < (vflat_max/np.sin(np.radians(inc_fixed))+50)
        and (x0_ns_fixed - 1.5) < x0_ns < (x0_ns_fixed + 1.5)
        and (y0_ns_fixed - 1.5) < y0_ns < (y0_ns_fixed + 1.5)
        and (x0_ao_fixed - 1.5) < x0_ao < (x0_ao_fixed + 1.5)
        and (y0_ao_fixed - 1.5) < y0_ao < (y0_ao_fixed + 1.5)):

        return np.log(np.sin(np.radians(inc))) - np.log(vflat) - np.log(rflat)

    return -np.inf
#-----------------------------------------------------------------------------------------------------------------------

def Lg_combine_cube(param, vel_ns, x_ns, y_ns, var_ns, model_ns, constant_inputs_ns, vel_ao, x_ao, y_ao, var_ao,
                    model_ao, constant_inputs_ao, i, resolution_ratio):

    pa, inc, rflat, vflat, x0_ns, y0_ns, x0_ao, y0_ao = param
    input_params_ns = [pa, inc, rflat/resolution_ratio, vflat, x0_ns, y0_ns, x0_ao, y0_ao]
    input_params_ao = [pa, inc, rflat, vflat, x0_ns, y0_ns, x0_ao, y0_ao]

    prob = np.nansum(- (vel_ns - model_ns(i, input_params_ns, constant_inputs_ns)) ** 2 / (2 * var_ns)) + \
           np.nansum(- (vel_ao - model_ao(i, input_params_ao, constant_inputs_ao)) ** 2 / (2 * var_ao))

    # punishment = (vflat - (np.nanmax(vel_ao))) + (vflat - (np.nanmax(vel_ns)))
    # if punishment < 0:
    #     return prob
    # if punishment > 0:
    #     return prob - punishment
    #punishment = (special.erf(vflat - (np.nanmax(vel_ao) + 30)) + 1) / 2
    #return prob * ((1 + punishment) / 5)

    return prob
#-----------------------------------------------------------------------------------------------------------------------

def lnprob_combine_cube(param, vel_ns, x_ns, y_ns, var_ns, model_ns, constant_inputs_ns, vel_ao, x_ao, y_ao, var_ao,
                        model_ao, constant_inputs_ao, i, resolution_ratio):
    x0_ns_fixed, y0_ns_fixed = constant_inputs_ns[-5], constant_inputs_ns[-4]
    inc_fixed = constant_inputs_ns[-3]
    x0_ao_fixed, y0_ao_fixed = constant_inputs_ao[-5], constant_inputs_ao[-4]
    rflat_lim = constant_inputs_ns[-2]
    vflat_max = constant_inputs_ao[-1]

    lp = lnprior_combine_cube(param, x0_ns_fixed, y0_ns_fixed, x0_ao_fixed, y0_ao_fixed, inc_fixed, rflat_lim,vflat_max)

    if not np.isfinite(lp):
        return -np.inf

    return lp + Lg_combine_cube(param, vel_ns, x_ns, y_ns, var_ns, model_ns, constant_inputs_ns, vel_ao, x_ao, y_ao,
                                var_ao, model_ao, constant_inputs_ao, i, resolution_ratio)

#--------------------------------------------------------------------------------------------------------------------

def lmfit_gaussian(params, x, y):
    x00 = params['x00']
    I = params['I']
    sig = params['sig']
    Hafit = (I / sig / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (x - x00) ** 2 / sig ** 2)
    return Hafit - y

#-----------------------------------------------------------------------------------------------------------------------
#@numba.jit(nopython=True)
def ns_maps(i, inputs, constant_inputs):

    pa, inc, rflat, vflat, x0, y0, x0_unsed, y0_unused = inputs
    vel_data, pixscale, r_d_guess, l0, wav, NS_kernel, AO_kernel, z, x0_fixed, y0_fixed, inc_fixed, rflat_lim, vflat_max = constant_inputs
    y, x = np.indices([vel_data.shape[0], vel_data.shape[1]])
    r_d_min = r_d_guess * (pixscale[1] / pixscale[0])

    # Use the optimized parameters to calculate AM
    vel_model = my_vel_model(x, y, pa, inc, rflat, vflat, x0, y0, None, None, i=i, data=vel_data)[0]
    vel_model[np.isnan(vel_data)] = np.nan
    den_model = density_profile_mcmc(x, y, 1, r_d_min, x0, y0, inc, pa)[0]
    den_model[np.isnan(vel_data)] = 0
    lc = vel_model * l0 / 3e5 + l0

    # Create cube
    cube = np.zeros([len(wav), vel_model.shape[0], vel_model.shape[1]])
    for ii, jj in itertools.product(np.arange(0, vel_model.shape[0],1), np.arange(0, vel_model.shape[1],1)):
        cube[:, ii, jj] = singlegaussian([lc[ii, jj], den_model[ii, jj], 1.79], wav)

    # Convolve cube
    for k in range(len(cube[:, 1, 1])):
        #cube[k] = convolve_fft(cube[k], NS_kernel, preserve_nan=True, normalize_kernel=True, fftn=pyfftw.interfaces.scipy_fft.fft)
        #cube[k] = convolve(cube[k], NS_kernel, boundary='extend', preserve_nan=True, fill_value=np.nan,
        #                   nan_treatment='fill', normalize_kernel=True, mask=vel_data)
        cube[k] = convolve(cube[k], NS_kernel, boundary='extend', preserve_nan=True, normalize_kernel=True)#False,
                           #mask=vel_data, fill_value=np.nan, nan_treatment='fill')

    cube[np.isnan(cube)] = 0.
    vel_modelll = copy(vel_model)

    # Fit gaussian to cube
    for iii, jjj in itertools.product(np.arange(0, vel_model.shape[0], 1), np.arange(0, vel_model.shape[1], 1)):
        if cube[:, iii, jjj].any() == np.nan:
            vel_modelll[iii, jjj] = np.nan
        else:
            # X = np.linspace(wav[0], wav[-1], 500)
            # fine_cube = np.interp(X,wav,cube[:, iii, jjj])
            # x = np.sum(X * fine_cube) / np.sum(fine_cube)
            # width = np.sqrt(np.abs(np.sum((X - x) ** 2 * fine_cube) / np.sum(fine_cube)))
            # max = fine_cube.max()
            # fit = singlegaussian([x, max, width], X)
            # vel_modelll[iii, jjj] = 3e5 * (X[np.argmax(fit, axis=0)] - l0) / X[np.argmax(fit, axis=0)]

            # The proper line fitting (uncomment in Ozstar):
            p0 = [lc[iii, jjj], den_model[iii, jjj], 1.79]
            output_parameters, covariance_matrix = cf(gaussian, wav, cube[:, iii, jjj], p0=p0, ftol=0.5, xtol=0.5)
            bounds=([wav[0], 0.0, 1],[wav[-1], 10, 2]),
            vel_modelll[iii, jjj] = 3e5 * (output_parameters[0] - l0) / output_parameters[0]

        # # Using lmfit (slower)
        # params = Parameters()
        # params.add('x00', min=wav[0], max=wav[-1])
        # params.add('I', value=den_model[iii, jjj], min=0.0, max=1.0)
        # params.add('sig', value=1.79,min=1.0, max=2.0)
        #
        # fitted_params = minimize(lmfit_gaussian, params, args=(wav, cube[:, iii, jjj],), method='least_squares')
        #
        # x00 = fitted_params.params['x00'].value
        # I = fitted_params.params['I'].value
        # sig = fitted_params.params['sig'].value

        #vel_modelll[iii, jjj] = 3e5 * (x00 - l0) / x00


    vel_modelll[np.isnan(vel_data)] = np.nan

    return vel_modelll

#-----------------------------------------------------------------------------------------------------------------------

def ao_maps(i, inputs, constant_inputs):

    pa, inc, rflat, vflat, x0_unsed, y0_unused, x0, y0 = inputs
    vel_data, pixscale, r_d_guess, l0, wav, NS_kernel, AO_kernel, z, x0_fixed, y0_fixed, inc_fixed, rflat_lim, vflat_max = constant_inputs
    y, x = np.indices([vel_data.shape[0], vel_data.shape[1]])
    r_d_min = r_d_guess

    vel_model = my_vel_model(x, y, pa, inc, rflat, vflat, x0, y0, None, None, i=i, data=vel_data)[0]
    vel_model[np.isnan(vel_data)] = np.nan
    den_model = density_profile_mcmc(x, y, 1, r_d_min, x0, y0, inc, pa)[0]
    den_model[np.isnan(vel_data)] = 0

    lc = vel_model * l0 / 3e5 + l0
    cube = np.zeros([len(wav), vel_model.shape[0], vel_model.shape[1]])
    for ii, jj in itertools.product(np.arange(0, vel_model.shape[0], 1), np.arange(0, vel_model.shape[1], 1)):
        cube[:, ii, jj] = singlegaussian([lc[ii, jj], den_model[ii, jj], 1.79], wav)

    for k in range(len(cube[:, 1, 1])):
        cube[k] = convolve(cube[k], AO_kernel, boundary='extend', preserve_nan=True, normalize_kernel=True)

    cube[np.isnan(cube)] = 0.
    vel_modelll = copy(vel_model)
    for iii, jjj in itertools.product(np.arange(0, vel_model.shape[0], 1), np.arange(0, vel_model.shape[1], 1)):
        if cube[:, iii, jjj].any() == np.nan:
            vel_modelll[iii, jjj] = np.nan
        else:
            # X = np.linspace(wav[0], wav[-1], 500)
            # fine_cube = np.interp(X, wav, cube[:, iii, jjj])
            # x = np.sum(X * fine_cube) / np.sum(fine_cube)
            # width = np.sqrt(np.abs(np.sum((X - x) ** 2 * fine_cube) / np.sum(fine_cube)))
            # max = fine_cube.max()
            # fit = singlegaussian([x, max, width], X)
            # vel_modelll[iii, jjj] = 3e5 * (X[np.argmax(fit, axis=0)] - l0) / X[np.argmax(fit, axis=0)]
            # p0 = [lc[iii, jjj], den_model[iii, jjj], 1.79]
            # output_parameters, covariance_matrix = cf(gaussian, wav, cube[:, iii, jjj], p0=p0)
            # vel_modelll[iii, jjj] = 3e5 * (output_parameters[0] - l0) / output_parameters[0]

            p0 = [lc[iii, jjj], den_model[iii, jjj], 1.79]
            output_parameters, covariance_matrix = cf(gaussian, wav, cube[:, iii, jjj], p0=p0, ftol=0.5, xtol=0.5)
            bounds = ([wav[0], 0.0, 1], [wav[-1], 10, 2]),
            vel_modelll[iii, jjj] = 3e5 * (output_parameters[0] - l0) / output_parameters[0]


    vel_modelll[np.isnan(vel_data)] = np.nan

    return vel_modelll

#-----------------------------------------------------------------------------------------------------------------------

def ns_maps_make_cube(i, inputs, constant_inputs):

    pa, inc, rflat, vflat, x0, y0, x0_unsed, y0_unused = inputs
    vel_data, pixscale, r_d_guess, l0, wav, NS_kernel, AO_kernel, z, x0_fixed, y0_fixed, inc_fixed, rflat_lim, vflat_max = constant_inputs
    y, x = np.indices([vel_data.shape[0], vel_data.shape[1]])
    r_d_min = r_d_guess * (pixscale[1] / pixscale[0])

    # Use the optimized parameters to calculate AM
    vel_model = my_vel_model(x, y, pa, inc, rflat, vflat, x0, y0, None, None, i=i, data=vel_data)[0]
    den_model = density_profile_mcmc(x, y, 1, r_d_min, x0, y0, inc, pa)[0]

    lc = vel_model * l0 / 3e5 + l0
    cube = np.zeros([len(wav), vel_model.shape[0], vel_model.shape[1]])
    for ii, jj in itertools.product(np.arange(0, vel_model.shape[0], 1), np.arange(0, vel_model.shape[1], 1)):
        cube[:, ii, jj] = singlegaussian([lc[ii, jj], den_model[ii, jj], 1.79], wav)

    original_cube = copy(cube)

    for k in range(len(cube[:, 1, 1])):
        #cube[k] = convolve_fft(cube[k], NS_kernel, boundary='extend', preserve_nan=True, normalize_kernel=True)
        cube[k] = convolve(cube[k], NS_kernel, boundary='extend', preserve_nan=True, normalize_kernel=True)


    return original_cube, cube

#-----------------------------------------------------------------------------------------------------------------------

def ao_maps_make_cube(i, inputs, constant_inputs):

    pa, inc, rflat, vflat, x0_unsed, y0_unused, x0, y0 = inputs
    vel_data, pixscale, r_d_guess, l0, wav, NS_kernel, AO_kernel, z, x0_fixed, y0_fixed, inc_fixed, rflat_lim, vflat_max = constant_inputs
    y, x = np.indices([vel_data.shape[0], vel_data.shape[1]])
    r_d_min = r_d_guess

    vel_model = my_vel_model(x, y, pa, inc, rflat, vflat, x0, y0, None, None, i=i, data=vel_data)[0]
    den_model = density_profile_mcmc(x, y, 1, r_d_min, x0, y0, inc, pa)[0]

    lc = vel_model * l0 / 3e5 + l0
    cube = np.zeros([len(wav), vel_model.shape[0], vel_model.shape[1]])
    for ii, jj in itertools.product(np.arange(0, vel_model.shape[0], 1), np.arange(0, vel_model.shape[1], 1)):
        cube[:, ii, jj] = singlegaussian([lc[ii, jj], den_model[ii, jj], 1.79], wav)

    original_cube = copy(cube)

    for k in range(len(cube[:, 1, 1])):
        cube[k] = convolve(cube[k], AO_kernel, boundary='extend', preserve_nan=True, normalize_kernel=True)

    return original_cube, cube

#-------------------------------------------------------------------------------------------------------------------

def calc_j_analitically(r_d, rflat, vflat, rmax, kpc_per_pix):
    from scipy.integrate import quad

    def numerator(r, r_d, rflat, vflat):
        return (r*kpc_per_pix)**2 * 1.0*(np.exp(-r/r_d)) * vflat*(1-np.exp(-r/rflat))

    def denominator(r, r_d):
        return r*kpc_per_pix * 1.0*(np.exp(-r/r_d))

    int_num = quad(numerator, 0, rmax, args=(r_d, rflat, vflat))
    int_den = quad(denominator, 0, rmax, args=(r_d))

    j = int_num[0] / int_den[0]

    return j


#-------------------------------------------------------------------------------------------------------------------

def radial_data(data, annulus_width=1, working_mask=None, x=None, y=None, rmax=None):
    """
    r = radial_data(data,annulus_width,working_mask,x,y)

    A function to reduce an image to a radial cross-section.

    INPUT:
    ------
    data   - whatever data you are radially averaging.  Data is
            binned into a series of annuli of width 'annulus_width'
            pixels.
    annulus_width - width of each annulus.  Default is 1.
    working_mask - array of same size as 'data', with zeros at
                      whichever 'data' points you don't want included
                      in the radial data computations.
      x,y - coordinate system in which the data exists (used to set
             the center of the data).  By default, these are set to
             integer meshgrids
      rmax -- maximum radial value over which to compute statistics

     OUTPUT:
     -------
      r - a data structure containing the following
                   statistics, computed across each annulus:
          .r      - the radial coordinate used (outer edge of annulus)
          .mean   - mean of the data in the annulus
          .std    - standard deviation of the data in the annulus
          .median - median value in the annulus
          .max    - maximum value in the annulus
          .min    - minimum value in the annulus
          .numel  - number of elements in the annulus
    """

    # 2010-03-10 19:22 IJC: Ported to python from Matlab
    # 2005/12/19 Added 'working_region' option (IJC)
    # 2005/12/15 Switched order of outputs (IJC)
    # 2005/12/12 IJC: Removed decifact, changed name, wrote comments.
    # 2005/11/04 by Ian Crossfield at the Jet Propulsion Laboratory

    class radialDat:
        """Empty object container.
        """

        def __init__(self):
            self.mean = None
            self.std = None
            self.median = None
            self.numel = None
            self.max = None
            self.min = None
            self.r = None

    # ---------------------
    # Set up input parameters
    # ---------------------
    data = np.array(data)

    if working_mask == None:
        working_mask = np.ones(data.shape, bool)

    npix, npiy = data.shape
    if x == None or y == None:
        x1 = np.arange(-npix / 2., npix / 2.)
        y1 = np.arange(-npiy / 2., npiy / 2.)
        x, y = np.meshgrid(y1, x1)

    r = abs(x + 1j * y)

    if rmax == None:
        rmax = r[working_mask].max()

    # ---------------------
    # Prepare the data container
    # ---------------------
    dr = np.abs([x[0, 0] - x[0, 1]]) * annulus_width
    radial = np.arange(rmax / dr) * dr + dr / 2.
    nrad = len(radial)
    radialdata = radialDat()
    radialdata.mean = np.zeros(nrad)
    radialdata.std = np.zeros(nrad)
    radialdata.median = np.zeros(nrad)
    radialdata.numel = np.zeros(nrad)
    radialdata.max = np.zeros(nrad)
    radialdata.min = np.zeros(nrad)
    radialdata.r = radial

    # ---------------------
    # Loop through the bins
    # ---------------------
    for irad in range(nrad):  # = 1:numel(radial)
        minrad = irad * dr
        maxrad = minrad + dr
        thisindex = (r >= minrad) * (r < maxrad) * working_mask
        if not thisindex.ravel().any():
            radialdata.mean[irad] = np.nan
            radialdata.std[irad] = np.nan
            radialdata.median[irad] = np.nan
            radialdata.numel[irad] = np.nan
            radialdata.max[irad] = np.nan
            radialdata.min[irad] = np.nan
        else:
            radialdata.mean[irad] = data[thisindex].mean()
            radialdata.std[irad] = data[thisindex].std()
            radialdata.median[irad] = np.median(data[thisindex])
            radialdata.numel[irad] = data[thisindex].size
            radialdata.max[irad] = data[thisindex].max()
            radialdata.min[irad] = data[thisindex].min()

    # ---------------------
    # Return with data
    # ---------------------

    return radialdata



#----------------------------------------------------------------------------------------------------------------------

def deproject_HST_map(x, y, s_d, r_d_min, x0, y0, inc_phot, pa_phot, HST_data):

    x_phot_dep = ((x - x0) * np.cos(np.radians(pa_phot)) + (y - y0) * np.sin(np.radians(pa_phot)))  # face-on x-coordinate
    y_phot_dep = ((y - y0) * np.cos(np.radians(pa_phot)) - (x - x0) * np.sin(np.radians(pa_phot))) / np.cos(np.radians(inc_phot)) + 1e-17  # face-on y-coordinate
    r_phot_dep = np.sqrt(x_phot_dep ** 2 + y_phot_dep ** 2)

    dep_fac = np.sin(np.radians(inc_phot)) / np.sqrt(1 + (y_phot_dep / x_phot_dep) ** 2)# * np.sign(x_phot_dep)

    profile = s_d * (np.exp(-r_phot_dep / r_d_min))

    faceon_profile = density_profile(x, y, s_d, r_d_min, x0, y0)

    deprojected =  faceon_profile * dep_fac

    return profile, faceon_profile, deprojected, dep_fac


#-----------------------------------------------------------------------------------------------------------------------

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

# ----------------------------------------------------------------------------------------------------------------------

def draw_halo(data, radius, center):
    """
    get a halo patch around the peak ready to be plotted with the specified radius in pixels
    """

    if center == "max":
        max_inten_location_h = np.unravel_index(np.argmax(data), data.shape)
        center_halo = PixCoord(max_inten_location_h[1], max_inten_location_h[0])
    elif center == "middle":
        x0, y0 = data.shape[0]/2, data.shape[1]/2
        center_halo = PixCoord(y0, x0)
    halo = CirclePixelRegion(center=center_halo, radius=radius)
    patch_halo = halo.as_artist(facecolor='w', edgecolor='w', color='w', fill=False, ls="--", lw=1)

    return patch_halo

# ----------------------------------------------------------------------------------------------------------------------

def draw_psf(sigma):
    """
    get a circular PSF patch ready to be plotted with the specified radius in pixels
    """

    center_psf = PixCoord(sigma + 1, sigma + 1)
    reg = CirclePixelRegion(center=center_psf, radius=1.175 * sigma)
    patch = reg.as_artist(facecolor='w', edgecolor='k', color='w', fill=True, lw=1)

    return patch

# ----------------------------------------------------------------------------------------------------------------------

def draw_elliptical_psf(sigma_x, sigma_y, theta):
    """
    get an elliptical PSF ready to be plotted with the specified radius in pixels, the axis ratio and the angle
    """

    center = PixCoord(x=sigma_x+1, y=sigma_y+1)
    reg = EllipsePixelRegion(center=center, width=2 * sigma_x, height=2 * sigma_y, angle=theta)
    patch = reg.as_artist(facecolor='w', edgecolor='k', color='w', fill=True, lw=1)

    return patch

# ----------------------------------------------------------------------------------------------------------------------

def bin_ndarray(ndarray, new_shape, operation='mean'):

    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray

#-----------------------------------------------------------------------

# Obtain the inclination from the photometric axis ratio
def inclination_from_axis_ratio(b_a, alpha):
    theta = np.arccos(np.sqrt( ((b_a) ** 2 - alpha ** 2) / (1 - alpha ** 2) ))

    return np.degrees(theta)

#------------------------------------------------------------------------

# Make a 2D array bigger for plotting purposes
def big(map, extension, fill_with):
    big_map = np.lib.pad(map, ((extension, extension), (extension, extension)), 'constant', constant_values=(fill_with))

    return big_map


#----------------------------------------------------------------------------------------------------------------------


def print_in_terminal(**params):
    """
    Function to print the values nicely without having to specify the format everytime

    """

    from termcolor import colored

    for k, v in params.items():
        print(colored(k, "blue"), "= %.2f" % v)

#----------------------------------------------------------------------------------------------------------------------

def fits_data(fitsfile):

    """
    Simply get the data from a fits file
    """
    hdul = fits.open(fitsfile)[0].data
    return hdul

#------------------------------------------------------------------------------------------------------------------

def normalize_kernel(kernel):
    """
    Normalize the kernel so that it doesn't have to be done in the convolve routine
    """
    return kernel / kernel.sum()

#-------------------------------------------------------------------------------------------------------------------

def remove_ticks(axs):
    """
    Takes a list of axes [ax1,ax2...etc] and removes the ticks and sets a black background for easy visualization
    """
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('black')

#---------------------------------------------------------------------------------------------------------------------

def draw_kin_and_zero_vel_axes(ax, rad, ang, x0, y0):
    ax.plot(rad * np.cos(ang) + x0, rad * np.sin(ang) + y0, ls='--', c="red",
             lw=3)  # Zero-vel
    ax.plot(-rad * np.sin(ang) + x0, rad * np.cos(ang) + y0, ls='--', c="lime",
             lw=3)  # Major ax