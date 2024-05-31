#COSMOLOGY CALCULATOR P.GURRI '2017'
#------------------------------------
#functions: (z is an array [1,2,3,4])
#ageuniverse() -- [Gyr](age of Universe)
#lookbacktime(z) -- [Gyr](time it took for light to arrive from z)
#timez(z) [Gyr] -- [Gyr](age when light was emitted from z)
#comovingdistance(z) -- [Mpc](comovingdistance to z)
#angulardistance(z) -- [Mpc](angulardistance to z)
#comovingvolume(z) -- [Gpc^3](comovingvolume to z)
#luminositydistance(z) [Mpc]-- (luminositydistance to z)

#import sys
#sys.path.append('/Users/polgurri/Desktop/SWINBURNE/coding/cosmology')
#import cosmo

from scipy.integrate import quad as _quad
from math import sqrt as _sqrt
from math import pi as _pi
import numpy as _np

#HardCoded values (possible to modify)
_H0   = 70
_Om_m = 0.3
_Om_v = 0.7
_Om_r = 0 #8.24 * 10**-5    #Carroll & Ostlie, 2007.

#Exact values
_c = 299792.458                      #km/s (exactly)
_Mpc = 648000 * 149597870700 / _pi    #m (exactly)
_JYear = 365.25 * 24 * 3600          #s (exacly)

#Definitions
_DH = _c / _H0
_tH = (_Mpc * 10**-6 / _JYear) / _H0
_Om_k = 1 - _Om_m - _Om_v - _Om_r


def lookbacktime(z):
    f = lambda t: _cosm_lbackt(t)
    vfunc = _np.vectorize(f)
    return _np.array(vfunc(z))

def comovingdistance(z):
    f = lambda t: _cosm_dist(t)
    vfunc = _np.vectorize(f)
    return _np.array(vfunc(z))

def ageuniverse():
    f = _quad(_cosm_lback, 0, 1, args=(_Om_k, _Om_m, _Om_v, _Om_r, _tH))
    return f[0]

def timez(z):
    f = lambda t: lookbacktime(t)
    vfunc = _np.vectorize(f)
    return ageuniverse() - vfunc(z)

def angulardistance(z):
    z = _np.array(z)
    az = 1 / (1 + z)
    if _Om_k == 0: return comovingdistance(z) * az
    elif _Om_k > 0:
        xx = _sqrt(abs(Om_k))
        return az * _DH / xx * _np.sinh(xx * comovingdistance(z)/_DH)
    elif _Om_k < 0:
        xx = _sqrt(abs(Om_k))
        return az * _DH / xx * _np.sin(xx * comovingdistance(z)/_DH)

def comovingvolume(z):
    z = _np.array(z)
    az = 1 / (1 + z)
    if _Om_k == 0: return 10**-9 * 4 * _pi * comovingdistance(z)**3 / 3
    elif _Om_k > 0:
        xx = _sqrt(abs(Om_k))
        y = 4 * _pi * _DH**3 / (2 * _Om_k)
        jj = comovingdistance(z) / DH * _np._sqrt(
            1 + _Om_k * comovingdistance(z) / _DH * comovingdistance(z) / _DH)
        return 10**-9 * y * (
            jj - _np.arcsinh(xx * comovingdistance(z) / _DH) / xx)
    elif _Om_k < 0:
        xx = _sqrt(abs(_Om_k))
        y = 4 * _pi * _DH**3 / (2 * _Om_k)
        jj = comovingdistance(z) / _DH * _np._sqrt(
            1 + _Om_k * comovingdistance(z) / _DH * comovingdistance(z) / _DH)
        return 10**-9 * y * (jj - _np.arcsin(
            xx * comovingdistance(z) / _DH) / xx)


def luminositydistance(z):
    z = _np.array(z)
    return (1+z) * comovingdistance(z)


def distancemodulus(z):
    f = 5 * _np.log10(luminositydistance(z)) + 25
    return f

# -----------------------------------------------------------------------------
# -- Internal functions -------------------------------------------------------
# -----------------------------------------------------------------------------

def _cosm_lback(a, Om_k, _Om_m, _Om_v, _Om_r, _tH):
    f = _sqrt(_Om_k + _Om_m/a + _Om_r/(a*a) + _Om_v*a*a)
    return _tH/f

def _cosm_ldist(a, Om_k, _Om_m, _Om_v, _Om_r, _tH):
    f = a * _sqrt(_Om_k + _Om_m/a + _Om_r/(a*a) + _Om_v*a*a)
    return _DH/f

def _cosm_lbackt(zz):
    az = 1 / (1 + zz)
    lookbacktime = _quad(_cosm_lback, az, 1,
        args=(_Om_k, _Om_m, _Om_v, _Om_r, _tH))
    return lookbacktime[0]

def _cosm_dist(zz):
    az = 1 / (1 + zz)
    comovingdistance = _quad(_cosm_ldist, az, 1,
        args=(_Om_k, _Om_m, _Om_v, _Om_r, _tH))
    return comovingdistance[0]

#
