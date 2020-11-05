import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from astropy_healpix import HEALPix
from astropy.coordinates import ICRS

import os
from shapely.geometry import Polygon, MultiPolygon

class SkyPlot:

    def __init__(self, figsize=(10, 10), projection=ccrs.Mollweide()):

        self.fig = plt.figure(figsize=figsize)
        self.ax = plt.axes(projection=projection)


    def plot_decam(self, ra, dec, zorder=1, color='#1f77b4', alpha=0.5):
        decam = DECamFocalPlane()
        corners = decam.rotate(ra, dec)
        corners.T[0][corners.T[0] > 180] -= 360
        corners.T[0] *= -1
        decam_poly = MultiPolygon(Polygon(corners[idx]) for idx in range(len(corners)))
        self.ax.add_geometries([decam_poly], crs=ccrs.PlateCarree(), facecolor=color,
                               alpha=alpha, edgecolor='black', zorder=zorder)
        return self

    def plot_hpix(self, hpix, nside, order='nested', color='blue', edgecolor='black', zorder=1, alpha=1):
        bounds = HEALPix(nside=nside, order=order, frame=ICRS())
        corners = bounds.boundaries_skycoord(np.unique(hpix), step=1)
        poly = MultiPolygon(Polygon(np.array([-(corners[idx].ra.deg - 360 * (corners[idx].ra.deg > 180)), corners[idx].dec.deg]).T) \
                for idx in range(len(corners.ra)) if not any(abs(corners[idx].ra.deg - 360 * (corners[idx].ra.deg > 180) - 180) == 0))
        self.ax.add_geometries([poly], crs=ccrs.PlateCarree(), facecolor=color,
                               alpha=alpha, edgecolor=edgecolor, zorder=zorder)


    def plot_ecliptic(self, color='red'):
        ecl = SkyCoord(lon=np.linspace(0, 2*np.pi, 10000) * u.rad, lat=np.zeros(10000) * u.rad, frame='geocentrictrueecliptic')
        ecl_order = ecl.icrs.ra.deg.argsort()
        self.ax.plot(-ecl.icrs.ra.deg[ecl_order], ecl.icrs.dec.deg[ecl_order], color=color, transform=ccrs.PlateCarree())


    def plot_galactic(self, color='red'):
        gal = SkyCoord(l=np.linspace(0, 2*np.pi, 10000) * u.rad, b=np.zeros(10000) * u.rad, frame='galactic')
        gal_order = gal.icrs.ra.deg.argsort()
        self.ax.plot(-gal.icrs.ra.deg[gal_order], gal.icrs.dec.deg[gal_order], color=color, transform=ccrs.PlateCarree())


    def scatter(self, ra, dec, color='black', alpha=0.5, zorder=1, s=5):
        self.ax.scatter(-ra, dec, color=color, alpha=alpha, transform=ccrs.PlateCarree(), zorder=zorder, s=s)


    def set_extent(self, bounds):
        ramin = -bounds[0]
        ramax = -bounds[1]
        decmin = bounds[2]
        decmax = bounds[3]
        self.ax.set_extent([ramin, ramax, decmin, decmax])

class DECamFocalPlane(object):
    """Class for storing and manipulating the corners of the DECam CCDs.
    """

    filename = '/Users/kjnapier/research/cartosky/ccd_corners_xy_fill.dat'

    def __init__(self):
        # This is not safe. Use yaml instead (extra dependency)
        self.ccd_dict = eval(''.join(open(self.filename).readlines()))

        # These are x,y coordinates
        self.corners = np.array(list(self.ccd_dict.values()))

        # Since we don't know the original projection of the DECam
        # focal plane into x,y it is probably not worth trying to
        # deproject it right now...

        #x,y = self.ccd_array[:,:,0],self.ccd_array[:,:,1]
        #ra,dec = Projector(0,0).image2sphere(x.flat,y.flat)
        #self.corners[:,:,0] = ra.reshape(x.shape)
        #self.corners[:,:,1] = dec.reshape(y.shape)

    def rotate(self, ra, dec):
        """Rotate the corners of the DECam CCDs to a given sky location.
        Parameters:
        -----------
        ra      : The right ascension (deg) of the focal plane center
        dec     : The declination (deg) of the focal plane center
        Returns:
        --------
        corners : The rotated corner locations of the CCDs
        """
        corners = np.copy(self.corners)

        R = SphericalRotator(ra,dec)
        _ra,_dec = R.rotate(corners[:,:,0].flat,corners[:,:,1].flat,invert=True)

        corners[:,:,0] = _ra.reshape(corners.shape[:2])
        corners[:,:,1] = _dec.reshape(corners.shape[:2])
        return corners

    def project(self, basemap, ra, dec):
        """Apply the given basemap projection to the DECam focal plane at a
        location given by ra,dec.
        Parameters:
        -----------
        basemap : The Basemap to project to.
        ra      : The right ascension (deg) of the focal plane center
        dec     : The declination (deg) of the focal plane center
        Returns:
        --------
        corners : Projected corner locations of the CCDs
        """
        corners = self.rotate(ra,dec)

        x,y = basemap.proj(corners[:,:,0],corners[:,:,1])

        # Remove CCDs that cross the map boundary
        x[(np.ptp(x,axis=1) > np.pi)] = np.nan

        corners[:,:,0] = x
        corners[:,:,1] = y
        return corners

class SphericalRotator:
    """
    Base class for rotating points on a sphere.
    The input is a fiducial point (deg) which becomes (0, 0) in rotated coordinates.
    """

    def __init__(self, lon_ref, lat_ref, zenithal=False):
        self.setReference(lon_ref, lat_ref, zenithal)

    def setReference(self, lon_ref, lat_ref, zenithal=False):

        if zenithal:
            phi = (np.pi / 2.) + np.radians(lon_ref)
            theta = (np.pi / 2.) - np.radians(lat_ref)
            psi = 0.
        if not zenithal:
            phi = (-np.pi / 2.) + np.radians(lon_ref)
            theta = np.radians(lat_ref)
            psi = np.radians(90.) # psi = 90 corresponds to (0, 0) psi = -90 corresponds to (180, 0)


        cos_psi,sin_psi = np.cos(psi),np.sin(psi)
        cos_phi,sin_phi = np.cos(phi),np.sin(phi)
        cos_theta,sin_theta = np.cos(theta),np.sin(theta)

        self.rotation_matrix = np.matrix([
            [cos_psi * cos_phi - cos_theta * sin_phi * sin_psi,
             cos_psi * sin_phi + cos_theta * cos_phi * sin_psi,
             sin_psi * sin_theta],
            [-sin_psi * cos_phi - cos_theta * sin_phi * cos_psi,
             -sin_psi * sin_phi + cos_theta * cos_phi * cos_psi,
             cos_psi * sin_theta],
            [sin_theta * sin_phi,
             -sin_theta * cos_phi,
             cos_theta]
        ])

        self.inverted_rotation_matrix = np.linalg.inv(self.rotation_matrix)

    def cartesian(self,lon,lat):
        lon = np.radians(lon)
        lat = np.radians(lat)

        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z =  np.sin(lat)
        return np.array([x,y,z])


    def rotate(self, lon, lat, invert=False):
        vec = self.cartesian(lon,lat)

        if invert:
            vec_prime = np.dot(np.array(self.inverted_rotation_matrix), vec)
        else:
            vec_prime = np.dot(np.array(self.rotation_matrix), vec)

        lon_prime = np.arctan2(vec_prime[1], vec_prime[0])
        lat_prime = np.arcsin(vec_prime[2])

        return (np.degrees(lon_prime) % 360.), np.degrees(lat_prime)
