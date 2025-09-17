import geopy.distance
import scipy
import netCDF4 as nc
import numpy as np
import skimage
from pyproj import Geod
from scipy.interpolate import RegularGridInterpolator

# def get_geodesic(p1, p2, step):
#     """Sample points along the geoid separating p1 and p2."""
#     d = geopy.distance.geodesic(p1, p2).nautical /60
#     npoints = max(int(np.ceil(d / step)), 1)
#     geoid = Geod(ellps="WGS84")
#     points = np.array(geoid.npts(p1[1], p1[0], p2[1], p2[0], npoints))  # (lon, lat)
#     real_step = d / npoints
#     return points[:, ::-1], real_step, real_step*111_000 # (lat, lon)

def get_geodesic(p1,p2, step):
    geod = Geod(ellps="WGS84")  # use WGS84 ellipsoid
    _, _, d = geod.inv(p1[1], p1[0], p2[1], p2[0])
    npoints = max(int(np.floor(d / (step*111_000 ))), 2)
    coordinates = geod.npts(p1[1], p1[0], p2[1], p2[0], npoints, initial_idx=1, terminus_idx=1) # (lon, lat)
    real_step_m = d / npoints #m
    real_step = real_step_m / 111_000
    coordinates = np.reshape(coordinates, (npoints,2))
    return coordinates[:, ::-1], real_step, real_step_m


def load_NetCDF(NetCDF_path, data_name, lat_bounds, lon_bounds, lat_name="lat", lon_name="lon"):
    """Load data from a NetCDF file with given bounds."""
    data = nc.Dataset(NetCDF_path)

    lat_resolution = data.variables[lat_name][1] - data.variables[lat_name][0]
    lon_resolution = data.variables[lon_name][1] - data.variables[lon_name][0]

    # index bounds in the lat/lon grid given the lat/lon bounds and the resolution
    lat_bounds = (max(0,int((lat_bounds[0] - data.variables[lat_name][0]) / lat_resolution)),
                  max(0,int((lat_bounds[1] - data.variables[lat_name][0]) / lat_resolution)))
    lon_bounds = (max(0,int((lon_bounds[0] - data.variables[lon_name][0]) / lon_resolution)),
                  max(0,int((lon_bounds[1] - data.variables[lon_name][0]) / lon_resolution)))

    lat, lon = (
        np.array(data.variables[lat_name])[lat_bounds[0]:lat_bounds[1] + 1],
        np.array(data.variables[lon_name])[lon_bounds[0]:lon_bounds[1] + 1],
    )

    grid = np.array(
        data.variables[data_name][lat_bounds[0]:lat_bounds[1]+1, lon_bounds[0]:lon_bounds[1]+1]
        .filled(fill_value=np.nan)
    )
    if len(grid.shape) == 3:  # remove unneeded temporal axis
        grid = grid[0]

    return grid, lat, lon, data


def reduce_grid(grid, lat, lon, reduction_factor, reduction_function):
    """Reduce grid size by block reduction."""
    block = (2 ** reduction_factor, 2 ** reduction_factor) if len(grid.shape) == 2 else \
        (2 ** reduction_factor, 2 ** reduction_factor, 1)
    data = skimage.measure.block_reduce(grid, block, reduction_function)
    lat = skimage.measure.block_reduce(lat, 2 ** reduction_factor, np.mean)
    lon = skimage.measure.block_reduce(lon, 2 ** reduction_factor, np.mean)
    return data, lat, lon

class BidimensionalGrid():
    def __init__(self, data, lat, lon, method="nearest"):
        """ Generic physics grid class, giving values for each lat/lon.
        :param data: Grid data as a 2D/3D array.
        :param lat: Latitudes represented in the grid.
        :param lon: Longitudes represented in the grid.
        """
        self.data, self.lat, self.lon = data, lat, lon
        self.lat_bounds, self.lon_bounds = (np.min(lat), np.max(lat)), (np.min(lon), np.max(lon))
        self.lat_resolution = lat[1] - lat[0]
        self.lon_resolution = lon[1] - lon[0]
        self.method = method
        self._build_interpolator()

    def _build_interpolator(self):
        """Initialize SciPy interpolator."""
        self._interpolator = RegularGridInterpolator(
            (self.lat, self.lon),
            self.data,
            method=self.method,
            bounds_error=False,
            fill_value=np.nan,
        )

    @classmethod
    def create_from_NetCDF(cls, NetCDF_path, data_name, lat_bounds=None, lon_bounds=None,
                           reduction_factor=0, reduction_function=np.mean, method="nearest"):
        lat_bounds, lon_bounds = lat_bounds or [-90, 90], lon_bounds or [-180, 180]
        grid, lat, lon, _ = load_NetCDF(NetCDF_path, data_name, lat_bounds, lon_bounds)
        grid, lat, lon = reduce_grid(grid, lat, lon, reduction_factor, reduction_function)
        return cls(grid, lat, lon, method=method)

    def check_positions(self, pos):
        """Ensure requested positions are inside the grid."""
        for p in pos:
            if not (self.lat[0] <= p[0] <= self.lat[-1] and self.lon[0] <= p[1] <= self.lon[-1]):
                raise ValueError(f"Point {p} outside grid bounds. pos : {pos}")


    def get_interpolated_values(self, coordinates, method=None):
        """Return interpolated values at given coordinates."""
        if method is not None and method != self.method:
            interpolator = RegularGridInterpolator(
                (self.lat, self.lon), self.data, method=method, bounds_error=False, fill_value=np.nan
            )
            return interpolator(coordinates)
        return self._interpolator(coordinates)

    def get_along_path(self, pos1, pos2, step=None, method=None, which="data", step_m = False):
        """
        Sample values along a geodesic path.

        :param pos1: First point (lat, lon).
        :param pos2: Second point (lat, lon).
        :param step: Step in degrees. Defaults to grid resolution.
        :param method: Interpolation method ('linear', 'nearest', 'cubic').
        :param which: 'data' (sound velocity) or 'sv_err' (uncertainty).
        :return: (values, actual_step)
        """
        self.check_positions([pos1, pos2])
        step = np.sqrt(self.lat_resolution ** 2 + self.lon_resolution ** 2) if step is None else step
        points, actual_step, actual_step_m = get_geodesic(pos1, pos2, step)

        # choisir le bon dataset
        if which == "data":
            dataset = self.data
        elif which == "sv_err":
            if self.sv_err is None:
                raise ValueError("SV_ERR data not available for this grid")
            dataset = self.sv_err
        else:
            raise ValueError("Invalid 'which' parameter, use 'data' or 'sv_err'")

        interpolator = RegularGridInterpolator(
            (self.lat, self.lon), dataset,
            method=method or self.method,
            bounds_error=False, fill_value=np.nan
        )
        values = interpolator(points)

        if step_m :
            return values, actual_step, actual_step_m

        return values, actual_step


    def save_as_NetCDF(self, path, description, data_name):
        """Save grid to NetCDF file."""
        root_grp = nc.Dataset(path, 'w', format='NETCDF4')
        root_grp.description = description

        root_grp.createDimension('lat', len(self.lat))
        root_grp.createDimension('lon', len(self.lon))

        lat_var = root_grp.createVariable('lat', 'f4', ('lat',))
        lat_var[:] = self.lat
        lon_var = root_grp.createVariable('lon', 'f4', ('lon',))
        lon_var[:] = self.lon
        data_var = root_grp.createVariable(data_name, 'f4', ("lat", "lon"))
        data_var[:, :] = self.data

        root_grp.close()


class SoundVelocityGrid(BidimensionalGrid):
    def __init__(self, data, lat, lon, sv_err=None, method="nearest"):
        super().__init__(data, lat, lon, method=method)
        self.sv_err = sv_err

    @classmethod
    def create_from_NetCDF(Grid, NetCDF_path, lat_bounds=None, lon_bounds=None,
                           interpolate=False, method="nearest"):
        lat_bounds, lon_bounds = lat_bounds or [-90, 90], lon_bounds or [-180, 180]
        grid, lat, lon, _ = load_NetCDF(NetCDF_path, "celerity", lat_bounds, lon_bounds)
        return Grid(grid, lat, lon, method=method)

    @classmethod
    def create_from_ISAS(Grid, NetCDF_path, lat_bounds=None, lon_bounds=None,
                         interpolate=False, method="nearest"):
        lat_bounds, lon_bounds = lat_bounds or [-90, 90], lon_bounds or [-180, 180]

        grid, lat, lon, _ = load_NetCDF(NetCDF_path, 'SOUND_VELOCITY',
                                        lat_bounds, lon_bounds,
                                        lat_name="latitude", lon_name="longitude")

        sv_err_grid, _, _, _ = load_NetCDF(NetCDF_path, 'SV_ERR',
                                           lat_bounds, lon_bounds,
                                           lat_name="latitude", lon_name="longitude")

        return Grid(grid, lat, lon, sv_err=sv_err_grid, method=method)

    def get_sound_speed(self, pos1, pos2, method=None):
        """Harmonic mean of sound speed along path."""
        if self.method == "nearest":
            velocities = self.get_along_path(pos1, pos2, method="nearest")[0]
        else:
            velocities = self.get_along_path(pos1, pos2, method=method)[0]

        velocities = velocities[~np.isnan(velocities)]
        return scipy.stats.hmean(velocities) if len(velocities) > 0 else np.nan

    def harmonic_mean_with_uncertainty(self, c, sigma_c, ds):
        """
        Harmonic mean of sound speed with propagated uncertainty.
        :param c: array of sound speeds [m/s]
        :param sigma_c: array of uncertainties (absolute, same units as c)
        :param ds: step length along path [m]
        :return: (c_eq, sigma_c_eq)
        """
        L = len(c) * ds
        T = np.sum(ds / c)  # total "time"
        c_eq = L / T
        # propagation of uncertainty
        coeffs = (ds / c ** 2) * sigma_c
        sigma_c_eq = (L / T ** 2) * np.sqrt(np.sum(coeffs ** 2))
        return c_eq, sigma_c_eq

    def get_sound_speed_with_uncertainty(self, pos1, pos2, method="nearest"):
        """Harmonic mean + uncertainty estimate."""
        if self.sv_err is None:
            return self.get_sound_speed(pos1, pos2, method=method), None

        velocities,actual_step, actual_step_m = self.get_along_path(pos1, pos2, method=method, which="data", step_m=True)
        uncertainties = self.get_along_path(pos1, pos2, method=method, which="sv_err")[0]

        valid_mask = ~(np.isnan(velocities) | np.isnan(uncertainties))
        velocities = velocities[valid_mask]
        uncertainties = uncertainties[valid_mask]

        if len(velocities) == 0:
            return np.nan, np.nan

        harmonic_mean, mean_uncertainty = self.harmonic_mean_with_uncertainty(velocities, uncertainties,actual_step_m)
        return harmonic_mean, mean_uncertainty
