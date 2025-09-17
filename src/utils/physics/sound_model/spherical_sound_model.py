import numpy as np
from pyproj import Geod
from utils.physics.constants import EARTH_RADIUS
from utils.physics.sound_model.sound_model import SoundModel
from utils.physics.sound_model.sound_velocity_grid import SoundVelocityGrid
from utils.physics.sound_model.ISAS_grid import load_ISAS_extracted

# sound model considering earth as a perfect sphere
class SphericalSoundModel(SoundModel):
    # t0 + di/c = ti (origin time + propagation time = detection time)
    # cost is expressed as ((t0 + di/c) - ti)^2
    # here di is computed with the great-circle distance formula considering the Earth as a perfect sphere
    # (see https://en.wikipedia.org/wiki/Great-circle_distance for more details)
    def _get_cost_function(self, sensor_positions, detection_times, velocities):
        def f(v):
            costs = np.zeros(len(sensor_positions))
            for i in range(len(sensor_positions)):
                ci = velocities[i]
                xi, yi = sensor_positions[i][0], sensor_positions[i][1]
                ti = detection_times[i]
                t0, x, y = v[0], v[1], v[2]

                delta_sigma = np.arccos(np.sin(x) * np.sin(xi) + np.cos(x) * np.cos(xi) * np.cos(np.abs(y - yi)))
                di = delta_sigma * EARTH_RADIUS

                costs[i] = ((t0 + di / ci) - ti) ** 2
            return costs
        return f

    # see comments on _get_cost_function for cost formula
    # the jacobian is then the derivatives of the cost with respect to the three variables (t0 and the 2 coordinates)
    def _get_jacobian_function(self, sensor_positions, detection_times, velocities):
        def jacobian(v):
            mat = np.zeros((len(sensor_positions), 3))
            for i in range(len(sensor_positions)):
                ci = velocities[i]
                xi, yi = sensor_positions[i][0], sensor_positions[i][1]
                ti = detection_times[i]
                t0, x, y = v[0], v[1], v[2]

                arc = np.sin(x) * np.sin(xi) + np.cos(x) * np.cos(xi) * np.cos(np.abs(y - yi))
                delta_sigma = np.arccos(arc)

                mat[i, 0] = 2 * t0 + 2 * delta_sigma * EARTH_RADIUS / ci - 2 * ti

                d_arc_dx = np.cos(x) * np.sin(xi) - np.sin(x) * np.cos(xi) * np.cos(np.abs(y - yi))
                d_sig_dx = - d_arc_dx / np.sqrt(1 - arc ** 2)
                d_sig2_dx = 2 * d_sig_dx * delta_sigma
                mat[i, 1] = (1 / ci**2) * (d_sig2_dx * EARTH_RADIUS**2) + (1/ci) * (d_sig_dx * EARTH_RADIUS) * 2 * (t0-ti)

                d_arc_dy = - np.cos(x) * np.cos(xi) * np.sin(np.abs(y - yi)) * np.sign(y - yi)
                d_sig_dy = - d_arc_dy / np.sqrt(1 - arc ** 2)
                d_sig2_dy = 2 * d_sig_dy * delta_sigma
                mat[i, 2] = (1 / ci**2) * (d_sig2_dy * EARTH_RADIUS**2) + (1/ci) * (d_sig_dy * EARTH_RADIUS) * 2 * (t0-ti)

            return mat
        return jacobian

    def _get_sound_travel_time(self, pos1, pos2, date=None, sound_velocity=None):
        x, y = pos1[0], pos1[1]
        xi, yi = pos2[0], pos2[1]
        arc = np.sin(x) * np.sin(xi) + np.cos(x) * np.cos(xi) * np.cos(np.abs(y - yi))
        delta_sigma = np.arccos(arc)
        di = delta_sigma * EARTH_RADIUS
        return di / sound_velocity

    # deg to radians
    def _transform_coordinates(self, pos):
        return np.deg2rad(pos)

    # radians to deg
    def _transform_coordinates_reverse(self, pos):
        return np.rad2deg(pos)


# constant velocity
class HomogeneousSphericalSoundModel(SphericalSoundModel):
    # sound speed in m/s
    def __init__(self, sound_speed=1480):
        self.sound_speed = sound_speed
        super().__init__()

    def get_sound_speed(self, source, sensor, date=None):
        return self.sound_speed


# WOA23-derived velocity
class GridSphericalSoundModel(SphericalSoundModel):
    def __init__(self, velocity_grid_paths, constant_velocity=1480, lat_bounds=None, lon_bounds=None):
        self.models = [SoundVelocityGrid.create_from_NetCDF(p, lat_bounds, lon_bounds) for p in velocity_grid_paths]
        self.constant_velocity = constant_velocity
        super().__init__()

    def get_sound_speed(self, source, sensor, date=None):
        return self.models[date.month-1].get_sound_speed(source, sensor)

    def localize_common_source(self, sensors_positions, detection_times, x_min=-90, y_min=-180, x_max=90,
                             y_max=180, t_min=-36_000, initial_pos=None, velocities=None):
        l = self._localize_common_source(sensors_positions, detection_times, x_min, y_min, x_max,
                             y_max, t_min, initial_pos, len(sensors_positions) * [self.constant_velocity])
        return self._localize_common_source(sensors_positions, detection_times, x_min, y_min, x_max,
                             y_max, t_min, l.x)