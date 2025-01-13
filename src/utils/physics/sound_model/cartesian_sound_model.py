from utils.physics.sound_model.sound_model import SoundModel
import numpy as np


# sound model of homogeneous sound velocity considering earth as a plane
class HomogeneousSoundModel(SoundModel):
    # sound speed in m/s
    def __init__(self, sound_speed=1480):
        super().__init__()
        self.sound_speed = sound_speed

    # t0 + di/c = ti (origin time + propagation time = detection time)
    # cost is expressed as ((t0 + di/c) - ti)^2
    # here di is a simple cartesian plane distance
    def _get_cost_function(self, sensor_positions, detection_times, velocities):
        def f(v):
            costs = np.zeros(len(sensor_positions))
            for i in range(len(sensor_positions)):
                c = velocities[i]
                xi, yi = sensor_positions[i][0], sensor_positions[i][1]
                ti = detection_times[i]
                t0, x, y = v[0], v[1], v[2]
                di = np.sqrt((x - xi) ** 2 + (y - yi) ** 2)

                costs[i] = ((t0 + di / c) - ti) ** 2
            return costs
        return f

    # see comments on _get_cost_function for cost formula
    # the jacobian is then the derivatives of the cost with respect to the three variables (t0 and the 2 coordinates
    def _get_jacobian_function(self, sensor_positions, detection_times, velocities):
        def jacobian(v):
            mat = np.zeros((len(sensor_positions), 3))
            for i in range(len(sensor_positions)):
                c = velocities[i]
                xi, yi = sensor_positions[i][0], sensor_positions[i][1]
                ti = detection_times[i]
                t0, x, y = v[0], v[1], v[2]
                di = np.sqrt((x - xi) ** 2 + (y - yi) ** 2)

                mat[i, 0] = 2 * t0 + 2 * di / c - 2 * ti
                mat[i, 1] = (2 * x - 2 * xi) / (c**2) + (x - xi) * 2 * (t0 - ti) / (c * di)
                mat[i, 2] = (2 * y - 2 * yi) / (c**2) + (y - yi) * 2 * (t0 - ti) / (c * di)
            return mat
        return jacobian

    def _get_sound_travel_time(self, pos1, pos2, date=None):
        c = self.sound_speed
        x, y = pos1[0], pos1[1]
        xi, yi = pos2[0], pos2[1]
        di = np.sqrt((x - xi) ** 2 + (y - yi) ** 2)
        return di / c

    # deg to m
    def _transform_coordinates(self, pos):
        new_pos = np.copy(pos)
        if type(pos) == list or type(pos) == np.array:
            new_pos[0] *= 111_000
            new_pos[1] *= 111_000
        else:
            new_pos *= 111_000
        return new_pos

    # m to deg
    def _transform_coordinates_reverse(self, pos):
        new_pos = np.copy(pos)
        if type(pos) == list or type(pos) == np.array:
            new_pos[0] /= 111_000
            new_pos[1] /= 111_000
        else:
            new_pos /= 111_000
        return new_pos

    def localize_common_source(self, sensors_positions, detection_times, x_min=-90, y_min=-180, x_max=90,
                             y_max=180, t_min=-36_000, initial_pos=None, velocities=None):
        velocities = len(sensors_positions) * [self.sound_speed]
        return self._localize_common_source(sensors_positions, detection_times, x_min, y_min, x_max,
                             y_max, t_min, initial_pos, velocities)
