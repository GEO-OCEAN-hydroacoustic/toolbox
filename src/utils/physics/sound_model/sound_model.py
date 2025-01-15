import numpy as np
from scipy.optimize import least_squares

# abstract class of a SoundModel, defining what a SoundModel should do
# note : every coordinate is given in decimal degrees (e.g. (-45.2345, 72.6789))
class SoundModel():
    # method to modify the coordinates from decimal degrees to intern computation system
    def _transform_coordinates(self, pos):
        return pos

    # method to modify the coordinates from intern computation system to decimal degrees
    def _transform_coordinates_reverse(self, pos):
        return pos

    # return the time, in s, that a sound emitted at one of the positions would require to reach the other one
    def get_sound_travel_time(self, pos1, pos2, date=None):
        pos1, pos2 = self._transform_coordinates(pos1), self._transform_coordinates(pos2)
        return self._get_sound_travel_time(pos1, pos2, date)

    # same as get_sound_travel_time but with input in intern computation system
    def _get_sound_travel_time(self, pos1, pos2, date=None):
        return None

    # return a function giving, for a guessed position and source time, a cost
    # also considers a velocity per sensor position
    def _get_cost_function(self, sensors_positions, detection_times, velocities):
        return None

    # return a function giving, for a guessed position and source time, a jacobian
    # also considers a velocity per sensor position
    def _get_jacobian_function(self, sensors_positions, detection_times, velocities):
        return None

    # given detection times and positions, try to locate the source
    def localize_common_source(self, sensors_positions, detection_times, x_min=-90, y_min=-180, x_max=90,
                             y_max=180, t_min=-36_000, initial_pos=None, velocities=None):
        return self._localize_common_source(sensors_positions, detection_times, x_min, y_min, x_max,
                             y_max, t_min, initial_pos, velocities)


    def _localize_common_source(self, sensors_positions, detection_times, x_min=-90, y_min=-180, x_max=90,
                             y_max=180, t_min=-36_000, initial_pos=None, velocities=None):
        min_date = np.argmin(detection_times)
        detection_times = np.array([(d-detection_times[min_date]).total_seconds() for d in detection_times])

        sensors_positions = np.array([self._transform_coordinates(p) for p in sensors_positions])
        x_min, y_min, x_max, y_max = (self._transform_coordinates(x_min), self._transform_coordinates(y_min),
                                      self._transform_coordinates(x_max), self._transform_coordinates(y_max))
        if initial_pos is None:
            initial_pos = np.mean(sensors_positions, axis=0)
        else:
            initial_pos = self._transform_coordinates(initial_pos)

        x0 = [0, *initial_pos]
        x0[0] = -self._get_sound_travel_time(x0[1:], sensors_positions[min_date], min_date)

        f = self._get_cost_function(sensors_positions, detection_times, velocities)
        jacobian = self._get_jacobian_function(sensors_positions, detection_times, velocities)

        res = least_squares(f, x0, bounds=([t_min, x_min, y_min], [0, x_max, y_max]),
                            jac=jacobian, method="dogbox", ftol=1e-12, xtol=1e-12, gtol=1e-12)

        res.x[1:] = self._transform_coordinates_reverse(res.x[1:])

        return res