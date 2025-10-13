import numpy as np
from pyproj import Geod
import datetime
from scipy.optimize import least_squares
from utils.physics.sound_model.sound_model import SoundModel
from utils.physics.sound_model.sound_velocity_grid import SoundVelocityGrid

# sound model considering earth as an ellipsoid (WGS84)
class EllipsoidalSoundModel(SoundModel):
    def __init__(self):
        self.geod = Geod(ellps="WGS84")
        super().__init__()

    def get_distance(self, pos1, pos2):
        """Calculate geodesic distance using WGS84 ellipsoid"""
        az12, az21, distance = self.geod.inv(pos1[1], pos1[0], pos2[1], pos2[0])  # lon, lat
        return distance

    def _get_sound_travel_time(self, pos1, pos2, date=None, sound_velocity=None):
        """Calculate travel time using ellipsoidal distance"""
        distance = self.get_distance(pos1, pos2)
        return distance / sound_velocity

    def get_sound_travel_time(self, pos1, pos2, date=None):
        """Public method for calculating sound travel time"""
        velocity = self.get_sound_speed(pos1, pos2, date)
        return self._get_sound_travel_time(pos1, pos2, date, velocity)

    def _theoretical_derivatives(self, lat1, lon1, lat2, lon2):
        """
        Compute derivatives of geodesic distance with respect to endpoint 2
        All inputs must be scalars or same-sized arrays
        Returns: ds/dlat2, ds/dlon2 in meters per radian
        """
        # Get geodesic parameters - all inputs must have same shape
        fwd_azi, back_azi, distance = self.geod.inv(lon1, lat1, lon2, lat2)

        # Convert to radians
        lat2_rad = np.radians(lat2)
        # Use back azimuth at point 2 (azi2 in our notation)
        azi2_rad = np.radians(back_azi)

        # WGS84 parameters
        a = self.geod.a  # semi-major axis
        f = self.geod.f  # flattening
        e2 = f * (2 - f)  # first eccentricity squared

        # Trigonometric components
        sin_lat2 = np.sin(lat2_rad)
        cos_lat2 = np.cos(lat2_rad)
        sin_azi2 = np.sin(azi2_rad)
        cos_azi2 = np.cos(azi2_rad)

        # Radii of curvature at point 2
        nu2 = 1 - e2 * sin_lat2 ** 2
        N2 = a / np.sqrt(nu2)  # Normal radius (east-west)
        M2 = a * (1 - e2) / (nu2 ** 1.5)  # Meridional radius (north-south)

        # Derivatives in meters per radian
        ds_dlat2 = M2 * cos_azi2  # ρ₂ cos(α₂)
        ds_dlon2 = N2 * cos_lat2 * sin_azi2  # ν₂ cos(φ₂) sin(α₂)

        return ds_dlat2, ds_dlon2

    def _get_cost_function(self, sensor_positions, detection_times, velocities):
        """TDOA residuals using first sensor as reference (returns residuals, not squared)"""

        def f(v):
            source_lat, source_lon = v[0], v[1]  # No t0 in TDOA
            n_eq = len(sensor_positions) - 1
            residuals = np.zeros(n_eq)

            # Reference sensor (index 0)
            ref_pos = sensor_positions[0]
            ref_time = detection_times[0]
            ref_velocity = velocities[0]
            ref_distance = self.get_distance([source_lat, source_lon], ref_pos)

            for i in range(1, len(sensor_positions)):
                sensor_pos = sensor_positions[i]
                sensor_time = detection_times[i]
                sensor_velocity = velocities[i]
                sensor_distance = self.get_distance([source_lat, source_lon], sensor_pos)

                # residual: predicted_tdoa - observed_tdoa
                predicted_tdoa = sensor_distance / sensor_velocity - ref_distance / ref_velocity
                observed_tdoa = sensor_time - ref_time

                residuals[i - 1] = predicted_tdoa - observed_tdoa

            return residuals

        return f

    def _get_jacobian_function(self, sensor_positions, detection_times, velocities):
        """TDOA analytical jacobian (derivatives of residuals wrt [lat, lon])"""

        def jacobian(v):
            source_lat, source_lon = v[0], v[1]
            n_equations = len(sensor_positions) - 1
            J = np.zeros((n_equations, 2))

            # Prepare arrays for _theoretical_derivatives
            n_sensors = len(sensor_positions)
            source_lat_array = np.full(n_sensors, source_lat)
            source_lon_array = np.full(n_sensors, source_lon)
            receivers_lat = np.array([pos[0] for pos in sensor_positions])
            receivers_lon = np.array([pos[1] for pos in sensor_positions])

            # Compute all derivatives at once using vectorization
            # Note: _theoretical_derivatives returns derivatives wrt to the SECOND point (sensor)
            # But we need derivatives wrt to the FIRST point (source)
            # So we swap the order: source becomes second point
            dsi_dlat, dsi_dlon = self._theoretical_derivatives(
                receivers_lat, receivers_lon,  # First point: sensors
                source_lat_array, source_lon_array  # Second point: source (we want derivatives wrt this)
            )

            # Convert derivatives to meters per degree
            deg_per_rad =  180 / np.pi #attention dsi_dlat en m/rad donc 1/deg_per_rad
            dsi_dlat = dsi_dlat / deg_per_rad
            dsi_dlon = dsi_dlon / deg_per_rad

            # Reference sensor derivatives and velocity
            ds0_dlat = dsi_dlat[0]
            ds0_dlon = dsi_dlon[0]
            v0 = velocities[0]

            # For each TDOA equation (sensor i vs reference sensor 0)
            for i in range(1, len(sensor_positions)):
                vi = velocities[i]

                # Residual: (s_i/v_i - s_0/v_0) - (t_i - t_0)
                # Derivative wrt lat: (1/v_i * ds_i/dlat - 1/v_0 * ds_0/dlat)
                J[i - 1, 0] = -((1 / vi) * dsi_dlat[i] - (1 / v0) * ds0_dlat)

                # Derivative wrt lon: (1/v_i * ds_i/dlon - 1/v_0 * ds_0/dlon)
                J[i - 1, 1] = -((1 / vi) * dsi_dlon[i] - (1 / v0 )* ds0_dlon)

            return J

        return jacobian
#%%
    def _get_cost_function_toa(self, sensor_positions, detection_times, velocities):
        """TOA cost function: residual_i = (t0 + d_i/c_i) - t_i"""
        def f(v):
            t0, lat, lon = v[0], v[1], v[2]  # t0 en premier
            n_sensors = len(sensor_positions)
            residuals = np.zeros(n_sensors)
            for i in range(n_sensors):
                sensor_pos = sensor_positions[i]
                observed_time = detection_times[i]
                c_i = velocities[i]
                # Calculate distance from source to sensor
                distance = self.get_distance([lat, lon], sensor_pos)
                # Predicted detection time
                predicted_time = t0 + distance / c_i
                #Residual: predicted - observed
                residuals[i] = predicted_time - observed_time
            return residuals
        return f


    def _get_jacobian_function_toa(self, sensor_positions, detection_times, velocities):
        """TOA Jacobian: derivatives of residuals wrt [t0, lat, lon]"""
        def jacobian(v):
            t0, lat, lon = v[0], v[1], v[2]
            n_sensors = len(sensor_positions)
            J = np.zeros((n_sensors, 3))
            # Prepare arrays for _theoretical_derivatives
            source_lat_array = np.full(n_sensors, lat)
            source_lon_array = np.full(n_sensors, lon)
            receivers_lat = np.array([pos[0] for pos in sensor_positions])
            receivers_lon = np.array([pos[1] for pos in sensor_positions])
            # Compute derivatives for all sensors
            # Note: _theoretical_derivatives returns derivatives wrt the SECOND point
            # So we put sensors as first point and source as second point
            d_di_dlat, d_di_dlon = self._theoretical_derivatives(
                receivers_lat, receivers_lon,  # First point: sensors
                source_lat_array, source_lon_array  # Second point: source
            )
            # Convert to seconds per degree
            rad_per_deg = np.pi / 180
            for i in range(n_sensors):
                c_i = velocities[i]
                # Derivative wrt t0: ∂/∂t0 (t0 + d_i/c_i - t_i) = 1
                J[i, 0] = 1.0
                # Derivative wrt lat: ∂/∂lat (t0 + d_i/c_i - t_i) = (1/c_i) * ∂d_i/∂lat
                J[i, 1] = -(1 / c_i) * d_di_dlat[i] * rad_per_deg
                # Derivative wrt lon: ∂/∂lon (t0 + d_i/c_i - t_i) = (1/c_i) * ∂d_i/∂lon
                J[i, 2] = -(1 / c_i) * d_di_dlon[i] * rad_per_deg
            return J
        return jacobian
#%%
    def check_jacobian(self,sensor_positions, detection_times, velocities, test_point, epsilon=1e-8):
        cost_func = self._get_cost_function(sensor_positions, detection_times, velocities)
        jac_func = self._get_jacobian_function(sensor_positions, detection_times, velocities)

        analytical_jac = jac_func(test_point)
        numerical_jac = np.zeros_like(analytical_jac)

        f0 = cost_func(test_point)

        # Derivative wrt lat
        point_perturbed = test_point.copy()
        point_perturbed[0] += epsilon
        f_perturbed = cost_func(point_perturbed)
        numerical_jac[:, 0] = (f_perturbed - f0) / epsilon

        # Derivative wrt lon
        point_perturbed = test_point.copy()
        point_perturbed[1] += epsilon
        f_perturbed = cost_func(point_perturbed)
        numerical_jac[:, 1] = (f_perturbed - f0) / epsilon

        print("Analytical Jacobian:\n", analytical_jac)
        print("Numerical Jacobian:\n", numerical_jac)
        print("Difference:\n", analytical_jac - numerical_jac)

    def _localize_common_source(self, sensors_positions, detection_times, x_min=-90, y_min=-180, x_max=90,
                                y_max=180, t_min=-36_000, initial_pos=None, velocities=None):
        """TDOA localization (no t0 parameter)"""
        from scipy.optimize import least_squares
        min_date = np.argmin(detection_times)

        if initial_pos is None:
            # Simple initial guess: centroid of sensors
            initial_pos = list(np.mean(sensors_positions, axis=0))
        else:
            initial_pos = list(initial_pos[1:])  # Remove t0 if present

        # velocities should be computed at the reference time (min_date)
        if velocities is None:
            velocities = [self.get_sound_speed(initial_pos, p, detection_times[min_date])
                          for p in sensors_positions]
        t0 = detection_times[min_date]
        # Convert times to relative seconds from earliest detection
        detection_times = np.array([(d -t0 ).total_seconds() for d in detection_times])
        f = self._get_cost_function(sensors_positions, detection_times, velocities)
        jacobian = self._get_jacobian_function(sensors_positions, detection_times, velocities)

        # check = self.check_jacobian(sensors_positions, detection_times, velocities, initial_pos, epsilon=1e-8)
        # print(check)

        res = least_squares(f, initial_pos, bounds=([x_min, y_min], [x_max, y_max]),
                                 jac = jacobian, method="dogbox" , ftol=1e-12, xtol=1e-12, gtol=1e-12)

        # Calculate true t0 (emission time relative to earliest detection)
        source_position = res.x
        travel_time_to_ref = self.get_sound_travel_time(source_position, sensors_positions[min_date])
        # ref_velocity = velocities[0]
        # travel_time_to_ref = ref_distance / ref_velocity
        t_origin = (t0 - datetime.timedelta(seconds=travel_time_to_ref)).timestamp()
        # Add t0 for compatibility with TOA interface
        result_x = np.concatenate([[t_origin], res.x])
        res.x = result_x

        return res

    def _define_x0(self, x0, sensors_positions, detection_times, idx):
        if x0 is None:
            return list(np.mean(sensors_positions, axis=0))
        elif len(x0) == 3:  # Si format [t0, lat, lon]
            return x0[1:]
        else:
            return x0

# constant velocity ellipsoidal model
class HomogeneousEllipsoidalSoundModel(EllipsoidalSoundModel):
    def __init__(self, sound_speed=1480):
        self.sound_speed = sound_speed
        super().__init__()

    def get_sound_speed(self, source, sensor, date=None):
        return self.sound_speed

# WOA23-derived velocity with ellipsoidal geometry
class GridEllipsoidalSoundModel(EllipsoidalSoundModel):
    def __init__(self, velocity_grid_paths, constant_velocity=1480, lat_bounds=None, lon_bounds=None, loader ="ISAS"):
        if loader == "ISAS":
            self.models = [SoundVelocityGrid.create_from_ISAS(p, lat_bounds, lon_bounds) for p in velocity_grid_paths]
        elif loader == "netcdf":
            self.models = [SoundVelocityGrid.create_from_NetCDF(p, lat_bounds, lon_bounds) for p in velocity_grid_paths]
        self.constant_velocity = constant_velocity
        super().__init__()

    def get_sound_speed(self, source, sensor, date=None):
        return self.models[date.month - 1].get_sound_speed(source, sensor)

    def get_sound_speed_with_uncertainty(self, source, sensor, date=None):
        return self.models[date.month - 1].get_sound_speed_with_uncertainty(source, sensor)

    def localize_common_source(self, sensors_positions, detection_times, x_min=-90, y_min=-180, x_max=90,
                             y_max=180, t_min=-36_000, initial_pos=None, velocities=None):
        l = self._localize_common_source(sensors_positions, detection_times, x_min, y_min, x_max,
                             y_max, t_min, initial_pos, velocities)
        return self._localize_common_source(sensors_positions, detection_times, x_min, y_min, x_max,
                             y_max, t_min, l.x, velocities)

    def get_bounds(self):
        lat_mins = [model.lat_bounds[0] for model in self.models]
        lat_maxs = [model.lat_bounds[1] for model in self.models]
        lon_mins = [model.lon_bounds[0] for model in self.models]
        lon_maxs = [model.lon_bounds[1] for model in self.models]
        return min(lat_mins), max(lat_maxs), min(lon_mins), max(lon_maxs)

    def _compute_observation_weights(self, sensors_positions, detection_times, velocities,
                                     drift_uncertainties=None, pick_uncertainties=None,
                                     velocity_uncertainties=None, source_position=None):
        """
        Calcule les poids des observations basés sur les incertitudes (WLS)

        La variance totale d'un TDOA est la somme de :
        1. Incertitude de picking (toujours présente)
        2. Incertitude de vitesse (optionnelle, si le modèle de vitesse est incertain)
        3. Dérive d'horloge (optionnelle, si les horloges ne sont pas synchronisées)
        """
        n_sensors = len(sensors_positions)
        n_equations = n_sensors - 1

        # Valeurs par défaut
        if drift_uncertainties is None:
            drift_uncertainties = np.zeros(n_sensors)
        if pick_uncertainties is None:
            pick_uncertainties = np.zeros(n_sensors)
        if velocity_uncertainties is None:
            velocity_uncertainties = np.zeros(n_sensors)

        if source_position is None:
            source_position = np.mean(sensors_positions, axis=0)

        distances = np.array([self.get_distance(source_position, pos) for pos in sensors_positions])
        variances_total = np.zeros(n_equations)

        ref_idx = 0
        for i, sensor_idx in enumerate(range(1, n_sensors)):
            # 1. Picking (incertitude dominante dans la plupart des cas)
            var_pick = pick_uncertainties[ref_idx] ** 2 + pick_uncertainties[sensor_idx] ** 2

            # 2. Vitesse (seulement si velocity_uncertainties est spécifié)
            if velocity_uncertainties is not None and np.any(velocity_uncertainties > 0):
                var_velocity_ref = (distances[ref_idx] / velocities[ref_idx] ** 2) ** 2 * velocity_uncertainties[
                    ref_idx] ** 2
                var_velocity_sensor = (distances[sensor_idx] / velocities[sensor_idx] ** 2) ** 2 * \
                                      velocity_uncertainties[sensor_idx] ** 2
                var_velocity = var_velocity_ref + var_velocity_sensor
            else:
                var_velocity = 0.0

            # 3. Dérive d'horloge
            var_drift = drift_uncertainties[ref_idx] ** 2 + drift_uncertainties[sensor_idx] ** 2

            # Variance totale
            variances_total[i] = var_pick + var_velocity + var_drift
        if variances_total.all() == 0 :
            print("Warning: variances_total = 0")
            return 1
        weights = 1.0 / np.sqrt(variances_total)
        return weights

    # def localize_with_uncertainties(self, sensors_positions, detection_times,
    #                                 drift_uncertainties=None, pick_uncertainties=None,
    #                                 velocity_uncertainties=None, x_min=-90, y_min=-180,
    #                                 x_max=90, y_max=180, initial_pos=None, velocities=None,
    #                                 max_iterations=5):
    #
    #     """TDOA localization (no t0 parameter)"""
    #     from scipy.optimize import least_squares
    #     min_date = np.argmin(detection_times)
    #
    #     if initial_pos is None:
    #         # Simple initial guess: centroid of sensors
    #         initial_pos = list(np.mean(sensors_positions, axis=0))
    #     else:
    #         initial_pos = list(initial_pos[1:])  # Remove t0 if present
    #
    #     # velocities should be computed at the reference time (min_date)
    #     if velocities is None:
    #         velocities = [self.get_sound_speed(initial_pos, p, detection_times[min_date])
    #                       for p in sensors_positions]
    #     t0 = detection_times[min_date]
    #     # Convert times to relative seconds from earliest detection
    #     detection_times = np.array([(d -t0 ).total_seconds() for d in detection_times])
    #     f = self._get_cost_function(sensors_positions, detection_times, velocities)
    #     jacobian = self._get_jacobian_function(sensors_positions, detection_times, velocities)
    #     weights = self._compute_observation_weights(sensors_positions, detection_times, velocities,
    #                                                 drift_uncertainties, pick_uncertainties,
    #                                                 velocity_uncertainties, source_position=initial_pos)
    #
    #     def f_weighted(x):
    #         return f(x) * weights
    #
    #     def jacobian_weighted(x):
    #         return jacobian(x) * weights[:, None]
    #
    #     # check = self.check_jacobian(sensors_positions, detection_times, velocities, initial_pos, epsilon=1e-8)
    #     # print(check)
    #
    #     res = least_squares(f_weighted, initial_pos, bounds=([x_min, y_min], [x_max, y_max]),
    #                              jac = jacobian_weighted, method="dogbox" , ftol=1e-12, xtol=1e-12, gtol=1e-12)
    #
    #     # Calculate true t0 (emission time relative to earliest detection)
    #     source_position = res.x
    #     travel_time_to_ref = self.get_sound_travel_time(source_position, sensors_positions[min_date], date=t0)
    #     # ref_velocity = velocities[0]
    #     # travel_time_to_ref = ref_distance / ref_velocity
    #     t_origin = (t0 - datetime.timedelta(seconds=travel_time_to_ref)).timestamp()
    #     # Add t0 for compatibility with TOA interface
    #     result_x = np.concatenate([[t_origin], res.x])
    #     res.x = result_x
    #
    #     return res


    def _get_cost_function_dynamic(self, sensor_positions, detection_times, t0_datetime):
        """
        TDOA cost function avec vitesses recalculées à chaque évaluation
        """

        def f(v):
            source_lat, source_lon = v[0], v[1]
            n_eq = len(sensor_positions) - 1
            residuals = np.zeros(n_eq)

            # Recalculer les vitesses à la position courante
            velocities = [self.get_sound_speed([source_lat, source_lon], pos, t0_datetime)
                          for pos in sensor_positions]

            ref_pos = sensor_positions[0]
            ref_time = detection_times[0]
            ref_velocity = velocities[0]
            ref_distance = self.get_distance([source_lat, source_lon], ref_pos)

            for i in range(1, len(sensor_positions)):
                sensor_pos = sensor_positions[i]
                sensor_time = detection_times[i]
                sensor_velocity = velocities[i]
                sensor_distance = self.get_distance([source_lat, source_lon], sensor_pos)

                predicted_tdoa = sensor_distance / sensor_velocity - ref_distance / ref_velocity
                observed_tdoa = sensor_time - ref_time
                residuals[i - 1] = predicted_tdoa - observed_tdoa

            return residuals

        return f

    def _get_jacobian_function_dynamic(self, sensor_positions, detection_times, t0_datetime):
        """
        Jacobienne avec vitesses recalculées dynamiquement
        """

        def jacobian(v):
            source_lat, source_lon = v[0], v[1]
            n_equations = len(sensor_positions) - 1
            J = np.zeros((n_equations, 2))
            n_sensors = len(sensor_positions)

            # Recalculer les vitesses
            velocities = [self.get_sound_speed([source_lat, source_lon], pos, t0_datetime)
                          for pos in sensor_positions]

            source_lat_array = np.full(n_sensors, source_lat)
            source_lon_array = np.full(n_sensors, source_lon)
            receivers_lat = np.array([pos[0] for pos in sensor_positions])
            receivers_lon = np.array([pos[1] for pos in sensor_positions])

            dsi_dlat, dsi_dlon = self._theoretical_derivatives(
                receivers_lat, receivers_lon,
                source_lat_array, source_lon_array
            )

            deg_per_rad = 180 / np.pi
            dsi_dlat = dsi_dlat / deg_per_rad
            dsi_dlon = dsi_dlon / deg_per_rad

            ds0_dlat = dsi_dlat[0]
            ds0_dlon = dsi_dlon[0]
            v0 = velocities[0]

            for i in range(1, len(sensor_positions)):
                vi = velocities[i]

                J[i - 1, 0] = -(1 / vi) * dsi_dlat[i] + (1 / v0) * ds0_dlat
                J[i - 1, 1] = -(1 / vi) * dsi_dlon[i] + (1 / v0) * ds0_dlon

            return J

        return jacobian

    def localize_with_uncertainties(self, sensors_positions, detection_times,
                                    drift_uncertainties=None, pick_uncertainties=None,
                                    velocity_uncertainties=None, x_min=-90, y_min=-180,
                                    x_max=90, y_max=180, initial_pos=None, velocities=None,
                                    max_iterations=5):
        """
        TDOA localization - scipy.least_squares gère l'itération
        """
        from scipy.optimize import least_squares

        min_date = np.argmin(detection_times)

        if initial_pos is None:
            initial_pos = list(np.mean(sensors_positions, axis=0))
        else:
            initial_pos = list(initial_pos[1:])

        t0 = detection_times[min_date]
        detection_times_rel = np.array([(d - t0).total_seconds() for d in detection_times])

        # Calculer vitesses et poids à la position initiale
        if velocities is None:
            velocities = [self.get_sound_speed(initial_pos, p, t0)
                          for p in sensors_positions]

        weights = self._compute_observation_weights(
            sensors_positions, detection_times_rel, velocities,
            drift_uncertainties, pick_uncertainties,
            velocity_uncertainties, source_position=initial_pos
        )

        # Utiliser les fonctions dynamiques qui recalculent les vitesses
        f = self._get_cost_function_dynamic(sensors_positions, detection_times_rel, t0)
        jacobian = self._get_jacobian_function_dynamic(sensors_positions, detection_times_rel, t0)

        def f_weighted(x):
            return f(x) * weights

        def jacobian_weighted(x):
            return jacobian(x) * weights[:, None]

        # UNE SEULE optimisation - least_squares itère tout seul
        res = least_squares(
            f_weighted, initial_pos,
            bounds=([x_min, y_min], [x_max, y_max]),
            jac=jacobian_weighted,
            method="dogbox",
            ftol=1e-12, xtol=1e-12, gtol=1e-12
        )

        # Calcul de t0
        source_position = res.x
        travel_time_to_ref = self.get_sound_travel_time(source_position, sensors_positions[min_date], date=t0)
        t_origin = (t0 - datetime.timedelta(seconds=travel_time_to_ref)).timestamp()

        result_x = np.concatenate([[t_origin], res.x])
        res.x = result_x

        return res

    def test_chi_square(self, res, n_params=2, alpha=0.05):
        """
        Test du chi² pour vérifier la cohérence résidus/incertitudes

        Returns: (chi2_statistic, p_value, passes_test, dof)
        """
        from scipy.stats import chi2

        residuals = res.fun
        chi2_stat = np.sum(residuals ** 2)
        dof = len(residuals) - n_params
        p_value = 1 - chi2.cdf(chi2_stat, dof)
        passes = p_value >= alpha

        return chi2_stat, p_value, passes, dof

    def detect_outliers(self, res, threshold=3.0, method='standardized'):
        """
        Détecte les observations aberrantes (fausses associations) dans les résidus

        Parameters:
        -----------
        res : OptimizeResult
            Résultat de least_squares contenant les résidus pondérés dans res.fun
        threshold : float
            Seuil de détection (en nombre d'écarts-types)
            - threshold=3.0 : détecte les outliers à 3σ (99.7%)
            - threshold=2.5 : plus strict
        method : str
            'standardized' : résidus standardisés (recommandé)
            'absolute' : valeur absolue des résidus pondérés

        Returns:
        --------
        dict avec :
            - outlier_indices : indices des capteurs suspects (relatifs au capteur 0)
            - outlier_scores : scores d'anomalie
            - is_outlier : masque booléen
            - residuals : résidus pondérés
        """
        residuals = res.fun  # Résidus pondérés (déjà normalisés)
        n_residuals = len(residuals)

        if method == 'standardized':
            # Les résidus pondérés devraient suivre N(0,1)
            # Un résidu > 3σ est suspect
            scores = np.abs(residuals)
            is_outlier = scores > threshold

        elif method == 'absolute':
            # Détection basée sur la médiane (plus robuste)
            median_residual = np.median(np.abs(residuals))
            mad = np.median(np.abs(residuals - np.median(residuals)))
            # MAD normalisé pour estimer σ
            sigma_robust = 1.4826 * mad
            scores = np.abs(residuals) / (sigma_robust + 1e-10)
            is_outlier = scores > threshold

        else:
            raise ValueError(f"Unknown method: {method}")

        outlier_indices = np.where(is_outlier)[0] + 1  # +1 car capteur 0 est référence

        return {
            'outlier_indices': outlier_indices,
            'outlier_scores': scores[is_outlier],
            'is_outlier': is_outlier,
            'residuals': residuals,
            'all_scores': scores
        }

    def evaluate_localization_quality(self, res, n_sensors):
        """
        Évalue la qualité globale de la localisation
        """
        chi2_stat, p_value, passes, dof = self.test_chi_square(res, n_params=2)
        outlier_info = self.detect_outliers(res, threshold=3.0)

        # Résidus normalisés RMS
        rms_residual = np.sqrt(np.mean(res.fun ** 2))

        # Score de qualité combiné
        quality_score = {
            'chi2_ok': passes,
            'p_value': p_value,
            'n_outliers': len(outlier_info['outlier_indices']),
            'rms_residual': rms_residual,
            'max_residual': np.max(np.abs(res.fun)),
            'is_good': passes and len(outlier_info['outlier_indices']) == 0 and rms_residual < 1.5
        }

        return quality_score


    def estimate_position_uncertainty(self, sensors_positions, detection_times,
                                      solution, weights, n_bootstrap=1000):
        """
        Estimation de l'incertitude de position par bootstrap pondéré
        """
        n_sensors = len(sensors_positions)
        n_equations = n_sensors - 1

        # Calculer les résidus finaux
        final_residuals = self._get_weighted_cost_function(
            sensors_positions, detection_times,
            [self.get_sound_speed(solution[1:], pos) for pos in sensors_positions],
            weights
        )(solution[1:])

        # Bootstrap pondéré
        positions = []
        for _ in range(n_bootstrap):
            # Échantillonnage pondéré des résidus
            bootstrap_indices = np.random.choice(n_equations, n_equations,
                                                 p=weights / np.sum(weights))

            # Perturbation des temps basée sur les résidus
            perturbed_times = detection_times.copy()
            for i, idx in enumerate(bootstrap_indices):
                perturbed_times[i + 1] += final_residuals[idx] / np.sqrt(weights[idx])

            # Re-localisation
            try:
                res_bootstrap = self.localize_common_source(
                    sensors_positions, [datetime.fromtimestamp(t) for t in perturbed_times]
                )
                positions.append(res_bootstrap.x[1:])
            except:
                continue

        if len(positions) > 0:
            positions = np.array(positions)
            uncertainty_ellipse = {
                'center': np.mean(positions, axis=0),
                'std_lat': np.std(positions[:, 0]),
                'std_lon': np.std(positions[:, 1]),
                'correlation': np.corrcoef(positions.T)[0, 1]
            }
            return uncertainty_ellipse
        else:
            return None