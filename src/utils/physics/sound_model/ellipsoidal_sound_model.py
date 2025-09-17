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
            n_equations = len(sensor_positions)-1
            J = np.zeros((n_equations, 2))

            # Prepare arrays for _theoretical_derivatives (it accepts vector inputs)
            n_sensors = len(sensor_positions)
            source_lat_array = np.full(n_sensors, source_lat)
            source_lon_array = np.full(n_sensors, source_lon)
            receivers_lat = np.array([pos[0] for pos in sensor_positions])
            receivers_lon = np.array([pos[1] for pos in sensor_positions])

            # Compute all derivatives at once using vectorization
            dsi_dlat, dsi_dlon = self._theoretical_derivatives(
                source_lat_array, source_lon_array,  # source positions (repeated)
                receivers_lat, receivers_lon  # all receiver positions
            )

            # Convert derivatives to meters per degree (because parameters are in degrees)
            rad_per_deg = np.pi / 180.0
            dsi_dlat = dsi_dlat * rad_per_deg
            dsi_dlon = dsi_dlon * rad_per_deg

            # Reference derivatives (sensor index 0)
            ds0_dlat = dsi_dlat[0]
            ds0_dlon = dsi_dlon[0]

            # Precompute ref distance (avoid recomputing inside loop)
            ref_distance = self.get_distance([source_lat, source_lon], sensor_positions[0])
            ref_velocity = velocities[0]

            for i in range(1,n_equations):
                sensor_velocity = velocities[i]
                # Difference derivatives: ∂(s_i - s_0)/∂source
                J[i-1, 0] = (dsi_dlat[i]/sensor_velocity  - ds0_dlat)/ref_velocity# ∂(s_i - s_0)/∂lat
                J[i-1, 1] = (dsi_dlon[i]/sensor_velocity - ds0_dlon)/ref_velocity# ∂(s_i - s_0)/∂lon

            return J

        return jacobian

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

        # Convert times to relative seconds from earliest detection
        detection_times = np.array([(d - detection_times[min_date]).total_seconds() for d in detection_times])
        f = self._get_cost_function(sensors_positions, detection_times, velocities)
        jacobian = self._get_jacobian_function(sensors_positions, detection_times, velocities)

        res = least_squares(f, initial_pos, bounds=([x_min, y_min], [x_max, y_max]),
                            jac=jacobian, method='trf', loss='soft_l1', f_scale=3.5, ftol=1e-12, xtol=1e-12, gtol=1e-12)

        # Calculate true t0 (emission time relative to earliest detection)
        source_position = res.x
        ref_distance = self.get_distance(source_position, sensors_positions[0])
        ref_velocity = velocities[0]
        travel_time_to_ref = ref_distance / ref_velocity
        t0 = -travel_time_to_ref
        # Add t0 for compatibility with TOA interface
        result_x = np.concatenate([[t0], res.x])
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
        return self._localize_common_source(sensors_positions, detection_times, x_min, y_min, x_max,
                             y_max, t_min, initial_pos, velocities)

    def get_bounds(self):
        lat_mins = [model.lat_bounds[0] for model in self.models]
        lat_maxs = [model.lat_bounds[1] for model in self.models]
        lon_mins = [model.lon_bounds[0] for model in self.models]
        lon_maxs = [model.lon_bounds[1] for model in self.models]
        return min(lat_mins), max(lat_maxs), min(lon_mins), max(lon_maxs)

#
# class WeightedEllipsoidalSoundModel(EllipsoidalSoundModel):
#     """
#     Extension du modèle ellipsoïdal avec pondération par les incertitudes
#     """
#
#     def __init__(self):
#         super().__init__()

    def _compute_observation_weights(self, sensors_positions, detection_times, velocities,
                                     drift_uncertainties=None, pick_uncertainties=None,
                                     velocity_uncertainties=None, source_position=None):
        """
        Calcule les poids des observations basés sur les incertitudes

        Parameters:
        -----------
        sensors_positions : array-like, positions des capteurs [(lat, lon), ...]
        detection_times : array-like, temps de détection
        velocities : array-like, vitesses du son
        drift_uncertainties : array-like, incertitudes of time drift
        pick_uncertainties : array-like, incertitudes de picking en secondes
        velocity_uncertainties : array-like, incertitudes de vitesse en m/s
        source_position : tuple, position estimée de la source (lat, lon)

        Returns:
        --------
        weights : array, poids pour chaque observation TDOA
        """
        n_sensors = len(sensors_positions)
        n_equations = n_sensors - 1  # TDOA: n-1 équations

        # Valeurs par défaut si non fournies
        if drift_uncertainties is None:
            drift_uncertainties = np.zeros(n_sensors)
        if pick_uncertainties is None:
            pick_uncertainties = np.full(n_sensors, 0.01)  # 10ms par défaut
        if velocity_uncertainties is None:
            velocity_uncertainties = np.full(n_sensors, 1.0)  # 1 m/s par défaut

        # Position de source approximative si non fournie
        if source_position is None:
            source_position = np.mean(sensors_positions, axis=0)

        # Calculer les distances source-capteur
        distances = np.array([self.get_distance(source_position, pos) for pos in sensors_positions])

        # Calculer les variances pour chaque type d'incertitude
        variances_total = np.zeros(n_equations)

        for i in range(n_equations):
            # Index des capteurs (i+1 vs référence 0)
            ref_idx = 0
            sensor_idx = i

            # 1. Variance due à l'incertitude de picking (TDOA)
            var_pick = pick_uncertainties[ref_idx] ** 2 + pick_uncertainties[sensor_idx] ** 2

            # 2. Variance due à l'incertitude de vitesse
            # σ²(t) = (d/v)² * (σ²(v)/v²) pour chaque capteur
            var_velocity_ref = (distances[ref_idx] / velocities[ref_idx]) ** 2 * \
                               (velocity_uncertainties[ref_idx] / velocities[ref_idx]) ** 2
            var_velocity_sensor = (distances[sensor_idx] / velocities[sensor_idx]) ** 2 * \
                                  (velocity_uncertainties[sensor_idx] / velocities[sensor_idx]) ** 2
            var_velocity = var_velocity_ref + var_velocity_sensor

            # 3. Variance due à l'incertitude de position (dérive)
            # où σ(d) est l'incertitude sur la distance due à la dérive d'horloge

            var_drift_ref = (drift_uncertainties[ref_idx] ) ** 2
            var_drift_sensor = (drift_uncertainties[sensor_idx]) ** 2
            var_drift = var_drift_ref + var_drift_sensor

            # Variance totale
            variances_total[i] = var_pick + var_velocity + var_drift

        # Poids = 1/variance (éviter division par zéro)
        weights = 1.0 / (variances_total + 1e-12)

        return weights

    def _get_weighted_cost_function(self, sensor_positions, detection_times, velocities, weights):
        """
        Fonction de coût pondérée par les incertitudes
        """

        def f(v):
            source_lat, source_lon = v[0], v[1]
            n_eq = len(sensor_positions) - 1
            residuals = np.zeros(n_eq)

            # Capteur de référence (index 0)
            ref_pos = sensor_positions[0]
            ref_time = detection_times[0]
            ref_velocity = velocities[0]
            ref_distance = self.get_distance([source_lat, source_lon], ref_pos)

            for i in range(1, len(sensor_positions)):
                sensor_pos = sensor_positions[i]
                sensor_time = detection_times[i]
                sensor_velocity = velocities[i]
                sensor_distance = self.get_distance([source_lat, source_lon], sensor_pos)

                # TDOA prédit vs observé
                predicted_tdoa = sensor_distance / sensor_velocity - ref_distance / ref_velocity
                observed_tdoa = sensor_time - ref_time

                # Résidu pondéré
                residuals[i - 1] = (predicted_tdoa - observed_tdoa) * np.sqrt(weights[i - 1])

            return residuals

        return f

    def _get_weighted_jacobian_function(self, sensor_positions, detection_times, velocities, weights):
        """
        Jacobien pondéré par les incertitudes
        """

        def jacobian(v):
            source_lat, source_lon = v[0], v[1]
            n_equations = len(sensor_positions) - 1
            J = np.zeros((n_equations, 2))

            # Calcul vectorisé des dérivées
            n_sensors = len(sensor_positions)
            source_lat_array = np.full(n_sensors, source_lat)
            source_lon_array = np.full(n_sensors, source_lon)
            receivers_lat = np.array([pos[0] for pos in sensor_positions])
            receivers_lon = np.array([pos[1] for pos in sensor_positions])

            dsi_dlat, dsi_dlon = self._theoretical_derivatives(
                source_lat_array, source_lon_array,
                receivers_lat, receivers_lon
            )

            # Conversion en dérivées par degré
            rad_per_deg = np.pi / 180.0
            dsi_dlat = dsi_dlat * rad_per_deg
            dsi_dlon = dsi_dlon * rad_per_deg

            # Dérivées de référence
            ds0_dlat = dsi_dlat[0]
            ds0_dlon = dsi_dlon[0]
            ref_velocity = velocities[0]

            for i in range(1, n_sensors):
                sensor_velocity = velocities[i]
                weight_sqrt = np.sqrt(weights[i - 1])

                # Jacobien pondéré
                J[i - 1, 0] = weight_sqrt * (dsi_dlat[i] / sensor_velocity - ds0_dlat / ref_velocity)
                J[i - 1, 1] = weight_sqrt * (dsi_dlon[i] / sensor_velocity - ds0_dlon / ref_velocity)

            return J

        return jacobian

    def localize_with_uncertainties(self, sensors_positions, detection_times,
                                    drift_uncertainties=None, pick_uncertainties=None,
                                    velocity_uncertainties=None, x_min=-90, y_min=-180,
                                    x_max=90, y_max=180, initial_pos=None, velocities=None,
                                    max_iterations=5):
        """
        Localisation avec pondération par les incertitudes

        Parameters:
        -----------
        sensors_positions : list, positions des capteurs [(lat, lon), ...]
        detection_times : list, temps de détection
        drift_uncertainties : array-like, incertitudes de position en mètres
        pick_uncertainties : array-like, incertitudes de picking en secondes
        velocity_uncertainties : array-like, incertitudes de vitesse en m/s
        max_iterations : int, nombre max d'itérations pour l'estimation itérative

        Returns:
        --------
        result : OptimizeResult, résultat de l'optimisation
        weights : array, poids finaux utilisés
        uncertainties : dict, détail des incertitudes
        """

        # Index du capteur de référence
        min_date = np.argmin(detection_times)

        # Position initiale
        if initial_pos is None:
            initial_pos = list(np.mean(sensors_positions, axis=0))
        else :
            print('initial pos ', initial_pos)
            if len(initial_pos) == 2 :
                initial_pos = list(initial_pos)
            if len(initial_pos) == 3 :
                initial_pos = list(initial_pos[1:])  # Retirer t0 si présent

        # Vitesses
        # if velocities is None:
        #     velocities = [self.get_sound_speed(initial_pos, p, detection_times[min_date])
        #                   for p in sensors_positions]
        if velocities is None:
            velocities = [self.get_sound_speed_with_uncertainty(initial_pos, p, detection_times[min_date])
                          for p in sensors_positions]
            velocities, velocity_uncertainties = np.array(velocities)[:,0], np.array(velocities)[:,1]

        # Temps relatifs
        detection_times_rel = np.array([(d - detection_times[min_date]).total_seconds()
                                        for d in detection_times])

        # Estimation itérative avec mise à jour des poids
        current_position = initial_pos.copy()

        for iteration in range(max_iterations):
            # Calcul des poids basés sur la position courante
            weights = self._compute_observation_weights(
                sensors_positions, detection_times_rel, velocities,
                drift_uncertainties, pick_uncertainties, velocity_uncertainties,
                current_position
            )

            # Optimisation pondérée
            f = self._get_weighted_cost_function(sensors_positions, detection_times_rel,
                                                 velocities, weights)
            jac = self._get_weighted_jacobian_function(sensors_positions, detection_times_rel,
                                                       velocities, weights)

            res = least_squares(f, current_position,
                                bounds=([x_min, y_min], [x_max, y_max]),
                                jac=jac, method='trf', loss='linear',
                                ftol=1e-12, xtol=1e-12, gtol=1e-12)

            # Vérifier la convergence
            if np.linalg.norm(np.array(res.x) - np.array(current_position)) < 1e-6:
                break

            current_position = res.x.copy()

        # Calcul du t0 final
        ref_distance = self.get_distance(res.x, sensors_positions[min_date])
        ref_velocity = velocities[min_date]
        t0 = -ref_distance / ref_velocity

        # Résultat final avec t0
        res.x = np.concatenate([[t0], res.x])

        # Informations sur les incertitudes
        uncertainties = {
            'drift_uncertainties': drift_uncertainties,
            'pick_uncertainties': pick_uncertainties,
            'velocity_uncertainties': velocity_uncertainties,
            'final_weights': weights,
            'iterations_used': iteration + 1
        }

        return res, weights, uncertainties

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