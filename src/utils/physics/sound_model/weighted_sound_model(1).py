import numpy as np
from scipy.optimize import least_squares
from datetime import datetime, timedelta
from tqdm import tqdm

from utils.physics.sound_model.ellipsoidal_sound_model import EllipsoidalSoundModel, HomogeneousEllipsoidalSoundModel, \
    GridEllipsoidalSoundModel


class WeightedEllipsoidalSoundModel(EllipsoidalSoundModel):
    """
    Extension du modèle ellipsoïdal avec pondération par les incertitudes
    """
    
    def __init__(self):
        super().__init__()
    
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
        drift_uncertainties : array-like, incertitudes de position des capteurs en mètres
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
            velocity_uncertainties = np.full(n_sensors, 10.0)  # 10 m/s par défaut
        
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
            sensor_idx = i + 1
            
            # 1. Variance due à l'incertitude de picking (TDOA)
            var_pick = pick_uncertainties[ref_idx]**2 + pick_uncertainties[sensor_idx]**2
            
            # 2. Variance due à l'incertitude de vitesse
            # σ²(t) = (d/v)² * (σ²(v)/v²) pour chaque capteur
            var_velocity_ref = (distances[ref_idx] / velocities[ref_idx])**2 * \
                              (velocity_uncertainties[ref_idx] / velocities[ref_idx])**2
            var_velocity_sensor = (distances[sensor_idx] / velocities[sensor_idx])**2 * \
                                 (velocity_uncertainties[sensor_idx] / velocities[sensor_idx])**2
            var_velocity = var_velocity_ref + var_velocity_sensor
            
            # 3. Variance due à l'incertitude de position (dérive)
            # Approximation: σ²(t) ≈ (1/v)² * σ²(d)
            # où σ(d) est l'incertitude sur la distance due à la dérive
            var_drift_ref = (drift_uncertainties[ref_idx] / velocities[ref_idx])**2
            var_drift_sensor = (drift_uncertainties[sensor_idx] / velocities[sensor_idx])**2
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
            print(residuals)
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
                weight_sqrt = np.sqrt(weights[i-1])
                
                # Jacobien pondéré
                J[i-1, 0] = weight_sqrt * (dsi_dlat[i]/sensor_velocity - ds0_dlat/ref_velocity)
                J[i-1, 1] = weight_sqrt * (dsi_dlon[i]/sensor_velocity - ds0_dlon/ref_velocity)
            
            return J
        
        return jacobian
    
    def localize_with_uncertainties(self, sensors_positions, detection_times, 
                                  drift_uncertainties=None, pick_uncertainties=None, 
                                  velocity_uncertainties=None, x_min=-90., y_min=-180.,
                                  x_max=90., y_max=180, initial_pos=None, velocities=None,
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
        else:
            initial_pos = list(initial_pos[1:])  # Retirer t0 si présent
        
        # Vitesses
        if velocities is None:
            velocities = [self.get_sound_speed(initial_pos, p, detection_times[min_date])
                         for p in sensors_positions]
        
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
            print(current_position)
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
                                               p=weights/np.sum(weights))
            
            # Perturbation des temps basée sur les résidus
            perturbed_times = detection_times.copy()
            for i, idx in enumerate(bootstrap_indices):
                perturbed_times[i+1] += final_residuals[idx] / np.sqrt(weights[idx])
            
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


class WeightedHomogeneousEllipsoidalSoundModel(HomogeneousEllipsoidalSoundModel):
    """
    Modèle ellipsoïdal homogène avec pondération par incertitudes
    """
    
    def __init__(self, sound_speed=1480):
        super().__init__(sound_speed)
        self._add_weighting_methods()
    
    def _add_weighting_methods(self):
        """Ajoute les méthodes de pondération à la classe"""
        # Copier les méthodes de pondération
        self._compute_observation_weights = WeightedEllipsoidalSoundModel._compute_observation_weights.__get__(self)
        self._get_weighted_cost_function = WeightedEllipsoidalSoundModel._get_weighted_cost_function.__get__(self)
        self._get_weighted_jacobian_function = WeightedEllipsoidalSoundModel._get_weighted_jacobian_function.__get__(self)
        self.localize_with_uncertainties = WeightedEllipsoidalSoundModel.localize_with_uncertainties.__get__(self)
        self.estimate_position_uncertainty = WeightedEllipsoidalSoundModel.estimate_position_uncertainty.__get__(self)


class WeightedGridEllipsoidalSoundModel(GridEllipsoidalSoundModel):
    """
    Modèle ellipsoïdal avec grille de vitesses ET pondération par incertitudes
    """
    
    def __init__(self, velocity_grid_paths, constant_velocity=1480, lat_bounds=None, lon_bounds=None, loader="ISAS"):
        super().__init__(velocity_grid_paths, constant_velocity, lat_bounds, lon_bounds, loader)
        self._add_weighting_methods()
    
    def _add_weighting_methods(self):
        """Ajoute les méthodes de pondération à la classe"""
        # Copier les méthodes de pondération
        self._compute_observation_weights = WeightedEllipsoidalSoundModel._compute_observation_weights.__get__(self)
        self._get_weighted_cost_function = WeightedEllipsoidalSoundModel._get_weighted_cost_function.__get__(self)
        self._get_weighted_jacobian_function = WeightedEllipsoidalSoundModel._get_weighted_jacobian_function.__get__(self)
        self.localize_with_uncertainties = WeightedEllipsoidalSoundModel.localize_with_uncertainties.__get__(self)
        self.estimate_position_uncertainty = WeightedEllipsoidalSoundModel.estimate_position_uncertainty.__get__(self)
    
    def get_velocity_at_positions(self, sensors_positions, detection_date):
        """
        Récupère les vitesses pour tous les capteurs à une date donnée
        
        Parameters:
        -----------
        sensors_positions : list, positions des capteurs [(lat, lon), ...]
        detection_date : datetime, date de référence
        
        Returns:
        --------
        velocities : list, vitesses du son pour chaque capteur
        """
        velocities = []
        for pos in sensors_positions:
            try:
                velocity = self.get_sound_speed([0, 0], pos, detection_date)  # Source dummy
                if velocity is None:
                    velocity = self.constant_velocity
                velocities.append(velocity)
            except (IndexError, KeyError):
                velocities.append(self.constant_velocity)
        
        return velocities
    
    def get_velocity_uncertainties_from_grid(self, sensors_positions, detection_date, 
                                           spatial_correlation_km=50):
        """
        Estime les incertitudes de vitesse basées sur la variabilité spatiale de la grille
        
        Parameters:
        -----------
        sensors_positions : list, positions des capteurs
        detection_date : datetime, date de référence  
        spatial_correlation_km : float, distance de corrélation spatiale
        
        Returns:
        --------
        velocity_uncertainties : array, incertitudes estimées pour chaque capteur
        """
        month_idx = detection_date.month - 1
        velocity_grid = self.models[month_idx]
        uncertainties = []
        
        for pos in sensors_positions:
            try:
                # Échantillonner autour de la position pour estimer la variabilité
                lat, lon = pos
                
                # Créer une grille locale
                dlat = spatial_correlation_km / 111.0  # ~1 degré = 111 km
                dlon = spatial_correlation_km / (111.0 * np.cos(np.radians(lat)))
                
                local_lats = np.linspace(lat - dlat, lat + dlat, 5)
                local_lons = np.linspace(lon - dlon, lon + dlon, 5)
                
                local_velocities = []
                for llat in local_lats:
                    for llon in local_lons:
                        try:
                            vel = velocity_grid.get_sound_speed([0, 0], [llat, llon])
                            if vel is not None:
                                local_velocities.append(vel)
                        except:
                            continue
                
                if len(local_velocities) > 1:
                    uncertainty = np.std(local_velocities)
                else:
                    uncertainty = self.constant_velocity * 0.05  # 5% par défaut
                    
            except:
                uncertainty = self.constant_velocity * 0.05
            
            uncertainties.append(max(uncertainty, 5.0))  # Minimum 5 m/s
        
        return np.array(uncertainties)


# Exemple d'utilisation pratique
def example_usage():
    """Exemple d'utilisation du modèle pondéré avec grille"""
    
    # 1. Modèle avec vitesse constante
    model_homo = WeightedHomogeneousEllipsoidalSoundModel(sound_speed=1480)
    
    # 2. Modèle avec grille de vitesses
    grid_paths = [f"../../../../data/sound_model/min-velocities_month-{i:02d}.nc" for i in range(1,13)]  # 12 fichiers
    model_grid = WeightedGridEllipsoidalSoundModel(
        velocity_grid_paths=grid_paths,
        constant_velocity=1480,
        loader="netcdf"

    )
    
    # Données d'exemple
    sensors_positions = [(0,1), (1, 1), (1, 0), (0, 0.0)]
    detection_times = [
        datetime(2024, 6, 15, 12, 30, 10, 123000),
        datetime(2024, 6, 15, 12, 30, 12, 456000),
        datetime(2024, 6, 15, 12, 30, 15, 789000),
        datetime(2024, 6, 15, 12, 30, 18, 120000)
    ]
    
    # Incertitudes
    drift_uncertainties = [100, 80, 120, 90]  # mètres
    pick_uncertainties = [0.01, 0.008, 0.015, 0.012]  # secondes
    
    # Pour le modèle homogène
    velocity_uncertainties_homo = [15, 15, 15, 15]  # m/s
    
    # Pour le modèle avec grille (estimation automatique)
    velocity_uncertainties_grid = model_grid.get_velocity_uncertainties_from_grid(
        sensors_positions, detection_times[0]
    )
    
    # Localisation avec modèle homogène
    result_homo, weights_homo, uncertainties_homo = model_homo.localize_with_uncertainties(
        sensors_positions=sensors_positions,
        detection_times=detection_times,
        drift_uncertainties=drift_uncertainties,
        pick_uncertainties=pick_uncertainties,
        velocity_uncertainties=velocity_uncertainties_homo
    )
    
    # Localisation avec modèle grille
    result_grid, weights_grid, uncertainties_grid = model_grid.localize_with_uncertainties(
        sensors_positions=sensors_positions,
        detection_times=detection_times,
        drift_uncertainties=drift_uncertainties,
        pick_uncertainties=pick_uncertainties,
        velocity_uncertainties=velocity_uncertainties_grid
    )
    
    print(f"Position homogène: {result_homo.x[1:]}")
    print(f"Position grille: {result_grid.x[1:]}")
    print(f"Poids finaux: {weights_grid}")
    
    return result_grid, weights_grid, uncertainties_grid

if __name__ == "__main__":
    example_usage()