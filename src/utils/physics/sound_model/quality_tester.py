import numpy as np
from scipy.stats import chi2, norm, t as student_t
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class QualityLevel(Enum):
    EXCELLENT = 4  # Test global OK, aucun outlier
    GOOD = 3  # Test global OK, outliers mineurs
    MARGINAL = 2  # Test global limite, exploitable
    POOR = 1  # Test global échoue, outliers détectés
    REJECTED = 0  # Données insuffisantes ou incohérentes


@dataclass
class QualityReport:
    """Rapport complet de qualité d'une localisation"""
    level: QualityLevel
    sigma2_hat: float  # Facteur unitaire de variance
    global_test_passed: bool
    chi2_bounds: tuple  # (lower, upper) pour test bilatéral
    outlier_indices: list  # Indices des résidus suspects
    outlier_scores: list  # Scores normalisés des outliers
    n_obs: int
    dof: int
    selection_score: float  # Score pour comparaison
    details: dict


class LocalizationQualityTester:
    """
    Tests statistiques rigoureux pour évaluation de localisation sismique
    Basé sur Sillard (2001) et Seubert & Touzé (2013)
    """

    def __init__(self, n_params=2, alpha_global=0.05, alpha_outlier=0.001,
                 bilateral_test=False):
        """
        Args:
            n_params: Nombre de paramètres estimés (2 pour x,y)
            alpha_global: Niveau de signification du test global (5%)
            alpha_outlier: Niveau pour détection outliers individuels (0.1%)
            bilateral_test: Si False, test unilatéral (rejette seulement σ² >> 1)
                           Si True, test bilatéral (rejette aussi σ² << 1)
        """
        self.n_params = n_params
        self.alpha_global = alpha_global
        self.alpha_outlier = alpha_outlier
        self.bilateral_test = bilateral_test

    def test_global_model(self, residuals: np.ndarray) -> dict:
        """
        Test du facteur unitaire de variance (test global)

        Mode unilatéral (défaut):
            H0: σ₀² <= 1   H1: σ₀² > 1
            Rejette seulement si résidus trop grands (outliers/mauvais modèle)

        Mode bilatéral:
            H0: σ₀² = 1    H1: σ₀² ≠ 1
            Rejette aussi si incertitudes surestimées (σ² << 1)

        Returns:
            dict avec sigma2_hat, passes, bounds, dof
        """
        n = len(residuals)
        dof = n - self.n_params

        if dof <= 0:
            return {
                'sigma2_hat': np.inf,
                'passes': False,
                'chi2_lower': np.nan,
                'chi2_upper': np.nan,
                'dof': dof,
                'reason': 'insufficient_dof'
            }

        # Somme des carrés des résidus normalisés
        vtpv = np.sum(residuals ** 2)

        # Facteur unitaire de variance a posteriori
        sigma2_hat = vtpv / dof

        if self.bilateral_test:
            # Test bilatéral : rejette si σ² trop grand OU trop petit
            chi2_lower = chi2.ppf(self.alpha_global / 2, dof)
            chi2_upper = chi2.ppf(1 - self.alpha_global / 2, dof)
            passes = chi2_lower < vtpv < chi2_upper
            reason = None
            if not passes:
                reason = 'sigma2_too_high' if vtpv >= chi2_upper else 'sigma2_too_low'
        else:
            # Test unilatéral : rejette SEULEMENT si σ² trop grand
            chi2_lower = 0.0  # Pas de borne inférieure
            chi2_upper = chi2.ppf(1 - self.alpha_global, dof)
            passes = vtpv < chi2_upper
            reason = 'sigma2_too_high' if not passes else None

        return {
            'sigma2_hat': sigma2_hat,
            'vtpv': vtpv,
            'passes': passes,
            'chi2_lower': chi2_lower,
            'chi2_upper': chi2_upper,
            'dof': dof,
            'reason': reason
        }

    def test_individual_residuals_baarda(self, residuals: np.ndarray) -> dict:
        """
        Test de Baarda (w-test / data snooping)

        Suppose que σ₀² = 1 est vérifié (test global passé)
        Les résidus normalisés suivent N(0,1)

        Returns:
            dict avec outliers détectés et leurs scores
        """
        n = len(residuals)

        # Seuil pour test bilatéral normal
        # Pour alpha=0.001 bilatéral → threshold ≈ 3.29
        threshold = norm.ppf(1 - self.alpha_outlier / 2)

        # Résidus normalisés (déjà normalisés en entrée)
        w_scores = np.abs(residuals)

        outlier_mask = w_scores > threshold
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_scores = w_scores[outlier_mask].tolist()

        return {
            'threshold': threshold,
            'outlier_indices': outlier_indices,
            'outlier_scores': outlier_scores,
            'n_outliers': len(outlier_indices),
            'max_score': float(np.max(w_scores)) if n > 0 else 0.0,
            'w_scores': w_scores
        }

    def test_individual_residuals_pope(self, residuals: np.ndarray) -> dict:
        """
        Test de Pope (τ-test)

        Plus robuste que Baarda quand σ₀² ≠ 1
        Utilise la distribution de Student au lieu de la normale

        Returns:
            dict avec outliers détectés
        """
        n = len(residuals)
        dof = n - self.n_params

        if dof <= 1:
            return {
                'threshold': np.inf,
                'outlier_indices': [],
                'outlier_scores': [],
                'n_outliers': 0,
                'reason': 'insufficient_dof'
            }

        # Facteur de variance a posteriori
        sigma2_hat = np.sum(residuals ** 2) / dof
        sigma_hat = np.sqrt(sigma2_hat)

        # Résidus studentisés
        tau_scores = np.abs(residuals) / sigma_hat

        # Seuil basé sur Student(dof-1)
        threshold = student_t.ppf(1 - self.alpha_outlier / 2, dof - 1)

        outlier_mask = tau_scores > threshold
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_scores = tau_scores[outlier_mask].tolist()

        return {
            'threshold': threshold,
            'outlier_indices': outlier_indices,
            'outlier_scores': outlier_scores,
            'n_outliers': len(outlier_indices),
            'max_score': float(np.max(tau_scores)) if n > 0 else 0.0,
            'sigma_hat': sigma_hat,
            'tau_scores': tau_scores
        }

    def compute_selection_score(self, global_result: dict,
                                outlier_result: dict,
                                n_obs: int,
                                azimuthal_gap: float = None) -> float:
        """
        Score optimisé basé sur l'analyse ROC-AUC empirique

        Pondérations basées sur le pouvoir discriminant mesuré :
        - outlier_fraction : AUC = 0.908 (poids fort)
        - p_value : AUC = 0.863 (poids fort)
        - max_zscore : AUC = 0.697 (poids modéré)
        - azimuthal_gap : AUC = 0.669 (poids modéré, si disponible)
        """
        if n_obs < 3 or global_result['dof'] <= 0:
            return -np.inf

        sigma2 = global_result['sigma2_hat']
        dof = global_result['dof']

        if sigma2 <= 0:
            return -np.inf

        # Calculer p_value du test χ²
        vtpv = global_result.get('vtpv', sigma2 * dof)
        p_value = 1 - chi2.cdf(vtpv, dof)

        # Récupérer les métriques d'outliers
        w_scores = outlier_result.get('w_scores', outlier_result.get('tau_scores', np.array([])))
        if len(w_scores) > 0:
            outlier_fraction = np.mean(np.abs(w_scores) > 2.5)
            max_zscore = np.max(np.abs(w_scores))
        else:
            outlier_fraction = 0.0
            max_zscore = 0.0

        # === Score composite pondéré par AUC ===
        # Plus l'AUC est élevé, plus la métrique est fiable

        # outlier_fraction (AUC=0.908) : 0 = parfait, 1 = mauvais
        score_outlier = 1.0 - outlier_fraction  # Inverser pour que haut = bon

        # p_value (AUC=0.863) : haut = bon
        score_pvalue = min(p_value, 1.0)

        # max_zscore (AUC=0.697) : bas = bon
        score_zscore = 1.0 / (1.0 + max_zscore / 5.0)

        # azimuthal_gap (AUC=0.669) : bas = bon (si fourni)
        if azimuthal_gap is not None:
            score_gap = 1.0 - min(azimuthal_gap / 360.0, 1.0)
        else:
            score_gap = 0.5  # Neutre si non fourni

        # Bonus observations
        score_obs = np.log(n_obs) / np.log(10)  # Normalisé ~[0.5, 1] pour n_obs ∈ [3, 10]

        # Pondération par AUC (normalisé)
        weights = {
            'outlier_fraction': 0.908,
            'p_value': 0.863,
            'max_zscore': 0.697,
            'azimuthal_gap': 0.669,
            'n_obs': 0.7  # Poids arbitraire pour robustesse
        }
        total_weight = sum(weights.values())

        score = (
                        weights['outlier_fraction'] * score_outlier +
                        weights['p_value'] * score_pvalue +
                        weights['max_zscore'] * score_zscore +
                        weights['azimuthal_gap'] * score_gap +
                        weights['n_obs'] * score_obs
                ) / total_weight

        return score

    def evaluate(self, result) -> QualityReport:
        """
        Évaluation complète d'une localisation

        Args:
            result: Objet OptimizeResult de scipy.optimize.least_squares

        Returns:
            QualityReport avec tous les diagnostics
        """
        residuals = np.asarray(result.fun)
        n_obs = len(residuals)

        # Cas dégénéré
        if n_obs < 3:
            return QualityReport(
                level=QualityLevel.REJECTED,
                sigma2_hat=np.inf,
                global_test_passed=False,
                chi2_bounds=(np.nan, np.nan),
                outlier_indices=[],
                outlier_scores=[],
                n_obs=n_obs,
                dof=n_obs - self.n_params,
                selection_score=-np.inf,
                details={'reason': 'insufficient_observations'}
            )

        # 1. Test global
        global_result = self.test_global_model(residuals)

        # 2. Test des résidus individuels
        # Utiliser Baarda si test global passe, Pope sinon
        if global_result['passes']:
            outlier_result = self.test_individual_residuals_baarda(residuals)
        else:
            outlier_result = self.test_individual_residuals_pope(residuals)

        # 3. Score de sélection
        selection_score = self.compute_selection_score(
            global_result, outlier_result, n_obs
        )

        # 4. Déterminer le niveau de qualité
        level = self._determine_quality_level(
            global_result, outlier_result, n_obs
        )

        return QualityReport(
            level=level,
            sigma2_hat=global_result['sigma2_hat'],
            global_test_passed=global_result['passes'],
            chi2_bounds=(global_result['chi2_lower'], global_result['chi2_upper']),
            outlier_indices=outlier_result['outlier_indices'],
            outlier_scores=outlier_result['outlier_scores'],
            n_obs=n_obs,
            dof=global_result['dof'],
            selection_score=selection_score,
            details={
                'global': global_result,
                'outliers': outlier_result
            }
        )

    def _determine_quality_level(self, global_result: dict,
                                 outlier_result: dict,
                                 n_obs: int) -> QualityLevel:
        """Détermine le niveau de qualité selon les tests"""

        if n_obs < 3 or global_result['dof'] <= 0:
            return QualityLevel.REJECTED

        sigma2 = global_result['sigma2_hat']
        passes_global = global_result['passes']
        n_outliers = outlier_result['n_outliers']

        # Excellent : test global OK et aucun outlier
        if passes_global and n_outliers == 0:
            return QualityLevel.EXCELLENT

        # Good : test global OK mais quelques outliers mineurs
        if passes_global and n_outliers <= 1:
            return QualityLevel.GOOD

        # Marginal : test global échoue mais σ² pas trop loin de 1
        if 0.5 < sigma2 < 2.0 and n_outliers <= 2:
            return QualityLevel.MARGINAL

        # Poor : problèmes significatifs
        if 0.2 < sigma2 < 5.0:
            return QualityLevel.POOR

        # Rejected : modèle complètement incohérent
        return QualityLevel.REJECTED