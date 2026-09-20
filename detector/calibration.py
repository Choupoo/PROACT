import numpy as np
from scipy.stats import beta, binom


def positive_integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError("{} must be a positive integer.".format(name))
    return int(value)


def fit_count_calibration(clean_alarms, task_size, alpha=0.05):
    values = np.asarray(clean_alarms)
    task_size = positive_integer(task_size, "task_size")
    if values.ndim != 1 or not values.size or not np.isin(values, [0, 1]).all():
        raise ValueError("clean_alarms must be a nonempty binary vector.")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1).")
    m, k = int(values.size), int(values.sum())
    estimation_error = float(alpha / 2)
    tail_error = float(alpha / 2)
    upper = 1.0 if k == m else float(beta.ppf(1 - estimation_error, k + 1, m - k))
    tails = binom.sf(np.arange(task_size + 1) - 1, task_size, upper)
    candidates = np.flatnonzero(tails <= tail_error)
    critical = int(candidates[0]) if candidates.size else task_size + 1
    return {
        "method": "clean_count_cp_binomial_v1",
        "task_size": task_size,
        "target_clean_frr": float(alpha),
        "calibration_original_images": m,
        "calibration_sample_alarms": k,
        "empirical_sample_fpr": k / m,
        "sample_fpr_upper_bound": upper,
        "estimation_error_budget": estimation_error,
        "tail_error_budget": tail_error,
        "critical_suspicious_count": critical,
        "detection_possible_at_this_size": critical <= task_size,
        "assumptions": "Frozen sample rule; calibration and incoming clean originals are independent draws with the same alarm probability. No guarantee under domain shift or dependent images. Reused simulated bags are not independent calibration observations.",
    }


def count_decision(counts, calibration):
    counts = np.asarray(counts)
    n = calibration["task_size"]
    if (
        not np.isfinite(counts).all()
        or np.any(counts != np.floor(counts))
        or np.any((counts < 0) | (counts > n))
    ):
        raise ValueError("Suspicious counts must be integers in 0..task_size.")
    tails = binom.sf(counts - 1, n, calibration["sample_fpr_upper_bound"])
    return counts >= calibration["critical_suspicious_count"], tails
