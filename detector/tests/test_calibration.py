"""Finite-clean-pool count calibration, including exact error-budget checks."""

import unittest
import numpy as np
from scipy.stats import beta, binom

from detector.calibration import count_decision, fit_count_calibration


class CountCalibrationTests(unittest.TestCase):
    def test_upper_bound_and_critical_count_match_independent_calculation(self):
        fitted = fit_count_calibration([1] * 20 + [0] * 480, 150, 0.05)
        upper = beta.ppf(0.975, 21, 480)
        self.assertAlmostEqual(fitted["sample_fpr_upper_bound"], upper)
        critical = fitted["critical_suspicious_count"]
        self.assertLessEqual(binom.sf(critical - 1, 150, upper), 0.025)
        self.assertGreater(binom.sf(critical - 2, 150, upper), 0.025)
        decisions, tails = count_decision([critical - 1, critical], fitted)
        np.testing.assert_array_equal(decisions, [False, True])
        self.assertGreater(tails[0], tails[1])

    def test_zero_observed_alarms_does_not_assume_zero_population_risk(self):
        fitted = fit_count_calibration(np.zeros(20), 10)
        self.assertGreater(fitted["sample_fpr_upper_bound"], 0)
        self.assertGreater(fitted["critical_suspicious_count"], 1)

    def test_all_clean_samples_alarm_exposes_no_power_instead_of_fake_success(self):
        fitted = fit_count_calibration(np.ones(20), 10)
        self.assertEqual(fitted["critical_suspicious_count"], 11)
        self.assertFalse(fitted["detection_possible_at_this_size"])
        self.assertFalse(count_decision(10, fitted)[0])

    def test_exact_unconditional_null_rejection_is_bounded_on_probability_grid(self):
        # Enumerate calibration outcomes, not a noisy Monte Carlo assertion.
        m, n, alpha = 30, 15, 0.05
        critical = np.array(
            [
                fit_count_calibration([1] * k + [0] * (m - k), n, alpha)[
                    "critical_suspicious_count"
                ]
                for k in range(m + 1)
            ]
        )
        for p in (0.001, 0.01, 0.05, 0.2, 0.5, 0.9, 0.999):
            rate = np.dot(
                binom.pmf(np.arange(m + 1), m, p), binom.sf(critical - 1, n, p)
            )
            self.assertLessEqual(rate, alpha + 1e-12)

    def test_invalid_inputs_rejected(self):
        for values, n, alpha in (
            ([], 10, 0.05),
            ([0.5], 10, 0.05),
            ([np.nan], 10, 0.05),
            ([0], True, 0.05),
            ([0], 10, 0),
        ):
            with self.subTest(values=values, size=n, alpha=alpha):
                with self.assertRaises(ValueError):
                    fit_count_calibration(values, n, alpha)
        fitted = fit_count_calibration([0, 1], 10)
        for counts in (-1, 11, 1.5, np.nan):
            with self.assertRaises(ValueError):
                count_decision(counts, fitted)


if __name__ == "__main__":
    unittest.main()
