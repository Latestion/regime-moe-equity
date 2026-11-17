import math
import unittest

import numpy as np

from core.forecast import research_statistics


class ResearchStatisticsTests(unittest.TestCase):
    def test_probabilistic_sharpe_uses_period_units(self):
        returns = np.tile([-0.01, 0.012], 250)
        result = research_statistics.probabilistic_sharpe_ratio(returns)
        self.assertAlmostEqual(
            result["sharpe_annual"],
            result["sharpe_period"] * math.sqrt(252),
        )
        self.assertGreater(result["psr"], 0.5)
        self.assertLessEqual(result["psr"], 1.0)
        self.assertIsNotNone(result["min_track_record_observations"])

    def test_deflation_raises_the_selection_hurdle(self):
        rng = np.random.default_rng(11)
        returns = rng.normal(0.0007, 0.01, 600)
        result = research_statistics.deflated_sharpe_ratio(
            returns,
            trial_sharpes_annual=[-0.2, 0.1, 0.4, 0.8, 1.1, 1.4],
            raw_trial_count=96,
        )
        self.assertGreater(result["expected_max_sharpe_annual"], 0.0)
        self.assertLessEqual(result["dsr"], result["psr"])
        self.assertEqual(result["raw_trial_count"], 96)

    def test_cscv_reports_all_symmetric_combinations(self):
        rng = np.random.default_rng(19)
        matrix = rng.normal(0.0, 0.01, size=(32, 3))
        matrix[:, 0] += 0.0002
        result = research_statistics.cscv_probability_of_backtest_overfitting(
            matrix, blocks=4,
        )
        self.assertEqual(result["combination_count"], 6)
        self.assertGreaterEqual(result["pbo"], 0.0)
        self.assertLessEqual(result["pbo"], 1.0)
        self.assertGreaterEqual(result["oos_probability_of_loss"], 0.0)
        self.assertLessEqual(result["oos_probability_of_loss"], 1.0)

    def test_invalid_return_streams_fail_closed(self):
        with self.assertRaises(ValueError):
            research_statistics.probabilistic_sharpe_ratio([0.01] * 20)
        with self.assertRaises(ValueError):
            research_statistics.cscv_probability_of_backtest_overfitting(
                np.ones((16, 2)), blocks=4,
            )


if __name__ == "__main__":
    unittest.main()
