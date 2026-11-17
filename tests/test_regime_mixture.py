import unittest
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core.forecast import ranking, regime_mixture


class _FakeArtifact:
    """Predicts a fixed per-token offset, so tests can check exactly which
    artifact's prediction landed on which row without training a real model.
    """
    def __init__(self, offsets: dict[str, float], momentum_weight: float = 0.0):
        self.offsets = offsets
        self.config = {"momentum_weight": momentum_weight}

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([self.offsets[token] for token in frame["token"]], dtype=float)


class ScoreRegimeRoutedTests(unittest.TestCase):
    def test_ranks_across_full_daily_cross_section_not_per_regime_subset(self):
        # Two names share a day: A is "trend", B is "chop". If ranking were
        # computed per-regime-subset (the bug), each name would be alone in
        # its group and get rank 1.0 (percentile rank of a singleton is
        # always 1.0) regardless of its actual score. Ranked against the
        # full day instead, the higher raw score must get the higher rank.
        test = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "token": ["A", "B"],
            "regime": ["trend", "chop"],
        })
        pooled = _FakeArtifact({"A": 0.0, "B": 0.0})
        specialists = {
            "trend": _FakeArtifact({"A": 10.0, "B": 0.0}),
            "chop": _FakeArtifact({"A": 0.0, "B": -10.0}),
        }
        out = regime_mixture._score_regime_routed(pooled, specialists, test)
        row_a = out.loc[out["token"] == "A"].iloc[0]
        row_b = out.loc[out["token"] == "B"].iloc[0]
        self.assertEqual(row_a["ml_score"], 10.0)
        self.assertEqual(row_b["ml_score"], -10.0)
        self.assertGreater(row_a["ml_rank"], row_b["ml_rank"])
        # The bug this guards against: ranking A and B separately within
        # their own singleton regime groups would give BOTH of them
        # percentile rank 1.0 (each is alone and "highest" in its own
        # group). Ranked together, only the true top name gets 1.0.
        self.assertEqual(row_a["ml_rank"], 1.0)
        self.assertNotEqual(row_b["ml_rank"], 1.0)

    def test_falls_back_to_pooled_artifact_when_no_specialist(self):
        test = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01"]),
            "token": ["A"], "regime": ["meanrev"],
        })
        pooled = _FakeArtifact({"A": 7.0})
        out = regime_mixture._score_regime_routed(pooled, {}, test)
        self.assertEqual(out.loc[0, "ml_score"], 7.0)


class RegimeMixtureConfigTests(unittest.TestCase):
    def test_rejects_non_positive_min_regime_train_rows(self):
        with self.assertRaises(ValueError):
            regime_mixture.RegimeMixtureConfig(base_config={}, min_regime_train_rows=0)


class RegimeIcReportTests(unittest.TestCase):
    def test_none_when_frame_has_no_usable_regime(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01"]),
            "token": ["A"], "regime": [np.nan],
            "score": [1.0], "xs_excess_5d": [0.01],
        })
        self.assertIsNone(regime_mixture._regime_ic_report(frame))


class PairedIcComparisonTests(unittest.TestCase):
    def test_empty_inputs_yield_empty_result(self):
        empty = pd.DataFrame()
        self.assertEqual(regime_mixture._paired_ic_comparison(empty, empty), {})

    def test_identical_pooled_and_mixture_gives_zero_diff(self):
        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2024-01-01", periods=20)
        rows = []
        for date in dates:
            for j in range(5):
                rows.append({
                    "date": date, "token": f"T{j}", "regime": "trend",
                    "score": float(rng.normal()), "xs_excess_5d": float(rng.normal()),
                })
        frame = pd.DataFrame(rows)
        result = regime_mixture._paired_ic_comparison(frame, frame.copy())
        self.assertIn("trend", result)
        self.assertAlmostEqual(result["trend"]["mean_ic_diff"], 0.0, places=9)


class SimulateRegimeConditionalPortfolioTests(unittest.TestCase):
    def _frame(self, regime_by_date: dict[str, str]) -> pd.DataFrame:
        rows = []
        for date_str, regime_label in regime_by_date.items():
            date = pd.Timestamp(date_str)
            for j, token in enumerate(["A", "B", "C", "D"]):
                rows.append({
                    "date": date, "token": token, "regime": regime_label,
                    "score": 1.0 - 0.1 * j, "ret_1d": 0.01,
                })
        return pd.DataFrame(rows)

    def test_dominant_regime_selects_matching_top_k(self):
        frame = self._frame({
            "2024-01-01": "trend", "2024-01-02": "trend", "2024-01-03": "chop",
        })
        regime_configs = {
            "trend": ranking.RankingConfig(top_k=1, rank_buffer=0, min_holding_days=1),
            "chop": ranking.RankingConfig(top_k=3, rank_buffer=0, min_holding_days=1),
        }
        default = ranking.RankingConfig(rebalance_every=1, top_k=2)
        _, _, trades = regime_mixture._simulate_regime_conditional_portfolio(
            frame, regime_configs, default_config=default,
        )
        by_date = {trade["date"]: trade for trade in trades}
        self.assertEqual(by_date["2024-01-01"]["dominant_regime"], "trend")
        self.assertEqual(len(by_date["2024-01-01"]["weights"]), 1)
        self.assertEqual(by_date["2024-01-03"]["dominant_regime"], "chop")
        self.assertEqual(len(by_date["2024-01-03"]["weights"]), 3)

    def test_falls_back_to_default_config_for_unmapped_regime(self):
        frame = self._frame({"2024-01-01": "meanrev"})
        regime_configs = {
            "trend": ranking.RankingConfig(top_k=1),
        }
        default = ranking.RankingConfig(rebalance_every=1, top_k=2)
        returns, _, trades = regime_mixture._simulate_regime_conditional_portfolio(
            frame, regime_configs, default_config=default,
        )
        self.assertEqual(len(returns), 1)
        self.assertEqual(trades[0]["dominant_regime"], "meanrev")


if __name__ == "__main__":
    unittest.main()
