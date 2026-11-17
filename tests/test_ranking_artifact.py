import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.forecast import models, ranking


class FakeLinear:
    def predict(self, X):
        return X.sum(axis=1).to_numpy()


class FakeJoblib:
    @staticmethod
    def load(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)


class RankingArtifactTests(unittest.TestCase):
    def test_buffered_ranking_retains_minimum_holding_names(self):
        dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
        rows = []
        for date, a_score, b_score in [
            (dates[0], 1.0, 0.0),
            (dates[1], 0.0, 1.0),
            (dates[2], 0.0, 1.0),
        ]:
            rows.extend([
                {
                    "date": date, "token": "A", "score": a_score,
                    "ret_1d": 0.0, "rv_20": 0.2, "sector": "Tech",
                },
                {
                    "date": date, "token": "B", "score": b_score,
                    "ret_1d": 0.0, "rv_20": 0.2, "sector": "Tech",
                },
            ])
        _, _, trades = ranking._simulate_buffered_portfolio(
            pd.DataFrame(rows),
            ranking.RankingConfig(
                top_k=1, rebalance_every=1, rank_buffer=0,
                min_holding_days=2, weighting="equal",
                max_name_weight=1.0, max_sector_weight=1.0,
                max_turnover=2.0, cost_bps=0, slippage_bps=0,
            ),
        )
        self.assertEqual(list(trades[0]["weights"]), ["A"])
        self.assertEqual(list(trades[1]["weights"]), ["A"])

    def test_buffered_ranking_caps_changed_notional(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01"] * 2 + ["2025-01-02"] * 2),
            "token": ["A", "B", "A", "B"],
            "score": [1.0, 0.9, 1.0, 0.9],
            "ret_1d": [0.0, 0.0, 0.0, 0.0],
            "rv_20": [0.2, 0.2, 0.2, 0.2],
            "sector": ["Tech", "Finance", "Tech", "Finance"],
        })
        _, _, trades = ranking._simulate_buffered_portfolio(
            frame,
            ranking.RankingConfig(
                top_k=2, rebalance_every=1, weighting="equal",
                max_name_weight=0.5, max_sector_weight=1.0,
                max_turnover=0.25, cost_bps=0, slippage_bps=0,
            ),
        )
        self.assertLessEqual(trades[0]["turnover"], 0.25 + 1e-12)
        self.assertAlmostEqual(sum(trades[0]["weights"].values()), 0.25)

    def test_ranking_selection_rewards_signal_quality(self):
        noisy = {
            "sharpe": 1.2, "sortino": 1.8, "calmar": 0.8,
            "cagr": 0.20, "max_drawdown": -0.20, "avg_turnover": 0.6,
            "rank_ic": -0.03, "top_k_excess_return": -0.002,
        }
        stable = {
            "sharpe": 1.1, "sortino": 1.7, "calmar": 0.8,
            "cagr": 0.19, "max_drawdown": -0.18, "avg_turnover": 0.5,
            "rank_ic": 0.04, "top_k_excess_return": 0.004,
        }
        self.assertGreater(
            ranking.ranking_selection_score(stable),
            ranking.ranking_selection_score(noisy),
        )

    def test_eta_duration_format_is_human_readable(self):
        self.assertEqual(ranking._format_duration(5), "5s")
        self.assertEqual(ranking._format_duration(65), "1m 05s")
        self.assertEqual(ranking._format_duration(3660), "1h 01m")

    def test_v3_artifact_load_preserves_predictions(self):
        frame = pd.DataFrame({"x": [1.0, 2.0], "sector": ["A", "B"]})
        expected_artifact = ranking.RankingArtifact(
            model_type="linear",
            model=FakeLinear(),
            feature_cols=["x", "sector"],
            config={"model_type": "linear"},
            categorical_maps={"sector": ["A", "B"]},
        )
        expected = expected_artifact.predict(frame)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "model.pkl").open("wb") as handle:
                pickle.dump(FakeLinear(), handle)
            (root / "meta.json").write_text(json.dumps({
                "artifact_version": 3,
                "model_type": "linear",
                "feature_cols": ["x", "sector"],
                "config": {"model_type": "linear"},
                "categorical_maps": {"sector": ["A", "B"]},
            }))
            with patch.object(models, "_joblib", return_value=FakeJoblib):
                loaded = ranking.load_artifact(root)
        np.testing.assert_allclose(loaded.predict(frame), expected)

    def test_research_ablation_feature_groups_are_explicit(self):
        self.assertIn("price_plus_fundamentals", ranking.RANK_ABLATIONS)
        self.assertIn("fund_roe", ranking.ablation_feature_columns(
            "price_plus_fundamentals",
        ))
        self.assertNotIn("fund_roe", ranking.ablation_feature_columns("price_only"))
        self.assertIn("graph_neighbour_shock", ranking.ablation_feature_columns(
            "price_plus_graph",
        ))

    def test_dataset_version_changes_with_panel_membership(self):
        a = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "token": ["A", "A"],
        })
        b = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "token": ["A", "B"],
        })
        self.assertNotEqual(
            ranking.dataset_version(a, universe="nifty200"),
            ranking.dataset_version(b, universe="nifty200"),
        )

    def test_ranking_boundaries_purge_forward_label_sessions(self):
        dates = pd.bdate_range("2025-01-01", periods=20)
        panel = pd.DataFrame({
            "date": dates.repeat(2),
            "token": ["A", "B"] * len(dates),
        })
        train = ranking._purged_before(
            panel, dates[10], horizon_sessions=5,
        )
        validation = ranking._purged_window(
            panel, dates[10], dates[18], horizon_sessions=5,
        )
        self.assertEqual(train["date"].max(), dates[4])
        self.assertEqual(validation["date"].max(), dates[12])

    def test_modal_config_tie_break_is_deterministic(self):
        left = ranking.RankingConfig(model_type="linear", top_k=10)
        right = ranking.RankingConfig(model_type="lambdarank", top_k=20)
        self.assertEqual(
            ranking._modal_config([left, right]),
            ranking._modal_config([right, left]),
        )

    def test_parameter_stability_requires_config_consensus(self):
        configs = [
            ranking.RankingConfig(model_type="linear", top_k=10),
            ranking.RankingConfig(model_type="lambdarank", top_k=15),
            ranking.RankingConfig(model_type="return_regression", top_k=20),
        ]
        unstable = [
            {"config": vars(config), "metrics": {"sharpe": 1.0}}
            for config in configs
        ]
        stable = [
            {"config": vars(configs[0]), "metrics": {"sharpe": value}}
            for value in (1.0, 0.8, -0.1)
        ]
        self.assertFalse(ranking._neighbour_stable(unstable))
        self.assertTrue(ranking._neighbour_stable(stable))


if __name__ == "__main__":
    unittest.main()
