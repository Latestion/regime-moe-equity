import unittest

import numpy as np
import pandas as pd
from unittest.mock import patch

from core.forecast import backtest
from core.forecast.backtest import BacktestConfig, _simulate_event_driven


class EventLedgerTests(unittest.TestCase):
    def test_drawdown_includes_loss_from_initial_capital(self):
        equity = pd.Series(
            [0.90, 0.95],
            index=pd.to_datetime(["2025-01-01", "2025-01-02"]),
        )
        drawdown, peak, trough = backtest._max_drawdown(equity)
        self.assertAlmostEqual(drawdown, -0.10)
        self.assertIsNone(peak)
        self.assertEqual(trough, "2025-01-01")

    def test_turnover_cost_and_daily_returns(self):
        dates = pd.to_datetime(["2025-01-01", "2025-01-02"])
        scored = pd.DataFrame([
            {"date": dates[0], "token": "A", "score": .9, "ret_1d": np.log1p(.01), "rv_20": .2},
            {"date": dates[0], "token": "B", "score": .1, "ret_1d": np.log1p(-.01), "rv_20": .2},
            {"date": dates[1], "token": "A", "score": .9, "ret_1d": np.log1p(.02), "rv_20": .2},
            {"date": dates[1], "token": "B", "score": .1, "ret_1d": np.log1p(-.02), "rv_20": .2},
        ])
        cfg = BacktestConfig(
            horizon="1d", top_k=1, cost_bps=10, slippage_bps=0,
            max_name_weight=1.0,
        )
        daily, equity, trades = _simulate_event_driven(scored, cfg=cfg, h=1)
        # A signal formed on Jan 1 can only earn the Jan 2 return.
        self.assertAlmostEqual(daily.iloc[0], .019, places=8)
        self.assertAlmostEqual(daily.iloc[1], 0.0, places=8)
        self.assertEqual(trades[0]["turnover"], 1.0)
        self.assertGreater(equity.iloc[-1], 1.0)

    def test_score_oos_carries_test_truth_without_name_error(self):
        dates = pd.bdate_range("2020-01-01", periods=600)
        panel = pd.DataFrame({
            "date": dates,
            "token": ["A"] * len(dates),
            "ret_1d": np.linspace(-.01, .01, len(dates)),
            "ret_5d": np.linspace(-.02, .02, len(dates)),
            "xs_up_5d": ([0.0, 1.0] * 300),
            "logret_1": np.linspace(-.01, .01, len(dates)),
        })

        class FakeClassifier:
            def predict_proba(self, X):
                return np.full(len(X), .52)

        def fake_spec(*args, **kwargs):
            return object()

        def fake_xy(df, spec):
            labelled = df.dropna(subset=["xs_up_5d"])
            return pd.DataFrame({"x": np.zeros(len(labelled))}, index=labelled.index), labelled["xs_up_5d"]

        with patch.object(backtest.models, "make_spec", side_effect=fake_spec), \
             patch.object(backtest.models, "prepare_xy", side_effect=fake_xy), \
             patch.object(backtest.models, "train_classifier_ensemble", return_value=FakeClassifier()):
            scored = backtest._score_oos(
                panel,
                horizon="5d",
                ablation="price_only",
                target="xs",
                train_years=1,
                test_months=6,
                min_train_rows=1,
            )
        self.assertIn("truth", scored.columns)
        self.assertTrue(set(scored["truth"].unique()).issubset({0.0, 1.0}))


if __name__ == "__main__":
    unittest.main()
