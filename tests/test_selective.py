import unittest

import numpy as np
import pandas as pd

from core.forecast import selective


def _synthetic_scored(regime_ic: dict[str, float], *, n_days: int = 60, n_names: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    regimes = list(regime_ic)
    for i, date in enumerate(dates):
        regime_label = regimes[i % len(regimes)]
        target_ic = regime_ic[regime_label]
        signal = rng.normal(size=n_names)
        noise = rng.normal(size=n_names)
        fwd = target_ic * signal + (1 - abs(target_ic)) * noise
        for j in range(n_names):
            rows.append({
                "date": date, "token": f"T{j}", "regime": regime_label,
                "score": float(signal[j]), "xs_excess_5d": float(fwd[j]),
            })
    return pd.DataFrame(rows)


class FitGateTests(unittest.TestCase):
    def test_significant_and_insignificant_regimes_are_separated(self):
        frame = _synthetic_scored({"trend": 0.6, "chop": 0.0}, n_days=80, seed=1)
        gate = selective.fit_gate(frame, ic_alpha=0.05, ic_correction="bonferroni")
        self.assertIn("trend", gate.significant_regimes)
        self.assertNotIn("chop", gate.significant_regimes)

    def test_fit_start_end_reflect_only_the_rows_given(self):
        frame = _synthetic_scored({"trend": 0.6}, n_days=40, seed=2)
        gate = selective.fit_gate(frame)
        self.assertEqual(gate.fit_start, str(frame["date"].min().date()))
        self.assertEqual(gate.fit_end, str(frame["date"].max().date()))

    def test_gate_cannot_see_rows_it_was_not_given(self):
        """Construction-level leakage guard: two disjoint slices of the same
        underlying process must be able to produce different gates, proving
        fit_gate's output is a function of exactly what it receives — not of
        any wider panel state it might otherwise have reached into."""
        early = _synthetic_scored({"trend": 0.6}, n_days=40, seed=3)
        late = _synthetic_scored({"trend": 0.0}, n_days=40, seed=4)
        late["date"] = late["date"] + pd.Timedelta(days=365)
        gate_early = selective.fit_gate(early)
        gate_late = selective.fit_gate(late)
        self.assertIn("trend", gate_early.significant_regimes)
        self.assertNotIn("trend", gate_late.significant_regimes)
        self.assertLess(gate_early.fit_end, gate_late.fit_start)

    def test_insufficient_observations_yields_no_significant_regimes(self):
        frame = _synthetic_scored({"trend": 0.6}, n_days=2, n_names=10, seed=5)
        self.assertLess(len(frame), 30)
        gate = selective.fit_gate(frame, min_regime_obs=30)
        self.assertEqual(gate.significant_regimes, ())


class ApplyGateTests(unittest.TestCase):
    def test_blocks_non_significant_regimes_only(self):
        gate = selective.SelectiveGate(
            significant_regimes=("trend",), regime_stats={},
            fit_start="2024-01-01", fit_end="2024-06-01", fit_n_rows=100,
        )
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "token": ["A", "B"],
            "regime": ["trend", "chop"],
            "score": [1.5, -0.3],
        })
        out = selective.apply_gate(frame, gate)
        self.assertEqual(out.loc[out["token"] == "A", "score"].iloc[0], 1.5)
        self.assertTrue(pd.isna(out.loc[out["token"] == "B", "score"].iloc[0]))

    def test_empty_gate_blocks_everything(self):
        gate = selective.SelectiveGate(
            significant_regimes=(), regime_stats={},
            fit_start="", fit_end="", fit_n_rows=0,
        )
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01"]),
            "token": ["A"], "regime": ["trend"], "score": [1.0],
        })
        out = selective.apply_gate(frame, gate)
        self.assertTrue(out["score"].isna().all())

    def test_is_pure_and_does_not_mutate_input(self):
        gate = selective.SelectiveGate(
            significant_regimes=("trend",), regime_stats={},
            fit_start="2024-01-01", fit_end="2024-06-01", fit_n_rows=100,
        )
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01"]),
            "token": ["A"], "regime": ["chop"], "score": [2.0],
        })
        first = selective.apply_gate(frame, gate)
        second = selective.apply_gate(frame, gate)
        self.assertEqual(frame["score"].iloc[0], 2.0)  # input untouched
        pd.testing.assert_frame_equal(first, second)  # deterministic, no hidden state


class SelectiveConfigTests(unittest.TestCase):
    def test_rejects_non_flat_abstain_mode(self):
        with self.assertRaises(ValueError):
            selective.SelectiveConfig(base_config={}, abstain_mode="equal_weight")


class LowVolTiltConfigTests(unittest.TestCase):
    def test_rejects_out_of_range_model_weight(self):
        with self.assertRaises(ValueError):
            selective.LowVolTiltConfig(base_config={}, model_weight=1.5)
        with self.assertRaises(ValueError):
            selective.LowVolTiltConfig(base_config={}, model_weight=-0.1)

    def test_rejects_non_positive_top_k(self):
        with self.assertRaises(ValueError):
            selective.LowVolTiltConfig(base_config={}, top_k=0)


class BlendLowvolForecastScoreTests(unittest.TestCase):
    def test_zero_model_weight_is_pure_lowvol_rank(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01"] * 3),
            "token": ["A", "B", "C"],
            "rv_20": [0.4, 0.1, 0.2],
            "score": [5.0, -3.0, 0.0],  # should be ignored entirely
        })
        out = selective.blend_lowvol_forecast_score(frame, model_weight=0.0)
        # Lowest rv_20 (B) should rank highest since ascending=False on rv_20.
        ranked = out.sort_values("score", ascending=False)["token"].tolist()
        self.assertEqual(ranked, ["B", "C", "A"])

    def test_one_model_weight_is_pure_model_rank(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01"] * 3),
            "token": ["A", "B", "C"],
            "rv_20": [0.4, 0.1, 0.2],  # should be ignored entirely
            "score": [5.0, -3.0, 0.0],
        })
        out = selective.blend_lowvol_forecast_score(frame, model_weight=1.0)
        ranked = out.sort_values("score", ascending=False)["token"].tolist()
        self.assertEqual(ranked, ["A", "C", "B"])

    def test_does_not_mutate_input(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01"] * 2),
            "token": ["A", "B"], "rv_20": [0.1, 0.2], "score": [1.0, 2.0],
        })
        selective.blend_lowvol_forecast_score(frame, model_weight=0.5)
        self.assertEqual(frame["score"].tolist(), [1.0, 2.0])

    def test_empty_frame_is_a_noop(self):
        frame = pd.DataFrame(columns=["date", "token", "rv_20", "score"])
        out = selective.blend_lowvol_forecast_score(frame, model_weight=0.5)
        self.assertTrue(out.empty)


class BlendLowvolForecastScoreConditionalTests(unittest.TestCase):
    def _gate(self, *regimes):
        return selective.SelectiveGate(
            significant_regimes=tuple(regimes), regime_stats={},
            fit_start="2024-01-01", fit_end="2024-06-01", fit_n_rows=100,
        )

    def test_non_significant_regime_ignores_model_entirely(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01"] * 3),
            "token": ["A", "B", "C"],
            "regime": ["chop", "chop", "chop"],
            "rv_20": [0.4, 0.1, 0.2],
            "score": [5.0, -3.0, 0.0],  # should be fully ignored: gate is closed for chop
        })
        gate = self._gate("trend")
        out = selective.blend_lowvol_forecast_score_conditional(
            frame, gate=gate, model_weight=0.9,
        )
        pure_lowvol = selective.blend_lowvol_forecast_score(frame, model_weight=0.0)
        pd.testing.assert_series_equal(out["score"], pure_lowvol["score"])

    def test_significant_regime_uses_full_model_weight(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01"] * 3),
            "token": ["A", "B", "C"],
            "regime": ["trend", "trend", "trend"],
            "rv_20": [0.4, 0.1, 0.2],
            "score": [5.0, -3.0, 0.0],
        })
        gate = self._gate("trend")
        out = selective.blend_lowvol_forecast_score_conditional(
            frame, gate=gate, model_weight=0.9,
        )
        full_weight = selective.blend_lowvol_forecast_score(frame, model_weight=0.9)
        pd.testing.assert_series_equal(out["score"], full_weight["score"])

    def test_mixed_regimes_within_one_day_are_split_correctly(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01"] * 2),
            "token": ["A", "B"],
            "regime": ["trend", "chop"],
            "rv_20": [0.4, 0.1],
            "score": [5.0, -3.0],
        })
        gate = self._gate("trend")
        out = selective.blend_lowvol_forecast_score_conditional(
            frame, gate=gate, model_weight=1.0,
        )
        lowvol_rank = frame.groupby("date")["rv_20"].rank(pct=True, ascending=False)
        model_rank = frame.groupby("date")["score"].rank(pct=True)
        self.assertAlmostEqual(out.loc[0, "score"], model_rank.iloc[0])  # trend: full model
        self.assertAlmostEqual(out.loc[1, "score"], lowvol_rank.iloc[1])  # chop: pure low-vol

    def test_empty_gate_reduces_to_pure_lowvol(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01"] * 2),
            "token": ["A", "B"],
            "regime": ["trend", "meanrev"],
            "rv_20": [0.4, 0.1],
            "score": [5.0, -3.0],
        })
        gate = self._gate()
        out = selective.blend_lowvol_forecast_score_conditional(
            frame, gate=gate, model_weight=1.0,
        )
        pure_lowvol = selective.blend_lowvol_forecast_score(frame, model_weight=0.0)
        pd.testing.assert_series_equal(out["score"], pure_lowvol["score"])


class RegimeConditionalTiltConfigTests(unittest.TestCase):
    def test_rejects_out_of_range_model_weight(self):
        with self.assertRaises(ValueError):
            selective.RegimeConditionalTiltConfig(base_config={}, model_weight=1.5)

    def test_rejects_non_positive_top_k(self):
        with self.assertRaises(ValueError):
            selective.RegimeConditionalTiltConfig(base_config={}, top_k=0)


if __name__ == "__main__":
    unittest.main()
