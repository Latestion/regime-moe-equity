import sqlite3
import unittest
from unittest.mock import patch

import pandas as pd

from core.forecast import data_quality


class DataQualityTests(unittest.TestCase):
    def test_fundamentals_respect_publication_date(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """CREATE TABLE fundamental_observations(
                 token TEXT, metric TEXT, observation_date TEXT,
                 available_at TEXT, value REAL, source TEXT
               )"""
        )
        conn.execute(
            """INSERT INTO fundamental_observations VALUES
               ('A','roe','2024-12-31','2025-02-15',0.20,'filing')"""
        )
        panel = pd.DataFrame({
            "date": pd.to_datetime(["2025-02-14", "2025-02-15"]),
            "token": ["A", "A"],
        })
        with patch.object(data_quality, "get_conn", return_value=conn):
            result = data_quality.attach_fundamentals(panel)
        self.assertTrue(pd.isna(result.iloc[0].get("fund_roe")))
        self.assertEqual(result.iloc[1]["fund_roe"], 0.20)
        self.assertEqual(result.iloc[1]["fund_is_stale"], 0.0)
        self.assertEqual(result.iloc[1]["fund_has_recent"], 1.0)

    def test_fundamental_derivatives_are_point_in_time(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """CREATE TABLE fundamental_observations(
                 token TEXT, metric TEXT, observation_date TEXT,
                 available_at TEXT, value REAL, source TEXT
               )"""
        )
        conn.executemany(
            "INSERT INTO fundamental_observations VALUES (?,?,?,?,?,?)",
            [
                ("A", "revenue", "2024-09-30", "2024-11-01", 100.0, "filing"),
                ("A", "revenue", "2024-12-31", "2025-02-01", 125.0, "filing"),
                ("A", "operating_cashflow", "2024-12-31", "2025-02-01", 40.0, "filing"),
                ("A", "capital_expenditure", "2024-12-31", "2025-02-01", -10.0, "filing"),
            ],
        )
        panel = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-15", "2025-02-01"]),
            "token": ["A", "A"],
        })
        with patch.object(data_quality, "get_conn", return_value=conn):
            result = data_quality.attach_fundamentals(panel)
        self.assertTrue(pd.isna(result.iloc[0].get("fund_revenue_growth_qoq")))
        self.assertAlmostEqual(result.iloc[1]["fund_revenue_growth_qoq"], 0.25)
        self.assertAlmostEqual(result.iloc[1]["fund_fcf"], 30.0)

    def test_point_in_time_membership_excludes_before_and_after_dates(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """CREATE TABLE universe_membership(
                 universe TEXT, token TEXT, symbol TEXT,
                 effective_from TEXT, effective_to TEXT, source TEXT
               )"""
        )
        conn.execute(
            """INSERT INTO universe_membership VALUES
               ('nifty200','A','A-EQ','2025-01-02','2025-01-03','test')"""
        )
        panel = pd.DataFrame({
            "date": pd.to_datetime([
                "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
            ]),
            "token": ["A"] * 4,
            "close": [10, 11, 12, 13],
        })
        with patch.object(data_quality, "get_conn", return_value=conn):
            result = data_quality.filter_point_in_time(
                panel, universe="nifty200",
            )
        self.assertEqual(
            result["date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2025-01-02", "2025-01-03"],
        )

    def test_split_adjustment_applies_only_before_action(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """CREATE TABLE corporate_actions(
                 token TEXT, action_date TEXT, price_factor REAL,
                 volume_factor REAL
               )"""
        )
        conn.execute(
            "INSERT INTO corporate_actions VALUES ('A','2025-01-03',0.5,2.0)"
        )
        candles = pd.DataFrame(
            {
                "open": [100.0, 102.0, 52.0],
                "high": [101.0, 103.0, 53.0],
                "low": [99.0, 101.0, 51.0],
                "close": [100.0, 102.0, 52.0],
                "volume": [1000.0, 1100.0, 2200.0],
            },
            index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        )
        with patch.object(data_quality, "get_conn", return_value=conn):
            adjusted = data_quality.apply_corporate_actions(candles, token="A")
        self.assertEqual(adjusted.iloc[0]["close"], 50.0)
        self.assertEqual(adjusted.iloc[0]["volume"], 2000.0)
        self.assertEqual(adjusted.iloc[2]["close"], 52.0)

    def test_proxy_membership_requires_explicit_override(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """CREATE TABLE universe_membership(
                 universe TEXT, token TEXT, symbol TEXT,
                 effective_from TEXT, effective_to TEXT, source TEXT
               )"""
        )
        conn.execute(
            """INSERT INTO universe_membership VALUES
               ('yahoo_angel200','A','A-EQ','2026-06-26',NULL,
                'yahoo_screen_current_proxy')"""
        )
        conn.execute(
            """CREATE TABLE eod_prices(token TEXT, trade_date TEXT)"""
        )
        conn.execute(
            """CREATE TABLE symbol_history(canonical_token TEXT)"""
        )
        conn.execute(
            """CREATE TABLE corporate_actions(token TEXT, action_date TEXT)"""
        )
        conn.execute(
            """CREATE TABLE fundamental_observations(token TEXT)"""
        )
        conn.execute(
            """CREATE TABLE macro_observations(series TEXT)"""
        )
        panel = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2026-06-26"]),
            "token": ["A", "A"],
            "ret_1d": [0.0, 0.01],
            "feature": [1.0, 2.0],
        })
        with patch.object(data_quality, "get_conn", return_value=conn):
            blocked = data_quality.validate_research_data(
                panel, universe="yahoo_angel200",
            )
            allowed = data_quality.validate_research_data(
                panel, universe="yahoo_angel200", allow_current_proxy=True,
            )
        self.assertFalse(blocked["passed"])
        self.assertIn(
            "current_proxy_membership_not_point_in_time",
            blocked["failures"],
        )
        self.assertTrue(allowed["passed"])
        self.assertTrue(allowed["membership_is_proxy"])
        self.assertTrue(allowed["exploratory_proxy_override"])


if __name__ == "__main__":
    unittest.main()
