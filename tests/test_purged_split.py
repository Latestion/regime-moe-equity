import unittest

import pandas as pd

from core.forecast.models import _time_ordered_holdout


class PurgedSplitTests(unittest.TestCase):
    def test_embargo_separates_training_and_validation(self):
        dates = pd.bdate_range("2025-01-01", periods=30)
        panel = pd.DataFrame({
            "date": dates.repeat(2),
            "token": ["A", "B"] * len(dates),
        })
        early, late = _time_ordered_holdout(panel, frac=.2, purge_days=5)
        self.assertFalse(early.empty)
        self.assertFalse(late.empty)
        self.assertLess(early["date"].max(), late["date"].min())
        between = pd.bdate_range(
            early["date"].max(), late["date"].min(), inclusive="neither",
        )
        self.assertGreaterEqual(len(between), 4)


if __name__ == "__main__":
    unittest.main()
