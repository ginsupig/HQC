import unittest

import pandas as pd

from backtest_runner import _bar_to_ticks


class TestBacktestReplayTicks(unittest.TestCase):
    def test_bar_to_ticks_replays_open_and_close_only(self):
        row = pd.Series({
            "timestamp": "2026-01-02T14:30:00Z",
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 101.0,
            "volume": 200.0,
        })
        ticks = _bar_to_ticks(row, "JPM")
        self.assertEqual(len(ticks), 2)
        self.assertEqual([t.payload["price"] for t in ticks], [100.0, 101.0])
        self.assertEqual([t.payload["volume"] for t in ticks], [100.0, 100.0])
        self.assertEqual(
            ticks[1].payload["timestamp"] - ticks[0].payload["timestamp"],
            30000,
        )


if __name__ == "__main__":
    unittest.main()
