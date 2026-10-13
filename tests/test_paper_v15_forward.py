import unittest

import numpy as np
import pandas as pd

import paper_v15_forward as fwd


def signal_frame(status="pending_entry"):
    # Mirror the persisted signals.csv schema. Date/status fields are explicitly
    # object dtype so pandas 3 cannot infer a float column from an all-NaN seed.
    return pd.DataFrame([{
        "signal_id": "V15-000001", "signal_key": "2026-10-14|123456",
        "signal_date": "2026-10-14", "ticker": "123456", "name": "A", "status": status,
        "decision_date": None, "entry_date": None, "entry_price": np.nan,
        "exit_date": None, "exit_price": np.nan, "net_return": np.nan,
        "benchmark_return": np.nan, "excess_return": np.nan,
    }])


class V15ForwardPaperTests(unittest.TestCase):
    def test_frozen_start_and_parameters(self):
        self.assertEqual(fwd.OOS_START_SIGNAL_DATE, pd.Timestamp("2026-10-14"))
        self.assertEqual(fwd.SIGNAL_WEIGHT, 0.10)
        self.assertEqual(fwd.MAX_POSITIONS, 10)
        self.assertEqual(fwd.HOLD_SESSIONS, 20)
        self.assertEqual(fwd.ROUND_TRIP_COST, 0.004)
        self.assertEqual(fwd.MEGA_TICKERS, {"005930", "000660"})

    def test_cooldown_is_20_observed_rows(self):
        x = pd.DataFrame({
            "ticker": ["000001"] * 4,
            "row_pos": [100, 119, 120, 140],
            "date": pd.to_datetime(["2026-01-01", "2026-01-20", "2026-01-21", "2026-02-20"]),
        })
        out = fwd.apply_cooldown(x, 20)
        self.assertEqual(out.row_pos.tolist(), [100, 120, 140])

    def test_exit_is_20th_observed_session_after_signal(self):
        dates = pd.bdate_range("2026-10-14", periods=25)
        stock = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1000}, index=dates)
        signal = dates[0]
        self.assertEqual(fwd.next_stock_session(stock, signal), dates[1])
        self.assertEqual(fwd.exit_stock_session(stock, signal, 20), dates[20])

    def test_duplicate_signal_is_not_appended(self):
        state = fwd._empty_state()
        signals = signal_frame()
        state["next_signal_id"] = 2
        selected = pd.DataFrame([{
            "ticker": "123456", "name": "A", "Close": 100.0, "ADV20": 3e9,
            "RS20": .1, "RS60": .2, "RS120": .3,
            "RS20_PCTL": .8, "RS60_PCTL": .8, "RS120_PCTL": .8,
            "KOSPI_RET60": .1, "KOSPI_RET60_PCTL756": .8,
            "LIQ_BREADTH60": .5, "LIQ_BREADTH60_PCTL756": .5,
        }])
        bundle = fwd.DataBundle(pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame(), "abc", 0)
        state, out = fwd.append_new_signals(selected, pd.Timestamp("2026-10-14"), state, signals, bundle)
        self.assertEqual(len(out), 1)
        self.assertEqual(state["next_signal_id"], 2)

    def test_entry_is_funded_from_kospi_and_total_equity_stays_one(self):
        dates = pd.bdate_range("2026-10-14", periods=22)
        bench = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0, "Volume": 1}, index=dates)
        stock = pd.DataFrame({"Open": 50.0, "High": 50.0, "Low": 50.0, "Close": 50.0, "Volume": 1000}, index=dates)
        bundle = fwd.DataBundle(pd.DataFrame(), pd.DataFrame(), {"123456": stock}, bench, pd.DataFrame(), "abc", 0)
        state = fwd._empty_state()
        state.update({
            "initialized": True,
            "initial_session": str(dates[0].date()),
            "index_units": 1.0 / (100.0 * (1 + fwd.SIDE_COST)),
            "benchmark_units": 1.0 / (100.0 * (1 + fwd.SIDE_COST)),
            "pending_signals": [{"signal_id": "V15-000001", "ticker": "123456", "name": "A", "signal_date": str(dates[0].date())}],
        })
        signals = signal_frame()
        trades = pd.DataFrame(columns=["signal_id"])
        before_units = state["index_units"]
        state, signals, trades, row = fwd.process_market_day(dates[1], state, bundle, signals, trades)
        self.assertEqual(len(state["active_positions"]), 1)
        self.assertLess(state["index_units"], before_units)
        self.assertAlmostEqual(row["total_equity_exposure"], 1.0, places=12)
        self.assertEqual(signals.loc[0, "status"], "active")

    def test_position_closes_on_twentieth_observed_session(self):
        dates = pd.bdate_range("2026-10-14", periods=22)
        bench = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0, "Volume": 1}, index=dates)
        stock = pd.DataFrame({"Open": 50.0, "High": 55.0, "Low": 49.0, "Close": 55.0, "Volume": 1000}, index=dates)
        bundle = fwd.DataBundle(pd.DataFrame(), pd.DataFrame(), {"123456": stock}, bench, pd.DataFrame(), "abc", 0)
        state = fwd._empty_state()
        state.update({
            "initialized": True, "initial_session": str(dates[0].date()),
            "index_units": 0.9 / 100.0, "benchmark_units": 1.0 / 100.0,
            "active_positions": [{
                "signal_id": "V15-000001", "ticker": "123456", "name": "A",
                "signal_date": str(dates[0].date()), "entry_date": str(dates[1].date()),
                "entry_price": 50.0, "shares": 0.1 / 50.0,
                "target_exit_date": str(dates[20].date()), "benchmark_entry_open": 100.0,
            }],
        })
        signals = signal_frame(status="active")
        trades = pd.DataFrame(columns=["signal_id"])
        state, signals, trades, row = fwd.process_market_day(dates[20], state, bundle, signals, trades)
        self.assertEqual(len(state["active_positions"]), 0)
        self.assertEqual(len(trades), 1)
        self.assertEqual(signals.loc[0, "status"], "closed")
        self.assertAlmostEqual(row["total_equity_exposure"], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
