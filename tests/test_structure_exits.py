import unittest

import numpy as np
import pandas as pd

from kospi_structure_exits import COST, prepare, initial_support, simulate_trade, replay


def candles(n=110):
    d = pd.bdate_range("2020-01-01", periods=n)
    close = 100. + np.arange(n)*.2
    return pd.DataFrame({"Open": close, "High": close+1, "Low": close-1,
                         "Close": close, "Volume": 1000.}, index=d)


class StructureExitTests(unittest.TestCase):
    def test_support_uses_only_signal_and_prior_nine_lows(self):
        x = candles()
        x.loc[x.index[61]:, "Low"] = 1
        self.assertEqual(initial_support(prepare(x), 60), x.Low.iloc[51:61].min())

    def test_no_twenty_day_forced_exit_while_trend_is_intact(self):
        x = candles(180)
        o = simulate_trade(prepare(x), x.index[60], "trend_ma20", x.index[-1])
        self.assertEqual(o["status"], "marked_open")
        self.assertGreater(o["hold_observations"], 20)
        self.assertEqual(o["exit_phase"], "mark")
        expected = x.Close.iloc[-1]/x.Open.iloc[61]/(1+COST/2)-1
        self.assertAlmostEqual(o["net_return"], expected)

    def test_fixed20_keeps_original_signal_relative_close(self):
        x = candles()
        o = simulate_trade(prepare(x), x.index[60], "fixed20", x.index[-1])
        self.assertEqual(o["exit_date"], x.index[80])
        self.assertEqual(o["exit_price"], x.Close.iloc[80])

    def test_gap_through_stop_executes_at_open_not_support(self):
        x = candles()
        stop = initial_support(prepare(x), 60)
        x.loc[x.index[62], ["Open", "Low", "High", "Close"]] = [stop-5, stop-6, stop-3, stop-4]
        o = simulate_trade(prepare(x), x.index[60], "trend_ma20", x.index[-1])
        self.assertEqual(o["exit_phase"], "open")
        self.assertEqual(o["exit_reason"], "support_gap")
        self.assertEqual(o["exit_price"], stop-5)

    def test_known_intraday_stop_ignores_exit_days_later_extremes(self):
        x = candles()
        stop = initial_support(prepare(x), 60)
        x.loc[x.index[62], "Low"] = stop-20
        x.loc[x.index[62], "High"] = 1000
        o = simulate_trade(prepare(x), x.index[60], "trend_ma20", x.index[-1])
        self.assertEqual(o["exit_phase"], "intraday")
        self.assertEqual(o["exit_price"], stop)
        self.assertAlmostEqual(o["mae"], stop/x.Open.iloc[61]-1)
        self.assertLess(o["mfe"], .1)

    def test_close_break_is_executed_next_open(self):
        x = candles()
        x.loc[x.index[62], "Close"] = 111.5
        x.loc[x.index[63], "Open"] = 110.5
        o = simulate_trade(prepare(x), x.index[60], "swing_ma5", x.index[-1])
        self.assertEqual(o["exit_reason"], "close_below_ma5")
        self.assertEqual(o["exit_date"], x.index[63])
        self.assertEqual(o["exit_price"], 110.5)

    def test_later_prices_cannot_change_already_executed_exit(self):
        x = candles()
        x.loc[x.index[62], "Close"] = 111.5
        before = simulate_trade(prepare(x), x.index[60], "swing_ma5", x.index[-1])
        x.loc[x.index[64]:, ["Open", "High", "Low", "Close"]] *= 10
        after = simulate_trade(prepare(x), x.index[60], "swing_ma5", x.index[-1])
        self.assertEqual(before, after)

    def test_ma_signal_on_last_close_is_marked_not_sold_at_that_close(self):
        x = candles(63)
        x.loc[x.index[62], "Close"] = 111.5
        o = simulate_trade(prepare(x), x.index[60], "swing_ma5", x.index[-1])
        self.assertEqual(o["status"], "marked_open")
        self.assertEqual(o["exit_reason"], "close_below_ma5")

    def test_halt_delays_pending_sell_to_next_traded_open(self):
        x = candles()
        x.loc[x.index[62], "Close"] = 111.5
        x.loc[x.index[63], ["Open", "High", "Low", "Volume"]] = 0
        x.loc[x.index[64], "Open"] = 109
        o = simulate_trade(prepare(x), x.index[60], "swing_ma5", x.index[-1])
        self.assertEqual(o["exit_date"], x.index[64])
        self.assertEqual(o["exit_price"], 109)

    def test_one_price_bar_defers_stop(self):
        x = candles()
        stop = initial_support(prepare(x), 60)
        x.loc[x.index[62], ["Open", "High", "Low", "Close"]] = stop-2
        x.loc[x.index[63], "Open"] = stop-4
        o = simulate_trade(prepare(x), x.index[60], "trend_ma20", x.index[-1])
        self.assertEqual(o["exit_date"], x.index[63])
        self.assertEqual(o["exit_price"], stop-4)
        self.assertEqual(o["exit_reason"], "support_locked_delayed")

    def test_one_price_entry_bar_is_not_assumed_filled(self):
        x = candles()
        x.loc[x.index[61], ["Open", "High", "Low", "Close"]] = 120
        self.assertEqual(simulate_trade(prepare(x), x.index[60], "trend_ma20", x.index[-1])["status"], "entry_unfilled")

    def test_trailing_low_uses_prior_bars_and_never_todays_low(self):
        x = candles()
        expected = x.Low.iloc[62:72].min()
        x.loc[x.index[72], "Low"] = 1
        o = simulate_trade(prepare(x), x.index[60], "trailing_low10", x.index[-1])
        self.assertEqual(o["exit_date"], x.index[72])
        self.assertEqual(o["exit_price"], expected)
        self.assertGreater(o["final_stop"], o["support"])

    def test_pure_ma_policy_does_not_apply_an_extra_support_stop(self):
        x = candles()
        stop = initial_support(prepare(x), 60)
        x.loc[x.index[62], "Open"] = stop-1
        x.loc[x.index[62], "Low"] = stop-2
        o = simulate_trade(prepare(x), x.index[60], "ma20_only", x.index[-1])
        self.assertEqual(o["status"], "marked_open")
        self.assertTrue(pd.isna(o["final_stop"]))

    def test_pure_ma20_uses_twenty_day_average_and_next_open(self):
        x = candles()
        x.loc[x.index[62], "Close"] = 109.5
        o = simulate_trade(prepare(x), x.index[60], "ma20_only", x.index[-1])
        self.assertEqual(o["exit_date"], x.index[63])
        self.assertEqual(o["exit_reason"], "close_below_ma20")


class StructureAccountingTests(unittest.TestCase):
    def setup_data(self, phase):
        x = candles(4)
        x[["Open", "Close"]] = 100.
        rows = [{"ticker": str(i), "status": "closed", "entry_date": x.index[0], "entry_price": 100.,
                 "exit_date": x.index[1] if i == 0 else x.index[-1],
                 "exit_price": 100., "exit_phase": phase if i == 0 else "close"} for i in range(10)]
        rows.append({"ticker": "extra", "status": "closed", "entry_date": x.index[1], "entry_price": 100.,
                     "exit_date": x.index[-1], "exit_price": 100., "exit_phase": "close"})
        return x, pd.DataFrame(rows), {str(i): prepare(x) for i in range(10)} | {"extra": prepare(x)}

    def test_intraday_exit_cannot_fund_same_day_open_entry(self):
        x, trades, paths = self.setup_data("intraday")
        d, _, skipped, s = replay(trades, paths, x, x.index[0], x.index[-1])
        self.assertEqual(s["accepted"], 10)
        self.assertEqual(s["skipped"], 1)
        self.assertTrue(d.cash.ge(-1e-12).all())

    def test_open_exit_releases_cash_before_new_open_entry(self):
        x, trades, paths = self.setup_data("open")
        # The exiting slot outperforms the others, leaving enough cash after fees
        # to fund a fresh 10% slot. A flat full account lacks that fee margin.
        winner = x.copy()
        winner.loc[x.index[1], ["Open", "Close"]] = 120.
        paths["0"] = prepare(winner)
        trades.loc[trades.ticker.eq("0"), "exit_price"] = 120.
        _, _, _, s = replay(trades, paths, x, x.index[0], x.index[-1])
        self.assertEqual(s["accepted"], 11)

    def test_same_ticker_is_not_bought_again_while_held(self):
        x, trades, paths = self.setup_data("open")
        trades = trades.iloc[[1, 10]].copy()
        trades.iloc[1, trades.columns.get_loc("ticker")] = "1"
        _, _, skipped, s = replay(trades, paths, x, x.index[0], x.index[-1])
        self.assertEqual(s["accepted"], 1)
        self.assertEqual(skipped.reason.iloc[0], "same_ticker_already_held")

    def test_marked_holding_is_not_forced_liquidated(self):
        x = candles(4)
        x[["Open", "Close"]] = 100.
        f = pd.DataFrame([{"ticker": "a", "status": "marked_open", "entry_date": x.index[0],
                           "entry_price": 100., "exit_date": x.index[-1], "exit_price": 100., "exit_phase": "mark"}])
        d, _, _, s = replay(f, {"a": prepare(x)}, x, x.index[0], x.index[-1])
        self.assertEqual(s["still_open"], 1)
        self.assertAlmostEqual(d.equity.iloc[-1], .9+.1/(1+COST/2))


if __name__ == "__main__":
    unittest.main()
