import unittest
import numpy as np
import pandas as pd

from robustness_v8 import COST, EXITS, choose_exit, metrics, outcomes, portfolio, select_model, split


def path():
    dates = pd.bdate_range("2024-01-02", periods=25)
    return pd.DataFrame({"Open": 100.0, "High": 102.0, "Low": 98.0,
                         "Close": 100.0, "ATR": 2.0}, index=dates)


class ExecutionTests(unittest.TestCase):
    def test_unarmed_exit_is_identical_to_fixed_close(self):
        x = path()
        x.iloc[20, x.columns.get_loc("Open")] = 80
        x.iloc[20, x.columns.get_loc("Close")] = 90
        result = outcomes(x, path(), 0)
        for method in EXITS:
            self.assertEqual(choose_exit(x, 0, method), (20, "close"))
            self.assertAlmostEqual(result[f"RET_{method}"], -0.10 - COST)

    def test_close_arms_and_later_close_executes_next_open(self):
        x = path()
        x.loc[x.index[2], "Close"] = 112
        x.loc[x.index[3], "Close"] = 105
        for method in EXITS[1:]:
            self.assertEqual(choose_exit(x, 0, method), (4, "open"))
        # Exit day's later extremes must never enter an open-exit MAE/MFE.
        x.loc[x.index[4], ["High", "Low"]] = [1000, 1]
        result = outcomes(x, path(), 0)
        self.assertAlmostEqual(result["MFE_profit10_dd5"], .02)
        self.assertAlmostEqual(result["MAE_profit10_dd5"], -.02)

    def test_intraday_high_does_not_arm(self):
        x = path()
        x.loc[x.index[2], "High"] = 140
        self.assertEqual(choose_exit(x, 0, "profit10_dd5"), (20, "close"))

    def test_entry_day_can_arm_but_cannot_exit_same_day(self):
        x = path()
        x.loc[x.index[1], "Close"] = 112
        x.loc[x.index[2], "Close"] = 105
        self.assertEqual(choose_exit(x, 0, "profit10_dd5"), (3, "open"))

    def test_future_edits_do_not_change_already_triggered_exit(self):
        x = path()
        x.loc[x.index[2], "Close"] = 112
        x.loc[x.index[3], "Close"] = 105
        before = choose_exit(x, 0, "profit10_dd5")
        x.loc[x.index[4]:, "Close"] = 10000
        self.assertEqual(before, choose_exit(x, 0, "profit10_dd5"))

    def test_missing_benchmark_candle_cannot_be_forward_filled(self):
        b = path().drop(path().index[15])
        with self.assertRaisesRegex(ValueError, "missing observed prices"):
            outcomes(path(), b, 0)


class AccountingTests(unittest.TestCase):
    def signals(self, end_phase):
        x = path()
        return pd.DataFrame([
            {"ticker": "a", "ENTRY_DATE": x.index[0], "ENTRY_PRICE": 100.,
             "EXIT_DATE_fixed20": x.index[2], "EXIT_PHASE_fixed20": end_phase,
             "EXIT_PRICE_fixed20": 100.},
            {"ticker": "b", "ENTRY_DATE": x.index[2], "ENTRY_PRICE": 100.,
             "EXIT_DATE_fixed20": x.index[3], "EXIT_PHASE_fixed20": "close",
             "EXIT_PRICE_fixed20": 100.},
        ])

    def test_close_exit_cannot_fund_same_day_open(self):
        daily, result = portfolio(self.signals("close"), "equal", "fixed20", path(),
                                  {"a": path(), "b": path()}, max_positions=1)
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(result["skipped"], 1)
        self.assertAlmostEqual(daily.equity.iloc[-1], (1-COST/2)/(1+COST/2))
        self.assertLess(result["max_drawdown"], 0)

    def test_open_exit_can_fund_same_open_without_leverage(self):
        daily, result = portfolio(self.signals("open"), "equal", "fixed20", path(),
                                  {"a": path(), "b": path()}, max_positions=1)
        self.assertEqual(result["accepted"], 2)
        self.assertTrue((daily.cash >= -1e-12).all())
        self.assertAlmostEqual(daily.equity.iloc[-1], ((1-COST/2)/(1+COST/2))**2)

    def test_horizon_excess_accounts_for_cash(self):
        f = pd.DataFrame({"PULLBACK_DRYUP": [False], "RET_fixed20": [.10],
                          "BENCH_fixed20": [.08], "EXCESS_fixed20": [.02],
                          "MAE_fixed20": [-.05], "HOLD_fixed20": [19]})
        m = metrics(f, "dryup", "fixed20")
        self.assertAlmostEqual(m["avg_budget_ret"], .05)
        self.assertAlmostEqual(m["avg_horizon_excess"], -.03)
        self.assertAlmostEqual(m["deployed_weighted_ret"], .10)

    def test_purge_removes_future_outcomes_from_selection_period(self):
        f = pd.DataFrame({"date": pd.to_datetime(["2023-12-01", "2023-12-27", "2024-12-29"]),
                          "EXIT_DATE_fixed20": pd.to_datetime(["2023-12-27", "2024-01-25", "2025-01-24"])})
        train, validation, evaluation = split(f, pd.Timestamp("2023-12-28"), pd.Timestamp("2024-12-30"))
        self.assertEqual(len(train), 1)
        self.assertTrue(validation.empty)
        self.assertTrue(evaluation.empty)

    def test_selection_is_invariant_to_evaluation_returns(self):
        records = []
        for date, terminal, count in (("2023-01-02", "2023-02-01", 40),
                                      ("2024-02-01", "2024-03-01", 20),
                                      ("2025-02-01", "2025-03-01", 20)):
            for _ in range(count):
                r = {"date": pd.Timestamp(date), "EXIT_DATE_fixed20": pd.Timestamp(terminal),
                     "QUALITY_SCORE": 2, "PULLBACK_DRYUP": True, "TRIGGER_RVOL_OK": True,
                     "BB_SQUEEZE_OK": True, "MA_COMPRESS_OK": True,
                     "RETEST_RECLAIM_OK": True, "MARKET_NOT_BEAR": True, "BENCH_fixed20": .01}
                for method in EXITS:
                    r.update({f"RET_{method}": .05 if method == "fixed20" else .07,
                              f"EXCESS_{method}": .04, f"MAE_{method}": -.02, f"HOLD_{method}": 10})
                records.append(r)
        f = pd.DataFrame(records)
        selected, grid = select_model(f, pd.Timestamp("2023-12-28"), pd.Timestamp("2024-12-30"))
        f.loc[f.date.dt.year == 2025, [f"RET_{m}" for m in EXITS]] = -999
        after, after_grid = select_model(f, pd.Timestamp("2023-12-28"), pd.Timestamp("2024-12-30"))
        self.assertEqual(selected, after)
        pd.testing.assert_frame_equal(grid, after_grid)


if __name__ == "__main__":
    unittest.main()
