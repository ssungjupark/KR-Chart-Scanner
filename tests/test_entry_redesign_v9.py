import unittest

import pandas as pd

import robustness_v7 as v7
from entry_redesign_v9 import MODEL_SPECS, entry_mask


def row(**overrides):
    base = {
        "ticker": "000001",
        "row_pos": 200,
        "setup": True,
        "RS20": 0.05,
        "RS_RATIO_SLOPE": 0.01,
        "RS20_PCTL": 0.60,
        "RS60_PCTL": 0.70,
        "RS120_PCTL": 0.80,
        "RS_REACCEL": True,
        "TRIG_BREAK3": True,
        "TRIG_PREV_HIGH": True,
        "TRIG_MA5_RECLAIM": True,
        "TRIG_MA20_RECLAIM": True,
        "SMA120_UP20": True,
        "PULLBACK_DRYUP": True,
        "CONTRACTION_OK": True,
        "MARKET_NOT_BEAR": True,
        "NEXT_ENTRY_GAP": 0.01,
    }
    base.update(overrides)
    return base


class V9EntryMaskTests(unittest.TestCase):
    def test_v6_baseline_keeps_original_rs_and_break3(self):
        frame = pd.DataFrame([
            row(),
            row(RS20_PCTL=0.49),
            row(TRIG_BREAK3=False),
        ])
        got = entry_mask(frame, MODEL_SPECS["v6_baseline"]).tolist()
        self.assertEqual(got, [True, False, False])

    def test_reaccel_model_drops_hard_rs20_percentile_floor(self):
        frame = pd.DataFrame([
            row(RS20_PCTL=0.20, RS60_PCTL=0.65, RS120_PCTL=0.75, RS_REACCEL=True),
        ])
        self.assertFalse(entry_mask(frame, MODEL_SPECS["v6_baseline"]).iloc[0])
        self.assertTrue(entry_mask(frame, MODEL_SPECS["diag_reaccel_ma5"]).iloc[0])

    def test_v9_core_requires_long_trend_non_bear_and_reclaim(self):
        good = row()
        flat_120 = row(SMA120_UP20=False)
        bear = row(MARKET_NOT_BEAR=False)
        no_reclaim = row(TRIG_MA5_RECLAIM=False)
        frame = pd.DataFrame([good, flat_120, bear, no_reclaim])
        got = entry_mask(frame, MODEL_SPECS["v9_core"]).tolist()
        self.assertEqual(got, [True, False, False, False])

    def test_gap_above_three_percent_is_not_chased(self):
        frame = pd.DataFrame([
            row(NEXT_ENTRY_GAP=0.03),
            row(NEXT_ENTRY_GAP=0.03001),
        ])
        got = entry_mask(frame, MODEL_SPECS["v9_core"]).tolist()
        self.assertEqual(got, [True, False])

    def test_skipped_gap_does_not_consume_cooldown(self):
        frame = pd.DataFrame([
            row(row_pos=200, NEXT_ENTRY_GAP=0.05),
            row(row_pos=205, NEXT_ENTRY_GAP=0.01),
        ])
        frame["date"] = pd.to_datetime(["2020-01-02", "2020-01-09"])
        filtered = frame.loc[entry_mask(frame, MODEL_SPECS["v9_core"])].copy()
        chosen = v7._cooldown(filtered, 20)
        self.assertEqual(chosen["row_pos"].tolist(), [205])

    def test_quality_overlay_accepts_dryup_or_contraction(self):
        frame = pd.DataFrame([
            row(PULLBACK_DRYUP=True, CONTRACTION_OK=False),
            row(PULLBACK_DRYUP=False, CONTRACTION_OK=True),
            row(PULLBACK_DRYUP=False, CONTRACTION_OK=False),
        ])
        got = entry_mask(frame, MODEL_SPECS["v9_full_quality"]).tolist()
        self.assertEqual(got, [True, True, False])


if __name__ == "__main__":
    unittest.main()
